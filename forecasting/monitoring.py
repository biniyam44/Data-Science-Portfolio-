"""Local prediction telemetry and delayed, idempotent outcome reconciliation."""

from datetime import datetime, timezone
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

from .serving import UNIT, _history, _json_bytes

REQUEST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _request_id(value):
    if not isinstance(value, str) or not REQUEST_RE.fullmatch(value):
        raise ValueError("Request ID must contain 1 to 128 safe identifier characters.")
    return value


def _timestamp(value, *, hourly=False):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tz is not None:
        raise ValueError("Use a finite timezone-naive source timestamp.")
    if hourly and timestamp != timestamp.floor("h"):
        raise ValueError("The target timestamp must be hour aligned.")
    return timestamp


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite nonnegative number.")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return result


def drift_report(history: pd.DataFrame, reference_profile: dict | None) -> dict:
    """Descriptive input shift, not an automatic retraining or failure decision."""
    values = _history(history)["y"].tail(168).to_numpy()
    if reference_profile is None:
        return {"available": False, "reason": "No training reference profile supplied."}
    edges = np.asarray(reference_profile["bin_edges"], dtype=float)
    expected_counts = np.asarray(reference_profile["bin_counts"], dtype=float)
    if (len(edges) != len(expected_counts) + 1 or np.any(np.diff(edges) <= 0)
            or not np.isfinite(edges).all() or not np.isfinite(expected_counts).all()
            or np.any(expected_counts < 0) or expected_counts.sum() <= 0):
        raise ValueError("Invalid drift reference profile.")
    edges[0], edges[-1] = -np.inf, np.inf
    observed_counts, _ = np.histogram(values, bins=edges)
    expected = (expected_counts + 0.5) / (expected_counts.sum() + 0.5 * len(expected_counts))
    observed = (observed_counts + 0.5) / (len(values) + 0.5 * len(expected_counts))
    psi = float(np.sum((observed - expected) * np.log(observed / expected)))
    mean = float(values.mean())
    return {
        "available": True, "window_rows": len(values), "mean": mean,
        "reference_mean": reference_profile["mean"],
        "mean_change": mean - reference_profile["mean"],
        "population_stability_index": psi,
        "outside_training_range_fraction": float(np.mean(
            (values < reference_profile["minimum"]) | (values > reference_profile["maximum"])
        )),
        "interpretation": "Descriptive shift only. Seasonality and this single-building sample limit interpretation.",
    }


class Monitor:
    """A single-host SQLite event store; do not expose it as an unauthenticated API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("Monitoring database must not be a symlink.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    request_id TEXT PRIMARY KEY, prediction TEXT NOT NULL,
                    history_sha256 TEXT NOT NULL, latency_ms REAL NOT NULL,
                    drift TEXT NOT NULL, recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    request_id TEXT NOT NULL REFERENCES forecasts(request_id),
                    target_timestamp TEXT NOT NULL, actual REAL NOT NULL,
                    observed_at TEXT NOT NULL, unit TEXT NOT NULL,
                    PRIMARY KEY (request_id, target_timestamp)
                );
                CREATE TABLE IF NOT EXISTS input_failures (
                    category TEXT NOT NULL, recorded_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record_forecast(self, request_id, prediction, latency_ms, history, reference_profile=None):
        request_id = _request_id(request_id)
        latency = _number(latency_ms, "Latency")
        validated = _history(history)
        if prediction.get("unit") != UNIT or prediction.get("origin") != validated.index[-1].isoformat():
            raise ValueError("Forecast units or origin do not match the input history.")
        encoded = _json_bytes(prediction).decode()
        history_hash = sha256(validated.to_csv(float_format="%.17g").encode()).hexdigest()
        drift = _json_bytes(drift_report(validated, reference_profile)).decode()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute("SELECT * FROM forecasts WHERE request_id=?", (request_id,)).fetchone()
            if prior:
                if prior["prediction"] != encoded or prior["history_sha256"] != history_hash:
                    raise ValueError("Request ID conflicts with a previously recorded forecast.")
                return {"request_id": request_id, "created": False}
            connection.execute("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?)", (
                request_id, encoded, history_hash, latency, drift, datetime.now(timezone.utc).isoformat(),
            ))
        return {"request_id": request_id, "created": True}

    def get_forecast(self, request_id):
        with self._connect() as connection:
            row = connection.execute("SELECT prediction FROM forecasts WHERE request_id=?",
                                     (_request_id(request_id),)).fetchone()
        return json.loads(row["prediction"]) if row else None

    def record_outcome(self, request_id, target_timestamp, actual, observed_at, unit=UNIT):
        request_id = _request_id(request_id)
        target = _timestamp(target_timestamp, hourly=True)
        observed = _timestamp(observed_at)
        actual = _number(actual, "Actual energy")
        if unit != UNIT:
            raise ValueError("Outcome units must match kWh per hour.")
        if observed < target + pd.Timedelta(hours=1):
            raise ValueError("The complete target hour must finish before an outcome is observed.")
        target_text, observed_text = target.isoformat(), observed.isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT prediction FROM forecasts WHERE request_id=?", (request_id,)).fetchone()
            if not row:
                raise ValueError("No forecast exists for this request ID.")
            forecast = json.loads(row["prediction"])
            if target_text not in {item["target_timestamp"] for item in forecast["forecasts"]}:
                raise ValueError("The outcome target is not one of the recorded forecast horizons.")
            prior = connection.execute("SELECT * FROM outcomes WHERE request_id=? AND target_timestamp=?",
                                       (request_id, target_text)).fetchone()
            if prior:
                if prior["actual"] != actual or prior["unit"] != unit or prior["observed_at"] != observed_text:
                    raise ValueError("Outcome conflicts with a previously recorded observation.")
                return {"request_id": request_id, "target_timestamp": target_text, "created": False}
            connection.execute("INSERT INTO outcomes VALUES (?, ?, ?, ?, ?)",
                               (request_id, target_text, actual, observed_text, unit))
        return {"request_id": request_id, "target_timestamp": target_text, "created": True}

    def record_input_failure(self, category="invalid_input"):
        if category not in {"invalid_input", "missing_input", "conflicting_request"}:
            raise ValueError("Unknown input failure category.")
        with self._connect() as connection:
            connection.execute("INSERT INTO input_failures VALUES (?, ?)",
                               (category, datetime.now(timezone.utc).isoformat()))

    def report(self, as_of=None):
        cutoff = _timestamp(as_of) if as_of is not None else None
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM forecasts ORDER BY recorded_at, request_id").fetchall()
            outcomes = {(row["request_id"], row["target_timestamp"]): dict(row)
                        for row in connection.execute("SELECT * FROM outcomes")}
            failures = {row["category"]: row["n"] for row in connection.execute(
                "SELECT category, COUNT(*) AS n FROM input_failures GROUP BY category")}
        groups, missing, pending = {}, 0, 0
        for row in rows:
            response = json.loads(row["prediction"])
            for forecast in response["forecasts"]:
                key = (response["version"], forecast["horizon_hours"])
                group = groups.setdefault(key, {"errors": [], "covered": [], "widths": [], "forecast_count": 0})
                group["forecast_count"] += 1
                outcome = outcomes.get((row["request_id"], forecast["target_timestamp"]))
                if outcome is None or (cutoff is not None and pd.Timestamp(outcome["observed_at"]) > cutoff):
                    if cutoff is not None and pd.Timestamp(forecast["target_timestamp"]) + pd.Timedelta(hours=1) <= cutoff:
                        missing += 1
                    else:
                        pending += 1
                    continue
                actual = outcome["actual"]
                group["errors"].append(forecast["prediction"] - actual)
                tolerance = 1e-9 * max(1.0, abs(actual))
                group["covered"].append(forecast["lower"] - tolerance <= actual <= forecast["upper"] + tolerance)
                group["widths"].append(forecast["upper"] - forecast["lower"])
        performance = []
        for (version, horizon), values in sorted(groups.items()):
            errors = np.asarray(values["errors"])
            performance.append({
                "version": version, "horizon_hours": horizon, "forecast_count": values["forecast_count"],
                "observed_count": len(errors), "mae": float(np.abs(errors).mean()) if len(errors) else None,
                "rmse": float(np.sqrt((errors ** 2).mean())) if len(errors) else None,
                "interval_coverage": float(np.mean(values["covered"])) if len(errors) else None,
                "mean_interval_width": float(np.mean(values["widths"])) if len(errors) else None,
            })
        latency = [row["latency_ms"] for row in rows]
        return {
            "forecast_requests": len(rows), "input_failures": failures,
            "pending_outcomes": pending, "missing_completed_outcomes": missing,
            "outcome_as_of": cutoff.isoformat() if cutoff is not None else None,
            "missingness_note": "Without an explicit source-clock as_of, unobserved targets are pending, not classified as overdue.",
            "latency_ms": {"p50": float(np.percentile(latency, 50)) if latency else None,
                           "p95": float(np.percentile(latency, 95)) if latency else None},
            "performance": performance,
            "latest_drift": json.loads(rows[-1]["drift"]) if rows else None,
            "unit": UNIT,
        }
