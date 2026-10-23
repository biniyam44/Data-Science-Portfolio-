"""Locked independent-household replication with observed-only evaluation targets."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd

from .evaluation import calibration_width, implementation_sha256, summarize_metrics, _package_versions
from .models import MODEL_NAMES, MODEL_PARAMETERS, predict_model

ROOT = Path(__file__).resolve().parents[1]
DATASET = "household_holdout"
WORKFLOW_FILES = ("forecasting/holdout.py", "scripts/prepare_holdout.py", "scripts/run_holdout.py")


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol["implementation_sha256"] != implementation_sha256():
        raise ValueError("Core implementation changed after protocol lock; do not relabel a new run as untouched.")
    if tuple(protocol["models"]) != MODEL_NAMES or protocol["horizons"] != [1, 6, 24]:
        raise ValueError("This runner requires the six fixed models and 1/6/24 hour protocol.")
    if protocol["split_fractions"] != [0.4, 0.2, 0.2, 0.2]:
        raise ValueError("The locked calendar split cannot be changed.")
    for key, expected in (("frequency", "h"), ("origin_step", 25), ("seasonal_lag", 24),
                          ("interval_level", 0.8), ("seed", 42)):
        if protocol[key] != expected:
            raise ValueError(f"Unsupported locked setting: {key}")
    return protocol


def exclusive_json(path: Path, value: dict) -> None:
    """Do not overwrite a protocol, acquisition or attempt record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    path.chmod(0o444)


def freeze_protocol(protocol_path: Path, lock_path: Path) -> dict:
    protocol = read_protocol(protocol_path)
    expected = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": digest_file(protocol_path),
        "implementation_sha256": implementation_sha256(),
        "workflow_sha256": {name: digest_file(ROOT / name) for name in WORKFLOW_FILES},
        "model_parameters": MODEL_PARAMETERS,
    }
    if lock_path.exists():
        saved = json.loads(lock_path.read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("The protocol or implementation differs from the acquisition lock.")
        return saved
    saved = {
        **expected, "locked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "locked before target download and measurement inspection",
        "scope": "Local development record, not a third-party timestamp or proof of model-pretraining exclusion.",
    }
    exclusive_json(lock_path, saved)
    return saved


def load_hourly(path: Path, protocol: dict) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if list(frame.columns) != ["timestamp", "y", "observed", "finite_minutes"]:
        raise ValueError("Unexpected holdout columns.")
    frame = frame.set_index("timestamp")
    expected = pd.date_range(protocol["window"]["start_inclusive"],
                             protocol["window"]["end_exclusive"], freq="h", inclusive="left")
    if not frame.index.equals(expected):
        raise ValueError("Holdout timestamps must match the locked complete hourly calendar.")
    if not frame["observed"].isin([True, False]).all():
        raise ValueError("Observed flags must be Boolean.")
    counts = pd.to_numeric(frame["finite_minutes"], errors="raise")
    if not (counts.between(0, 60) & (counts == np.floor(counts))).all():
        raise ValueError("Finite minute counts must be integers from zero through sixty.")
    observed = counts == 60
    if not (frame["observed"] == observed).all():
        raise ValueError("An observed target needs exactly sixty finite minute readings.")
    y = pd.to_numeric(frame["y"], errors="raise")
    if not np.isfinite(y[observed]).all() or (y[observed] < 0).any() or y[~observed].notna().any():
        raise ValueError("Only complete hours may contain finite nonnegative target values.")
    frame.index = pd.DatetimeIndex(frame.index, freq="h", name="timestamp")
    return frame


def history_at(frame: pd.DataFrame, origin: int) -> tuple[pd.DataFrame, int]:
    """Forward-fill only the prefix available at issue time, never its future."""
    prefix = frame.iloc[:origin + 1]["y"]
    first = prefix.first_valid_index()
    if first is None:
        raise ValueError("No observed hourly history before the origin.")
    prefix = prefix.loc[first:]
    imputed = int(prefix.isna().sum())
    history = prefix.ffill().to_frame("y")
    if len(history) < 168:
        raise ValueError("Fewer than 168 historical hours after removing leading missing values.")
    history.index = pd.DatetimeIndex(history.index, freq="h", name="timestamp")
    return history, imputed


def make_plan(frame: pd.DataFrame, protocol: dict) -> tuple[dict[str, list[int]], dict]:
    n = len(frame)
    bounds = [0, int(n * .4), int(n * .6), int(n * .8), n]
    horizons = protocol["horizons"]
    plan, splits = {}, {}
    for i, phase in enumerate(("training", "validation", "calibration", "test")):
        start, end = bounds[i:i + 2]
        if start == end:
            raise ValueError("The holdout calendar is too short to split.")
        splits[phase] = {
            "start": frame.index[start].isoformat(), "end": frame.index[end - 1].isoformat(),
            "start_index": start, "end_index_exclusive": end, "rows": end - start,
            "observed_hours": int(frame.iloc[start:end]["observed"].sum()),
            "missing_hours": int((~frame.iloc[start:end]["observed"]).sum()),
        }
        if phase == "training":
            continue
        candidates = list(range(start - 1, end - max(horizons), protocol["origin_step"]))
        accepted, excluded = [], []
        for origin in candidates:
            missing = [frame.index[origin + h].isoformat() for h in horizons
                       if not frame.iloc[origin + h]["observed"]]
            reason = "one or more requested target hours are incomplete" if missing else None
            try:
                history_at(frame, origin)
            except ValueError as error:
                reason = str(error)
            if reason:
                excluded.append({"origin": frame.index[origin].isoformat(), "reason": reason,
                                 "incomplete_targets": missing})
            else:
                accepted.append(origin)
        if not accepted or (phase == "calibration" and len(accepted) < 5):
            raise ValueError(f"Insufficient complete target origins in {phase}; do not change the locked window.")
        plan[phase] = accepted
        splits[phase].update({
            "candidate_origins": len(candidates), "n_origins": len(accepted), "excluded_origins": excluded,
            "origins": [frame.index[j].isoformat() for j in accepted],
            "origin_hour_counts": dict(sorted(Counter(str(frame.index[j].hour) for j in accepted).items())),
            "origin_weekday_counts": dict(sorted(Counter(frame.index[j].day_name() for j in accepted).items())),
        })
    return plan, splits


def run_replication(frame: pd.DataFrame, protocol: dict, progress=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    plan, splits = make_plan(frame, protocol)
    rows, widths, selected = [], {}, {}
    notices = Counter()
    for phase, origins in plan.items():
        for origin in origins:
            history, imputed = history_at(frame, origin)
            y = history["y"].to_numpy(dtype=float)
            scale = float(np.abs(y[24:] - y[:-24]).mean())
            for model in protocol["models"]:
                if progress:
                    progress(f"{phase}: {history.index[-1].isoformat()} {model}")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimates = predict_model(model, history, protocol["horizons"], "h", 24)
                for warning in caught:
                    notices[(model, phase, warning.category.__name__, str(warning.message))] += 1
                for horizon, prediction in zip(protocol["horizons"], estimates, strict=True):
                    width = widths.get((model, horizon)) if phase == "test" else None
                    rows.append({
                        "dataset": DATASET, "phase": phase, "model": model,
                        "origin": history.index[-1], "timestamp": frame.index[origin + horizon],
                        "horizon": horizon, "actual": float(frame.iloc[origin + horizon]["y"]),
                        "prediction": float(prediction), "scale": scale,
                        "lower": max(0., float(prediction) - width) if width is not None else None,
                        "upper": float(prediction) + width if width is not None else None,
                        "interval_level": .8 if width is not None else None,
                        "history_rows": len(history), "history_imputed_hours": imputed,
                        "origin_imputed": not bool(frame.iloc[origin]["observed"]), "target_observed": True,
                    })
        phase_rows = pd.DataFrame([row for row in rows if row["phase"] == phase])
        if phase == "validation":
            ranks = summarize_metrics(phase_rows).sort_values(["horizon", "mae", "model"])
            selected = {str(int(row.horizon)): row.model for row in ranks.groupby("horizon").first().reset_index().itertuples()}
        elif phase == "calibration":
            for (model, horizon), group in phase_rows.groupby(["model", "horizon"]):
                widths[(model, horizon)] = calibration_width(np.abs(group.actual - group.prediction), .8)
    predictions = pd.DataFrame(rows)
    metrics = summarize_metrics(predictions)
    test = predictions[predictions.phase == "test"].copy()
    test["absolute_error"] = abs(test.prediction - test.actual)
    summary = {
        "schema_version": 1, "dataset": DATASET,
        "title": "Locked household energy replication", "unit": "kWh", "frequency": "h",
        "horizons": protocol["horizons"], "protocol_id": protocol["protocol_id"],
        "source": protocol["source"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": implementation_sha256(),
        "config": {"models": protocol["models"], "model_parameters": MODEL_PARAMETERS,
                   "package_versions": _package_versions(), "python_version": platform.python_version(),
                   "origin_step": 25, "interval_level": .8, "selection": protocol["selection"],
                   "rolling_policy": protocol["rolling_policy"]},
        "splits": splits, "selected_models": selected,
        "metrics": json.loads(metrics.to_json(orient="records", double_precision=15)),
        "calibration_widths": [{"model": m, "horizon": int(h), "width": w} for (m, h), w in sorted(widths.items())],
        "largest_test_errors": json.loads(test.sort_values("absolute_error", ascending=False).groupby(
            ["model", "horizon"]).head(3).to_json(orient="records", date_format="iso", double_precision=15)),
        "warnings": [{"model": m, "phase": p, "category": c, "message": msg, "count": count}
                     for (m, p, c, msg), count in sorted(notices.items())],
        "limitations": [
            "A different household and whole-household consumption target, not a future holdout of appliance energy.",
            "Protocol locked locally before downloading and inspecting target measurements; no external preregistration or pretraining exclusion claim.",
            "Only complete sixty-minute target hours are scored; exclusions may make coverage unrepresentative during data outages.",
            "Forward-filled historical inputs and training labels are flagged and can bias fits during long outages; evaluation labels are never filled.",
            "Expanding histories may include past test observations available before later forecast origins.",
            "Sparse origins, one household and four months do not establish production or cross-season generalization.",
            "Serial dependence means nominal 80% intervals have no guaranteed coverage.",
        ],
    }
    return predictions, metrics, summary
