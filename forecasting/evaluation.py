"""Separate selection, interval calibration and final rolling evaluation."""

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd

from .data import DATASETS, validate_data
from .models import MODEL_NAMES, MODEL_PARAMETERS, predict_model

INTERVAL_LEVEL = 0.8


def make_origin_plan(
    data: pd.DataFrame, dataset: str, quick: bool = False
) -> tuple[dict[str, list[int]], dict[str, dict]]:
    """Use the same origins for all models, with targets contained in each block."""
    settings = DATASETS[dataset]
    n = len(data)
    boundaries = [0, int(n * 0.4), int(n * 0.6), int(n * 0.8), n]
    names = ("training", "validation", "calibration", "test")
    plan = {}
    splits = {}
    for index, name in enumerate(names):
        start, end = boundaries[index:index + 2]
        if start == end:
            raise ValueError("The dataset is too short to split.")
        splits[name] = {
            "start": data.index[start].isoformat(), "end": data.index[end - 1].isoformat(),
            "rows": end - start, "start_index": start, "end_index_exclusive": end,
        }
        if name == "training":
            continue
        origins = list(range(start - 1, end - max(settings.horizons), settings.origin_step))
        if not origins:
            raise ValueError(f"The {name} block is shorter than the maximum horizon.")
        if name == "calibration" and len(origins) < 5:
            raise ValueError("At least five calibration origins are required for 80% intervals.")
        if quick:
            count = min(len(origins), 5 if name == "calibration" else 2)
            origins = [origins[i] for i in np.linspace(0, len(origins) - 1, count, dtype=int)]
        plan[name] = origins
        splits[name]["n_origins"] = len(origins)
        splits[name]["origins"] = [data.index[i].isoformat() for i in origins]
        splits[name]["origin_weekday_counts"] = dict(sorted(Counter(
            data.index[i].day_name() for i in origins
        ).items()))
        if settings.frequency == "h":
            splits[name]["origin_hour_counts"] = dict(sorted(Counter(
                f"{data.index[i].hour:02d}:00" for i in origins
            ).items()))
    return plan, splits


def calibration_width(errors: np.ndarray | list[float], level: float = INTERVAL_LEVEL) -> float:
    """Finite-sample absolute-residual order statistic, not a coverage guarantee."""
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Calibration errors must be finite, nonnegative and nonempty.")
    if not 0 < level < 1:
        raise ValueError("The interval level must be between zero and one.")
    rank = math.ceil((len(values) + 1) * level)
    if rank > len(values):
        raise ValueError("Too few calibration errors for the requested interval level.")
    return float(np.sort(values)[rank - 1])


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAPE is a percentage; zero actuals and zero MASE scales are counted separately."""
    records = []
    for (dataset, phase, model, horizon), group in predictions.groupby(
        ["dataset", "phase", "model", "horizon"], sort=True
    ):
        actual = group["actual"].to_numpy(dtype=float)
        error = group["prediction"].to_numpy(dtype=float) - actual
        absolute = np.abs(error)
        nonzero_actual = actual != 0
        scales = group["scale"].to_numpy(dtype=float)
        valid_scales = np.isfinite(scales) & (scales > 0)
        intervals = group["lower"].notna() & group["upper"].notna()
        coverage = None
        width = None
        if intervals.any():
            bounded = group.loc[intervals]
            tolerance = 1e-9 * np.maximum(1.0, np.abs(bounded["actual"]))
            coverage = float(((bounded["actual"] >= bounded["lower"] - tolerance) &
                              (bounded["actual"] <= bounded["upper"] + tolerance)).mean())
            width = float((bounded["upper"] - bounded["lower"]).mean())
        records.append({
            "dataset": dataset, "phase": phase, "model": model, "horizon": int(horizon),
            "n": len(group), "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mape_pct": float(np.mean(absolute[nonzero_actual] / np.abs(actual[nonzero_actual])) * 100)
            if nonzero_actual.any() else None,
            "mape_n": int(nonzero_actual.sum()),
            "mase": float(np.mean(absolute[valid_scales] / scales[valid_scales]))
            if valid_scales.any() else None,
            "mase_n": int(valid_scales.sum()), "coverage": coverage, "mean_interval_width": width,
        })
    return pd.DataFrame(records)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def _package_versions() -> dict[str, str]:
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "prophet", "cmdstanpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return packages


def implementation_sha256() -> str:
    """Fingerprint the modules that define the experiment, excluding its plots."""
    root = Path(__file__).resolve().parent
    digest = sha256()
    for name in ("__init__.py", "data.py", "features.py", "models.py", "evaluation.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def run_experiment(
    data: pd.DataFrame,
    dataset: str = "crypto",
    models: tuple[str, ...] | list[str] | None = None,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate fixed models; choose on validation and freeze widths before test."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}.")
    settings = DATASETS[dataset]
    data = validate_data(data, settings.frequency)
    chosen_models = list(MODEL_NAMES if models is None else models)
    if not chosen_models or len(set(chosen_models)) != len(chosen_models):
        raise ValueError("Supply at least one model, without duplicates.")
    if set(chosen_models) - set(MODEL_NAMES):
        raise ValueError(f"Choose models from {', '.join(MODEL_NAMES)}.")
    plan, splits = make_origin_plan(data, dataset, quick)
    rows = []
    selected = {}
    widths = {}
    notices: Counter = Counter()
    for phase, origins in plan.items():
        for origin_index in origins:
            history = data.iloc[:origin_index + 1]
            y = history["y"].to_numpy(dtype=float)
            scale = float(np.mean(np.abs(y[settings.seasonal_lag:] - y[:-settings.seasonal_lag])))
            for model in chosen_models:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        forecasts = predict_model(
                            model, history, settings.horizons, settings.frequency, settings.seasonal_lag
                        )
                    for warning in caught:
                        notices[(model, phase, warning.category.__name__, str(warning.message))] += 1
                except Exception as error:
                    raise RuntimeError(
                        f"{model} failed at {history.index[-1].isoformat()} during {phase}: {error}"
                    ) from error
                for horizon, prediction in zip(settings.horizons, forecasts, strict=True):
                    width = widths.get((model, horizon)) if phase == "test" else None
                    rows.append({
                        "dataset": dataset, "phase": phase, "model": model,
                        "origin": history.index[-1], "timestamp": data.index[origin_index + horizon],
                        "horizon": horizon, "actual": float(data["y"].iloc[origin_index + horizon]),
                        "prediction": float(prediction),
                        "lower": max(0.0, float(prediction) - width) if width is not None else None,
                        "upper": float(prediction) + width if width is not None else None,
                        "interval_level": INTERVAL_LEVEL if width is not None else None, "scale": scale,
                    })
        phase_rows = pd.DataFrame([row for row in rows if row["phase"] == phase])
        if phase == "validation":
            ranking = summarize_metrics(phase_rows).sort_values(["horizon", "mae", "model"])
            selected = {
                str(int(row.horizon)): row.model
                for row in ranking.groupby("horizon").first().reset_index().itertuples()
            }
        elif phase == "calibration":
            for (model, horizon), group in phase_rows.groupby(["model", "horizon"]):
                widths[(model, horizon)] = calibration_width(np.abs(group["actual"] - group["prediction"]))
    predictions = pd.DataFrame(rows)
    metrics = summarize_metrics(predictions)
    normalized_hash = sha256(data.to_csv(float_format="%.15g").encode("utf-8")).hexdigest()
    test = predictions.loc[predictions["phase"] == "test"].copy()
    test["absolute_error"] = (test["prediction"] - test["actual"]).abs()
    largest_errors = test.sort_values("absolute_error", ascending=False).groupby(
        ["model", "horizon"], sort=True
    ).head(3)
    summary = {
        "schema_version": 1, "dataset": dataset, "title": settings.title,
        "unit": settings.unit, "frequency": settings.frequency, "horizons": list(settings.horizons),
        "source": {"title": settings.source_title, "url": settings.source_url, "license": settings.source_license},
        "input_sha256": data.attrs.get("input_sha256", normalized_hash),
        "input_file": data.attrs.get("input_file"), "normalized_input_sha256": normalized_hash,
        "implementation_sha256": implementation_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_summary": {"start": data.index[0].isoformat(), "end": data.index[-1].isoformat(), "n_rows": len(data)},
        "config": {
            "models": chosen_models, "model_parameters": {key: MODEL_PARAMETERS[key] for key in chosen_models},
            "quick": quick, "seed": 42, "split_fractions": [0.4, 0.2, 0.2, 0.2],
            "origin_step": settings.origin_step, "seasonal_lag": settings.seasonal_lag,
            "origin_schedule": "5-day or 25-hour spacing visits all weekdays or hours instead of fixing one clock slot",
            "python_version": platform.python_version(),
            "forecast_issue_time": "after the origin period has ended and its measurements are available",
            "selection_metric": "validation MAE separately per horizon; ties broken by model name",
            "evaluation_policy": "expanding-window rolling refits, including already observed earlier test values",
            "interval_level": INTERVAL_LEVEL,
            "interval_boundary_tolerance": "1e-9 * max(1, abs(actual)) for numeric serialization at inclusive bounds",
            "interval_method": "absolute calibration residual at rank ceil((n+1)*0.8), frozen per model and horizon",
            "negative_forecasts": "clipped to zero for every model; interval lower bounds also clipped",
            "package_versions": _package_versions(),
        },
        "splits": splits, "selected_models": selected, "metrics": _json_records(metrics),
        "calibration_widths": [
            {"model": model, "horizon": int(horizon), "width": width}
            for (model, horizon), width in sorted(widths.items())
        ],
        "largest_test_errors": _json_records(largest_errors),
        "warnings": [
            {"model": model, "phase": phase, "category": category, "message": message, "count": count}
            for (model, phase, category, message), count in sorted(notices.items())
        ],
        "limitations": [
            "Historical experiments, not live forecasts or trading recommendations.",
            "Fixed configurations compare model families, not exhaustive hyperparameter searches.",
            "Selection uses validation only. Test values are not used to choose models or interval widths.",
            "Later test forecasts may learn from earlier test observations once those observations are available.",
            "Calibration errors are serially dependent; nominal 80% intervals have no guaranteed coverage.",
            "Origins are sparse, so metrics do not describe forecasts issued at every possible timestamp.",
            "Development runs preceded a calendar-sampling correction; these historical results are not fresh external validation.",
            "MAPE excludes zero actuals; MASE excludes zero seasonal training scales, with counts reported.",
        ] + (["Quick mode is a smoke test with fewer origins, not the full comparison."] if quick else []),
    }
    return predictions, metrics, summary


def write_artifacts(
    predictions: pd.DataFrame, metrics: pd.DataFrame, summary: dict, output_dir: str | Path
) -> None:
    """Save inspectable records; JSON cannot silently contain NaN or infinity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, float_format="%.12g")
    metrics.to_csv(output_dir / "metrics.csv", index=False, float_format="%.12g")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output_dir / "splits.json").write_text(json.dumps(summary["splits"], indent=2, allow_nan=False) + "\n")
