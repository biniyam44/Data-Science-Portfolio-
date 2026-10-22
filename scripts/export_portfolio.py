"""Export checked results to the local portfolio checkout; never deploy or push."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from forecasting.data import DATASETS, load_data
from forecasting.evaluation import calibration_width, implementation_sha256, make_origin_plan
from forecasting.models import MODEL_NAMES


ROOT = Path(__file__).resolve().parents[1]


def validate_results(result_dir, data_path):
    summary = json.loads((result_dir / "summary.json").read_text())
    if summary.get("implementation_sha256") != implementation_sha256():
        raise ValueError("Forecasting implementation has changed; rerun the comparison before exporting")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if summary["input_sha256"] != digest:
        raise ValueError(f"The input has changed since {result_dir.name} was evaluated")
    dataset = summary["dataset"]
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset in saved results")
    settings = DATASETS[dataset]
    data = load_data(data_path, dataset)
    normalized_digest = hashlib.sha256(data.to_csv(float_format="%.15g").encode("utf-8")).hexdigest()
    if summary["normalized_input_sha256"] != normalized_digest:
        raise ValueError("Evaluated measurements differ from the source dataset")
    config = summary["config"]
    models = config["models"]
    if not isinstance(models, list) or not models or len(set(models)) != len(models) or set(models) - set(MODEL_NAMES):
        raise ValueError("Saved model list must contain distinct supported models")
    if not isinstance(config["quick"], bool):
        raise ValueError("Saved quick mode must be a boolean")
    if summary["horizons"] != list(settings.horizons) or summary["unit"] != settings.unit or summary["frequency"] != settings.frequency:
        raise ValueError("Saved horizons, units or frequency differ from the dataset contract")
    plan, expected_splits = make_origin_plan(data, dataset, quick=config["quick"])
    saved_splits = json.loads((result_dir / "splits.json").read_text())
    if summary["splits"] != expected_splits or saved_splits != expected_splits:
        raise ValueError("Saved splits differ from the source-derived evaluation plan")
    expected_data_summary = {
        "start": data.index[0].isoformat(), "end": data.index[-1].isoformat(), "n_rows": len(data),
    }
    if summary["data_summary"] != expected_data_summary:
        raise ValueError("Saved data bounds or row count differ from the source")
    predictions = pd.read_csv(result_dir / "predictions.csv")
    metrics = pd.read_csv(result_dir / "metrics.csv")
    if predictions.empty or metrics.empty:
        raise ValueError("No evaluated forecasts to export")
    required = {
        "dataset", "phase", "model", "origin", "timestamp", "horizon", "actual", "prediction",
        "lower", "upper", "interval_level", "scale",
    }
    if not required.issubset(predictions):
        raise ValueError("Saved predictions are missing required columns")
    if not predictions.dataset.eq(dataset).all() or not metrics.dataset.eq(dataset).all():
        raise ValueError("Saved rows contain another dataset")
    for column in ("origin", "timestamp"):
        dates = pd.DatetimeIndex(pd.to_datetime(predictions[column], errors="raise"))
        if dates.hasnans or dates.tz is not None:
            raise ValueError("Forecast dates must be present and source-local without timezone offsets")
        predictions[column] = dates
    if not np.isfinite(predictions[["actual", "prediction"]].to_numpy()).all():
        raise ValueError("Predictions contain non-finite values")
    if (predictions.prediction < 0).any():
        raise ValueError("Saved forecasts violate the nonnegative forecast policy")
    if predictions.duplicated(["phase", "model", "origin", "horizon"]).any():
        raise ValueError("Forecast keys are not unique")
    source_rows = {}
    for phase, origins in plan.items():
        for origin_index in origins:
            history = data.y.iloc[:origin_index + 1].to_numpy(dtype=float)
            lag = settings.seasonal_lag
            scale = float(np.mean(np.abs(history[lag:] - history[:-lag])))
            for model in models:
                for horizon in settings.horizons:
                    source_rows[(phase, model, data.index[origin_index], horizon)] = (
                        data.index[origin_index + horizon], float(data.y.iloc[origin_index + horizon]), scale,
                    )
    actual_forecast_keys = set(predictions[["phase", "model", "origin", "horizon"]].itertuples(index=False, name=None))
    if actual_forecast_keys != set(source_rows):
        raise ValueError("Forecast keys do not match every model, origin and horizon in the evaluation plan")
    for row in predictions.itertuples(index=False):
        target_date, actual, scale = source_rows[(row.phase, row.model, row.origin, row.horizon)]
        if row.timestamp != target_date or not np.isclose(row.actual, actual, rtol=1e-9, atol=1e-9):
            raise ValueError("Saved target dates or actuals do not match the source observations")
        if not np.isclose(row.scale, scale, rtol=1e-9, atol=1e-9):
            raise ValueError("Saved MASE scale uses a different history than the forecast origin")
    widths = {}
    for (model, horizon), rows in predictions.loc[predictions.phase.eq("calibration")].groupby(["model", "horizon"]):
        widths[(model, horizon)] = calibration_width((rows.actual - rows.prediction).abs().to_numpy())
    saved_widths = summary["calibration_widths"]
    width_keys = [(row["model"], row["horizon"]) for row in saved_widths]
    if len(set(width_keys)) != len(width_keys) or set(width_keys) != set(widths):
        raise ValueError("Saved calibration widths do not cover the evaluated model-horizon pairs")
    for saved in saved_widths:
        if not np.isclose(saved["width"], widths[(saved["model"], saved["horizon"])], rtol=1e-7, atol=1e-8):
            raise ValueError("Saved interval widths differ from calibration-only residuals")
    for (model, horizon), rows in predictions.loc[predictions.phase.eq("test")].groupby(["model", "horizon"]):
        width = widths[(model, horizon)]
        if not np.allclose(rows.lower, np.maximum(0.0, rows.prediction - width), rtol=1e-7, atol=1e-8) or not np.allclose(rows.upper, rows.prediction + width, rtol=1e-7, atol=1e-8):
            raise ValueError("Test intervals do not use the frozen calibration widths")
        if not rows.interval_level.eq(0.8).all():
            raise ValueError("Saved interval levels differ from the nominal 80% policy")
    if predictions.loc[predictions.phase.ne("test"), ["lower", "upper", "interval_level"]].notna().any().any():
        raise ValueError("Validation or calibration rows unexpectedly contain calibrated test intervals")
    expected_keys = set()
    for key, rows in predictions.groupby(["phase", "model", "horizon"]):
        phase, model, horizon = key
        expected_keys.add(key)
        saved = metrics.loc[metrics.phase.eq(phase) & metrics.model.eq(model) & metrics.horizon.eq(horizon)]
        if len(saved) != 1:
            raise ValueError(f"Missing or repeated metric group: {key}")
        saved = saved.iloc[0]
        error = rows.prediction - rows.actual
        nonzero = rows.actual.ne(0)
        valid_scale = np.isfinite(rows.scale) & rows.scale.gt(0)
        checks = [(len(rows), saved.n), (np.abs(error).mean(), saved.mae),
                  (np.sqrt(np.square(error).mean()), saved.rmse),
                  (int(nonzero.sum()), saved.mape_n), (int(valid_scale.sum()), saved.mase_n),
                  ((error[nonzero].abs() / rows.actual[nonzero].abs()).mean() * 100 if nonzero.any() else np.nan, saved.mape_pct),
                  ((error[valid_scale].abs() / rows.scale[valid_scale]).mean() if valid_scale.any() else np.nan, saved.mase)]
        if phase == "test":
            if not np.isfinite(rows[["lower", "upper"]].to_numpy()).all():
                raise ValueError("Test intervals are missing")
            if (rows.lower > rows.upper).any():
                raise ValueError("An interval has reversed bounds")
            tolerance = 1e-9 * np.maximum(1.0, rows.actual.abs())
            coverage = ((rows.actual >= rows.lower - tolerance) & (rows.actual <= rows.upper + tolerance)).mean()
            checks.extend([(coverage, saved.coverage), ((rows.upper - rows.lower).mean(), saved.mean_interval_width)])
        else:
            checks.extend([(np.nan, saved.coverage), (np.nan, saved.mean_interval_width)])
        if not all(np.isclose(a, b, rtol=1e-7, atol=1e-8, equal_nan=True) for a, b in checks):
            raise ValueError(f"Metrics do not match prediction rows: {key}")
    actual_keys = set(metrics[["phase", "model", "horizon"]].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ValueError("Metric groups differ from prediction groups")
    validation = metrics.loc[metrics.phase.eq("validation")]
    if set(summary["selected_models"]) != {str(horizon) for horizon in settings.horizons}:
        raise ValueError("Selected models do not cover the evaluated horizons")
    for horizon in summary["horizons"]:
        rows = validation.loc[validation.horizon.eq(horizon)]
        model = summary["selected_models"][str(horizon)]
        selected = rows.loc[rows.model.eq(model)]
        if len(selected) != 1 or not np.isclose(selected.iloc[0].mae, rows.mae.min()):
            raise ValueError("A selected model does not minimize validation MAE")
    summary["metrics"] = json.loads(metrics.to_json(orient="records", double_precision=12))
    return predictions, metrics, summary


def require_full_comparison(summary):
    """Do not publish development runs or incomplete model comparisons."""
    if summary["config"]["quick"] is not False:
        raise ValueError("Quick results are smoke checks and cannot be exported to the portfolio")
    if set(summary["config"]["models"]) != set(MODEL_NAMES):
        raise ValueError("Portfolio results must include all six comparison models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="Existing local website checkout, not a URL")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not (destination / "index.html").is_file():
        parser.error("--destination must contain the existing portfolio index.html")
    output = destination / "forecasting" / "data"
    checked = []
    for dataset in ("crypto", "energy"):
        result_dir = ROOT / "results" / dataset
        data_path = ROOT / "data" / ("dc.csv" if dataset == "crypto" else "appliances_hourly.csv")
        predictions, _, summary = validate_results(result_dir, data_path)
        if summary["dataset"] != dataset:
            raise ValueError("The result directory contains a different dataset")
        require_full_comparison(summary)
        checked.append((dataset, result_dir, predictions, summary))
    output.mkdir(parents=True, exist_ok=True)
    for dataset, result_dir, predictions, summary in checked:
        payload = dict(summary)
        payload["predictions"] = json.loads(predictions.loc[predictions.phase.eq("test")].to_json(
            orient="records", date_format="iso", double_precision=12))
        (output / f"{dataset}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        for name in ("predictions.csv", "metrics.csv", "summary.json", "splits.json"):
            shutil.copyfile(result_dir / name, output / f"{dataset}-{name}")
        print(f"Exported checked {dataset} results to {output}")


if __name__ == "__main__":
    main()
