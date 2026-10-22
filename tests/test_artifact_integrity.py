"""Saved results must match the source and complete chronological evaluation."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from forecasting import load_data, run_experiment, write_artifacts
from forecasting.evaluation import summarize_metrics
from forecasting.models import MODEL_NAMES
from scripts.export_portfolio import require_full_comparison, validate_results


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/dc.csv"


@pytest.fixture(scope="module")
def baseline_results():
    return run_experiment(load_data(SOURCE, "crypto"), "crypto", ["naive"], quick=True)


def save_changed_rows(location, predictions, summary):
    write_artifacts(predictions, summarize_metrics(predictions), copy.deepcopy(summary), location)


def test_unchanged_baseline_results_validate(baseline_results, tmp_path):
    predictions, metrics, summary = baseline_results
    write_artifacts(predictions, metrics, summary, tmp_path)
    validate_results(tmp_path, SOURCE)


def test_fabricated_actuals_fail_even_when_metrics_are_recomputed(baseline_results, tmp_path):
    predictions, _, summary = baseline_results
    changed = predictions.copy()
    test = changed.phase.eq("test")
    changed.loc[test, "actual"] = changed.loc[test, "prediction"]
    save_changed_rows(tmp_path, changed, summary)
    with pytest.raises(ValueError, match="actuals do not match"):
        validate_results(tmp_path, SOURCE)


def test_missing_origin_fails_even_when_metrics_are_recomputed(baseline_results, tmp_path):
    predictions, _, summary = baseline_results
    first = predictions.loc[predictions.phase.eq("test"), "origin"].min()
    changed = predictions.loc[~(predictions.phase.eq("test") & predictions.origin.eq(first))]
    save_changed_rows(tmp_path, changed, summary)
    with pytest.raises(ValueError, match="Forecast keys do not match"):
        validate_results(tmp_path, SOURCE)


@pytest.mark.parametrize("column", ["origin", "timestamp"])
def test_shifted_forecast_dates_are_rejected(column, baseline_results, tmp_path):
    predictions, _, summary = baseline_results
    changed = predictions.copy()
    test = changed.phase.eq("test")
    changed.loc[test, column] = changed.loc[test, column] + pd.Timedelta(days=1)
    save_changed_rows(tmp_path, changed, summary)
    with pytest.raises(ValueError, match="Forecast keys|target dates"):
        validate_results(tmp_path, SOURCE)


def test_foreign_mase_scale_is_rejected(baseline_results, tmp_path):
    predictions, _, summary = baseline_results
    changed = predictions.copy()
    changed.loc[changed.phase.eq("test"), "scale"] *= 2
    save_changed_rows(tmp_path, changed, summary)
    with pytest.raises(ValueError, match="MASE scale"):
        validate_results(tmp_path, SOURCE)


def test_inflated_test_intervals_cannot_claim_calibration(baseline_results, tmp_path):
    predictions, _, summary = baseline_results
    changed = predictions.copy()
    changed.loc[changed.phase.eq("test"), "upper"] += 1000
    save_changed_rows(tmp_path, changed, summary)
    with pytest.raises(ValueError, match="frozen calibration widths"):
        validate_results(tmp_path, SOURCE)


def test_changed_calibration_metadata_is_rejected(baseline_results, tmp_path):
    predictions, metrics, summary = baseline_results
    changed = copy.deepcopy(summary)
    changed["calibration_widths"][0]["width"] += 10
    write_artifacts(predictions, metrics, changed, tmp_path)
    with pytest.raises(ValueError, match="calibration-only residuals"):
        validate_results(tmp_path, SOURCE)


def test_downloaded_split_record_must_match_source_plan(baseline_results, tmp_path):
    predictions, metrics, summary = baseline_results
    write_artifacts(predictions, metrics, summary, tmp_path)
    splits_path = tmp_path / "splits.json"
    changed = json.loads(splits_path.read_text())
    changed["test"]["rows"] -= 1
    splits_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Saved splits"):
        validate_results(tmp_path, SOURCE)


@pytest.mark.parametrize("metric", ["mase", "mase_n"])
def test_mase_metrics_are_checked(metric, baseline_results, tmp_path):
    predictions, metrics, summary = baseline_results
    changed = metrics.copy()
    changed.loc[0, metric] += 1
    write_artifacts(predictions, changed, summary, tmp_path)
    with pytest.raises(ValueError, match="Metrics do not match"):
        validate_results(tmp_path, SOURCE)


def test_unit_metadata_cannot_silently_change(baseline_results, tmp_path):
    predictions, metrics, summary = baseline_results
    changed = copy.deepcopy(summary)
    changed["unit"] = "kWh per hour"
    write_artifacts(predictions, metrics, changed, tmp_path)
    with pytest.raises(ValueError, match="units"):
        validate_results(tmp_path, SOURCE)


def test_old_implementation_results_are_rejected(baseline_results, tmp_path):
    predictions, metrics, summary = baseline_results
    changed = copy.deepcopy(summary)
    changed["implementation_sha256"] = "0" * 64
    write_artifacts(predictions, metrics, changed, tmp_path)
    with pytest.raises(ValueError, match="implementation has changed"):
        validate_results(tmp_path, SOURCE)


def test_development_runs_are_valid_but_not_publishable(baseline_results):
    _, _, summary = baseline_results
    with pytest.raises(ValueError, match="Quick results"):
        require_full_comparison(summary)
    changed = copy.deepcopy(summary)
    changed["config"]["quick"] = False
    with pytest.raises(ValueError, match="all six"):
        require_full_comparison(changed)
    changed["config"]["models"] = list(MODEL_NAMES)
    require_full_comparison(changed)
