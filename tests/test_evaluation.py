import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from forecasting import load_data, run_experiment, write_artifacts
from forecasting.data import DATASETS
from forecasting.evaluation import calibration_width, make_origin_plan, summarize_metrics
from scripts.export_portfolio import validate_results


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def crypto():
    return load_data(ROOT / "data/dc.csv", "crypto")


@pytest.mark.parametrize("dataset", ["crypto", "energy"])
def test_all_target_dates_stay_inside_their_split(dataset):
    data = load_data(ROOT / DATASETS[dataset].default_path, dataset)
    plan, splits = make_origin_plan(data, dataset)
    for phase, origins in plan.items():
        start = splits[phase]["start_index"]
        end = splits[phase]["end_index_exclusive"]
        for origin in origins:
            assert origin < end
            assert origin + min(DATASETS[dataset].horizons) >= start
            assert origin + max(DATASETS[dataset].horizons) < end
    assert max(plan["calibration"]) + max(DATASETS[dataset].horizons) < splits["test"]["start_index"]
    for origins in plan.values():
        assert len({data.index[i].dayofweek for i in origins}) == 7
        if dataset == "energy":
            assert len({data.index[i].hour for i in origins}) == 24


def test_calibration_order_statistic_and_invalid_inputs():
    assert calibration_width([1, 2, 3, 4, 5]) == 5
    assert calibration_width(list(range(1, 26))) == 21
    for bad in ([], [np.nan], [-1], [1, 2]):
        with pytest.raises(ValueError):
            calibration_width(bad)


def test_metric_units_and_zero_denominators():
    frame = pd.DataFrame({
        "dataset": ["example"] * 3, "phase": ["test"] * 3,
        "model": ["naive"] * 3, "horizon": [1] * 3,
        "actual": [100, 200, 0], "prediction": [110, 180, 3],
        "scale": [10, 20, 0], "lower": [90, 170, 0], "upper": [120, 210, 5],
    })
    row = summarize_metrics(frame).iloc[0]
    assert row.mape_pct == pytest.approx(10)
    assert row.mape_n == 2
    assert row.mase == pytest.approx(1)
    assert row.mase_n == 2
    assert row.mae == 11
    assert row.coverage == 1


def test_interval_boundary_rounding_does_not_change_coverage():
    frame = pd.DataFrame({
        "dataset": ["example"] * 2, "phase": ["test"] * 2,
        "model": ["naive"] * 2, "horizon": [1] * 2,
        "actual": [0.55, 0.56], "prediction": [0.5, 0.5],
        "scale": [1, 1], "lower": [0.1, 0.1], "upper": [0.5499999999999999, 0.55],
    })
    assert summarize_metrics(frame).iloc[0].coverage == 0.5


def test_holdout_cannot_change_selection_or_intervals(crypto):
    before, _, summary_before = run_experiment(crypto, "crypto", ["naive", "drift"], quick=True)
    changed = crypto.copy()
    changed.iloc[int(len(changed) * 0.8):, 0] *= 4
    after, _, summary_after = run_experiment(changed, "crypto", ["naive", "drift"], quick=True)
    assert summary_before["selected_models"] == summary_after["selected_models"]
    assert summary_before["calibration_widths"] == summary_after["calibration_widths"]
    pd.testing.assert_frame_equal(before.loc[before.phase.ne("test")], after.loc[after.phase.ne("test")])
    assert not before.loc[before.phase.eq("test"), "actual"].equals(after.loc[after.phase.eq("test"), "actual"])


def test_identical_forecast_keys_and_history_bounds(crypto, monkeypatch):
    import forecasting.evaluation as evaluation

    seen = []

    def predictor(name, history, horizons, frequency, seasonal_lag):
        seen.append((name, history.index[-1], len(history)))
        return np.repeat(history.y.iloc[-1], len(horizons))

    monkeypatch.setattr(evaluation, "predict_model", predictor)
    predictions, _, _ = run_experiment(crypto, "crypto", ["naive", "drift"], quick=True)
    for model in ("naive", "drift"):
        keys = predictions.loc[predictions.model.eq(model), ["phase", "origin", "timestamp", "horizon"]].reset_index(drop=True)
        if model == "naive":
            baseline = keys
        else:
            pd.testing.assert_frame_equal(baseline, keys)
    for _, last, count in seen:
        assert crypto.index[count - 1] == last
    assert (predictions.timestamp > predictions.origin).all()


def test_saved_results_are_audited_and_tampering_rejected(crypto, tmp_path):
    predictions, metrics, summary = run_experiment(crypto, "crypto", ["naive"], quick=True)
    write_artifacts(predictions, metrics, summary, tmp_path)
    validate_results(tmp_path, ROOT / "data/dc.csv")
    assert json.loads((tmp_path / "summary.json").read_text())["config"]["quick"] is True
    saved = pd.read_csv(tmp_path / "metrics.csv")
    saved.loc[0, "mae"] += 100
    saved.to_csv(tmp_path / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="Metrics do not match"):
        validate_results(tmp_path, ROOT / "data/dc.csv")


def test_cli_runs_outside_repository_directory(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "time_series_forecasting.py"), "--dataset", "crypto",
        "--models", "naive", "--quick", "--output", str(tmp_path / "run"),
    ], cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run/predictions.csv").is_file()


def test_cli_reports_invalid_data_path(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "time_series_forecasting.py"), "--data-path", str(tmp_path / "absent.csv"),
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "Forecasting failed" in result.stderr
