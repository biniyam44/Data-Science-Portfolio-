from copy import deepcopy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from forecasting.monitoring import Monitor, drift_report
from forecasting.serving import UNIT, _profile


@pytest.fixture
def prediction():
    history = pd.DataFrame({"y": np.linspace(0.5, 1.5, 169)},
                           index=pd.date_range("2020-01-01", periods=169, freq="h", name="timestamp"))
    origin = history.index[-1]
    result = {
        "version": "energy-" + "a" * 24, "origin": origin.isoformat(), "unit": UNIT,
        "forecasts": [{"target_timestamp": (origin + pd.Timedelta(hours=h)).isoformat(),
                       "horizon_hours": h, "model": "naive", "prediction": 1.0, "lower": 0.5, "upper": 1.5}
                      for h in (1, 6, 24)],
    }
    return history, result


def test_request_retries_are_idempotent_and_conflicts_rejected(tmp_path, prediction):
    history, result = prediction
    monitor = Monitor(tmp_path / "events.sqlite")
    profile = _profile(history["y"].to_numpy())
    assert monitor.record_forecast("r-1", result, 5.0, history, profile)["created"]
    assert not monitor.record_forecast("r-1", result, 10.0, history, profile)["created"]
    assert monitor.report()["forecast_requests"] == 1
    assert monitor.get_forecast("r-1") == result
    altered = history.copy()
    altered.iloc[0, 0] += 0.1
    with pytest.raises(ValueError, match="conflicts"):
        monitor.record_forecast("r-1", result, 5, altered, profile)
    assert monitor.report()["latency_ms"]["p50"] == 5


def test_delayed_outcomes_and_metrics(tmp_path, prediction):
    history, result = prediction
    monitor = Monitor(tmp_path / "events.sqlite")
    monitor.record_forecast("r-1", result, 4, history)
    target = result["forecasts"][0]["target_timestamp"]
    observed = (pd.Timestamp(target) + pd.Timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError, match="complete target hour"):
        monitor.record_outcome("r-1", target, 1.25, target)
    assert monitor.record_outcome("r-1", target, 1.25, observed)["created"]
    assert not monitor.record_outcome("r-1", target, 1.25, observed)["created"]
    with pytest.raises(ValueError, match="conflicts"):
        monitor.record_outcome("r-1", target, 1.4, observed)
    with pytest.raises(ValueError, match="units"):
        monitor.record_outcome("r-1", target, 1.25, observed, unit="Wh")
    before = monitor.report(as_of=target)
    assert before["performance"][0]["observed_count"] == 0
    report = monitor.report(as_of=pd.Timestamp(target) + pd.Timedelta(hours=30))
    first = report["performance"][0]
    assert first["observed_count"] == 1
    assert first["mae"] == first["rmse"] == 0.25
    assert first["interval_coverage"] == 1.0
    assert report["missing_completed_outcomes"] == 2


@pytest.mark.parametrize("actual", [-1, float("nan"), float("inf"), True, "1.2"])
def test_invalid_actual_rejected(tmp_path, prediction, actual):
    history, result = prediction
    monitor = Monitor(tmp_path / "events.sqlite")
    monitor.record_forecast("r-1", result, 1, history)
    target = result["forecasts"][0]["target_timestamp"]
    with pytest.raises(ValueError):
        monitor.record_outcome("r-1", target, actual, pd.Timestamp(target) + pd.Timedelta(hours=2))


def test_unknown_request_and_target_rejected(tmp_path, prediction):
    history, result = prediction
    monitor = Monitor(tmp_path / "events.sqlite")
    target = result["forecasts"][0]["target_timestamp"]
    later = pd.Timestamp(target) + pd.Timedelta(hours=5)
    with pytest.raises(ValueError, match="No forecast"):
        monitor.record_outcome("missing", target, 1, later)
    monitor.record_forecast("r-1", result, 1, history)
    with pytest.raises(ValueError, match="not one of"):
        monitor.record_outcome("r-1", pd.Timestamp(target) + pd.Timedelta(hours=1), 1, later)


def test_version_metrics_remain_separate(tmp_path, prediction):
    history, result = prediction
    monitor = Monitor(tmp_path / "events.sqlite")
    monitor.record_forecast("first", result, 1, history)
    second = deepcopy(result)
    second["version"] = "energy-" + "b" * 24
    monitor.record_forecast("second", second, 2, history)
    monitor.record_input_failure("missing_input")
    assert len(monitor.report()["performance"]) == 6
    assert monitor.report()["input_failures"] == {"missing_input": 1}
    assert monitor.report()["pending_outcomes"] == 6
    assert not monitor.report()["latest_drift"]["available"]


def test_drift_uses_training_reference_and_detects_shift(prediction):
    history, _ = prediction
    profile = _profile(history["y"].to_numpy())
    stable = drift_report(history, profile)
    shifted = history * 20
    changed = drift_report(shifted, profile)
    assert changed["population_stability_index"] > stable["population_stability_index"]
    assert changed["outside_training_range_fraction"] == 1.0
    assert profile["bin_edges"][0] == 0.5


def test_database_connections_close_after_each_operation(tmp_path):
    monitor = Monitor(tmp_path / "events.sqlite")
    with monitor._connect() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
