"""Synthetic boundary checks do not inspect the locked household measurements."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import forecasting.holdout as holdout
from scripts.prepare_holdout import aggregate_minute_rows


def protocol():
    return json.loads((Path(__file__).parents[1] / "docs/holdout-protocol.json").read_text())


def calendar(n=3000):
    index = pd.date_range("2010-07-01", periods=n, freq="h", name="timestamp")
    return pd.DataFrame({"y": 1 + np.sin(np.arange(n) / 24) ** 2,
                         "observed": True, "finite_minutes": 60}, index=index)


def test_kw_to_kwh_complete_hours_only():
    minutes = pd.date_range("2010-07-01", periods=120, freq="min")
    frame = pd.DataFrame({"timestamp": minutes, "power_kw": 2.0})
    hourly = aggregate_minute_rows(frame, "2010-07-01", "2010-07-01T02:00:00")
    assert hourly.y.tolist() == [2., 2.]
    frame.loc[65, "power_kw"] = np.nan
    missing = aggregate_minute_rows(frame, "2010-07-01", "2010-07-01T02:00:00")
    assert missing.iloc[0].y == 2.
    assert np.isnan(missing.iloc[1].y)
    assert missing.finite_minutes.tolist() == [60, 59]
    assert missing.observed.tolist() == [True, False]


@pytest.mark.parametrize("value", [-1, float("inf"), "corrupt"])
def test_bad_power_rejected(value):
    frame = pd.DataFrame({"timestamp": pd.date_range("2010-07-01", periods=60, freq="min"),
                          "power_kw": [value] + [1.] * 59})
    with pytest.raises(ValueError):
        aggregate_minute_rows(frame, "2010-07-01", "2010-07-01T01:00:00")


def test_duplicate_minute_rejected_and_missing_minute_not_filled():
    frame = pd.DataFrame({"timestamp": pd.date_range("2010-07-01", periods=60, freq="min"), "power_kw": 1.})
    with pytest.raises(ValueError, match="unique"):
        aggregate_minute_rows(pd.concat([frame, frame.iloc[:1]]), "2010-07-01", "2010-07-01T01:00:00")
    result = aggregate_minute_rows(frame.iloc[1:], "2010-07-01", "2010-07-01T01:00:00")
    assert not result.iloc[0].observed and np.isnan(result.iloc[0].y)


def test_history_fill_is_causal_and_leading_missing_is_not_backfilled():
    frame = calendar()
    frame.loc[frame.index[:3], "y"] = np.nan
    frame.loc[frame.index[210:215], "y"] = np.nan
    history, imputed = holdout.history_at(frame, 212)
    assert history.index[0] == frame.index[3]
    assert imputed == 3 and history.iloc[-1].y == frame.iloc[209].y
    frame.loc[frame.index[213:], "y"] = 1e12
    same, count = holdout.history_at(frame, 212)
    pd.testing.assert_frame_equal(same, history)
    assert count == imputed


def test_insufficient_initial_history_rejected():
    with pytest.raises(ValueError, match="168"):
        holdout.history_at(calendar(), 100)


def test_calendar_splits_and_shared_target_exclusions():
    frame = calendar()
    config = protocol()
    plan, splits = holdout.make_plan(frame, config)
    origin = plan["test"][0]
    frame.loc[frame.index[origin + 6], ["y", "observed", "finite_minutes"]] = [np.nan, False, 59]
    changed, updated = holdout.make_plan(frame, config)
    assert origin not in changed["test"]
    assert len(changed["test"]) == len(plan["test"]) - 1
    assert updated["test"]["rows"] == splits["test"]["rows"]
    assert len(updated["test"]["excluded_origins"]) == 1
    for phase, origins in changed.items():
        for i in origins:
            assert all(updated[phase]["start_index"] <= i+h < updated[phase]["end_index_exclusive"]
                       for h in config["horizons"])


def test_protocol_lock_rejects_changed_settings(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "docs/holdout-protocol.json"
    local = tmp_path / "protocol.json"
    local.write_bytes(source.read_bytes())
    monkeypatch.setattr(holdout, "implementation_sha256", lambda: protocol()["implementation_sha256"])
    lock = tmp_path / "lock.json"
    saved = holdout.freeze_protocol(local, lock)
    assert saved == holdout.freeze_protocol(local, lock)
    modified = json.loads(local.read_text())
    modified["purpose"] += " modified"
    local.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="differs"):
        holdout.freeze_protocol(local, lock)


def test_hourly_loader_does_not_admit_imputed_targets(tmp_path):
    config = protocol()
    config["window"]["end_exclusive"] = "2010-07-02T00:00:00"
    frame = calendar(24)
    path = tmp_path / "hourly.csv"
    frame.to_csv(path)
    assert len(holdout.load_hourly(path, config)) == 24
    frame.loc[frame.index[1], ["observed", "finite_minutes"]] = [False, 59]
    frame.to_csv(path)
    with pytest.raises(ValueError, match="Only complete"):
        holdout.load_hourly(path, config)


def test_every_model_shares_origins_and_calibration_cannot_see_test(monkeypatch):
    config = protocol()
    frame = calendar()
    plan, _ = holdout.make_plan(frame, config)
    frame.loc[frame.index[plan["test"][0] + 6], ["y", "observed", "finite_minutes"]] = [np.nan, False, 59]
    calls = []

    def fake_model(name, history, horizons, frequency, seasonal_lag):
        calls.append((name, history.index[-1], history.y.isna().any()))
        return np.repeat(history.iloc[-1].y, len(horizons))

    monkeypatch.setattr(holdout, "predict_model", fake_model)
    original, _, summary = holdout.run_replication(frame, config)
    for phase in ("validation", "calibration", "test"):
        sets = [set(zip(group.origin, group.horizon)) for _, group in original[original.phase == phase].groupby("model")]
        assert all(item == sets[0] for item in sets)
    assert not any(missing for _, _, missing in calls)
    assert original.actual.notna().all() and original.target_observed.all()
    assert original[original.phase != "test"].lower.isna().all()
    changed = frame.copy()
    boundary = summary["splits"]["test"]["start_index"]
    observed = changed.iloc[boundary:]["observed"]
    changed.loc[observed.index[observed], "y"] = 5000
    other, _, next_summary = holdout.run_replication(changed, config)
    assert summary["selected_models"] == next_summary["selected_models"]
    assert summary["calibration_widths"] == next_summary["calibration_widths"]
    pd.testing.assert_frame_equal(original[original.phase != "test"], other[other.phase != "test"])


def test_model_failure_aborts_instead_of_dropping_a_case(monkeypatch):
    def broken(*args):
        raise RuntimeError("a model failed")
    monkeypatch.setattr(holdout, "predict_model", broken)
    with pytest.raises(RuntimeError, match="a model failed"):
        holdout.run_replication(calendar(), protocol())


def test_saved_replication_matches_sources_calendar_and_calibration():
    root = Path(__file__).parents[1]
    result = root / "results/holdout"
    if not (result / "summary.json").exists():
        pytest.skip("The first locked experiment has not completed.")
    config = protocol()
    lock = holdout.freeze_protocol(root / "docs/holdout-protocol.json", result / "protocol-lock.json")
    receipt = json.loads((result / "acquisition.json").read_text())
    source = json.loads((root / "data/household_holdout_source.json").read_text())
    summary = json.loads((result / "summary.json").read_text())
    completion = json.loads((result / "attempt-001-completed.json").read_text())
    assert completion["summary_sha256"] == holdout.digest_file(result / "summary.json")
    assert lock["locked_utc"] < receipt["acquired_utc"] < completion["started_utc"]
    assert source["derived_sha256"] == holdout.digest_file(root / "data/household_holdout_hourly.csv")
    assert source["derived_sha256"] == summary["input_sha256"]
    assert receipt["raw_sha256"] == source["raw_sha256"] == summary["raw_sha256"]
    frame = holdout.load_hourly(root / "data/household_holdout_hourly.csv", config)
    plan, splits = holdout.make_plan(frame, config)
    assert splits == summary["splits"]
    predictions = pd.read_csv(result / "predictions.csv", parse_dates=["origin", "timestamp"])
    keys = ["phase", "model", "origin", "horizon"]
    assert not predictions.duplicated(keys).any()
    expected = {(phase, model, frame.index[i], horizon)
                for phase, indices in plan.items() for model in config["models"]
                for i in indices for horizon in config["horizons"]}
    assert set(predictions[keys].itertuples(index=False, name=None)) == expected
    assert len(predictions) == completion["prediction_rows"]
    assert (predictions.timestamp == predictions.origin + pd.to_timedelta(predictions.horizon, unit="h")).all()
    assert predictions.target_observed.all()
    assert frame.loc[predictions.timestamp, "observed"].all()
    np.testing.assert_allclose(predictions.actual, frame.loc[predictions.timestamp, "y"], rtol=1e-10)
    for (model, horizon), calibration in predictions[predictions.phase == "calibration"].groupby(["model", "horizon"]):
        width = holdout.calibration_width(abs(calibration.actual - calibration.prediction))
        test = predictions[(predictions.phase == "test") & (predictions.model == model) & (predictions.horizon == horizon)]
        np.testing.assert_allclose(test.lower, np.maximum(0, test.prediction - width), atol=1e-10)
        np.testing.assert_allclose(test.upper, test.prediction + width, atol=1e-10)
    actual_metrics = holdout.summarize_metrics(predictions)
    saved = pd.read_csv(result / "metrics.csv")
    assert list(saved.columns) == list(actual_metrics.columns)
    for column in saved.columns:
        if pd.api.types.is_numeric_dtype(saved[column]):
            np.testing.assert_allclose(saved[column], actual_metrics[column], atol=1e-10, equal_nan=True)
        else:
            assert saved[column].tolist() == actual_metrics[column].tolist()
