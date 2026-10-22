from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.data import DATASETS, load_data, validate_data
from forecasting.features import feature_matrix, supervised_rows
from forecasting.models import predict_model


ROOT = Path(__file__).resolve().parents[1]


def series(length=400, frequency="D"):
    return pd.DataFrame(
        {"y": 100 + np.arange(length, dtype=float), "volume": 1000 + np.arange(length)},
        index=pd.date_range("2020-01-01", periods=length, freq=frequency, name="timestamp"),
    )


def test_crypto_is_sorted_and_target_conversion_is_excluded():
    result = load_data(ROOT / "data/dc.csv", "crypto")
    assert len(result) == 1000
    assert result.index.is_monotonic_increasing
    assert result.index[0] == pd.Timestamp("2018-05-07")
    assert set(result.columns) == {"y", "volume"}
    assert len(result.attrs["input_sha256"]) == 64


def test_energy_units_and_contiguous_hours():
    result = load_data(ROOT / "data/appliances_hourly.csv", "energy")
    assert len(result) == 3289
    assert DATASETS["energy"].unit == "kWh per hour"
    assert (result.index[1:] - result.index[:-1] == pd.Timedelta(hours=1)).all()


@pytest.mark.parametrize("corruption", ["duplicate", "gap", "nan", "infinity", "negative", "leak"])
def test_bad_data_fails_closed(corruption):
    data = series()
    if corruption == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    elif corruption == "gap":
        data = data.drop(data.index[5])
    elif corruption == "nan":
        data.iloc[5, 0] = np.nan
    elif corruption == "infinity":
        data.iloc[5, 0] = np.inf
    elif corruption == "negative":
        data.iloc[5, 0] = -1
    else:
        data["close_SAR"] = data.y * 3.7504
    with pytest.raises(ValueError):
        validate_data(data, "D")


@pytest.mark.parametrize("frequency,horizon", [("D", 30), ("h", 24)])
def test_future_changes_cannot_change_earlier_features(frequency, horizon):
    original = series(frequency=frequency)
    changed = original.copy()
    changed.iloc[250:] *= 10
    before = feature_matrix(original, horizon, frequency).iloc[:250]
    after = feature_matrix(changed, horizon, frequency).iloc[:250]
    pd.testing.assert_frame_equal(before, after)


def test_direct_labels_do_not_reach_past_the_origin():
    history = series()
    features, labels, next_row = supervised_rows(history, 30, "D")
    assert features.index.max() == history.index[-31]
    assert next_row.index[0] == history.index[-1]
    np.testing.assert_allclose(labels, 30)
    assert next_row.y_now.iloc[0] == history.y.iloc[-1]
    assert next_row.y_lag_1.iloc[0] == history.y.iloc[-2]


def test_zero_values_do_not_create_infinite_return_features():
    history = series()
    history.loc[history.index[30:40], "y"] = 0
    features = feature_matrix(history, 7, "D")
    assert not np.isinf(features.to_numpy()).any()


def test_baselines_have_known_answers():
    history = series(50)
    np.testing.assert_allclose(predict_model("naive", history, [1, 7, 30], "D", 7), [149] * 3)
    np.testing.assert_allclose(predict_model("drift", history, [1, 7, 30], "D", 7), [150, 156, 179])
    np.testing.assert_allclose(predict_model("seasonal_naive", history, [1, 7, 8], "D", 7), [143, 149, 143])


@pytest.mark.parametrize("name", ["arima", "prophet", "boosted"])
def test_each_model_fits_and_produces_finite_forecasts(name):
    history = series(220)
    result = predict_model(name, history, [1, 7, 30], "D", 7)
    assert result.shape == (3,)
    assert np.isfinite(result).all()
    assert (result >= 0).all()


def test_unknown_model_is_not_replaced_by_a_baseline():
    with pytest.raises(ValueError, match="Unknown model"):
        predict_model("not-a-model", series(), [1], "D", 7)


@pytest.mark.parametrize("horizons", [[1.9], [True], [0], [], [-1]])
def test_invalid_horizons_are_not_silently_truncated(horizons):
    with pytest.raises(ValueError, match="positive integers"):
        predict_model("naive", series(), horizons, "D", 7)
