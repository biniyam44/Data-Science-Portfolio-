"""Features indexed by their issue time, never by an unavailable target."""

import numpy as np
import pandas as pd


def feature_matrix(history: pd.DataFrame, horizon: int, frequency: str) -> pd.DataFrame:
    """Build causal features and calendar values known for a future target time."""
    if horizon < 1:
        raise ValueError("A forecast horizon must be positive.")
    hourly = frequency == "h"
    lags = (1, 2, 3, 6, 12, 24, 48, 168) if hourly else (1, 2, 3, 7, 14, 28)
    windows = (6, 24, 168) if hourly else (7, 14, 28)
    y = history["y"]
    log_returns = np.log1p(y).diff()
    features = pd.DataFrame({"y_now": y}, index=history.index)
    for lag in lags:
        features[f"y_lag_{lag}"] = y.shift(lag)
    features["log_return_1"] = log_returns
    features["log_return_lag_1"] = log_returns.shift(1)
    for window in windows:
        features[f"rolling_mean_{window}"] = y.rolling(window).mean()
        features[f"rolling_volatility_{window}"] = log_returns.rolling(window).std(ddof=0)
    if "volume" in history:
        volume = history["volume"]
        features["volume_now"] = volume
        features["volume_lag_1"] = volume.shift(1)
        features["volume_lag_7"] = volume.shift(7)
        features["volume_mean_7"] = volume.rolling(7).mean()
    target_dates = history.index + horizon * pd.tseries.frequencies.to_offset(frequency)
    features["target_weekday_sin"] = np.sin(2 * np.pi * target_dates.dayofweek / 7)
    features["target_weekday_cos"] = np.cos(2 * np.pi * target_dates.dayofweek / 7)
    features["target_year_sin"] = np.sin(2 * np.pi * (target_dates.dayofyear - 1) / 365.25)
    features["target_year_cos"] = np.cos(2 * np.pi * (target_dates.dayofyear - 1) / 365.25)
    if hourly:
        features["target_hour_sin"] = np.sin(2 * np.pi * target_dates.hour / 24)
        features["target_hour_cos"] = np.cos(2 * np.pi * target_dates.hour / 24)
    return features


def supervised_rows(
    history: pd.DataFrame, horizon: int, frequency: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Training labels stop at the origin; prediction uses the last feature row."""
    features = feature_matrix(history, horizon, frequency)
    changes = history["y"].shift(-horizon) - history["y"]
    eligible = features.notna().all(axis=1) & changes.notna()
    training_features = features.loc[eligible]
    if len(training_features) < 30:
        raise ValueError("Too little history remains after lags and future-label exclusion.")
    prediction_features = features.iloc[[-1]]
    if prediction_features.isna().any().any():
        raise ValueError("The forecast origin has incomplete lag features.")
    return training_features, changes.loc[eligible], prediction_features
