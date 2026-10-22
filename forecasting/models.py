"""Fixed model configurations used without test-set tuning."""

import logging

import numpy as np
import pandas as pd

from .features import supervised_rows

MODEL_NAMES = ("naive", "drift", "seasonal_naive", "arima", "prophet", "boosted")
MODEL_PARAMETERS = {
    "naive": {"method": "last observed value"},
    "drift": {"method": "last value plus full-history average change"},
    "seasonal_naive": {"method": "repeat the last complete seasonal cycle"},
    "arima": {"order": [2, 1, 0], "trend": "n"},
    "prophet": {
        "growth": "linear", "weekly_seasonality": True, "yearly_seasonality": False,
        "daily_seasonality": "hourly data only", "uncertainty_samples": 0, "seed": 42,
    },
    "boosted": {
        "estimator": "HistGradientBoostingRegressor", "max_iter": 80,
        "max_leaf_nodes": 15, "early_stopping": False, "random_state": 42,
        "target": "future value minus value at issue time", "strategy": "direct per horizon",
    },
}


def predict_model(
    name: str, history: pd.DataFrame, horizons: tuple[int, ...] | list[int],
    frequency: str, seasonal_lag: int,
) -> np.ndarray:
    """Return forecasts for the requested horizons using only supplied history."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}.")
    if any(isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer))
           for horizon in horizons):
        raise ValueError("Horizons must be positive integers.")
    horizon_array = np.asarray(horizons, dtype=int)
    if not len(horizon_array) or np.any(horizon_array < 1):
        raise ValueError("Horizons must be positive integers.")
    y = history["y"].to_numpy(dtype=float)
    last = float(y[-1])
    maximum = int(horizon_array.max())
    if name == "naive":
        forecasts = np.full(len(horizon_array), last)
    elif name == "drift":
        if len(y) < 2:
            raise ValueError("Drift needs at least two observations.")
        forecasts = last + horizon_array * (last - y[0]) / (len(y) - 1)
    elif name == "seasonal_naive":
        if len(y) < seasonal_lag:
            raise ValueError("Seasonal naive needs one complete seasonal cycle.")
        forecasts = y[-seasonal_lag:][(horizon_array - 1) % seasonal_lag]
    elif name == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        fitted = ARIMA(history["y"], order=(2, 1, 0), trend="n").fit()
        forecasts = np.asarray(fitted.forecast(steps=maximum))[horizon_array - 1]
    elif name == "prophet":
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        fitted = Prophet(
            growth="linear", weekly_seasonality=True, yearly_seasonality=False,
            daily_seasonality=frequency == "h", uncertainty_samples=0,
        )
        fitted.fit(pd.DataFrame({"ds": history.index, "y": y}), seed=42)
        dates = pd.date_range(history.index[-1], periods=maximum + 1, freq=frequency)[1:]
        forecasts = fitted.predict(pd.DataFrame({"ds": dates}))["yhat"].to_numpy()[horizon_array - 1]
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits

        estimates = []
        with threadpool_limits(limits=1):
            for horizon in horizon_array:
                x_train, y_train, x_next = supervised_rows(history, int(horizon), frequency)
                model = HistGradientBoostingRegressor(
                    max_iter=80, max_leaf_nodes=15, early_stopping=False, random_state=42,
                )
                model.fit(x_train, y_train)
                estimates.append(last + float(model.predict(x_next)[0]))
        forecasts = np.asarray(estimates)
    if not np.isfinite(forecasts).all():
        raise ValueError(f"{name} returned a nonfinite forecast.")
    return np.maximum(np.asarray(forecasts, dtype=float), 0.0)
