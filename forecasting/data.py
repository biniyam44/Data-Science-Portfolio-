"""Input contracts for the two historical forecasting studies."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    title: str
    unit: str
    frequency: str
    horizons: tuple[int, ...]
    seasonal_lag: int
    origin_step: int
    default_path: str
    source_url: str
    source_title: str
    source_license: str


DATASETS = {
    "crypto": Dataset(
        title="Digital currency closing-price forecasts", unit="USD", frequency="D",
        horizons=(1, 7, 30), seasonal_lag=7, origin_step=5, default_path="data/dc.csv",
        source_url="https://www.kaggle.com/datasets/ahmedadam415/digital-currency-time-series",
        source_title="Digital Currency Time Series", source_license="Not independently confirmed",
    ),
    "energy": Dataset(
        title="Household appliance energy forecasts", unit="kWh per hour", frequency="h",
        horizons=(1, 6, 24), seasonal_lag=24, origin_step=25,
        default_path="data/appliances_hourly.csv",
        source_url="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
        source_title="UCI Appliances Energy Prediction", source_license="CC BY 4.0",
    ),
}


def validate_data(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Return a sorted copy; reject duplicates, gaps and invalid measurements."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The input needs a DatetimeIndex.")
    if data.empty or "y" not in data:
        raise ValueError("The input must contain a nonempty y column.")
    if data.index.hasnans or data.index.has_duplicates:
        raise ValueError("Timestamps must be present and unique.")
    if data.index.tz is not None:
        raise ValueError("Use timezone-naive source timestamps consistently.")
    unexpected = set(data.columns) - {"y", "volume"}
    if unexpected:
        raise ValueError(f"Unexpected model inputs: {sorted(unexpected)}")
    result = data.sort_index().copy()
    expected = pd.date_range(result.index[0], periods=len(result), freq=frequency)
    if not np.array_equal(expected.values, result.index.values):
        raise ValueError(f"Timestamps must be contiguous at frequency {frequency}; no gaps are filled.")
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if not np.isfinite(result[column]).all() or (result[column] < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative measurements.")
    result.index = pd.DatetimeIndex(result.index, freq=frequency, name="timestamp")
    return result


def load_data(path: str | Path, dataset: str = "crypto") -> pd.DataFrame:
    """Load only measurements available by the end of a forecast origin period."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}.")
    path = Path(path)
    raw = pd.read_csv(path)
    if dataset == "crypto":
        date_column = "timestamp" if "timestamp" in raw else raw.columns[0]
        if "close_USD" not in raw:
            raise ValueError("The crypto CSV requires close_USD.")
        columns = {"y": raw["close_USD"].to_numpy()}
        if "volume" in raw:
            columns["volume"] = raw["volume"].to_numpy()
    else:
        date_column = "timestamp"
        if not {"timestamp", "y"}.issubset(raw.columns):
            raise ValueError("The energy CSV requires timestamp and y.")
        columns = {"y": raw["y"].to_numpy()}
    index = pd.DatetimeIndex(pd.to_datetime(raw[date_column], errors="raise"))
    data = validate_data(pd.DataFrame(columns, index=index), DATASETS[dataset].frequency)
    data.attrs["input_sha256"] = sha256(path.read_bytes()).hexdigest()
    data.attrs["input_file"] = path.name
    return data
