"""Acquire the locked UCI household source and preserve missing hourly targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from forecasting.holdout import ROOT, digest_file, exclusive_json, freeze_protocol, read_protocol

MAX_ZIP_BYTES = 30_000_000
MAX_MEMBER_BYTES = 160_000_000


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def aggregate_minute_rows(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Convert mean kW to hourly kWh; one missing minute makes its target missing."""
    if list(frame.columns) != ["timestamp", "power_kw"]:
        raise ValueError("Expected timestamp and power_kw columns.")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame[(frame.timestamp >= pd.Timestamp(start)) & (frame.timestamp < pd.Timestamp(end))]
    if frame.empty or frame.timestamp.isna().any() or frame.timestamp.duplicated().any():
        raise ValueError("Window timestamps must be nonempty, present and unique.")
    if frame.timestamp.dt.tz is not None or not (frame.timestamp == frame.timestamp.dt.floor("min")).all():
        raise ValueError("Minute timestamps must be source-naive and on the minute grid.")
    frame["power_kw"] = pd.to_numeric(frame["power_kw"], errors="raise")
    finite = frame.power_kw.dropna()
    if not np.isfinite(finite).all() or (finite < 0).any():
        raise ValueError("Power must be nonnegative and finite or explicitly missing.")
    series = frame.set_index("timestamp").sort_index()["power_kw"]
    calendar = pd.date_range(start, end, freq="h", inclusive="left", name="timestamp")
    counts = series.resample("h").count().reindex(calendar, fill_value=0)
    if (counts > 60).any():
        raise ValueError("An hour cannot contain more than sixty minute measurements.")
    energy = series.resample("h").sum(min_count=60).reindex(calendar) / 60.
    observed = counts == 60
    return pd.DataFrame({"y": energy.where(observed), "observed": observed, "finite_minutes": counts}, index=calendar)


def read_zip_hourly(content: bytes, protocol: dict) -> pd.DataFrame:
    with ZipFile(io.BytesIO(content)) as archive:
        member = protocol["source"]["zip_member"]
        if archive.namelist().count(member) != 1:
            raise ValueError("The expected UCI source member must occur exactly once.")
        if archive.getinfo(member).file_size > MAX_MEMBER_BYTES:
            raise ValueError("Source member exceeds the expected size limit.")
        with archive.open(member) as handle:
            raw = pd.read_csv(handle, sep=";", usecols=["Date", "Time", "Global_active_power"],
                              dtype=str, keep_default_na=False)
    dates = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S", errors="raise")
    power = raw["Global_active_power"].replace({"?": np.nan, "": np.nan})
    frame = pd.DataFrame({"timestamp": dates, "power_kw": power})
    return aggregate_minute_rows(frame, protocol["window"]["start_inclusive"], protocol["window"]["end_exclusive"])


def prepare(protocol_path: Path, source_path: Path, output: Path, metadata: Path, records: Path) -> dict:
    protocol = read_protocol(protocol_path)
    lock_path = records / "protocol-lock.json"
    if source_path.exists() and not lock_path.exists():
        raise ValueError("A cached source predates the lock; this cannot be called a first acquisition.")
    lock = freeze_protocol(protocol_path, lock_path)
    acquisition_path = records / "acquisition.json"
    if source_path.exists():
        raw = source_path.read_bytes()
        if not acquisition_path.exists():
            raise ValueError("Cached source lacks its acquisition receipt.")
        receipt = json.loads(acquisition_path.read_text())
        if sha256(raw).hexdigest() != receipt["raw_sha256"]:
            raise ValueError("Cached source checksum changed.")
    else:
        if acquisition_path.exists():
            raise ValueError("An acquisition receipt exists but its source is missing; restore the pinned raw cache.")
        with urlopen(protocol["source"]["download_url"], timeout=90) as response:
            if response.geturl() != protocol["source"]["download_url"]:
                raise ValueError("Unexpected download redirect; inspect the source before proceeding.")
            raw = response.read(MAX_ZIP_BYTES + 1)
        if len(raw) > MAX_ZIP_BYTES:
            raise ValueError("Download exceeded the source size limit.")
        receipt = {"acquired_utc": datetime.now(timezone.utc).isoformat(), "raw_sha256": sha256(raw).hexdigest(),
                   "raw_bytes": len(raw), "source_url": protocol["source"]["download_url"],
                   "protocol_sha256": lock["protocol_sha256"], "protocol_locked_utc": lock["locked_utc"]}
        exclusive_json(acquisition_path, receipt)
        atomic_write(source_path, raw)
    if receipt["protocol_sha256"] != lock["protocol_sha256"]:
        raise ValueError("Acquisition and protocol receipt disagree.")
    hourly = read_zip_hourly(raw, protocol)
    content = hourly.to_csv(float_format="%.15g").encode()
    source = {
        **protocol["source"], "schema_version": 1, "protocol_sha256": lock["protocol_sha256"],
        "raw_sha256": receipt["raw_sha256"], "raw_bytes": len(raw), "acquired_utc": receipt["acquired_utc"],
        "derived_sha256": sha256(content).hexdigest(), "derived_file": output.name,
        "window": protocol["window"], "calendar_hours": len(hourly),
        "observed_hours": int(hourly.observed.sum()), "incomplete_hours": int((~hourly.observed).sum()),
        "finite_minute_readings": int(hourly.finite_minutes.sum()),
        "missing_minute_readings": int(len(hourly) * 60 - hourly.finite_minutes.sum()),
        "preparation_script": "scripts/prepare_holdout.py",
        "scope": "One whole household in 2010, a separate target from the appliance energy project.",
    }
    if metadata.exists() and json.loads(metadata.read_text()) != source:
        raise ValueError("Prepared source metadata differs from the first acquisition.")
    if output.exists() and digest_file(output) != source["derived_sha256"]:
        raise ValueError("Existing prepared targets differ; do not overwrite the locked source.")
    atomic_write(output, content)
    atomic_write(metadata, (json.dumps(source, indent=2, allow_nan=False) + "\n").encode())
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/holdout-protocol.json")
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/household_power_consumption.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/household_holdout_hourly.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/household_holdout_source.json")
    parser.add_argument("--records", type=Path, default=ROOT / "results/holdout")
    args = parser.parse_args()
    source = prepare(args.protocol, args.source, args.output, args.metadata, args.records)
    print(f"Prepared {source['calendar_hours']} calendar hours: {source['observed_hours']} observed and {source['incomplete_hours']} incomplete.")


if __name__ == "__main__":
    main()
