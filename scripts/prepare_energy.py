"""Prepare complete hourly appliance-energy observations from the UCI source."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/"
    "energydata_complete.csv"
)
SOURCE_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
SOURCE_ROWS = 19_735
MAX_DOWNLOAD_BYTES = 30_000_000


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    """Replace a generated file only after its complete contents are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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


def load_source(path: Path) -> bytes:
    """Reuse the pinned source file, or download it without accepting new bytes."""
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            content = response.read(MAX_DOWNLOAD_BYTES + 1)
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise ValueError("UCI download exceeded the expected size limit")
    actual = sha256(content)
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"Source checksum mismatch: expected {SOURCE_SHA256}, received {actual}. "
            "Inspect the upstream change before updating the recorded checksum."
        )
    if not path.exists():
        atomic_write(path, content)
    return content


def parse_measurements(content: bytes) -> list[tuple[datetime, Decimal]]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if not {"date", "Appliances"}.issubset(reader.fieldnames or []):
        raise ValueError("Source must contain date and Appliances columns")
    records: list[tuple[datetime, Decimal]] = []
    for line, row in enumerate(reader, start=2):
        try:
            stamp = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
            energy_wh = Decimal(row["Appliances"].strip())
        except (ValueError, InvalidOperation, AttributeError) as exc:
            raise ValueError(f"Invalid date or energy value on source line {line}") from exc
        if stamp.minute % 10 or stamp.second or stamp.microsecond:
            raise ValueError(f"Off-grid timestamp on source line {line}: {stamp}")
        if not energy_wh.is_finite() or energy_wh < 0:
            raise ValueError(f"Invalid appliance energy on source line {line}")
        if records and stamp - records[-1][0] != timedelta(minutes=10):
            raise ValueError(
                f"Source timestamps must be unique, ordered and ten minutes apart: {stamp}"
            )
        records.append((stamp, energy_wh))
    if not records:
        raise ValueError("Source contains no measurements")
    return records


def aggregate_hourly(
    records: list[tuple[datetime, Decimal]],
) -> tuple[list[tuple[datetime, Decimal]], list[dict[str, object]]]:
    """Sum six ten-minute Wh readings, then convert to kWh per complete hour."""
    buckets: dict[datetime, list[tuple[datetime, Decimal]]] = {}
    for stamp, energy_wh in records:
        hour = stamp.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour, []).append((stamp, energy_wh))
    hours = sorted(buckets)
    rows: list[tuple[datetime, Decimal]] = []
    dropped: list[dict[str, object]] = []
    for hour in hours:
        readings = buckets[hour]
        slots = [stamp.minute for stamp, _ in readings]
        if slots != [0, 10, 20, 30, 40, 50]:
            if hour not in {hours[0], hours[-1]}:
                raise ValueError(f"Incomplete interior hour cannot be discarded: {hour}")
            dropped.append({"timestamp": hour.isoformat(), "readings": len(readings)})
            continue
        rows.append((hour, sum(value for _, value in readings) / Decimal(1000)))
    if not rows:
        raise ValueError("Source contains no complete hours")
    if any(right[0] - left[0] != timedelta(hours=1) for left, right in zip(rows, rows[1:])):
        raise ValueError("Hourly output is not continuous")
    return rows, dropped


def prepare(source: Path, output: Path, metadata: Path) -> dict[str, object]:
    raw = load_source(source)
    records = parse_measurements(raw)
    if len(records) != SOURCE_ROWS:
        raise ValueError(f"Expected {SOURCE_ROWS} raw rows, received {len(records)}")
    hourly, dropped = aggregate_hourly(records)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["timestamp", "y"])
    writer.writerows((stamp.isoformat(), format(value, "f")) for stamp, value in hourly)
    derived = buffer.getvalue().encode("utf-8")
    provenance: dict[str, object] = {
        "dataset": "Appliances Energy Prediction",
        "creator": "Luis Candanedo",
        "citation": (
            "Candanedo, L. (2017). Appliances Energy Prediction [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G."
        ),
        "source_url": SOURCE_URL,
        "dataset_url": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
        "doi": "10.24432/C5VC8G",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "raw_sha256": sha256(raw),
        "raw_rows": len(records),
        "raw_start": records[0][0].isoformat(),
        "raw_end": records[-1][0].isoformat(),
        "derived_sha256": sha256(derived),
        "derived_file": output.name,
        "derived_rows": len(hourly),
        "derived_start": hourly[0][0].isoformat(),
        "derived_end": hourly[-1][0].isoformat(),
        "target": "Appliances",
        "source_unit": "Wh per ten-minute reading",
        "derived_unit": "kWh per hour",
        "aggregation": "Sum six distinct ten-minute Appliances readings per hour, divide by 1000.",
        "timestamp_semantics": (
            "Hour start in the source's naive local clock. No UTC conversion or "
            "daylight-saving adjustment is inferred. The value is known after that hour."
        ),
        "excluded_partial_edge_hours": dropped,
        "missing_interior_intervals": 0,
        "duplicate_timestamps": 0,
        "preparation_script": "scripts/prepare_energy.py",
        "scope": "One low-energy building, about 4.5 months in 2016; not whole-building load.",
    }
    atomic_write(output, derived)
    atomic_write(metadata, (json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode())
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/energydata_complete.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/appliances_hourly.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/energy_source.json")
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.metadata)
    print(f"Prepared {result['derived_rows']} complete hours; SHA-256 {result['derived_sha256']}")


if __name__ == "__main__":
    main()
