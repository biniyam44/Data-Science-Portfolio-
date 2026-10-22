"""Checks for the public energy-source transformation."""

import csv
from datetime import datetime, timedelta
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_energy import aggregate_hourly, load_source, parse_measurements, sha256


ROOT = Path(__file__).resolve().parents[1]


def source_bytes(rows: list[tuple[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["date", "Appliances"])
    writer.writerows(rows)
    return buffer.getvalue().encode()


class PrepareEnergyTests(unittest.TestCase):
    def test_wh_readings_are_summed_and_converted_to_kwh(self):
        start = datetime(2016, 1, 11, 17)
        rows = [
            ((start + timedelta(minutes=10 * index)).strftime("%Y-%m-%d %H:%M:%S"), 100)
            for index in range(12)
        ]
        hourly, excluded = aggregate_hourly(parse_measurements(source_bytes(rows)))
        self.assertEqual(hourly, [(start, Decimal("0.6")), (start + timedelta(hours=1), Decimal("0.6"))])
        self.assertEqual(excluded, [])

    def test_only_partial_edge_hours_are_dropped(self):
        start = datetime(2016, 1, 11, 17, 20)
        rows = [(start + timedelta(minutes=10 * index), Decimal(50)) for index in range(12)]
        hourly, excluded = aggregate_hourly(rows)
        self.assertEqual(hourly, [(datetime(2016, 1, 11, 18), Decimal("0.3"))])
        self.assertEqual([item["readings"] for item in excluded], [4, 2])

    def test_missing_reading_in_middle_hour_is_rejected(self):
        start = datetime(2016, 1, 11, 17)
        records = [(start + timedelta(minutes=10 * index), Decimal(50)) for index in range(18) if index != 8]
        with self.assertRaisesRegex(ValueError, "Incomplete interior hour"):
            aggregate_hourly(records)

    def test_duplicate_gap_and_out_of_order_timestamps_are_rejected(self):
        first = "2016-01-11 17:00:00"
        for second in [first, "2016-01-11 17:20:00", "2016-01-11 16:50:00"]:
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "unique, ordered"):
                parse_measurements(source_bytes([(first, 60), (second, 60)]))

    def test_off_grid_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Off-grid"):
            parse_measurements(source_bytes([("2016-01-11 17:05:00", 60)]))

    def test_nonfinite_negative_and_missing_values_are_rejected(self):
        for value in ["NaN", "Infinity", "-1", ""]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_measurements(source_bytes([("2016-01-11 17:00:00", value)]))

    def test_changed_cached_source_is_rejected_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            content = source_bytes([("2016-01-11 17:00:00", 60)])
            source.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                load_source(source)
            self.assertEqual(source.read_bytes(), content)

    def test_committed_hourly_data_matches_its_provenance(self):
        content = (ROOT / "data/appliances_hourly.csv").read_bytes()
        metadata = json.loads((ROOT / "data/energy_source.json").read_text())
        rows = list(csv.DictReader(io.StringIO(content.decode())))
        self.assertEqual(sha256(content), metadata["derived_sha256"])
        self.assertEqual(len(rows), 3289)
        self.assertEqual(len(rows), metadata["derived_rows"])
        self.assertEqual(rows[0]["timestamp"], metadata["derived_start"])
        self.assertEqual(rows[-1]["timestamp"], metadata["derived_end"])
        self.assertEqual(metadata["derived_unit"], "kWh per hour")


if __name__ == "__main__":
    unittest.main()
