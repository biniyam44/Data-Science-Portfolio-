"""Run the fixed first household replication once, retaining an attempt receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from forecasting.evaluation import write_artifacts
from forecasting.holdout import ROOT, digest_file, exclusive_json, freeze_protocol, load_hourly, read_protocol, run_replication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/holdout-protocol.json")
    parser.add_argument("--input", type=Path, default=ROOT / "data/household_holdout_hourly.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/household_holdout_source.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/holdout")
    args = parser.parse_args()
    protocol = read_protocol(args.protocol)
    if not (args.output / "protocol-lock.json").exists():
        raise ValueError("Prepare the source under its lock before running evaluation.")
    lock = freeze_protocol(args.protocol, args.output / "protocol-lock.json")
    source = json.loads(args.metadata.read_text())
    if source["derived_sha256"] != digest_file(args.input) or source["protocol_sha256"] != lock["protocol_sha256"]:
        raise ValueError("Prepared source or protocol changed since acquisition.")
    if any(args.output.glob("attempt-*.json")) or (args.output / "summary.json").exists():
        raise ValueError("An attempt already exists. Preserve it and document any retry under a separate reviewed protocol.")
    frame = load_hourly(args.input, protocol)
    started = datetime.now(timezone.utc).isoformat()
    attempt_path = args.output / "attempt-001.json"
    exclusive_json(attempt_path, {"started_utc": started, "protocol_sha256": lock["protocol_sha256"],
                                "input_sha256": source["derived_sha256"], "status": "started; completion is recorded separately"})
    timer = time.perf_counter()
    try:
        predictions, metrics, summary = run_replication(frame, protocol, progress=lambda message: print(message, flush=True))
        summary.update({"protocol_sha256": lock["protocol_sha256"], "input_sha256": source["derived_sha256"],
                        "raw_sha256": source["raw_sha256"], "lock": lock, "data_quality": source,
                        "first_run_status": "first locked comparison completed without score-driven retries",
                        "elapsed_seconds": time.perf_counter() - timer})
        write_artifacts(predictions, metrics, summary, args.output)
        exclusive_json(args.output / "attempt-001-completed.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(), "started_utc": started,
            "status": "completed", "elapsed_seconds": summary["elapsed_seconds"],
            "prediction_rows": len(predictions), "summary_sha256": digest_file(args.output / "summary.json")})
        print(f"Completed first locked comparison: {len(predictions)} predictions, {summary['elapsed_seconds']:.1f} seconds.")
    except Exception:
        exclusive_json(args.output / "attempt-001-failed.json", {
            "failed_utc": datetime.now(timezone.utc).isoformat(), "status": "failed, not a model performance result",
            "elapsed_seconds": time.perf_counter() - timer, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
