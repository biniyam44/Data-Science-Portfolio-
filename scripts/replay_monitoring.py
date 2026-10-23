"""Replay saved test-period outcomes against a frozen service bundle locally."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.data import load_data
from forecasting.evaluation import make_origin_plan
from forecasting.monitoring import Monitor
from forecasting.serving import load_bundle, predict


def replay(registry, data_path, output):
    data = load_data(data_path, "energy")
    bundle = load_bundle(registry)
    plan, _ = make_origin_plan(data, "energy")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    with tempfile.TemporaryDirectory(prefix="forecast-replay-") as temporary:
        monitor = Monitor(Path(temporary) / "monitor.db")
        for index, origin in enumerate(plan["test"]):
            history = data.iloc[:origin + 1].tail(169)
            started = time.perf_counter()
            prediction = predict(bundle, history)
            latency = (time.perf_counter() - started) * 1000
            request_id = f"replay-{index:04d}"
            monitor.record_forecast(request_id, prediction, latency, history, bundle.metadata["reference_profile"])
            for item in prediction["forecasts"]:
                target = pd.Timestamp(item["target_timestamp"])
                monitor.record_outcome(request_id, target, float(data.loc[target, "y"]), target + pd.Timedelta(hours=1))
            predictions.append({"request_id": request_id, **prediction})
        report = monitor.report(as_of=data.index[-1] + pd.Timedelta(hours=1))
    first = data.iloc[:plan["test"][0] + 1].tail(169)
    request = {"unit": "kWh per hour", "history": [
        {"timestamp": timestamp.isoformat(), "y": float(value)} for timestamp, value in first["y"].items()
    ]}
    artifacts = {
        "model.json": {"version": bundle.version, **bundle.metadata},
        "sample-request.json": request,
        "sample-response.json": predictions[0],
        "predictions.json": predictions,
        "monitoring.json": {
            "mode": "historical local replay, not live telemetry",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": "Frozen service model chosen using validation only, distinct from rolling study models.",
            "report": report,
        },
    }
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/energy"))
    parser.add_argument("--data", type=Path, default=Path("data/appliances_hourly.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/service-replay"))
    args = parser.parse_args()
    print(json.dumps(replay(args.registry, args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
