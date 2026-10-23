"""Log a verified energy model bundle to local SQLite-backed MLflow tracking."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.tracking import track_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/energy"))
    parser.add_argument("--version", help="Bundle identifier; omit to log the active local version.")
    parser.add_argument("--database", default="artifacts/tracking/mlflow.sqlite")
    parser.add_argument("--artifact-root", default="artifacts/tracking/files")
    parser.add_argument("--experiment", default="energy-forecast-service")
    args = parser.parse_args()
    print(json.dumps(track_bundle(args.registry, args.version, args.database, args.artifact_root, args.experiment), indent=2))


if __name__ == "__main__":
    main()
