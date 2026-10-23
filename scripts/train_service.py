"""Build a local energy-service bundle without changing the active model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.serving import train_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/appliances_hourly.csv"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/energy"))
    args = parser.parse_args()
    metadata = train_bundle(args.data, args.registry)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
