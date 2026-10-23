"""Inspect, promote or restore a locally trained energy-service version."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecasting.serving import load_bundle, promote, rollback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/energy"))
    subparsers = parser.add_subparsers(dest="action", required=True)
    show = subparsers.add_parser("show")
    show.add_argument("--version")
    activate = subparsers.add_parser("promote")
    activate.add_argument("version")
    subparsers.add_parser("rollback")
    args = parser.parse_args()
    if args.action == "show":
        bundle = load_bundle(args.registry, args.version)
        result = {"version": bundle.version, **bundle.metadata}
    elif args.action == "promote":
        result = promote(args.registry, args.version)
    else:
        result = rollback(args.registry)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
