"""Ask the local forecast evidence store; optional paid inference requires opt-in."""

import argparse
import json
from pathlib import Path

from forecasting.analyst import AnalystError, CompatibleToolPlanner, EvidenceStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--llm", action="store_true", help="Use the explicitly configured provider for tool selection")
    parser.add_argument("--allow-paid", action="store_true", help="Permit one metered request, subject to the configured budget")
    args = parser.parse_args()
    try:
        if args.allow_paid and not args.llm:
            parser.error("--allow-paid requires --llm")
        planner = CompatibleToolPlanner.from_env(allow_paid=args.allow_paid) if args.llm else None
        result = EvidenceStore(Path(__file__).resolve().parents[1]).answer(args.question, planner)
    except (AnalystError, ValueError) as error:
        parser.exit(2, f"Request not run: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
