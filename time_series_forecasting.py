"""Run the historical price or appliance-energy comparison from the command line."""

import argparse
from pathlib import Path
import sys

from forecasting import DATASETS, load_data, run_experiment, write_artifacts
from forecasting.models import MODEL_NAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="crypto")
    parser.add_argument("--data-path", type=Path, help="Override the dataset CSV path.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output", type=Path, help="Output directory; defaults to outputs/<dataset>.")
    parser.add_argument("--quick", action="store_true", help="Use fewer origins for a smoke test.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    data_path = args.data_path or root / DATASETS[args.dataset].default_path
    output = args.output or root / "outputs" / args.dataset
    try:
        data = load_data(data_path, args.dataset)
        predictions, metrics, summary = run_experiment(data, args.dataset, args.models, args.quick)
        write_artifacts(predictions, metrics, summary, output)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Forecasting failed: {error}", file=sys.stderr)
        return 1
    mode = "smoke test" if args.quick else "full rolling comparison"
    print(f"Completed {args.dataset} {mode}: {len(predictions)} forecasts across {len(args.models)} models.")
    for horizon, model in summary["selected_models"].items():
        score = metrics[(metrics["phase"] == "test") & (metrics["model"] == model) &
                        (metrics["horizon"] == int(horizon))].iloc[0]
        print(f"Horizon {horizon}: validation selected {model}; test MAE {score['mae']:.3f}.")
    print(f"Artifacts: {output.resolve()}")
    if summary["warnings"]:
        print(f"Model warnings recorded in summary.json: {len(summary['warnings'])} distinct warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
