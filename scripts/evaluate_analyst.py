"""Evaluate the offline router and evidence renderer, without LLM requests."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from forecasting.analyst import EvidenceStore


ROOT = Path(__file__).resolve().parents[1]


def check_citations(root: Path, citations: list) -> bool:
    if not citations:
        return False
    for citation in citations:
        path = root / citation["file"]
        if path.resolve().is_relative_to(root.resolve()) is False:
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != citation["sha256"]:
            return False
        lines = path.read_text().splitlines()
        if not 1 <= citation["line_start"] <= citation["line_end"] <= len(lines):
            return False
        if path.suffix == ".csv":
            records = list(csv.DictReader(lines))
            row = records[citation["line_start"] - 2]
            for key, expected in citation["selector"].items():
                actual = datetime.fromisoformat(row[key]).isoformat() if key == "origin" else row[key]
                if str(expected) != actual:
                    return False
    return True


def check_numbers(root: Path, case: dict, facts: dict) -> bool:
    """Independent standard-library aggregation, not the implementation's metric helper."""
    if case["tool"] == "read_methodology":
        return facts["topic"] == case["arguments"]["topic"] and facts["excerpt"] in (root / "docs/methodology.md").read_text()
    args = case["arguments"]
    with (root / f"results/{args['dataset']}/predictions.csv").open() as source:
        records = [row for row in csv.DictReader(source) if row["phase"] == args["phase"] and int(row["horizon"]) == args["horizon"]]
    groups = {}
    for row in records:
        groups.setdefault(row["model"], []).append(row)
    if "models" in facts and {result["model"] for result in facts["models"]} != set(groups):
        return False
    for result in facts.get("models", []):
        errors = [float(row["prediction"]) - float(row["actual"]) for row in groups[result["model"]]]
        if result["n"] != len(errors) or not math.isclose(result["mae"], statistics.mean(map(abs, errors)), rel_tol=1e-10):
            return False
        if not math.isclose(result["rmse"], math.sqrt(statistics.mean(error * error for error in errors)), rel_tol=1e-10):
            return False
        if args["phase"] == "test":
            covered = [float(row["lower"]) - 1e-9 * max(1, abs(float(row["actual"]))) <= float(row["actual"]) <= float(row["upper"]) + 1e-9 * max(1, abs(float(row["actual"])))
                       for row in groups[result["model"]]]
            if not math.isclose(result["coverage"], statistics.mean(covered), abs_tol=1e-12):
                return False
    if "best_model" in case:
        if facts["best_model"] != case["best_model"] or not math.isclose(facts["models"][0]["mae"], case["mae"], rel_tol=1e-9):
            return False
    if "worst_errors" in facts:
        model = facts.get("best_model", args.get("model"))
        expected = sorted(groups[model], key=lambda row: (-abs(float(row["prediction"]) - float(row["actual"])), row["timestamp"]))[:3]
        if len(facts["worst_errors"]) != len(expected):
            return False
        for actual, row in zip(facts["worst_errors"], expected, strict=True):
            if actual["timestamp"] != datetime.fromisoformat(row["timestamp"]).isoformat() or not math.isclose(actual["absolute_error"], abs(float(row["prediction"]) - float(row["actual"])), rel_tol=1e-10):
                return False
            if any(not math.isclose(actual[key], float(row[key]), rel_tol=1e-10) for key in ("actual", "prediction")):
                return False
    return True


def evaluate(root: Path = ROOT) -> dict:
    started = time.perf_counter()
    store = EvidenceStore(root)
    startup_ms = round((time.perf_counter() - started) * 1000, 3)
    fixture = root / "tests/fixtures/analyst_questions.json"
    cases = json.loads(fixture.read_text())
    outcomes = []
    for case in cases:
        response = store.answer(case["question"])
        expected_refusal = case.get("refused", False)
        checks = {"refusal": response["refused"] == expected_refusal}
        if not expected_refusal:
            checks["tool_selection"] = response["trace"] == [{"tool": case["tool"], "arguments": case["arguments"]}]
            checks["citations"] = check_citations(root, response["citations"])
            checks["numerical_accuracy"] = not response["refused"] and check_numbers(root, case, response["facts"])
        else:
            checks["no_evidence_on_refusal"] = response["facts"] == {} and response["citations"] == []
        outcomes.append({"id": case["id"], "passed": all(checks.values()), "checks": checks, "response": response})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "deterministic-evidence", "llm_evaluation": False, "provider_requests": 0,
        "scope": "Offline routing, typed tools, checked citations, independent arithmetic and refusal fixtures. Not LLM quality, cost or prompt-injection robustness calibration.",
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256((root / "forecasting/analyst.py").read_bytes()).hexdigest(),
        "evaluation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": len(cases), "passed": sum(row["passed"] for row in outcomes),
        "latency_ms": {"startup_validation": startup_ms,
                       "median": statistics.median(row["response"]["latency_ms"] for row in outcomes),
                       "answered_median": statistics.median(row["response"]["latency_ms"] for row in outcomes if not row["response"]["refused"]),
                       "refused_median": statistics.median(row["response"]["latency_ms"] for row in outcomes if row["response"]["refused"]),
                       "max": max(row["response"]["latency_ms"] for row in outcomes)},
        "estimated_cost_usd": None, "results": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analyst/offline_evaluation.json")
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Offline evidence checks: {result['passed']}/{result['cases']}; provider requests: 0")
    if result["passed"] != result["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
