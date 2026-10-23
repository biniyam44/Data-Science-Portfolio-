from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import export_engineering as module


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for folder in ("data", "docs", "results", "forecasting", "scripts", "tests/fixtures"):
        shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "raw"))
    request = json.loads((root / "results/service/sample-request.json").read_text())
    response = json.loads((root / "results/service/sample-response.json").read_text())
    # Test-only local measurement fixture. Never exported to the real portfolio.
    measurement = {
        "kind": "local HTTP load test", "model_version": response["version"],
        "requests": 2, "concurrency": 1, "successes": 2, "failures": 0,
        "warmup_requests_excluded": 1, "status_codes": [200, 200],
        "elapsed_seconds": 2.0, "successful_requests_per_second": 1.0,
        "latency_ms": {"p50": 1.0, "p95": 2.0},
        "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
    }
    (root / "results/service/load-test.json").write_text(json.dumps(measurement))
    monkeypatch.setattr(module, "ROOT", root)
    return root


def mutate(root, relative, function):
    path = root / relative
    content = json.loads(path.read_text())
    function(content)
    path.write_text(json.dumps(content))


def test_export_checks_and_writes_six_recorded_artifacts(evidence, tmp_path):
    destination = tmp_path / "website"
    destination.mkdir()
    (destination / "index.html").write_text("<title>Portfolio</title>")
    module.export(destination)
    output = destination / "engineering/data"
    assert {path.stem for path in output.glob("*.json")} == {
        "service-request", "service-response", "monitoring", "load-test", "analyst", "holdout"}
    analyst = json.loads((output / "analyst.json").read_text())
    assert [example["result"]["refused"] for example in analyst["examples"]] == [False, False, False, True]
    assert "Zero provider requests" in analyst["evaluation_note"]
    holdout = json.loads((output / "holdout.json").read_text())
    assert {row["horizon"] for row in holdout["selected"]} == {1, 6, 24}
    assert holdout["limitations"]


def test_http_response_metadata_can_differ_from_replay(evidence):
    mutate(evidence, "results/service/sample-response.json", lambda row: row.update(
        request_id="a" * 32, latency_ms=12.3, created=True))
    assert module.validate_service()[1]["created"] is True


@pytest.mark.parametrize("changes", [
    {"model_version": "different"}, {"requests": 0}, {"requests": True},
    {"successes": 1}, {"failures": 1}, {"status_codes": [200, 500]},
    {"concurrency": 0}, {"elapsed_seconds": 0}, {"elapsed_seconds": float("nan")},
    {"successful_requests_per_second": 99}, {"request_sha256": "bad"},
    {"latency_ms": {"p50": 20, "p95": 10}}, {"latency_ms": {"p50": -1, "p95": 1}},
])
def test_load_evidence_must_match_successful_bounded_workload(evidence, changes):
    mutate(evidence, "results/service/load-test.json", lambda row: row.update(changes))
    with pytest.raises(ValueError):
        module.validate_service()


@pytest.mark.parametrize("changes", [
    {"input_sha256": "bad"}, {"feature_implementation_sha256": "old"}, {"version": "another"},
])
def test_service_source_and_implementation_hashes_required(evidence, changes):
    mutate(evidence, "results/service/model.json", lambda row: row.update(changes))
    with pytest.raises(ValueError):
        module.validate_service()


@pytest.mark.parametrize("changes", [
    {"performance": []}, {"forecast_requests": 0}, {"missing_completed_outcomes": 1},
    {"pending_outcomes": 1}, {"outcome_as_of": "2020-01-01T00:00:00"},
])
def test_monitoring_needs_complete_matching_replay(evidence, changes):
    mutate(evidence, "results/service/monitoring.json", lambda row: row["report"].update(changes))
    with pytest.raises(ValueError):
        module.validate_service()


@pytest.mark.parametrize("field,value", [("mae", 0), ("rmse", 0), ("interval_coverage", 0), ("observed_count", 0), ("version", "different")])
def test_monitoring_metrics_recomputed_from_predictions_and_source(evidence, field, value):
    mutate(evidence, "results/service/monitoring.json", lambda row: row["report"]["performance"][0].update({field: value}))
    with pytest.raises(ValueError):
        module.validate_service()


@pytest.mark.parametrize("changes", [{"created": 1}, {"latency_ms": float("nan")}, {"request_id": "../bad"}, {"injected": "field"}])
def test_http_metadata_has_a_bounded_contract(evidence, changes):
    mutate(evidence, "results/service/sample-response.json", lambda row: row.update(changes))
    with pytest.raises(ValueError):
        module.validate_service()


def test_tampered_http_prediction_is_not_hidden_by_metadata_exclusion(evidence):
    mutate(evidence, "results/service/sample-response.json", lambda row: row["forecasts"][0].update(prediction=999))
    with pytest.raises(ValueError, match="stable prediction"):
        module.validate_service()


@pytest.mark.parametrize("changes", [
    {"cases": 0, "passed": 0}, {"fixture_sha256": "old"}, {"evaluation_sha256": "old"},
    {"implementation_sha256": "old"}, {"passed": 1}, {"provider_requests": 1},
    {"llm_evaluation": True}, {"mode": "llm-tool-planner"}, {"results": []},
])
def test_analyst_report_pins_nonempty_current_offline_cases(evidence, changes):
    mutate(evidence, "results/analyst/offline_evaluation.json", lambda row: row.update(changes))
    with pytest.raises(ValueError):
        module.validate_analyst()


def test_analyst_report_cannot_claim_fabricated_numbers(evidence):
    mutate(evidence, "results/analyst/offline_evaluation.json", lambda row: row["results"][0]["response"]["facts"]["models"][0].update(mae=0))
    with pytest.raises(ValueError, match="evidence"):
        module.validate_analyst()


def test_holdout_source_digest_required(evidence):
    path = evidence / "data/household_holdout_hourly.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source"):
        module.validate_holdout()


def test_holdout_metric_values_recomputed(evidence):
    path = evidence / "results/holdout/metrics.csv"
    frame = module.pd.read_csv(path)
    frame.loc[0, "mae"] = 999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="metrics"):
        module.validate_holdout()


def test_holdout_target_values_must_match_source(evidence):
    path = evidence / "results/holdout/predictions.csv"
    frame = module.pd.read_csv(path)
    frame.loc[0, "actual"] = 999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source targets"):
        module.validate_holdout()


def test_holdout_summary_completion_receipt_required(evidence):
    mutate(evidence, "results/holdout/summary.json", lambda row: row.update(limitations=[]))
    with pytest.raises(ValueError, match="completion"):
        module.validate_holdout()


def test_invalid_evidence_writes_nothing(evidence, tmp_path):
    destination = tmp_path / "website"
    destination.mkdir()
    (destination / "index.html").write_text("<title>Portfolio</title>")
    mutate(evidence, "results/service/load-test.json", lambda row: row.update(failures=1))
    with pytest.raises(ValueError):
        module.export(destination)
    assert not (destination / "engineering").exists()


def test_destination_must_be_existing_site(tmp_path):
    with pytest.raises(ValueError, match="existing"):
        module.export(tmp_path / "new-site")
