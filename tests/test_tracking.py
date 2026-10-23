import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["DO_NOT_TRACK"] = "1"
pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from forecasting.serving import train_bundle
from forecasting.tracking import local_client, track_bundle


@pytest.fixture(scope="module")
def tracked_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("tracked-model")
    index = pd.date_range("2020-01-01", periods=1000, freq="h", name="timestamp")
    data = pd.DataFrame({"y": 1 + .3 * np.sin(np.arange(1000) * 2 * np.pi / 24)}, index=index)
    source = root / "energy.csv"
    data.to_csv(source)
    metadata = train_bundle(source, root / "registry")
    return root / "registry", metadata


def test_real_local_mlflow_roundtrip(tracked_bundle, tmp_path, monkeypatch):
    registry, metadata = tracked_bundle
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://invalid.example/remote")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "https://invalid.example/remote")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")
    database = tmp_path / "mlflow.sqlite"
    result = track_bundle(registry, metadata["version"], database, tmp_path / "files")
    assert database.is_file() and result["status"] == "FINISHED"
    client = local_client(database)
    run = client.get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.params["bundle_version"] == metadata["version"]
    assert run.data.params["training_cutoff"] == metadata["training_cutoff"]
    for horizon, scores in metadata["validation_mae"].items():
        for model, value in scores.items():
            assert run.data.metrics[f"validation_mae_h{horizon}_{model}"] == value
        assert run.data.metrics[f"interval_half_width_h{horizon}"] == metadata["interval_half_widths"][horizon]
    assert len(run.data.metrics) == 12
    assert {artifact.path for artifact in client.list_artifacts(result["run_id"], "bundle")} == {
        "bundle/manifest.json", "bundle/metadata.json", "bundle/models.joblib"}
    for name in ("manifest.json", "metadata.json", "models.joblib"):
        downloaded = Path(client.download_artifacts(result["run_id"], f"bundle/{name}"))
        assert downloaded.read_bytes() == (registry / "bundles" / metadata["version"] / name).read_bytes()
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert os.environ["DO_NOT_TRACK"] == "1"
    assert not (registry / "registry.json").exists()
    another = track_bundle(registry, metadata["version"], database, tmp_path / "files")
    assert another["experiment_id"] == result["experiment_id"]
    assert another["run_id"] != result["run_id"]


@pytest.mark.parametrize("database,artifacts", [
    ("https://example.org/runs", "files"),
    ("sqlite:///tmp/runs.db", "files"),
    ("runs.sqlite", "s3://bucket/models"),
    ("runs.txt", "files"),
    ("runs?mode=readonly.sqlite", "files"),
])
def test_remote_or_ambiguous_paths_are_rejected(tracked_bundle, database, artifacts):
    registry, metadata = tracked_bundle
    with pytest.raises(ValueError):
        track_bundle(registry, metadata["version"], database, artifacts)


def test_corruption_rejected_before_database_created(tracked_bundle, tmp_path):
    registry, metadata = tracked_bundle
    corrupt = tmp_path / "registry"
    shutil.copytree(registry, corrupt)
    (corrupt / "bundles" / metadata["version"] / "models.joblib").write_bytes(b"corrupt")
    database = tmp_path / "runs.sqlite"
    with pytest.raises(ValueError, match="hash mismatch"):
        track_bundle(corrupt, metadata["version"], database, tmp_path / "files")
    assert not database.exists()


def test_existing_remote_artifacts_rejected(tracked_bundle, tmp_path):
    registry, metadata = tracked_bundle
    database = tmp_path / "runs.sqlite"
    client = local_client(database)
    client.create_experiment("energy-forecast-service", artifact_location="https://invalid.example/artifacts")
    with pytest.raises(ValueError, match="nonlocal"):
        track_bundle(registry, metadata["version"], database, tmp_path / "files")
    assert client.search_runs([client.get_experiment_by_name("energy-forecast-service").experiment_id]) == []


def test_symlinked_artifact_root_rejected(tracked_bundle, tmp_path):
    registry, metadata = tracked_bundle
    actual = tmp_path / "real"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual)
    with pytest.raises(ValueError, match="symbolic"):
        track_bundle(registry, metadata["version"], tmp_path / "runs.sqlite", alias)
