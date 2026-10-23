"""Record verified service bundles in an explicitly local MLflow database."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.parse import unquote, urlsplit

from .serving import load_bundle

EXPERIMENT_NAME = "energy-forecast-service"


def _local_path(value: str | Path, *, directory: bool) -> Path:
    text = str(value)
    parsed = urlsplit(text)
    if not text or parsed.scheme or parsed.query or parsed.fragment or text.startswith("//"):
        raise ValueError("Tracking accepts local filesystem paths, not remote URIs.")
    path = Path(value).expanduser().absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Tracking paths must not contain symbolic links.")
    if directory:
        if path.exists() and not path.is_dir():
            raise ValueError("The artifact root must be a directory.")
    elif path.suffix not in {".db", ".sqlite", ".sqlite3"} or (path.exists() and not path.is_file()):
        raise ValueError("The tracking database must be a local .db or .sqlite file.")
    return path


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Only regular bundle files can be tracked.")
        return handle.read()


def _assert_artifacts_inside(uri: str, root: Path) -> None:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
        raise ValueError("Existing MLflow experiment points to a nonlocal artifact store.")
    path = _local_path(unquote(parsed.path), directory=True)
    if not path.is_relative_to(root):
        raise ValueError("Existing MLflow artifacts are outside the configured artifact root.")


def local_client(database: str | Path):
    """Ignore remote tracking defaults and disable telemetry before importing MLflow."""
    path = _local_path(database, directory=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "false"
    from mlflow.tracking import MlflowClient

    uri = "sqlite:///" + path.as_posix()
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def track_bundle(
    registry: str | Path, version: str | None, database: str | Path,
    artifact_root: str | Path, experiment_name: str = EXPERIMENT_NAME,
) -> dict:
    """Log validation evidence, not an invented production or test measurement.

    The registry is trusted and server-owned. Its hash checks detect corruption,
    not malicious replacements by an owner who can execute Python via joblib.
    """
    database = _local_path(database, directory=False)
    artifact_root = _local_path(artifact_root, directory=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", experiment_name):
        raise ValueError("Experiment names must be short letters, digits, underscores or hyphens.")
    bundle = load_bundle(registry, version)
    folder = Path(registry).absolute() / "bundles" / bundle.version
    snapshot = {name: _read_regular(folder / name)
                for name in ("manifest.json", "metadata.json", "models.joblib")}
    manifest = json.loads(snapshot["manifest.json"])
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if "energy-" + sha256(canonical).hexdigest()[:24] != bundle.version:
        raise ValueError("Bundle manifest changed after validation.")
    for name, expected in manifest["files"].items():
        if sha256(snapshot[name]).hexdigest() != expected:
            raise ValueError("Bundle changed while creating the tracking snapshot.")
    if json.loads(snapshot["metadata.json"]) != bundle.metadata:
        raise ValueError("Bundle metadata changed after validation.")
    artifact_root.mkdir(parents=True, exist_ok=True)
    client = local_client(database)
    from mlflow.entities import Metric, Param

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=(artifact_root / experiment_name).as_uri(),
            tags={"scope": "local-portfolio-experiment", "dataset": "energy"},
        )
        experiment = client.get_experiment(experiment_id)
    elif experiment.lifecycle_stage != "active":
        raise ValueError("The configured MLflow experiment is not active.")
    _assert_artifacts_inside(experiment.artifact_location, artifact_root)
    run = client.create_run(experiment.experiment_id, tags={
        "mlflow.runName": bundle.version, "bundle_version": bundle.version,
        "evaluation_status": "validation and interval calibration only; not production or new holdout evidence",
        "model_promotion": "separate local registry action, not performed by tracking",
        "input_sha256": bundle.metadata["input_sha256"],
    })
    try:
        _assert_artifacts_inside(run.info.artifact_uri, artifact_root)
        metadata = bundle.metadata
        params = {
            "bundle_version": bundle.version, "dataset": metadata["dataset"], "unit": metadata["unit"],
            "training_cutoff": metadata["training_cutoff"], "calibration_cutoff": metadata["calibration_cutoff"],
            "selection_training_cutoff": metadata["selection_training_cutoff"],
            "interval_level": metadata["interval_level"], "calibration_origins": metadata["calibration_origins"],
            "minimum_history": metadata["minimum_history"],
            "input_sha256": metadata["input_sha256"],
            "feature_implementation_sha256": metadata["feature_implementation_sha256"],
            **{f"boosted_{name}": value for name, value in metadata["parameters"].items()},
            **{f"selected_h{h}": value for h, value in metadata["selected_models"].items()},
        }
        now_ms = int(time.time() * 1000)
        metrics = [Metric(f"validation_mae_h{h}_{name}", float(value), now_ms, 0)
                   for h, models in metadata["validation_mae"].items() for name, value in models.items()]
        metrics += [Metric(f"interval_half_width_h{h}", float(value), now_ms, 0)
                    for h, value in metadata["interval_half_widths"].items()]
        client.log_batch(run.info.run_id, metrics=metrics,
                         params=[Param(name, str(value)) for name, value in params.items()])
        with tempfile.TemporaryDirectory(prefix="forecast-tracking-") as temporary:
            for name, content in snapshot.items():
                path = Path(temporary) / name
                path.write_bytes(content)
                client.log_artifact(run.info.run_id, str(path), artifact_path="bundle")
        client.set_terminated(run.info.run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise
    return {
        "run_id": run.info.run_id, "experiment_id": experiment.experiment_id,
        "bundle_version": bundle.version, "database": str(database),
        "artifact_uri": run.info.artifact_uri, "status": "FINISHED",
        "remote_requests": "none requested; explicit local SQLite and file artifacts",
    }
