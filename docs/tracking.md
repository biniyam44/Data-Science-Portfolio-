# Local experiment tracking

The energy service has a local serving registry for promotion and rollback. MLflow is a separate record of how a bundle was trained and evaluated. Logging a run does not activate a model, retrain it, or send anything to a hosted service.

## Record a trained version

Install the full locked environment, then train a bundle using `scripts/train_service.py`. Copy its printed `version` into the tracking command:

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

Record that version with:

```bash
python scripts/track_service.py \
  --registry artifacts/energy \
  --version energy-REPLACE_WITH_THE_PRINTED_IDENTIFIER \
  --database artifacts/tracking/mlflow.sqlite \
  --artifact-root artifacts/tracking/files
```

Omit `--version` only when an active version has already been promoted in the local serving registry. A successful call prints the run ID, experiment ID, bundle version, database path and local artifact URI.

The helper validates the bundle hashes, model types, feature implementation, package versions and validation-selection rule before creating a tracking database. It logs the actual training parameters, selected model at each horizon, all nine validation MAEs and three calibrated interval half-widths. Manifest, metadata and serialized model files are copied into the run's local artifacts. Each logging request creates its own run, even when it records an already logged version.

These metrics come from historical validation and interval calibration. They are not production measurements, fresh holdout scores or a claim that the model is deployed. The separate household replication has its own data and protocol.

## Local-only boundary

The database must be an explicit local SQLite file. The artifact root must be a local directory without symbolic links. Remote URI arguments and preexisting experiments with remote or out-of-root artifact locations are rejected. The client receives explicit local tracking and registry URIs, so a remote `MLFLOW_TRACKING_URI` or `MLFLOW_REGISTRY_URI` environment setting is not used. MLflow telemetry and asynchronous logging are disabled before importing its client.

No tracking server is started and no credentials are required. The service API never loads uploaded models. The serving registry must remain trusted and server-owned: joblib files can execute Python, and checksums do not make a malicious owner's artifacts safe.

## Inspect a run

```python
from forecasting.tracking import local_client

client = local_client("artifacts/tracking/mlflow.sqlite")
experiment = client.get_experiment_by_name("energy-forecast-service")
for run in client.search_runs([experiment.experiment_id]):
    print(run.info.run_id, run.info.status, run.data.params["bundle_version"])
    print(run.data.metrics)
```

The test suite performs a real SQLite-backed MLflow round trip and compares downloaded artifact bytes with the validated bundle. It also checks remote-path refusal, inherited remote tracking settings, corrupt artifacts and symbolic links:

```bash
python -m pytest tests/test_tracking.py
```

Upstream reference: [MLflow tracking with a local database](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/).
