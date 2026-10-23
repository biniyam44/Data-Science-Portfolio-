# Run the energy service

This is a single-host portfolio demonstration. It has no public deployment, uptime claim, or customer data. The website displays recorded examples and does not hold a service token.

## Build and promote a model locally

From the repository root, install `requirements.lock` and the project as described in the README. Training and promotion are separate commands:

```bash
python scripts/train_service.py --registry artifacts/energy
python scripts/model_registry.py --registry artifacts/energy promote ENERGY_VERSION_FROM_TRAINING
```

Replace `ENERGY_VERSION_FROM_TRAINING` with the returned `energy-...` identifier. The identifier includes the model and metadata hashes. Rebuilding on another platform can produce another identifier; do not paste a recorded example's version into a different registry.

The service loads the active model once. Restart the process after promotion or rollback. It never fits an estimator inside a prediction request. Model files are executable pickle-based artifacts: load only bundles trained by you in an administrator-controlled registry. Hash checking detects accidental changes, not an attacker who can replace both a model and its manifest.

## Start a loopback API

Generate a temporary local token without printing it, then start one worker:

```bash
export FORECAST_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m uvicorn forecasting.api:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Keep the token in the server process environment, not a tracked file, browser script, URL, or command-line argument to a remote service. Use a separate terminal with the same locally configured token when calling protected routes. In a deployed environment, use the host's secret manager.

Routes:

- `GET /health`: public process liveness only; it does not assert a model is ready.
- `GET /ready`: authenticated readiness, including model loading.
- `GET /openapi.json`: authenticated API schema.
- `GET /v1/model`: trained model metadata and baseline comparisons.
- `POST /v1/forecast`: completed hourly measurements, returning horizons 1, 6 and 24.
- `POST /v1/outcomes`: an actual observation for a recorded request and target hour.
- `GET /v1/monitoring`: errors, interval coverage, input failures, latency and descriptive input shift.
- `POST /v1/ask`: a saved-evidence question. The default makes no model-provider calls.

Use `Authorization: Bearer <token>`. The complete request example is [sample-request.json](../results/service/sample-request.json). Its values are historical appliance energy in kWh per hour, not live meter readings. Source timestamps are naive hourly clock labels; each origin is usable only after its measurement hour has finished. Do not attach a guessed timezone.

Input limits are documented in [service-models.md](service-models.md). An optional 32-character lowercase hexadecimal `request_id` makes forecast retries idempotent; different inputs with the same identifier are a conflict. Outcomes require a matching request and target and an observation time after the complete target hour. Identical outcome retries do not inflate metrics.

## Docker

The base image is pinned by digest. The context allowlist excludes secrets, local models, telemetry and development files. Training remains outside the serving image. Models are mounted read-only; SQLite state has a separate named volume. The image runs as UID 10001.

```bash
docker compose build
docker compose up
```

`FORECAST_API_TOKEN` must already be set in that shell. The Compose port binds only to `127.0.0.1`. On native Linux, ensure UID 10001 can read the trusted model directories; do not make the registry world-writable. Container and training dependencies must match. If an artifact is rejected for dependency drift, train a fresh bundle in the intended environment rather than disabling the check.

The health check reports liveness. Use authenticated `/ready` for readiness before sending prediction traffic. Stop the local service with `docker compose down`; the named monitoring volume is retained.

## Replay and measurement

```bash
python scripts/replay_monitoring.py --registry artifacts/energy --output outputs/service-replay
python scripts/benchmark_service.py --output outputs/load-test.json
python scripts/evaluate_analyst.py --output outputs/analyst-evaluation.json
```

The replay uses held-out historical outcomes and a frozen model. It is not online telemetry. The benchmark sends one warmup and twenty authenticated HTTP requests with four workers to a loopback server. It refuses external target hosts. Its latency is a local workload measurement, not a production capacity estimate. Keep the server idle beforehand so other requests do not consume its rate allowance.

The API's returned `latency_ms` measures its prediction path before the monitoring database write. The load-test record measures the complete HTTP round trip, including that write and response transfer. Do not compare these as if they were the same metric.

The [research assistant](research-assistant.md) has an optional bounded provider-backed tool planner. It is disabled by default and has not been evaluated against a live provider in this project. No key belongs in the static portfolio. The optional [MLflow helper](tracking.md) writes only to local SQLite and local artifact storage.

## Before any public deployment

Choose a host and budget first. A public demo needs TLS termination, managed secrets, per-user authorization, a shared rate/budget store, request timeouts at the proxy, monitoring retention limits, backups, and operational alerts. The current rate limiter is process-local and shared by one token; it is not a distributed quota. The current SQLite monitor is for one host, not a multi-replica ingestion service. No automatic retraining or promotion follows a drift statistic.

Do not expose the API token in GitHub Pages or allow anonymous provider calls. A future public demonstration should use a backend session boundary and a hard provider-side spending cap. The code's per-request estimate is not a guarantee about billing.
