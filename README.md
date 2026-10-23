# Forecasting studies and ML systems

Two historical forecasting projects: daily digital-currency prices and hourly appliance energy use. Both use the same evaluation code so that a more complicated model has to earn its place against simple forecasts.

The price study extends the original ARIMA and Prophet notebook. The energy study asks how well recent consumption predicts the next hour, six hours ahead, and the following day in one measured building. An independent whole-household replication, a versioned prediction API, and restricted research tools extend the work beyond notebooks. These are local engineering demonstrations, not claims about electricity savings or a production deployment.

## Run locally

Use Python 3.11. The commands below run from this repository's root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest
```

Run either complete comparison:

```bash
python time_series_forecasting.py --dataset crypto --output results/crypto
python time_series_forecasting.py --dataset energy --output results/energy
```

For a shorter development run, select fewer models and origins. These results are smoke checks, not the published comparison:

```bash
python time_series_forecasting.py --dataset energy --models naive seasonal_naive boosted --quick --output outputs/energy-smoke
python time_series_forecasting.py --help
```

Each run writes `predictions.csv`, `metrics.csv`, `splits.json`, and `summary.json`. The split record and source hashes make the evaluation inputs inspectable. Re-running into `results/` replaces that project's saved results; use `outputs/` for experiments you do not intend to publish.

## Read the studies

- [Daily price forecasts](Time_Series_Forecasting.ipynb)
- [Hourly appliance-energy forecasts](Energy_Demand_Forecasting.ipynb)
- [Evaluation decisions and limitations](docs/methodology.md)
- [Dataset sources and preparation](data/README.md)
- [Locked independent-household replication](docs/holdout.md)
- [Energy API and deployment](docs/deployment.md)
- [Model training, promotion and rollback](docs/service-models.md)
- [Local MLflow tracking](docs/tracking.md)
- [Evidence-grounded research assistant](docs/research-assistant.md)

Rebuild and execute both notebooks from the saved result files:

```bash
python scripts/build_notebooks.py --execute
```

The notebooks inspect the source data and saved forecasts, calculate comparisons, and draw the charts. They do not quietly retrain the models when opened. Run the forecasting commands first when changing a model or dataset.

## What is compared

The comparison includes last-value, drift and seasonal-naive baselines, ARIMA, Prophet, and gradient-boosted trees. Models use the same forecast origins and target dates. Tree inputs are historical observations and calendar features available at forecast time. The price model does not use the target's converted-currency column or future prices and volumes.

The time series is divided into an initial training period, validation, interval calibration, and a final test period. The final period is excluded from model selection and interval calibration. Evaluation is rolling: after a past observation becomes available, later forecast origins can train on it. This is not a single forecast made before the entire test period.

The projects report error in original units, percentage error, scaled error, and measured coverage of nominal 80% intervals. No model is assumed to beat a baseline. A lower test error is evidence for this dataset and period, not a promise of future performance.

## Data and scope

The existing price file has 1,000 daily observations from May 2018 to January 2021. Its original Kaggle link is recorded, but the upstream redistribution license has not been confirmed. Do not treat this repository as a new license grant for that file.

The energy file contains 3,289 complete hours derived from the UCI Appliances Energy Prediction dataset, attributed to Luis Candanedo and available under CC BY 4.0. Preparation validates the pinned source checksum, the ten-minute clock, and complete hourly groups. It excludes the final partial hour, rather than treating missing readings as zero. The dataset covers one building for about four and a half months, so it cannot establish annual seasonality or performance across buildings.

## Portfolio presentation

The corresponding interactive case study is prepared for [Biniyam Berga's portfolio](https://biniyam44.github.io/forecasting/). To copy checked results into an existing local checkout of that site:

```bash
python scripts/export_portfolio.py --destination ../biniyam-berga
```

The export checks source hashes, forecast keys, and metric calculations before copying the evaluation files. It does not push or deploy either repository. Code, data provenance, and downloadable evaluation files remain the source of evidence behind the presentation.

## Energy forecasting API and monitoring

Train a model offline, inspect the returned version, then promote that version explicitly:

```bash
python scripts/train_service.py --registry artifacts/energy
python scripts/model_registry.py --registry artifacts/energy promote ENERGY_VERSION_FROM_TRAINING
export FORECAST_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m uvicorn forecasting.api:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

The API accepts completed, contiguous hourly appliance-energy measurements. It returns 1-, 6- and 24-hour forecasts, empirical intervals, a model version and training cutoff. Training, validation selection and calibration happen before serving. The HTTP process only loads and predicts from saved models.

Monitoring stores request identities and delayed outcomes in local SQLite. It reports forecast error, interval coverage, request latency, input validation failures and descriptive input shift. Historical replay results are in [results/service](results/service). They are distinct from the expanding-window research results in `results/energy`.

Use [deployment.md](docs/deployment.md) for Docker, authenticated requests, input limits, operational caveats and the loopback load test. Optional MLflow tracking records model metadata, validation comparisons and artifacts in a local SQLite-backed store. No hosted tracking account is required.

## Forecast research assistant

```bash
python scripts/ask_forecasts.py "Which model was best for energy at 6 hours on test?"
python scripts/evaluate_analyst.py --output outputs/analyst-evaluation.json
```

The default is deterministic evidence retrieval. It validates the recorded experiments, uses restricted tools, calculates answers in Python, and cites file lines and hashes. It can explain methodology, compare models and inspect errors; it cannot answer live questions or infer missing experiment scope.

An optional provider-backed tool planner is implemented, but disabled by default. It requires explicit configuration, server-side credentials and a per-request token/cost budget. Offline fixtures and mocked provider tests are not evidence of live LLM quality or real provider cost. No paid calls were made to produce the committed example evaluation.

## Independent replication

The original final periods were inspected during development. To test the workflow on separate evidence, the source, date window, model settings and missingness policy for a UCI whole-household dataset were locked locally before acquiring measurements. The [first recorded comparison](docs/holdout.md#first-recorded-result) preserves all six models and reports mixed results: the one-hour validation winner lost to the last-value baseline on test. This is a different household and target, not a future holdout from the appliance building.

The first-run commands refuse to overwrite completed receipts. Use the documented reproduction path for numerical re-runs, and do not label those as untouched evaluations.

## Recorded engineering showcase

After refreshing the local service, analyst and benchmark records, copy their checked examples into the existing portfolio checkout:

```bash
python scripts/export_engineering.py --destination ../biniyam-berga
```

The static page contains no API key and makes no paid or live service requests. Public hosting, production operations and live provider evaluation are separate steps.
