# Forecasting studies

Two historical forecasting projects: daily digital-currency prices and hourly appliance energy use. Both use the same evaluation code so that a more complicated model has to earn its place against simple forecasts.

The price study extends the original ARIMA and Prophet notebook. The energy study asks how well recent consumption predicts the next hour, six hours ahead, and the following day in one measured building. It is an example of short-term demand planning, not a claim about electricity savings or a deployed service.

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
