# Household energy replication

The appliance experiment was inspected while its evaluation code was being developed. Its final period therefore remains a historical development result. This experiment checks the fixed forecasting approach on a separate public dataset under a protocol written before acquiring its measurements.

The new target is whole-household hourly energy in 2010. It is not the appliance-only target from the original 2016 building. A result here cannot establish future accuracy for that original building, and an old public dataset cannot establish exclusion from a language model's pretraining.

## Source and target

The source is [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), by Georges Hebrail and Alice Berard, distributed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citation: Hebrail, G. and Berard, A. (2006), DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54).

The fixed window is July 1 through October 31, 2010. `Global_active_power` is minute-averaged power in kW. Each hourly target is the sum of sixty valid minute readings divided by sixty, in kWh. Duplicate timestamps, off-grid times and invalid negative or infinite values are errors. One missing minute makes that hour an unobserved target. Missing hours remain on the calendar rather than compressing time.

Timestamps retain the source's naive clock. No timezone or daylight-saving correction is invented. Forecasts are issued after the origin hour has finished.

## What was locked

[holdout-protocol.json](holdout-protocol.json) specifies the source, window, models, horizons, splits, missing-data policy and failure policy. Before the download, the acquisition script creates an exclusive read-only `results/holdout/protocol-lock.json` with the protocol hash, original model-core hash and three workflow-file hashes. Every later acquisition and run checks those hashes. File permissions and local timestamps provide an auditable development record, not independent notarization or tamper-proof storage.

The models are last value, drift, seasonal naive, ARIMA, Prophet and gradient boosting, with the same fixed configurations as the earlier study. Horizons are 1, 6 and 24 hours. Calendar rows are split 40% training, 20% validation, 20% calibration and 20% test. Origins advance by 25 hours, and every requested target must be inside its phase.

At each origin, missing historical values are forward-filled from earlier observed hours only. Leading missing hours are removed; at least 168 historical rows must remain. Imputation also affects historical supervised training labels and is explicitly counted in every forecast record. Future values and evaluation targets are never filled. If any requested target hour is incomplete, that origin is excluded for every model and horizon. Exclusions and calendar coverage are saved in `splits.json`.

Validation MAE chooses a model per horizon, with model-name ordering breaking ties. Calibration residuals set 80% interval widths before test. Test forecasts refit on all history available at each origin, including earlier test observations. This is rolling evaluation, not a forecast of the entire test period from one frozen origin. Serial dependence means 80% is a nominal level, not a coverage guarantee.

## Running and inspecting

Install the repository's pinned environment first. A first acquisition and evaluation use:

```bash
python scripts/prepare_holdout.py
python scripts/run_holdout.py
python -m pytest tests/test_holdout.py
```

The acquisition script writes a source checksum receipt before parsing measurements. It then writes the hourly data and detailed missing-minute counts. A cached source must match that receipt. The initial sandboxed download attempt failed DNS before receiving data; the network-enabled retry used the unchanged locked files.

The evaluation writes a first-attempt receipt before running any model and a separate completion or failure receipt afterward. It refuses to overwrite an existing attempt, change model settings, omit failed models or silently retry after seeing scores. The saved `predictions.csv`, `metrics.csv`, `summary.json` and `splits.json` are the inspection path for the completed comparison. The first-run command intentionally will not run again over these records.

For a clearly labeled numerical reproduction after results are public, use the same API without replacing the first-run record:

```python
from pathlib import Path
from forecasting.holdout import read_protocol, load_hourly, run_replication
from forecasting.evaluation import write_artifacts

protocol = read_protocol(Path("docs/holdout-protocol.json"))
data = load_hourly(Path("data/household_holdout_hourly.csv"), protocol)
predictions, metrics, summary = run_replication(data, protocol)
summary["first_run_status"] = "Reproduction after the original results were public."
write_artifacts(predictions, metrics, summary, "outputs/household-reproduction")
```

This reproduction is not another untouched experiment. A scientific change to the protocol needs a new source/window and a new lock. Infrastructure failures are not model failures; any retry must retain and disclose its earlier receipt.

## Limits

This is one household, four months and sparse forecast origins. Missing observations can make the scored subset unrepresentative of outage periods. Long historical gaps require forward filling and can distort fits; the saved counts make that visible. There is no live integration, production claim or energy-savings estimate. The original appliance and crypto results remain unchanged.

## First recorded result

The first run completed on October 23, 2026, in 55.1 seconds without changing the locked protocol or model code. It produced 1,080 forecast records from 19 validation, 18 calibration and 23 test origins. There are 2,952 calendar hours in the source window: 2,740 complete and 212 incomplete. The common completeness rule excluded four validation origins and five calibration origins; no scheduled test origin was excluded.

Validation selected seasonal naive for one hour, gradient boosting for six hours, and Prophet for twenty-four hours. On test, their MAEs were 0.6547, 0.5272 and 0.6606 kWh respectively. Last-value forecasts scored 0.4169, 1.0288 and 0.6983 kWh on those same horizons. The six-hour model improved on that baseline, but the one-hour selection did not carry over successfully. The selected models' interval coverage was 69.6%, 95.7% and 73.9%, against a nominal 80% level.

All six models and all horizons remain in the downloadable records, including models that were not selected. These mixed outcomes are reported without changing model choices or rerunning to improve the comparison. Saved warnings are library deprecations and a processor-count fallback, not failed forecasts.
