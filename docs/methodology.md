# Evaluation decisions

## Forecast origin and target

Each origin is the last fully observed timestamp available to the model. Price forecasts use daily closing observations and predict 1, 7, or 30 days ahead. Energy forecasts use complete hourly appliance consumption and predict 1, 6, or 24 hours ahead. An energy row is labelled by the hour's start, but it is not available until that hour ends.

All models receive the same history, issue dates, and target dates. Errors are measured on the target at each stated horizon, not on a mixture of unrelated forecast lengths. Seasonal baselines use the same clock and cannot inspect the target they predict.

## Chronological partitions

The first 40% of observations provide initial training history. The next 20% are validation, the next 20% calibrate forecast intervals, and the last 20% are the final test period. Exact boundaries and origins are saved in `splits.json`. Origins whose requested horizon crosses a partition boundary are excluded from that partition's comparison.

Models refit on expanding history. Origins are spaced five days apart in the daily price study and 25 hours apart in the hourly energy study. These strides visit every weekday or hour instead of repeatedly sampling a single clock slot. The split record includes weekday counts and, for energy, hour counts. The `--quick` option reduces the number of origins for development; quick output is not a substitute for the full result set.

For each horizon, the reported selected model is the one with the lowest validation MAE; ties are broken by model name. All models are still evaluated and shown. The final test period is not used to select a model, tune hyperparameters, or calibrate intervals in the recorded run. A later test origin may incorporate observations from earlier in the test period after they become known. The test therefore measures rolling deployment with refits, not one fixed model asked to forecast the entire period at once.

The configurations in the code are fixed rather than the result of an exhaustive search. If those configurations are changed after inspecting the final results, that period is no longer an untouched test for the revised experiment. A new holdout period or dataset is needed for a fresh confirmation.

Development runs exposed a schedule that sampled different fixed weekdays or hours in each partition. The recorded comparison corrects that calendar-sampling problem. The model configurations were not tuned to improve the test scores, but this historical period has been inspected during development and is not a fresh external confirmation.

## Models and features

The simple models establish whether extra complexity is useful: last observed value, linear drift from the observed history, and a seasonal repeat. The repeat period is seven days for prices and 24 hours for energy. ARIMA and Prophet are fitted separately at every origin. Gradient-boosted trees predict the change from the last observed value directly for each horizon, with features built from past target values, lagged changes, rolling statistics, and known calendar values. The price model also uses historical volume.

Training examples for a particular horizon are retained only if their outcome was available by that origin. Rolling calculations must look backward. Features may include the current completed observation, but not a future observation or a column containing the target in another currency. A future timestamp's calendar fields are known; its measured demand, price, and volume are not.

The energy experiment excludes future weather and indoor sensor measurements. Supplying observed future weather would test a different information setting and could give an unfair advantage over models that receive only consumption history. Negative forecasts are clipped to zero for every model, consistently with the nonnegative targets.

## Errors and intervals

MAE and RMSE stay in the target's units: USD for price and kWh for hourly appliance consumption. MAPE is reported as a percentage, not a fraction with a percent sign. Zero-valued targets are excluded from MAPE, and `mape_n` reports its denominator. MASE scales absolute error by a seasonal-naive error estimated from the history at each origin, not from its future outcomes. Zero scales are excluded, with the valid count reported as `mase_n`.

Nominal 80% prediction intervals use absolute errors collected in the calibration period, separately by model and horizon. For `n` calibration errors, the half-width is the sorted error at rank `ceil((n + 1) × 0.8)`. At least five calibration origins are required. Half-widths are fixed before the final test forecasts are scored, with lower bounds clipped to zero. The results report observed test coverage and interval width, so a wide but unhelpful interval is not presented as an accurate point forecast.

These are empirical residual intervals. Time-series errors can be dependent, and changing volatility or demand can move the test distribution away from calibration. The nominal level is therefore not a guarantee of 80% future coverage. Sparse calibration samples and overlapping long-horizon forecasts also limit the precision of comparisons.

Bounds are inclusive. Coverage comparisons allow `1e-9 × max(1, abs(actual))` at a boundary so decimal CSV serialization cannot change whether an observation exactly on that boundary is counted. This tolerance is far below the source measurement precision.

## Reading a result

Compare a model against the relevant baseline at the same horizon and origins. Then inspect the errors over time: an average alone can hide a price regime change or a few demand spikes. Validation and test rankings may disagree. That disagreement is useful evidence and should not be removed from the case study.

The saved result files include individual forecasts, metrics, split definitions, and source fingerprints. `summary.json` also records each model's three largest test errors per horizon and any fitting warnings. A failed model fit stops the experiment instead of silently dropping that model or origin from the comparison. The notebooks read these files rather than recreating a second implementation. Results from `--quick` runs should stay in `outputs/` and should not replace the full evaluation used on the website.

## Limits

Both projects use historical datasets. The price series ends in January 2021 and is not a live market feed. It does not establish trading performance, transaction-cost-adjusted returns, or investment value.

The energy series covers about four and a half months in one building. It is too short to establish annual behavior and too narrow to establish generalization across households. No tariff, control policy, or operational trial is included, so forecast accuracy does not demonstrate cost or energy savings.

Dependency versions, random seeds, source hashes, and commands are recorded for reproducibility. Numerical libraries can still produce small platform-dependent differences. The repository does not claim production readiness solely because a historical backtest completes.
