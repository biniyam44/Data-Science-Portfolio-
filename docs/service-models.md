# Energy model lifecycle

The service predicts hourly appliance energy at one, six and 24 hours ahead. It uses the same causal feature definitions as the historical experiments, but fitting happens offline. A request does not retrain the model.

## Training and selection

`python scripts/train_service.py --data data/appliances_hourly.csv --registry artifacts/energy` produces a versioned bundle. It does not change the active model. The training CLI, registry CLI and API use `artifacts/energy` by default.

The first 40 percent of readings trains a direct gradient-boosted model for each horizon. Those fixed models compete with last-value and daily seasonal baselines on the following 20 percent. All candidates use identical origins, and their targets stay within that block. The lowest validation MAE wins each horizon; model-name order breaks ties. The service can therefore serve a baseline when the more complicated model does not improve it.

Selected boosted models are refitted through the end of validation and then frozen. The next 20 percent supplies absolute residuals for an empirical 80 percent interval. The interval width is the finite-sample residual order statistic used by the experiment pipeline. It is not a guarantee of future coverage, especially with serial dependence or a changed household.

The training command does not use the final 20 percent for fitting, selection, calibration or promotion. Changing those target values leaves the fitted predictions and calibration unchanged; a regression test checks this. The complete input file is hashed for provenance, so its version still changes. This existing historical dataset was inspected during earlier project development. Reserving a block in this workflow does not turn it into fresh external validation.

Each bundle records the training and calibration cutoffs, validation comparisons, selected estimators, interval widths, package versions, feature names and implementation hash, source hash, split boundaries, and a training-input distribution reference. Its content-derived version identifies both its metadata and serialized estimators.

## Promotion and rollback

Inspect a candidate:

```bash
python scripts/model_registry.py --registry artifacts/energy show --version energy-VERSION
```

Use the actual identifier returned by training, not the placeholder. Promote it only after reviewing validation comparisons and operational checks:

```bash
python scripts/model_registry.py --registry artifacts/energy promote energy-VERSION
python scripts/model_registry.py --registry artifacts/energy rollback
```

Promotion verifies artifact hashes and the validation-selection rule. A lock and atomic pointer replacement serialize registry changes. Rollback restores the previous promoted version. Restart a service process that has already loaded its bundle to adopt the new active pointer. Neither action changes the historical experiment results or claims that the new version outperforms an incumbent in production.

## Input and time contract

Send at least 169 contiguous, increasing, hour-aligned observations. Values are finite nonnegative kWh consumed during each complete hour, not instantaneous kW. The service accepts the original dataset's timezone-naive source clock; it does not infer a time zone or silently convert daylight-saving timestamps. A real deployment needs an explicit clock contract for its new source.

The HTTP demonstration accepts at most 1,680 readings and values up to 1,000,000 kWh per hour. This deliberately generous upper bound prevents numerical overflow from untrusted extreme inputs; it is not a plausible household consumption limit. Authentication, body-size limits and per-process request limits apply before request validation. Supply an optional 32-character lowercase hexadecimal `request_id` to retry a forecast without recording it twice.

The final reading is the forecast origin. A forecast labelled with a target timestamp refers to the hour starting at that timestamp. Only completed input hours may be supplied. The request origin cannot precede the bundle's calibration cutoff, because that would reuse future information from interval fitting. Inputs with gaps, duplicates, reversed order, an extra measurement column, missing values or inconsistent timestamps are rejected rather than imputed.

## Registry trust boundary

Model bundles are local, trusted artifacts written by the offline training command. The server must own the registry; untrusted users must not have write access. The API must not offer model uploads, arbitrary artifact paths or promotion endpoints. Version identifiers cannot contain paths, regular bundle files are hash checked, and package and feature implementations must match before loading.

The estimators use joblib, which deserializes Python objects. File hashes detect accidental corruption but do not authenticate an attacker-controlled registry. Never load a bundle supplied by an unknown party, even if its manifest is internally consistent. Production distribution would require authenticated artifact storage and signatures, plus an isolated training and release process.

## Monitoring

SQLite stores request IDs, versioned forecasts, input hashes, measured inference latency and descriptive input shift. Repeating the same request is idempotent; reusing its ID with different input or predictions is a conflict. A per-request training reference is used for shift calculations so different model versions are not silently compared against one shared reference.

An outcome is attached to a recorded request and one of its target timestamps. It must use the same unit, and its `observed_at` must be at or after the end of the complete target hour. Exact retries are accepted without double counting; conflicting observations are rejected. The caller is responsible for supplying genuine observations and a truthful source-clock observation time.

Metrics remain separate by model version and forecast horizon. They include MAE, RMSE, empirical interval coverage and width, latency percentiles, and input rejection counts. Missing completed outcomes require an explicit source-clock `as_of`; without it, unobserved targets are simply pending. Reported drift is a descriptive population-stability index and out-of-training-range fraction over the last 168 inputs, not an automatic retraining decision.

This is a single-host demonstration. It does not claim distributed registry transactions, authenticated sensor ingestion, continuous production monitoring, load-tested capacity on a public host, or a universal anomaly threshold.
