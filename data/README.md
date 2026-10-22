# Data sources

## Daily prices: `dc.csv`

This is the file already tracked in the original project. It contains 1,000 daily records from 2018-05-07 to 2021-01-30, stored newest first. The forecasting loader sorts them by date and checks the daily sequence before fitting a model.

The original source link is [Digital Currency Time Series on Kaggle](https://www.kaggle.com/datasets/ahmedadam415/digital-currency-time-series). The upstream license and instrument identity have not been independently confirmed. The project therefore describes the series as digital-currency prices and does not grant redistribution rights over the original file. Confirm its terms before reusing the raw data elsewhere.

The target is `close_USD`. The corresponding `close_SAR` value is exactly `close_USD × 3.7504` in this file, so it is excluded from model inputs. Forecasts are issued after an observation closes. Only volume and price information available by that point may enter the features; later high, low, close, and volume values may not.

## Appliance energy: `appliances_hourly.csv`

Source: Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G).

The [UCI dataset page](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) licenses the data under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). This project's hourly file is an adaptation: it retains the `date` and `Appliances` fields, groups six ten-minute measurements into each complete hour, and converts their summed Wh to kWh. Credit the original creator when reusing it.

`timestamp` identifies the start of the recorded hour; `y` is appliance energy consumed in that hour, in kWh. It becomes available only after the hour is complete. The raw timestamps have no timezone offset. They remain in the source's recorded local clock; the transformation does not invent a UTC offset or a daylight-saving correction.

The source has 19,735 measurements from 2016-01-11 17:00 to 2016-05-27 18:00. There are no duplicate timestamps or interior gaps in its ten-minute sequence. The final 18:00 hour contains only one measurement and is excluded. The resulting file has 3,289 consecutive complete hours, ending at 2016-05-27 17:00.

Recreate the hourly file with the Python standard library:

```bash
python scripts/prepare_energy.py
```

The first run downloads the original CSV into the ignored `data/raw/` directory. Subsequent runs reuse that pinned file. To use an existing download:

```bash
python scripts/prepare_energy.py --source /path/to/energydata_complete.csv
```

The script rejects changed source bytes, malformed values, duplicate or out-of-order timestamps, and missing interior intervals. It writes generated files atomically. [energy_source.json](energy_source.json) records the raw and derived SHA-256 hashes, date bounds, dropped edge readings, transformation, citation, and license.

Only past appliance consumption and known calendar values are used in this forecasting study. Indoor measurements, interpolated weather, lights, and the source's random variables are not supplied as future predictors. This is one building's appliance load, not a utility demand series or a whole-building electricity meter.
