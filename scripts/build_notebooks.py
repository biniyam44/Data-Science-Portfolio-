"""Build and optionally execute the two notebooks from saved experiment outputs."""

import argparse
import base64
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parents[1]


def make_notebook(dataset):
    energy = dataset == "energy"
    title = "Hourly appliance energy demand" if energy else "Historical digital-currency prices"
    source_note = (
        "This study uses one Belgian household measured in early 2016. Six complete "
        "ten-minute readings are summed and divided by 1,000 to obtain hourly kWh. "
        "This is appliance energy, not total household energy or instantaneous power. "
        "Source: [UCI, Candanedo (2017), CC BY 4.0](https://doi.org/10.24432/C5VC8G)."
        if energy else
        "The original repository contains 1,000 daily observations from 2018 to 2021. "
        "The target is `close_USD`. Its SAR conversion is excluded because it contains "
        "the same target information. The original dataset license and instrument metadata "
        "remain unverified. [Source listed by the original project](https://www.kaggle.com/datasets/ahmedadam415/digital-currency-time-series)."
    )
    cells = [
        nbformat.v4.new_markdown_cell(f"# {title}\n\nBiniyam Berga\n\n"
                                     "## Results at a glance\n\n"
                                     "The cells below recompute metrics from the saved prediction rows. "
                                     "The headline model is selected on validation error, not test error. "
                                     "All results are historical experiments, not live forecasts."),
        nbformat.v4.new_code_cell(f'''%matplotlib inline
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from forecasting import load_data
from forecasting.evaluation import implementation_sha256
from forecasting.plots import set_style, comparison_plot, forecast_plot, error_plot

ROOT = Path.cwd()
DATASET = "{dataset}"
result_dir = ROOT / "results" / DATASET
summary = json.loads((result_dir / "summary.json").read_text())
predictions = pd.read_csv(result_dir / "predictions.csv")
metrics = pd.read_csv(result_dir / "metrics.csv")
source_path = ROOT / "data" / ("appliances_hourly.csv" if DATASET == "energy" else "dc.csv")
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == summary["input_sha256"]
assert implementation_sha256() == summary["implementation_sha256"]
data = load_data(source_path, DATASET)
set_style()
for horizon in summary["horizons"]:
    model = summary["selected_models"][str(horizon)]
    row = metrics.loc[(metrics.phase == "test") & (metrics.model == model) & (metrics.horizon == horizon)].iloc[0]
    print(f"{{horizon}} steps: {{model}}, MAE {{row.mae:.4f}} {{summary['unit']}}, {{int(row.n)}} forecasts, interval coverage {{row.coverage:.1%}}")'''),
        nbformat.v4.new_markdown_cell("## Context and methods\n\n"
                                     "The first 40% of observations seed the expanding training window. "
                                     "The following three 20% blocks are validation, interval calibration, "
                                     "and final test. Every model forecasts the same origins and horizons. "
                                     "Parameters are fixed in code. Validation chooses the model separately "
                                     "for each horizon; calibration fixes its error-band width.\n\n"
                                     "### Key assumptions\n\n"
                                     "- A forecast is issued after its origin observation becomes available.\n"
                                     "- Refits can use earlier observed test values, but never future values. "
                                     "Test data do not select models or calibrate intervals.\n"
                                     "- The 80% intervals use past absolute forecast errors. Time dependence "
                                     "and regime changes mean 80% future coverage is not guaranteed.\n"
                                     "- One historical series does not establish production or trading performance.\n\n"
                                     f"To regenerate model outputs first: `python time_series_forecasting.py --dataset {dataset} --output results/{dataset}`. "
                                     "This notebook analyzes those outputs without fitting a second, different pipeline."),
        nbformat.v4.new_markdown_cell("## Data\n\n" + source_note),
        nbformat.v4.new_code_cell('''print(f"{len(data):,} observations: {data.index.min()} to {data.index.max()}")
display(data.head(3))
display(pd.DataFrame(summary["splits"]).T[["start", "end", "rows", "n_origins"]])
assert not data.index.has_duplicates
assert data.index.is_monotonic_increasing
assert np.isfinite(data.y).all()'''),
        nbformat.v4.new_markdown_cell("## Results\n\n### Independent metric check\n\n"
                                     "MAE and RMSE are recomputed below from all saved prediction rows. "
                                     "MAPE is a percentage; zero actuals are excluded and counted separately."),
        nbformat.v4.new_code_cell('''for key, rows in predictions.groupby(["phase", "model", "horizon"]):
    phase, model, horizon = key
    saved = metrics.loc[(metrics.phase == phase) & (metrics.model == model) & (metrics.horizon == horizon)].iloc[0]
    error = rows.prediction - rows.actual
    assert np.isclose(np.abs(error).mean(), saved.mae)
    assert np.isclose(np.sqrt(np.square(error).mean()), saved.rmse)
    nonzero = rows.actual.ne(0)
    assert int(nonzero.sum()) == int(saved.mape_n)
    if nonzero.any():
        assert np.isclose((error[nonzero].abs() / rows.actual[nonzero].abs()).mean() * 100, saved.mape_pct)
print("Saved MAE, RMSE, MAPE, and denominators match every prediction group.")
display(metrics.loc[metrics.phase.eq("test"), ["model", "horizon", "n", "mae", "rmse", "mape_pct", "coverage"]].round(4))'''),
        nbformat.v4.new_code_cell('''figure = comparison_plot(metrics, summary["horizons"], summary["unit"], summary["frequency"])
plt.show()'''),
        nbformat.v4.new_markdown_cell("### Forecasts and uncertainty\n\n"
                                     "The longest requested horizon is shown below. The band is calibrated "
                                     "before test and stays fixed. The lines connect separate rolling forecasts; "
                                     "they are not a single forecast path issued on one date."),
        nbformat.v4.new_code_cell('''horizon = max(summary["horizons"])
figure = forecast_plot(predictions, summary, horizon)
plt.show()'''),
        nbformat.v4.new_markdown_cell("### Where the selected model misses\n\n"
                                     "The sign of an error distinguishes underprediction from overprediction. "
                                     "These observations describe this test period, not a causal explanation."),
        nbformat.v4.new_code_cell('''figure = error_plot(predictions, summary, horizon)
plt.show()
model = summary["selected_models"][str(horizon)]
selected = predictions.loc[predictions.phase.eq("test") & predictions.model.eq(model) & predictions.horizon.eq(horizon)].copy()
selected["absolute_error"] = (selected.prediction - selected.actual).abs()
display(selected.nlargest(5, "absolute_error")[["origin", "timestamp", "actual", "prediction", "absolute_error"]].round(4))'''),
        nbformat.v4.new_markdown_cell("## Takeaways\n\n"
                                     "A model selected on validation can still lose to the last-value baseline "
                                     "during test. The comparison below reports that outcome without changing "
                                     "the selection. Larger models are useful only when the measured errors support them."),
        nbformat.v4.new_code_cell('''test = metrics.loc[metrics.phase.eq("test")]
for horizon in summary["horizons"]:
    model = summary["selected_models"][str(horizon)]
    chosen = test.loc[test.model.eq(model) & test.horizon.eq(horizon)].iloc[0]
    baseline = test.loc[test.model.eq("naive") & test.horizon.eq(horizon)].iloc[0]
    relative = 100 * (chosen.mae / baseline.mae - 1) if baseline.mae else np.nan
    print(f"{horizon} steps: {model} MAE differs from last-value baseline by {relative:+.1f}% (negative is better).")
for caveat in summary["limitations"]:
    print(f"- {caveat}")'''),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"{dataset}-{index:02d}"
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "title": title,
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for dataset, filename in (("crypto", "Time_Series_Forecasting.ipynb"),
                              ("energy", "Energy_Demand_Forecasting.ipynb")):
        notebook = make_notebook(dataset)
        if args.execute:
            NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, ROOT / filename)
        if args.execute:
            body, _ = HTMLExporter().from_notebook_node(notebook)
            preview = ROOT / "preview"
            preview.mkdir(exist_ok=True)
            (preview / f"{dataset}.html").write_text(body)
            for cell_index, cell in enumerate(notebook.cells):
                for output_index, cell_output in enumerate(cell.get("outputs", [])):
                    png = cell_output.get("data", {}).get("image/png")
                    if png:
                        (preview / f"{dataset}-{cell_index}-{output_index}.png").write_bytes(base64.b64decode(png))
        print(f"{'Executed' if args.execute else 'Built'} {filename}")


if __name__ == "__main__":
    main()
