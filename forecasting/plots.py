"""Figures shared by the two case-study notebooks."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "naive": "Last value", "drift": "Drift", "seasonal_naive": "Seasonal naive",
    "arima": "ARIMA", "prophet": "Prophet", "boosted": "Boosted trees",
}


def set_style():
    plt.rcParams.update({
        "figure.figsize": (10, 4.5), "figure.dpi": 120,
        "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#26352e",
        "text.color": "#26352e", "axes.titlelocation": "left",
        "axes.titlepad": 15, "savefig.bbox": "tight",
    })


def comparison_plot(metrics, horizons, unit, frequency):
    """Compare every requested model, without selecting on test performance."""
    test = metrics.loc[metrics.phase.eq("test")]
    fig, axes = plt.subplots(1, len(horizons), figsize=(13, 4), layout="constrained")
    for ax, horizon in zip(axes, horizons):
        rows = test.loc[test.horizon.eq(horizon)].sort_values("mae", ascending=False)
        ax.barh([LABELS.get(m, m) for m in rows.model], rows.mae, color="#315f51")
        period = "day" if frequency == "D" else "hour"
        ax.set_title(f"{horizon} {period}{'' if horizon == 1 else 's'} ahead")
        ax.set_xlabel(f"Test MAE ({unit})")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out rolling forecasts: MAE (each panel has its own scale)", fontsize=14)
    return fig


def forecast_plot(predictions, summary, horizon):
    model = summary["selected_models"][str(horizon)]
    selected = predictions.loc[
        predictions.phase.eq("test") & predictions.model.eq(model)
        & predictions.horizon.eq(horizon)
    ].sort_values("timestamp")
    dates = pd.to_datetime(selected.timestamp)
    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(dates, selected.lower, selected.upper, color="#315f51", alpha=0.16,
                    label="80% empirical prediction interval")
    ax.plot(dates, selected.actual, "o-", color="#26352e", label="Observed", markersize=3)
    ax.plot(dates, selected.prediction, "s--", color="#a85c23", label="Forecast", markersize=3)
    ax.set_title(f"{LABELS.get(model, model)} at {horizon} steps: selected on validation MAE")
    ax.set_ylabel(summary["unit"])
    ax.set_xlabel("Target date; each point comes from a separate forecast origin")
    tick_indexes = sorted({0, len(dates) // 3, 2 * len(dates) // 3, len(dates) - 1})
    ax.set_xticks(dates.iloc[tick_indexes])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.18)
    return fig


def error_plot(predictions, summary, horizon):
    model = summary["selected_models"][str(horizon)]
    selected = predictions.loc[
        predictions.phase.eq("test") & predictions.model.eq(model)
        & predictions.horizon.eq(horizon)
    ]
    fig, ax = plt.subplots(layout="constrained")
    error = selected.prediction - selected.actual
    ax.scatter(selected.actual, error, color="#315f51", edgecolor="#26352e", alpha=0.85)
    ax.axhline(0, color="#26352e", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Observed value ({summary['unit']})")
    ax.set_ylabel(f"Forecast minus observed ({summary['unit']})")
    ax.set_title(f"Held-out errors: {LABELS.get(model, model)}, {horizon} steps ahead")
    ax.grid(alpha=0.18)
    return fig
