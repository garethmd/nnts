from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def subplot_forecast_horizon(
    selected_scenarios: List[Any],
    dataset: str,
    model_name: str,
    ax: plt.Axes,
    loader_fn: callable,
):
    """plot forecast horizon for a given model and dataset"""
    for scenario in selected_scenarios:
        forecast_horizon_metrics = loader_fn(f"{model_name}/{dataset}/{scenario.name}")
        ax.plot(forecast_horizon_metrics, label=scenario.name)

    ax.set_xlim(1, len(forecast_horizon_metrics))
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Error (sMAPE)")
    ax.set_title(f"{model_name}")
    ax.legend()


def plot_forecast_horizon_trajectories(
    selected_scenarios: List[Any],
    dataset: str,
    covariates: List[int] | int,
    model_names: List[str] = None,
    loader_fn: callable = None,
    figsize: Tuple[int, int] = (20, 5),
    path: str = None,
):
    # FORECAST HORIZON TRAJECTORIES
    if model_names is None:
        raise ValueError("Please provide a list of model names to compare")
    if loader_fn is None:
        raise ValueError("Please provide a loader function to load the data")

    fig, axes = plt.subplots(
        nrows=1, ncols=len(model_names), figsize=figsize, sharey=True
    )

    for i, model_name in enumerate(model_names):
        ax = axes[i] if len(model_names) > 1 else axes
        subplot_forecast_horizon(
            selected_scenarios, dataset, model_name, ax, loader_fn=loader_fn
        )

    plt.suptitle(f"{dataset} forecast horizon trajectories {covariates} covariates")
    plt.tight_layout()
    if path:
        if isinstance(covariates, list):
            covariates = "_".join(map(str, covariates))
        model_name = "_".join(model_names)
        full_path = f"{path}/{model_name}_{dataset}_k_{covariates}_trajectory.png"
        print("Saving to", full_path)
        plt.savefig(full_path)
    else:
        plt.show()

    return plt


def plot(df_test, prediction_length, start_idx=0):
    num_plots = min(len(df_test), 4)
    fig, axes = plt.subplots(
        nrows=num_plots // 2 + num_plots % 2, ncols=min(num_plots, 2), figsize=(20, 10)
    )
    axes = np.ravel(axes)  # Flatten the axes array

    for idx, ax in enumerate(axes):
        if idx < len(df_test):
            df_test[start_idx + idx].set_index("ds").tail(prediction_length * 5)[
                ["y", "y_hat"]
            ].plot(ax=ax)
        else:
            ax.axis("off")  # Hide empty subplots if df_test length is less than 4
    return fig


def plot_forecasts_vs_actuals(y, y_hat):
    """
    Plots the forecasts vs actuals for the first 4 time series.

    Parameters:
    - y: Actual values tensor of shape [320, 24]
    - y_hat: Forecasted values tensor of shape [320, 24]
    """
    num_series_to_plot = 4
    timesteps = y.shape[1]

    fig, axes = plt.subplots(num_series_to_plot, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Forecasts vs Actuals for First 4 Time Series")

    for i in range(num_series_to_plot):
        axes[i].plot(range(timesteps), y[i], label="Actual", color="blue")
        axes[i].plot(
            range(timesteps),
            y_hat[i],
            label="Forecast",
            color="orange",
            linestyle="dashed",
        )
        axes[i].set_title(f"Time Series {i+1}")
        axes[i].legend()

    plt.xlabel("Timesteps")
    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )  # Adjust layout to make room for the main title
    plt.show()


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Function to plot the forecasts vs actuals for the first 4 time series
def plotly_forecasts_vs_actuals(y, y_hat):
    """
    Plots the forecasts vs actuals for the first 4 time series using Plotly.

    Parameters:
    - y: Actual values tensor of shape [320, 24]
    - y_hat: Forecasted values tensor of shape [320, 24]
    """
    y_hat = y_hat.squeeze(-1)
    y = y.squeeze(-1)
    num_series_to_plot = 4
    timesteps = y.shape[1]

    fig = make_subplots(
        rows=num_series_to_plot,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Time Series {i+1}" for i in range(num_series_to_plot)],
    )

    for i in range(num_series_to_plot):
        fig.add_trace(
            go.Scatter(
                x=list(range(timesteps)),
                y=y[i],
                mode="lines",
                name=f"Actual {i+1}",
                line=dict(color="blue"),
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(timesteps)),
                y=y_hat[i],
                mode="lines",
                name=f"Forecast {i+1}",
                line=dict(color="orange", dash="dash"),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=1000,
        width=800,
        title_text="Forecasts vs Actuals for First 4 Time Series",
        showlegend=True,
    )

    fig.update_xaxes(title_text="Timesteps", row=num_series_to_plot, col=1)

    fig.show()
