from typing import List, Tuple

import matplotlib.pyplot as plt

from . import CovariateScenario


def subplot_forecast_horizon(
    selected_scenarios: List[CovariateScenario],
    dataset: str,
    model_name: str,
    ax: plt.Axes,
    loader_fn: callable,
):
    """plot forecast horizon for a given model and dataset"""
    PATH = f"results/{model_name}/{dataset}"

    for scenario in selected_scenarios:
        forecast_horizon_metrics = loader_fn(PATH, scenario.name)
        ax.plot(forecast_horizon_metrics, label=scenario.name)

    ax.set_xlim(1, len(forecast_horizon_metrics))
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Error (sMAPE)")
    ax.set_title(f"{model_name}")
    ax.legend()


def plot_forecast_horizon_trajectories(
    selected_scenarios: List[CovariateScenario],
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
