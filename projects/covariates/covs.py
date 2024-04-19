import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

import nnts.data.metadata
import nnts.experiments.scenarios
import nnts.experiments.scenarios as scenarios
import nnts.metrics

errors = {
    "us_births": np.linspace(0, 0.195, 8).tolist(),
    "tourism": np.linspace(0, 1.65, 8).tolist(),
    "solar": np.linspace(0, 0.702, 8).tolist(),
    "hospital": np.linspace(0, 1.65, 8).tolist(),
    "electricity": np.linspace(0, 1.65, 8).tolist(),
    "traffic": np.linspace(0, 0.6, 8).tolist(),
}


def calculate_pearson(df):
    df = df.copy()
    df["y_lead"] = df[["y_lead_1", "unique_id"]].groupby("unique_id").shift(1)
    df = df.dropna()
    pearson = scipy.stats.pearsonr(df["y"], df["y_lead"])
    print(pearson)
    return pearson[0]


def copy_with_noise(x, level, lead):
    noisy_x = (
        x
        + (x.std() * np.random.randn(len(x)) * level)
        + (x.mean() * np.random.randn(len(x)) * level)
    )
    return noisy_x.shift(-lead)


def prepare(data, scenario):
    pearson = 0
    conts = []
    noise = 0
    if scenario.covariates > 0:
        data["y_lead_1"] = (
            data[["y", "unique_id"]]
            .groupby("unique_id")
            .transform(copy_with_noise, scenario.error, 1)
            .ffill()
        )
        pearson = calculate_pearson(data)
        conts.append("y_lead_1")
    if scenario.covariates > 1 and scenario.skip != 1:
        data["y_lead_2"] = (
            data[["y", "unique_id"]]
            .groupby("unique_id")
            .transform(copy_with_noise, scenario.error, 2)
            .ffill()
        )
        conts.append("y_lead_2")
    if scenario.covariates > 2:
        data["y_lead_3"] = (
            data[["y", "unique_id"]]
            .groupby("unique_id")
            .transform(copy_with_noise, scenario.error, 3)
            .ffill()
        )
        conts.append("y_lead_3")
    scenario.conts = conts
    scenario.pearson = pearson
    scenario.noise = noise
    scenario.covariates = len(conts)
    return data, scenario


def get_chart_data(df, fh, cov, metric):
    df[f"base_{metric}"] = df.loc[
        (df["covariates"] == 0) & (df["prediction_length"] == fh) & (df["seed"] == 42),
        metric,
    ].item()
    return (
        df[
            (df["covariates"] == cov)
            & (df["prediction_length"] == fh)
            & (df["seed"] == 42)
        ]
        .set_index("pearson")
        .sort_index(ascending=False)[[f"base_{metric}", metric]]
    )


class CSVFileAggregator:
    def __init__(self, path: str, filename: str):
        self.path = path
        self.filename = filename

    def __call__(self) -> pd.DataFrame:
        data_list = []
        for filename in os.listdir(self.path):
            if filename.endswith(".json"):
                with open(os.path.join(self.path, filename), "r") as file:
                    data = json.load(file)
                    data_list.append(data)
        # Concatenate DataFrames if needed
        results = pd.DataFrame(data_list)
        results.to_csv(f"{self.path}/{self.filename}.csv", index=False)
        return results


def add_y_hat(df, y_hat, prediction_length):
    i = 0
    df_list = []
    for name, group in df.groupby("unique_id", sort=False):
        group["y_hat"] = None
        group["y_hat"][-prediction_length:] = y_hat[i].squeeze()
        i += 1
        df_list.append(group)
    return df_list


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


def prepare_scenarios(
    df_orig, scenario_list: List[nnts.experiments.CovariateScenario]
) -> List[nnts.experiments.CovariateScenario]:
    new_scenario_list = []
    for scenario in scenario_list:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        new_scenario_list.append(prepare(df_orig.copy(), scenario)[1])
    scenario_list = new_scenario_list
    return scenario_list


def univariate_results(
    scenario: nnts.experiments.CovariateScenario,
    metadata: nnts.data.metadata.Metadata,
    forecast_horizon: int,
    path: str,
):
    y = torch.load(f"{path}/{scenario.name}_y.pt")
    y_hat = torch.load(f"{path}/{scenario.name}_y_hat.pt")

    return nnts.metrics.calc_metrics(
        y[:, :forecast_horizon, :],
        y_hat[:, :forecast_horizon, :],
        metadata.freq,
        metadata.seasonality,
    )


def plot_pcc_charts(
    model_name: str, scenario_covariate: int, dataset_list: List[str], path: str = None
):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(dataset_list), figsize=(20, 5), sharey=True
    )
    for i, dataset in enumerate(dataset_list):
        df_orig, metadata = nnts.data.load(dataset)
        PATH = f"results/{model_name}/{metadata.dataset}"
        scenario_list: List[nnts.experiments.CovariateScenario] = []
        # Models for full forecast horizon with covariates
        scenario_list.append(
            nnts.experiments.CovariateScenario(
                metadata.prediction_length, 0, covariates=0
            ),
        )
        for covariates in [scenario_covariate]:
            for error in errors[metadata.dataset]:
                scenario_list.append(
                    nnts.experiments.CovariateScenario(
                        metadata.prediction_length, error, covariates=covariates
                    )
                )

        scenario_list = prepare_scenarios(df_orig, scenario_list)

        # Select the univariate scenario
        univariate_scenario = scenario_list[0]
        forecast_horizon = scenario_covariate
        metrics_list = []
        pearson_list = []

        # Calculate metrics and collect data for the univariate scenario
        metrics_list.append(
            univariate_results(univariate_scenario, metadata, forecast_horizon, PATH)[
                "smape"
            ]
        )

        # Calculate metrics and collect data for other scenarios
        for scenario in scenario_list[1:]:
            if scenario.covariates == forecast_horizon:
                y = torch.load(f"{PATH}/{scenario.name}_y.pt")
                y_hat = torch.load(f"{PATH}/{scenario.name}_y_hat.pt")

                metrics = nnts.metrics.calc_metrics(
                    y[:, :forecast_horizon, :],
                    y_hat[:, :forecast_horizon, :],
                    metadata.freq,
                    metadata.seasonality,
                )["smape"]
                metrics_list.append(metrics)
                pearson_list.append(scenario.pearson)

        # Format the pearson_list values to two decimal places
        pearson_list = [round(value, 2) for value in pearson_list]

        # Plot the data on the current subplot
        ax = axes[i]
        ax.plot(pearson_list, metrics_list[:1] * 8, label="univariate")
        ax.plot(pearson_list, metrics_list[1:], label="covariate")
        ax.set_xticks(pearson_list)
        ax.set_xlabel("Correlation (PCC)")
        ax.set_ylabel("Error (sMAPE)")
        ax.set_title(f"{metadata.dataset}")
        ax.legend()

    # Adjust layout and display the figure
    plt.suptitle(
        f"{model_name}  sMAPE vs pearson values. forecast horizon: {forecast_horizon}, covariates: {forecast_horizon}"
    )
    plt.tight_layout()
    if path:
        full_path = f"{path}/{model_name}_k_{scenario_covariate}_smape_vs_pearson.png"
        print("Saving to", full_path)
        plt.savefig(full_path)
    else:
        plt.show()
    plt.close()
    # plt.savefig(f"results/{model_name}/{model_name}_smape_vs_pearson.png")
    # plt.savefig(f"{ARTICLE_PATH}/{model_name}_smape_vs_pearson.png")
    # plt.show()
    return plt
