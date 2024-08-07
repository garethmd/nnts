import json
import os
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import metrics_old
import numpy as np
import pandas as pd
import scipy
import torch

import nnts.datasets
import nnts.torch.models
import nnts.torch.preprocessing as preprocessing
from nnts import datasets, utils


@dataclass
class CovariateScenario:
    prediction_length: int
    error: int
    conts: list = field(default_factory=list)
    pearson: float = 0
    noise: float = 0
    covariates: int = 0
    seed: int = 42
    skip: int = 0

    def copy(self):
        return self.__class__(
            prediction_length=self.prediction_length,
            error=self.error,
            conts=self.conts.copy(),
            pearson=self.pearson,
            noise=self.noise,
            covariates=self.covariates,
            seed=self.seed,
            skip=self.skip,
        )

    @property
    def name(self):
        if self.skip == 1:
            return f"cov-{self.covariates}-pearsn-{str(round(self.pearson, 2))}-pl-{str(self.prediction_length)}-seed-{self.seed}-skip-{self.skip}"
        return f"cov-{self.covariates}-pearsn-{str(round(self.pearson, 2))}-pl-{str(self.prediction_length)}-seed-{self.seed}"


errors = {
    "us_births": np.linspace(0, 0.195, 8).tolist(),
    "tourism_monthly": np.linspace(0, 1.65, 8).tolist(),
    "solar": np.linspace(0, 0.702, 8).tolist(),
    "hospital": np.linspace(0, 1.65, 8).tolist(),
    "electricity_hourly": np.linspace(0, 1.65, 8).tolist(),
    "traffic_weekly": np.linspace(0, 0.6, 8).tolist(),
}


file_map = {
    "tourism_monthly": "tourism_monthly_dataset.tsf",
    "hospital": "hospital_dataset.tsf",
    "traffic_weekly": "traffic_weekly_dataset.tsf",
    "electricity_hourly": "electricity_hourly_dataset.tsf",
}


def list_available_datasets() -> List[str]:
    return ["traffic_weekly", "electricity_hourly", "tourism_monthly", "hospital"]


def list_available_models() -> List[str]:
    return ["base-lstm", "seg-lstm"]


def calculate_pearson(df):
    df = df.copy()
    df["y_lead"] = df[["y_lead_1", "unique_id"]].groupby("unique_id").shift(1)
    df = df.dropna()
    pearson = scipy.stats.pearsonr(df["y"], df["y_lead"])
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


def add_y_hat(df, y_hat, prediction_length):
    i = 0
    df_list = []
    for name, group in df.groupby("unique_id", sort=False):
        group["y_hat"] = None
        group["y_hat"][-prediction_length:] = y_hat[i].squeeze()
        i += 1
        df_list.append(group)
    return df_list


def prepare_scenarios(
    df_orig, scenario_list: List[CovariateScenario]
) -> List[CovariateScenario]:
    new_scenario_list = []
    for scenario in scenario_list:
        nnts.torch.utils.seed_everything(scenario.seed)
        new_scenario_list.append(prepare(df_orig.copy(), scenario)[1])
    scenario_list = new_scenario_list
    return scenario_list


def univariate_results(
    scenario: CovariateScenario,
    metadata: datasets.Metadata,
    forecast_horizon: int,
    path: str,
):
    y = torch.load(f"{path}/{scenario.name}_y.pt")
    y_hat = torch.load(f"{path}/{scenario.name}_y_hat.pt")

    return metrics_old.calc_metrics(
        y[:, :forecast_horizon, :],
        y_hat[:, :forecast_horizon, :],
        metadata.freq,
        metadata.seasonality,
    )


def plot_pcc_charts(
    model_name: str,
    scenario_covariate: int,
    dataset_list: List[str],
    path: str = None,
    results_path: str = "nb-results",
    data_path: str = "data",
):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(dataset_list), figsize=(20, 5), sharey=True
    )
    for i, dataset_name in enumerate(dataset_list):
        df_orig, metadata = nnts.datasets.load(
            dataset_name,
            data_path,
            metadata_filename=f"{model_name}-monash.json",
        )
        PATH = os.path.join(results_path, model_name, metadata.dataset)
        scenario_list: List[CovariateScenario] = []
        # Models for full forecast horizon with covariates
        scenario_list.append(
            CovariateScenario(metadata.prediction_length, 0, covariates=0),
        )
        for covariates in [scenario_covariate]:
            for error in errors[metadata.dataset]:
                scenario_list.append(
                    CovariateScenario(
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

                metrics = metrics_old.calc_metrics(
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
    return plt


def model_factory(
    model_name: str,
    params: utils.Hyperparams,
    scenario: CovariateScenario,
    metadata: datasets.Metadata,
):
    if model_name == "base-lstm":
        return nnts.torch.models.BaseLSTM(
            nnts.torch.models.LinearModel,
            params,
            preprocessing.masked_mean_abs_scaling,
            scenario.covariates + 1,
        )
    elif model_name == "base-future-covariate-lstm":
        return nnts.torch.models.BaseFutureCovariateLSTM(
            nnts.torch.models.LinearModel,
            params,
            preprocessing.masked_mean_abs_scaling,
            1,
            known_future_covariates=scenario.covariates,
        )
    elif model_name == "seg-lstm":
        return nnts.torch.models.SegLSTM(
            nnts.torch.models.LinearModel,
            params,
            preprocessing.masked_mean_abs_scaling,
            scenario.covariates + 1,
            metadata.seasonality,
        )
    elif model_name == "unrolled-lstm":
        return nnts.torch.models.UnrolledLSTM(
            nnts.torch.models.LinearModel,
            params,
            preprocessing.masked_mean_abs_scaling,
            scenario.covariates + 1,
        )
    elif model_name == "unrolled-future-covariate-lstm":
        return nnts.torch.models.UnrolledFutureCovariateLSTM(
            nnts.torch.models.LinearModel,
            params,
            preprocessing.masked_mean_abs_scaling,
            1,
            known_future_covariates=scenario.covariates,
        )
    else:
        raise ValueError(f"Model {model_name} not found.")
