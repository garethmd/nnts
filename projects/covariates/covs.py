import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

import nnts.experiments.scenarios as scenarios

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
        )
        pearson = calculate_pearson(data)
        conts.append("y_lead_1")
    if scenario.covariates > 1:
        data["y_lead_2"] = (
            data[["y", "unique_id"]]
            .groupby("unique_id")
            .transform(copy_with_noise, scenario.error, 2)
        )
        conts.append("y_lead_2")
    if scenario.covariates > 2:
        data["y_lead_3"] = (
            data[["y", "unique_id"]]
            .groupby("unique_id")
            .transform(copy_with_noise, scenario.error, 3)
        )
        conts.append("y_lead_3")
    data = data.dropna()
    scenario.conts = conts
    scenario.pearson = pearson
    scenario.noise = noise
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
