import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

import nnts
import nnts.data.datasets
import nnts.experiments.scenarios as scenarios
import nnts.metrics
import nnts.models.hyperparams
import nnts.torch.data.preprocessing as preprocessing
import nnts.torch.models.tsarlstm as tsarlstm
import nnts.torch.trainer as trainer


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


def prepare_covariates(data, scenario):
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
            df_test[start_idx + idx].set_index("ds").tail(prediction_length * 5).plot(
                ax=ax
            )
        else:
            ax.axis("off")  # Hide empty subplots if df_test length is less than 4
    return fig


def main():
    errors = {"us_births": np.linspace(0, 0.195, 8).tolist()}
    errors["us_births"]

    df, metadata = nnts.data.datasets.load("us_births")
    scenario = scenarios.CovariateScenario(
        metadata.prediction_length, 0.0, covariates=0
    )
    params = nnts.models.hyperparams.Hyperparams()

    df, scenario = prepare_covariates(df, scenario)
    trn_dl, val_dl, test_dl = preprocessing.split(df, metadata, scenario, params)
    name = f"cov-{scenario.covariates}-pearsn-{str(round(scenario.pearson, 2))}-pl-{str(scenario.prediction_length)}"

    net = tsarlstm.TsarLSTM(
        tsarlstm.LinearModel, params, preprocessing.masked_mean_abs_scaling
    )
    # best_state_dict = trainer.train(
    #    net, trn_dl, val_dl, params, metadata, name, logger=None
    # )
    best_state_dict = torch.load(name)
    net.load_state_dict(best_state_dict)

    y, y_hat = trainer.eval(
        net, test_dl, scenario.prediction_length, metadata.context_length
    )
    test_metrics = nnts.metrics.calc_metrics(
        y, y_hat, metadata.freq, metadata.seasonality
    )
    print(test_metrics)

    df_list = add_y_hat(df, y_hat, scenario.prediction_length)
    sample_preds = plot(df_list, scenario.prediction_length)


if __name__ == "__main__":
    main()
