import argparse
import os
from typing import List

import gluonts
import pandas as pd
import torch

import nnts
import nnts.data
import nnts.experiments
import nnts.loggers
import nnts.metrics
import nnts.models
import nnts.pandas
import nnts.torch.data
import nnts.torch.data.datasets
import nnts.torch.data.preprocessing
import nnts.torch.models
import nnts.torch.models.trainers as trainers


def main(
    data_path: str,
    model_name: str,
    base_model_name: str,
    dataset_name: str,
    results_path: str,
):
    # Set up paths and load metadata
    data_path = "projects/rnn-covariates/data/"

    metadata_path = os.path.join(data_path, f"{base_model_name}-monash.json")
    metadata = nnts.data.metadata.load(dataset_name, path=metadata_path)
    datafile_path = os.path.join(data_path, metadata.filename)
    PATH = os.path.join(results_path, model_name, metadata.dataset)

    # Load data
    df_orig, *_ = nnts.pandas.read_tsf(datafile_path)
    params = nnts.models.Hyperparams()
    splitter = nnts.pandas.LastHorizonSplitter()

    # Create output directory if it doesn't exist
    nnts.loggers.makedirs_if_not_exists(PATH)

    # Set parameters
    params.batch_size = 32
    params.batches_per_epoch = 50

    # Calculate next month and unix timestamp
    next_month = df_orig["ds"] + pd.DateOffset(months=1)
    df_orig["month"] = next_month.dt.month
    df_orig["unix_timestamp"] = (
        df_orig["ds"] - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")

    # Normalize data
    max_min_scaler = nnts.torch.data.preprocessing.MaxMinScaler()
    max_min_scaler.fit(df_orig, ["month", "unix_timestamp"])
    df_orig = max_min_scaler.transform(df_orig, ["month", "unix_timestamp"])

    lag_seq = gluonts.time_feature.lag.get_lags_for_frequency(metadata.freq)

    df_orig, lag_conts = prepare_lags(df_orig, lag_seq)
    df_orig = df_orig.dropna()

    scenario_list: List[nnts.experiments.Scenario] = []

    # Add the baseline scenarios
    for seed in [42]:
        scenario_list.append(
            nnts.experiments.Scenario(
                metadata.prediction_length,
                conts=lag_conts + ["month", "unix_timestamp"],
                seed=seed,
            )
        )

    params.training_method = nnts.models.hyperparams.TrainingMethod.TEACHER_FORCING
    lag_seq = [lag - 1 for lag in lag_seq if lag > 1]

    for scenario in scenario_list[:1]:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        df = df_orig.copy()
        split_data = splitter.split_test_train(df, metadata)
        trn_dl, test_dl = nnts.data.train_test_to_dataloaders(
            split_data,
            metadata,
            scenario,
            params,
            nnts.torch.data.TorchTimeseriesDataLoaderFactory(),
        )
        net = nnts.torch.models.UnrolledFutureCovariateLSTM(
            nnts.torch.models.LinearModel,
            params,
            nnts.torch.data.preprocessing.masked_mean_abs_scaling,
            1,
            lag_seq=lag_seq,
        )
        trner = trainers.TorchEpochTrainer(
            nnts.models.TrainerState(),
            net,
            params,
            metadata,
            os.path.join(PATH, f"{scenario.name}.pt"),
        )

        evaluator = trner.train(trn_dl)
        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )
        test_metrics = nnts.metrics.calc_metrics(y, y_hat, trn_dl, metadata)
        print(test_metrics)


def prepare_lags(data, lag_seq):
    data = data.copy()
    conts = []
    for lag in range(1, max(lag_seq)):
        data[f"y_lag_{lag}"] = data[["y", "unique_id"]].groupby("unique_id").shift(lag)
        conts.append(f"y_lag_{lag}")
    return data, conts


def add_y_hat(df, y_hat, prediction_length):
    i = 0
    df_list = []
    for name, group in df.groupby("unique_id", sort=False):
        group["y_hat"] = None
        group["y_hat"][-prediction_length:] = y_hat[i].squeeze()
        i += 1
        df_list.append(group)
    return df_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training and evaluation script."
    )
    parser.add_argument(
        "--data-path", type=str, default="data", help="Path to the data directory."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unrolled-future-covariate-lstm",
        help="Name of the model.",
    )
    parser.add_argument(
        "--base-model-name", type=str, default="base-lstm", help="Base model name."
    )
    parser.add_argument(
        "--dataset-name", type=str, default="tourism", help="Name of the dataset."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="ablation-results",
        help="Path to the results directory.",
    )
    args = parser.parse_args()

    main(
        args.data_path,
        args.model_name,
        args.base_model_name,
        args.dataset_name,
        args.results_path,
    )
