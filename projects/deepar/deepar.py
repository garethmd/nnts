import argparse
import os
from dataclasses import dataclass, field
from typing import List

import gluonts
import numpy as np
import pandas as pd
import torch
import trainers

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


def create_time_features(df_orig: pd.DataFrame):
    df_orig["day_of_week"] = df_orig["ds"].dt.day_of_week
    df_orig["hour"] = df_orig["ds"].dt.hour
    df_orig["week"] = df_orig["ds"].dt.isocalendar().week
    df_orig["week"] = df_orig["week"].astype(np.float32)

    # GluonTS uses the following code to generate the age covariate
    # age = np.log10(2.0 + np.arange(length, dtype=self.dtype))
    # length = the length of the time series. In GluonTS this length depends on the length of the training set and test set.
    # but we do it once on the complete dataset.
    # Also note that this doesn't align to the most recent time point, but to the first time point which
    # intuitively doesn't make sense.
    df_orig["month"] = df_orig["ds"].dt.month

    df_orig["unix_timestamp"] = np.log10(2.0 + df_orig.groupby("unique_id").cumcount())
    return df_orig


@dataclass
class LagScenario(nnts.experiments.scenarios.BaseScenario):
    # covariates: int = field(init=False)
    dataset: str = ""
    lag_seq: List[int] = field(default_factory=list)
    scaled_covariates: List[str] = field(default_factory=list)

    def copy(self):
        return LagScenario(
            prediction_length=self.prediction_length,
            conts=self.conts.copy(),
            seed=self.seed,
            lag_seq=self.lag_seq.copy(),
            scaled_covariates=self.scaled_covariates.copy(),
        )

    def scaled_covariate_names(self):
        return "-".join(self.scaled_covariates)

    @property
    def name(self):
        return f"cov-{self.scaled_covariate_names()}-lags-{len(self.lag_seq)}-ds-{self.dataset}-seed-{self.seed}"


def create_scenarios(metadata, lag_seq):
    scaled_covariates = ["month", "unix_timestamp", nnts.torch.models.deepar.FEAT_SCALE]
    scaled_covariate_selection_matrix = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    scenario_list: List[LagScenario] = []

    for seed in [42, 43, 44, 45, 46]:
        for row in scaled_covariate_selection_matrix:
            selected_combination = [
                covariate
                for covariate, select in zip(scaled_covariates, row)
                if select == 1
            ]
            scenario_list.append(
                LagScenario(
                    metadata.prediction_length,
                    conts=[
                        cov
                        for cov in selected_combination
                        if cov != nnts.torch.models.deepar.FEAT_SCALE
                    ],
                    scaled_covariates=selected_combination,
                    lag_seq=lag_seq,
                    seed=seed,
                    dataset=metadata.dataset,
                )
            )
    return scenario_list


def create_lag_scenarios(metadata, lag_seq):
    scaled_covariates = ["month", "unix_timestamp", nnts.torch.models.deepar.FEAT_SCALE]

    scenario_list: List[LagScenario] = []

    for seed in [42, 43, 44, 45, 46]:
        for lags in range(1, len(lag_seq) + 1):
            scenario_list.append(
                LagScenario(
                    metadata.prediction_length,
                    conts=[
                        cov
                        for cov in scaled_covariates
                        if cov != nnts.torch.models.deepar.FEAT_SCALE
                    ],
                    scaled_covariates=scaled_covariates,
                    lag_seq=lag_seq[:lags],
                    seed=seed,
                    dataset=metadata.dataset,
                )
            )
    return scenario_list


def main(
    model_name: str,
    dataset_name: str,
    data_path: str,
    base_model_name: str,
    results_path: str,
):
    # Set up paths and load metadata

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
    df_orig = create_time_features(df_orig)

    # Normalize data
    max_min_scaler = nnts.torch.data.preprocessing.MaxMinScaler()
    max_min_scaler.fit(df_orig, ["month"])
    df_orig = max_min_scaler.transform(df_orig, ["month"])

    lag_seq = gluonts.time_feature.lag.get_lags_for_frequency(metadata.freq)
    lag_seq = [lag - 1 for lag in lag_seq if lag > 1]

    scenario_list = create_lag_scenarios(metadata, lag_seq)

    params.training_method = nnts.models.hyperparams.TrainingMethod.TEACHER_FORCING

    for scenario in scenario_list:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        df = df_orig.copy()
        context_length = metadata.context_length + max(scenario.lag_seq)
        split_data = nnts.pandas.split_test_train_last_horizon(
            df, context_length, metadata.prediction_length
        )
        trn_dl, test_dl = nnts.data.create_trn_test_dataloaders(
            split_data,
            metadata,
            scenario,
            params,
            nnts.torch.data.preprocessing.TorchTimeseriesLagsDataLoaderFactory(),
        )
        logger = nnts.loggers.LocalFileRun(
            project=f"{model_name}-{metadata.dataset}",
            name=scenario.name,
            config={
                **params.__dict__,
                **metadata.__dict__,
                **scenario.__dict__,
            },
            path=PATH,
        )
        net = nnts.torch.models.DeepAR(
            nnts.torch.models.LinearModel,
            params,
            nnts.torch.data.preprocessing.masked_mean_abs_scaling,
            1,
            lag_seq=scenario.lag_seq,
            scaled_features=scenario.scaled_covariates,
        )
        trner = trainers.TorchEpochTrainer(
            nnts.models.TrainerState(),
            net,
            params,
            metadata,
            os.path.join(PATH, f"{scenario.name}.pt"),
        )
        logger.configure(trner.events)

        evaluator = trner.train(trn_dl)
        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )
        test_metrics = nnts.metrics.calc_metrics(
            y, y_hat, nnts.metrics.calculate_seasonal_error(trn_dl, metadata)
        )
        print(test_metrics)
        logger.log(test_metrics)
        logger.finish()


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
        "--model",
        type=str,
        default="deepar",
        help="Name of the model.",
    )
    parser.add_argument(
        "--dataset", type=str, default="tourism", help="Name of the dataset."
    )
    parser.add_argument(
        "--data-path", type=str, default="data", help="Path to the data directory."
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="ablation-results",
        help="Path to the results directory.",
    )
    args = parser.parse_args()

    main(
        args.model,
        args.dataset,
        args.data_path,
        "base-lstm",
        args.results_path,
    )
