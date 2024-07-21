"""
This script runs the ablation study for the DeepAR model. 
It trains and evaluates the model on different scenarios and logs the results. 
"""

import argparse
import os
from typing import Iterable, List

import features
import torch
import torch.distributions as td
import torch.nn.functional as F
import trainers as project_trainers

import nnts
import nnts.data
import nnts.datasets
import nnts.loggers
import nnts.metrics
import nnts.torch.datasets
import nnts.torch.models
import nnts.torch.preprocessing
import nnts.torch.utils
import nnts.trainers
from nnts import datasets, utils


def calculate_seasonal_error(trn_dl: Iterable, metadata: datasets.Metadata):
    se_list = []
    for batch in trn_dl:
        past_data = batch["target"]
        se = nnts.metrics.gluon_metrics.calculate_seasonal_error(
            past_data, metadata.freq, metadata.seasonality
        )
        se_list.append(se)
    return torch.tensor(se_list).unsqueeze(1)


def generate_one_hot_matrix(n):
    # Total number of rows in the matrix
    num_rows = 2**n

    # Initialize the matrix
    one_hot_matrix = []

    # Generate each combination of binary values
    for i in range(num_rows):
        # Convert the number to its binary representation and fill with leading zeros
        binary_representation = format(i, f"0{n}b")
        # Convert the binary string to a list of integers
        one_hot_row = [int(bit) for bit in binary_representation]
        # Append the one-hot row to the matrix
        one_hot_matrix.append(one_hot_row)

    return one_hot_matrix


def create_scenarios(metadata: datasets.Metadata, lag_seq):
    scaled_covariates = [
        "month",
        "unix_timestamp",
        "unique_id_0",
        "static_cont",
        nnts.torch.models.deepar.FEAT_SCALE,
    ]

    # Example usage for n=5
    n = len(scaled_covariates)
    scaled_covariate_selection_matrix = generate_one_hot_matrix(n)
    scenario_list: List[features.LagScenario] = []

    for seed in [42, 43, 44, 45, 46]:
        for row in scaled_covariate_selection_matrix:
            selected_combination = [
                covariate
                for covariate, select in zip(scaled_covariates, row)
                if select == 1
            ]
            scenario_list.append(
                features.LagScenario(
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


def create_lag_scenarios(metadata: datasets.Metadata, lag_seq: List[int]):
    """
    Creates scenarios that vary the number of lags in the model.
    """
    conts = [
        "day_of_week",
        "hour",
        "dayofyear",
        "dayofmonth",
        "unix_timestamp",
        "unique_id_0",
        "static_cont",
    ]

    scenario_list: List[features.LagScenario] = []

    # BASELINE
    scenario_list = []
    i = 1
    while i < len(lag_seq):
        for seed in [42, 43, 44, 45, 46]:
            scenario = features.LagScenario(
                metadata.prediction_length,
                conts=conts,
                scaled_covariates=conts
                + [
                    nnts.torch.models.deepar.FEAT_SCALE,
                ],
                lag_seq=lag_seq[:i],
                seed=seed,
                dataset=metadata.dataset,
            )
            scenario_list.append(scenario)
        i += 1
    return scenario_list


def distr_nll(distr: td.Distribution, target: torch.Tensor) -> torch.Tensor:
    nll = -distr.log_prob(target)
    # nll = nll.mean(dim=(1,))
    return nll.mean()


def main(
    model_name: str,
    dataset_name: str,
    data_path: str,
    base_model_name: str,
    results_path: str,
):
    # Set up paths and load metadata

    df_orig, metadata = nnts.datasets.load_dataset(dataset_name)

    # Set parameters
    params = utils.GluonTsDefaultWithOneCycle(
        optimizer=torch.optim.Adam,
        loss_fn=distr_nll,
        scheduler=utils.Scheduler.REDUCE_LR_ON_PLATEAU,
        training_method=utils.TrainingMethod.TEACHER_FORCING,
        batches_per_epoch=100,
    )

    PATH = os.path.join(results_path, model_name, metadata.dataset)
    utils.makedirs_if_not_exists(PATH)

    # Calculate next month and unix timestamp
    df_orig = features.create_time_features(df_orig)
    df_orig = features.create_dummy_unique_ids(df_orig)
    lag_seq = features.create_lag_seq(metadata.freq)
    scenario_list = create_scenarios(metadata, lag_seq)
    scenario_list += create_lag_scenarios(metadata, lag_seq)

    for scenario in scenario_list:
        nnts.torch.utils.seed_everything(scenario.seed)
        df = df_orig.copy()
        lag_processor = nnts.torch.preprocessing.LagProcessor(scenario.lag_seq)

        context_length = metadata.context_length + max(scenario.lag_seq)

        dataset_options = {
            "context_length": metadata.context_length,
            "prediction_length": metadata.prediction_length,
            "conts": scenario.conts,
            "lag_seq": scenario.lag_seq,
        }
        trn_dl, test_dl = nnts.torch.utils.create_dataloaders(
            df,
            datasets.split_test_train_last_horizon,
            context_length,
            metadata.prediction_length,
            Dataset=nnts.torch.datasets.TimeseriesDataset,
            dataset_options=dataset_options,
            Sampler=nnts.torch.datasets.TimeSeriesSampler,
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
        trner = create_trainer(
            model_name, metadata, PATH, params, scenario, lag_processor
        )
        logger.configure(trner.events)

        evaluator = trner.train(trn_dl)

        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )

        test_metrics = nnts.metrics.calc_metrics(
            y_hat,
            y,
            nnts.metrics.calculate_seasonal_error(trn_dl, metadata.seasonality),
        )
        logger.log(test_metrics)
        print(test_metrics)
        logger.finish()

    csv_aggregator = utils.CSVFileAggregator(PATH, "results")
    results = csv_aggregator()
    univariate_results = results.loc[:, ["smape", "mase"]]
    print(
        univariate_results.mean(), univariate_results.std(), univariate_results.count()
    )


def create_trainer(model_name, metadata, PATH, params, scenario, lag_processor):
    if model_name == "deepar-studentt":
        net = nnts.torch.models.DistrDeepAR(
            nnts.torch.models.deepar.StudentTHead,
            params,
            nnts.torch.preprocessing.masked_mean_abs_scaling,
            1,
            lag_processor=lag_processor,
            scaled_features=scenario.scaled_covariates,
            context_length=metadata.context_length,
            cat_idx=scenario.cat_idx,
        )
        trner = project_trainers.TorchEpochTrainer(
            nnts.trainers.TrainerState(),
            net,
            params,
            metadata,
            os.path.join(PATH, f"{scenario.name}.pt"),
            loss_fn=distr_nll,
        )
    elif model_name == "deepar-point":
        net = nnts.torch.models.DeepARPoint(
            nnts.torch.models.LinearModel,
            params,
            nnts.torch.preprocessing.masked_mean_abs_scaling,
            1,
            lag_processor=lag_processor,
            scaled_features=scenario.scaled_covariates,
            context_length=metadata.context_length,
            cat_idx=scenario.cat_idx,
        )
        trner = project_trainers.TorchEpochTrainer(
            nnts.trainers.TrainerState(),
            net,
            params,
            metadata,
            os.path.join(PATH, f"{scenario.name}.pt"),
            F.l1_loss,
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return trner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training and evaluation script."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepar-studentt",
        help="Name of the model.",
    )
    parser.add_argument(
        "--dataset", type=str, default="tourism_monthly", help="Name of the dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="projects/deepar/data",
        help="Path to the data directory.",
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="projects/deepar/ablation-results",
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
