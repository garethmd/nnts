import argparse
import os
from typing import Iterable, List

import features
import gluonadapt
import torch
import torch.distributions as td
import torch.nn.functional as F
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
import nnts.torch.utils


def calculate_seasonal_error(trn_dl: Iterable, metadata: nnts.data.metadata.Metadata):
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


def create_scenarios(metadata, lag_seq):
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


def create_lag_scenarios(metadata, lag_seq):
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
    for seed in [42, 43, 44, 45, 46]:
        scenario = features.LagScenario(
            metadata.prediction_length,
            conts=conts,
            scaled_covariates=conts
            + [
                nnts.torch.models.deepar.FEAT_SCALE,
            ],
            lag_seq=lag_seq,
            seed=seed,
            dataset=metadata.dataset,
        )
        scenario_list.append(scenario)
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
    metadata = nnts.data.metadata.load(
        dataset_name, path=os.path.join(data_path, f"{base_model_name}-monash.json")
    )
    PATH = os.path.join(results_path, model_name, metadata.dataset)
    nnts.loggers.makedirs_if_not_exists(PATH)

    # Load data
    df_orig, *_ = nnts.pandas.read_tsf(os.path.join(data_path, metadata.filename))

    # Set parameters
    params = nnts.models.Hyperparams()
    params.batch_size = 32
    params.batches_per_epoch = 100

    # Calculate next month and unix timestamp
    df_orig = features.create_time_features(df_orig)
    df_orig = features.create_dummy_unique_ids(df_orig)

    # Normalize data
    # max_min_scaler = nnts.torch.data.preprocessing.MaxMinScaler()
    # max_min_scaler.fit(df_orig, ["month"])
    # df_orig = max_min_scaler.transform(df_orig, ["month"])

    lag_seq = features.create_lag_seq(metadata.freq)
    scenario_list = create_lag_scenarios(metadata, lag_seq)

    params.training_method = nnts.models.hyperparams.TrainingMethod.TEACHER_FORCING
    params.scheduler = nnts.models.hyperparams.Scheduler.REDUCE_LR_ON_PLATEAU

    for scenario in scenario_list[:1]:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        df = df_orig.copy()
        lag_processor = nnts.torch.data.preprocessing.LagProcessor(scenario.lag_seq)

        context_length = metadata.context_length + max(scenario.lag_seq)
        split_data = nnts.pandas.split_test_train_last_horizon(
            df, context_length, metadata.prediction_length
        )
        trn_dl, test_dl = nnts.data.create_trn_test_dataloaders(
            split_data,
            metadata,
            scenario,
            params,
            nnts.torch.data.datasets.TorchTimeseriesLagsDataLoaderFactory(),
            Sampler=nnts.torch.data.datasets.TimeSeriesSampler,
        )
        # trn_dl = gluonadapt.trn_dl
        # test_dl = gluonadapt.test_dl
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

        # net = gluonadapt.load_gluonts_weights(net)
        # evaluator = nnts.torch.models.trainers.TorchEvaluator(net)

        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )

        test_metrics = nnts.metrics.calc_metrics(
            y_hat, y, nnts.metrics.calculate_seasonal_error(trn_dl, metadata)
        )
        logger.log(test_metrics)
        print(test_metrics)
        logger.finish()

    csv_aggregator = nnts.pandas.CSVFileAggregator(PATH, "results")
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
            nnts.torch.data.preprocessing.masked_mean_abs_scaling,
            1,
            lag_processor=lag_processor,
            scaled_features=scenario.scaled_covariates,
            context_length=metadata.context_length,
            cat_idx=scenario.cat_idx,
        )
        trner = trainers.TorchEpochTrainer(
            nnts.models.TrainerState(),
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
            nnts.torch.data.preprocessing.masked_mean_abs_scaling,
            1,
            lag_processor=lag_processor,
            scaled_features=scenario.scaled_covariates,
            context_length=metadata.context_length,
            cat_idx=scenario.cat_idx,
        )
        trner = trainers.TorchEpochTrainer(
            nnts.models.TrainerState(),
            net,
            params,
            metadata,
            os.path.join(PATH, f"{scenario.name}.pt"),
            F.l1_loss,
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return trner


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
        default="deepar-studentt",
        help="Name of the model.",
    )
    parser.add_argument(
        "--dataset", type=str, default="traffic_hourly", help="Name of the dataset."
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
