import argparse
import os
from typing import List

import covs
import metric_generator
import pandas as pd
import torch.nn.functional as F
import torch.optim
import trainers

import nnts
import nnts.data
import nnts.datasets
import nnts.loggers
import nnts.metrics
import nnts.torch.data
import nnts.torch.datasets
import nnts.torch.models
import nnts.torch.trainers
import nnts.torch.utils
import nnts.trainers
from nnts import datasets, utils


def run_scenario(
    scenario: covs.CovariateScenario,
    df_orig: pd.DataFrame,
    metadata: datasets.Metadata,
    params: utils.Hyperparams,
    model_name: str,
    path: str,
):
    nnts.torch.utils.seed_everything(scenario.seed)
    df, scenario = covs.prepare(df_orig.copy(), scenario.copy())
    trn_dl, val_dl, test_dl = nnts.torch.utils.create_dataloaders(
        df,
        nnts.datasets.split_test_val_train_last_horizon,
        metadata.context_length,
        metadata.prediction_length,
        Dataset=nnts.torch.datasets.TimeseriesDataset,
        dataset_options={
            "context_length": metadata.context_length,
            "prediction_length": metadata.prediction_length,
            "conts": scenario.conts,
        },
        batch_size=params.batch_size,
    )
    logger = nnts.loggers.LocalFileRun(
        project=f"{model_name}-{metadata.dataset}",
        name=scenario.name,
        config={
            **params.__dict__,
            **metadata.__dict__,
            **scenario.__dict__,
        },
        path=path,
    )

    net = covs.model_factory(model_name, params, scenario, metadata)
    trner = trainers.ValidationTorchEpochTrainer(
        trainers.TrainerState(),
        net,
        params,
        metadata,
        os.path.join(path, f"{scenario.name}.pt"),
    )
    logger.configure(trner.events)
    evaluator = trner.train(trn_dl, val_dl)
    y_hat, y = evaluator.evaluate(
        test_dl, scenario.prediction_length, metadata.context_length
    )
    test_metrics = nnts.metrics.calc_metrics(
        y, y_hat, nnts.metrics.calculate_seasonal_error(trn_dl, metadata.seasonality)
    )
    logger.log(test_metrics)
    logger.finish()


def run_experiment(
    model_names: List[str],
    dataset_names: List[str],
    data_path: str = "data",
    results_path: str = "script-results",
    generate_metrics: bool = False,
):
    if "all" in dataset_names:
        dataset_names = covs.list_available_datasets()

    if "all" in model_names:
        model_names = covs.list_available_models()

    for dataset_name in dataset_names:
        for model_name in model_names:
            metadata = datasets.load_metadata(
                dataset_name, path=os.path.join(data_path, f"{model_name}-monash.json")
            )
            df_orig, *_ = nnts.datasets.read_tsf(
                os.path.join(data_path, covs.file_map[dataset_name])
            )

            params = utils.Hyperparams(
                optimizer=torch.optim.AdamW, loss_fn=F.smooth_l1_loss
            )
            path = os.path.join(results_path, model_name, metadata.dataset)
            utils.makedirs_if_not_exists(path)

            scenario_list: List[covs.CovariateScenario] = []

            # Add the baseline scenarios
            for seed in [42, 43, 44, 45, 46]:
                scenario_list.append(
                    covs.CovariateScenario(
                        metadata.prediction_length, error=0.0, covariates=0, seed=seed
                    )
                )

            # Models for full forecast horizon with covariates
            for covariates in [1, 2, 3]:
                for error in covs.errors[metadata.dataset]:
                    scenario_list.append(
                        covs.CovariateScenario(
                            metadata.prediction_length, error, covariates=covariates
                        )
                    )

            scenario_list.append(
                covs.CovariateScenario(
                    metadata.prediction_length, 0, covariates=3, skip=1
                )
            )

            for scenario in scenario_list:
                run_scenario(
                    scenario,
                    df_orig,
                    metadata,
                    params,
                    model_name,
                    path,
                )
            csv_aggregator = utils.CSVFileAggregator(path, "results")
            csv_aggregator()

            if generate_metrics:
                metric_generator.generate(
                    scenario_list, df_orig, metadata, params, model_name, path
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment with specified dataset names and model names."
    )
    parser.add_argument(
        "model_names",
        nargs="+",
        type=str,
        help="Model names. Use 'all' to run for all models.",
        default=["base-lstm"],
    )
    parser.add_argument(
        "dataset_names",
        nargs="+",
        type=str,
        help="Names of the datasets. Use 'all' to run for all datasets.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="File path to the dataset (defaults: data)",
        default="data",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Results path (default: script-results)",
        default="script-results",
    )
    parser.add_argument(
        "--generate_metrics",
        type=bool,
        help="Flag to generate additional metrics from the test set",
        default=False,
    )

    args = parser.parse_args()
    run_experiment(
        args.model_names,
        args.dataset_names,
        args.data_path,
        args.results_path,
        args.generate_metrics,
    )
