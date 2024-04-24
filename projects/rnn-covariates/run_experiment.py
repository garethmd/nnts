import argparse
from typing import List

import covs
import metric_generator
import pandas as pd

import nnts
import nnts.data
import nnts.experiments
import nnts.loggers
import nnts.metrics
import nnts.models
import nnts.pandas
import nnts.torch.data
import nnts.torch.data.datasets
import nnts.torch.models
import nnts.torch.models.trainers as trainers


def run_scenario(
    scenario: nnts.experiments.CovariateScenario,
    df_orig: pd.DataFrame,
    metadata: nnts.data.metadata.Metadata,
    params: nnts.models.Hyperparams,
    splitter: nnts.data.Splitter,
    model_name: str,
    path: str,
):
    nnts.torch.data.datasets.seed_everything(scenario.seed)
    df, scenario = covs.prepare(df_orig.copy(), scenario)
    split_data = splitter.split(df, metadata)
    trn_dl, val_dl, test_dl = nnts.data.map_to_dataloaders(
        split_data,
        metadata,
        scenario,
        params,
        nnts.torch.data.TorchTimeseriesDataLoaderFactory(),
    )
    logger = nnts.loggers.ProjectRun(
        nnts.loggers.JsonFileHandler(path=path, filename=f"{scenario.name}.json"),
        project=f"{model_name}-{metadata.dataset}",
        run=scenario.name,
        config={
            **params.__dict__,
            **metadata.__dict__,
            **scenario.__dict__,
        },
    )

    net = covs.model_factory(model_name, params, scenario, metadata)
    trner = trainers.TorchEpochTrainer(
        nnts.models.TrainerState(),
        net,
        params,
        metadata,
        f"{path}/{scenario.name}.pt",
        logger=logger,
    )
    evaluator = trner.train(trn_dl, val_dl)
    y_hat, y = evaluator.evaluate(
        test_dl, scenario.prediction_length, metadata.context_length
    )
    test_metrics = nnts.metrics.calc_metrics(
        y, y_hat, metadata.freq, metadata.seasonality
    )
    logger.log(test_metrics)
    logger.finish()


def run_experiment(
    dataset_name: str,
    data_path: str,
    model_name: str = "base-lstm",
    results_path: str = "script-results",
    generate_metrics: bool = False,
):
    metadata = nnts.data.metadata.load(dataset_name, path="monash.json")
    df_orig, *_ = nnts.pandas.read_tsf(data_path)

    params = nnts.models.Hyperparams()
    splitter = nnts.data.PandasSplitter()
    path = f"{results_path}/{model_name}/{metadata.dataset}"
    nnts.loggers.makedirs_if_not_exists(path)

    scenario_list: List[nnts.experiments.CovariateScenario] = []

    # Add the baseline scenarios
    for seed in [42, 43, 44, 45, 46]:
        scenario_list.append(
            nnts.experiments.CovariateScenario(
                metadata.prediction_length, error=0.0, covariates=0, seed=seed
            )
        )

    # Models for full forecast horizon with covariates
    for covariates in [1, 2, 3]:
        for error in covs.errors[metadata.dataset]:
            scenario_list.append(
                nnts.experiments.CovariateScenario(
                    metadata.prediction_length, error, covariates=covariates
                )
            )

    for scenario in scenario_list:
        run_scenario(scenario, df_orig, metadata, params, splitter, model_name, path)
    csv_aggregator = covs.CSVFileAggregator(path, "results")
    csv_aggregator()

    if generate_metrics:
        metric_generator.generate(
            scenario_list, df_orig, metadata, params, model_name, path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment with specified dataset name."
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("data_path", type=str, help="File path to the dataset")
    parser.add_argument("model_name", type=str, help="model name", default="base-lstm")
    parser.add_argument(
        "results_path", type=str, help="results path", default="script-results"
    )
    parser.add_argument(
        "generate_metrics",
        type=bool,
        help="flag to generate additional metrics from the test set",
        default=False,
        nargs="?",
    )

    args = parser.parse_args()
    run_experiment(
        args.dataset_name, args.data_path, args.model_name, args.results_path
    )
