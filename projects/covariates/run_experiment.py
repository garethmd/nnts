import argparse
from typing import List

import covs
import matplotlib.pyplot as plt
import torch

import nnts
import nnts.data
import nnts.experiments
import nnts.loggers
import nnts.metrics
import nnts.models
import nnts.torch.data
import nnts.torch.data.datasets
import nnts.torch.data.preprocessing as preprocessing
import nnts.torch.models
import nnts.torch.models.trainers as trainers


def run_experiment(dataset_name: str):
    df_orig, metadata = nnts.data.load(dataset_name)
    params = nnts.models.Hyperparams()
    splitter = nnts.data.PandasSplitter()
    model_name = "base-lstm"
    PATH = f"results/{model_name}/{metadata.dataset}"

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

    # Models with short forecast horizon with covariates
    for covariates in [1, 2, 3]:
        for error in covs.errors[metadata.dataset]:
            scenario = nnts.experiments.CovariateScenario(
                covariates, error, covariates=covariates
            )
            scenario_list.append(scenario)

    for scenario in scenario_list:
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
        name = f"cov-{scenario.covariates}-pearsn-{str(round(scenario.pearson, 2))}-pl-{str(scenario.prediction_length)}-seed-{scenario.seed}"
        logger = nnts.loggers.ProjectRun(
            nnts.loggers.JsonFileHandler(path=PATH, filename=f"{name}.json"),
            project=f"{model_name}-{metadata.dataset}",
            run=name,
            config={
                **params.__dict__,
                **metadata.__dict__,
                **scenario.__dict__,
            },
        )

        net = nnts.torch.models.BaseLSTM(
            nnts.torch.models.LinearModel,
            params,
            preprocessing.masked_mean_abs_scaling,
            scenario.covariates + 1,
        )
        trner = trainers.TorchEpochTrainer(
            nnts.models.TrainerState(),
            net,
            params,
            metadata,
            f"{PATH}/{name}.pt",
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

    scenario_list = [
        nnts.experiments.CovariateScenario(
            1, 0, conts=[], pearson=1, noise=0, covariates=0
        ),
        nnts.experiments.CovariateScenario(
            2, 0, conts=[], pearson=1, noise=0, covariates=0
        ),
        nnts.experiments.CovariateScenario(
            3, 0, conts=[], pearson=1, noise=0, covariates=0
        ),
    ]

    for scenario in scenario_list:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        df, scenario = covs.prepare(df_orig.copy(), scenario)
        split_data = splitter.split(df, metadata)
        _, _, test_dl = nnts.data.map_to_dataloaders(
            split_data,
            metadata,
            scenario,
            params,
            nnts.torch.data.TorchTimeseriesDataLoaderFactory(),
        )
        name = f"cov-{scenario.covariates}-pearsn-{str(round(scenario.pearson, 2))}-pl-{str(scenario.prediction_length)}-seed-{scenario.seed}"
        logger = nnts.loggers.ProjectRun(
            nnts.loggers.JsonFileHandler(path=PATH, filename=f"{name}.json"),
            project=f"{model_name}-{metadata.dataset}",
            run=name,
            config={
                **params.__dict__,
                **metadata.__dict__,
                **scenario.__dict__,
            },
        )

        net = nnts.torch.models.BaseLSTM(
            nnts.torch.models.LinearModel,
            params,
            preprocessing.masked_mean_abs_scaling,
            1,
        )
        best_state_dict = torch.load(
            f"{PATH}/cov-{scenario.covariates}-pearsn-{str(round(scenario.pearson, 2))}-pl-{metadata.prediction_length}-seed-{scenario.seed}.pt"
        )
        net.load_state_dict(best_state_dict)
        evaluator = nnts.torch.models.trainers.TorchEvaluator(net)
        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )
        test_metrics = nnts.metrics.calc_metrics(
            y, y_hat, metadata.freq, metadata.seasonality
        )
        logger.log(test_metrics)
        logger.finish()

    csv_aggregator = covs.CSVFileAggregator(PATH, "results")
    results = csv_aggregator()

    for metric in ["smape", "mape", "rmse", "mae"]:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharey=True)
        covs.get_chart_data(results, 1, 1, metric).plot(
            kind="line",
            ax=axes[0, 0],
            title=f"{metadata.dataset} {metric} covariates = 1, forecast horizon = 1",
        )
        covs.get_chart_data(results, 2, 2, metric).plot(
            kind="line",
            ax=axes[0, 1],
            title=f"{metadata.dataset} {metric} covariates = 2, forecast horizon = 2",
        )
        covs.get_chart_data(results, 3, 3, metric).plot(
            kind="line",
            ax=axes[0, 2],
            title=f"{metadata.dataset} {metric} covariates = 3, forecast horizon = 3",
        )
        covs.get_chart_data(results, metadata.prediction_length, 1, metric).plot(
            kind="line",
            ax=axes[1, 0],
            title=f"{metadata.dataset} {metric} covariates = 1, forecast horizon = {metadata.prediction_length}",
        )
        covs.get_chart_data(results, metadata.prediction_length, 2, metric).plot(
            kind="line",
            ax=axes[1, 1],
            title=f"{metadata.dataset} {metric} covariates = 2, forecast horizon = {metadata.prediction_length}",
        )
        covs.get_chart_data(results, metadata.prediction_length, 3, metric).plot(
            kind="line",
            ax=axes[1, 2],
            title=f"{metadata.dataset} {metric} covariates = 3, forecast horizon = {metadata.prediction_length}",
        )
        fig.tight_layout()
        fig.savefig(f"{PATH}/{metric}.png")

    df_list = covs.add_y_hat(df, y_hat, scenario.prediction_length)
    sample_preds = covs.plot(df_list, scenario.prediction_length)

    results_mean = results.loc[
        (results["covariates"] == 0)
        & (results["prediction_length"] == metadata.prediction_length),
        ["smape", "mape", "rmse", "mae"],
    ].mean()
    print(results_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment with specified dataset name."
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    args = parser.parse_args()
    run_experiment(args.dataset_name)
