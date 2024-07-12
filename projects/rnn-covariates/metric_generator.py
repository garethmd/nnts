from typing import List

import covs
import metrics_old
import pandas as pd
import torch

import nnts.datasets
import nnts.torch.datasets
import nnts.torch.models
import nnts.torch.trainers
import nnts.torch.utils
from nnts import datasets, utils


def save_results(y_hat, y, path, name):
    torch.save(y_hat, f"{path}/{name}_y_hat.pt")
    torch.save(y, f"{path}/{name}_y.pt")


def save_metrics(metrics, path, name):
    torch.save(metrics, f"{path}/{name}_metrics.pt")


def calculate_forecast_horizon_metrics(y_hat, y, metadata, metric="mae"):
    forecast_horizon_metrics = []
    for i in range(0, metadata.prediction_length + 1):
        metrics = metrics_old.calc_metrics(
            y[:, :i, :], y_hat[:, :i, :], metadata.freq, metadata.seasonality
        )
        forecast_horizon_metrics.append(metrics[metric])
    return forecast_horizon_metrics


def generate(
    scenario_list: List[covs.CovariateScenario],
    df_orig: pd.DataFrame,
    metadata: datasets.Metadata,
    params: utils.Hyperparams,
    model_name: str,
    path: str,
):
    for scenario in scenario_list:
        assert isinstance(scenario, covs.CovariateScenario)
        assert isinstance(df_orig, pd.DataFrame)
        nnts.torch.utils.seed_everything(scenario.seed)
        df, scenario = covs.prepare(df_orig.copy(), scenario.copy())

        _, _, test_dl = nnts.torch.utils.create_dataloaders(
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

        net = covs.model_factory(model_name, params, scenario, metadata)
        best_state_dict = torch.load(f"{path}/{scenario.name}.pt")
        net.load_state_dict(best_state_dict)
        evaluator = nnts.torch.trainers.TorchEvaluator(net)
        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )
        test_metrics = metrics_old.calc_metrics(
            y, y_hat, metadata.freq, metadata.seasonality
        )
        save_results(y_hat, y, path, scenario.name)
        forecast_horizon_metrics = calculate_forecast_horizon_metrics(
            y_hat, y, metadata, "smape"
        )
        save_metrics(forecast_horizon_metrics, path, scenario.name)
