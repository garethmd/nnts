from typing import List

import covs
import pandas as pd
import torch

import nnts.data.metadata
import nnts.experiments
import nnts.models
import nnts.torch.data.datasets
import nnts.torch.data.preprocessing as preprocessing
import nnts.torch.models
import nnts.torch.models.trainers


def save_results(y_hat, y, path, name):
    torch.save(y_hat, f"{path}/{name}_y_hat.pt")
    torch.save(y, f"{path}/{name}_y.pt")


def save_metrics(metrics, path, name):
    torch.save(metrics, f"{path}/{name}_metrics.pt")


def calculate_forecast_horizon_metrics(y_hat, y, metadata, metric="mae"):
    forecast_horizon_metrics = []
    for i in range(0, metadata.prediction_length + 1):
        metrics = nnts.metrics.calc_metrics(
            y[:, :i, :], y_hat[:, :i, :], metadata.freq, metadata.seasonality
        )
        forecast_horizon_metrics.append(metrics[metric])
    return forecast_horizon_metrics


def generate(
    scenario_list: List[nnts.experiments.CovariateScenario],
    df_orig: pd.DataFrame,
    metadata: nnts.data.metadata.Metadata,
    params: nnts.models.Hyperparams,
    model_name: str,
):
    for scenario in scenario_list:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        df, scenario = covs.prepare(df_orig.copy(), scenario)
        splitter = nnts.data.PandasSplitter()
        split_data = splitter.split(df, metadata)
        _, _, test_dl = nnts.data.map_to_dataloaders(
            split_data,
            metadata,
            scenario,
            params,
            nnts.torch.data.TorchTimeseriesDataLoaderFactory(),
        )
        path = f"results/{model_name}/{metadata.dataset}"

        if model_name == "seg-lstm":
            net = nnts.torch.models.SegLSTM(
                nnts.torch.models.LinearModel,
                params,
                preprocessing.masked_mean_abs_scaling,
                scenario.covariates + 1,
                metadata.seasonality,
            )
        else:
            net = nnts.torch.models.BaseLSTM(
                nnts.torch.models.LinearModel,
                params,
                preprocessing.masked_mean_abs_scaling,
                scenario.covariates + 1,
            )

        best_state_dict = torch.load(f"{path}/{scenario.name}.pt")
        net.load_state_dict(best_state_dict)
        evaluator = nnts.torch.models.trainers.TorchEvaluator(net)
        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )
        test_metrics = nnts.metrics.calc_metrics(
            y, y_hat, metadata.freq, metadata.seasonality
        )
        save_results(y_hat, y, path, scenario.name)
        forecast_horizon_metrics = calculate_forecast_horizon_metrics(
            y_hat, y, metadata, "smape"
        )
        save_metrics(forecast_horizon_metrics, path, scenario.name)
