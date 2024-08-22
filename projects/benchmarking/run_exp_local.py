import argparse
import os

import torch.nn.functional as F
import torch.optim

import nnts
import nnts.data
import nnts.datasets
import nnts.loggers
import nnts.metrics
import nnts.torch
import nnts.torch.datasets
import nnts.torch.models
import nnts.torch.preprocessing
import nnts.torch.trainers
import nnts.torch.utils
import nnts.trainers
from nnts import utils

DATASET_NAMES = [
    # "bitcoin",
    "car_parts",
    "cif_2016",
    "covid_deaths",
    # "dominick",
    "electricity_hourly",
    "electricity_weekly",
    "fred_md",
    "hospital",
    # "kaggle_web_traffic",
    "kdd_cup",
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    # "m3_monthly",
    # "m3_quarterly",
    # "m3_yearly",
    # "m4_daily",
    # "m4_hourly",
    # "m4_monthly",
    # "m4_quarterly",
    # "m4_yearly",
    # "m4_weekly",
    "nn5_daily",
    "nn5_weekly",
    "pedestrian_counts",
    "rideshare",
    "saugeen_river_flow",
    "solar_10_minutes",
    "solar_weekly",
    "sunspot",
    # "temperature_rain",
    "tourism_monthly",
    "tourism_quarterly",
    "tourism_yearly",
    "traffic_hourly",
    "traffic_weekly",
    "us_births",
    "vehicle_trips",
    "weather",
    "australian_electricity_demand",
]


def model_factory(model_name: str, metadata: nnts.datasets.Metadata, **kwargs):
    if model_name == "dlinear":
        return nnts.torch.models.DLinear(metadata, **kwargs)
    elif model_name == "nlinear":
        return nnts.torch.models.NLinear(metadata, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def benchmark_dataset(model_name: str, dataset_name: str, results_path: str):
    df, metadata = nnts.datasets.load_dataset(dataset_name)
    unique_ids = df["unique_id"].unique()

    params = nnts.torch.models.dlinear.Hyperparams(
        optimizer=torch.optim.Adam,
        loss_fn=F.l1_loss,
        batch_size=32,
        batches_per_epoch=50,
        training_method=utils.TrainingMethod.DMS,
        model_file_path=f"logs",
        lr=0.005,
        weight_decay=0.0,
    )
    dataset_options = {
        "context_length": metadata.context_length,
        "prediction_length": metadata.prediction_length,
        "conts": [],
    }

    seed = 42
    nnts.torch.utils.seed_everything(seed)
    logger = nnts.loggers.WandbRun(
        project=model_name + "-local-separate",
        name=f"{dataset_name}-seed-{seed}",
        config={
            **params.__dict__,
            **metadata.__dict__,
        },
        path=os.path.join(results_path, model_name, metadata.dataset),
    )

    y_hat_list = []
    y_list = []
    seasonal_error_list = []

    for unique_id in unique_ids:
        df_local = df[df["unique_id"] == unique_id]

        trn_dl, test_dl = nnts.torch.utils.create_dataloaders(
            df_local,
            nnts.datasets.split_test_train_last_horizon,
            metadata.context_length,
            metadata.prediction_length,
            Dataset=nnts.torch.datasets.TimeseriesDataset,
            dataset_options=dataset_options,
            Sampler=nnts.torch.datasets.TimeSeriesSampler,
        )

        net = model_factory(
            model_name,
            metadata,
            enc_in=trn_dl.dataset[0].data.shape[1],
            individual=False,
        )

        trner = nnts.torch.trainers.TorchEpochTrainer(net, params, metadata)
        trner.events.add_listener(nnts.trainers.EpochTrainComplete, logger)
        trner.events.add_listener(nnts.trainers.EpochValidateComplete, logger)
        evaluator = trner.train(trn_dl)
        y_hat, y = evaluator.evaluate(
            test_dl, metadata.prediction_length, metadata.context_length
        )
        y_hat_list.append(y_hat)
        y_list.append(y)
        seasonal_error_list.append(
            nnts.metrics.calculate_seasonal_error(trn_dl, metadata.seasonality)
        )

    y_hat = torch.cat(y_hat_list, dim=0)
    y = torch.cat(y_list, dim=0)
    seasonal_error = torch.cat(seasonal_error_list, dim=0)

    test_metrics = nnts.metrics.calc_metrics(y_hat, y, seasonal_error=seasonal_error)

    logger.log(test_metrics)
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training and evaluation script."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nlinear",
        help="Name of the model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="projects/benchmarking/results",
        help="Path to the results directory.",
    )
    args = parser.parse_args()

    if args.dataset not in DATASET_NAMES and args.dataset != "all":
        raise ValueError(f"Unknown dataset name: {args.dataset}")

    if args.dataset == "all":
        for dataset_name in DATASET_NAMES:
            benchmark_dataset(
                args.model,
                dataset_name,
                args.results_path,
            )
    else:
        benchmark_dataset(
            args.model,
            args.dataset,
            args.results_path,
        )
