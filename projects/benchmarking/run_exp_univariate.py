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
    # "car_parts",
    # "cif_2016",
    # "covid_deaths",
    # "dominick",
    # "electricity_hourly",
    # "electricity_weekly",
    # "fred_md",
    # "hospital",
    # "kaggle_web_traffic",
    # "kdd_cup",
    # "m1_monthly",
    # "m1_quarterly",
    # "m1_yearly",
    # "m3_monthly",
    # "m3_quarterly",
    # "m3_yearly",
    # "m4_daily",
    # "m4_hourly",
    # "m4_monthly",
    # "m4_quarterly",
    # "m4_weekly",
    # "m4_yearly",
    # "nn5_daily",
    # "nn5_weekly",
    # "pedestrian_counts",
    # "rideshare",
    # "saugeen_river_flow",
    "solar_10_minutes",
    "solar_weekly",
    "sunspot",
    "temperature_rain",
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


def model_factory(model_name: str, metadata: nnts.datasets.Metadata, enc_in, params):
    if model_name == "dlinear":
        return nnts.torch.models.DLinear(
            metadata.prediction_length, metadata.context_length, enc_in, params
        )
    elif model_name == "nlinear":
        return nnts.torch.models.NLinear(
            metadata.prediction_length, metadata.context_length, enc_in, params
        )
    elif model_name == "nhits":
        return nnts.torch.models.NHITS(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            n_blocks=[1, 1, 1],
            mlp_units=[[512, 512], [512, 512]],
            n_pool_kernel_size=[1, 1, 1],
            n_freq_downsample=[1, 1, 1],
        )
    elif model_name == "tide":
        return nnts.torch.models.TiDE(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_hyperparams(model_name: str):
    if model_name == "dlinear":
        return nnts.torch.models.dlinear.Hyperparams()
    elif model_name == "nlinear":
        return nnts.torch.models.nlinear.Hyperparams()
    elif model_name == "nhits":
        return nnts.torch.models.nhits.Hyperparams()
    elif model_name == "tide":
        return nnts.torch.models.tide.Hyperparams()


def benchmark_dataset(model_name: str, dataset_name: str, results_path: str, seed=42):
    df, metadata = nnts.datasets.load_dataset(dataset_name)
    metadata.context_length = metadata.prediction_length * 2

    params = get_hyperparams(model_name)
    dataset_options = {
        "context_length": metadata.context_length,
        "prediction_length": metadata.prediction_length,
        "conts": [],
    }

    nnts.torch.utils.seed_everything(seed)
    logger = nnts.loggers.WandbRun(
        project=model_name + "-global",
        name=f"{dataset_name}-seed-{seed}",
        config={
            **params.__dict__,
            **metadata.__dict__,
        },
        path=os.path.join(results_path, model_name, metadata.dataset),
    )

    trn_dl, test_dl = nnts.torch.utils.create_dataloaders(
        df,
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
        params=params,
    )

    trner = nnts.torch.trainers.TorchEpochTrainer(
        net, params, metadata, model_path="univariate.pth"
    )
    logger.configure(trner.events)
    evaluator = trner.train(trn_dl)
    y_hat, y = evaluator.evaluate(
        test_dl, metadata.prediction_length, metadata.context_length
    )
    # y_hat = y_hat.permute(2, 1, 0)  # .reshape(7, -1)
    # y = y.permute(2, 1, 0)  # .reshape(7, -1)
    test_metrics = nnts.metrics.calc_metrics(
        y_hat,
        y,
        nnts.metrics.calculate_seasonal_error(trn_dl, metadata.seasonality),
    )
    logger.log(test_metrics)
    param_count = nnts.torch.utils.count_of_params_in(net)
    logger.log({"param_count": param_count})
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
        for seed in [43]:
            for dataset_name in DATASET_NAMES:
                benchmark_dataset(
                    args.model,
                    dataset_name,
                    args.results_path,
                    seed=seed,
                )
    else:
        benchmark_dataset(
            args.model,
            args.dataset,
            args.results_path,
        )
