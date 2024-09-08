import argparse
import os

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
from nnts.utils import Scheduler

DATASET_NAMES = [
    # "car_parts",
    # "covid_deaths",
    # "electricity_weekly",
    # "fred_md",
    # "hospital",
    # "nn5_daily",
    # "nn5_weekly",
    # "rideshare",
    # "saugeen_river_flow",
    # "solar_weekly",  # fails patchtst
    # "sunspot",
    # "traffic_weekly",
    # "us_births",
    "traffic_hourly",  # fails too big
    "electricity_hourly",  # fails too big
    "solar_10_minutes",  # fails too big
    # "kaggle_web_traffic",  # fails too big
    # "temperature_rain",  # fails patchtst
]


def model_factory(
    model_name: str, metadata: nnts.datasets.Metadata, params, enc_in, **kwargs
):
    if model_name == "dlinear":
        return nnts.torch.models.DLinear(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            c_in=enc_in,
            configs=params,
            **kwargs,
        )
    elif model_name == "nlinear":
        return nnts.torch.models.NLinear(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            c_in=enc_in,
            configs=params,
            **kwargs,
        )
    elif model_name == "patchtst":
        return nnts.torch.models.PatchTST(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            c_in=enc_in,
            configs=params,
        )
    elif model_name == "autoformer":
        return nnts.torch.models.Autoformer(
            h=metadata.prediction_length,
            input_size=metadata.context_length,
            c_in=enc_in,
            configs=params,
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
    elif model_name == "patchtst":
        return nnts.torch.models.patchtst.Hyperparams()
    elif model_name == "autoformer":
        return nnts.torch.models.autoformer.Hyperparams()


import torch
import torch.nn.functional as F


def benchmark_dataset(model_name: str, dataset_name: str, results_path: str, seed=42):
    CONTEXT_LENGTH_ITEM = 2
    df, metadata = nnts.datasets.load_dataset(dataset_name)
    metadata.context_length = metadata.context_lengths[CONTEXT_LENGTH_ITEM]

    params = get_hyperparams(model_name)

    # Baseline multi-variate params for all models in the benchmark
    params.scheduler = Scheduler.REDUCE_LR_ON_PLATEAU
    params.individual = False
    params.optimizer = torch.optim.Adam
    params.loss_fn = F.l1_loss
    params.batch_size = 32
    params.epochs = 100
    params.patience = 10
    params.early_stopper_patience = 30
    params.batches_per_epoch = 50

    dataset_options = {
        "context_length": metadata.context_length,
        "prediction_length": metadata.prediction_length,
        "conts": [],
    }

    nnts.torch.utils.seed_everything(seed)
    logger = nnts.loggers.WandbRun(
        project=f"{model_name}-multivariate-{CONTEXT_LENGTH_ITEM}",
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
        Dataset=nnts.torch.datasets.MultivariateTimeSeriesDataset,
        dataset_options=dataset_options,
    )

    net = model_factory(
        model_name,
        metadata,
        params=params,
        enc_in=trn_dl.dataset[0].data.shape[1],
    )
    param_count = nnts.torch.utils.count_of_params_in(net)
    print(f"Number of parameters: {param_count}")
    logger.log({"param_count": param_count})

    trner = nnts.torch.trainers.TorchEpochTrainer(
        net, params, metadata, model_path="multivariate.pt"
    )
    trner.events.add_listener(nnts.trainers.EpochTrainComplete, logger)
    trner.events.add_listener(nnts.trainers.EpochValidateComplete, logger)
    evaluator = trner.train(trn_dl)
    y_hat, y = evaluator.evaluate(
        test_dl, metadata.prediction_length, metadata.context_length
    )
    y_hat = y_hat.permute(2, 1, 0)  # .reshape(7, -1)
    y = y.permute(2, 1, 0)  # .reshape(7, -1)
    test_metrics = nnts.metrics.calc_metrics(
        y_hat,
        y,
        seasonal_error=nnts.metrics.calculate_seasonal_error(
            trn_dl, metadata.seasonality
        ),
    )

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
        for model_name in ["patchtst"]:
            for seed in [42, 43, 44, 45, 46]:
                for dataset_name in DATASET_NAMES:
                    benchmark_dataset(
                        model_name,
                        dataset_name,
                        args.results_path,
                        seed=seed,
                    )
    else:
        for seed in [42, 43, 44, 45, 46]:
            benchmark_dataset(
                args.model,
                args.dataset,
                args.results_path,
                seed=seed,
            )
