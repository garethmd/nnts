import argparse
import importlib
import os

import pandas as pd
import torch

import nnts
import nnts.data
import nnts.loggers
import nnts.metrics
import nnts.torch.datasets
import nnts.torch.trainers
import nnts.torch.utils
from nnts import datasets, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a time-series model."
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Path to the dataset"
    )
    parser.add_argument(
        "--model_module",
        type=str,
        required=True,
        help="Module where the model is located",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model class name"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="ETTh1", help="Dataset name"
    )
    parser.add_argument(
        "--results_path", type=str, default="nb-results", help="Path to save results"
    )
    parser.add_argument(
        "--metadata_path", type=str, default="informer.json", help="Metadata file path"
    )
    return parser.parse_args()


def load_data(args):
    metadata = datasets.load_metadata(args.dataset_name, path=args.metadata_path)
    datafile_path = os.path.join(args.data_path, metadata.filename)
    results_dir = os.path.join(args.results_path, args.model_name, metadata.dataset)
    utils.makedirs_if_not_exists(results_dir)
    df = pd.read_csv(datafile_path)
    return df, metadata, results_dir


def preprocess_data(df):
    df = df.rename({"date": "ds", "OT": "y"}, axis="columns")
    df["unique_id"] = "T1"
    return df


def split_dataset(df, metadata):
    torch.manual_seed(2021)
    trn_length = 8640
    val_test_length = 2880
    return datasets.split_test_val_train(
        df, trn_length, val_test_length, val_test_length
    )


def create_dataloaders(split_data, metadata, params):
    dataset_options = {
        "context_length": metadata.context_length,
        "prediction_length": metadata.prediction_length,
        "conts": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
    }
    return nnts.torch.utils.create_dataloaders_from_split_data(
        split_data,
        Dataset=nnts.torch.datasets.MultivariateTimeSeriesDataset,
        dataset_options=dataset_options,
        batch_size=params.batch_size,
        transforms=[nnts.torch.preprocessing.StandardScaler()],
    )


def main():
    args = parse_args()
    df, metadata, results_dir = load_data(args)
    df = preprocess_data(df)
    split_data = split_dataset(df, metadata)

    # Dynamically import the model module and retrieve the model class
    model_module = importlib.import_module(args.model_module)
    model_class = getattr(model_module, args.model_name)

    params_fn = getattr(model_module, "get_mutlivariate_params")
    params = params_fn()
    params.enc_in = 1
    params.individual = False

    trn_dl, val_dl, test_dl = create_dataloaders(split_data, metadata, params)
    model = model_class(metadata.context_length, metadata.prediction_length, 7, params)

    trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(model, params, metadata)
    evaluator = trainer.train(trn_dl, val_dl)
    y_hat, y = evaluator.evaluate(
        test_dl, metadata.prediction_length, metadata.context_length
    )

    test_metrics = nnts.metrics.calc_metrics(
        y_hat, y, nnts.metrics.calculate_seasonal_error(trn_dl, metadata.seasonality)
    )
    print("Test Metrics:", test_metrics)

    torch.save(y_hat, os.path.join(results_dir, "y_hat.pt"))
    torch.save(y, os.path.join(results_dir, "y.pt"))


if __name__ == "__main__":
    main()
