import json
import os
from typing import Iterable, Tuple

import pandas as pd

import nnts.experiments
import nnts.models

from . import metadata, splitter, tsf


def load(dataset: str) -> Tuple[pd.DataFrame, metadata.Metadata]:
    m = load_metadata(dataset)
    return load_data(m), m


def load_data(m: metadata.Metadata) -> Tuple[pd.DataFrame]:
    datai = tsf.convert_tsf_to_dataframe(m.path)
    df = pd.DataFrame(datai[0])
    df = pd.concat([tsf.unpack(df.iloc[x], freq=m.freq) for x in range(len(df))])
    return df


def load_metadata(
    dataset: str,
    repository: str = "monash",
    path: str = None,
) -> metadata.Metadata:
    # Get the directory of the current script
    if path is None:
        path = f"{repository}.json"
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Join the script directory with the relative path
        path = os.path.join(script_dir, path)

    with open(path) as f:
        data = json.load(f)
    return metadata.Metadata(**data[dataset])


class DataLoaderFactory:
    def __call__(
        self,
        data: pd.DataFrame,
        metadata: metadata.Metadata,
        scenario: nnts.experiments.CovariateScenario,
        params: nnts.models.Hyperparams,
        shuffle: bool,
    ) -> Iterable:
        raise NotImplementedError


def map_to_dataloaders(
    split_data: splitter.SplitData,
    metadata: metadata.Metadata,
    scenario: nnts.experiments.CovariateScenario,
    params: nnts.models.Hyperparams,
    dataloader_factory: DataLoaderFactory,
) -> Tuple[Iterable, Iterable, Iterable]:
    """Generate PyTorch dataloaders for training, validation, and testing.

    Args:
        split_data (nnts.data.SplitData):
        metadata (nnts.data.Metadata): metadata for the dataset
        scenario (nnts.experiments.CovariateScenario): scenario for the experiment
        params (nnts.models.Hyperparams): hyperparameters for the model
        dataloader_factory (callable, optional): function to create the dataloader. Defaults to pytorch_timeseries_dataloader_factory.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: training, validation, and testing dataloaders
    """
    trn_dl = dataloader_factory(split_data.train, metadata, scenario, params, True)
    val_dl = dataloader_factory(
        split_data.validation, metadata, scenario, params, False
    )
    test_dl = dataloader_factory(split_data.test, metadata, scenario, params, False)
    return trn_dl, val_dl, test_dl
