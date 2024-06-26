from typing import Any, Iterable, List, Tuple, Type

import nnts.experiments
import nnts.models

from . import metadata, preprocessing, splitter


class DataLoaderFactory:

    def __call__(
        self,
        data: Any,
        metadata: metadata.Metadata,
        scenario: nnts.experiments.Scenario,
        params: nnts.models.Hyperparams,
        shuffle: bool,
        transforms: List[preprocessing.Transformation] = None,
        Sampler: Type = None,
    ) -> Iterable:
        raise NotImplementedError


def create_trn_val_test_dataloaders(
    split_data: splitter.SplitData,
    metadata: metadata.Metadata,
    scenario: nnts.experiments.Scenario,
    params: nnts.models.Hyperparams,
    dataloader_factory: DataLoaderFactory,
    transforms: List[preprocessing.Transformation] = None,
    Sampler: Type = None,
) -> Tuple[Iterable, Iterable, Iterable]:
    """Generate Iterable dataloaders for training, validation, and testing.

    Args:
        split_data (nnts.data.SplitData):
        metadata (nnts.data.Metadata): metadata for the dataset
        scenario (nnts.experiments.Scenario): scenario for the experiment
        params (nnts.models.Hyperparams): hyperparameters for the model
        dataloader_factory (callable, optional): function to create the dataloader. Defaults to pytorch_timeseries_dataloader_factory.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: training, validation, and testing dataloaders
    """
    if transforms is not None:
        transforms = [
            transform.fit(
                split_data.train,
            )
            for transform in transforms
        ]

    trn_dl = dataloader_factory(
        split_data.train,
        metadata,
        scenario,
        params,
        True,
        transforms=transforms,
        Sampler=Sampler,
    )
    val_dl = dataloader_factory(
        split_data.validation,
        metadata,
        scenario,
        params,
        False,
        transforms=transforms,
        Sampler=Sampler,
    )
    test_dl = dataloader_factory(
        split_data.test,
        metadata,
        scenario,
        params,
        False,
        transforms=transforms,
        Sampler=Sampler,
    )
    return trn_dl, val_dl, test_dl


def create_trn_test_dataloaders(
    split_data: splitter.SplitTrainTest,
    metadata: metadata.Metadata,
    scenario: nnts.experiments.Scenario,
    params: nnts.models.Hyperparams,
    dataloader_factory: DataLoaderFactory,
    transforms: List[preprocessing.Transformation] = None,
    Sampler: Type = None,
) -> Tuple[Iterable, Iterable, Iterable]:
    """Generate Iterable dataloaders for training, and testing.

    Args:
        split_data (nnts.data.SplitTrainTest):
        metadata (nnts.data.Metadata): metadata for the dataset
        scenario (nnts.experiments.CovariateScenario): scenario for the experiment
        params (nnts.models.Hyperparams): hyperparameters for the model
        dataloader_factory (callable, optional): function to create the dataloader. Defaults to pytorch_timeseries_dataloader_factory.

    Returns:
        Tuple[DataLoader, DataLoader]: training, and testing dataloaders
    """
    if transforms is not None:
        transforms = [
            transform.fit(
                split_data.train,
            )
            for transform in transforms
        ]

    trn_dl = dataloader_factory(
        split_data.train,
        metadata,
        scenario,
        params,
        True,
        transforms=transforms,
        Sampler=Sampler,
    )
    test_dl = dataloader_factory(
        split_data.test,
        metadata,
        scenario,
        params,
        False,
        transforms=transforms,
        Sampler=Sampler,
    )
    return trn_dl, test_dl
