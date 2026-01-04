from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Type

import pandas as pd
import torch
from torch.utils.data import DataLoader

import nnts.preprocessing
import nnts.utils


def count_of_params_in(net: torch.nn.Module):
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return pytorch_total_params


class DataLoaderBuilder:

    def __init__(self, data):
        self.data = data
        self.transforms = None
        self.Dataset = None
        self.dataset_options = {}
        self.DataLoader = None
        self.dataloader_options = {}
        self.Sampler = None

    def set_transforms(
        self, transforms: List[nnts.preprocessing.Transformation]
    ) -> "DataLoaderBuilder":
        self.transforms = transforms
        return self

    def set_dataset(self, Dataset: Type, dataset_options) -> "DataLoaderBuilder":
        self.Dataset = Dataset
        self.dataset_options = dataset_options
        return self

    def set_dataloader(
        self, DataLoader: Type, **dataloader_options
    ) -> "DataLoaderBuilder":
        self.DataLoader = DataLoader
        self.dataloader_options = dataloader_options
        return self

    def set_Sampler(self, Sampler: Type) -> "DataLoaderBuilder":
        self.Sampler = Sampler
        return self

    def to_dataloader(self) -> Iterable:
        if self.transforms is not None:
            for transform in self.transforms:
                self.data = transform.transform(self.data)

        ts = self.Dataset(self.data, **self.dataset_options).build()

        if self.Sampler is not None:
            sampler = self.Sampler(ts)
            self.dataloader_options["sampler"] = sampler
            self.dataloader_options.pop("shuffle", None)

        dl = self.DataLoader(ts, **self.dataloader_options)
        return dl


def create_dataloaders(
    df: pd.DataFrame,
    splitter_fn: Callable,
    context_length: int,
    prediction_length: int,
    Dataset: Type,
    dataset_options: dict = {},
    Sampler: Type | None = None,
    batch_size: int = 32,
    transforms: List[nnts.preprocessing.Transformation] = None,
) -> Tuple[Iterable, Iterable, Iterable]:
    """takes split data -> fits transforms -> creates dataloaders"""
    split_data = splitter_fn(df, context_length, prediction_length)
    return create_dataloaders_from_split_data(
        split_data,
        Dataset,
        dataset_options,
        Sampler,
        batch_size,
        transforms,
    )


def create_dataloaders_from_split_data(
    split_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Dataset: Type,
    dataset_options: dict = {},
    Sampler: Type | None = None,
    batch_size: int = 32,
    transforms: List[nnts.preprocessing.Transformation] = None,
) -> Tuple[Iterable, Iterable, Iterable]:
    """takes split data -> fits transforms -> creates dataloaders"""

    has_validation = hasattr(split_data, "validation")

    trn_builder = DataLoaderBuilder(split_data.train)
    test_builder = DataLoaderBuilder(split_data.test)
    if has_validation:
        val_builder = DataLoaderBuilder(split_data.validation)

    if transforms is not None:
        transforms = [
            transform.fit(
                split_data.train,
            )
            for transform in transforms
        ]
        trn_builder.set_transforms(transforms)
        test_builder.set_transforms(transforms)
        if has_validation:
            val_builder.set_transforms(transforms)

    if Sampler is not None:
        trn_builder.set_Sampler(Sampler)
        # test_builder.set_Sampler(Sampler)
        if has_validation:
            val_builder.set_Sampler(Sampler)

    trn_dl = (
        trn_builder.set_dataset(
            Dataset,
            dataset_options=dataset_options,
        )
        .set_dataloader(DataLoader, batch_size=batch_size, shuffle=True, drop_last=True)
        .to_dataloader()
    )
    test_dl = (
        test_builder.set_dataset(
            Dataset,
            dataset_options=dataset_options,
        )
        .set_dataloader(DataLoader, batch_size=batch_size, shuffle=False)
        .to_dataloader()
    )
    if has_validation:
        val_dl = (
            val_builder.set_dataset(
                Dataset,
                dataset_options=dataset_options,
            )
            .set_dataloader(
                DataLoader, batch_size=batch_size, shuffle=False, drop_last=False
            )
            .to_dataloader()
        )
        return trn_dl, val_dl, test_dl
    return trn_dl, test_dl


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
