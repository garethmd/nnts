from typing import Iterator, List, Optional, Sized, Type

import pandas as pd
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Sampler

import nnts.data


def right_pad_sequence(
    seq: list[torch.Tensor], padding_value: int = 0, min_length: int = 0
):
    """Pads a list of 2D tensors to the right with a given padding value."""
    max_lengths = max([t.shape[0] for t in seq])
    padded_tensor = torch.zeros(len(seq), max_lengths, seq[0].shape[1]) + padding_value
    padded_mask = torch.zeros(len(seq), max_lengths).bool()
    padded_mask[:, :min_length] = True
    for i in range(len(seq)):
        start = min_length - seq[i].shape[0]
        start = max(0, start)
        padded_tensor[i, start : start + seq[i].shape[0], ...] = seq[i]
        padded_mask[i, start : start + seq[i].shape[0]] = True

    return padded_tensor, padded_mask


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


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        conts: List[int] = [],
    ):
        self.df = df.copy()
        self.conts = conts.copy()
        self.context_length = torch.tensor(context_length).long()
        self.prediction_length = torch.tensor(prediction_length).long()

        lengths = (
            self.df.groupby("unique_id", sort=False)["ds"].count()
            - context_length
            - prediction_length
            + 1  # we return x and y so add 1 for teacher forcing
        )
        cum_lengths = lengths.cumsum()
        self.shifted_cum_lengths = torch.tensor(
            cum_lengths.shift().fillna(0).astype(int).values
        )
        self.len = lengths.sum()

    def build(self):
        ts = []
        for unique_id, grp in self.df.groupby("unique_id", sort=False):
            ts.append(torch.from_numpy(grp[["y"] + self.conts].values))
        ts, mask = right_pad_sequence(ts)
        self.X = ts[:, :, :]
        self.pad_mask = mask
        return self

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        shifted_ln_lt_i = self.shifted_cum_lengths[self.shifted_cum_lengths <= i]
        result = torch.max(shifted_ln_lt_i, 0)
        locator = result.indices, i - result.values
        X = self.X[
            locator[0],
            locator[1] : locator[1] + self.context_length + self.prediction_length,
            ...,
        ]
        pad_mask = self.pad_mask[
            locator[0],
            locator[1] : locator[1] + self.context_length + self.prediction_length,
        ]
        return {
            "X": X,
            "pad_mask": pad_mask,
        }


class TimeseriesLagsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        conts: List[int] = [],
        lag_seq: List[int] = None,
    ):
        self.df = df.copy()
        self.conts = conts.copy()

        if lag_seq is None:
            raise ValueError("Lag sequence must be provided.")

        self.lag_length = max(lag_seq)
        self.context_length = torch.tensor(context_length).long()
        self.prediction_length = torch.tensor(prediction_length).long()

        lengths = (
            self.df.groupby("unique_id", sort=False)["ds"].count()
            - context_length
            - prediction_length
            - self.lag_length
            + 1  # we return x and y so add 1 for teacher forcing
        ).clip(1)
        cum_lengths = lengths.cumsum()
        self.shifted_cum_lengths = torch.tensor(
            cum_lengths.shift().fillna(0).astype(int).values
        )
        self.len = lengths.sum()
        self.min_length = context_length + prediction_length + self.lag_length
        self.lengths = lengths

    def build(self):
        ts = []
        for unique_id, grp in self.df.groupby("unique_id", sort=False):
            ts.append(torch.from_numpy(grp[["y"] + self.conts].values))
        ts, mask = right_pad_sequence(ts, min_length=self.min_length)
        self.X = ts[:, :, :]
        self.pad_mask = mask
        return self

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        shifted_ln_lt_i = self.shifted_cum_lengths[self.shifted_cum_lengths <= i]
        result = torch.max(shifted_ln_lt_i, 0)
        locator = result.indices, i - result.values
        X = self.X[
            locator[0],
            locator[1] : locator[1]
            + self.lag_length
            + self.context_length
            + self.prediction_length,
            ...,
        ]
        pad_mask = self.pad_mask[
            locator[0],
            locator[1] : locator[1]
            + self.lag_length
            + self.context_length
            + self.prediction_length,
        ]
        # pad_mask[: self.lag_length] = False
        return {
            "X": X,
            "pad_mask": pad_mask,
        }


class TimeSeriesSampler(Sampler[int]):

    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None):
        self.data_source = data_source
        self._num_samples = len(data_source) if num_samples is None else num_samples

        to = torch.cat(
            [
                data_source.shifted_cum_lengths,
                torch.tensor([len(data_source)]),
            ]
        )[1:]
        self.ranges = torch.stack([data_source.shifted_cum_lengths, to]).T

    def __iter__(self) -> Iterator[int]:
        count = 0
        while True:
            for i in range(self.ranges.shape[0]):
                count += 1
                if count > self._num_samples:
                    return
                start, end = self.ranges[i]
                val = torch.randint(start, end, size=(1,))
                yield val

    def __len__(self) -> int:
        return self._num_samples


class TorchTimeseriesDataLoaderFactory(nnts.data.DataLoaderFactory):

    def __call__(
        self,
        data: pd.DataFrame,
        metadata: nnts.data.Metadata,
        scenario: nnts.experiments.Scenario,
        params: nnts.models.Hyperparams,
        shuffle: bool,
        transforms: List[nnts.data.preprocessing.Transformation] = None,
        Sampler: Type = None,
    ) -> DataLoader:

        if transforms is not None:
            for transform in transforms:
                data = transform.transform(data)

        ts = TimeseriesDataset(
            data,
            conts=scenario.conts,
            context_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
        ).build()

        if Sampler is not None:
            result = DataLoader(ts, batch_size=params.batch_size, sampler=Sampler(ts))
        else:
            result = DataLoader(ts, batch_size=params.batch_size, shuffle=shuffle)
        return result


class TorchTimeseriesLagsDataLoaderFactory(nnts.data.DataLoaderFactory):
    def __call__(
        self,
        data: pd.DataFrame,
        metadata: nnts.data.Metadata,
        scenario: nnts.experiments.Scenario,
        params: nnts.models.Hyperparams,
        shuffle: bool,
        transforms: List[nnts.data.preprocessing.Transformation] = None,
        Sampler: Type = None,
    ) -> DataLoader:

        if transforms is not None:
            for transform in transforms:
                data = transform.transform(data)

        ts = TimeseriesLagsDataset(
            data,
            conts=scenario.conts,
            context_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
            lag_seq=scenario.lag_seq,
        ).build()

        if Sampler is not None:
            result = DataLoader(
                ts,
                batch_size=params.batch_size,
                sampler=Sampler(ts),
            )
        else:
            result = DataLoader(
                ts,
                batch_size=params.batch_size,
                shuffle=shuffle,
            )
        return result
