from collections import namedtuple
from typing import Iterator, List, Optional, Sized

import pandas as pd
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Sampler

PaddedData = namedtuple("PaddedData", ["data", "pad_mask"])


def right_pad_sequence(
    seq: list[torch.Tensor], padding_value: int = 0, min_length: int = 0
):
    """Pads a list of 2D tensors to the right with a given padding value."""
    max_lengths = max([t.shape[0] for t in seq])
    max_lengths = max(max_lengths, min_length)
    padded_tensor = torch.zeros(len(seq), max_lengths, seq[0].shape[1]) + padding_value
    padded_mask = torch.zeros(len(seq), max_lengths).bool()
    # padded_mask[:, :min_length] = True
    for i in range(len(seq)):
        start = min_length - seq[i].shape[0]
        start = max(0, start)
        padded_tensor[i, start : start + seq[i].shape[0], ...] = seq[i]
        padded_mask[i, start : start + seq[i].shape[0]] = True

    return padded_tensor, padded_mask


def left_pad_sequence(
    seq: list[torch.Tensor], padding_value: int = 0, min_length: int = 0
):
    """Pads a list of 2D tensors to the left with a given padding value."""
    max_lengths = max([t.shape[0] for t in seq])
    max_lengths = max(max_lengths, min_length)
    padded_tensor = torch.full((len(seq), max_lengths, seq[0].shape[1]), padding_value)
    padded_mask = torch.zeros(len(seq), max_lengths).bool()
    for i in range(len(seq)):
        pad_len = max_lengths - seq[i].shape[0]
        pad_len = max(pad_len, min_length - seq[i].shape[0])
        padded_tensor[i, pad_len : pad_len + seq[i].shape[0], ...] = seq[i]
        padded_mask[i, pad_len : pad_len + seq[i].shape[0]] = True

    return padded_tensor.float(), padded_mask


class TimeseriesDataset(torch.utils.data.Dataset):
    """Time series dataset for use with Global models where given a
    a dataset containing multiple time series we want to sample windows
    of a fixed length determined by the context_length and prediction_length
    and optionally lag length for each time series.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        conts: List[int] = [],
        lag_seq: List[int] = None,
        pad_fn=right_pad_sequence,
    ):
        self.df = df.copy()
        self.conts = conts.copy()

        self.lag_length = 0 if lag_seq is None else max(lag_seq)
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
        self.pad_fn = pad_fn

    def build(self) -> "TimeseriesDataset":
        ts = []
        for unique_id, grp in self.df.groupby("unique_id", sort=False):
            ts.append(torch.from_numpy(grp[["y"] + self.conts].values))
        ts, mask = self.pad_fn(ts, min_length=self.min_length)
        self.X = ts[:, :, :]
        self.pad_mask = mask
        return self

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, i: int) -> PaddedData:
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
        return PaddedData(data=X, pad_mask=pad_mask)


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


class TimeseriesDataset(torch.utils.data.Dataset):
    """Time series dataset for use with Global models where given a
    a dataset containing multiple time series we want to sample windows
    of a fixed length determined by the context_length and prediction_length
    and optionally lag length for each time series.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        conts: List[int] = [],
        lag_seq: List[int] = None,
        pad_fn=right_pad_sequence,
    ):
        self.df = df.copy()
        self.conts = conts.copy()

        self.lag_length = 0 if lag_seq is None else max(lag_seq)
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
        self.pad_fn = pad_fn

    def build(self) -> "TimeseriesDataset":
        ts = []
        for unique_id, grp in self.df.groupby("unique_id", sort=False):
            ts.append(torch.from_numpy(grp[["y"] + self.conts].values))
        ts, mask = self.pad_fn(ts, min_length=self.min_length)
        self.X = ts[:, :, :]
        self.pad_mask = mask
        return self

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, i: int) -> PaddedData:
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
        return PaddedData(data=X, pad_mask=pad_mask)


class MultivariateTimeSeriesDatasetLong(torch.utils.data.Dataset):
    """Time series dataset for use with Multivariate or Local models where given a
    a dataset containing multiple time series, N, we want to sample windows
    of a fixed length determined by the context_length and prediction_length
    across all time series so each sample will have shape (N, context_length + prediction_length)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        conts: List[int] = [],
        lag_seq: List[int] = None,
        pad_fn=right_pad_sequence,
    ):
        self.df = df.copy()
        self.conts = conts.copy()

        self.lag_length = 0 if lag_seq is None else max(lag_seq)
        self.context_length = torch.tensor(context_length).long()
        self.prediction_length = torch.tensor(prediction_length).long()

        lengths = (
            self.df.groupby("unique_id", sort=False)["ds"].count()
            - context_length
            - prediction_length
            - self.lag_length
            + 1  # we return x and y so add 1 for teacher forcing
        ).clip(1)
        self.max_length = lengths.max()
        self.window_size = context_length + prediction_length + self.lag_length
        self.lengths = lengths
        self.pad_fn = pad_fn

    def build(self) -> "MultivariateTimeSeriesDataset":
        ts = []
        for unique_id, grp in self.df.groupby("unique_id", sort=False):
            ts.append(torch.from_numpy(grp[["y"] + self.conts].values))
        ts, mask = self.pad_fn(ts, min_length=self.window_size)
        self.X = ts[:, :, :]
        self.pad_mask = mask
        return self

    def __len__(self) -> int:
        return self.max_length

    def __getitem__(self, i: int) -> PaddedData:
        X = self.X[:, i : i + self.window_size, 0].permute(1, 0)  # [N, window_size]
        pad_mask = self.pad_mask[:, i : i + self.window_size].permute(
            1, 0
        )  # [N, window_size]
        return PaddedData(data=X, pad_mask=pad_mask)


class MultivariateTimeSeriesDataset(torch.utils.data.Dataset):
    """Time series dataset for use with Multivariate or Local models where given a
    a dataset containing multiple time series, N, we want to sample windows
    of a fixed length determined by the context_length and prediction_length
    across all time series so each sample will have shape (N, context_length + prediction_length)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        conts: List[int] = [],
        lag_seq: List[int] = None,
        pad_fn=right_pad_sequence,
    ):
        self.df = df.copy()
        self.conts = conts.copy()

        self.lag_length = 0 if lag_seq is None else max(lag_seq)
        self.context_length = torch.tensor(context_length).long()
        self.prediction_length = torch.tensor(prediction_length).long()

        lengths = (
            self.df.groupby("unique_id", sort=False)["ds"].count()
            - context_length
            - prediction_length
            - self.lag_length
            + 1  # we return x and y so add 1 for teacher forcing
        ).clip(1)
        self.max_length = lengths.max()
        self.window_size = context_length + prediction_length + self.lag_length
        self.lengths = lengths
        self.pad_fn = pad_fn

    def build(self) -> "MultivariateTimeSeriesDataset":
        ts = []
        for unique_id, grp in self.df.groupby("unique_id", sort=False):
            ts.append(torch.from_numpy(grp[self.conts + ["y"]].values))
        ts, mask = self.pad_fn(ts, min_length=self.window_size)
        self.X = ts[:, :, :]
        self.pad_mask = mask
        return self

    def __len__(self) -> int:
        return self.max_length

    def __getitem__(self, i: int) -> PaddedData:
        X = self.X[0, i : i + self.window_size, :]
        pad_mask = self.pad_mask[0, i : i + self.window_size]
        return PaddedData(data=X, pad_mask=pad_mask)
