from typing import List

import pandas as pd
import torch

import nnts.data
import nnts.preprocessing


def masked_mean_abs_scaling(
    seq: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-10,
    dim: int = 1,
):
    if mask is None:
        mask = torch.ones_like(seq)
    if len(mask.shape) == 2 and len(seq.shape) == 3:
        mask = mask[:, :, None]

    if len(mask.shape) != len(seq.shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match sequence shape {seq.shape}"
        )
    seq_sum = (seq * mask).abs().sum(dim, keepdim=True)
    item_count = mask.sum(dim, keepdim=True).clamp(min=1)

    scale = seq_sum / item_count

    if scale.max() < 1:
        scale = torch.clamp(scale, min=eps)
    else:
        scale = torch.clamp(scale, min=1)
    return scale


class StandardScaler(nnts.preprocessing.Transformation):

    def __init__(self, mean=None, std=None, cols=None):
        self.mean = mean
        self.std = std
        self.cols = cols

    def fit(self, data: pd.DataFrame, cols=None):
        if self.cols is not None:
            cols = self.cols

        numeric_data = (
            data.select_dtypes(include=["number"]) if cols is None else data[cols]
        )
        self.mean = numeric_data.mean()
        self.std = numeric_data.std()
        return self

    def transform(self, data: pd.DataFrame, cols=None):
        if self.cols is not None:
            cols = self.cols

        numeric_data = (
            data.select_dtypes(include=["number"]) if cols is None else data[cols]
        )
        numeric_cols = numeric_data.columns
        data[numeric_cols] = (numeric_data - self.mean) / self.std
        return data

    def inverse_transform(self, data: pd.DataFrame, cols=None):
        if self.cols is not None:
            cols = self.cols
        numeric_data = (
            data.select_dtypes(include=["number"]) if cols is None else data[cols]
        )
        numeric_cols = numeric_data.columns
        data[numeric_cols] = numeric_data * self.std + self.mean
        return data


class MaxMinScaler(nnts.preprocessing.Transformation):
    def __init__(self, max=None, min=None, cols=None):
        self.max = max
        self.min = min
        self.cols = cols

    def fit(self, data: pd.DataFrame, cols=None):
        if self.cols is not None:
            cols = self.cols
        numeric_data = (
            data.select_dtypes(include=["number"]) if cols is None else data[cols]
        )
        self.max = numeric_data.max()
        self.min = numeric_data.min()
        return self

    def transform(self, data: pd.DataFrame, cols=None):
        if self.cols is not None:
            cols = self.cols
        numeric_data = (
            data.select_dtypes(include=["number"]) if cols is None else data[cols]
        )
        numeric_cols = numeric_data.columns
        data[numeric_cols] = ((numeric_data - self.min) / (self.max - self.min)) - 0.5
        return data

    def inverse_transform(self, data: pd.DataFrame, cols=None):
        if self.cols is not None:
            cols = self.cols
        numeric_data = (
            data.select_dtypes(include=["number"]) if cols is None else data[cols]
        )
        numeric_cols = numeric_data.columns
        data[numeric_cols] = numeric_data * (self.max - self.min) + self.min
        return data


def create_lags(
    n_timesteps: int, past_target: torch.tensor, lag_seq: List[int]
) -> torch.tensor:
    lag_features = []
    for t in range(0, n_timesteps):
        lag_seq = lag_seq + 1
        lag_for_step = past_target.index_select(1, lag_seq)
        lag_features.append(lag_for_step)
    return torch.stack(lag_features, dim=1).flip(1)


class LagProcessor:

    def __init__(self, lag_seq: List[int]):
        self.lag_seq = torch.tensor(lag_seq)

    def create(self, n_timesteps: int, past_target: torch.tensor):
        return create_lags(n_timesteps, past_target, self.lag_seq - 1)

    def __len__(self):
        return len(self.lag_seq)

    def max(self):
        return max(self.lag_seq)
