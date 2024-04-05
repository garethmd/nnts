import pandas as pd
import torch
from torch.utils.data import DataLoader

import nnts.data
import nnts.experiments
import nnts.models

from . import datasets


def masked_mean_abs_scaling(
    seq: torch.Tensor, mask: torch.Tensor = None, eps: float = 1, dim: int = 1
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
    scale = torch.clamp(scale, min=eps)
    return scale


class TorchTimeseriesDataLoaderFactory(nnts.data.DataLoaderFactory):
    def __call__(
        self,
        data: pd.DataFrame,
        metadata: nnts.data.Metadata,
        scenario: nnts.experiments.CovariateScenario,
        params: nnts.models.Hyperparams,
        shuffle: bool,
    ) -> DataLoader:
        ts = datasets.TimeseriesDataset(
            data,
            conts=scenario.conts,
            context_length=metadata.context_length,
            prediction_length=metadata.prediction_length,
        ).build()

        return DataLoader(ts, batch_size=params.batch_size, shuffle=shuffle)
