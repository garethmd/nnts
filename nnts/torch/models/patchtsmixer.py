# Third Party
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    PatchTSMixerModel,
    Trainer,
    TrainingArguments,
)

from nnts.utils import Scheduler, TrainingMethod

from ..datasets import PaddedData


class PatchTSMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()

        self.prediction_channel_indices = config.prediction_channel_indices

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()

        self.dropout_layer = nn.Dropout(config.head_dropout)
        if distribution_output is None:
            self.base_forecast_block = nn.Linear(
                (config.num_patches * config.d_model), config.prediction_length
            )
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(
                config.num_patches * config.d_model
            )

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features):
        """

        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, d_model)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, nvars)`.

        """

        hidden_features = self.flatten(
            hidden_features
        )  # [batch_size x n_vars x num_patch * d_model]
        hidden_features = self.dropout_layer(
            hidden_features
        )  # [batch_size x n_vars x num_patch * d_model]
        forecast = self.base_forecast_block(
            hidden_features
        )  # [batch_size x n_vars x prediction_length]
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)
        else:
            forecast = forecast.transpose(
                -1, -2
            )  # [batch_size x prediction_length x n_vars]

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(
                    z[..., self.prediction_channel_indices] for z in forecast
                )
            else:
                forecast = forecast[
                    ..., self.prediction_channel_indices
                ]  # [batch_size x prediction_length x n_vars]

        return forecast


@dataclass
class Hyperparams:
    patch_length: int
    num_input_channels: int
    optimizer: callable = torch.optim.Adam
    loss_fn: callable = F.l1_loss
    dropout: float = 0.0
    batch_size: int = 32
    lr: float = 0.005
    epochs: int = 100
    patience: int = 10
    early_stopper_patience: int = 30
    batches_per_epoch: int = 50
    weight_decay: float = 0.0
    training_method: TrainingMethod = TrainingMethod.DMS
    scheduler: Scheduler = Scheduler.REDUCE_LR_ON_PLATEAU
    model_file_path: str = f"logs"
    d_model: int = 48
    num_layers: int = 3
    expansion_factor: int = 3
    dropout: float = 0.5
    head_dropout: float = 0.7
    mode: str = (
        "common_channel"  # change it `mix_channel` if we need to explicitly model channel correlations
    )
    scaling: str = "std"


class PatchTSMixer(PatchTSMixerModel):

    def __init__(
        self, context_length: int, prediction_length: int, params: Hyperparams
    ):

        config = PatchTSMixerConfig(
            context_length=context_length,
            prediction_length=prediction_length,
            patch_length=params.patch_length,
            num_input_channels=params.num_input_channels,
            patch_stride=params.patch_length,
            d_model=params.d_model,
            num_layers=params.num_layers,
            expansion_factor=params.expansion_factor,
            dropout=params.dropout,
            head_dropout=params.head_dropout,
            mode=params.mode,  # change it `mix_channel` if we need to explicitly model channel correlations
            scaling=params.scaling,
        )
        super().__init__(config)
        self.head = PatchTSMixerForPredictionHead(
            config=config,
            distribution_output=None,
        )
        self.seq_len = context_length

    def train_output(
        self, batch: PaddedData, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = batch.data[:, : self.seq_len, :]
        pad_mask = batch.pad_mask[:, : self.seq_len]
        model_output = self(x)
        y_hat = self.head(model_output.last_hidden_state)
        y_hat = y_hat * model_output.scale + model_output.loc

        y = batch.data[:, self.seq_len :, :]

        pad_mask = batch.pad_mask[:, self.seq_len :]
        y_hat = y_hat[pad_mask]
        y = y[pad_mask]

        return y_hat, y

    def generate(
        self, x: torch.Tensor, pad_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        model_output = self(x)
        y_hat = self.head(model_output.last_hidden_state)
        y_hat = y_hat * model_output.scale + model_output.loc
        return y_hat
