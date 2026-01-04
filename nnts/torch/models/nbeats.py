# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
N-BEATS Model.
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnts.utils import Scheduler, TrainingMethod

from ..datasets import PaddedData


@dataclass
class Hyperparams:
    """Class for keeping track of training and model params."""

    optimizer: callable = torch.optim.Adam
    loss_fn: callable = F.l1_loss
    dropout: float = 0.0

    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    early_stopper_patience: int = 30
    batches_per_epoch: int = 50
    weight_decay: float = 0.0
    training_method: TrainingMethod = TrainingMethod.DMS
    scheduler: Scheduler = Scheduler.REDUCE_LR_ON_PLATEAU
    model_file_path = f"logs"

    # Paper
    # batch_size: int = 1024

    # GluonTS
    theta_size: int = 32
    num_stacks: int = 30
    num_layers: int = 2
    layer_size: int = 512
    batch_size: int = 32

    # Darts
    # theta_size: int = 5
    # num_stacks: int = 30
    # num_layers: int = 4
    # layer_size: int = 256


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, theta_size: int):
        super().__init__()
        self.theta_size = theta_size

    def forward(self, theta: torch.Tensor):
        return theta[:, : self.theta_size], theta[:, -self.theta_size :]


class TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(
        self, degree_of_polynomial: int, backcast_size: int, forecast_size: int
    ):
        super().__init__()
        self.polynomial_size = (
            degree_of_polynomial + 1
        )  # degree of polynomial with constant term

        # Create the backcast time tensor using PyTorch methods
        backcast_time = torch.cat(
            [
                (torch.arange(backcast_size, dtype=torch.float32) / backcast_size)
                .pow(i)
                .unsqueeze(0)
                for i in range(self.polynomial_size)
            ]
        )
        self.backcast_time = nn.Parameter(backcast_time, requires_grad=False)

        # Create the forecast time tensor using PyTorch methods
        forecast_time = torch.cat(
            [
                (torch.arange(forecast_size, dtype=torch.float32) / forecast_size)
                .pow(i)
                .unsqueeze(0)
                for i in range(self.polynomial_size)
            ]
        )
        self.forecast_time = nn.Parameter(forecast_time, requires_grad=False)

    def forward(self, theta: torch.Tensor):
        # Compute the backcast and forecast using einsum for matrix multiplication
        backcast = torch.einsum(
            "bp,pt->bt", theta[:, self.polynomial_size :], self.backcast_time
        )
        forecast = torch.einsum(
            "bp,pt->bt", theta[:, : self.polynomial_size], self.forecast_time
        )
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Harmonic functions to model seasonality.
    """

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()

        # Define frequency tensor
        frequency = torch.cat(
            (
                torch.zeros(1, dtype=torch.float32),
                torch.arange(
                    harmonics, harmonics / 2 * forecast_size, dtype=torch.float32
                )
                / harmonics,
            )
        ).unsqueeze(
            0
        )  # Shape: (1, num_frequencies)

        # Define backcast and forecast grids
        backcast_grid = (
            -2
            * torch.pi
            * (
                torch.arange(backcast_size, dtype=torch.float32).unsqueeze(1)
                / forecast_size
            )
            * frequency
        )
        forecast_grid = (
            2
            * torch.pi
            * (
                torch.arange(forecast_size, dtype=torch.float32).unsqueeze(1)
                / forecast_size
            )
            * frequency
        )

        # Create cos and sin templates for backcast and forecast using PyTorch
        self.backcast_cos_template = nn.Parameter(
            torch.cos(backcast_grid).transpose(0, 1), requires_grad=False
        )
        self.backcast_sin_template = nn.Parameter(
            torch.sin(backcast_grid).transpose(0, 1), requires_grad=False
        )
        self.forecast_cos_template = nn.Parameter(
            torch.cos(forecast_grid).transpose(0, 1), requires_grad=False
        )
        self.forecast_sin_template = nn.Parameter(
            torch.sin(forecast_grid).transpose(0, 1), requires_grad=False
        )

    def forward(self, theta: torch.Tensor):
        # Calculate the number of parameters per harmonic
        params_per_harmonic = theta.shape[1] // 4

        # Compute backcast harmonics using einsum for efficient batched multiplication
        backcast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, 2 * params_per_harmonic : 3 * params_per_harmonic],
            self.backcast_cos_template,
        )
        backcast_harmonics_sin = torch.einsum(
            "bp,pt->bt", theta[:, 3 * params_per_harmonic :], self.backcast_sin_template
        )
        backcast = backcast_harmonics_cos + backcast_harmonics_sin

        # Compute forecast harmonics using einsum
        forecast_harmonics_cos = torch.einsum(
            "bp,pt->bt", theta[:, :params_per_harmonic], self.forecast_cos_template
        )
        forecast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, params_per_harmonic : 2 * params_per_harmonic],
            self.forecast_sin_template,
        )
        forecast = forecast_harmonics_cos + forecast_harmonics_sin

        return backcast, forecast


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        h: int,
        input_size,
        theta_size: int,
        basis_function: nn.Module,
        layers: int,
        layer_size: int,
    ):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)]
            + [
                nn.Linear(in_features=layer_size, out_features=layer_size)
                for _ in range(layers - 1)
            ]
        )
        self.basis_parameters = nn.Linear(
            in_features=layer_size, out_features=theta_size * 2
        )
        self.basis_function = basis_function
        self.backcast = nn.Linear(in_features=theta_size, out_features=input_size)
        self.forecast = nn.Linear(in_features=theta_size, out_features=h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = F.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        backcast, forecast = self.basis_function(basis_parameters)
        return self.backcast(backcast), self.forecast(forecast)


class NBeats(nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, h: int, input_size: int, configs: Hyperparams):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(
                    h=h,
                    input_size=input_size,
                    theta_size=configs.theta_size,
                    basis_function=GenericBasis(configs.theta_size),
                    layers=configs.num_layers,
                    layer_size=configs.layer_size,
                )
                for _ in range(configs.num_stacks)
            ]
        )
        self.seq_len = input_size
        self.h = h

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        input_mask = input_mask.squeeze(-1)
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast.unsqueeze(-1)

    def train_output(
        self, batch: PaddedData, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = batch.data[:, : self.seq_len, :]
        x_pad_mask = batch.pad_mask[:, : self.seq_len]
        y_hat = self(x, x_pad_mask)
        y = batch.data[:, self.seq_len :, :]

        pad_mask = batch.pad_mask[:, self.seq_len :]
        y_hat = y_hat[pad_mask]
        y = y[pad_mask]

        return y_hat, y

    def generate(
        self, x: torch.Tensor, pad_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        y_hat = self(x, pad_mask)
        return y_hat
