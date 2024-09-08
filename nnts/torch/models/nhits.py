from dataclasses import dataclass
from typing import Tuple

import numpy as np
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
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    early_stopper_patience: int = 30
    batches_per_epoch: int = 50
    weight_decay: float = 0.0
    training_method: TrainingMethod = TrainingMethod.DMS
    scheduler: Scheduler = Scheduler.REDUCE_LR_ON_PLATEAU
    model_file_path = f"logs"

    n_blocks = [1, 1, 1]
    mlp_units = [[512, 512], [512, 512]]
    n_pool_kernel_size = [1, 1, 1]
    n_freq_downsample = [1, 1, 1]


def domain_map(y_hat: torch.Tensor):
    """
    Univariate loss operates in dimension [B,T,H]/[B,H]
    This changes the network's output from [B,H,1]->[B,H]
    """
    return y_hat.squeeze(-1)


class _IdentityBasis(nn.Module):
    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
        out_features: int = 1,
    ):
        super().__init__()
        assert interpolation_mode in ["linear", "nearest"]
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode
        self.out_features = out_features

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # Interpolation is performed on default dim=-1 := H
        knots = knots.reshape(len(knots), self.out_features, -1)
        # knots = knots[:,None,:]
        forecast = F.interpolate(
            knots, size=self.forecast_size, mode=self.interpolation_mode
        )
        # forecast = forecast[:,0,:]

        # [B,Q,H] -> [B,H,Q]
        forecast = forecast.permute(0, 2, 1)
        return backcast, forecast


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

POOLING = ["MaxPool1d", "AvgPool1d"]


class NHITSBlock(nn.Module):
    """
    NHITS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        input_size = pooled_hist_size

        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        # Block MLPs
        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                # raise NotImplementedError('dropout')
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class NHITS(nn.Module):
    """NHITS

    The Neural Hierarchical Interpolation for Time Series (NHITS), is an MLP-based deep
    neural architecture with backward and forward residual links. NHITS tackles volatility and
    memory complexity challenges, by locally specializing its sequential predictions into
    the signals frequencies with hierarchical interpolation and pooling.

    **Parameters:**
    `h`: int, Forecast horizon.
    `input_size`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].
    `stat_exog_list`: str list, static exogenous columns.
    `hist_exog_list`: str list, historic exogenous columns.
    `futr_exog_list`: str list, future exogenous columns.
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.
    `activation`: str, activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'].
    `stack_types`: List[str], stacks list in the form N * ['identity'], to be deprecated in favor of `n_stacks`. Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
    `n_blocks`: List[int], Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).
    `mlp_units`: List[List[int]], Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).
    `n_freq_downsample`: List[int], list with the stack's coefficients (inverse expressivity ratios). Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
    `interpolation_mode`: str='linear', interpolation basis from ['linear', 'nearest', 'cubic'].
    `n_pool_kernel_size`: List[int], list with the size of the windows to take a max/avg over. Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
    `pooling_mode`: str, input pooling module from ['MaxPool1d', 'AvgPool1d'].
    `dropout_prob_theta`: float, Float between (0, 1). Dropout for NHITS basis.
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).
    `max_steps`: int=1000, maximum number of training steps.
    `learning_rate`: float=1e-3, Learning rate between (0, 1).
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.
    `val_check_steps`: int=100, Number of training steps between every validation loss check.
    `batch_size`: int=32, number of different series in each batch.
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.
    `inference_windows_batch_size`: int=-1, number of windows to sample in each inference batch, -1 uses all.
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.
    `step_size`: int=1, step size between each window of temporal data.
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).
    `random_seed`: int, random_seed for pytorch initializer and numpy generators.
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.
    `alias`: str, optional,  Custom name of the model.
    `optimizer`: Subclass of 'torch.optim.Optimizer', optional, user specified optimizer instead of the default choice (Adam).
    `optimizer_kwargs`: dict, optional, list of parameters used by the user specified `optimizer`.
    `lr_scheduler`: Subclass of 'torch.optim.lr_scheduler.LRScheduler', optional, user specified lr_scheduler instead of the default choice (StepLR).
    `lr_scheduler_kwargs`: dict, optional, list of parameters used by the user specified `lr_scheduler`.
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    **References:**
    -[Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza,
    Max Mergenthaler-Canseco, Artur Dubrawski (2023). "NHITS: Neural Hierarchical Interpolation for Time Series Forecasting".
    Accepted at the Thirty-Seventh AAAI Conference on Artificial Intelligence.](https://arxiv.org/abs/2201.12886)
    """

    # Class attributes

    def __init__(
        self,
        h,
        input_size,
        configs: Hyperparams,
        exclude_insample_y=False,
        stack_types: list = ["identity", "identity", "identity"],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta=0.0,
        activation="ReLU",
    ):

        super().__init__()
        # Architecture
        self.seq_len = input_size
        self.h = h
        blocks = self.create_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            n_blocks=configs.n_blocks,
            mlp_units=configs.mlp_units,
            n_pool_kernel_size=configs.n_pool_kernel_size,
            n_freq_downsample=configs.n_freq_downsample,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
        )
        self.blocks = torch.nn.ModuleList(blocks)
        self.decompose_forecast = False

    def create_stack(
        self,
        h,
        input_size,
        stack_types,
        n_blocks,
        mlp_units,
        n_pool_kernel_size,
        n_freq_downsample,
        pooling_mode,
        interpolation_mode,
        dropout_prob_theta,
        activation,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                assert (
                    stack_types[i] == "identity"
                ), f"Block type {stack_types[i]} not found!"

                n_theta = input_size + max(h // n_freq_downsample[i], 1)
                basis = _IdentityBasis(
                    backcast_size=input_size,
                    forecast_size=h,
                    out_features=1,
                    interpolation_mode=interpolation_mode,
                )

                nbeats_block = NHITSBlock(
                    h=h,
                    input_size=input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    n_pool_kernel_size=n_pool_kernel_size[i],
                    pooling_mode=pooling_mode,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    def forward(self, x, pad_mask):

        # Parse windows_batch
        insample_y = x.squeeze(-1)
        insample_mask = pad_mask

        # insample
        residuals = insample_y.flip(dims=(-1,))  # backcast init reverse time order
        insample_mask = insample_mask.flip(
            dims=(-1,)
        )  # backcast init reverse time order

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        block_forecasts = [forecast.repeat(1, self.h, 1)]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

            if self.decompose_forecast:
                block_forecasts.append(block_forecast)

        # Adapting output's domain
        forecast = domain_map(forecast)

        if self.decompose_forecast:
            # (n_batch, n_blocks, h, output_size)
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2, 3)
            block_forecasts = block_forecasts.squeeze(-1)  # univariate output
            return block_forecasts
        else:
            return forecast

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

        return y_hat.unsqueeze(-1), y

    def generate(
        self, x: torch.Tensor, pad_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        y_hat = self(x, pad_mask)
        return y_hat.unsqueeze(-1)
