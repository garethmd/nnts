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
    dropout: float = 0.3
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

    hidden_size: int = 256
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    decoder_output_dim: int = 8
    temporal_decoder_dim: int = 128
    output_dim: int = 1


class ResidualBlock(nn.Module):
    """Residual Block. We use the residual block as the basic layer in our architecture. It is an MLP with one
    hidden layer with ReLU activation. It also has a skip connection that is fully linear. We use dropout on the
    linear layer that maps the hidden layer to the output and also use layer norm at the output.
    We separate the model into encoding and decoding sections. The encoding section has a novel feature
    projection step followed by a dense MLP encoder. The decoder section consists of a dense decoder followed
    by a novel temporal decoder. Note that the dense encoder (green block with ne layers) and decoder blocks
    (yellow block with nd layers) in Figure 1 can be merged into a single block. For the sake of exposition we keep
    them separate as we tune the hidden layer size in the two blocks separately. Also the last layer of the decoder
    block is unique in the sense that its output dimension needs to be H × p before the reshape operation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        dropout: float = 0.0,
        norm: bool = True,
    ):
        super(ResidualBlock, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        self.norm = norm

    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        out = self.dropout(out)
        skip = self.skip(x)
        out += skip
        if self.norm:
            out = self.layer_norm(out)
        return out


class TiDE(nn.Module):

    def __init__(
        self,
        h: int,
        input_size: int,
        configs: Hyperparams,
    ):
        super(TiDE, self).__init__()
        self.h = h
        self.seq_len = input_size

        dense_encoder_layers = [
            ResidualBlock(
                input_dim=input_size if i == 0 else configs.hidden_size,
                hidden_size=configs.hidden_size,
                output_dim=configs.hidden_size,
                dropout=configs.dropout,
            )
            for i in range(configs.num_encoder_layers)
        ]
        self.dense_encoder = nn.Sequential(*dense_encoder_layers)

        # Decoder
        decoder_output_size = configs.decoder_output_dim * h
        dense_decoder_layers = [
            ResidualBlock(
                input_dim=configs.hidden_size,
                hidden_size=configs.hidden_size,
                output_dim=(
                    decoder_output_size
                    if i == configs.num_decoder_layers - 1
                    else configs.hidden_size
                ),
                dropout=configs.dropout,
            )
            for i in range(configs.num_decoder_layers)
        ]
        self.dense_decoder = nn.Sequential(*dense_decoder_layers)

        self.temporal_decoder = ResidualBlock(
            input_dim=configs.decoder_output_dim,
            hidden_size=configs.temporal_decoder_dim,
            output_dim=configs.output_dim,
            dropout=configs.dropout,
            norm=True,
        )

        self.global_skip = nn.Linear(
            in_features=input_size, out_features=h * configs.output_dim
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]

        # Flatten insample_y
        x = x.reshape(batch_size, -1)  #   [B, L, 1] -> [B, L]

        # Global skip connection
        x_skip = self.global_skip(x)  #   [B, L] -> [B, h * n_outputs]
        x_skip = x_skip.reshape(
            batch_size, self.h, -1
        )  #   [B, h * n_outputs] -> [B, h, n_outputs]

        # Dense encoder
        x = self.dense_encoder(
            x
        )  #   [B, L * (1 + 2 * temporal_width) + h * temporal_width + S] -> [B, hidden_size]

        # Dense decoder
        x = self.dense_decoder(x)  #   [B, hidden_size] ->  [B, decoder_output_dim * h]
        x = x.reshape(
            batch_size, self.h, -1
        )  #   [B, decoder_output_dim * h] -> [B, h, decoder_output_dim]

        # Temporal decoder
        x = self.temporal_decoder(
            x
        )  #  [B, h, temporal_width + decoder_output_dim] -> [B, h, n_outputs]

        # Map to output domain
        forecast = x + x_skip
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

        return y_hat, y

    def generate(
        self, x: torch.Tensor, pad_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        y_hat = self(x, pad_mask)
        return y_hat
