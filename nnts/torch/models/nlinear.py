from dataclasses import dataclass
from typing import Any, Tuple

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
    lr: float = 0.005
    epochs: int = 100
    patience: int = 10
    early_stopper_patience: int = 30
    batches_per_epoch: int = 50
    weight_decay: float = 0.0
    training_method: TrainingMethod = TrainingMethod.DMS
    scheduler: Scheduler = Scheduler.REDUCE_LR_ON_PLATEAU
    model_file_path: str = f"logs"
    individual: bool = True
    kernel_size = 25
    scaling_fn: Any = None
    enc_in: int = 1


def get_mutlivariate_params():
    params = Hyperparams(
        optimizer=torch.optim.Adam,
        loss_fn=torch.nn.MSELoss(),
        batch_size=32,
        batches_per_epoch=100,
        training_method=TrainingMethod.DMS,
        model_file_path="logs",
        epochs=10,
        scheduler=Scheduler.STEP_LR,
        lr=0.005,
        weight_decay=0.0,
    )
    return params


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, h: int, input_size: int, c_in: int, configs: Hyperparams):
        super(NLinear, self).__init__()
        print("enc_in", configs.enc_in)
        self.seq_len = input_size
        self.pred_len = h
        self.scaling_fn = configs.scaling_fn

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = c_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
            ).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]

    def train_output(
        self, batch: PaddedData, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = batch.data
        y_hat = self(x[:, : self.seq_len, :])
        y = x[:, self.seq_len :, :]
        return y_hat, y

    def generate(
        self, x: torch.Tensor, pad_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        y_hat = self(x)
        return y_hat
