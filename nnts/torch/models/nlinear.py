from typing import Tuple

import torch
import torch.nn as nn

from ..datasets import PaddedData


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs, individual=True, enc_in=1):
        super(NLinear, self).__init__()
        self.seq_len = configs.context_length
        self.pred_len = configs.prediction_length

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
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
