from typing import Dict, Tuple

import torch
import torch.nn as nn

from nnts import utils

from .. import datasets


class UnrolledLSTMDecoder(nn.Module):
    def __init__(self, params: utils.Hyperparams, output_dim: int):
        super(UnrolledLSTMDecoder, self).__init__()
        self.params = params

        if params.rnn_type == "gru":
            self.rnn = nn.GRU(
                output_dim + 1,
                params.hidden_dim,
                params.n_layers,
                dropout=params.dropout,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                output_dim + 1,
                params.hidden_dim,
                params.n_layers,
                dropout=params.dropout,
                batch_first=True,
            )
        self.rnn_type = params.rnn_type

    def init_hidden_zeros(self, batch_size: int):
        if self.rnn_type == "gru":
            hidden = torch.zeros(
                self.params.n_layers, batch_size, self.params.hidden_dim
            )
        else:
            h0 = torch.zeros(self.params.n_layers, batch_size, self.params.hidden_dim)
            c0 = torch.zeros(self.params.n_layers, batch_size, self.params.hidden_dim)
            hidden = (h0, c0)
        return hidden

    def forward(self, X: torch.tensor, hidden=None):
        """
        H: number of steps to unroll
        """
        B, T, C = X.shape
        if hidden is None:
            hidden = self.init_hidden_zeros(B)
        out, hidden = self.rnn(X, hidden)
        return out, hidden


class UnrolledLSTM(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: utils.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
    ):
        super(UnrolledLSTM, self).__init__()
        self.scaling_fn = scaling_fn
        self.decoder = UnrolledLSTMDecoder(params, output_dim)
        self.distribution = Distribution(params.hidden_dim, output_dim)

    def forward(self, X: torch.tensor, pad_mask: torch.tensor, H: int) -> torch.tensor:
        X = X.clone()
        B, T, C = X.shape
        y_hat = torch.zeros(B, T + H, C)

        target_scale = self.scaling_fn(X, pad_mask)
        conts = X / target_scale
        embedded = torch.cat(
            [conts, torch.log(target_scale[:, :, :1]).expand(B, T, 1)], 2
        )
        out, hidden = self.decoder(embedded)
        out = self.distribution(out, target_scale=None)
        y_hat[:, :T, :] = out
        for t in range(0, H):
            embedded = torch.cat([out[:, -1:, :], torch.log(target_scale[:, :, :1])], 2)
            out, hidden = self.decoder(embedded, hidden)
            out = self.distribution(out, target_scale=None)
            y_hat[:, T + t, :] = out[:, -1, :]

        y_hat = y_hat * target_scale
        return y_hat

    def generate(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        prediction_length: int,
        context_length: int,
    ) -> torch.tensor:
        y_hat = self.forward(
            X[:, -context_length:, :], pad_mask[:, -context_length:], prediction_length
        )
        y_hat = y_hat[:, -prediction_length:, :]
        return y_hat

    def free_running(
        self, batch: datasets.PaddedData, context_length: int, prediction_length: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = batch.data
        pad_mask = batch.pad_mask

        y = x[:, 1:, :]

        y_hat = self.forward(
            x[:, :context_length, :],
            pad_mask[:, :context_length],
            prediction_length - 1,
        )
        y_hat = y_hat[pad_mask[:, 1:]]
        y = y[pad_mask[:, 1:]]
        return y_hat, y

    def teacher_forcing_output(
        self, batch: datasets.PaddedData
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = batch.data
        pad_mask = batch.pad_mask
        y_hat = self(x[:, :-1, :], pad_mask[:, :-1], H=0)
        y = x[:, 1:, :]
        y_hat = y_hat[pad_mask[:, 1:]]
        y = y[pad_mask[:, 1:]]
        return y_hat, y
