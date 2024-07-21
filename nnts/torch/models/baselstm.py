from typing import Dict, Tuple

import torch
import torch.nn as nn

from nnts import utils

from .. import datasets


class LinearModel(nn.Module):
    """
    This model predicts a point predction for ordinal data.
    """

    def __init__(self, hidden_size: int, output_size: int):
        super(LinearModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.tensor, target_scale: torch.tensor):
        y_hat = self.main(x)
        if target_scale is not None:
            y_hat = y_hat * target_scale
        return y_hat


class BaseLSTMDecoder(nn.Module):
    def __init__(self, params: utils.Hyperparams, output_dim: int):
        super(BaseLSTMDecoder, self).__init__()
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
        B, T, C = X.shape
        if hidden is None:
            hidden = self.init_hidden_zeros(B)
        return self.rnn(X, hidden)


class BaseLSTM(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: utils.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
    ):
        super(BaseLSTM, self).__init__()
        self.scaling_fn = scaling_fn
        self.decoder = BaseLSTMDecoder(params, output_dim)
        self.distribution = Distribution(params.hidden_dim, output_dim)

    def forward(self, X: torch.tensor, pad_mask: torch.tensor) -> torch.tensor:
        X = X.clone()
        B, T, C = X.shape
        target_scale = self.scaling_fn(X, pad_mask)
        conts = X / target_scale
        embedded = torch.cat(
            [conts, torch.log(target_scale[:, :, :1]).expand(B, T, 1)], 2
        )
        out, _ = self.decoder(embedded)
        distr = self.distribution(out, target_scale=target_scale)
        return distr

    def generate(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        prediction_length: int,
        context_length: int,
    ) -> torch.tensor:
        pred_list = []
        while True:
            pad_mask = pad_mask[:, -context_length:]
            assert (
                pad_mask.sum()
                == X[:, -context_length:, :].shape[0]
                * X[:, -context_length:, :].shape[1]
            )
            preds = self.forward(X[:, -context_length:, :], pad_mask)
            # focus only on the last time step
            preds = preds[:, -1:, :]  # becomes (B, 1, C)
            pred_list.append(preds)

            if len(pred_list) >= prediction_length:
                break
            y_hat = preds.detach().clone()
            X = torch.cat((X, y_hat), dim=1)  # (B, T+1)
            pad_mask = torch.cat((pad_mask, torch.ones_like(preds[:, :, 0])), dim=1)
        return torch.cat(pred_list, 1)

    def teacher_forcing_output(self, batch: datasets.PaddedData, *args, **kwargs):
        x = batch.data
        pad_mask = batch.pad_mask
        y_hat = self(x[:, :-1, :], pad_mask[:, :-1])
        y = x[:, 1:, :]

        y_hat = y_hat[pad_mask[:, 1:]]
        y = y[pad_mask[:, 1:]]
        return y_hat, y


class BaseFutureCovariateLSTM(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: utils.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        known_future_covariates: int,
    ):
        super(BaseFutureCovariateLSTM, self).__init__()
        self.scaling_fn = scaling_fn
        self.decoder = BaseLSTMDecoder(params, output_dim + known_future_covariates)
        self.distribution = Distribution(params.hidden_dim, output_dim)
        self.known_future_covariates = known_future_covariates
        self.output_dim = output_dim

    def forward(
        self, X: torch.tensor, pad_mask: torch.tensor, conts_future: torch.tensor = None
    ) -> torch.tensor:
        X = X.clone()
        B, T, _ = X.shape
        target_scale = self.scaling_fn(X, pad_mask)
        conts = X / target_scale
        conts_list = [conts, torch.log(target_scale[:, :, :1]).expand(B, T, 1)]

        if conts_future is not None:
            conts_list.append(conts_future)

        embedded = torch.cat(conts_list, 2)
        out, _ = self.decoder(embedded)
        distr = self.distribution(out, target_scale=target_scale)
        return distr

    def teacher_forcing_output(
        self, batch: datasets.PaddedData
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = batch.data
        conts_future = x[:, :, -self.known_future_covariates :]
        x = x[:, :, : -self.known_future_covariates]

        pad_mask = batch.pad_mask
        y_hat = self(
            x[:, :-1, :], pad_mask[:, :-1], conts_future=conts_future[:, :-1, :]
        )
        y = x[:, 1:, :1]

        y_hat = y_hat[pad_mask[:, 1:]]
        y = y[pad_mask[:, 1:]]
        return y_hat, y

    def generate(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        prediction_length: int,
        context_length: int,
    ) -> torch.tensor:
        y_hat = torch.zeros(X.shape[0], prediction_length, self.output_dim)
        past_conts = X[:, :context_length, :]
        past_pad_mask = pad_mask[:, :context_length]

        future_conts = X[:, context_length:, -self.known_future_covariates :]
        future_pad_mask = pad_mask[:, context_length:]

        for x in range(prediction_length):
            preds = self(
                past_conts[:, :, : -self.known_future_covariates],
                past_pad_mask,
                conts_future=past_conts[:, :, -self.known_future_covariates :],
            )
            preds = preds[:, -1:, :].detach().clone()  # becomes (B, 1, C)
            y_hat[:, x, :] = preds[:, 0, :]

            # append the prediction to the past
            past_conts[:, :, : -self.known_future_covariates] = torch.cat(
                (past_conts[:, :, : -self.known_future_covariates], preds), dim=1
            )[:, 1:, :]
            past_conts[:, :, -self.known_future_covariates :] = torch.cat(
                [
                    past_conts[:, :, -self.known_future_covariates :],
                    future_conts[:, x : x + 1, :],
                ],
                dim=1,
            )[:, 1:, :]
            past_pad_mask = torch.cat(
                [past_pad_mask, future_pad_mask[:, x : x + 1]], dim=1
            )[:, 1:]

        return y_hat

    def validate(self, batch: datasets.PaddedData, prediction_length, context_length):
        y = batch.data[
            :, context_length : context_length + prediction_length, : self.output_dim
        ]
        y_hat = self.generate(
            batch.data,
            batch.pad_mask,
            prediction_length=prediction_length,
            context_length=context_length,
        )
        y_hat = y_hat[:, -prediction_length:, ...]
        return y_hat, y
