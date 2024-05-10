from typing import Dict, List

import torch
import torch.nn as nn

import nnts.models

from .. import models


class DeepAR(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: nnts.models.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        lag_seq: List[int],
    ):
        super(DeepAR, self).__init__()
        self.scaling_fn = scaling_fn
        self.output_dim = output_dim
        self.scaled_features = 3
        self.lag_seq = (
            [0]
            + lag_seq
            + [max(lag_seq) + sf for sf in range(1, self.scaled_features + 1)]
        )
        self.decoder = models.unrolledlstm.UnrolledLSTMDecoder(
            params, len(self.lag_seq) - 1
        )
        self.distribution = Distribution(params.hidden_dim, output_dim)

    def forward(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        H: int,
        context_length: int,
    ) -> torch.tensor:
        X = X.clone()
        if H > 0:
            X[:, context_length:, : -self.scaled_features + 1] = (
                0.0  # zero out the target
            )
        B, T, C = X.shape

        y_hat = torch.zeros(B, T, self.output_dim)

        target_scale = self.scaling_fn(
            X[:, :context_length, :1], pad_mask[:, :context_length]
        )
        X = torch.cat(
            [
                X,
                torch.log(target_scale[:, :, :1]).expand(B, T, 1),
            ],
            2,
        )

        X[:, :, : -self.scaled_features] = (
            X[:, :, : -self.scaled_features] / target_scale
        )

        input = X[:, :context_length, :].index_select(2, torch.tensor(self.lag_seq))
        out, hidden = self.decoder(input)
        out = self.distribution(out, target_scale=None)
        y_hat[:, :context_length, :] = out
        for t in range(1, H):
            X[
                :, context_length + t - 1 : context_length + t, : -self.scaled_features
            ] = torch.cat(
                [
                    out[:, -1:, :],
                    X[
                        :,
                        context_length + t - 2 : context_length + t - 1,
                        : -self.scaled_features - 1,
                    ],
                ],
                dim=2,
            )

            # select the lag features and detach from the graph to prevent backpropagation
            input = (
                X[:, context_length + t - 1 : context_length + t, :]
                .index_select(2, torch.tensor(self.lag_seq))
                .clone()
                .detach()
            )
            out, hidden = self.decoder(
                input,
                hidden,
            )
            out = self.distribution(out, target_scale=None)
            y_hat[:, context_length + t - 1, :] = out[:, -1, :]

        y_hat = y_hat * target_scale[:, :, : self.output_dim]
        return y_hat

    def generate(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        prediction_length: int,
        context_length: int,
    ) -> torch.tensor:
        y_hat = self(X, pad_mask, prediction_length, context_length)
        y_hat = y_hat[:, -prediction_length:, :]
        return y_hat

    def free_running(
        self, data: Dict, context_length: int, prediction_length: int
    ) -> torch.tensor:
        x = data["X"]
        pad_mask = data["pad_mask"]
        y = x[:, 1:, : self.output_dim]
        y_hat = self(
            x[:, :-1, :],
            pad_mask[:, :-1],
            prediction_length,
            context_length,
        )
        y_hat = y_hat[pad_mask[:, 1:]]
        y = y[pad_mask[:, 1:]]
        return y_hat, y

    def teacher_forcing_output(self, data, prediction_length, context_length):
        """
        data: dict with keys "X" and "pad_mask"
        """
        x = data["X"]
        pad_mask = data["pad_mask"]
        y = x[:, 1:, : self.output_dim]
        y_hat = self(x[:, :-1, :], pad_mask[:, :-1], 1, x.shape[1] - 1)
        y_hat = y_hat[pad_mask[:, 1:]]
        y = y[pad_mask[:, 1:]]
        return y_hat, y

    def validate(self, batch, prediction_length, context_length):
        y = batch["X"][
            :, context_length : context_length + prediction_length, : self.output_dim
        ]
        y_hat = self.generate(
            batch["X"][:, :-1, ...],
            batch["pad_mask"][:, :-1],
            prediction_length=prediction_length,
            context_length=context_length,
        )
        return y_hat, y
