from typing import Dict, List

import torch
import torch.nn as nn

import nnts.models

from .. import models

FEAT_SCALE: str = "feat_scale"


class DeepAR(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: nnts.models.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        lag_seq: List[int],
        scaled_features: List[str],
    ):
        super(DeepAR, self).__init__()
        self.scaling_fn = scaling_fn
        self.output_dim = output_dim
        self.scaled_features = scaled_features
        self.n_scaled_features = len(scaled_features)
        self.n_scaled_features_excluding_feat_scale = len(
            [f for f in scaled_features if f != FEAT_SCALE]
        )
        self.lag_seq = torch.tensor(lag_seq) - 1
        self.decoder = models.unrolledlstm.UnrolledLSTMDecoder(
            params, self.n_scaled_features + len(self.lag_seq)
        )
        self.distribution = Distribution(params.hidden_dim, output_dim)
        self.max_lag = max(lag_seq)

    def create_lags(
        self, n_timesteps: int, past_target: torch.tensor, lag_seq: List[int]
    ) -> torch.tensor:
        lag_features = []
        for t in range(0, n_timesteps):
            lag_seq = lag_seq + 1
            lag_for_step = past_target.index_select(1, lag_seq)
            lag_features.append(lag_for_step)
        return torch.stack(lag_features, dim=1).flip(1)

    def forward(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        H: int,
        context_length: int,
    ) -> torch.tensor:
        X = X.clone()
        if H > 0:
            h = H - 1
            X[:, -h:, 0] = 0.0
            past_target = X[:, :-h, 0].flip(1)
        else:
            past_target = X[:, :, 0].flip(1)
        X = X[:, self.max_lag :, ...]
        pad_mask = pad_mask[:, self.max_lag :]
        B, T, C = X.shape
        y_hat = torch.zeros(B, T, self.output_dim)

        target_scale = self.scaling_fn(
            X[:, :context_length, :1], pad_mask[:, :context_length]
        )
        X[:, :, :1] = X[:, :, :1] / target_scale
        past_target = past_target / target_scale.squeeze(2)
        lags = self.create_lags(context_length, past_target, self.lag_seq)

        if FEAT_SCALE in self.scaled_features:
            X = torch.cat(
                [
                    X,
                    torch.log(target_scale[:, :, :1]).expand(B, T, 1),
                ],
                2,
            )
            C = C + 1

        input = torch.zeros(B, T, C + lags.shape[2])
        input[:, :, :C] = X
        input[:, :context_length, C:] = lags.squeeze(1)

        out, hidden = self.decoder(input[:, :context_length, :])
        out = self.distribution(out, target_scale=None)
        y_hat[:, :context_length, :] = out

        # free running for H steps
        for t in range(0, H - 1):
            past_target = torch.cat([out[:, -1:, 0], past_target], 1)
            lags = self.create_lags(1, past_target, self.lag_seq)
            input[:, context_length + t, 0] = out[:, -1, 0]
            input[:, context_length + t, -lags.shape[2] :] = lags.squeeze(1)

            # select the lag features and detach from the graph to prevent backpropagation
            out, hidden = self.decoder(
                input[:, context_length + t : context_length + t + 1, ...]
                .clone()
                .detach(),
                hidden,
            )
            out = self.distribution(out, target_scale=None)
            y_hat[:, context_length + t, :] = out[:, -1, :]

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
        self, data: Dict, prediction_length: int, context_length: int
    ) -> torch.tensor:
        x = data["X"]
        pad_mask = data["pad_mask"]
        y = x[:, 1 - context_length - prediction_length :, : self.output_dim]
        y_hat = self(
            x[:, :-1, :],
            pad_mask[:, :-1],
            prediction_length,
            context_length,
        )
        return y_hat, y

    def teacher_forcing_output(self, data, prediction_length, context_length):
        """
        data: dict with keys "X" and "pad_mask"
        """
        x = data["X"]
        pad_mask = data["pad_mask"]
        y = x[:, 1:, : self.output_dim]
        y_hat = self(
            x[:, :-1, :], pad_mask[:, :-1], 0, context_length + prediction_length - 1
        )
        y = y[:, 1 - context_length - prediction_length :, ...]
        return y_hat, y

    def validate(self, batch, prediction_length, context_length):
        y = batch["X"][:, -prediction_length:, : self.output_dim]
        y_hat = self.generate(
            batch["X"][:, :-1, ...],
            batch["pad_mask"][:, :-1],
            prediction_length=prediction_length,
            context_length=context_length,
        )
        return y_hat, y
