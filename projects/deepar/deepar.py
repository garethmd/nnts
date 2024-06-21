from typing import Dict, List

import torch
import torch.nn as nn

import nnts.models
from nnts.torch import models

FEAT_SCALE: str = "feat_scale"


import torch.distributions as td
from torch.distributions import AffineTransform, Distribution, TransformedDistribution


class AffineTransformed(TransformedDistribution):
    """
    Represents the distribution of an affinely transformed random variable.

    This is the distribution of ``Y = scale * X + loc``, where ``X`` is a
    random variable distributed according to ``base_distribution``.

    Parameters
    ----------
    base_distribution
        Original distribution
    loc
        Translation parameter of the affine transformation.
    scale
        Scaling parameter of the affine transformation.
    """

    def __init__(self, base_distribution: Distribution, loc=None, scale=None):
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc

        super().__init__(base_distribution, [AffineTransform(self.loc, self.scale)])

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale**2

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()


class DeepARPoint(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: nnts.models.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        lag_seq: List[int],
        scaled_features: List[str],
        context_length: int = None,
    ):
        super(DeepARPoint, self).__init__()
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
        self.context_length = context_length

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
            X[:, : self.context_length, :1], pad_mask[:, : self.context_length]
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


class DistrDeepAR(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: nnts.models.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        lag_seq: List[int],
        scaled_features: List[str],
        context_length: int = 30,
    ):
        super(DistrDeepAR, self).__init__()
        self.scaling_fn = scaling_fn
        self.output_dim = output_dim
        self.scaled_features = scaled_features
        self.n_scaled_features = len(scaled_features)
        self.n_scaled_features_excluding_feat_scale = len(
            [f for f in scaled_features if f != FEAT_SCALE]
        )
        self.lag_seq = torch.tensor(lag_seq) - 1
        self.decoder = models.unrolledlstm.UnrolledLSTMDecoder(
            params, self.n_scaled_features + len(self.lag_seq) + 1
        )
        self.distribution = Distribution(params.hidden_dim, output_dim)
        self.embbeder = nn.Embedding(1, self.output_dim)
        self.max_lag = max(lag_seq)
        self.context_length = context_length

    def create_lags(
        self, n_timesteps: int, past_target: torch.tensor, lag_seq: List[int]
    ) -> torch.tensor:
        lag_features = []
        for t in range(0, n_timesteps):
            lag_seq = lag_seq + 1
            lag_for_step = past_target.index_select(1, lag_seq)
            lag_features.append(lag_for_step)
        return torch.stack(lag_features, dim=1).flip(1)

    def _distr(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        H: int,
        context_length: int,
    ) -> torch.tensor:
        X = X.clone()
        if H > 0:
            X[:, -H:, 0] = 0.0
            past_target = X[:, :-H, 0].flip(1)
        else:
            past_target = X[:, :, 0].flip(1)
        X = X[:, self.max_lag :, ...]
        pad_mask = pad_mask[:, self.max_lag :]
        B, T, C = X.shape

        target_scale = self.scaling_fn(
            X[:, : self.context_length, :1], pad_mask[:, : self.context_length]
        )
        X[:, :, :1] = X[:, :, :1] / target_scale
        past_target = past_target / target_scale.squeeze(2)
        lags = torch.zeros(B, T, len(self.lag_seq))
        lags[:, :context_length, :] = self.create_lags(
            context_length, past_target, self.lag_seq
        )

        if FEAT_SCALE in self.scaled_features:
            X = torch.cat(
                [
                    X[:, :, :1],
                    lags,
                    self.embbeder(torch.zeros(B, 1).long()).expand(B, T, 1),
                    X[:, :, -1:],
                    torch.log(target_scale[:, :, :1]).expand(B, T, 1),
                    X[:, :, 1:-1],
                ],
                2,
            )
            C = C + 2

        # input = torch.zeros(B, T, C + lags.shape[2])
        # input[:, :, :C] = X
        # input[:, :context_length, C:] = lags.squeeze(1)
        input = X

        out, hidden = self.decoder(input[:, :context_length, :])
        params = self.distribution(out, target_scale=target_scale)
        distr = AffineTransformed(
            td.StudentT(params[0], params[1], params[2]), None, target_scale.squeeze(-1)
        )

        return distr

    def forward(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        H: int,
        context_length: int,
        n_samples: int = 100,
    ) -> torch.tensor:
        X = X.clone()
        if H > 0:
            X[:, -H:, 0] = 0.0
            past_target = X[:, :-H, 0].flip(1)
        else:
            past_target = X[:, :, 0].flip(1)
        X = X[:, self.max_lag :, ...]
        pad_mask = pad_mask[:, self.max_lag :]
        B, T, C = X.shape
        y_hat = torch.zeros(n_samples * B, T - 1, 1)

        target_scale = self.scaling_fn(
            X[:, :context_length, :1], pad_mask[:, :context_length]
        )
        X[:, :, :1] = X[:, :, :1] / target_scale
        past_target = past_target / target_scale.squeeze(2)
        lags = torch.zeros(B, T, len(self.lag_seq))
        lags[:, :context_length, :] = self.create_lags(
            context_length, past_target, self.lag_seq
        )

        if FEAT_SCALE in self.scaled_features:
            X = torch.cat(
                [
                    X[:, :, :1],
                    lags,
                    self.embbeder(torch.zeros(B, 1).long()).expand(B, T, 1),
                    X[:, :, -1:],
                    torch.log(target_scale[:, :, :1]).expand(B, T, 1),
                    X[:, :, 1:-1],
                ],
                2,
            )
            C = C + 2
        input = X[:, :-1, :]
        # input = torch.zeros(B, T, C + lags.shape[2])
        # input[:, :, :C] = X
        # input[:, :context_length, C:] = lags.squeeze(1)

        out, hidden = self.decoder(input[:, :context_length, :])

        # Expand the input and target_scale to N samples
        # out = out.reshape(n_samples * B, -1).unsqueeze(-1)  # N*B, T, C
        # out = out.repeat_interleave(n_samples, 0)  # N*B, T, C
        params = self.distribution(out, target_scale=target_scale)
        params = tuple(
            param.repeat_interleave(n_samples, 0) for param in params
        )  # N*B, T, C

        input = input.repeat_interleave(n_samples, 0)  # N*B, T, C
        past_target = past_target.repeat_interleave(n_samples, 0)  # N*B, T
        target_scale = target_scale.repeat_interleave(n_samples, 0)  # N*B, T, 1
        lags = lags.repeat_interleave(n_samples, 0)  # N*B, T, L
        hidden = tuple(
            state.repeat_interleave(n_samples, 1) for state in hidden
        )  # N*B, H
        distr = AffineTransformed(
            td.StudentT(params[0], params[1], params[2]), None, target_scale.squeeze(-1)
        )
        out = distr.sample()  # N*B, T

        # distr = self.distribution(out, target_scale=target_scale)
        # out = distr.sample((n_samples,))  # N, B, T

        y_hat[..., :context_length, 0] = out

        # free running for H steps
        for t in range(0, H - 1):
            out[:, -1:] = out[:, -1:] / target_scale.squeeze(2)
            past_target = torch.cat([out[:, -1:], past_target], 1)
            lags = self.create_lags(1, past_target, self.lag_seq)
            input[..., context_length + t, 0] = out[:, -1]
            input[..., context_length + t, 1 : 1 + lags.shape[2]] = lags.squeeze(1)

            # select the lag features and detach from the graph to prevent backpropagation
            out, hidden = self.decoder(
                input[:, context_length + t : context_length + t + 1, ...]
                .clone()
                .detach(),
                hidden,
            )
            params = self.distribution(out, target_scale=target_scale)
            distr = AffineTransformed(
                td.StudentT(params[0], params[1], params[2]),
                None,
                target_scale.squeeze(-1),
            )
            out = distr.sample()
            y_hat[..., context_length + t, :] = out

        # y_hat = y_hat * target_scale[:, :, : self.output_dim]
        y_hat = y_hat.reshape(B, n_samples, -1).permute(1, 0, 2)
        y_hat = y_hat.median(dim=0)[0]
        return y_hat.unsqueeze(-1)

    def generate(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        prediction_length: int,
        context_length: int,
    ) -> torch.tensor:
        assert pad_mask.sum() == pad_mask.numel()
        y_hat = self(X, pad_mask, prediction_length, context_length)
        y_hat = y_hat[:, -prediction_length:]
        return y_hat

    def free_running(
        self, data: Dict, prediction_length: int, context_length: int
    ) -> torch.tensor:
        x = data["X"]
        pad_mask = data["pad_mask"]
        assert pad_mask.sum() == pad_mask.numel()
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
        y_hat = self._distr(
            x[:, :-1, :], pad_mask[:, :-1], 0, context_length + prediction_length - 1
        )
        y = y[:, 1 - context_length - prediction_length :, ...]
        return y_hat, y

    def validate(self, batch, prediction_length, context_length):
        y = batch["X"][:, -prediction_length:, : self.output_dim]
        y_hat = self.generate(
            batch["X"],
            batch["pad_mask"],
            prediction_length=prediction_length,
            context_length=context_length,
        )
        return y_hat, y
