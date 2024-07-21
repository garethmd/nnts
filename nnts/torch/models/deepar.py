from typing import Dict, List

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import AffineTransform, Distribution, TransformedDistribution

import nnts.data
import nnts.torch.preprocessing
from nnts import utils

from .. import datasets, models

FEAT_SCALE: str = "feat_scale"


def distr_nll(distr: td.Distribution, target: torch.Tensor) -> torch.Tensor:
    nll = -distr.log_prob(target)
    return nll.mean()


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


class StudentTHead(nn.Module):
    """
    This model outputs a studentT distribution.
    """

    PARAMS = 3

    def __init__(self, hidden_size: int, output_size: int):
        super(StudentTHead, self).__init__()

        self.main = nn.ModuleList(
            [nn.Linear(hidden_size, output_size) for _ in range(StudentTHead.PARAMS)]
        )

    def forward(self, x: torch.tensor, target_scale: torch.tensor):
        df, loc, scale = tuple(self.main[i](x) for i in range(StudentTHead.PARAMS))
        df = 2.0 + F.softplus(df)
        epsilon = torch.finfo(scale.dtype).eps
        scale = F.softplus(scale).clamp_min(epsilon)
        return df, loc, scale


class DeepARMixin:
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

    def validate(self, batch: datasets.PaddedData, prediction_length, context_length):
        y = batch.data[:, -prediction_length:, : self.output_dim]
        y_hat = self.generate(
            batch.data,
            batch.pad_mask,
            prediction_length=prediction_length,
            context_length=context_length,
        )
        return y_hat, y

    def free_running(
        self, batch: datasets.PaddedData, prediction_length: int, context_length: int
    ) -> torch.tensor:
        x = batch.data
        pad_mask = batch.pad_mask
        y = x[:, 1 - context_length - prediction_length :, : self.output_dim]
        y_hat = self(
            x,
            pad_mask,
            prediction_length,
            context_length,
        )
        return y_hat, y

    def prep_input(self, X, context_length, past_target, target_scale):
        B, T, _ = X.shape
        features = []
        if self.lag_processor is not None:
            lags = torch.zeros(B, T, len(self.lag_processor))
            lags[:, :context_length, :] = self.lag_processor.create(
                context_length, past_target
            )
            features.append(lags)

        if FEAT_SCALE in self.scaled_features:
            features.append(
                torch.log(target_scale[:, :, :1]).expand(B, T, 1),
            )

        if self.cat_idx is not None:
            X[..., self.cat_idx] = self.embedder(X[..., self.cat_idx].long()).squeeze(
                -1
            )

        if self.seq_cat_idx is not None:
            features.append(self.seq_embedder(X[..., self.seq_cat_idx].long()))
            X = torch.cat(
                [X[..., : self.seq_cat_idx], X[..., self.seq_cat_idx + 1 :]], 2
            )

        input = torch.cat([X[:, :, :1]] + features + [X[:, :, 1:]], 2)
        return lags, input


class DeepARPoint(nn.Module, DeepARMixin):

    def __init__(
        self,
        Distribution: nn.Module,
        params: utils.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        lag_processor: nnts.torch.preprocessing.LagProcessor,
        scaled_features: List[str],
        context_length: int = 15,
        cat_idx: int = None,
        seq_cat_idx=None,
        emb_dim=5,
    ):
        super(DeepARPoint, self).__init__()
        self.scaling_fn = scaling_fn
        self.output_dim = output_dim
        self.scaled_features = scaled_features
        self.n_scaled_features = len(scaled_features)
        self.n_scaled_features_excluding_feat_scale = len(
            [f for f in scaled_features if f != FEAT_SCALE]
        )
        self.lag_processor = lag_processor
        self.decoder = models.unrolledlstm.UnrolledLSTMDecoder(
            params, self.n_scaled_features + len(self.lag_processor) + emb_dim - 1
        )
        self.distribution = Distribution(params.hidden_dim, output_dim)
        self.context_length = context_length
        self.cat_idx = cat_idx
        if cat_idx is not None:
            self.embedder = nn.Embedding(1, 1)

        self.seq_cat_idx = seq_cat_idx
        if seq_cat_idx is not None:
            self.seq_embedder = nn.Embedding(12, emb_dim)

    def forward(
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
        X = X[:, self.lag_processor.max() :, ...]
        pad_mask = pad_mask[:, self.lag_processor.max() :]
        B, T, C = X.shape

        target_scale = self.scaling_fn(
            X[:, : self.context_length, :1], pad_mask[:, : self.context_length]
        )
        X[:, :, :1] = X[:, :, :1] / target_scale
        past_target = past_target / target_scale.squeeze(2)
        _, input = self.prep_input(X, context_length, past_target, target_scale)

        if H > 0:
            input = input[:, :-1, :]
            y_hat = torch.zeros(B, T - 1, self.output_dim)
        else:
            y_hat = torch.zeros(B, T, self.output_dim)

        out, hidden = self.decoder(input[:, :context_length, :])
        out = self.distribution(out, target_scale=None)

        y_hat[:, :context_length, :] = out

        # free running for H steps
        for t in range(0, H - 1):
            past_target = torch.cat([out[:, -1:, 0], past_target], 1)
            lags = self.lag_processor.create(1, past_target)
            input[..., context_length + t, 0] = out[:, -1, 0]
            input[..., context_length + t, 1 : 1 + lags.shape[2]] = lags.squeeze(1)

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

    def teacher_forcing_output(
        self, batch: datasets.PaddedData, prediction_length, context_length
    ):
        """
        data: dict with keys "X" and "pad_mask"
        """
        x = batch.data
        pad_mask = batch.pad_mask
        y = x[:, 1:, : self.output_dim]
        y_hat = self(
            x[:, :-1, :], pad_mask[:, :-1], 0, context_length + prediction_length - 1
        )
        y = y[:, 1 - context_length - prediction_length :, ...]
        return y_hat, y


class DistrDeepAR(nn.Module, DeepARMixin):

    def __init__(
        self,
        Distribution: nn.Module,
        params: utils.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        lag_processor: nnts.torch.preprocessing.LagProcessor,
        scaled_features: List[str],
        context_length: int = 15,
        cat_idx: int = None,
        seq_cat_idx=None,
        emb_dim=1,
    ):
        super(DistrDeepAR, self).__init__()

        self.scaling_fn = scaling_fn
        self.output_dim = output_dim
        self.scaled_features = scaled_features
        self.n_scaled_features = len(scaled_features)
        self.n_scaled_features_excluding_feat_scale = len(
            [f for f in scaled_features if f != FEAT_SCALE]
        )
        self.lag_processor = lag_processor
        self.decoder = models.unrolledlstm.UnrolledLSTMDecoder(
            params, self.n_scaled_features + len(self.lag_processor) + emb_dim - 1
        )
        self.distribution = Distribution(params.hidden_dim, output_dim)

        self.context_length = context_length
        self.cat_idx = cat_idx
        if cat_idx is not None:
            self.embedder = nn.Embedding(1, 1)

        self.seq_cat_idx = seq_cat_idx
        if seq_cat_idx is not None:
            self.seq_embedder = nn.Embedding(12, emb_dim)

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
        X = X[:, self.lag_processor.max() :, ...]
        pad_mask = pad_mask[:, self.lag_processor.max() :]

        target_scale = self.scaling_fn(
            X[:, : self.context_length, :1], pad_mask[:, : self.context_length]
        )
        X[:, :, :1] = X[:, :, :1] / target_scale
        past_target = past_target / target_scale.squeeze(2)
        _, input = self.prep_input(X, context_length, past_target, target_scale)

        if H > 0:
            input = input[:, :-1, :]

        out, _ = self.decoder(input[:, :context_length, :])
        params = self.distribution(out, target_scale=target_scale)
        distr = AffineTransformed(
            td.StudentT(params[0], params[1], params[2]), None, target_scale
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
        X = X[:, self.lag_processor.max() :, ...]
        pad_mask = pad_mask[:, self.lag_processor.max() :]
        B, T, C = X.shape

        target_scale = self.scaling_fn(
            X[:, :context_length, :1], pad_mask[:, :context_length]
        )
        X[:, :, :1] = X[:, :, :1] / target_scale
        past_target = past_target / target_scale.squeeze(2)

        lags, input = self.prep_input(X, context_length, past_target, target_scale)
        input = input[:, :-1, :]

        out, hidden = self.decoder(input[:, :context_length, :])

        # Expand the input and target_scale to N samples
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
            td.StudentT(params[0], params[1], params[2]), None, target_scale
        )
        out = distr.sample()  # N*B, T

        y_hat = torch.zeros(n_samples * B, T - 1, 1)
        y_hat[..., :context_length, :] = out

        # free running for H steps
        for t in range(0, H - 1):
            out[:, -1:, 0] = out[:, -1:, 0] / target_scale.squeeze(2)
            past_target = torch.cat([out[:, -1:, 0], past_target], 1)
            lags = self.lag_processor.create(1, past_target)
            input[..., context_length + t, 0] = out[:, -1, 0]
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
                target_scale,
            )
            out = distr.sample()
            y_hat[..., context_length + t, :] = out[:, -1, :]

        y_hat = y_hat.reshape(B, n_samples, -1).permute(1, 0, 2)
        y_hat = y_hat.median(dim=0)[0]
        return y_hat.unsqueeze(-1)

    def teacher_forcing_output(
        self, batch: datasets.PaddedData, prediction_length, context_length
    ):
        """
        data: dict with keys "X" and "pad_mask"
        """
        x = batch.data
        pad_mask = batch.pad_mask
        y = x[:, 1:, : self.output_dim]
        y_hat = self._distr(
            x[:, :-1, :], pad_mask[:, :-1], 0, context_length + prediction_length - 1
        )
        y = y[:, 1 - context_length - prediction_length :, ...]
        return y_hat, y
