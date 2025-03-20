from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from nnts.utils import Scheduler, TrainingMethod

from ..datasets import PaddedData


@dataclass
class Hyperparams:
    """Class for keeping track of training and model params."""

    optimizer: callable = torch.optim.Adam
    loss_fn: callable = F.l1_loss
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 0.01
    epochs: int = 100
    patience: int = 20
    early_stopper_patience: int = 30
    batches_per_epoch: int = 50
    weight_decay: float = 0.0
    training_method: TrainingMethod = TrainingMethod.DMS
    scheduler: Scheduler = Scheduler.REDUCE_LR_ON_PLATEAU
    model_file_path: str = f"logs"
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    task_name: str = "short_term_forecast"
    label_len: int = 0
    down_sampling_window: int = 2
    down_sampling_layers: int = 1
    d_model: int = 32
    d_ff: int = 32
    e_layers: int = 4
    factor: int = 3
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    down_sampling_method: str = "avg"
    channel_independence: int = 1
    moving_avg: int = 25
    embed: str = "timeF"
    freq: str = None  # h, d, w, m
    use_future_temporal_feature: int = 0
    decomp_method: str = "moving_avg"
    use_norm: int = 1
    num_class: int = None


def get_mutlivariate_params():
    params = Hyperparams(
        optimizer=torch.optim.Adam,
        loss_fn=torch.nn.MSELoss(),
        batch_size=32,
        batches_per_epoch=250,
        training_method=TrainingMethod.DMS,
        model_file_path="logs",
        epochs=10,
        scheduler=Scheduler.STEP_LR,
        lr=0.001,
        weight_decay=0.0,
        blocks=4,
        dropout=0.9,
        num_blocks=4,
        ff_dim=256,
        output_channels=7,
    )
    return params


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        # freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        freq_map = {
            "ME": 1,
            "QE": 1,
            "YE": 1,
            "1D": 1,
            "1H": 4,
            "1W": 2,
            "30min": 5,
            "10min": 5,
            "h": 4,
        }
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, seq_len: int, pred_len: int, params: Hyperparams):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (params.down_sampling_window**i),
                        seq_len // (params.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (params.down_sampling_window ** (i + 1)),
                        seq_len // (params.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(params.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, seq_len: int, pred_len: int, params: Hyperparams):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (params.down_sampling_window ** (i + 1)),
                        seq_len // (params.down_sampling_window**i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (params.down_sampling_window**i),
                        seq_len // (params.down_sampling_window**i),
                    ),
                )
                for i in reversed(range(params.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, params: Hyperparams):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = params.down_sampling_window

        self.layer_norm = nn.LayerNorm(params.d_model)
        self.dropout = nn.Dropout(params.dropout)
        self.channel_independence = params.channel_independence

        if params.decomp_method == "moving_avg":
            self.decompsition = series_decomp(params.moving_avg)
        elif params.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(params.top_k)
        else:
            raise ValueError("decompsition is error")

        if params.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=params.d_model, out_features=params.d_ff),
                nn.GELU(),
                nn.Linear(in_features=params.d_ff, out_features=params.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, pred_len, params
        )

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(seq_len, pred_len, params)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=params.d_model, out_features=params.d_ff),
            nn.GELU(),
            nn.Linear(in_features=params.d_ff, out_features=params.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):

    def __init__(
        self,
        sequence_length,
        prediction_length,
        params: Hyperparams,
    ):
        super(TimeMixer, self).__init__()
        self.params = params
        self.task_name = params.task_name
        self.seq_len = sequence_length
        self.label_len = params.label_len
        self.pred_len = prediction_length
        self.down_sampling_window = params.down_sampling_window
        self.channel_independence = params.channel_independence
        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(self.seq_len, self.pred_len, params)
                for _ in range(params.e_layers)
            ]
        )

        self.preprocess = series_decomp(params.moving_avg)
        self.enc_in = params.enc_in
        self.use_future_temporal_feature = params.use_future_temporal_feature

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, params.d_model, params.embed, params.freq, params.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                params.enc_in,
                params.d_model,
                params.embed,
                params.freq,
                params.dropout,
            )

        self.layer = params.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                RevIN(
                    self.params.enc_in,
                    affine=True,
                )
                for i in range(params.down_sampling_layers + 1)
            ]
        )

        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.seq_len // (params.down_sampling_window**i),
                        self.pred_len,
                    )
                    for i in range(params.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(params.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    params.d_model, params.c_out, bias=True
                )

                self.out_res_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            self.seq_len // (params.down_sampling_window**i),
                            self.seq_len // (params.down_sampling_window**i),
                        )
                        for i in range(params.down_sampling_layers + 1)
                    ]
                )

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            self.seq_len // (params.down_sampling_window**i),
                            self.pred_len,
                        )
                        for i in range(params.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(params.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    params.d_model, params.c_out, bias=True
                )
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(params.dropout)
            self.projection = nn.Linear(params.d_model * self.seq_len, params.num_class)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.params.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(
                self.params.down_sampling_window, return_indices=False
            )
        elif self.params.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.params.down_sampling_window)
        elif self.params.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.params.enc_in,
                out_channels=self.params.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.params.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.params.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, :: self.params.down_sampling_window, :]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[
                    :, :: self.params.down_sampling_window, :
                ]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc):

        # if self.use_future_temporal_feature:
        #     if self.channel_independence == 1:
        #         B, T, N = x_enc.size()
        #         x_mark_dec = x_mark_dec.repeat(N, 1, 1)
        #         self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        #     else:
        #         self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(
                range(len(x_enc)),
                x_enc,
            ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = (
                    dec_out.reshape(B, self.params.c_out, self.pred_len)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(
                range(len(x_list[0])), enc_out_list, x_list[1]
            ):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(
            range(len(x_enc)),
            x_enc,
        ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, "norm")
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = (
            dec_out.reshape(B, self.params.c_out, -1).permute(0, 2, 1).contiguous()
        )

        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(
                range(len(x_enc)),
                x_enc,
            ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = (
            dec_out.reshape(B, self.params.c_out, -1).permute(0, 2, 1).contiguous()
        )

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, mask=None):
        x_mark_enc = None
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Other tasks implemented yet")

    def train_output(
        self, batch: PaddedData, *args, **kwargs
    ) -> Tuple[torch.tensor, torch.tensor]:
        x = batch.data[:, : self.seq_len, :]
        pad_mask = batch.data[:, : self.seq_len]
        y_hat = self(x, pad_mask)
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


if __name__ == "__main__":
    m = TimeMixer(10, 5, 2, output_channels=4)
    x = torch.randn(3, 10, 2)
    y = m(x)
