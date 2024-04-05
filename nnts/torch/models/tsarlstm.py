import torch
import torch.nn as nn

import nnts.models


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

    def forward(self, x, target_scale):
        y_hat = self.main(x)
        y_hat = y_hat * target_scale
        return y_hat


class TsarLSTMDecoder(nn.Module):
    def __init__(self, params, output_dim):
        super(TsarLSTMDecoder, self).__init__()
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

    def init_hidden_zeros(self, batch_size):
        if self.rnn_type == "gru":
            hidden = torch.zeros(
                self.params.n_layers, batch_size, self.params.hidden_dim
            )
        else:
            h0 = torch.zeros(self.params.n_layers, batch_size, self.params.hidden_dim)
            c0 = torch.zeros(self.params.n_layers, batch_size, self.params.hidden_dim)
            hidden = (h0, c0)
        return hidden

    def forward(self, X, hidden=None):
        B, T, C = X.shape
        if hidden is None:
            hidden = self.init_hidden_zeros(B)
        return self.rnn(X, hidden)


class TsarLSTM(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: nnts.models.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
    ):
        super(TsarLSTM, self).__init__()
        self.encoder = TsarLSTMDecoder(params, output_dim)
        self.distribution = Distribution(params.hidden_dim, output_dim)
        self.scaling_fn = scaling_fn

    def forward(self, X, mask):
        X = X.clone()
        B, T, C = X.shape
        target_scale = self.scaling_fn(X[:, :, :1], mask)
        conts = X / target_scale
        embedded = torch.cat([conts, torch.log(target_scale).expand(B, T, 1)], 2)
        out, _ = self.encoder(embedded)
        distr = self.distribution(out, target_scale=target_scale)
        return distr
