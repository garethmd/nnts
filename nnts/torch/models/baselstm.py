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

    def forward(self, x: torch.tensor, target_scale: torch.tensor):
        y_hat = self.main(x)
        if target_scale is not None:
            y_hat = y_hat * target_scale
        return y_hat


class BaseLSTMDecoder(nn.Module):
    def __init__(self, params: nnts.models.Hyperparams, output_dim: int):
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
        params: nnts.models.Hyperparams,
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
        target_scale = self.scaling_fn(X[:, :, :1], pad_mask)
        conts = X / target_scale
        embedded = torch.cat([conts, torch.log(target_scale).expand(B, T, 1)], 2)
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

    def teacher_forcing_output(self, data):
        """
        data: dict with keys "X" and "pad_mask"
        """
        x = data["X"]
        pad_mask = data["pad_mask"]
        y_hat = self(x[:, :-1, :], pad_mask[:, :-1])
        y = x[:, 1:, :]

        y_hat = y_hat[pad_mask[:, 1:]]
        y = y[pad_mask[:, 1:]]
        return y_hat, y
