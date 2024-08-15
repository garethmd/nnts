import torch
import torch.nn as nn

from nnts import utils

from .. import datasets


class SegLSTMDecoder(nn.Module):

    def __init__(self, params: utils.Hyperparams, input_dim: int, hidden_dim: int):
        super(SegLSTMDecoder, self).__init__()
        self.params = params
        self.dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(
            self.dim,
            self.hidden_dim,
            params.n_layers,
            dropout=params.dropout,
            batch_first=True,
        )
        self.rnn_type = params.rnn_type

    def init_hidden_zeros(self, batch_size):
        if self.rnn_type == "gru":
            hidden = torch.zeros(
                self.params.n_layers,
                batch_size,
                self.hidden_dim,
            )
        else:
            h0 = torch.zeros(
                self.params.n_layers,
                batch_size,
                self.hidden_dim,
            )
            c0 = torch.zeros(
                self.params.n_layers,
                batch_size,
                self.hidden_dim,
            )
            hidden = (h0, c0)
        return hidden

    def forward(self, X: torch.tensor, hidden: bool = None):
        """
        X: B, T, T_dim, C
         - B: batch size
         - T: time steps
         - T_dim: segment length
         - C: covariate length + 1
        """
        B, T, T_dim, C = X.shape
        X = X.reshape(B, T, -1)
        if hidden is None:
            hidden = self.init_hidden_zeros(B)
        out, hidden = self.rnn(X, hidden)
        return out


class SegLSTM(nn.Module):

    def __init__(
        self,
        Distribution: nn.Module,
        params: utils.Hyperparams,
        scaling_fn: callable,
        output_dim: int,
        segment_length: int,
    ):
        """
        Initialize the SegLSTM model.

        Args:
            Distribution (nn.Module): The distribution module used for generating output.
            params (utils.Hyperparams): Hyperparameters for the model.
            scaling_fn (callable): A scaling function applied to the output.
            output_dim (int): The dimension of the output. (ie y + covariates).
            segment_length (int): The segment length (eg seasonality).

        """
        super(SegLSTM, self).__init__()
        self.decoder = SegLSTMDecoder(params, segment_length * output_dim, 40)
        self.distribution = Distribution(40, segment_length * output_dim)
        self.output_dim = output_dim
        self.scaling_fn = scaling_fn
        self.segment_length = segment_length

    def forward(self, X: torch.tensor, pad_mask: torch.tensor) -> torch.tensor:

        X = X.clone()
        B, T, C = X.shape

        if self.scaling_fn is None:
            target_scale = None
            x = X
        else:
            target_scale = self.scaling_fn(X, pad_mask)
            x = X / target_scale

        x = x.reshape(B, -1, self.segment_length, C)  # B, T/12, 12, C
        B, T, T_dim, C = x.shape

        x = self.decoder(x)
        x = self.distribution(x, None)
        x = x.reshape(B, T, T_dim, C)
        x = x * target_scale.unsqueeze(2)
        return x

    def generate(
        self,
        X: torch.tensor,
        pad_mask: torch.tensor,
        prediction_length: int,
        context_length: int,
    ) -> torch.tensor:
        pred_list = []
        while True:
            pad_mask = pad_mask[:, -context_length + 1 :]
            preds = self.forward(X[:, -context_length + 1 :, :], pad_mask)
            # focus only on the last time step
            preds = preds[:, -1:, -1, :]
            pred_list.append(preds)

            if len(pred_list) >= prediction_length:
                break
            y_hat = preds.detach().clone()
            X = torch.cat((X, y_hat), dim=1)  # (B, T+1)
            pad_mask = torch.cat((pad_mask, torch.ones_like(preds[:, :, 0])), dim=1)
        return torch.cat(pred_list, 1)

    def teacher_forcing_output(self, batch: datasets.PaddedData, *args, **kwargs):
        x = batch.data
        B, T, C = x.shape
        pad_mask = batch.pad_mask
        y_hat = self(x[:, :-1, :], pad_mask[:, :-1])
        y_hat = y_hat[:, :, -1:, ...]  # B, T, S, C
        # y_hat = y_hat[:, :, :, ...]  # B, T, S, C

        y = x[:, 1:, :]
        y = y.reshape(B, -1, self.segment_length, C)
        y = y[:, :, -1:, ...]  # B, T, S, C
        # y = y[:, :, :, ...]  # B, T, S, C
        return y_hat, y
