import types

import pytest
import torch

import nnts.torch.models
import nnts.torch.trainers
import nnts.trainers


class FakeNet:
    def __init__(self):
        pass

    def forward(self, x, pad_mask):
        return x + 1

    def eval(self):
        pass

    def generate(self):
        pass


@pytest.fixture
def fake_net():
    net = FakeNet()
    net.generate = types.MethodType(nnts.torch.models.BaseLSTM.generate, net)
    return net


def test_early_stopper_should_stop_when_loss_is_nan():
    # Arrange
    stopper = nnts.torch.trainers.EarlyStopper(patience=1)
    state = nnts.trainers.TrainerState()
    state.valid_loss = float("nan")

    # Act
    result = stopper.early_stopping(state.valid_loss)

    # Assert
    assert result == True


def test_should_early_stop_after_patience_expired():
    # Arrange
    stopper = nnts.torch.trainers.EarlyStopper(patience=5)
    state = nnts.trainers.TrainerState()
    state.valid_loss = 1.0

    for i in range(5):
        stopper.early_stopping(state.valid_loss)

    # Act
    result = stopper.early_stopping(state.valid_loss)

    # Assert
    assert result == True


def test_should_generate_correctly(fake_net):
    # Arrange
    X = torch.arange(1, 10).float()
    X = X[None, :, None]
    pad_mask = torch.ones((1, 10)).long()
    prediction_length = 5
    context_length = 3

    y = X[:, context_length : context_length + prediction_length, ...]
    X = X[:, :context_length, ...]
    # Act
    result = fake_net.generate(X, pad_mask, prediction_length, context_length)
    # Assert
    assert result.shape == (1, 5, 1)
    assert torch.all(result[0, :, 0] == y[0, :, 0])


def test_should_validate_correctly(fake_net):
    # Arrange
    X = torch.arange(1, 10).float()
    X = X[None, :, None]
    pad_mask = torch.ones((1, 10)).long()
    prediction_length = 5
    context_length = 3
    batch = {"X": X, "pad_mask": pad_mask}

    # Act
    y_hat, y = nnts.torch.trainers.validate(
        fake_net, batch, prediction_length, context_length
    )

    # Assert
    assert y_hat.shape == (1, 5, 1)
    assert torch.all(
        y_hat[0, :, 0] == X[0, context_length : context_length + prediction_length, 0]
    )
    assert torch.all(
        y[0, :, 0] == X[0, context_length : context_length + prediction_length, 0]
    )


def test_should_eval_correctly(fake_net):
    # Arrange

    X = torch.arange(1, 10).float()
    X = X[None, :, None]
    pad_mask = torch.ones((1, 10)).long()
    prediction_length = 5
    context_length = 3
    batch = {"X": X, "pad_mask": pad_mask}

    def fake_dl():
        yield batch

    # Act
    y, y_hat = nnts.torch.trainers.eval(
        fake_net, fake_dl(), prediction_length, context_length
    )

    # Assert
    assert y.shape == (1, 5, 1)
    assert torch.all(y[0, :, 0] == torch.arange(4, 9).float())
    assert y_hat.shape == (1, 5, 1)
    assert torch.all(y_hat[0, :, 0] == torch.arange(4, 9).float())
