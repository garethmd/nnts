import types

import pytest
import torch

import nnts.datasets
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


class TestTorchForecaster:

    def test_should_forecast_batch(self, fake_net):
        forecaster = nnts.torch.trainers.TorchForecaster(fake_net)
        # Arrange
        X = torch.arange(1, 10).float()
        X = X[None, :, None]
        pad_mask = torch.ones((1, 10)).long()
        prediction_length = 5
        context_length = 3

        y = X[:, context_length : context_length + prediction_length, ...]
        X = X[:, :context_length, ...]

        # Act
        result = forecaster.forecast_batch(
            {"X": X, "pad_mask": pad_mask}, prediction_length, context_length
        )

        # Assert
        assert result.shape == (1, 5, 1)

    def test_should_forecast(self, fake_net):
        forecaster = nnts.torch.trainers.TorchForecaster(fake_net)
        # Arrange
        X = torch.arange(1, 10).float()
        X = X[None, :, None]
        pad_mask = torch.ones((1, 10)).long()

        prediction_length = 5
        context_length = 3

        y = X[:, context_length : context_length + prediction_length, ...]
        X = X[:, :context_length, ...]
        fake_dl = [{"X": X, "pad_mask": pad_mask}, {"X": X, "pad_mask": pad_mask}]

        # Act
        result = forecaster.forecast(fake_dl, prediction_length, context_length)

        # Assert
        assert result.shape == (2, 5, 1)


import numpy as np

import nnts.utils


class TestValidationTorchEpochTrainer:

    @pytest.mark.parametrize(
        "scheduler, expected",
        [
            (nnts.utils.Scheduler.ONE_CYCLE, torch.optim.lr_scheduler.OneCycleLR),
            (
                nnts.utils.Scheduler.REDUCE_LR_ON_PLATEAU,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ),
        ],
    )
    def test_should_create_correct_scheduler(self, scheduler, expected):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            scheduler=scheduler,
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(
            fake_net, params, metadata
        )
        trainer.before_train(None)

        # Assert
        assert isinstance(trainer.scheduler, expected)

    def test_should_initialise_best_loss(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(
            fake_net, params, metadata
        )
        trainer.before_train(None)

        # Assert
        assert trainer.best_loss is np.inf
        # Assert
        assert trainer.best_loss is np.inf

    def test_should_create_early_stopper(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
            early_stopper_patience=5,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(
            fake_net, params, metadata
        )
        trainer.before_train(None)

        # Assert
        assert trainer.early_stopper is not None

    def test_should_not_create_early_stopper(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
            early_stopper_patience=None,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(
            fake_net, params, metadata
        )
        trainer.before_train(None)

        # Assert
        assert trainer.early_stopper is None

    def test_should_set_state_stop_with_early_stopper(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
            early_stopper_patience=1,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(
            fake_net, params, metadata
        )
        trainer.before_train(None)

        trainer.state.best_loss = np.inf
        trainer.state.valid_loss = 99
        trainer.after_validate_epoch()

        assert trainer.state.stop is False

        trainer.state.valid_loss = 99
        trainer.after_validate_epoch()
        # Assert

        assert trainer.state.stop is True

    def test_should_set_best_loss(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
            early_stopper_patience=1,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(
            fake_net, params, metadata
        )
        trainer.before_train(None)

        trainer.state.best_loss = np.inf
        trainer.state.valid_loss = 99
        trainer.after_validate_epoch()

        assert trainer.state.best_loss == 99

    def test_should_return_evaluator(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.ValidationTorchEpochTrainer(
            fake_net, params, metadata
        )
        evaluator = trainer.create_evaluator()

        assert isinstance(evaluator, nnts.torch.trainers.TorchEvaluator)


class TestTorchEpochTrainer:

    @pytest.mark.parametrize(
        "scheduler, expected",
        [
            (nnts.utils.Scheduler.ONE_CYCLE, torch.optim.lr_scheduler.OneCycleLR),
            (
                nnts.utils.Scheduler.REDUCE_LR_ON_PLATEAU,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ),
        ],
    )
    def test_should_create_correct_scheduler(self, scheduler, expected):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            scheduler=scheduler,
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.TorchEpochTrainer(fake_net, params, metadata)
        trainer.before_train(None)

        # Assert
        assert isinstance(trainer.scheduler, expected)

    def test_should_initialise_best_loss(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.TorchEpochTrainer(fake_net, params, metadata)
        trainer.before_train(None)

        # Assert
        assert trainer.best_loss is np.inf

    def test_should_set_best_loss(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.TorchEpochTrainer(fake_net, params, metadata)
        trainer.before_train(None)

        trainer.state.best_loss = np.inf
        trainer.state.train_loss = 99
        trainer.after_train_epoch()

        assert trainer.state.best_loss == 99

    def test_should_return_evaluator(self):
        fake_net = torch.nn.Linear(10, 1)

        params = nnts.utils.Hyperparams(
            optimizer=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss,
        )
        metadata = nnts.datasets.Metadata(
            filename="hospital.tsf",
            dataset="hospital",
            context_length=3,
            prediction_length=5,
            freq="ME",
            seasonality=1,
        )

        # Act
        trainer = nnts.torch.trainers.TorchEpochTrainer(fake_net, params, metadata)
        evaluator = trainer.create_evaluator()

        assert isinstance(evaluator, nnts.torch.trainers.TorchEvaluator)
