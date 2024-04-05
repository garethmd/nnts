import types

import pytest

import nnts.models.hyperparams as hyperparams
import nnts.models.trainers as trainers


class MockEvaluator(trainers.Evaluator):
    def __init__(self):
        self.evaluated = False

    def evaluate(self):
        self.evaluated = True


class MockEpochTrainer(trainers.EpochTrainer):
    def __init__(self, state, params):
        super().__init__(state, params)
        self.before_train_called = False
        self.before_train_epoch_called = False
        self.before_validate_epoch_called = False
        self.after_validate_epoch_called = False

    def _train_batch(self, i, batch):
        return 10

    def _validate_batch(self, i, batch):
        return 20

    def before_train(self, train_dl: trainers.Iterable) -> None:
        self.before_train_called = True

    def before_train_epoch(self) -> None:
        self.before_train_epoch_called = True

    def before_validate_epoch(self) -> None:
        self.before_validate_epoch_called = True

    def after_validate_epoch(self) -> None:
        self.after_validate_epoch_called = True

    def create_evaluator(self) -> None:
        return MockEvaluator()


@pytest.fixture
def evaluator():
    return MockEvaluator()


@pytest.fixture
def params():
    return hyperparams.Hyperparams(epochs=5)


@pytest.fixture
def state():
    return trainers.TrainerState()


def test_should_initialise_with_evaluator_and_state(state, params):
    trainer = MockEpochTrainer(state, params)
    assert trainer.state == state


def test_should_train_epoch_batch(state, params):
    trainer = MockEpochTrainer(state, params)
    L = trainer._train_batch(0, None)
    assert L == 10


def test_should_validate_epoch_batch(state, params):
    trainer = MockEpochTrainer(state, params)
    L = trainer._validate_batch(0, None)
    assert L == 20


def test_should_train_epoch(state, params):
    trainer = MockEpochTrainer(state, params)
    train_dl = [1, 2, 3, 4, 5]
    L = trainer._train_epoch(train_dl)
    assert L == 50 / len(train_dl)


def test_should_validate_epoch(state, params):
    trainer = MockEpochTrainer(state, params)
    valid_dl = [1, 2, 3, 4, 5]
    L = trainer._validate_epoch(valid_dl)
    assert L == 100 / len(valid_dl)


def test_should_train(state, params):
    trainer = MockEpochTrainer(state, params)
    train_dl = [1, 2, 3, 4, 5]
    valid_dl = [1, 2, 3, 4, 5]
    evaluator = trainer.train(train_dl, valid_dl)
    assert isinstance(evaluator, trainers.Evaluator)
    assert state.epoch == 5
    assert state.train_loss == 50 / len(train_dl)
    assert state.valid_loss == 100 / len(valid_dl)
    assert not state.stop


def test_should_call_before_train(state, params):
    trainer = MockEpochTrainer(state, params)
    train_dl = [1, 2, 3, 4, 5]
    valid_dl = [1, 2, 3, 4, 5]
    trainer.train(train_dl, valid_dl)
    assert trainer.before_train_called


def test_should_call_before_train_epoch(state, params):
    trainer = MockEpochTrainer(state, params)
    train_dl = [1, 2, 3, 4, 5]
    trainer._train_epoch(train_dl)
    assert trainer.before_train_epoch_called


def test_should_call_before_validate_epoch(state, params):
    trainer = MockEpochTrainer(state, params)
    valid_dl = [1, 2, 3, 4, 5]
    trainer._validate_epoch(valid_dl)
    assert trainer.before_validate_epoch_called


def test_should_call_after_validate_epoch(state, params):
    trainer = MockEpochTrainer(state, params)
    valid_dl = [1, 2, 3, 4, 5]
    trainer._validate_epoch(valid_dl)
    assert trainer.after_validate_epoch_called


def test_should_stop_train_on_state_stop(state, params):
    trainer = MockEpochTrainer(state, params)

    # override the existing after validate epoch method
    def stop_training(self):
        if self.state.epoch == 2:
            self.state.stop = True

    trainer.after_validate_epoch = types.MethodType(stop_training, trainer)

    train_dl = [1, 2, 3, 4, 5]
    valid_dl = [1, 2, 3, 4, 5]
    evaluator = trainer.train(train_dl, valid_dl)
    assert trainer.state.stop == True
    assert state.epoch == 2
