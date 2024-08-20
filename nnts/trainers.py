from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

import numpy as np
from pydantic import BaseModel, PositiveInt

import nnts.events

from .utils import Hyperparams


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self) -> None:
        pass


class Forecaster(ABC):
    @abstractmethod
    def forecast(self, data, h: int) -> Any:
        pass


class Trainer(ABC):

    @abstractmethod
    def train(self) -> Evaluator:
        pass

    @abstractmethod
    def create_evaluator(self) -> None:
        pass


class TrainerState(BaseModel):
    epoch: int = 0
    stop: bool = False
    train_loss: float = 0
    valid_loss: float = 0
    best_loss: float = np.inf


class TrainerEvent:
    def __init__(self, state: TrainerState):
        self.state = state


class EpochValidateComplete(TrainerEvent):
    def __init__(self, state: TrainerState):
        super().__init__(state)


class EpochTrainComplete(TrainerEvent):
    def __init__(self, state: TrainerState):
        super().__init__(state)


class EpochBestModel:
    def __init__(self, path: str):
        self.path = path


class EpochTrainer(Trainer):

    def __init__(self, state: TrainerState, params: Hyperparams):
        if state is None:
            state = TrainerState()
        self.state = state
        self.params = params
        self.events = nnts.events.EventManager()

    def before_train(self, train_dl: Iterable) -> None:
        pass

    def train(self, train_dl, valid_dl=None) -> Evaluator:
        self.before_train(train_dl)
        for epoch in range(1, self.params.epochs + 1):
            if self.state.stop:
                break
            self.state.epoch = epoch
            self._train_epoch(train_dl)
            if valid_dl:
                self._validate_epoch(valid_dl)

        return self.create_evaluator()

    def before_train_epoch(self) -> None:
        pass

    def after_train_epoch(self) -> None:
        pass

    def _train_epoch(self, train_dl: Iterable) -> Any:
        if self.params.batches_per_epoch is None:
            self.params.batches_per_epoch = len(train_dl)
        self.before_train_epoch()
        loss = 0
        for i, batch in enumerate(train_dl):
            if i > self.params.batches_per_epoch:
                break
            L = self._train_batch(i, batch)
            loss += L
        loss /= self.params.batches_per_epoch  # len(train_dl
        self.state.train_loss = loss
        self.after_train_epoch()
        self.events.notify(EpochTrainComplete(self.state))
        return loss

    def before_validate_epoch(self) -> None:
        pass

    def after_validate_epoch(self) -> None:
        pass

    def _validate_epoch(self, valid_dl: Iterable) -> Any:
        self.before_validate_epoch()
        loss = 0
        for i, batch in enumerate(valid_dl):
            L = self._validate_batch(i, batch)
            loss += L
        loss /= len(valid_dl)
        self.state.valid_loss = loss
        self.after_validate_epoch()
        self.events.notify(EpochValidateComplete(self.state))
        return loss

    @abstractmethod
    def _train_batch(self, i, batch) -> Any:
        pass

    @abstractmethod
    def _validate_batch(self, i, batch) -> Any:
        pass
