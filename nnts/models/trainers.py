from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, PositiveInt

from .hyperparams import Hyperparams


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self) -> None:
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


class EpochTrainer(Trainer):

    def __init__(self, state: TrainerState, params: Hyperparams):
        self.state = state
        self.params = params

    def before_train(self, train_dl: Iterable) -> None:
        pass

    def train(self, train_dl, valid_dl) -> Evaluator:
        self.before_train(train_dl)
        for epoch in range(1, self.params.epochs + 1):
            if self.state.stop:
                break
            self.state.epoch = epoch
            self._train_epoch(train_dl)
            self._validate_epoch(valid_dl)

        return self.create_evaluator()

    def before_train_epoch(self) -> None:
        pass

    def _train_epoch(self, train_dl: Iterable) -> Any:
        self.before_train_epoch()
        loss = 0
        for i, batch in enumerate(train_dl):
            if i > self.params.batches_per_epoch:
                break
            L = self._train_batch(i, batch)
            loss += L
        loss /= len(train_dl)
        self.state.train_loss = loss
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
        return loss

    @abstractmethod
    def _train_batch(self, i, batch) -> Any:
        pass

    @abstractmethod
    def _validate_batch(self, i, batch) -> Any:
        pass
