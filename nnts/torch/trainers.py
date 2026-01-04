import os
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

import nnts.data
import nnts.loggers
import nnts.metrics
import nnts.trainers
from nnts import datasets, utils

from .datasets import PaddedData


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.min_loss = np.inf
        self.counter = 0

    def early_stopping(self, loss: float) -> bool:
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience or np.isnan(loss):
            print("early stopping")
            return True

        return False


def validate(net, batch: PaddedData, prediction_length, context_length):
    if hasattr(net, "validate"):
        return net.validate(batch, prediction_length, context_length)
    else:
        y = batch.data[:, context_length : context_length + prediction_length, ...]
        y_hat = net.generate(
            batch.data[:, :context_length, ...],
            batch.pad_mask[:, :context_length],
            prediction_length=prediction_length,
            context_length=context_length,
        )
        y_hat = y_hat[:, -prediction_length:, ...]
        return y_hat, y


def eval(
    net, dl, prediction_length: int, context_length: int, hooks: Any = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    net.eval()
    with torch.no_grad():
        y_list = []
        y_hat_list = []
        for batch in dl:
            y_hat, y = validate(net, batch, prediction_length, context_length)
            y_list.append(y[:, :, :])
            y_hat_list.append(y_hat[:, :, :])

            # TODO: improve this. Removing hook after first batch to make visualisation work
            if hooks is not None:
                hooks.remove()

        y = torch.cat(y_list, dim=0)
        y_hat = torch.cat(y_hat_list, dim=0)
    return y_hat, y


class TorchEvaluator(nnts.trainers.Evaluator):
    def __init__(self, net: torch.nn.Module):
        self.net = net

    def evaluate(
        self,
        dl: torch.utils.data.DataLoader,
        prediction_length: int,
        context_length: int,
        hooks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return eval(self.net, dl, prediction_length, context_length, hooks=hooks)


class TorchForecaster(nnts.trainers.Forecaster):
    def __init__(self, net: torch.nn.Module):
        self.net = net

    def forecast_batch(
        self, batch: PaddedData, prediction_length: int, context_length: int
    ) -> Any:
        y_hat = self.net.generate(
            batch.data,
            batch.pad_mask,
            prediction_length=prediction_length,
            context_length=context_length,
        )

        y_hat = y_hat[:, -prediction_length:, ...]
        return y_hat

    def forecast(
        self,
        dl: torch.utils.data.DataLoader,
        prediction_length: int,
        context_length: int,
    ) -> Any:
        self.net.eval()
        with torch.no_grad():
            y_hat_list = []
            for batch in dl:
                y_hat = self.forecast_batch(batch, prediction_length, context_length)
                y_hat_list.append(y_hat[:, :, :1])

            y_hat = torch.cat(y_hat_list, dim=0)
        return y_hat


class ValidationTorchEpochTrainer(nnts.trainers.EpochTrainer):

    def __init__(
        self,
        net: torch.nn.Module,
        params: utils.Hyperparams,
        metadata: datasets.Metadata,
        state: nnts.trainers.TrainerState = None,
        model_path: str = "best_model.pt",
    ):
        super().__init__(state, params)
        self.net = net
        self.metadata = metadata
        utils.makedirs_if_not_exists(params.model_file_path)
        self.path = os.path.join(params.model_file_path, model_path)
        self.loss_fn = params.loss_fn
        self.Optimizer = params.optimizer

    def before_train(self, train_dl):
        print(self.net)
        if self.params.batches_per_epoch is None:
            self.params.batches_per_epoch = len(train_dl)
        self.optimizer = self.Optimizer(
            self.net.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=self.params.patience
            )
        elif self.params.scheduler == utils.Scheduler.STEP_LR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=0.5
            )
        elif self.params.scheduler == utils.Scheduler.ONE_CYCLE:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.params.lr,
                steps_per_epoch=self.params.batches_per_epoch,
                epochs=self.params.epochs,
                pct_start=0.3,
            )
        self.early_stopper = (
            None
            if self.params.early_stopper_patience is None
            else EarlyStopper(patience=self.params.early_stopper_patience)
        )

        self.best_loss = np.inf

    def before_train_epoch(self) -> None:
        self.net.train()

    def before_validate_epoch(self) -> None:
        self.net.eval()

    def after_validate_epoch(self) -> None:
        if self.early_stopper:
            if self.early_stopper.early_stopping(self.state.valid_loss):
                self.state.stop = True

        if self.state.valid_loss < self.state.best_loss:
            print(
                f"Epoch {self.state.epoch} Train Loss: {self.state.train_loss}  Valid Loss: {self.state.valid_loss}"
            )
            print(f"saving model to {self.path}")
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.valid_loss
            self.events.notify(nnts.trainers.EpochBestModel(self.path))

        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler.step(self.state.valid_loss)
        elif self.params.scheduler == utils.Scheduler.STEP_LR:
            if self.state.epoch > 1:
                self.scheduler.step()
                print("reducing lr", self.scheduler.get_last_lr())

    def _train_batch(self, i, batch):
        self.optimizer.zero_grad()
        y_hat, y = self.net.train_output(
            batch,
            self.metadata.prediction_length,
            self.metadata.context_length,
            training_method=self.params.training_method,
        )

        L = self.loss_fn(y_hat, y)
        L.backward()
        self.optimizer.step()
        if self.params.scheduler == utils.Scheduler.ONE_CYCLE:
            self.scheduler.step()  # This is required for OneCycleLR
        return L

    def _validate_batch(self, i, batch):
        with torch.no_grad():
            y_hat, y = validate(
                self.net,
                batch,
                self.metadata.prediction_length,
                self.metadata.context_length,
            )
            L = self.loss_fn(y_hat, y)
        return L

    def create_evaluator(self) -> nnts.trainers.Evaluator:
        state_dict = torch.load(self.path)
        self.net.load_state_dict(state_dict)
        return TorchEvaluator(self.net)


class TorchEpochTrainer(nnts.trainers.EpochTrainer):

    def __init__(
        self,
        net: torch.nn.Module,
        params: utils.Hyperparams,
        metadata: datasets.Metadata,
        state: nnts.trainers.TrainerState = None,
        model_path: str = "best_model.pt",
    ):
        super().__init__(state, params)
        self.net = net
        self.metadata = metadata
        utils.makedirs_if_not_exists(params.model_file_path)
        self.path = os.path.join(params.model_file_path, model_path)
        self.loss_fn = params.loss_fn
        self.Optimizer = params.optimizer

    def before_train(self, train_dl):
        print(self.net)
        self.optimizer = self.Optimizer(
            self.net.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=self.params.patience
            )
        elif self.params.scheduler == utils.Scheduler.STEP_LR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=0.5
            )
        elif self.params.scheduler == utils.Scheduler.ONE_CYCLE:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.params.lr * 3,
                steps_per_epoch=self.params.batches_per_epoch,
                epochs=self.params.epochs + 2,
            )
        self.best_loss = np.inf

    def before_train_epoch(self) -> None:
        self.net.train()

    def after_train_epoch(self) -> None:
        if self.state.train_loss < self.state.best_loss:
            print(f"saving model to {self.path}")
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.train_loss

        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler.step(self.state.train_loss)
        elif self.params.scheduler == utils.Scheduler.STEP_LR:
            self.scheduler.step()

        print(f"Epoch {self.state.epoch} Train Loss: {self.state.train_loss}")

    def _train_batch(self, i, batch):
        self.optimizer.zero_grad()
        y_hat, y = self.net.train_output(
            batch,
            self.metadata.prediction_length,
            self.metadata.context_length,
            training_method=self.params.training_method,
        )

        L = self.loss_fn(y_hat, y)
        L.backward()
        # torch.nn.utils.clip_grad_value_(self.net.parameters(), 10.0)
        self.optimizer.step()

        if self.params.scheduler == utils.Scheduler.ONE_CYCLE:
            self.scheduler.step()  # This is required for OneCycleLR
        return L

    def _validate_batch(self, i, batch):
        pass

    def create_evaluator(self) -> nnts.trainers.Evaluator:
        self.events.notify(nnts.trainers.EpochBestModel(self.path))
        state_dict = torch.load(self.path)
        self.net.load_state_dict(state_dict)
        print(f"Best model loaded from {self.path}")
        return TorchEvaluator(self.net)
