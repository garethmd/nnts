from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import nnts.data
import nnts.loggers
import nnts.trainers
from nnts import utils


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


def validate(net, batch, prediction_length, context_length):
    if hasattr(net, "validate"):
        return net.validate(batch, prediction_length, context_length)
    else:
        y = batch["X"][:, context_length : context_length + prediction_length, ...]
        y_hat = net.generate(
            batch["X"][:, :context_length, ...],
            batch["pad_mask"][:, :context_length],
            prediction_length=prediction_length,
            context_length=context_length,
        )
        y_hat = y_hat[:, -prediction_length:, ...]
        return y_hat, y


def eval(net, dl, prediction_length: int, context_length: int, hooks: Any = None):
    net.eval()
    with torch.no_grad():
        y_list = []
        y_hat_list = []
        for batch in dl:
            y_hat, y = validate(net, batch, prediction_length, context_length)
            y_list.append(y[:, :, :1])
            y_hat_list.append(y_hat[:, :, :1])

            # TODO: improve this. Removing hook after first batch to make visualisation work
            if hooks is not None:
                hooks.remove()

        y = torch.cat(y_list, dim=0)
        y_hat = torch.cat(y_hat_list, dim=0)
    return y_hat, y


class TorchEvaluator(nnts.trainers.Evaluator):
    def __init__(self, net):
        self.net = net

    def evaluate(self, dl, prediction_length, context_length, hooks=None):
        return eval(self.net, dl, prediction_length, context_length, hooks=hooks)


class ValidationTorchEpochTrainer(nnts.trainers.EpochTrainer):
    def __init__(
        self,
        state: nnts.trainers.TrainerState,
        net: torch.nn.Module,
        params: utils.Hyperparams,
        metadata: utils.Metadata,
        path: str,
        loss_fn=F.l1_loss,
    ):
        super().__init__(state, params)
        self.net = net
        self.metadata = metadata
        self.path = path
        self.loss_fn = loss_fn
        self.Optimizer = params.optimizer or torch.optim.AdamW

    def before_train(self, train_dl):
        print(self.net)
        self.optimizer = self.Optimizer(
            self.net.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )
        elif self.params.scheduler == utils.Scheduler.ONE_CYCLE:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.params.lr * 3,
                steps_per_epoch=self.params.batches_per_epoch,
                epochs=self.params.epochs + 2,
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
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.valid_loss
            self.events.notify(nnts.trainers.EpochBestModel(self.path))

        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler.step(self.state.valid_loss)

    def _train_batch(self, i, batch):
        self.optimizer.zero_grad()

        if self.params.training_method == utils.TrainingMethod.FREE_RUNNING:
            y_hat, y = self.net.free_running(
                batch,
                self.metadata.context_length,
                self.metadata.prediction_length,
            )
        else:
            y_hat, y = self.net.teacher_forcing_output(
                batch, self.metadata.context_length, self.metadata.prediction_length
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
        state: nnts.trainers.TrainerState,
        net: torch.nn.Module,
        params: utils.Hyperparams,
        metadata: utils.Metadata,
        path: str,
        loss_fn=F.l1_loss,
    ):
        super().__init__(state, params)
        self.net = net
        self.metadata = metadata
        self.path = path
        self.loss_fn = loss_fn
        self.Optimizer = params.optimizer or torch.optim.AdamW

    def before_train(self, train_dl):
        print(self.net)
        self.optimizer = self.Optimizer(
            self.net.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
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
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.train_loss
            self.events.notify(nnts.trainers.EpochBestModel(self.path))

        if self.params.scheduler == utils.Scheduler.REDUCE_LR_ON_PLATEAU:
            self.scheduler.step(self.state.train_loss)

        print(f"Epoch {self.state.epoch} Train Loss: {self.state.train_loss}")

    def before_validate_epoch(self) -> None:
        self.net.eval()

    def after_validate_epoch(self) -> None:
        if self.state.valid_loss < self.state.best_loss:
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.valid_loss
            self.events.notify(nnts.trainers.EpochBestModel(self.path))

    def _train_batch(self, i, batch):
        self.optimizer.zero_grad()

        if self.params.training_method == utils.TrainingMethod.FREE_RUNNING:
            y_hat, y = self.net.free_running(
                batch,
                self.metadata.prediction_length,
                self.metadata.context_length,
            )
        else:
            y_hat, y = self.net.teacher_forcing_output(
                batch,
                self.metadata.prediction_length,
                self.metadata.context_length,
            )

        L = self.loss_fn(y_hat, y)
        L.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 10.0)
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
        print(f"Best model loaded from {self.path}")
        return TorchEvaluator(self.net)
