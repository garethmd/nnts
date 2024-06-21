from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import nnts.data
import nnts.loggers
import nnts.models
import nnts.torch
import nnts.torch.models
import nnts.torch.models.trainers


class TorchEpochTrainer(nnts.models.EpochTrainer):
    """A GluonTS like trainer that mimics the default behaviour of a GluonTS estimator.
    In particular it uses:
        No validation set
        Adam optimizer
        ReduceLROnPlateau scheduler.
        A l1_loss loss function (for point predictions)
    """

    def __init__(
        self,
        state: nnts.models.TrainerState,
        net: torch.nn.Module,
        params: nnts.models.Hyperparams,
        metadata: nnts.data.Metadata,
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
        if (
            self.params.scheduler
            == nnts.models.hyperparams.Scheduler.REDUCE_LR_ON_PLATEAU
        ):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )
        elif self.params.scheduler == nnts.models.hyperparams.Scheduler.ONE_CYCLE:
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
            self.events.notify(nnts.models.trainers.EpochBestModel(self.path))

        if (
            self.params.scheduler
            == nnts.models.hyperparams.Scheduler.REDUCE_LR_ON_PLATEAU
        ):
            self.scheduler.step(self.state.train_loss)

        print(f"Epoch {self.state.epoch} Train Loss: {self.state.train_loss}")

    def before_validate_epoch(self) -> None:
        self.net.eval()

    def after_validate_epoch(self) -> None:
        if self.state.valid_loss < self.state.best_loss:
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.valid_loss
            self.events.notify(nnts.models.trainers.EpochBestModel(self.path))

    def _train_batch(self, i, batch):
        self.optimizer.zero_grad()

        if (
            self.params.training_method
            == nnts.models.hyperparams.TrainingMethod.FREE_RUNNING
        ):
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

        if self.params.scheduler == nnts.models.hyperparams.Scheduler.ONE_CYCLE:
            self.scheduler.step()  # This is required for OneCycleLR
        return L

    def _validate_batch(self, i, batch):
        with torch.no_grad():
            y_hat, y = nnts.torch.models.trainers.validate(
                self.net,
                batch,
                self.metadata.prediction_length,
                self.metadata.context_length,
            )
            L = self.loss_fn(y_hat, y)
        return L

    def create_evaluator(self) -> nnts.models.Evaluator:
        state_dict = torch.load(self.path)
        self.net.load_state_dict(state_dict)
        print(f"Best model loaded from {self.path}")
        return nnts.torch.models.trainers.TorchEvaluator(self.net)
