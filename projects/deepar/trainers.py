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
        A smooth L1 loss function (for point predictions)
    """

    def __init__(
        self,
        state: nnts.models.TrainerState,
        net: torch.nn.Module,
        params: nnts.models.Hyperparams,
        metadata: nnts.data.Metadata,
        path: str,
        loss_fn=F.smooth_l1_loss,
    ):
        super().__init__(state, params)
        self.net = net
        self.metadata = metadata
        self.path = path
        self.loss_fn = loss_fn

    def before_train(self, train_dl):
        print(self.net)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.best_loss = np.inf

    def before_train_epoch(self) -> None:
        self.net.train()

    def after_train_epoch(self) -> None:
        if self.state.train_loss < self.state.best_loss:
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.train_loss
            self.events.notify(nnts.models.trainers.EpochBestModel(self.path))

        print(f"Train Loss: {self.state.train_loss}")
        self.scheduler.step(self.state.train_loss)

    def before_validate_epoch(self) -> None:
        self.net.eval()

    def after_validate_epoch(self) -> None:
        if self.state.valid_loss < self.state.best_loss:
            torch.save(self.net.state_dict(), self.path)
            self.state.best_loss = self.state.valid_loss
            self.events.notify(nnts.models.trainers.EpochBestModel(self.path))

        self.scheduler.step(self.state.valid_loss)

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
        self.optimizer.step()
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
        state_dict = torch.load(self.path, map_location=torch.device("cpu"))
        self.net.load_state_dict(state_dict)
        return nnts.torch.models.trainers.TorchEvaluator(self.net)
