import timeit
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import nnts.data
import nnts.models


def point_loss_all(loss_fn, self, data):
    x = data["X"]
    pad_mask = data["pad_mask"]
    y_hat = self(x[:, :-1, :], pad_mask[:, :-1])
    y = x[:, 1:, :]

    y_hat = y_hat[pad_mask[:, 1:]]
    y = y[pad_mask[:, 1:]]
    return loss_fn(y_hat, y)


smooth_l1_loss_all = partial(point_loss_all, F.smooth_l1_loss)


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

        if self.counter > self.patience:
            print("early stopping")
            return True

        return False


def generate(
    self, X, pad_mask, prediction_length, context_length
):  # PREDICTION_LENGTH, CONTEXT_LENGTH
    pred_list = []
    while True:
        pad_mask = pad_mask[:, -context_length:]
        assert (
            pad_mask.sum()
            == X[:, -context_length:, :].shape[0] * X[:, -context_length:, :].shape[1]
        )
        preds = self.forward(X[:, -context_length:, :], pad_mask)
        # focus only on the last time step
        preds = preds[:, -1:, :]  # becomes (B, 1, C)
        pred_list.append(preds)

        if len(pred_list) >= prediction_length:
            break
        y_hat = preds.detach().clone()
        X = torch.cat((X, y_hat), dim=1)  # (B, T+1)
        pad_mask = torch.cat((pad_mask, torch.ones_like(preds[:, :, 0])), dim=1)
    return torch.cat(pred_list, 1)


def validate(net, batch, prediction_length, context_length):
    """forecast horizon: the number of steps to forecast
    prediction length: the potential maximum forecast horizon available
    """
    y = batch["X"][:, context_length : context_length + prediction_length, ...]
    y_hat = generate(
        net,
        batch["X"][:, :context_length, ...],
        batch["pad_mask"][:, :context_length],
        prediction_length=prediction_length,
        context_length=context_length,
    )
    y_hat = y_hat[:, -prediction_length:, ...]
    return y, y_hat


def train(net, trn_dl, val_dl, params, metadata, path, logger=None, seed_fn=None):
    if seed_fn is not None:
        seed_fn(42)
    print(net)
    optimizer = optim.AdamW(
        net.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #                optimizer,
    #                mode="min",
    #                #factor=0.5,
    #                patience=10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.lr * 3,
        steps_per_epoch=len(trn_dl),
        epochs=params.epochs,
    )
    early_stopper = (
        None
        if params.early_stopper_patience is None
        else EarlyStopper(patience=params.early_stopper_patience)
    )
    net.train()
    best_loss = np.inf
    for epoch in range(params.epochs):
        start_time = timeit.default_timer()
        net.train()
        train_loss = 0
        for i, batch in enumerate(trn_dl):
            if i > params.batches_per_epoch:
                break
            optimizer.zero_grad()
            L = smooth_l1_loss_all(net, batch).mean()
            L.backward()
            optimizer.step()
            scheduler.step()
            train_loss += L
        train_loss = train_loss / len(trn_dl)

        # validation
        with torch.no_grad():
            net.eval()
            valid_loss = 0
            for i, batch in enumerate(val_dl):
                y, y_hat = validate(
                    net, batch, metadata.prediction_length, metadata.context_length
                )

                L = F.smooth_l1_loss(y_hat, y)
                valid_loss += L

            valid_loss = valid_loss / len(val_dl)

            if valid_loss < best_loss:
                torch.save(net.state_dict(), path)
                best_loss = valid_loss
                if logger is not None:
                    logger.log_model(name=f"{metadata.dataset}-{logger.id}", path=path)
            if logger is not None:
                logger.log({"train_loss": train_loss, "valid_loss": valid_loss})

        elapsed_time = timeit.default_timer() - start_time
        print(
            f"epoch {epoch}, loss {train_loss} valid loss {valid_loss}, elapsed time {elapsed_time}"
        )
        if early_stopper:
            if early_stopper.early_stopping(valid_loss):
                break

    net = torch.load(path, map_location=torch.device("cpu"))
    return net


def eval(net, dl, prediction_length: int, context_length: int):
    net.eval()
    with torch.no_grad():
        y_list = []
        y_hat_list = []
        for batch in dl:
            y, y_hat = validate(net, batch, prediction_length, context_length)
            y_list.append(y[:, :, :1])
            y_hat_list.append(y_hat[:, :, :1])

        y = torch.cat(y_list, dim=0)
        y_hat = torch.cat(y_hat_list, dim=0)
    return y, y_hat


class TorchEvaluator(nnts.models.Evaluator):
    def __init__(self, net):
        self.net = net

    def evaluate(self, dl, prediction_length, context_length):
        return eval(self.net, dl, prediction_length, context_length)


class TorchEpochTrainer(nnts.models.EpochTrainer):
    def __init__(
        self,
        state: nnts.models.TrainerState,
        net: torch.nn.Module,
        params: nnts.models.Hyperparams,
        metadata: nnts.data.Metadata,
        path: str,
        seed_fn: callable = None,
    ):
        super().__init__(state, params)
        self.net = net
        self.metadata = metadata
        self.path = path
        self.seed_fn = seed_fn

    def before_train(self, train_dl):
        print(self.net)
        if self.seed_fn is not None:
            print("setting seed")
            self.seed_fn(42)

        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.params.lr * 3,
            steps_per_epoch=len(train_dl),
            epochs=self.params.epochs,
        )
        self.early_stopper = (
            None
            if self.params.early_stopper_patience is None
            else EarlyStopper(patience=self.params.early_stopper_patience)
        )

        self.best_loss = np.inf

    def before_train_epoch(self) -> None:
        self.start_time = timeit.default_timer()
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

        elapsed_time = timeit.default_timer() - self.start_time
        print(
            f"epoch {self.state.epoch}, loss {self.state.train_loss} valid loss {self.state.valid_loss}, elapsed time {elapsed_time}"
        )

    def _train_batch(self, i, batch):
        self.optimizer.zero_grad()
        L = smooth_l1_loss_all(self.net, batch).mean()
        L.backward()
        self.optimizer.step()
        self.scheduler.step()
        return L

    def _validate_batch(self, i, batch):
        with torch.no_grad():
            y, y_hat = validate(
                self.net,
                batch,
                self.metadata.prediction_length,
                self.metadata.context_length,
            )
            L = F.smooth_l1_loss(y_hat, y)
        return L

    def create_evaluator(self) -> nnts.models.Evaluator:
        state_dict = torch.load(self.path, map_location=torch.device("cpu"))
        self.net.load_state_dict(state_dict)
        return TorchEvaluator(self.net)
