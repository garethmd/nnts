from dataclasses import dataclass
from typing import Callable

import torch

from nnts import utils


@dataclass
class GluonTsDefaultWithOneCycle(utils.Hyperparams):
    """Class for keeping track of training and model params."""

    input_dim: int = 1
    hidden_dim: int = 40
    n_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    rnn_type: str = "lstm"
    early_stopper_patience: int = 30
    batches_per_epoch: int = 50
    weight_decay: float = 1e-8
    training_method: utils.TrainingMethod = utils.TrainingMethod.TEACHER_FORCING
    scheduler: utils.Scheduler = utils.Scheduler.ONE_CYCLE
    optimizer: Callable = torch.optim.AdamW
    loss_fn: Callable = torch.nn.NLLLoss
