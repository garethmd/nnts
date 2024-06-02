from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TrainingMethod(Enum):
    TEACHER_FORCING = auto()
    FREE_RUNNING = auto()


@dataclass
class Hyperparams:
    """Class for keeping track of training and model params."""

    input_dim: int = 1
    hidden_dim: int = 40
    n_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    rnn_type: str = "lstm"
    early_stopper_patience: int = 30
    batches_per_epoch: int = 200
    weight_decay: float = 1e-8
    training_method: TrainingMethod = TrainingMethod.TEACHER_FORCING
    optimizer: Any = None
