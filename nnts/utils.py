import json
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel

SplitData = namedtuple("SplitData", ["train", "validation", "test"])
SplitTrainTest = namedtuple("SplitTrainTest", ["train", "test"])

FREQUENCY_MAP: dict = {
    "minutely": "1min",
    "10_minutes": "10min",
    "half_hourly": "30min",
    "hourly": "1H",
    "daily": "1D",
    "weekly": "1W",
    "monthly": "1M",
    "quarterly": "1Q",
    "yearly": "1Y",
}


class Metadata(BaseModel):
    """Class for storing dataset metadata"""

    filename: str
    dataset: str
    context_length: int
    prediction_length: int
    freq: str
    seasonality: int


def load(
    dataset: str,
    path: str = None,
) -> Metadata:
    # Get the directory of the current script
    with open(path) as f:
        data = json.load(f)
    return Metadata(**data[dataset])


class TrainingMethod(Enum):
    TEACHER_FORCING = auto()
    FREE_RUNNING = auto()


class Scheduler(Enum):
    ONE_CYCLE = auto()
    REDUCE_LR_ON_PLATEAU = auto()


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
    scheduler: Scheduler = Scheduler.ONE_CYCLE
