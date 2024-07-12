import json
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import pandas as pd


def makedirs_if_not_exists(directory: str) -> None:
    if len(directory) == 0:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)


class TrainingMethod(Enum):
    TEACHER_FORCING = auto()
    FREE_RUNNING = auto()


class Scheduler(Enum):
    ONE_CYCLE = auto()
    REDUCE_LR_ON_PLATEAU = auto()


@dataclass
class Hyperparams:
    """Class for keeping track of training and model params."""

    optimizer: Any
    loss_fn: Any
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
    scheduler: Scheduler = Scheduler.ONE_CYCLE
    model_file_path: str = ""


@dataclass
class GluonTsDefaultWithOneCycle(Hyperparams):
    """Class for keeping track of training and model params."""

    optimizer: Callable
    loss_fn: Callable
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
    training_method: TrainingMethod = TrainingMethod.TEACHER_FORCING
    scheduler: Scheduler = Scheduler.ONE_CYCLE


class CSVFileAggregator:
    def __init__(self, path: str, filename: str):
        self.path = path
        self.filename = filename

    def __call__(self) -> pd.DataFrame:
        data_list = []
        for filename in os.listdir(self.path):
            if filename.endswith(".json"):
                with open(os.path.join(self.path, filename), "r") as file:
                    data = json.load(file)
                    data_list.append(data)
        # Concatenate DataFrames if needed
        results = pd.DataFrame(data_list)
        results.to_csv(f"{self.path}/{self.filename}.csv", index=False)
        return results
