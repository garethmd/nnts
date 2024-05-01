import json
import os
import shutil
import timeit
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

import numpy as np

"""
run = wandb.init(
    project=f"06-rnn-covariates-{metadata.dataset}",
    name=name,
    config={
        **params.__dict__,
        **metadata.__dict__,
        **scenario.__dict__,
    },
)

def init(project:str, name: str, config: Dict[str, Any]) -> Run:
    return PrintRun("test")
"""


def makedirs_if_not_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_np_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError


class Handler(ABC):

    @abstractmethod
    def handle(self, data: Any) -> None:
        pass

    @abstractmethod
    def artifact(self, source_file: str) -> None:
        pass


class PrintHandler(Handler):
    def handle(self, data: Any) -> None:
        print(data)

    def artifact(self, source_file: str) -> None:
        print(f"Artifact saved to {source_file}")


class JsonFileHandler(Handler):
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def artifact(self, source_file: str) -> None:
        print(f"Artifact saved to {source_file}")
        try:
            shutil.copy(source_file, self.path)
        except shutil.SameFileError:
            pass

    def handle(self, data: Any) -> None:
        makedirs_if_not_exists(self.path)
        with open(os.path.join(self.path, self.filename), "w") as json_file:
            json.dump(data, json_file, indent=4, default=convert_np_float)

    def handle_outputs(self, data: Dict[str, np.array]) -> None:
        makedirs_if_not_exists(self.path)
        numpy_dict = {key: value for key, value in data.items()}
        # Save each numpy array to a separate text file
        for key, value in numpy_dict.items():
            file_name = key + ".txt"  # You can use the key as the file name
            np.savetxt(
                os.path.join(self.path, file_name), value, fmt="%f"
            )  # Save the numpy array to a text file


class Run(ABC):

    @abstractmethod
    def log(self, data: Any) -> None:
        pass

    @abstractmethod
    def log_model(self, name: str, path: str) -> None:
        pass


class ProjectRun(Run):

    def __init__(
        self, handler: Handler, project: str, run: str, config: Dict[str, Any] = None
    ):
        self.project = project
        self.run = run
        self.static_data = config
        self.handler = handler
        self.start_time = timeit.default_timer()

    def log(self, data: Any) -> None:
        print(data)
        self.static_data = {**self.static_data, **data}

    def log_model(self, source_file: str) -> None:
        self.handler.artifact(source_file)

    def log_outputs(self, data: Dict[str, Any]) -> None:
        self.handler.handle_outputs(data)

    def hook_fn(self, module, input, output):
        # self.activation_values.append(output.detach().numpy())
        self.handler.handle_outputs(
            {"activations": input[0].detach().reshape(-1).numpy()}
        )

    def finish(self) -> None:
        run_time = timeit.default_timer() - self.start_time
        self.static_data["run_time"] = run_time
        self.handler.handle(self.static_data)
        print(f"Run {self.run} finished")


import wandb


class WandbRun(Run):

    def __init__(self, project: str, name: str, config: Dict[str, Any] = None):
        self.project = project
        self.name = name
        self.static_data = config
        self.run = wandb.init(
            project=self.project, name=self.name, config=self.static_data
        )

    def log(self, data: Any) -> None:
        self.run.log(data)

    def log_model(self, source_file: str) -> None:
        self.run.log_model(name=f"{self.name}-{self.run.id}", path=source_file)

    def finish(self) -> None:
        print(f"Run {self.name} finished")
        self.run.finish()
