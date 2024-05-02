import json
import os
import shutil
import timeit
from abc import ABC, abstractmethod
from enum import Enum
from functools import singledispatchmethod
from typing import Any, Dict

import numpy as np
import wandb

import nnts.events
import nnts.models.trainers


def makedirs_if_not_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_np_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class Handler(ABC):

    @abstractmethod
    def handle(self, data: Any) -> None:
        pass


class PrintHandler(Handler):
    def handle(self, data: Any) -> None:
        print(data)


class JsonFileHandler(Handler):
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def handle(self, data: Any) -> None:
        makedirs_if_not_exists(self.path)
        with open(os.path.join(self.path, self.filename), "w") as json_file:
            json.dump(data, json_file, indent=4, default=convert_np_float)


class EpochEventMixin(nnts.events.Listener):

    @singledispatchmethod
    def notify(self, event: Dict[str, Any]) -> None:
        self.log(event)

    @notify.register(nnts.models.trainers.EpochTrainComplete)
    def _(self, event: nnts.models.trainers.EpochTrainComplete) -> None:
        self.log(
            {
                "epoch": event.state.epoch,
                "train_loss": event.state.train_loss.detach().item(),
            }
        )

    @notify.register(nnts.models.trainers.EpochValidateComplete)
    def _(self, event: nnts.models.trainers.EpochValidateComplete) -> None:
        self.log(
            {
                "valid_loss": event.state.valid_loss.detach().item(),
            }
        )
        print(
            f"Epoch {event.state.epoch} train loss: {event.state.train_loss.detach().item()}, valid loss: {event.state.valid_loss.detach().item()}"
        )

    @notify.register(nnts.models.trainers.EpochBestModel)
    def _(self, event: nnts.models.trainers.EpochBestModel) -> None:
        self.log_model(event.path)


class Run(ABC):

    @abstractmethod
    def log(self, data: Any) -> None:
        pass

    @abstractmethod
    def log_model(self, name: str, path: str) -> None:
        pass


class LocalFileRun(Run, EpochEventMixin):

    def __init__(
        self, project: str, name: str, config: Dict[str, Any] = None, path: str = ""
    ):
        self.project = project
        self.name = name
        self.static_data = config
        self.path = path
        self.handler = JsonFileHandler(path=path, filename=f"{name}.json")
        self.start_time = timeit.default_timer()

    def log(self, data: Any) -> None:
        self.static_data = {**self.static_data, **data}

    def log_model(self, source_file: str) -> None:
        print(f"Artifact saved to {source_file}")
        try:
            shutil.copy(source_file, self.path)
        except shutil.SameFileError:
            pass

    def log_outputs(self, data: Dict[str, Any]) -> None:
        makedirs_if_not_exists(self.path)
        numpy_dict = {key: value for key, value in data.items()}
        # Save each numpy array to a separate text file
        for key, value in numpy_dict.items():
            file_name = key + ".txt"  # You can use the key as the file name
            np.savetxt(
                os.path.join(self.path, file_name), value, fmt="%f"
            )  # Save the numpy array to a text file

    def hook_fn(self, module, input, output):
        # self.activation_values.append(output.detach().numpy())
        self.handler.handle_outputs(
            {"activations": input[0].detach().reshape(-1).numpy()}
        )

    def finish(self) -> None:
        run_time = timeit.default_timer() - self.start_time
        self.static_data["run_time"] = run_time
        self.handler.handle(self.static_data)
        print(f"Run {self.name} finished")


class PrintRun(Run, EpochEventMixin):

    def __init__(self, project: str, name: str, config: Dict[str, Any] = None):
        self.project = project
        self.name = name
        self.static_data = config
        self.handler = PrintHandler()
        self.start_time = timeit.default_timer()

    def log(self, data: Any) -> None:
        self.static_data = {**self.static_data, **data}

    def log_model(self, source_file: str) -> None:
        print(f"Artifact saved to {source_file}")

    def log_outputs(self, data: Dict[str, Any]) -> None:
        print(data)

    def hook_fn(self, module, input, output):
        print("input", input)
        print("output", output)

    def finish(self) -> None:
        run_time = timeit.default_timer() - self.start_time
        self.static_data["run_time"] = run_time
        print(self.static_data)
        print(f"Run {self.name} finished")


class ProjectRun(Run, EpochEventMixin):

    def __init__(
        self, handler: Handler, project: str, name: str, config: Dict[str, Any] = None
    ):
        self.project = project
        self.name = name
        self.static_data = config
        self.handler = handler
        self.start_time = timeit.default_timer()

    def log(self, data: Any) -> None:
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
        print(f"Run {self.name} finished")


class WandbRun(Run, EpochEventMixin):

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
