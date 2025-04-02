import json
import os
import shutil
import timeit
import types
from abc import ABC, abstractmethod
from enum import Enum
from functools import singledispatchmethod
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb

from . import events, trainers, utils


class Handler(ABC):

    @abstractmethod
    def handle(self, data: Any) -> None:
        pass


class PrintHandler(Handler):
    def handle(self, data: Dict[str, Any]) -> None:
        print(data)


def convert_np_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class JsonFileHandler(Handler):
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename + ".json"

    def handle(self, data: Dict[str, Any]) -> None:
        with open(os.path.join(self.path, self.filename), "w") as json_file:
            json.dump(data, json_file, indent=4, default=convert_np_float)


class EpochEventMixin(events.Listener):

    @singledispatchmethod
    def notify(self, event: Dict[str, Any]) -> None:
        self.log(event)

    @notify.register(trainers.EpochTrainComplete)
    def _(self, event: trainers.EpochTrainComplete) -> None:
        self.log(
            {
                "epoch": event.state.epoch,
                "train_loss": event.state.train_loss,  # event.state.train_loss.detach().item(),
            }
        )

    @notify.register(trainers.EpochValidateComplete)
    def _(self, event: trainers.EpochValidateComplete) -> None:
        self.log(
            {
                "valid_loss": event.state.valid_loss  # event.state.valid_loss.detach().item(),
            }
        )
        print(
            f"Epoch {event.state.epoch} train loss: {event.state.train_loss}, valid loss: {event.state.valid_loss}"
        )

    @notify.register(trainers.EpochBestModel)
    def _(self, event: trainers.EpochBestModel) -> None:
        self.log_model(event.path)

    def configure(self, evts: events.EventManager) -> None:
        evts.add_listener(trainers.EpochTrainComplete, self)
        evts.add_listener(trainers.EpochValidateComplete, self)
        evts.add_listener(trainers.EpochBestModel, self)


class Run(ABC):

    @abstractmethod
    def log(self, data: Any) -> None:
        pass

    @abstractmethod
    def log_model(self, name: str, path: str) -> None:
        pass

    @abstractmethod
    def log_activations(self, module, input, output) -> None:
        pass


class ActivationVisualizer:
    def __init__(self):
        self.activations = []

    def has_activations(self):
        return len(self.activations) > 0

    def save_heatmap(self, path):
        if len(self.activations) > 0:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.activations, cmap="coolwarm", linewidths=0.5)
            plt.savefig(path)

    def append(self, output):
        if isinstance(output, tuple):
            output = output[0]
        input_0 = output[0]
        self.activations.append(input_0[:, -1].detach().cpu().numpy())


class LocalFileRun(Run, EpochEventMixin):

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any] = None,
        path: str = "",
        Handler: Handler = JsonFileHandler,
    ):
        self.project = project
        self.name = name
        self.static_data = config
        self.path = path
        utils.makedirs_if_not_exists(self.path)
        self.handler = Handler(path=path, filename=name)
        self.start_time = timeit.default_timer()
        self.activation_visualizer = ActivationVisualizer()

    def log(self, data: Any) -> None:
        self.static_data = {**self.static_data, **data}

    def log_model(self, source_file: str) -> None:
        print(f"Artifact saved to {source_file}")
        try:
            shutil.copy(source_file, self.path)
        except shutil.SameFileError:
            pass

    def log_outputs(self, data: Dict[str, Any]) -> None:
        numpy_dict = {key: value for key, value in data.items()}
        # Save each numpy array to a separate text file
        for key, value in numpy_dict.items():
            file_name = key + ".txt"  # You can use the key as the file name
            np.savetxt(
                os.path.join(self.path, file_name), value, fmt="%f"
            )  # Save the numpy array to a text file

    def log_activations(self, module, input, output) -> None:
        self.activation_visualizer.append(output)

    def finish(self) -> None:
        run_time = timeit.default_timer() - self.start_time
        self.static_data["run_time"] = run_time
        self.handler.handle(self.static_data)
        if self.activation_visualizer.has_activations():
            try:
                self.activation_visualizer.save_heatmap(
                    os.path.join(self.path, "activations.png")
                )
            except Exception as e:
                print(f"Error saving activations: {e}")
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

    def log_activations(self, module, input, output) -> None:
        print("input", input)
        print("output", output)

    def finish(self) -> None:
        run_time = timeit.default_timer() - self.start_time
        self.static_data["run_time"] = run_time
        print(self.static_data)
        print(f"Run {self.name} finished")


class WandbRun(Run, EpochEventMixin):

    def __init__(
        self, project: str, name: str, config: Dict[str, Any] = None, path: str = ""
    ):
        self.project = project
        self.name = name
        self.static_data = config
        self.run = wandb.init(
            project=self.project, name=self.name, config=self.static_data
        )
        self.path = path
        utils.makedirs_if_not_exists(self.path)
        self.activation_visualizer = ActivationVisualizer()

    def log(self, data: Any) -> None:
        self.run.log(data)

    def log_model(self, source_file: str) -> None:
        self.run.log_model(name=f"{self.name}-{self.run.id}", path=source_file)

    def log_activations(self, module, input, output) -> None:
        self.activation_visualizer.append(output)

    def finish(self) -> None:
        print(f"Run {self.name} finished")

        if self.activation_visualizer.has_activations():
            try:
                activation_image_path = os.path.join(self.path, "activations.png")
                self.activation_visualizer.save_heatmap(activation_image_path)
                self.run.log({"activations": wandb.Image(activation_image_path)})
            except Exception as e:
                print(f"Error saving activations: {e}")

        self.run.finish()

    def log_table(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        table = wandb.Table(dataframe=df)
        table_artifact = wandb.Artifact(metadata.dataset, type="dataset")
        table_artifact.add(table, "table")
        self.run.log_artifact(table_artifact)
        self.log({"results": table})


"""
test_table = wandb.Table(dataframe=test_df)
  test_table_artifact = wandb.Artifact(metadata.dataset, type="dataset")
  test_table_artifact.add(test_table, "test_table")

  logger.log(test_metrics)
  logger.log({"results": test_table})

  # and Log as an Artifact to increase the available row limit!
  lxogger.run.log_artifact(test_table_artifact)
"""
