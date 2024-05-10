from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd

from . import metadata

SplitData = namedtuple("SplitData", ["train", "validation", "test"])
SplitTrainTest = namedtuple("SplitTrainTest", ["train", "test"])


class Splitter(ABC):
    @abstractmethod
    def split(self, data, *args, **kwargs) -> namedtuple:
        pass
