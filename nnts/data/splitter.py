from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd

from . import metadata

SplitData = namedtuple("SplitData", ["train", "validation", "test"])


class Splitter(ABC):
    @abstractmethod
    def split(self, data, *args, **kwargs) -> namedtuple:
        pass


class PandasSplitter(Splitter):

    def split(
        self, data: pd.DataFrame, metadata: metadata.Metadata, *args, **kwargs
    ) -> SplitData:
        trn_val = data.groupby("unique_id").head(-metadata.prediction_length)
        trn = trn_val.groupby("unique_id").head(-metadata.prediction_length)
        val = trn_val.groupby("unique_id").tail(
            metadata.context_length + metadata.prediction_length
        )
        test = data.groupby("unique_id").tail(
            metadata.context_length + metadata.prediction_length
        )
        return SplitData(train=trn, validation=val, test=test)
