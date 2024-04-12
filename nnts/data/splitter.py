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


def slice_rows(group: pd.DataFrame, start, end) -> pd.DataFrame:
    return group.iloc[start:end]


def split_dataframe(
    data: pd.DataFrame, train_size: int, val_size: int, test_size: int
) -> SplitData:
    trn = data.groupby("unique_id").head(train_size)
    val = (
        data.groupby("unique_id")
        .apply(slice_rows, train_size, train_size + val_size)
        .reset_index(drop=True)
    )
    test = data.groupby("unique_id").tail(test_size)
    return SplitData(train=trn, validation=val, test=test)
