import json
import os
from typing import Tuple

import pandas as pd

import nnts.data.metadata
import nnts.data.splitter as splitter
import nnts.data.tsf as tsf

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


def unpack(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    timesteps = pd.date_range(
        df["start_timestamp"], periods=len(df["series_value"]), freq=freq
    )
    unpacked_df = pd.DataFrame(
        data={"y": df["series_value"].to_numpy(), "ds": timesteps}
    )
    unpacked_df["unique_id"] = df["series_name"]
    return unpacked_df


def read_tsf(path: str, url: str = None) -> pd.DataFrame:
    (
        all_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = tsf.read(path, url)
    df = pd.DataFrame(all_data)
    freq = FREQUENCY_MAP[frequency]
    if "start_timestamp" not in df.columns:
        df["start_timestamp"] = pd.Timestamp("1970-01-01")
    df = pd.concat([unpack(df.iloc[x], freq=freq) for x in range(len(df))])
    return df, freq, forecast_horizon, contain_missing_values, contain_equal_length


def load(
    dataset_name: str, data_path: str, metadata_filename: str
) -> Tuple[pd.DataFrame, nnts.data.metadata.Metadata]:

    metadata_path = os.path.join(data_path, metadata_filename)
    metadata = nnts.data.metadata.load(dataset_name, path=metadata_path)
    datafile_path = os.path.join(data_path, metadata.filename)
    df, freq, forecast_horizon, *_ = read_tsf(datafile_path)
    metadata.freq = freq
    if forecast_horizon is not None:
        metadata.prediction_length = forecast_horizon
    return df, metadata


def split_test_train_last_horizon(
    data: pd.DataFrame, context_length: int, prediction_length: int
) -> splitter.SplitTrainTest:
    trn = data.groupby("unique_id").head(-prediction_length)
    test = data.groupby("unique_id").tail(context_length + prediction_length)
    return splitter.SplitTrainTest(train=trn, test=test)


def split_test_val_train_last_horizon(
    data: pd.DataFrame, context_length: int, prediction_length: int
) -> splitter.SplitTrainTest:
    trn_val = data.groupby("unique_id").head(-prediction_length)
    trn = trn_val.groupby("unique_id").head(-prediction_length)
    val = trn_val.groupby("unique_id").tail(context_length + prediction_length)
    test = data.groupby("unique_id").tail(context_length + prediction_length)
    return splitter.SplitData(train=trn, validation=val, test=test)


class LastHorizonSplitter(splitter.Splitter):

    def split(
        self, data: pd.DataFrame, metadata: nnts.data.metadata.Metadata, *args, **kwargs
    ) -> splitter.SplitData:
        """
        Splits the data into train, validation, and test sets based on the provided metadata.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be split.
            metadata (metadata.Metadata): The metadata object containing information about the split.

        Returns:
            splitter.SplitData: An object containing the train, validation, and test sets.
        """

        trn_val = data.groupby("unique_id").head(-metadata.prediction_length)
        trn = trn_val.groupby("unique_id").head(-metadata.prediction_length)
        val = trn_val.groupby("unique_id").tail(
            metadata.context_length + metadata.prediction_length
        )
        test = data.groupby("unique_id").tail(
            metadata.context_length + metadata.prediction_length
        )
        return splitter.SplitData(train=trn, validation=val, test=test)


def slice_rows(group: pd.DataFrame, start, end) -> pd.DataFrame:
    return group.iloc[start:end]


class FixedSizeSplitter(splitter.Splitter):

    def split(
        self, data: pd.DataFrame, train_size: int, val_size: int, test_size: int
    ) -> splitter.SplitData:
        trn = data.groupby("unique_id").head(train_size)
        val = (
            data.groupby("unique_id")
            .apply(slice_rows, train_size, train_size + val_size)
            .reset_index(drop=True)
        )
        test = data.groupby("unique_id").tail(test_size)
        return splitter.SplitData(train=trn, validation=val, test=test)


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
