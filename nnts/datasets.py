import json
import os
from collections import namedtuple
from typing import Tuple

import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from pydantic import BaseModel

import nnts.data.tsf as tsf

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
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
    url: str = None


def unpack(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    try:
        timesteps = pd.date_range(
            df["start_timestamp"], periods=len(df["series_value"]), freq=freq
        )
    except OutOfBoundsDatetime:
        timesteps = pd.date_range(
            pd.Timestamp("1970-01-01"),
            periods=len(df["series_value"]),
            freq=freq,
            unit="s",
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


def load_metadata(
    dataset: str,
    path: str = None,
) -> Metadata:
    # Get the directory of the current script
    with open(path) as f:
        data = json.load(f)
    if dataset not in data:
        raise ValueError(
            f"Dataset {dataset} not found in metadata {path} choose from {data.keys()}"
        )
    return Metadata(**data[dataset])


def load_dataset(
    name: str, repository: str = "monash"
) -> Tuple[pd.DataFrame, Metadata]:
    """
    Load dataset and its metadata.

    Parameters:
    - name (str): Name of the dataset to load.
    - repository (str): Name of the repository where the dataset metadata is stored. Default is 'monash'.

    Returns:
    - Tuple[pd.DataFrame, Metadata]: A tuple containing the loaded DataFrame and its metadata.
    """
    path = os.path.join(DATA_PATH, f"{repository}.json")
    metadata = load_metadata(name, path=path)

    df, *_ = read_tsf(metadata.filename, metadata.url)
    return df, metadata


def load(
    dataset_name: str, data_path: str, metadata_filename: str
) -> Tuple[pd.DataFrame, Metadata]:

    metadata_path = os.path.join(data_path, metadata_filename)
    metadata = load_metadata(dataset_name, path=metadata_path)
    datafile_path = os.path.join(data_path, metadata.filename)
    df, freq, forecast_horizon, *_ = read_tsf(datafile_path)
    metadata.freq = freq
    if forecast_horizon is not None:
        metadata.prediction_length = forecast_horizon
    return df, metadata


def split_test_train_last_horizon(
    data: pd.DataFrame, context_length: int, prediction_length: int
) -> SplitTrainTest:
    trn = data.groupby("unique_id").head(-prediction_length)
    test = data.groupby("unique_id").tail(context_length + prediction_length)
    return SplitTrainTest(train=trn, test=test)


def split_test_val_train_last_horizon(
    data: pd.DataFrame, context_length: int, prediction_length: int
) -> SplitTrainTest:
    trn_val = data.groupby("unique_id").head(-prediction_length)
    trn = trn_val.groupby("unique_id").head(-prediction_length)
    val = trn_val.groupby("unique_id").tail(context_length + prediction_length)
    test = data.groupby("unique_id").tail(context_length + prediction_length)
    return SplitData(train=trn, validation=val, test=test)


def slice_rows(group: pd.DataFrame, start, end) -> pd.DataFrame:
    return group.iloc[start:end]
