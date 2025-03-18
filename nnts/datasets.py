import json
import os
import warnings
from collections import namedtuple
from typing import Tuple

import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from pydantic import BaseModel

import nnts.data.tsf as tsf

warnings.simplefilter(action="ignore", category=FutureWarning)

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
    """Class for storing metadata information about a dataset.

    Attributes:
    filename (str): The name of the file containing the dataset.
    dataset (str): The name of the dataset.
    context_length (int): The number of past observations used as context for making predictions.
    prediction_length (int): The number of future observations to predict.
    freq (str): The frequency of the time series data (e.g., 'D' for daily, 'H' for hourly).
    seasonality (int): The seasonal period of the time series data.
    url (str, optional): The URL where the dataset can be accessed. Defaults to None.
    """

    filename: str
    dataset: str
    context_length: int
    prediction_length: int
    freq: str
    seasonality: int
    url: str = None
    context_lengths: list = None
    multivariate: bool = False


def unpack(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
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
        data={"y": df["series_value"].to_numpy(), "ds": timesteps.astype(str)}
    )
    unpacked_df["unique_id"] = df["series_name"]
    return unpacked_df


def read_tsf(path: str, url: str = None) -> pd.DataFrame:
    """Reads a time series forecasting (TSF) file from a specified path or URL and returns
    the data as a pandas DataFrame along with metadata.

    Parameters
    ----------
    path: str
        The file path to the TSF file.
    url: :obj: `str` optional
        The URL to the TSF file. If specified, the file will be downloaded from the URL. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the time series data with the necessary transformations applied.
    str
        The frequency of the time series data.
    int
        The forecast horizon.
    bool
        Indicator whether the data contains missing values.
    bool
        Indicator whether the time series data have equal lengths.

    Example
    -------
    >>> df, freq, forecast_horizon, contain_missing_values, contain_equal_length = read_tsf('path/to/file.tsf')
    >>> print(df.head())
    >>> print(freq)
    >>> print(forecast_horizon)
    >>> print(contain_missing_values)
    >>> print(contain_equal_length)
    """
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
    """
    Loads metadata for a specified dataset from a JSON file.

    Parameters
    ----------
    dataset: str
        The name of the dataset for which to load metadata.
    path: :obj: `str` optional
        The path to the JSON file containing metadata. If not specified, a default path should be provided by the caller.

    Returns
    -------
    Metadata
        An instance of the Metadata class initialized with the data corresponding to the specified dataset.

    Raises
    ------
    ValueError
        If the specified dataset is not found in the metadata JSON file.

    Example
    -------
    >>> metadata = load_metadata('my_dataset', 'path/to/metadata.json')
    >>> print(metadata)
    Metadata(attribute1=value1, attribute2=value2, ...)
    """
    with open(path) as f:
        data = json.load(f)
    if dataset not in data:
        raise ValueError(
            f"Dataset {dataset} not found in metadata {path} choose from: {', '.join(data.keys())}"
        )
    return Metadata(**data[dataset])


def load_dataset(
    name: str, repository: str = "monash"
) -> Tuple[pd.DataFrame, Metadata]:
    """
    Load dataset and its metadata.

    Parameters
    ----------
    name: str
        Name of the dataset to load.
    repository: str
        Name of the repository where the dataset metadata is stored. Default is 'monash'.

    Returns
    -------
    Tuple[pd.DataFrame, Metadata]
        A tuple containing the loaded DataFrame and its metadata.
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
    """
    Splits the given time series data into training and testing sets based on the
    specified context length and prediction length.

    Parameters
    ----------
    data: pd.DataFrame
        The input time series data containing a 'unique_id' column to identify different series.
    context_length: int
        The number of past observations to use as context for making predictions.
    prediction_length: int
        The number of future observations to predict.

    Returns
    -------
    SplitTrainTest
        A named tuple containing two DataFrames:
            - train: The training set, which excludes the last 'prediction_length' observations for each unique series.
            - test: The testing set, which includes the last 'context_length + prediction_length' observations for each unique series.

    """
    trn = data.groupby("unique_id").head(-prediction_length)
    test = data.groupby("unique_id").tail(context_length + prediction_length)
    return SplitTrainTest(train=trn, test=test)


def split_test_val_train_last_horizon(
    data: pd.DataFrame, context_length: int, prediction_length: int
) -> SplitTrainTest:
    """
    Splits the given time series data into training, validation, and testing sets based
    on the specified context length and prediction length.

    Parameters
    ----------
    data: pd.DataFrame
        The input time series data containing a 'unique_id' column to identify different series.
    context_length: int
        The number of past observations to use as context for making predictions.
    prediction_length: int
        The number of future observations to predict.

    Returns
    -------
    SplitData
        A named tuple containing three DataFrames:
            - train: The training set, which excludes the last '2 * prediction_length' observations for each unique series.
            - validation: The validation set, which excludes the last 'prediction_length' and includes a time-range of context_length + prediction_length or each unique series.
            - test: The testing set, which includes the last 'context_length + prediction_length' observations for each unique series.
    """
    trn_val = data.groupby("unique_id").head(-prediction_length)
    trn = trn_val.groupby("unique_id").head(-prediction_length)
    val = trn_val.groupby("unique_id").tail(context_length + prediction_length)
    test = data.groupby("unique_id").tail(context_length + prediction_length)
    return SplitData(train=trn, validation=val, test=test)


def split_test_val_train(
    data: pd.DataFrame,
    trn_length: int,
    val_length: int,
    test_length: int,
    context_length: int = 0,
    prediction_length: int = 336,
) -> SplitTrainTest:
    data = data.groupby("unique_id").head(trn_length + val_length + test_length)
    trn_val = data.groupby("unique_id").head(trn_length + val_length - context_length)
    trn = trn_val.groupby("unique_id").head(trn_length)
    val = trn_val.groupby("unique_id").tail(val_length + prediction_length)
    test = data.groupby("unique_id").tail(test_length + prediction_length)
    return SplitData(train=trn, validation=val, test=test)
