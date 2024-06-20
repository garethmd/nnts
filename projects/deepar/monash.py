from datetime import datetime

import loader
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

BASE_DIR = "projects/deepar/data/"

# The name of the column containing time series values after loading data from the .tsf file into a dataframe
VALUE_COL_NAME = "series_value"

# The name of the column containing timestamps after loading data from the .tsf file into a dataframe
TIME_COL_NAME = "start_timestamp"

# Seasonality values corresponding with the frequencies: minutely, 10_minutes, half_hourly, hourly, daily, weekly, monthly, quarterly and yearly
# Consider multiple seasonalities for frequencies less than daily
SEASONALITY_MAP = {
    "minutely": [1440, 10080, 525960],
    "10_minutes": [144, 1008, 52596],
    "half_hourly": [48, 336, 17532],
    "hourly": [24, 168, 8766],
    "daily": 7,
    "weekly": 365.25 / 7,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}

# Frequencies used by GluonTS framework
FREQUENCY_MAP = {
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


def get_deep_nn_forecasts(
    dataset_name,
    lag,
    input_file_name,
    method,
    external_forecast_horizon=None,
    integer_conversion=False,
):
    print("Started loading " + dataset_name)

    (
        df,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = loader.convert_tsf_to_dataframe(
        BASE_DIR + input_file_name, "NaN", VALUE_COL_NAME
    )

    train_series_list = []
    test_series_list = []
    train_series_full_list = []
    test_series_full_list = []
    final_forecasts = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    if isinstance(seasonality, list):
        seasonality = min(seasonality)  # Use to calculate MASE

    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        if external_forecast_horizon is None:
            raise Exception("Please provide the required forecast horizon")
        else:
            forecast_horizon = external_forecast_horizon

    start_exec_time = datetime.now()

    for index, row in df.iterrows():
        if TIME_COL_NAME in df.columns:
            train_start_time = row[TIME_COL_NAME]
        else:
            train_start_time = datetime.strptime(
                "1900-01-01 00-00-00", "%Y-%m-%d %H-%M-%S"
            )  # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

        series_data = row[VALUE_COL_NAME]

        # Creating training and test series. Test series will be only used during evaluation
        train_series_data = series_data[: len(series_data) - forecast_horizon]
        test_series_data = series_data[
            (len(series_data) - forecast_horizon) : len(series_data)
        ]

        train_series_list.append(train_series_data)
        test_series_list.append(test_series_data)

        # We use full length training series to train the model as we do not tune hyperparameters
        train_series_full_list.append(
            {
                FieldName.TARGET: train_series_data,
                FieldName.START: pd.Timestamp(train_start_time),
            }
        )

        test_series_full_list.append(
            {
                FieldName.TARGET: series_data,
                FieldName.START: pd.Timestamp(train_start_time),
            }
        )

    train_ds = ListDataset(train_series_full_list, freq=freq)
    test_ds = ListDataset(test_series_full_list, freq=freq)
    return train_ds, test_ds
