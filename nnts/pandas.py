from typing import Tuple

import pandas as pd

import nnts.data.metadata as metadata
import nnts.data.splitter as splitter
import nnts.data.tsf as tsf


def read_tsf(path: str, url: str):
    return pd.DataFrame(tsf.handle_zip_file_http_request(path, url, tsf.read_tsf))


def unpack(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    timesteps = pd.date_range(
        df["start_timestamp"], periods=len(df["series_value"]), freq=freq
    )
    unpacked_df = pd.DataFrame(
        data={"y": df["series_value"].to_numpy(), "ds": timesteps}
    )
    unpacked_df["unique_id"] = df["series_name"]
    return unpacked_df


class LastHorizonSplitter(splitter.Splitter):

    def split(
        self, data: pd.DataFrame, metadata: metadata.Metadata, *args, **kwargs
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


def load_data(m: metadata.Metadata) -> Tuple[pd.DataFrame]:
    datai = tsf.convert_tsf_to_dataframe(m.path)
    df = pd.DataFrame(datai[0])
    df = pd.concat([tsf.unpack(df.iloc[x], freq=m.freq) for x in range(len(df))])
    return df
