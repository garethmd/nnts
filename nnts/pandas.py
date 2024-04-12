import pandas as pd

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
