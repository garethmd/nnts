from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

import nnts
import nnts.lags
import nnts.torch.preprocessing


@dataclass
class BaseScenario:
    prediction_length: int
    conts: list = field(default_factory=list)
    seed: int = 42

    def copy(self):
        return self.__class__(
            prediction_length=self.prediction_length,
            conts=self.conts.copy(),
            seed=self.seed,
        )


@dataclass
class LagScenario(BaseScenario):
    # covariates: int = field(init=False)
    dataset: str = ""
    lag_seq: List[int] = field(default_factory=list)
    scaled_covariates: List[str] = field(default_factory=list)

    def copy(self):
        return LagScenario(
            prediction_length=self.prediction_length,
            conts=self.conts.copy(),
            seed=self.seed,
            lag_seq=self.lag_seq.copy(),
            scaled_covariates=self.scaled_covariates.copy(),
        )

    def scaled_covariate_names(self):
        return "-".join(self.scaled_covariates)

    @property
    def name(self):
        return f"cov-{self.scaled_covariate_names()}-lags-{len(self.lag_seq)}-ds-{self.dataset}-seed-{self.seed}"

    @property
    def cat_idx(self):
        return (
            self.scaled_covariates.index("unique_id_0") + 1
            if "unique_id_0" in self.scaled_covariates
            else None
        )

    @property
    def month_idx(self):
        return (
            self.scaled_covariates.index("month") + 1
            if "month" in self.scaled_covariates
            else None
        )


def create_time_features(df_orig: pd.DataFrame):
    df_orig["day_of_week"] = df_orig["ds"].dt.day_of_week / 6.0 - 0.5
    df_orig["hour"] = df_orig["ds"].dt.hour / 23.0 - 0.5
    df_orig["dayofyear"] = df_orig["ds"].dt.dayofyear / 365.0 - 0.5
    df_orig["dayofmonth"] = df_orig["ds"].dt.day / 30.0 - 0.5

    # GluonTS uses the following code to generate the age covariate
    # age = np.log10(2.0 + np.arange(length, dtype=self.dtype))
    # length = the length of the time series. In GluonTS this length depends on the length of the training set and test set.
    # but we do it once on the complete dataset.
    # Also note that this doesn't align to the most recent time point, but to the first time point which
    # intuitively doesn't make sense.
    df_orig["month"] = (df_orig["ds"] + pd.DateOffset(months=1)).dt.month
    max_min_scaler = nnts.torch.preprocessing.MaxMinScaler()
    max_min_scaler.fit(df_orig, ["month"])
    df_orig = max_min_scaler.transform(df_orig, ["month"])

    df_orig["unix_timestamp"] = np.log10(
        2.0 + df_orig.groupby("unique_id").cumcount() + 1
    )
    return df_orig


def create_dummy_unique_ids(df_orig: pd.DataFrame):
    df_orig["unique_id_0"] = 0.0
    df_orig["static_cont"] = 0.0
    return df_orig


def create_lag_seq(freq: str):
    lag_seq = nnts.lags.get_lags_for_frequency(freq)
    lag_seq = [lag - 1 for lag in lag_seq if lag > 1]
    return lag_seq
