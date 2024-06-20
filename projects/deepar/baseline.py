import argparse
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import deepar
import features
import gluondata
import gluonts
import numpy as np
import pandas as pd
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import trainers

import nnts
import nnts.data
import nnts.data.datasets
import nnts.experiments
import nnts.loggers
import nnts.metrics
import nnts.models
import nnts.pandas
import nnts.torch.data
import nnts.torch.data.datasets
import nnts.torch.data.preprocessing
import nnts.torch.models
import nnts.torch.utils


def calculate_seasonal_error(trn_dl: Iterable, metadata: nnts.data.metadata.Metadata):
    se_list = []
    for batch in trn_dl:
        past_data = batch["target"]
        se = nnts.metrics.gluon_metrics.calculate_seasonal_error(
            past_data, metadata.freq, metadata.seasonality
        )
        se_list.append(se)
    return torch.tensor(se_list).unsqueeze(1)


def create_time_features(df_orig: pd.DataFrame):
    df_orig["day_of_week"] = df_orig["ds"].dt.day_of_week
    df_orig["hour"] = df_orig["ds"].dt.hour
    df_orig["week"] = df_orig["ds"].dt.isocalendar().week
    df_orig["week"] = df_orig["week"].astype(np.float32)

    # GluonTS uses the following code to generate the age covariate
    # age = np.log10(2.0 + np.arange(length, dtype=self.dtype))
    # length = the length of the time series. In GluonTS this length depends on the length of the training set and test set.
    # but we do it once on the complete dataset.
    # Also note that this doesn't align to the most recent time point, but to the first time point which
    # intuitively doesn't make sense.
    df_orig["month"] = (df_orig["ds"] + pd.DateOffset(months=1)).dt.month

    df_orig["unix_timestamp"] = np.log10(
        2.0 + df_orig.groupby("unique_id").cumcount() + 1
    )
    return df_orig


@dataclass
class LagScenario(nnts.experiments.scenarios.BaseScenario):
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


def create_scenarios(metadata, lag_seq):
    scaled_covariates = ["month", "unix_timestamp", nnts.torch.models.deepar.FEAT_SCALE]
    scaled_covariate_selection_matrix = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    scenario_list: List[LagScenario] = []

    for seed in [42, 43, 44, 45, 46]:
        for row in scaled_covariate_selection_matrix:
            selected_combination = [
                covariate
                for covariate, select in zip(scaled_covariates, row)
                if select == 1
            ]
            scenario_list.append(
                LagScenario(
                    metadata.prediction_length,
                    conts=[
                        cov
                        for cov in selected_combination
                        if cov != nnts.torch.models.deepar.FEAT_SCALE
                    ],
                    scaled_covariates=selected_combination,
                    lag_seq=lag_seq,
                    seed=seed,
                    dataset=metadata.dataset,
                )
            )
    return scenario_list


def create_lag_scenarios(metadata, lag_seq):
    conts = [
        "hour",
        "day_of_week",
        "dayofmonth",
        "dayofyear",
        "unix_timestamp",
        "unique_id_0",
        # "static_cont",
    ]

    scenario_list: List[LagScenario] = []

    # BASELINE
    scenario_list = []
    for seed in [42, 43, 44, 45, 46]:
        scenario = LagScenario(
            metadata.prediction_length,
            conts=conts,
            scaled_covariates=conts
            + [
                nnts.torch.models.deepar.FEAT_SCALE,
            ],
            lag_seq=lag_seq,
            seed=seed,
            dataset=metadata.dataset,
        )
        scenario_list.append(scenario)
    return scenario_list


from torch.distributions import AffineTransform, Distribution, TransformedDistribution


class AffineTransformed(TransformedDistribution):
    """
    Represents the distribution of an affinely transformed random variable.

    This is the distribution of ``Y = scale * X + loc``, where ``X`` is a
    random variable distributed according to ``base_distribution``.

    Parameters
    ----------
    base_distribution
        Original distribution
    loc
        Translation parameter of the affine transformation.
    scale
        Scaling parameter of the affine transformation.
    """

    def __init__(self, base_distribution: Distribution, loc=None, scale=None):
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc

        super().__init__(base_distribution, [AffineTransform(self.loc, self.scale)])

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale**2

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()


from typing import Iterator, Optional, Sized

from torch.utils.data import Sampler


class TimeSeriesSampler(Sampler[int]):

    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None):
        self.data_source = data_source
        self._num_samples = len(data_source) if num_samples is None else num_samples

        to = torch.cat(
            [
                data_source.shifted_cum_lengths,
                torch.tensor([len(data_source)]),
            ]
        )[1:]
        self.ranges = torch.stack([data_source.shifted_cum_lengths, to]).T

    def __iter__(self) -> Iterator[int]:
        count = 0
        while True:
            for i in range(self.ranges.shape[0]):
                count += 1
                if count > self._num_samples:
                    return
                start, end = self.ranges[i]
                val = torch.randint(start, end, size=(1,))
                yield val

    def __len__(self) -> int:
        return self._num_samples


class StudentTHead(nn.Module):
    """
    This model outputs a studentT distribution.
    """

    PARAMS = 3

    def __init__(self, hidden_size: int, output_size: int):
        super(StudentTHead, self).__init__()

        # self.main = nn.Sequential(
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, output_size * StudentTHead.PARAMS),
        # )
        # self.main = nn.Linear(hidden_size, output_size * StudentTHead.PARAMS)
        self.main = nn.ModuleList(
            [nn.Linear(hidden_size, output_size) for _ in range(StudentTHead.PARAMS)]
        )

    def forward(self, x: torch.tensor, target_scale: torch.tensor):
        df, loc, scale = tuple(self.main[i](x) for i in range(StudentTHead.PARAMS))
        df = 2.0 + F.softplus(df)
        epsilon = torch.finfo(scale.dtype).eps
        scale = F.softplus(scale).clamp_min(epsilon)
        # student_t = td.StudentT(df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1))
        # return AffineTransformed(student_t, None, target_scale.squeeze(-1))
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


def distr_nll(distr: td.Distribution, target: torch.Tensor) -> torch.Tensor:
    nll = -distr.log_prob(target.squeeze(-1))
    nll = nll.mean(dim=(1,))
    return nll.mean()


def test_dataloader(
    model_name: str,
    dataset_name: str,
    data_path: str,
    base_model_name: str,
    results_path: str,
):
    # Set up paths and load metadata

    metadata_path = os.path.join(data_path, f"{base_model_name}-monash.json")
    metadata = nnts.data.metadata.load(dataset_name, path=metadata_path)
    datafile_path = os.path.join(data_path, metadata.filename)
    PATH = os.path.join(results_path, model_name, metadata.dataset)

    # Load data
    df_orig, *_ = nnts.pandas.read_tsf(datafile_path)
    params = nnts.models.Hyperparams()

    # Create output directory if it doesn't exist
    nnts.loggers.makedirs_if_not_exists(PATH)

    # Set parameters
    params.batch_size = 32
    params.batches_per_epoch = 50

    # Calculate next month and unix timestamp
    df_orig = features.create_time_features(df_orig)

    # Create dummy unique ids
    df_orig = create_dummy_unique_ids(df_orig)

    # Normalize data
    max_min_scaler = nnts.torch.data.preprocessing.MaxMinScaler()
    max_min_scaler.fit(df_orig, ["month"])
    df_orig = max_min_scaler.transform(df_orig, ["month"])

    lag_seq = gluonts.time_feature.lag.get_lags_for_frequency(metadata.freq)
    lag_seq = [lag - 1 for lag in lag_seq if lag > 1]

    scenario_list = create_lag_scenarios(metadata, lag_seq)

    params.training_method = nnts.models.hyperparams.TrainingMethod.TEACHER_FORCING

    for scenario in scenario_list[:1]:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        df = df_orig.copy()
        context_length = metadata.context_length + max(scenario.lag_seq)
        split_data = nnts.pandas.split_test_train_last_horizon(
            df, context_length, metadata.prediction_length
        )
        trn_dl, _ = nnts.data.create_trn_test_dataloaders(
            split_data,
            metadata,
            scenario,
            params,
            nnts.torch.data.datasets.TorchTimeseriesLagsDataLoaderFactory(),
            Sampler=None,
        )
        for batch in trn_dl:
            print(batch)


def remove_prefix(s, prefix):
    return s[len(prefix) :] if s.startswith(prefix) else s


def load_gluonts_weights(net):
    state_dict = torch.load(
        "/Users/garethdavies/Development/workspaces/nnts/projects/deepar/gluonts.pt"
    )
    rnn = {k: v for k, v in state_dict.items() if k.startswith("rnn")}
    net.decoder.load_state_dict(rnn)
    net.embbeder.load_state_dict({"weight": state_dict["embedder._embedders.0.weight"]})
    proj = {
        remove_prefix(k, "param_proj.proj."): v
        for k, v in state_dict.items()
        if k.startswith("param_proj")
    }
    net.distribution.main.load_state_dict(proj)
    return net


def main(
    model_name: str,
    dataset_name: str,
    data_path: str,
    base_model_name: str,
    results_path: str,
):
    # Set up paths and load metadata

    metadata_path = os.path.join(data_path, f"{base_model_name}-monash.json")
    metadata = nnts.data.metadata.load(dataset_name, path=metadata_path)
    datafile_path = os.path.join(data_path, metadata.filename)
    PATH = os.path.join(results_path, model_name, metadata.dataset)

    # Load data
    df_orig, *_ = nnts.pandas.read_tsf(datafile_path)
    params = nnts.models.Hyperparams()

    # Create output directory if it doesn't exist
    nnts.loggers.makedirs_if_not_exists(PATH)

    # Set parameters
    params.batch_size = 32
    params.batches_per_epoch = 50
    params.scheduler = nnts.models.hyperparams.Scheduler.REDUCE_LR_ON_PLATEAU
    params.training_method = nnts.models.hyperparams.TrainingMethod.TEACHER_FORCING

    # Calculate next month and unix timestamp
    df_orig = features.create_time_features(df_orig)
    df_orig = features.create_dummy_unique_ids(df_orig)

    # Normalize data
    max_min_scaler = nnts.torch.data.preprocessing.MaxMinScaler()
    max_min_scaler.fit(df_orig, ["month"])
    df_orig = max_min_scaler.transform(df_orig, ["month"])

    lag_seq = gluonts.time_feature.lag.get_lags_for_frequency(metadata.freq)
    lag_seq = [lag - 1 for lag in lag_seq if lag > 1]

    scenario_list = create_lag_scenarios(metadata, lag_seq)

    for scenario in scenario_list[:1]:
        nnts.torch.data.datasets.seed_everything(scenario.seed)
        df = df_orig.copy()
        context_length = metadata.context_length + max(scenario.lag_seq)
        split_data = nnts.pandas.split_test_train_last_horizon(
            df, context_length, metadata.prediction_length
        )
        trn_dl_alt, test_dl = nnts.data.create_trn_test_dataloaders(
            split_data,
            metadata,
            scenario,
            params,
            nnts.torch.data.datasets.TorchTimeseriesLagsDataLoaderFactory(),
            Sampler=TimeSeriesSampler,
        )
        trn_dl = gluondata.get_train_dl()
        # test_dl = gluonadapt.test_dl
        logger = nnts.loggers.LocalFileRun(
            project=f"{model_name}-{metadata.dataset}",
            name=scenario.name,
            config={
                **params.__dict__,
                **metadata.__dict__,
                **scenario.__dict__,
            },
            path=PATH,
        )
        net = deepar.DistrDeepAR(
            StudentTHead,
            params,
            nnts.torch.data.preprocessing.masked_mean_abs_scaling,
            1,
            lag_seq=lag_seq,
            scaled_features=scenario.scaled_covariates,
        )
        print(nnts.torch.utils.count_of_params_in(net))
        trner = trainers.TorchEpochTrainer(
            nnts.models.TrainerState(),
            net,
            params,
            metadata,
            os.path.join(PATH, f"{scenario.name}.pt"),
            loss_fn=distr_nll,
        )
        logger.configure(trner.events)

        evaluator = trner.train(trn_dl)

        # net = load_gluonts_weights(net)
        # evaluator = nnts.torch.models.trainers.TorchEvaluator(net)

        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )
        test_metrics = nnts.metrics.calc_metrics(
            y_hat, y, nnts.metrics.calculate_seasonal_error(trn_dl_alt, metadata)
        )
        logger.log(test_metrics)
        print(test_metrics)
        logger.finish()


def add_y_hat(df, y_hat, prediction_length):
    i = 0
    df_list = []
    for name, group in df.groupby("unique_id", sort=False):
        group["y_hat"] = None
        group["y_hat"][-prediction_length:] = y_hat[i].squeeze()
        i += 1
        df_list.append(group)
    return df_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training and evaluation script."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepar",
        help="Name of the model.",
    )
    parser.add_argument(
        "--dataset", type=str, default="traffic_hourly", help="Name of the dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="projects/deepar/data",
        help="Path to the data directory.",
    )

    parser.add_argument(
        "--results-path",
        type=str,
        default="ablation-results",
        help="Path to the results directory.",
    )
    args = parser.parse_args()

    main(
        args.model,
        args.dataset,
        args.data_path,
        "base-lstm",
        args.results_path,
    )
