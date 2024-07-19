import argparse
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sized

import deepar
import features
import gluonts.time_feature
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import trainers as project_trainers
from torch.distributions import (AffineTransform, Distribution,
                                 TransformedDistribution)
from torch.utils.data import Sampler

import nnts.metrics
import nnts.torch.datasets
import nnts.torch.models
import nnts.torch.utils
from nnts import datasets, loggers, trainers, utils

"""
The purpose of this script is to train and evaluate a deepar model that is as close as possible to the gluonts implementation.
"""


@dataclass
class LagScenario(features.BaseScenario):
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


def create_scenarios(metadata: datasets.Metadata, lag_seq):
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


def create_lag_scenarios(metadata: datasets.Metadata, lag_seq: List[int]):
    if metadata.freq == "1H":
        conts = [
            "hour",
            "day_of_week",
            "dayofmonth",
            "dayofyear",
            "unix_timestamp",
            "unique_id_0",
            # "static_cont",
        ]
    elif metadata.freq == "ME":
        conts = [
            "month",
            "unix_timestamp",
            "unique_id_0",
        ]
    else:
        raise ValueError(f"Unsupported frequency {metadata.freq}")

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


def masked_mean_abs_scaling(
    seq: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-10, dim: int = 1
):
    if mask is None:
        mask = torch.ones_like(seq)
    if len(mask.shape) == 2 and len(seq.shape) == 3:
        mask = mask[:, :, None]

    if len(mask.shape) != len(seq.shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match sequence shape {seq.shape}"
        )
    seq_sum = (seq * mask).abs().sum(dim, keepdim=True)
    item_count = mask.sum(dim, keepdim=True).clamp(min=1)

    scale = seq_sum / item_count
    scale = torch.clamp(scale, min=eps)
    return scale


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
        self.main = nn.ModuleList(
            [nn.Linear(hidden_size, output_size) for _ in range(StudentTHead.PARAMS)]
        )

    def forward(self, x: torch.tensor, target_scale: torch.tensor):
        df, loc, scale = tuple(self.main[i](x) for i in range(StudentTHead.PARAMS))
        df = 2.0 + F.softplus(df)
        epsilon = torch.finfo(scale.dtype).eps
        scale = F.softplus(scale).clamp_min(epsilon)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


def distr_nll(distr: td.Distribution, target: torch.Tensor) -> torch.Tensor:
    nll = -distr.log_prob(target.squeeze(-1))
    nll = nll.mean(dim=(1,))
    return nll.mean()


def main(
    model_name: str,
    dataset_name: str,
    data_path: str,
    base_model_name: str,
    results_path: str,
):

    # Load data
    df_orig, metadata = datasets.load_dataset(dataset_name)
    PATH = os.path.join(results_path, model_name, metadata.dataset)

    params = utils.Hyperparams(
        optimizer=torch.optim.Adam,
        loss_fn=distr_nll,
        scheduler=utils.Scheduler.REDUCE_LR_ON_PLATEAU,
        training_method=utils.TrainingMethod.TEACHER_FORCING,
        batch_size=32,
        batches_per_epoch=50,
    )

    # Calculate next month and unix timestamp
    df_orig = features.create_time_features(df_orig)
    df_orig = features.create_dummy_unique_ids(df_orig)
    lag_seq = gluonts.time_feature.lag.get_lags_for_frequency(metadata.freq)
    lag_seq = [lag - 1 for lag in lag_seq if lag > 1]
    scenario_list = create_lag_scenarios(metadata, lag_seq)

    for scenario in scenario_list:
        nnts.torch.utils.seed_everything(scenario.seed)
        df = df_orig.copy()
        context_length = metadata.context_length + max(scenario.lag_seq)

        dataset_options = {
            "context_length": metadata.context_length,
            "prediction_length": metadata.prediction_length,
            "conts": scenario.conts,
            "lag_seq": scenario.lag_seq,
        }

        trn_dl, test_dl = nnts.torch.utils.create_dataloaders(
            df,
            datasets.split_test_train_last_horizon,
            context_length,
            metadata.prediction_length,
            Dataset=nnts.torch.datasets.TimeseriesLagsDataset,
            dataset_options=dataset_options,
            Sampler=nnts.torch.datasets.TimeSeriesSampler,
        )

        logger = loggers.LocalFileRun(
            project=f"{model_name}-{metadata.dataset}",
            name=scenario.name,
            config={
                **params.__dict__,
                **metadata.__dict__,
                **scenario.__dict__,
            },
            path=PATH,
        )
        if model_name == "deepar-point":
            net = deepar.DeepARPoint(
                nnts.torch.models.LinearModel,
                params,
                masked_mean_abs_scaling,
                1,
                lag_seq=lag_seq,
                scaled_features=scenario.scaled_covariates,
                context_length=metadata.context_length,
            )
            trner = project_trainers.TorchEpochTrainer(
                trainers.TrainerState(),
                net,
                params,
                metadata,
                os.path.join(PATH, f"{scenario.name}.pt"),
                F.l1_loss,
            )
        else:
            net = deepar.DistrDeepAR(
                StudentTHead,
                params,
                masked_mean_abs_scaling,
                1,
                lag_seq=lag_seq,
                scaled_features=scenario.scaled_covariates,
                context_length=metadata.context_length,
            )
            print(nnts.torch.utils.count_of_params_in(net))
            trner = project_trainers.TorchEpochTrainer(
                trainers.TrainerState(),
                net,
                params,
                metadata,
                os.path.join(PATH, f"{scenario.name}.pt"),
                loss_fn=distr_nll,
            )
        logger.configure(trner.events)

        evaluator = trner.train(trn_dl)

        y_hat, y = evaluator.evaluate(
            test_dl, scenario.prediction_length, metadata.context_length
        )
        test_metrics = nnts.metrics.calc_metrics(
            y_hat,
            y,
            nnts.metrics.calculate_seasonal_error(trn_dl, metadata.seasonality),
        )
        logger.log(test_metrics)
        print(test_metrics)
        logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training and evaluation script."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepar-point",
        help="Name of the model.",
    )
    parser.add_argument(
        "--dataset", type=str, default="tourism_monthly", help="Name of the dataset."
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
        default="projects/deepar/baseline-results",
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
    )
