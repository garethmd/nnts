import argparse
import os
from dataclasses import dataclass, field
from typing import List

import features
import torch.nn.functional as F
import torch.optim

import nnts
import nnts.data
import nnts.datasets
import nnts.loggers
import nnts.metrics
import nnts.torch
import nnts.torch.datasets
import nnts.torch.models
import nnts.torch.preprocessing
import nnts.torch.trainers
import nnts.torch.utils
import nnts.trainers
from nnts import datasets, utils


@dataclass
class SchedulerScenario(features.BaseScenario):
    # covariates: int = field(init=False)
    dataset: str = ""
    lag_seq: List[int] = field(default_factory=list)
    scaled_covariates: List[str] = field(default_factory=list)
    scheduler_name: str = ""

    def copy(self):
        return SchedulerScenario(
            prediction_length=self.prediction_length,
            conts=self.conts.copy(),
            seed=self.seed,
            lag_seq=self.lag_seq.copy(),
            scaled_covariates=self.scaled_covariates.copy(),
            scheduler_name=self.scheduler_name,
        )

    def scaled_covariate_names(self):
        return "-".join(self.scaled_covariates)

    @property
    def name(self):
        return f"scheduler-{self.scheduler_name}-ds-{self.dataset}-seed-{self.seed}"

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


# EXPERIMENT SETUP
def generate_one_hot_matrix(n):
    # Total number of rows in the matrix
    num_rows = 2**n

    # Initialize the matrix
    one_hot_matrix = []

    # Generate each combination of binary values
    for i in range(num_rows):
        # Convert the number to its binary representation and fill with leading zeros
        binary_representation = format(i, f"0{n}b")
        # Convert the binary string to a list of integers
        one_hot_row = [int(bit) for bit in binary_representation]
        # Append the one-hot row to the matrix
        one_hot_matrix.append(one_hot_row)

    return one_hot_matrix


def create_scenarios(metadata, lag_seq):
    scaled_covariates = [
        "unix_timestamp",
        "unique_id_0",
        "static_cont",
        nnts.torch.models.deepar.FEAT_SCALE,
    ]

    # Example usage for n=5
    n = len(scaled_covariates)
    scaled_covariate_selection_matrix = generate_one_hot_matrix(n)
    scenario_list: List[features.LagScenario] = []

    for seed in [42, 43, 44, 45, 46]:
        for row in scaled_covariate_selection_matrix:
            selected_combination = [
                covariate
                for covariate, select in zip(scaled_covariates, row)
                if select == 1
            ]
            scenario_list.append(
                SchedulerScenario(
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


def create_scheduler_scenarios(metadata: datasets.Metadata, lag_seq, scheduler_name):
    conts = []
    scenario_list: List[SchedulerScenario] = []

    # BASELINE
    scenario_list = []
    for seed in [42, 43, 44, 45, 46]:
        scenario = SchedulerScenario(
            metadata.prediction_length,
            conts=conts,
            scaled_covariates=conts,
            lag_seq=lag_seq,
            seed=seed,
            dataset=metadata.dataset,
            scheduler_name=scheduler_name,
        )
        scenario_list.append(scenario)
    return scenario_list


def main(
    model_name: str,
    dataset_name: str,
    results_path: str,
):
    # Set up paths and load metadata
    df_orig, metadata = nnts.datasets.load_dataset(dataset_name)

    # Set parameters
    params = utils.GluonTsDefaultWithOneCycle(
        optimizer=torch.optim.AdamW, loss_fn=nnts.torch.models.deepar.distr_nll
    )

    # Calculate next month and unix timestamp
    df_orig = features.create_dummy_unique_ids(df_orig)
    lag_seq = features.create_lag_seq(metadata.freq)
    scenario_list = create_scheduler_scenarios(metadata, lag_seq, "ONE_CYCLE")

    for scenario in scenario_list:
        nnts.torch.utils.seed_everything(scenario.seed)
        df = df_orig.copy()
        lag_processor = nnts.torch.preprocessing.LagProcessor(scenario.lag_seq)

        context_length = metadata.context_length + max(scenario.lag_seq)
        # end of experiment setup

        logger = nnts.loggers.LocalFileRun(
            project=f"{model_name}-{metadata.dataset}-scheduler",
            name=scenario.name,
            config={
                **params.__dict__,
                **metadata.__dict__,
                **scenario.__dict__,
            },
            path=os.path.join(results_path, model_name, metadata.dataset),
        )

        dataset_options = {
            "context_length": metadata.context_length,
            "prediction_length": metadata.prediction_length,
            "conts": scenario.conts,
            "lag_seq": scenario.lag_seq,
        }

        trn_dl, test_dl = nnts.torch.utils.create_dataloaders(
            df,
            nnts.datasets.split_test_train_last_horizon,
            context_length,
            metadata.prediction_length,
            Dataset=nnts.torch.datasets.TimeseriesLagsDataset,
            dataset_options=dataset_options,
            Sampler=nnts.torch.datasets.TimeSeriesSampler,
        )

        net = create_net(model_name, metadata, params, scenario, lag_processor)
        trner = nnts.torch.trainers.TorchEpochTrainer(
            net,
            params,
            metadata,
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


def create_net(model_name, metadata, params, scenario, lag_processor):
    if model_name == "better-deepar-studentt":
        net = nnts.torch.models.DistrDeepAR(
            nnts.torch.models.deepar.StudentTHead,
            params,
            nnts.torch.preprocessing.masked_mean_abs_scaling,
            1,
            lag_processor=lag_processor,
            scaled_features=scenario.scaled_covariates,
            context_length=metadata.context_length,
            cat_idx=scenario.cat_idx,
            seq_cat_idx=scenario.month_idx,
        )
    elif model_name == "better-deepar-point":
        net = nnts.torch.models.DeepARPoint(
            nnts.torch.models.LinearModel,
            params,
            nnts.torch.preprocessing.masked_mean_abs_scaling,
            1,
            lag_processor=lag_processor,
            scaled_features=scenario.scaled_covariates,
            context_length=metadata.context_length,
            cat_idx=scenario.cat_idx,
            seq_cat_idx=scenario.month_idx,
        )

    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training and evaluation script."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="better-deepar-studentt",
        help="Name of the model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tourism_monthly",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="projects/deepar/scheduler-results",
        help="Path to the results directory.",
    )
    args = parser.parse_args()

    main(
        args.model,
        args.dataset,
        args.results_path,
    )
