import os
import random

import numpy as np
import pandas as pd
import torch.optim

import nnts
import nnts.data
import nnts.loggers
import nnts.metrics
import nnts.torch.datasets
import nnts.torch.models
import nnts.torch.trainers
import nnts.torch.utils
from nnts import datasets, utils

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
model_name = "dlinear"
data_path = os.path.join(BASE_PATH, "data")
dataset_name = "ETTh1"
results_path = os.path.join(BASE_PATH, "nb-results")
metadata_path = os.path.join(BASE_PATH, "informer.json")

metadata = datasets.load_metadata(dataset_name, path=metadata_path)
datafile_path = os.path.join(data_path, metadata.filename)
PATH = os.path.join(results_path, model_name, metadata.dataset)
df = pd.read_csv(datafile_path)
df = df.rename({"date": "ds"}, axis="columns")
FEATURES = ["OT"]  # ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
utils.makedirs_if_not_exists(PATH)

# DLinear paper params
params = utils.Hyperparams(
    optimizer=torch.optim.Adam,
    loss_fn=torch.nn.L1Loss(),
    batch_size=32,
    batches_per_epoch=None,
    training_method=utils.TrainingMethod.DMS,
    model_file_path="logs",
    epochs=100,
    scheduler=utils.Scheduler.REDUCE_LR_ON_PLATEAU,
    lr=0.005,
    weight_decay=0.0,
)

df = pd.melt(
    df,
    id_vars=["ds"],
    value_vars=FEATURES,
    var_name="unique_id",
    value_name="y",
)

for unique_id in FEATURES:
    local_df = df[df["unique_id"] == unique_id]

    # trn_length = int(24 * 365.25)
    # val_test_length = int(24 * 365.25 * (4 / 12))
    trn_length = 8640
    val_test_length = 3216
    split_data = datasets.split_test_val_train(
        local_df,
        trn_length,
        val_test_length,
        val_test_length,
        context_length=metadata.context_length,
    )

    for seed in [2021, 2022, 2023, 2024, 2025]:
        nnts.torch.utils.seed_everything(seed)

        dataset_options = {
            "context_length": metadata.context_length,
            "prediction_length": metadata.prediction_length,
            "conts": [],
        }

        trn_dl, val_dl, test_dl = nnts.torch.utils.create_dataloaders_from_split_data(
            split_data,
            Dataset=nnts.torch.datasets.TimeseriesDataset,
            dataset_options=dataset_options,
            # Sampler=nnts.torch.datasets.TimeSeriesSampler,
            batch_size=params.batch_size,
            transforms=[nnts.torch.preprocessing.StandardScaler()],
        )

        net = nnts.torch.models.NLinear(metadata, individual=False)
        trner = nnts.torch.trainers.ValidationTorchEpochTrainer(net, params, metadata)
        evaluator = trner.train(trn_dl, val_dl)
        y_hat, y = evaluator.evaluate(
            test_dl, metadata.prediction_length, metadata.context_length
        )
        test_metrics = nnts.metrics.calc_metrics(
            y_hat,
            y,
            nnts.metrics.calculate_seasonal_error(trn_dl, metadata.seasonality),
        )
        print(test_metrics)
