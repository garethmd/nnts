import warnings
from typing import Iterable, List, Optional

warnings.filterwarnings("ignore")

import gluonts
import numpy as np
import torch
from gluonts.dataset.common import DataBatch, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import InferenceDataLoader, as_stacked_batches
from gluonts.dataset.repository import dataset_names, get_dataset
from gluonts.itertools import Cyclic
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    DummyValueImputation,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    Transformation,
    VstackFeatures,
)

"""
This module creates dataloaders from the GluonTS library which
is particularly useful to generate training data in an identical
fashion as GluonTS and using the GluonTS sampling mechanism
"""


def create_transformation(prediction_length: int = 24) -> Transformation:
    num_feat_static_cat = 0
    num_feat_static_real = 0
    num_feat_dynamic_real = 0
    imputation_method = DummyValueImputation(dummy_value=0.0)
    freq = "1M"
    time_features = time_features_from_frequency_str(freq)

    remove_field_names = []
    remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
            if not num_feat_static_cat > 0
            else []
        )
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
            if not num_feat_static_real > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=int,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
            ),
            AsNumpyArray(
                field=FieldName.TARGET,
                # in the following line, we add 1 for the time dimension
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
                imputation_method=imputation_method,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + ([FieldName.FEAT_DYNAMIC_REAL] if num_feat_dynamic_real > 0 else []),
            ),
            AsNumpyArray(FieldName.FEAT_TIME, expected_ndim=2),
        ]
    )


CONTEXT_LENGTH = 30
MAX_LAGS = 720


def create_training_data_loader(
    data: Dataset,
    shuffle_buffer_length: Optional[int] = None,
    batch_size: int = 32,
    num_batches_per_epoch: int = 50,
    prediction_length=24,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "feat_static_cat",
        "feat_static_real",
        "past_time_feat",
        "past_target",
        "past_observed_values",
        "future_time_feat",
    ]
    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_target",
        "future_observed_values",
    ]

    transformation = create_transformation(prediction_length=prediction_length)
    transformed_training_data = transformation.apply(data, is_train=True)

    data = Cyclic(transformed_training_data).stream()
    instance_sampler = ExpectedNumInstanceSampler(
        num_instances=1.0, min_future=prediction_length
    )

    splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=CONTEXT_LENGTH + MAX_LAGS,  # context length + max lags
        future_length=prediction_length,
        time_series_fields=[
            FieldName.FEAT_TIME,
            FieldName.OBSERVED_VALUES,
        ],
        dummy_value=0.0,
    )

    instances = splitter.apply(data, is_train=True)
    return as_stacked_batches(
        instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def stack(data, device: Optional[torch.device] = None):
    if isinstance(data[0], np.ndarray):
        data = torch.tensor(np.array(data), device=device)
    elif isinstance(data[0], (list, tuple)):
        return list(stack(t, device=device) for t in zip(*data))
    return data


def batchify(data: List[dict], device: Optional[torch.device] = None) -> DataBatch:
    return {
        key: stack(data=[item[key] for item in data], device=device)
        for key in data[0].keys()
    }


def create_inference_data_loader(dataset, prediction_length=24, batch_size=32):
    required_fields = [
        # "forecast_start",
        "item_id",
        # "info",
        "feat_static_cat",
        "feat_static_real",
        "past_time_feat",
        "past_target",
        "past_observed_values",
        "future_time_feat",
    ]
    input_transform = create_transformation(prediction_length=prediction_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_data_loader = InferenceDataLoader(
        dataset,
        transform=input_transform + SelectFields(required_fields, allow_missing=False),
        batch_size=batch_size,
        stack_fn=lambda data: batchify(data, device),
    )
    return inference_data_loader


def gluonts_to_nnts(batch):
    if "future_target" not in batch:
        batch["future_target"] = torch.zeros(
            batch["future_time_feat"].shape[0],
            batch["future_time_feat"].shape[1],
        )
    if "future_observed_values" not in batch:
        batch["future_observed_values"] = torch.ones(
            batch["future_time_feat"].shape[0],
            batch["future_time_feat"].shape[1],
        ).bool()
    target = torch.cat([batch["past_target"], batch["future_target"]], dim=1).unsqueeze(
        -1
    )
    B, T, _ = target.shape
    time_feat = torch.cat([batch["past_time_feat"], batch["future_time_feat"]], dim=1)
    pad_mask = torch.cat(
        [batch["past_observed_values"], batch["future_observed_values"]], dim=1
    )
    feat_static_cat = batch["feat_static_cat"].expand(B, T).unsqueeze(-1)
    feat_static_real = batch["feat_static_real"].expand(B, T).unsqueeze(-1)
    X = torch.cat([target, time_feat, feat_static_real], dim=-1)
    return {"X": X, "pad_mask": pad_mask}


class DecoratedDataLoader:
    def __init__(self, dataloader, decorator_fn):
        self.dataloader = dataloader
        self.decorator_fn = decorator_fn

    def __iter__(self):
        for batch in self.dataloader:
            yield self.decorator_fn(batch)


dataset = get_dataset("traffic")

trn_dl = DecoratedDataLoader(
    create_training_data_loader(dataset.train, batch_size=32, prediction_length=168),
    gluonts_to_nnts,
)
# test_dl = create_inference_data_loader(dataset.test, batch_size=32)


def get_test_dataloader():
    input = torch.load(
        "/Users/garethdavies/Development/workspaces/gluonts/examples/input.pt"
    )
    return [gluonts_to_nnts(i) for i in input]


test_dl = get_test_dataloader()


def load_gluonts_weights(net):
    def remove_prefix(s, prefix):
        return s[len(prefix) :] if s.startswith(prefix) else s

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
