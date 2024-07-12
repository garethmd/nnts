from typing import Iterable, List, Optional

import monash
import numpy as np
import torch
from gluonts.dataset.common import DataBatch, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
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
    TestSplitSampler,
    Transformation,
    VstackFeatures,
)

FREQ_MAP = {
    "tourism_monthly": "1M",
    "electricity": "1H",
    "traffic_hourly": "1H",
    "hospital": "1M",
}

CONTEXT_LENGTH_MAP = {
    "tourism_monthly": 15,
    "electricity": 30,
    "traffic_hourly": 30,
    "hospital": 15,
}
PREDICTION_LENGTH_MAP = {
    "tourism_monthly": 24,
    "electricity": 168,
    "traffic_hourly": 168,
    "hospital": 12,
}
FILE_NAME_MAP = {
    "tourism_monthly": "tourism_monthly_dataset.tsf",
    "electricity": "electricity_hourly_dataset.tsf",
    "traffic_hourly": "traffic_hourly_dataset.tsf",
    "hospital": "hospital_dataset.tsf",
}


def create_transformation(prediction_length: int, freq) -> Transformation:
    num_feat_static_cat = 0
    num_feat_static_real = 0
    num_feat_dynamic_real = 0
    imputation_method = DummyValueImputation(dummy_value=0.0)
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


def create_training_data_loader(
    data: Dataset,
    batch_size,
    num_batches_per_epoch,
    prediction_length,
    context_length,
    max_lags,
    freq,
    shuffle_buffer_length: Optional[int] = None,
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

    transformation = create_transformation(
        prediction_length=prediction_length, freq=freq
    )
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
        past_length=context_length + max_lags,  # context length + max lags
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


class DecoratedDataLoader:
    def __init__(self, dataloader, decorator_fn):
        self.dataloader = dataloader
        self.decorator_fn = decorator_fn

    def __iter__(self):
        for batch in self.dataloader:
            yield self.decorator_fn(batch)


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


def get_train_dl(dataset_name, max_lags=0):
    batch_size = 32
    num_batches_per_epoch = 50
    prediction_length = PREDICTION_LENGTH_MAP[dataset_name]
    context_length = CONTEXT_LENGTH_MAP[dataset_name]

    train_ds, test_ds = monash.get_deep_nn_forecasts(
        dataset_name,
        context_length,
        FILE_NAME_MAP[dataset_name],
        "feed_forward",
        prediction_length,
        True,
    )

    data_loader = create_training_data_loader(
        train_ds,
        batch_size,
        num_batches_per_epoch,
        prediction_length=prediction_length,
        context_length=context_length,
        max_lags=max_lags,
        freq=FREQ_MAP[dataset_name],
    )

    trn_dl = DecoratedDataLoader(
        data_loader,
        gluonts_to_nnts,
    )
    return trn_dl


if __name__ == "__main__":
    trn_dl = get_train_dl()

    for i, batch in enumerate(trn_dl):
        print(batch)
        if i > 5:
            break
