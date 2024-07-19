from typing import Callable

import pandas as pd
import pytest

from nnts import datasets


def test_load_data_should_return_dataframe():
    # Arrange

    # Act
    data, *_ = datasets.read_tsf("tests/artifacts/hospital_dataset.tsf")

    # Assert
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert len(data.columns) > 0
    assert len(data.index) > 0


@pytest.fixture
def sample_data():
    unique_ids = ["T1", "T2"]

    # Create 50 records for each unique_id
    result = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "value": range(1, 51),
                    "ds": pd.date_range(start="2024-04-01", periods=50, freq="ME"),
                }
            )
            for unique_id in unique_ids
        ],
        ignore_index=True,
    )

    return result


@pytest.fixture
def sample_metadata() -> datasets.Metadata:
    metadata = datasets.Metadata(
        filename="fake_path",
        dataset="fake_dataset",
        context_length=15,
        prediction_length=12,
        freq="MS",
        seasonality=12,
    )
    return metadata


def test_should_should_contain_correct_counts(sample_data, sample_metadata):
    split_data = datasets.split_test_val_train_last_horizon(
        sample_data, sample_metadata.context_length, sample_metadata.prediction_length
    )  # 50 records for each unique_id

    assert split_data.train.groupby("unique_id").get_group("T1").shape == (
        50 - (12 * 2),
        3,
    )
    assert split_data.validation.groupby("unique_id").get_group("T1").shape == (
        15 + 12,
        3,
    )
    assert split_data.test.groupby("unique_id").get_group("T1").shape == (15 + 12, 3)


def test_validation_should_not_overlap_dates(
    sample_data: pd.DataFrame, sample_metadata: datasets.Metadata
):
    split_data = datasets.split_test_val_train_last_horizon(
        sample_data, sample_metadata.context_length, sample_metadata.prediction_length
    )  # 50 records for each unique_id

    assert (
        split_data.validation.groupby("unique_id").get_group("T1").max().ds
        < split_data.test.groupby("unique_id")
        .get_group("T1")[-sample_metadata.prediction_length :]
        .ds.min()
    )

    assert (
        split_data.validation.groupby("unique_id").get_group("T1").max().ds
        == split_data.test.groupby("unique_id")
        .get_group("T1")[-sample_metadata.prediction_length - 1 :]
        .ds.min()
    )


@pytest.mark.parametrize(
    "splitter_fn",
    [
        datasets.split_test_train_last_horizon,
        datasets.split_test_val_train_last_horizon,
    ],
)
def test_train_should_not_overlap_dates(
    splitter_fn: Callable,
    sample_data: pd.DataFrame,
    sample_metadata: datasets.Metadata,
):
    split_data = splitter_fn(
        sample_data, sample_metadata.context_length, sample_metadata.prediction_length
    )  # 50 records for each unique_id

    assert (
        split_data.train.groupby("unique_id").get_group("T1").max().ds
        < split_data.test.groupby("unique_id")
        .get_group("T1")[-sample_metadata.prediction_length :]
        .ds.min()
    )


def test_should_should_contain_correct_counts_given_train_test_split(
    sample_data, sample_metadata
):
    split_data = datasets.split_test_train_last_horizon(
        sample_data, sample_metadata.context_length, sample_metadata.prediction_length
    )  # 50 records for each unique_id

    assert split_data.train.groupby("unique_id").get_group("T1").shape == (
        50 - 12,
        3,
    )

    assert split_data.test.groupby("unique_id").get_group("T1").shape == (15 + 12, 3)
