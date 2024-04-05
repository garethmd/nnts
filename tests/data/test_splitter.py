import pandas as pd
import pytest

from nnts.data import Metadata, PandasSplitter, SplitData


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
                    "ds": pd.date_range(start="2024-04-01", periods=50, freq="M"),
                }
            )
            for unique_id in unique_ids
        ],
        ignore_index=True,
    )

    return result


@pytest.fixture
def sample_metadata():
    metadata = Metadata(
        path="fake_path",
        dataset="fake_dataset",
        context_length=15,
        prediction_length=12,
        freq="MS",
        seasonality=12,
    )
    return metadata


def test_should_split_data(sample_data, sample_metadata):
    splitter = PandasSplitter()
    split_data = splitter.split(sample_data, sample_metadata)
    assert isinstance(split_data, SplitData)


def test_should_should_contain_correct_counts(sample_data, sample_metadata):
    splitter = PandasSplitter()
    split_data = splitter.split(
        sample_data, sample_metadata
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


def test_train_should_not_overlap_dates(sample_data, sample_metadata):
    splitter = PandasSplitter()
    split_data = splitter.split(
        sample_data, sample_metadata
    )  # 50 records for each unique_id

    assert (
        split_data.train.groupby("unique_id").get_group("T1").max().ds
        < split_data.validation.groupby("unique_id")
        .get_group("T1")[-sample_metadata.prediction_length :]
        .ds.min()
    )

    assert (
        split_data.train.groupby("unique_id").get_group("T1").max().ds
        == split_data.validation.groupby("unique_id")
        .get_group("T1")[-sample_metadata.prediction_length - 1 :]
        .ds.min()
    )


def test_validation_should_not_overlap_dates(sample_data, sample_metadata):
    splitter = PandasSplitter()
    split_data = splitter.split(
        sample_data, sample_metadata
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
