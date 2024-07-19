import pandas as pd

import nnts.datasets
from nnts import datasets


def test_load_metadata_should_return_metadata():
    # Arrange
    dataset = "hospital"

    # Act
    metadata = datasets.load_metadata(dataset, path="tests/artifacts/monash.json")

    # Assert
    assert isinstance(metadata, datasets.Metadata)
    assert metadata.dataset == dataset
    assert metadata.context_length > 0
    assert metadata.prediction_length > 0
    assert metadata.freq is not None
    assert metadata.seasonality > 0


def test_should_load_monash_metadata():
    metadata = datasets.load_metadata("hospital", path="tests/artifacts/monash.json")
    assert isinstance(metadata, datasets.Metadata)


def test_should_load_monash_metadata_and_data():
    response = nnts.datasets.load(
        "hospital",
        data_path="tests/artifacts",
        metadata_filename="monash.json",
    )
    assert isinstance(response[0], pd.DataFrame)
    assert isinstance(response[1], datasets.Metadata)


def test_should_read_tsf_data():
    df, freq, forecast_horizon, missing_values, equal_length = nnts.datasets.read_tsf(
        "tests/artifacts/hospital_dataset.tsf"
    )
    assert isinstance(df, pd.DataFrame)
    assert freq == "1M"
    assert forecast_horizon == None
    assert missing_values == False
    assert equal_length == True


def test_should_raise_error_given_missing_repository():
    try:
        nnts.datasets.load_metadata(
            "missing_dataset", path="tests/artifacts/monash.json"
        )
    except ValueError as e:
        assert (
            str(e)
            == "Dataset missing_dataset not found in metadata tests/artifacts/monash.json choose from: us_births, tourism_monthly, hospital"
        )
