import pandas as pd

import nnts.data.metadata
import nnts.pandas


def test_load_metadata_should_return_metadata():
    # Arrange
    dataset = "hospital"

    # Act
    metadata = nnts.data.metadata.load(dataset, path="tests/artifacts/monash.json")

    # Assert
    assert isinstance(metadata, nnts.data.metadata.Metadata)
    assert metadata.dataset == dataset
    assert metadata.context_length > 0
    assert metadata.prediction_length > 0
    assert metadata.freq is not None
    assert metadata.seasonality > 0


def test_should_load_monash_metadata():
    metadata = nnts.data.metadata.load("hospital", path="tests/artifacts/monash.json")
    assert isinstance(metadata, nnts.data.metadata.Metadata)


def test_should_load_monash_metadata_and_data():
    response = nnts.pandas.load(
        "hospital",
        metadata_path="tests/artifacts/monash.json",
    )
    assert isinstance(response[0], pd.DataFrame)
    assert isinstance(response[1], nnts.data.metadata.Metadata)
