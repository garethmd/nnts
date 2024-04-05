import pandas as pd

from nnts.data import Metadata, loader


def test_load_data_should_return_dataframe():
    # Arrange
    metadata = Metadata(
        path="tests/artifacts/hospital_dataset.tsf",
        dataset="hospital",
        context_length=15,
        prediction_length=12,
        freq="M",
        seasonality=12,
    )

    # Act
    data = loader.load_data(metadata)

    # Assert
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert len(data.columns) > 0
    assert len(data.index) > 0


def test_load_metadata_should_return_metadata():
    # Arrange
    dataset = "hospital"

    # Act
    metadata = loader.load_metadata(dataset, path="tests/artifacts/monash.json")

    # Assert
    assert isinstance(metadata, Metadata)
    assert metadata.dataset == dataset
    assert metadata.context_length > 0
    assert metadata.prediction_length > 0
    assert metadata.freq is not None
    assert metadata.seasonality > 0


def test_should_load_monash_metadata():
    metadata = loader.load_metadata("hospital")
    assert isinstance(metadata, Metadata)


def test_should_load_monash_metadata_and_data():
    response = loader.load("hospital")
    assert isinstance(response[0], pd.DataFrame)
    assert isinstance(response[1], Metadata)
