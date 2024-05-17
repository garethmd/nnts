import pytest

import nnts.data
import nnts.experiments
import nnts.models


@pytest.fixture
def sample_split_data():
    return nnts.data.SplitData(train="train", validation="validation", test="test")


@pytest.fixture
def sample_metadata():
    return nnts.data.Metadata(
        filename="fake_path",
        dataset="fake_dataset",
        context_length=15,
        prediction_length=12,
        freq="MS",
        seasonality=12,
    )


@pytest.fixture
def sample_scenario():
    return nnts.experiments.CovariateScenario(
        prediction_length=12, error=0, conts=["cont1", "cont2"]
    )


@pytest.fixture
def sample_params():
    return nnts.models.Hyperparams(batch_size=2, epochs=10, lr=0.01)


def test_should_create_trn_val_test_dataloaders(
    sample_split_data, sample_metadata, sample_scenario, sample_params
):
    # Arrange
    class MockDataLoaderFactory(nnts.data.DataLoaderFactory):
        def __call__(self, data, *args, **kwargs):
            return data

    mock_factory = MockDataLoaderFactory()

    # Act
    trn_dl, val_dl, test_dl = nnts.data.loader.create_trn_val_test_dataloaders(
        sample_split_data,
        sample_metadata,
        sample_scenario,
        sample_params,
        dataloader_factory=mock_factory,
    )

    print(trn_dl)
    # Assert
    assert trn_dl == "train"
    assert val_dl == "validation"
    assert test_dl == "test"
