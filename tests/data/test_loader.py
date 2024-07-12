import pytest

import nnts.data
import nnts.datasets
import nnts.torch.datasets
from nnts import datasets, utils


@pytest.fixture
def sample_split_data():
    return nnts.datasets.SplitData(train="train", validation="validation", test="test")


@pytest.fixture
def sample_metadata():
    return datasets.Metadata(
        filename="fake_path",
        dataset="fake_dataset",
        context_length=15,
        prediction_length=12,
        freq="MS",
        seasonality=12,
    )


@pytest.fixture
def sample_scenario():
    return covs.CovariateScenario(
        prediction_length=12, error=0, conts=["cont1", "cont2"]
    )


@pytest.fixture
def sample_params():
    return utils.Hyperparams(batch_size=2, epochs=10, lr=0.01)
