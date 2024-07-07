import pytest

import nnts.data
import nnts.experiments
import nnts.hyperparams
import nnts.torch.datasets


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
    return nnts.hyperparams.Hyperparams(batch_size=2, epochs=10, lr=0.01)
