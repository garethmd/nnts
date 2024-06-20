import pandas as pd
import pytest
import torch

import nnts.torch.data.datasets as datasets


def test_right_pad_sequence():
    seq = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])]
    padded_tensor, padded_mask = datasets.right_pad_sequence(seq)
    assert padded_tensor.shape == (2, 2, 2)
    assert padded_mask.shape == (2, 2)
    assert torch.all(padded_tensor[0, 0, :] == torch.tensor([1, 2]))
    assert torch.all(padded_tensor[0, 1, :] == torch.tensor([3, 4]))
    assert torch.all(padded_tensor[1, 0, :] == torch.tensor([5, 6]))
    assert torch.all(padded_tensor[1, 1, :] == torch.tensor([0, 0]))
    assert torch.all(padded_mask[0, :] == torch.tensor([True, True]))
    assert torch.all(padded_mask[1, :] == torch.tensor([True, False]))


@pytest.fixture
def sample_data():
    unique_ids = ["T1", "T2"]

    # Create 50 records for each unique_id
    result = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "y": range(1, 51),
                    "ds": pd.date_range(start="2024-04-01", periods=50, freq="M"),
                }
            )
            for unique_id in unique_ids
        ],
        ignore_index=True,
    )

    return result


class TestTimeseriesDataset:

    def test_should_load_dataframe(self, sample_data):
        ds = datasets.TimeseriesDataset(sample_data, 15, 24)
        assert ds.df.equals(sample_data)

    def test_should_calculate_length_correctly(self, sample_data):
        ds = datasets.TimeseriesDataset(sample_data, 15, 24)
        print(sample_data.shape)
        assert len(ds) == 2 * (50 - 15 - 24 + 1)

    def test_should_build_correctly(self, sample_data):
        ds = datasets.TimeseriesDataset(sample_data, 15, 24)
        ds.build()
        assert ds.X.shape == (2, 50, 1)

    def test_should_index_correctly(self, sample_data):
        ds = datasets.TimeseriesDataset(sample_data, 15, 24)
        ds.build()
        data = ds[0]
        assert data["X"].shape == (15 + 24, 1)

    def test_should_iterate_whole_dataset_start(self, sample_data):
        ds = datasets.TimeseriesDataset(sample_data, 15, 24)
        ds.build()
        data = ds[0]
        expected = sample_data.head(15 + 24)
        assert torch.allclose(
            torch.tensor(expected["y"].values).float(), data["X"].squeeze().float()
        )

    def test_should_iterate_whole_dataset_second(self, sample_data):
        ds = datasets.TimeseriesDataset(sample_data, 15, 24)
        ds.build()
        data = ds[1]
        expected = sample_data[1 : 1 + 15 + 24]
        assert torch.allclose(
            torch.tensor(expected["y"].values).float(), data["X"].squeeze().float()
        )

    def test_should_iterate_whole_dataset_end(self, sample_data):
        ds = datasets.TimeseriesDataset(sample_data, 15, 24)
        ds.build()
        last_idx = len(ds) - 1
        data = ds[last_idx]
        expected = sample_data.tail(15 + 24)
        assert torch.allclose(
            torch.tensor(expected["y"].values).float(), data["X"].squeeze().float()
        )


@pytest.fixture
def sample_dataset():
    import torch.utils.data

    sample_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 5), torch.randint(0, 2, (100,))
    )
    return sample_dataset


class TestTruncatedDataLoader:

    def test_should_return_correct_length_given_no_batches_per_epoch(
        self, sample_dataset
    ):
        loader = datasets.TruncatedDataLoader(sample_dataset, batch_size=32)
        assert len(loader) == 4

    def test_should_return_correct_length_given_set_batches_per_epoch(
        self, sample_dataset
    ):
        loader = datasets.TruncatedDataLoader(
            sample_dataset, batch_size=32, batches_per_epoch=2
        )
        assert len(loader) == 2
