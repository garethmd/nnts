import pandas as pd
import torch
from torch.utils.data import DataLoader

from nnts.torch.datasets import LastWindowDataset
from nnts.torch.models.nlinear import Hyperparams, NLinear
from nnts.torch.trainers import TorchForecaster


def test_length_equals_series_count():
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "c", "c", "c"],
            "ds": [1, 2, 1, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    dataset = LastWindowDataset(frame, context=2)

    assert len(dataset) == 3


def test_short_series_is_left_padded_and_masked():
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "y": [10.0, 20.0],
        }
    )

    item = LastWindowDataset(frame, context=4)[0]

    assert item.data.shape == (4, 1)
    assert torch.equal(item.data, torch.tensor([[0.0], [0.0], [10.0], [20.0]]))
    assert torch.equal(item.pad_mask, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_long_series_keeps_its_last_context_values():
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "a", "a", "a"],
            "ds": [1, 2, 3, 4, 5, 6],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    item = LastWindowDataset(frame, context=3)[0]

    assert torch.equal(item.data, torch.tensor([[4.0], [5.0], [6.0]]))
    assert torch.equal(item.pad_mask, torch.tensor([1.0, 1.0, 1.0]))


def test_series_as_long_as_context_is_unpadded():
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a"],
            "ds": [1, 2, 3],
            "y": [4.0, 5.0, 6.0],
        }
    )

    item = LastWindowDataset(frame, context=3)[0]

    assert torch.equal(item.data, torch.tensor([[4.0], [5.0], [6.0]]))
    assert torch.equal(item.pad_mask, torch.tensor([1.0, 1.0, 1.0]))


def test_unique_ids_line_up_with_items():
    frame = pd.DataFrame(
        {
            "unique_id": ["b", "a", "a", "b"],
            "ds": [1, 1, 2, 2],
            "y": [7.0, 1.0, 2.0, 8.0],
        }
    )

    dataset = LastWindowDataset(frame, context=2)

    assert dataset.unique_ids == ["b", "a"]
    assert torch.equal(dataset[0].data, torch.tensor([[7.0], [8.0]]))
    assert torch.equal(dataset[1].data, torch.tensor([[1.0], [2.0]]))


def test_forecast_over_last_windows_returns_one_path_per_series():
    frame = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "a", "b", "b"],
            "ds": [1, 2, 3, 4, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0],
        }
    )
    context = 3
    prediction_length = 2
    dataset = LastWindowDataset(frame, context)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    net = NLinear(
        h=prediction_length,
        input_size=context,
        c_in=1,
        configs=Hyperparams(),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    history = torch.randn(4, context, 1)
    target = torch.randn(4, prediction_length, 1)
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(net(history), target)
        loss.backward()
        optimizer.step()

    y_hat = TorchForecaster(net).forecast(loader, prediction_length, context)

    assert tuple(y_hat.shape) == (2, prediction_length, 1)
