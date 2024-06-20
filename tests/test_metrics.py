import numpy as np
import pytest
import torch

import nnts.metrics


@pytest.mark.parametrize(
    "y_hat, y, expected",
    [
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([0.5, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[0.0, 4.0, 6.0, 8.0], [0.0, 2.0, 3.0, 4.0]]),
            torch.tensor([0.5, 0.0]),
        ),
    ],
)
def test_should_calculate_mape_correctly(y_hat, y, expected):
    mape = nnts.metrics.mape(y_hat, y)
    assert torch.allclose(mape, expected)


def test_should_handle_divide_by_zero():
    y_hat = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([0.0, 0.0, 0.0, 0.0])
    mape = nnts.metrics.mape(y_hat, y)
    assert torch.isnan(mape)
