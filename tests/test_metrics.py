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
            torch.tensor([2.5, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.25]),
        ),
    ],
)
def test_should_calculate_mae_correctly(y_hat, y, expected):
    mae = nnts.metrics.mae(y_hat, y)
    assert torch.allclose(mae, expected)


@pytest.mark.parametrize(
    "y_hat, y, sesonal_errors, expected",
    [
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([2.0, 4.0]),
            torch.tensor([1.25, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0, 4.0, 6.0, 8.0]),
            torch.tensor([2.0]),
            torch.tensor([1.25]),
        ),
    ],
)
def test_should_calculate_mase_correctly(y_hat, y, sesonal_errors, expected):
    mase = nnts.metrics.mase(y_hat, y, sesonal_errors)
    assert torch.allclose(mase, expected)


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
            torch.tensor([10, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0]),
        ),
    ],
)
def test_should_calculate_abs_error_correctly(y_hat, y, expected):
    mae = nnts.metrics.abs_error(y_hat, y)
    assert torch.allclose(mae, expected)


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
            torch.tensor([10, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0]),
        ),
    ],
)
def test_should_calculate_abs_error_correctly(y_hat, y, expected):
    mae = nnts.metrics.abs_error(y_hat, y)
    assert torch.allclose(mae, expected)


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
            torch.tensor([7.5, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.25]),
        ),
    ],
)
def test_should_calculate_mse_correctly(y_hat, y, expected):
    mae = nnts.metrics.mse(y_hat, y)
    assert torch.allclose(mae, expected)


@pytest.mark.parametrize(
    "y, seasonality, expected",
    [
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            1,
            torch.tensor([1.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            2,
            torch.tensor([2.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            5,
            torch.tensor([1.0]),
        ),
        (
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            1,
            torch.tensor([0.0]),
        ),
    ],
)
def test_should_calculate_seasonal_error_correctly(y, seasonality, expected):
    seasonal_errors = nnts.metrics._calculate_seasonal_error(y, seasonality)
    assert torch.allclose(seasonal_errors, expected)


@pytest.mark.parametrize(
    "y, expected",
    [
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([10.0]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
            torch.tensor([10.0, 20.0]),
        ),
    ],
)
def test_should_calculate_abs_target_sum_correctly(y, expected):
    abs_sum = nnts.metrics.abs_target_sum(y)
    assert torch.allclose(abs_sum, expected)


@pytest.mark.parametrize(
    "y, expected",
    [
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([2.5]),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
            torch.tensor([2.5, 5.0]),
        ),
    ],
)
def test_should_calculate_abs_target_mean_correctly(y, expected):
    abs_mean = nnts.metrics.abs_target_mean(y)
    assert torch.allclose(abs_mean, expected)


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
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            torch.tensor([1.5]),
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
            torch.tensor([2 / 3, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 0.0, 1.0, 1.0]),
            torch.tensor(4.2 / 4),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[0.0, 4.0, 6.0, 8.0], [0.0, 2.0, 3.0, 4.0]]),
            torch.tensor([1.0, 0.5]),
        ),
    ],
)
def test_should_calculate_smape_correctly(y_hat, y, expected):
    smape = nnts.metrics.smape(y_hat, y)
    assert torch.allclose(smape, expected)


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
            torch.tensor([0.6553, 0.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.0, 0.0, 1.0, 1.0]),
            torch.tensor(1.0142),
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[0.0, 4.0, 6.0, 8.0], [0.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[0.9486, 0.4545]]),
        ),
    ],
)
def test_should_calculate_msmape_correctly(y_hat, y, expected):
    smape = nnts.metrics.msmape(y_hat, y)
    assert torch.allclose(smape, expected, atol=1e-4)


def test_should_get_metrics_per_ts():
    y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    y = torch.tensor([[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]])
    seasonal_error = torch.tensor([2.0, 4.0])
    metrics = nnts.metrics.get_metrics_per_ts(y_hat, y, seasonal_error)
    assert isinstance(metrics, dict)


def test_should_handle_divide_by_zero():
    y_hat = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([0.0, 0.0, 0.0, 0.0])
    mape = nnts.metrics.mape(y_hat, y)
    assert torch.isnan(mape)
