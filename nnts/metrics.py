from typing import Dict, Optional

import torch
import torch.utils.data


def _calculate_seasonal_error(
    past_data: torch.Tensor,
    seasonality: Optional[int] = 1,
) -> torch.Tensor:
    """
    Calculates the seasonal error for a given time series data based on the specified seasonality.

    The seasonal error is computed by comparing the values of the time series with those
    lagged by the seasonality period. If the seasonality is larger than the length of the
    time series, the lag is set to 1 (t-1).

    Parameters
    ----------
    past_data: torch.Tensor
        A tensor containing the past time series data.
    seasonality: :obj:`int`
        The seasonality period. If not specified or if the length of past_data is less than the seasonality, a fallback value of 1 (i.e., t-1) is used.

    Returns
    -------
    torch.Tensor
        A tensor containing the mean absolute seasonal error.

    Example
    -------
    >>> past_data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> seasonality = 2
    >>> error = _calculate_seasonal_error(past_data, seasonality)
    >>> print(error)
    tensor(20.0)
    """
    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        forecast_freq = 1  # Fallback to t-1 if series length < seasonality. From Monash
        # https://github.com/rakshitha123/TSForecasting/blob/d187b057af25a203f8b321ad52edf11bc3c8b619/utils/error_calculator.R#L57

    y_t = past_data[:-forecast_freq]
    y_tm = past_data[forecast_freq:]

    result = torch.mean(torch.abs(y_t - y_tm))
    return result


def calculate_seasonal_error(
    trn_dl: torch.utils.data.DataLoader, seasonality: int
) -> torch.Tensor:
    seasonal_errors_per_series = []
    for series, mask in zip(trn_dl.dataset.X[:, :, 0], trn_dl.dataset.pad_mask):
        past_data = series[mask]
        error = _calculate_seasonal_error(past_data, seasonality)
        seasonal_errors_per_series.append(error)
    seasonal_errors = torch.stack(seasonal_errors_per_series, dim=0)
    return seasonal_errors.unsqueeze(1)


def mse(y_hat: torch.tensor, y: torch.tensor, dim=1) -> torch.tensor:
    """
    Calculates the Mean Squared Error (MSE) between the predicted values and the actual values.

    The function computes the MSE by taking the element-wise difference between the predicted
    values and the actual values, squaring these differences, and then averaging them. If the
    input tensors have more than one dimension, the averaging is done along the specified dimension.

    Parameters
    ----------
    y_hat: torch.tensor
        The predicted values.
    y: torch.tensor
        The actual values.
    dim: :obj: `int`, optional
        The dimension along which to compute the mean if the input tensors have more than one dimension. Defaults to 1.

    Returns
    -------
    torch.tensor
        A tensor containing the mean squared error.

    Example
    -------
    >>> y_hat = torch.tensor([2.5, 0.0, 2.0, 8.0])
    >>> y = torch.tensor([3.0, -0.5, 2.0, 7.0])
    >>> error = mse(y_hat, y)
    >>> print(error)
    tensor(0.3750)
    """
    return (
        ((y_hat - y) ** 2).mean(dim=dim)
        if len(y.shape) > 1
        else ((y_hat - y) ** 2).mean()
    )


def abs_error(y_hat: torch.tensor, y: torch.tensor, dim=1) -> torch.tensor:
    """
    Calculates the absolute error between the predicted values and the actual values.

    The function computes the absolute error by taking the element-wise absolute difference
    between the predicted values and the actual values, and then summing these differences.
    If the input tensors have more than one dimension, the summation is done along dimension 1.

    Parameters
    ----------
    y_hat: torch.tensor
        The predicted values.
    y: torch.tensor
        The actual values.

    Returns
    -------
    torch.tensor:
        A tensor containing the sum of absolute errors.

    Example
    -------
    >>> y_hat = torch.tensor([2.5, 0.0, 2.0, 8.0])
    >>> y = torch.tensor([3.0, -0.5, 2.0, 7.0])
    >>> error = abs_error(y_hat, y)
    >>> print(error)
    tensor(2.0)
    """
    return (
        (y_hat - y).abs().sum(dim=dim) if len(y.shape) > 1 else (y_hat - y).abs().sum()
    )


def abs_target_sum(y: torch.tensor, dim: int = 1) -> torch.tensor:
    """
    Calculates the sum of the absolute values of the target tensor.

    The function computes the sum of the absolute values of the target tensor. If the input
    tensor has more than one dimension, the summation is done along the specified dimension.

    Parameters
    ----------
    y: torch.tensor
        The target values.
    dim: :obj: `int`, optional
        The dimension along which to compute the sum if the input tensor has more than one dimension. Defaults to 1.

    Returns
    -------
    torch.tensor
        A tensor containing the sum of absolute values.

    Example
    -------
    >>> y = torch.tensor([3.0, -0.5, 2.0, -7.0])
    >>> result = abs_target_sum(y)
    >>> print(result)
    tensor(12.5)
    """
    return y.abs().sum(dim=dim) if len(y.shape) > 1 else y.abs().sum()


def abs_target_mean(y: torch.tensor, dim: int = 1) -> torch.tensor:
    """
    Calculates the mean of the absolute values of the target tensor.

    The function computes the mean of the absolute values of the target tensor. If the input
    tensor has more than one dimension, the mean is calculated along the specified dimension.

    Parameters
    ----------
    y: torch.tensor
        The target values.
    dim: :obj: `int`, optional
        The dimension along which to compute the mean if the input tensor has more than one dimension. Defaults to 1.

    Returns
    -------
    torch.tensor
        A tensor containing the mean of absolute values.

    Example
    -------
    >>> y = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    >>> result = abs_target_mean(y)
    >>> print(result)
    tensor([2.5, 5.0])
    """
    return y.abs().mean(dim=dim) if len(y.shape) > 1 else y.abs().mean()


def mae(y_hat: torch.tensor, y: torch.tensor, dim: int = 1) -> torch.tensor:
    """
    Calculates the Mean Absolute Error (MAE) between the predicted values and the actual values.

    The function computes the MAE by taking the element-wise absolute difference between the
    predicted values and the actual values, and then averaging these differences. If the input
    tensors have more than one dimension, the averaging is done along the specified dimension.

    Parameters
    ----------
    y_hat: torch.tensor)
        The predicted values.
    y: torch.tensor
        The actual values.
    dim: :obj:`int`, optional
        The dimension along which to compute the mean if the input tensors have more than one dimension. Defaults to 1.

    Returns
    -------
    torch.tensor
        A tensor containing the mean absolute error.

    Example
    -------
    >>> y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    >>> y = torch.tensor([[2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0]])
    >>> error = mae(y_hat, y)
    >>> print(error)
    tensor([2.5, 0.0])
    """
    return (
        (y_hat - y).abs().mean(dim=dim)
        if len(y.shape) > 1
        else (y_hat - y).abs().mean()
    )


def mase(
    y_hat: torch.tensor,
    y: torch.tensor,
    seasonal_errors: torch.tensor,
    dim: int = 1,
) -> torch.tensor:
    """
    Calculates the Mean Absolute Scaled Error (MASE) between the predicted values and the actual values.

    The function computes the MASE by dividing the mean absolute error (MAE) by the seasonal
    errors. If the input tensors have more than one dimension, the MAE is calculated along the
    specified dimension.

    Parameters
    ----------
    y_hat: torch.tensor)
        The predicted values.
    y: torch.tensor
        The actual values.
    seasonal_errors: torch.tensor)
        The seasonal errors used for scaling the mean absolute error (MAE).
    dim: :obj:`int`, optional
        The dimension along which to compute the mean if the input tensors have more than one dimension. Defaults to 1.

    Returns
    -------
    torch.tensor: A tensor containing the mean absolute scaled error.

    Example
    -------

    >>> y_hat = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> y = torch.tensor([0.0, 4.0, 6.0, 8.0])
    >>> seasonal_errors = torch.tensor([2.0])
    >>> error = mase(y_hat, y, seasonal_errors)
    >>> print(error)
    tensor([1.25])
    """
    if len(y.shape) > 2 and len(seasonal_errors.shape) == 1:
        seasonal_errors = seasonal_errors.unsqueeze(-1)
    return mae(y_hat, y, dim=dim) / seasonal_errors


def rmsse(
    y_hat: torch.tensor,
    y: torch.tensor,
    seasonal_errors: torch.tensor,
    dim: int = 1,
) -> torch.tensor:
    """
    Calculates the Mean Absolute Scaled Error (MASE) between the predicted values and the actual values.

    The function computes the MASE by dividing the mean absolute error (MAE) by the seasonal
    errors. If the input tensors have more than one dimension, the MAE is calculated along the
    specified dimension.

    Parameters
    ----------
    y_hat: torch.tensor)
        The predicted values.
    y: torch.tensor
        The actual values.
    seasonal_errors: torch.tensor)
        The seasonal errors used for scaling the mean absolute error (MAE).
    dim: :obj:`int`, optional
        The dimension along which to compute the mean if the input tensors have more than one dimension. Defaults to 1.

    Returns
    -------
    torch.tensor: A tensor containing the mean absolute scaled error.

    Example
    -------

    >>> y_hat = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> y = torch.tensor([0.0, 4.0, 6.0, 8.0])
    >>> seasonal_errors = torch.tensor([2.0])
    >>> error = mase(y_hat, y, seasonal_errors)
    >>> print(error)
    tensor([1.25])
    """
    if len(y.shape) > 2 and len(seasonal_errors.shape) == 1:
        seasonal_errors = seasonal_errors.unsqueeze(-1)
    return mse(y_hat, y, dim=dim).sqrt() / seasonal_errors


def _single_ts_mape(y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
    ape = (y_hat - y).abs() / y.abs()
    mask = torch.isfinite(ape)
    ape_filtered = ape[mask]
    result = torch.mean(ape_filtered)
    return result


def mape(y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between the predicted values and the actual values.

    The function computes the MAPE by calculating the absolute percentage error for each time
    series and then averaging these errors. If the input tensors contain multiple time series
    (i.e., are two-dimensional), the MAPE is calculated for each time series independently.

    Parameters
    ----------
    y_hat: torch.tensor
        The predicted values.
    y: torch.tensor)
        The actual values.

    Returns
    -------
    torch.tensor: A tensor containing the mean absolute percentage error for each time series.

    Example
    -------
    >>> y_hat = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> y = torch.tensor([1.0, 1.0, 1.0, 1.0])
    >>> error = mape(y_hat, y)
    >>> print(error)
    tensor([1.5])

    >>> y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    >>> y = torch.tensor([[0.0, 4.0, 6.0, 8.0], [0.0, 2.0, 3.0, 4.0]])
    >>> error = mape(y_hat, y)
    >>> print(error)
    tensor([0.5, 0.0])

    """
    if len(y.shape) > 1:
        return torch.stack(
            [_single_ts_mape(y_hat[i], y[i]) for i in range(y.shape[0])], dim=0
        )
    else:
        return _single_ts_mape(y_hat, y)


def smape(y_hat: torch.tensor, y: torch.tensor, dim: int = 1) -> torch.tensor:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (sMAPE) between the predicted values and the actual values.

    The function computes the sMAPE by calculating the symmetric absolute percentage error
    for each prediction, which is given by the formula 2 * |y_hat - y| / (|y| + |y_hat|).
    If the input tensors have more than one dimension, the mean is calculated along the specified dimension.

    Parameters
    ----------
    y_hat: torch.tensor
        The predicted values.
    y: torch.tensor
        The actual values.
    dim: :obj:`int`, optional
        The dimension along which to compute the mean if the input tensors have more than one dimension. Defaults to 1.

    Returns
    -------
    torch.tensor
        A tensor containing the symmetric mean absolute percentage error.

    Example
    -------
    >>> y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    >>> y = torch.tensor([[0.0, 4.0, 6.0, 8.0], [0.0, 2.0, 3.0, 4.0]])
    >>> error = smape(y_hat, y)
    >>> print(error)
    tensor([1.0, 0.5])

    """
    pe = 2 * (y_hat - y).abs() / (y.abs() + y_hat.abs())
    return pe.mean(dim=dim) if len(y.shape) > 1 else pe.mean()


def msmape(y_hat: torch.tensor, y: torch.tensor, dim: int = 1, eps=0.1) -> torch.tensor:
    """
    Calculates the Modified Symmetric Mean Absolute Percentage Error (msMAPE) between the predicted values and the actual values.

    The function computes the msMAPE by calculating the modified symmetric absolute percentage error
    for each prediction, which is given by the formula 2 * |y_hat - y| / max((|y| + |y_hat| + eps), (0.5 + eps)).
    If the input tensors have more than one dimension, the mean is calculated along the specified dimension.

    Parameters
    ----------
    y_hat: torch.tensor)
        The predicted values.
    y: torch.tensor)
        The actual values.
    dim: :obj:`int`, optional
        The dimension along which to compute the mean if the input tensors have more than one dimension. Defaults to 1.
    eps: :obj:`float`, optional
        Epsilon value to prevent division by zero. Defaults to 0.1.

    Returns
    -------
    torch.tensor
        A tensor containing the modified symmetric mean absolute percentage error.


    Example
    -------
    >>> y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    >>> y = torch.tensor([[0.0, 4.0, 6.0, 8.0], [0.0, 2.0, 3.0, 4.0]])
    >>> error = msmape(y_hat, y)
    >>> print(error)
    tensor([1.0, 0.5])

    """
    pe = (
        2
        * (y_hat - y).abs()
        / torch.max((y.abs() + y_hat.abs() + eps), torch.ones_like(y) * 0.5 + eps)
    )
    return pe.mean(dim=dim) if len(y.shape) > 1 else pe.mean()


def nd(y_hat: torch.tensor, y: torch.tensor, dim: int = 1) -> torch.tensor:
    return abs_error(y_hat, y, dim=dim) / abs_target_sum(y, dim=dim)


def get_metrics_per_ts(
    y_hat: torch.tensor, y: torch.tensor, seasonal_error: torch.tensor
) -> Dict[str, torch.tensor]:
    mase_per_ts = mase(y_hat, y, seasonal_error)
    mase_per_ts = mase_per_ts[mase_per_ts.isfinite()]
    rmsse_per_ts = rmsse(y_hat, y, seasonal_error)
    rmsse_per_ts = rmsse_per_ts[rmsse_per_ts.isfinite()]
    return {
        "mse": mse(y_hat, y),
        "abs_error": abs_error(y_hat, y),
        "abs_target_sum": abs_target_sum(y),
        "abs_target_mean": abs_target_mean(y),
        "mase": mase_per_ts,
        "rmsse": rmsse_per_ts,
        "mape": mape(y_hat, y),
        "smape": smape(y_hat, y),
        "msmape": msmape(y_hat, y),
        "nd": nd(y_hat, y),
        "mae": mae(y_hat, y),
        "rmse": mse(y_hat, y).sqrt(),
        "seasonal_error": seasonal_error,
    }


def aggregate_base_metrics(metrics_per_ts: Dict[str, torch.tensor]) -> Dict[str, float]:
    aggregate_map = {
        "mse": "mean",
        "abs_error": "sum",
        "abs_target_sum": "sum",
        "abs_target_mean": "mean",
        "seasonal_error": "mean",
    }
    return {
        k: getattr(v, aggregate_map[k])().item()
        for k, v in metrics_per_ts.items()
        if k in aggregate_map
    }


def aggregate_mean_metrics(metrics_per_ts: Dict[str, torch.tensor]) -> Dict[str, float]:
    aggregate_map = {
        "mase": "mean",
        "rmsse": "mean",
        "mape": "nanmean",
        "smape": "mean",
        "msmape": "mean",
        "mae": "mean",
        "rmse": "mean",
    }
    return {
        f"mean_{k}": getattr(v, aggregate_map[k])().item()
        for k, v in metrics_per_ts.items()
        if k in aggregate_map
    }


def aggregate_median_metrics(
    metrics_per_ts: Dict[str, torch.tensor],
) -> Dict[str, float]:
    aggregate_map = {
        "mase": "median",
        "rmsse": "median",
        # "mape": "nanmedian",
        "smape": "median",
        "msmape": "median",
        "mae": "median",
        "rmse": "median",
    }
    return {
        f"median_{k}": getattr(v, aggregate_map[k])().item()
        for k, v in metrics_per_ts.items()
        if k in aggregate_map
    }


def calc_metrics(
    y_hat: torch.tensor, y: torch.tensor, seasonal_error: torch.tensor
) -> Dict[str, float]:
    metrics_per_ts = get_metrics_per_ts(y_hat, y, seasonal_error)
    base_metrics = aggregate_base_metrics(metrics_per_ts)
    mean_metrics = aggregate_mean_metrics(metrics_per_ts)
    median_metrics = aggregate_median_metrics(metrics_per_ts)
    metrics = {**base_metrics, **mean_metrics, **median_metrics}
    return metrics
