from typing import Dict, Optional

import torch
import torch.nn.functional as F

import nnts.data.metadata


def _calculate_seasonal_error(
    past_data: torch.tensor,
    seasonality: Optional[int] = None,
) -> torch.tensor:
    r"""
    .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

    where m is the seasonal frequency. See [HA21]_ for more details.
    """
    # Check if the length of the time series is larger than the seasonal
    # frequency
    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        forecast_freq = 1

    y_t = past_data[:-forecast_freq]
    y_tm = past_data[forecast_freq:]

    result = torch.mean(torch.abs(y_t - y_tm))
    # assert torch.isfinite(result)
    return result


def calculate_seasonal_error(
    trn_dl: torch.utils.data.DataLoader, metadata: nnts.data.metadata.Metadata
) -> torch.tensor:
    seasonal_errors_per_series = []
    for series, mask in zip(trn_dl.dataset.X[:, :, 0], trn_dl.dataset.pad_mask):
        past_data = series[mask]
        error = _calculate_seasonal_error(past_data, metadata.seasonality)
        if error == 0.0:
            error = _calculate_seasonal_error(past_data, 1)

        seasonal_errors_per_series.append(error)
    seasonal_errors = torch.stack(seasonal_errors_per_series, dim=0)
    # assert torch.isfinite(seasonal_errors).all()
    return seasonal_errors.unsqueeze(1)


def mse(y_hat: torch.tensor, y: torch.tensor, dim=1) -> torch.tensor:
    return (
        ((y_hat - y) ** 2).mean(dim=dim)
        if dim is not None
        else ((y_hat - y) ** 2).mean()
    )


def abs_error(y_hat: torch.tensor, y: torch.tensor, dim=1) -> torch.tensor:
    return (
        (y_hat - y).abs().sum(dim=dim) if dim is not None else (y_hat - y).abs().sum()
    )


def abs_target_sum(y: torch.tensor, dim: int = 1):
    return y.abs().sum(dim=dim) if dim is not None else y.abs().sum()


def abs_target_mean(y, dim=1):
    return y.abs().mean(dim=dim) if dim is not None else y.abs().mean()


def mae(y_hat, y, dim=1) -> torch.tensor:
    return (
        (y_hat - y).abs().mean(dim=dim) if dim is not None else (y_hat - y).abs().mean()
    )


def _single_ts_mase(
    y_hat: torch.tensor, y: torch.tensor, seasonal_errors: torch.tensor
) -> torch.tensor:
    ape = (y_hat - y).abs() / y.abs()
    mask = torch.isfinite(ape)
    ape_filtered = ape[mask]
    result = torch.mean(ape_filtered)
    result /= seasonal_errors
    # assert torch.isfinite(result)
    return result


def _mase(
    y_hat: torch.tensor, y: torch.tensor, seasonal_errors: torch.tensor
) -> torch.tensor:
    if len(y.shape) > 1:
        result = torch.stack(
            [
                _single_ts_mase(y_hat[i], y[i], seasonal_errors[i])
                for i in range(y.shape[0])
            ],
            dim=0,
        )
        return result
    else:
        return _single_ts_mase(y_hat, y, seasonal_errors)


def mase(
    y_hat: torch.tensor, y: torch.tensor, seasonal_errors: torch.tensor, dim: int = 1
) -> torch.tensor:
    return mae(y_hat, y, dim=dim) / seasonal_errors


def _single_ts_mape(y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
    ape = (y_hat - y).abs() / y.abs()
    mask = torch.isfinite(ape)
    ape_filtered = ape[mask]
    result = torch.mean(ape_filtered)
    return result


def mape(y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
    if len(y.shape) > 1:
        return torch.stack(
            [_single_ts_mape(y_hat[i], y[i]) for i in range(y.shape[0])], dim=0
        )
    else:
        return _single_ts_mape(y_hat, y)


def smape(y_hat: torch.tensor, y: torch.tensor, dim: int = 1) -> torch.tensor:
    pe = 2 * (y_hat - y).abs() / (y.abs() + y_hat.abs())
    return pe.mean(dim=dim) if dim is not None else pe.mean()


def nd(y_hat: torch.tensor, y: torch.tensor, dim: int = 1) -> torch.tensor:
    return abs_error(y_hat, y, dim=dim) / abs_target_sum(y, dim=dim)


def get_metrics_per_ts(
    y_hat: torch.tensor, y: torch.tensor, seasonal_error: torch.tensor
) -> Dict[str, torch.tensor]:
    mase_per_ts = mase(y_hat, y, seasonal_error)
    mase_per_ts = mase_per_ts[mase_per_ts.isfinite()]
    return {
        "mse": mse(y_hat, y),
        "abs_error": abs_error(y_hat, y),
        "abs_target_sum": abs_target_sum(y),
        "abs_target_mean": abs_target_mean(y),
        "mase": mase_per_ts,
        "mape": mape(y_hat, y),
        "smape": smape(y_hat, y),
        "nd": nd(y_hat, y),
        "mae": mae(y_hat, y),
        "rmse": mse(y_hat, y).sqrt(),
        "seasonal_error": seasonal_error,
    }


def aggregate_metrics(metrics_per_ts: Dict[str, torch.tensor]) -> Dict[str, float]:
    aggregate_map = {
        "mse": "mean",
        "abs_error": "sum",
        "abs_target_sum": "sum",
        "abs_target_mean": "mean",
        "mase": "mean",
        "mape": "nanmean",
        "smape": "mean",
        "nd": "mean",
        "mae": "mean",
        "rmse": "mean",
        "seasonal_error": "mean",
    }
    return {k: getattr(v, aggregate_map[k])().item() for k, v in metrics_per_ts.items()}


def calc_metrics(
    y_hat: torch.tensor, y: torch.tensor, seasonal_error: torch.tensor
) -> Dict[str, float]:
    metrics_per_ts = get_metrics_per_ts(y_hat, y, seasonal_error)
    result = aggregate_metrics(metrics_per_ts)
    return result
