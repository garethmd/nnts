import gluonts.evaluation.metrics as gluon_metrics
import torch
import torch.nn.functional as F

import nnts.data.metadata


def calculate_seasonal_error(
    trn_dl: torch.utils.data.DataLoader, metadata: nnts.data.metadata.Metadata
):
    seasonal_errors_per_series = []
    for series, mask in zip(trn_dl.dataset.X[:, :, 0], trn_dl.dataset.pad_mask):
        past_data = series[mask]
        seasonal_errors_per_series.append(
            gluon_metrics.calculate_seasonal_error(
                past_data.numpy(), metadata.freq, metadata.seasonality
            )
        )
    seasonal_errors = torch.tensor(seasonal_errors_per_series)
    return seasonal_errors.unsqueeze(1)


def mse(y_hat, y, dim=1):
    return (
        ((y_hat - y) ** 2).mean(dim=dim)
        if dim is not None
        else ((y_hat - y) ** 2).mean()
    )


def abs_error(y_hat, y, dim=1):
    return (
        (y_hat - y).abs().sum(dim=dim) if dim is not None else (y_hat - y).abs().sum()
    )


def abs_target_sum(y, dim=1):
    return y.abs().sum(dim=dim) if dim is not None else y.abs().sum()


def abs_target_mean(y, dim=1):
    return y.abs().mean(dim=dim) if dim is not None else y.abs().mean()


def mae(y_hat, y, dim=1):
    return (
        (y_hat - y).abs().mean(dim=dim) if dim is not None else (y_hat - y).abs().mean()
    )


def mase(y_hat, y, seasonal_errors, dim=1):
    return mae(y_hat, y, dim=dim) / seasonal_errors


def _single_ts_mape(y_hat, y):
    ape = (y_hat - y).abs() / y.abs()
    mask = torch.isfinite(ape)
    ape_filtered = ape[mask]
    result = torch.mean(ape_filtered)
    return result


def mape(y_hat, y):
    if len(y.shape) > 1:
        return torch.stack(
            [_single_ts_mape(y_hat[i], y[i]) for i in range(y.shape[0])], dim=0
        )
    else:
        return _single_ts_mape(y_hat, y)


def smape(y_hat, y, dim=1):
    pe = 2 * (y_hat - y).abs() / (y.abs() + y_hat.abs())
    return pe.mean(dim=dim) if dim is not None else pe.mean()


def nd(y_hat, y, dim=1):
    return abs_error(y_hat, y, dim=dim) / abs_target_sum(y, dim=dim)


def get_metrics_per_ts(y_hat, y, seasonal_error):
    return {
        "mse": mse(y_hat, y),
        "abs_error": abs_error(y_hat, y),
        "abs_target_sum": abs_target_sum(y),
        "abs_target_mean": abs_target_mean(y),
        "mase": mase(y_hat, y, seasonal_error),
        "mape": mape(y_hat, y),
        "smape": smape(y_hat, y),
        "nd": nd(y_hat, y),
        "mae": mae(y_hat, y),
        "rmse": mse(y_hat, y).sqrt(),
        "seasonal_error": seasonal_error,
    }


def aggregate_metrics(metrics_per_ts):
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


def calc_metrics(y_hat, y, seasonal_error):
    metrics_per_ts = get_metrics_per_ts(y_hat, y, seasonal_error)
    return aggregate_metrics(metrics_per_ts)
