import gluonts.evaluation.metrics as metrics
import torch
import torch.nn.functional as F


def mean_absolute_error(x: torch.Tensor, y: torch.Tensor, reduction="mean"):
    if reduction == "median":
        return (x - y).abs().median(1)
    else:
        return (x - y).abs().mean(1)


def rmse(x: torch.Tensor, y: torch.Tensor, reduction="mean"):
    if reduction == "median":
        return (x - y).pow(2).mean().sqrt().median(1)
    else:
        return (x - y).pow(2).mean(1).sqrt()


def calc_metrics(y, y_hat, period="ME", seasonality=12):
    mae = mean_absolute_error(y, y_hat).mean().item()
    test_rmse = rmse(y, y_hat).mean()
    mse = metrics.mse(y.numpy(), y_hat.numpy())
    mape = metrics.mape(y.numpy(), y_hat.numpy())
    smape = metrics.smape(y.numpy(), y_hat.numpy())
    abs_error = metrics.abs_error(y.numpy(), y_hat.numpy())
    seasonal_error = metrics.calculate_seasonal_error(
        y.numpy(), period, seasonality
    )  # needs to be prior history
    mase = metrics.mase(y.numpy(), y_hat.numpy(), seasonal_error)
    test_rmse = rmse(y, y_hat).mean().item()

    test_metrics = {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "abs_error": abs_error,
        "mase": mase,
        "rmse": test_rmse,
    }
    return test_metrics
