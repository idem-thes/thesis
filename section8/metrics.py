"""Forecast-skill metrics for Section 8 - MAE, directional accuracy, Diebold-Mariano.

"""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else float("nan")


def directional_accuracy(forecast, realized) -> float:
    """Fraction of non-zero pairs where sign(forecast) == sign(realized)."""
    forecast = np.asarray(forecast, float)
    realized = np.asarray(realized, float)
    m = np.isfinite(forecast) & np.isfinite(realized)
    f, r = forecast[m], realized[m]
    valid = (f != 0) & (r != 0)
    return float(np.mean(np.sign(f[valid]) == np.sign(r[valid]))) if valid.any() else float("nan")


def dm_test(forecast_model, forecast_baseline, realized, lag: int = 1) -> tuple[float, float]:
    """Diebold-Mariano with Newey-West HAC variance; squared-error loss.


    """
    fm = np.asarray(forecast_model, float)
    fb = np.asarray(forecast_baseline, float)
    rz = np.asarray(realized, float)
    m = np.isfinite(fm) & np.isfinite(fb) & np.isfinite(rz)
    if m.sum() < max(2, lag + 1):
        return float("nan"), float("nan")
    d = (rz[m] - fm[m]) ** 2 - (rz[m] - fb[m]) ** 2
    T = len(d)
    d_bar = float(np.mean(d))
    var_lr = float(np.mean((d - d_bar) ** 2))
    for k in range(1, lag + 1):
        if k >= T:
            break
        cov_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var_lr += 2.0 * (1.0 - k / (lag + 1.0)) * cov_k
    if var_lr <= 0.0 or not np.isfinite(var_lr):
        return float("nan"), float("nan")
    dm_stat = d_bar / np.sqrt(var_lr / T)
    from scipy.stats import norm

    return float(dm_stat), float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))


def forecast_report(realized, pred, *, baseline, lag: int = 1) -> dict:
    """{mae, mae_baseline, dir_acc, dm_stat, dm_p, n} for one (horizon, model)."""
    realized = np.asarray(realized, float)
    pred = np.asarray(pred, float)
    baseline = np.asarray(baseline, float)
    stat, p = dm_test(pred, baseline, realized, lag=lag)
    m = np.isfinite(realized) & np.isfinite(pred)
    return {
        "mae": mae(realized, pred),
        "mae_baseline": mae(realized, baseline),
        "dir_acc": directional_accuracy(pred, realized),
        "dm_stat": stat,
        "dm_p": p,
        "n": int(m.sum()),
    }
