"""Method A forecast - kernel anchored at t with r_t = 0.

For each evaluation date t, computes the trader's 10am ex-ante forecast of VIX_t:
sigma_t^{(A)} = beta_0 + beta_1 * (R_1_unnorm / Z_trunc_1) + beta_2 * sqrt(R_2_unnorm / Z_trunc_2),
where R_n_unnorm = sum_{i>=1} K(i*dt) * r_{t-i*dt}^n (lag-0 slot empty by construction;
no dt factor - matches Guyon's compute_kernel_weighted_sum convention).

The reported betas in perform_empirical_study's opt_params dict already absorb
the kernel normaliser (code_guyon/empirical_study/empirical_study_tspl.py:476):
beta_reported = beta_raw * Z_trunc^j. To use them we divide unnormalised R_n by
Z_trunc (j=1) before the linear part and by Z_trunc (under sqrt, j=1/2) before
the volatility part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from code_section6.motivation import _truncated_Z


def _r_n_unnorm_method_a(
    history_returns: np.ndarray,
    n: int,
    alpha: float,
    delta: float,
    dt: float,
    max_delta: int,
) -> float:
    """Sum K(i*dt) * r_{t-i*dt}^n over i=1..max_delta-1 (skip lag 0 => r_t = 0)

    history_returns is ordered most-recent-last. We reverse so index 0 = r_{t-1}
    """
    n_avail = min(len(history_returns), max_delta - 1)
    if n_avail == 0:
        return 0.0
    r_lagged = history_returns[-n_avail:][::-1]
    lags = (np.arange(1, n_avail + 1)) * dt
    K = (lags + delta) ** (-alpha)
    return float(np.sum(K * r_lagged**n))


def method_a_forecast(
    spx_returns: pd.Series,
    eval_dates: pd.DatetimeIndex,
    params: dict,
    dt: float = 1.0 / 252,
    max_delta: int = 1000,
) -> pd.Series:
    """Compute sigma_t^{(A)} for each t in eval_dates using returns strictly before t."""
    Z1 = _truncated_Z(params["alpha_1"], params["delta_1"], max_delta=max_delta, dt_grid=dt)
    Z2 = _truncated_Z(params["alpha_2"], params["delta_2"], max_delta=max_delta, dt_grid=dt)

    forecasts: dict = {}
    for t in eval_dates:
        history_returns = spx_returns[spx_returns.index < t].to_numpy()

        R1_unnorm = _r_n_unnorm_method_a(
            history_returns,
            n=1,
            alpha=params["alpha_1"],
            delta=params["delta_1"],
            dt=dt,
            max_delta=max_delta,
        )
        R2_unnorm = _r_n_unnorm_method_a(
            history_returns,
            n=2,
            alpha=params["alpha_2"],
            delta=params["delta_2"],
            dt=dt,
            max_delta=max_delta,
        )
        feature_1 = R1_unnorm / Z1
        feature_2 = float(np.sqrt(max(0.0, R2_unnorm / Z2)))
        forecasts[t] = (
            params["beta_0"] + params["beta_1"] * feature_1 + params["beta_2"] * feature_2
        )

    return pd.Series(forecasts, name="method_a")


def evaluate_forecast(forecast: pd.Series, target: pd.Series) -> dict:
    """Level + dynamics metrics for a forecast against a target series."""
    common = forecast.index.intersection(target.index)
    f = forecast.reindex(common).dropna()
    t = target.reindex(f.index).dropna()
    f = f.reindex(t.index)

    rmse = float(np.sqrt(np.mean((f - t) ** 2)))

    df = f.diff().dropna()
    dt_target = t.diff().reindex(df.index).dropna()
    df = df.reindex(dt_target.index)

    sign_hit = float((np.sign(df) == np.sign(dt_target)).mean())
    direc_corr = float(df.corr(dt_target))

    abs_dvix = dt_target.abs()
    median_dvix = abs_dvix.median()
    calm_mask = abs_dvix < median_dvix
    volatile_mask = ~calm_mask
    calm_sign_hit = float((np.sign(df[calm_mask]) == np.sign(dt_target[calm_mask])).mean())
    volatile_sign_hit = float(
        (np.sign(df[volatile_mask]) == np.sign(dt_target[volatile_mask])).mean()
    )

    mz_beta, mz_alpha = np.polyfit(f.to_numpy(), t.to_numpy(), 1)

    return {
        "n_obs": int(len(f)),
        "rmse": rmse,
        "sign_hit_rate": sign_hit,
        "directional_corr": direc_corr,
        "sign_hit_calm": calm_sign_hit,
        "sign_hit_volatile": volatile_sign_hit,
        "mz_alpha": float(mz_alpha),
        "mz_beta": float(mz_beta),
    }
