"""Section 6 motivation figures - pure functions for the three figures.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def lag_zero_weight(alpha: float, delta: float, dt: float) -> float:
    """Weight on the most recent (lag-0) return in the discretised normalised TSPL feature.

    Implements the closed form (alpha - 1) / delta * dt, derived from
    K_norm(0) = (alpha - 1) / delta with the kernel
    K_norm(tau) = (tau + delta)^{-alpha} / Z, Z = delta^{1 - alpha} / (alpha - 1).
    See Guyon_model_main_article.pdf Section 3.1 eq (11), (12).
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1 for normalisable TSPL, got {alpha}")
    return (alpha - 1.0) / delta * dt


def tspl_kernel(tau: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """Normalised time-shifted power-law kernel K_norm(tau) = (tau + delta)^{-alpha} / Z.

    Z = delta^{1 - alpha} / (alpha - 1) is the integral of the un-normalised kernel
    over [0, infty). See Guyon_model_main_article.pdf Section 3.1 eq (11), (12).
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1 for normalisable TSPL, got {alpha}")
    Z = delta ** (1.0 - alpha) / (alpha - 1.0)
    return (tau + delta) ** (-alpha) / Z


def compute_R_n_at(
    return_times: np.ndarray,
    returns: np.ndarray,
    eval_time: float,
    n: int,
    alpha: float,
    delta: float,
    dt: float,
) -> float:
    """Discrete TSPL feature R_{n,t} = sum K(t - t_i) * r_{t_i}^n * dt over t_i <= eval_time.

    Implements Guyon_model_main_article.pdf Section 3.1 eq (10) with the normalised TSPL
    kernel of eq (11), (12) and a per-observation step `dt` (in years).
    """
    if return_times.shape != returns.shape:
        raise ValueError(
            f"return_times {return_times.shape} and returns {returns.shape} must match"
        )
    mask = return_times <= eval_time
    lags = eval_time - return_times[mask]
    K = tspl_kernel(lags, alpha=alpha, delta=delta)
    return float(np.sum(K * returns[mask] ** n) * dt)


def sigma_from_features(R1: float, R2: float, beta_0: float, beta_1: float, beta_2: float) -> float:
    """Linear PDV combiner sigma_t = beta_0 + beta_1 R_1 + beta_2 sqrt(R_2).

    See Guyon_model_main_article.pdf Section 3.2 eq (9). R_2 is clipped at zero before
    sqrt to defend against tiny numerical negatives
    """
    return beta_0 + beta_1 * R1 + beta_2 * np.sqrt(max(0.0, R2))


def _truncated_Z(
    alpha: float, delta: float, max_delta: int = 1000, dt_grid: float = 1.0 / 252
) -> float:
    """Empirical truncated normaliser sum_{i=0}^{max_delta-1} (i*dt + delta)^{-alpha} * dt.

    Mirrors code_guyon/empirical_study/empirical_study_tspl.py:466-467.
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1 for normalisable TSPL, got {alpha}")
    return float(np.sum((np.arange(max_delta) * dt_grid + delta) ** (-alpha))) * dt_grid


def _compute_R_n_truncated(
    return_times: np.ndarray,
    returns: np.ndarray,
    eval_time: float,
    n: int,
    alpha: float,
    delta: float,
    Z: float,
) -> float:
    """Guyon-convention feature: sum_i K_unnorm(t - t_i) * r_i^n / Z. No dt factor.

    Matches code_guyon/empirical_study/utils.py:147-161 (compute_kernel_weighted_sum):
    the feature is the un-normalised kernel sum, divided by Z so reported betas in
    perform_empirical_study's opt_params dict apply directly (those betas absorb
    Z^j via empirical_study_tspl.py:476).
    """
    if return_times.shape != returns.shape:
        raise ValueError(
            f"return_times {return_times.shape} and returns {returns.shape} must match"
        )
    mask = return_times <= eval_time
    lags = eval_time - return_times[mask]
    K = (lags + delta) ** (-alpha) / Z
    return float(np.sum(K * returns[mask] ** n))


def intraday_substep_demo(
    daily_return: float,
    sub_returns: tuple[float, ...],
    table3: dict,
    R_init1: float = 0.0,
    R_init2: float = 0.0,
) -> dict:
    """Compare cumulative sigma evolution under daily vs N-substep encoding.

    Returns are simple returns r_t = P_t/P_{t-1} - 1 (code_guyon/empirical_study/
    utils.py:175 convention).

    Uses Guyon's empirical truncated kernel (max_delta=1000 trading days) for both
    R_1 and R_2 - matches the normalisation that Table 3 betas were calibrated for.

    R_init1, R_init2 represent pre-history kernel-weighted feature values just before
    the demo's day, i.e. (R_1_unnorm / Z_1) and (R_2_unnorm / Z_2) computed from real
    SPX history at the as-of date. 

    Returns dict with:
        sigma_initial: sigma at the supplied pre-history.
        sigma_daily: sigma after one daily step with return daily_return.
        sigma_substep: sigma after all n substeps.
        dsigma_daily: sigma_daily - sigma_initial.
        dsigma_substep_total: sigma_substep - sigma_initial.
        sigma_path: [sigma_initial, sigma_after_step_1, ..., sigma_after_step_n].
        dsigma_per_step: per-substep change, length n_sub.
        delta_sigma: sigma_substep - sigma_daily (path-dependence gap).
    """
    expected = 1.0 + daily_return
    actual = float(np.prod([1.0 + r for r in sub_returns]))
    if not math.isclose(expected, actual, rel_tol=1e-6):
        raise ValueError(
            f"sub_returns {sub_returns} must compound to (1+daily_return)={expected:.6f}, "
            f"got {actual:.6f}"
        )

    dt = 1.0 / 252
    n_sub = len(sub_returns)
    sub_dt = dt / n_sub
    sub_times = np.array([1.0 - (n_sub - 1 - i) * sub_dt for i in range(n_sub)])
    sub_returns_arr = np.array(sub_returns)

    beta_0 = table3["beta_0"]
    beta_1 = table3["beta_1"]
    beta_2 = table3["beta_2"]
    alpha_1, delta_1 = table3["alpha_1"], table3["delta_1"]
    alpha_2, delta_2 = table3["alpha_2"], table3["delta_2"]

    Z1 = _truncated_Z(alpha_1, delta_1)
    Z2 = _truncated_Z(alpha_2, delta_2)

    sigma_initial = sigma_from_features(R_init1, R_init2, beta_0, beta_1, beta_2)

    R1_daily_step = _compute_R_n_truncated(
        np.array([1.0]),
        np.array([daily_return]),
        1.0,
        n=1,
        alpha=alpha_1,
        delta=delta_1,
        Z=Z1,
    )
    R2_daily_step = _compute_R_n_truncated(
        np.array([1.0]),
        np.array([daily_return]),
        1.0,
        n=2,
        alpha=alpha_2,
        delta=delta_2,
        Z=Z2,
    )
    sigma_daily = sigma_from_features(
        R_init1 + R1_daily_step, R_init2 + R2_daily_step, beta_0, beta_1, beta_2
    )

    sigma_path = [sigma_initial]
    dsigma_per_step = []
    for k in range(1, n_sub + 1):
        times_k = sub_times[:k]
        returns_k = sub_returns_arr[:k]
        R1_step_k = _compute_R_n_truncated(
            times_k,
            returns_k,
            float(times_k[-1]),
            n=1,
            alpha=alpha_1,
            delta=delta_1,
            Z=Z1,
        )
        R2_step_k = _compute_R_n_truncated(
            times_k,
            returns_k,
            float(times_k[-1]),
            n=2,
            alpha=alpha_2,
            delta=delta_2,
            Z=Z2,
        )
        sigma_k = sigma_from_features(
            R_init1 + R1_step_k, R_init2 + R2_step_k, beta_0, beta_1, beta_2
        )
        sigma_path.append(sigma_k)
        dsigma_per_step.append(sigma_k - sigma_path[-2])

    sigma_substep = sigma_path[-1]
    return {
        "sigma_initial": sigma_initial,
        "sigma_daily": sigma_daily,
        "sigma_substep": sigma_substep,
        "dsigma_daily": sigma_daily - sigma_initial,
        "dsigma_substep_total": sigma_substep - sigma_initial,
        "sigma_path": sigma_path,
        "dsigma_per_step": dsigma_per_step,
        "delta_sigma": sigma_substep - sigma_daily,
    }


def initialize_R_exp_markov(
    lam: Sequence[float],
    past_prices: pd.Series,
    max_delta: int = 1000,
    squared: bool = False,
) -> np.ndarray:
    """Init two-component R_{n,j} (j=0,1) from past prices via exponential kernels.

    Mirrors code_guyon/calibration/torch_montecarlo.py:18-34 (initialize_R) but in
    numpy. Returns are (1 - P_{t-1}/P_t) per Guyon's torch_montecarlo convention;
    `squared=True` raises returns to the second power (used for R_2 init).
    """
    lam_arr = np.asarray(lam, dtype=float)
    if lam_arr.shape != (2,):
        raise ValueError(f"lam must have shape (2,), got {lam_arr.shape}")
    rets = (1.0 - past_prices.shift(1) / past_prices).iloc[-max_delta:][::-1].to_numpy()
    rets = rets[~np.isnan(rets)]
    ts = np.arange(rets.shape[0]) * (1.0 / 252)
    x = rets**2 if squared else rets
    out = np.empty(2)
    for j in range(2):
        out[j] = float(np.sum(lam_arr[j] * np.exp(-lam_arr[j] * ts) * x))
    return out


def markov_bridge_intraday_demo(
    chunk_targets: Sequence[float],
    N_per_chunk: int,
    lam1: Sequence[float],
    lam2: Sequence[float],
    betas: Sequence[float],
    theta1: float,
    theta2: float,
    R_init1: Sequence[float] = (0.0, 0.0),
    R_init2: Sequence[float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
    vol_cap: float | None = 1.5,
) -> dict:
    """4FMPDV Markovian Brownian bridge for one trading day, sigma evolved per substep.

    Implements Guyon_model_main_article.pdf Section 5.4 eq (44)-(47):
        dS_t / S_t = sigma_t dW_t,
        dR_{1,j,t} = lam1[j] * (sigma_t dW_t - R_{1,j,t} dt),    j in {0, 1},
        dR_{2,j,t} = lam2[j] * (sigma_t^2 - R_{2,j,t}) dt,       j in {0, 1},
        R_{n,t} = (1 - theta_n) R_{n,0,t} + theta_n R_{n,1,t},
        sigma_t = beta_0 + beta_1 R_{1,t} + beta_2 sqrt(R_{2,t}).
    Per-substep update mirrors code_guyon/calibration/torch_montecarlo.py:166-167:
        R_{1,j} <- exp(-lam1[j] * sub_dt) * (R_{1,j} + lam1[j] * dx),
        R_{2,j} <- exp(-lam2[j] * sub_dt) * (R_{2,j} + lam2[j] * sigma_now^2 * sub_dt).
    Within chunk c with target log-return L_c = log(1 + chunk_targets[c]), log-prices
    follow a Markovian Brownian bridge conditioned on hitting L_c at the chunk's end.
    Last substep in each chunk is forced (steps_left=1, var=0) so multiplicative
    compound is exact. Optional vol_cap clamps sigma_t (default 1.5, matching
    torch_montecarlo); pass None to disable.
    """
    n_chunks = len(chunk_targets)
    if n_chunks == 0:
        raise ValueError("chunk_targets must be non-empty")
    if N_per_chunk < 1:
        raise ValueError(f"N_per_chunk must be >= 1, got {N_per_chunk}")

    lam1_arr = np.asarray(lam1, dtype=float)
    lam2_arr = np.asarray(lam2, dtype=float)
    R1 = np.asarray(R_init1, dtype=float).copy()
    R2 = np.asarray(R_init2, dtype=float).copy()
    if lam1_arr.shape != (2,) or lam2_arr.shape != (2,):
        raise ValueError("lam1 and lam2 must each have shape (2,)")
    if R1.shape != (2,) or R2.shape != (2,):
        raise ValueError("R_init1 and R_init2 must each have shape (2,)")
    betas_arr = np.asarray(betas, dtype=float)
    if betas_arr.shape != (3,):
        raise ValueError(f"betas must have shape (3,), got {betas_arr.shape}")
    beta_0, beta_1, beta_2 = float(betas_arr[0]), float(betas_arr[1]), float(betas_arr[2])

    if rng is None:
        from code_section6.seeds import seed_for

        rng = np.random.default_rng(seed_for("markov_bridge_intraday_demo"))

    dt_day = 1.0 / 252
    n_sub = n_chunks * N_per_chunk
    sub_dt = dt_day / n_sub
    sub_times = np.array([1.0 - (n_sub - 1 - i) * sub_dt for i in range(n_sub)])
    L_chunks = np.array([math.log(1.0 + c) for c in chunk_targets])
    chunk_end_x = np.cumsum(L_chunks)

    def _sigma(R1_pair: np.ndarray, R2_pair: np.ndarray) -> float:
        R1_blend = (1.0 - theta1) * R1_pair[0] + theta1 * R1_pair[1]
        R2_blend = (1.0 - theta2) * R2_pair[0] + theta2 * R2_pair[1]
        sig = beta_0 + beta_1 * R1_blend + beta_2 * math.sqrt(max(0.0, R2_blend))
        if vol_cap is None:
            return float(sig)
        return float(min(sig, vol_cap))

    sigma_initial = _sigma(R1, R2)

    R1_d = R1.copy()
    R2_d = R2.copy()
    L_total = float(L_chunks.sum())
    for j in range(2):
        R1_d[j] = math.exp(-lam1_arr[j] * dt_day) * (R1_d[j] + lam1_arr[j] * L_total)
        R2_d[j] = math.exp(-lam2_arr[j] * dt_day) * (
            R2_d[j] + lam2_arr[j] * sigma_initial**2 * dt_day
        )
    sigma_daily = _sigma(R1_d, R2_d)

    sub_returns = np.empty(n_sub)
    sigma_path: list[float] = [sigma_initial]
    R1_path: list[np.ndarray] = [R1.copy()]
    R2_path: list[np.ndarray] = [R2.copy()]
    x_current = 0.0

    for c in range(n_chunks):
        x_target = float(chunk_end_x[c])
        for i_in_chunk in range(N_per_chunk):
            sigma_now = sigma_path[-1]
            steps_left = N_per_chunk - i_in_chunk
            L_rem = x_target - x_current
            if steps_left == 1:
                dx = L_rem
            else:
                bridge_mean = L_rem / steps_left
                bridge_var = (sigma_now**2) * sub_dt * (steps_left - 1) / steps_left
                bridge_std = math.sqrt(max(0.0, bridge_var))
                dx = bridge_mean + bridge_std * float(rng.standard_normal())

            for j in range(2):
                R1[j] = math.exp(-lam1_arr[j] * sub_dt) * (R1[j] + lam1_arr[j] * dx)
                R2[j] = math.exp(-lam2_arr[j] * sub_dt) * (
                    R2[j] + lam2_arr[j] * sigma_now**2 * sub_dt
                )

            sigma_path.append(_sigma(R1, R2))
            R1_path.append(R1.copy())
            R2_path.append(R2.copy())

            i_total = c * N_per_chunk + i_in_chunk
            sub_returns[i_total] = float(math.exp(dx) - 1.0)
            x_current += dx

    return {
        "sub_returns": sub_returns,
        "sub_times": sub_times,
        "sigma_path": sigma_path,
        "R1_path": np.array(R1_path),
        "R2_path": np.array(R2_path),
        "sigma_initial": sigma_initial,
        "sigma_daily": sigma_daily,
        "sigma_substep": sigma_path[-1],
        "compound": float(np.prod(1.0 + sub_returns)),
    }


def fetch_spx_vix_yahoo(start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """Fetch SPX close + VIX close from Yahoo Finance, normalised to date-only index.

    Mirrors code_guyon/empirical_study.ipynb cells 3-5. VIX is returned in decimal
    form (close / 100) so it sits on the [0, 1] scale the linear PDV model fits.
    """
    import yfinance as yf

    spx_data = yf.Ticker("^GSPC").history(start=start, end=end)
    vix_data = yf.Ticker("^VIX").history(start=start, end=end)
    spx_data.index = pd.to_datetime(spx_data.index.date)
    vix_data.index = pd.to_datetime(vix_data.index.date)
    return spx_data["Close"], vix_data["Close"] / 100
