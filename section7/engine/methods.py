"""Calibration methods for the Section 7 synthetic-PDV recovery engine.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


_GUYON_ROOT = Path(__file__).resolve().parents[2] / "code_guyon"
if str(_GUYON_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUYON_ROOT))

from empirical_study.empirical_study_2exp import (  # noqa: E402
    find_optimal_parameters_exp,
    fit_betas_exp,
    linear_of_kernels_exp,
)
from empirical_study.utils import dataframe_of_returns  # noqa: E402

# ---- engine-wide constants (simulator-internal year per 2026-05-14 amendment) ----
MINUTES_PER_YEAR = 252 * 1440  # 362_880; simulator's dt=1/252 * 1440 steps/day
DAY_STEPS = 1440  # 1-min steps per simulator-day
RTH_STEPS = 390  # RTH minutes per trading day (09:30 -> 16:00)
OVERNIGHT_GAP_STEPS = DAY_STEPS - RTH_STEPS  # 1050
WEEKEND_GAP_STEPS = 2 * DAY_STEPS + OVERNIGHT_GAP_STEPS  # 3930

# Parameter ordering (9 params):
#   [lam1_0, lam1_1, lam2_0, lam2_1, beta_0, beta_1, beta_2, theta_1, theta_2]
PARAM_ORDER = (
    "lam1_0",
    "lam1_1",
    "lam2_0",
    "lam2_1",
    "beta_0",
    "beta_1",
    "beta_2",
    "theta_1",
    "theta_2",
)

# Guyon-style 1000-day lookback (user-decreed; non-negotiable).
GUYON_MAX_DELTA = 1000


@dataclass
class FitResult:
    """Engine-internal fit container. Reused by Tasks 8-9 (M3a/M3b/M4).
    """

    theta_hat: np.ndarray
    fit_time_sec: float
    method_name: str
    diagnostics: dict


def _scalar_dt_years(dt_array: np.ndarray) -> float:
    """Collapse per-bar Delta t to a single scalar in simulator-years.
    """
    dt_min = float(np.median(dt_array))
    return dt_min / MINUTES_PER_YEAR


def _to_guyon_series(
    arrays: dict[str, pd.DataFrame],
    path_id: int,
    start_date: str = "1995-01-01",
) -> tuple[pd.Series, pd.Series]:
    """Convert one path's (S, sigma) into pd.Series with business-day DatetimeIndex.
    """
    n = len(arrays["S"])
    idx = pd.bdate_range(start=start_date, periods=n)
    prices = pd.Series(arrays["S"].iloc[:, path_id].to_numpy(), index=idx)
    vol = pd.Series(arrays["sigma"].iloc[:, path_id].to_numpy(), index=idx)
    return prices, vol


def _guyon_dict_to_param_vector(opt_params: dict) -> np.ndarray:
    """Map Guyon's ``opt_params`` dict to our 9-vector ordered per ``PARAM_ORDER``.
    """
    return np.array(
        [
            float(opt_params["lambda_1"][0]),  # lambda_1_0 (kernel 1's first decay rate)
            float(opt_params["lambda_2"][0]),  # lambda_1_1 (kernel 1's second decay rate)
            float(opt_params["lambda_1"][1]),  # lambda_2_0 (kernel 2's first decay rate)
            float(opt_params["lambda_2"][1]),  # lambda_2_1 (kernel 2's second decay rate)
            float(opt_params["beta_0"]),
            float(opt_params["beta_1"]),
            float(opt_params["beta_2"]),
            float(opt_params["theta_1"]),
            float(opt_params["theta_2"]),
        ]
    )


def _train_test_dates(
    prices: pd.Series,
    max_delta: int = GUYON_MAX_DELTA,
    test_size: int = 50,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Pick (train_start, test_start, test_end) so the convolution lookback fits.
    """
    n = len(prices)
    if n <= max_delta + test_size + 1:
        raise ValueError(
            f"Path too short for Guyon calibration: need > {max_delta + test_size + 1} "
            f"daily bars (1000 lookback + test buffer), got {n}."
        )
    train_start_date = prices.index[max_delta]
    test_end_date = prices.index[-1]
    test_start_date = prices.index[-test_size]
    return train_start_date, test_start_date, test_end_date


def _fit_via_guyon(
    arrays: dict[str, pd.DataFrame],
    dt_array: np.ndarray,
    path_id: int,
    method_name: str,
    use_jacob: bool,
) -> FitResult:

    dt_years = _scalar_dt_years(dt_array)
    prices, vol = _to_guyon_series(arrays, path_id)
    train_start_date, test_start_date, test_end_date = _train_test_dates(prices)
    t0 = time.perf_counter()
    sol = find_optimal_parameters_exp(
        vol=vol,
        index=prices,
        p=1,
        setting=[(1, 1), (2, 1 / 2)],
        train_start_date=train_start_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        max_delta=GUYON_MAX_DELTA,
        use_jacob=use_jacob,
    )
    elapsed = time.perf_counter() - t0
    theta_hat = _guyon_dict_to_param_vector(sol["opt_params"])
    return FitResult(
        theta_hat=theta_hat,
        fit_time_sec=elapsed,
        method_name=method_name,
        diagnostics={
            "opt_params": sol["opt_params"],
            "train_r2": float(sol["train_r2"]),
            "test_r2": float(sol["test_r2"]),
            "train_rmse": float(sol["train_rmse"]),
            "test_rmse": float(sol["test_rmse"]),
            "dt_years_used": dt_years,
            "max_delta": GUYON_MAX_DELTA,
            "use_jacob": use_jacob,
        },
    )


def fit_m1(
    arrays: dict[str, pd.DataFrame],
    dt_array: np.ndarray,
    path_id: int,
    x0: np.ndarray | None = None,  # noqa: ARG001 - unused, kept for downstream symmetry
) -> FitResult:
    """M1: Guyon ``find_optimal_parameters_exp`` with numerical Jacobian.
    """
    return _fit_via_guyon(arrays, dt_array, path_id, method_name="M1", use_jacob=False)


def fit_m2(
    arrays: dict[str, pd.DataFrame],
    dt_array: np.ndarray,
    path_id: int,
    x0: np.ndarray | None = None,  # noqa: ARG001 - unused, kept for downstream symmetry
) -> FitResult:
    """M2: Guyon ``find_optimal_parameters_exp`` with analytical Jacobian.

    Identical to :func:`fit_m1` except ``use_jacob=True`` - scipy's TRF
    consumes the analytical Jacobian defined in
    :func:`empirical_study.empirical_study_2exp.find_optimal_parameters_exp`
    (its inner ``jacobian`` function). All other plumbing - smart x0,
    1000-day convolution lookback - is shared with M1.
    """
    return _fit_via_guyon(arrays, dt_array, path_id, method_name="M2", use_jacob=True)



GUYON_SETTING = [(1, (1,)), (2, (1 / 2,))]


def _theta_hat_to_guyon_params(theta_hat: np.ndarray) -> np.ndarray:
    """Map our 9-vector (``PARAM_ORDER``) to Guyon's flat parameter layout.


    """
    lam1_0, lam1_1, lam2_0, lam2_1, beta_0, beta_1, beta_2, theta_1, theta_2 = theta_hat
    return np.array(
        [beta_0, beta_1, beta_2, theta_1, theta_2, lam1_0, lam1_1, lam2_0, lam2_1],
        dtype=float,
    )


def _stack_features_for_design(features: dict) -> np.ndarray:
    """Stack Guyon's nested feature dict into an N*n_features design matrix.
    """
    cols = []
    for key in features:
        cols.extend(list(features[key].values()))
    return np.asarray(cols, dtype=float).T


def _build_fast_design(
    fast_arrays: dict[str, pd.DataFrame],
    path_id: int,
    stage1_theta_hat: np.ndarray,
    test_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the Stage-2 train/test design matrices on the fast tape.

    """
    prices_fast, vol_fast = _to_guyon_series(fast_arrays, path_id)
    df_returns = dataframe_of_returns(index=prices_fast, vol=vol_fast, max_delta=GUYON_MAX_DELTA)
    df_returns = df_returns.dropna()
    if len(df_returns) <= test_size + 1:
        raise ValueError(
            f"Fast tape too short after dropna for Stage 2: need > {test_size + 1} "
            f"bars after 1000-bar lookback warmup, got {len(df_returns)}."
        )
    cols = [f"r_(t-{lag})" for lag in range(GUYON_MAX_DELTA)]
    X_returns = df_returns[cols]
    y_vol = df_returns["vol"].to_numpy()

    guyon_params = _theta_hat_to_guyon_params(stage1_theta_hat)
    features_full, _ = linear_of_kernels_exp(
        returns=X_returns,
        setting=GUYON_SETTING,
        parameters=guyon_params,
        return_features=True,
    )
    X_features = _stack_features_for_design(features_full)
    # Prepend intercept column so X is directly consumable by QuantReg /
    # manual residual math.
    X_design = np.column_stack([np.ones(X_features.shape[0]), X_features])

    train_X = X_design[:-test_size]
    train_y = y_vol[:-test_size]
    test_X = X_design[-test_size:]
    test_y = y_vol[-test_size:]
    return train_X, train_y, test_X, test_y, X_returns


def _r2_and_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Closed-form R^2 + RMSE on sigma. Matches ``sklearn.metrics.r2_score`` for

    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y_true)))
    return r2, rmse


def fit_m3a(
    daily_arrays: dict[str, pd.DataFrame],
    fast_arrays: dict[str, pd.DataFrame],
    fast_dt_array: np.ndarray,
    path_id: int,
    loss: str = "L2",
) -> FitResult:
    """M3a: tiered (Theta from daily M2, beta re-fit on fast tape).

    """
    if loss not in ("L2", "MAE"):
        raise ValueError(f"fit_m3a: unsupported loss '{loss}'. Use 'L2' or 'MAE'.")
    n_daily = len(daily_arrays["sigma"])
    daily_dt_array = np.full(n_daily, float(DAY_STEPS))
    stage1 = fit_m2(daily_arrays, daily_dt_array, path_id)

    t0 = time.perf_counter()
    train_X, train_y, test_X, test_y, _ = _build_fast_design(fast_arrays, path_id, stage1.theta_hat)

    if loss == "L2":
        # Reuse Guyon's helper for parity with M1/M2's beta-pathway. It
        # internally re-builds the features (so we pay a small redundant
        # cost vs. our already-stacked design); the wins are: (a) we run
        # exactly the operation Guyon's own code does, no chance of
        # subtle column-order drift, (b) the L2 closed-form is fast.
        guyon_params = _theta_hat_to_guyon_params(stage1.theta_hat)
        prices_fast, vol_fast = _to_guyon_series(fast_arrays, path_id)
        df_returns = dataframe_of_returns(
            index=prices_fast, vol=vol_fast, max_delta=GUYON_MAX_DELTA
        ).dropna()
        cols = [f"r_(t-{lag})" for lag in range(GUYON_MAX_DELTA)]
        # Stage 2 fit window = all-but-test-tail (same as _build_fast_design)
        train_returns = df_returns.iloc[:-50][cols]
        train_y_full = df_returns.iloc[:-50]["vol"]
        beta_vec = fit_betas_exp(
            parameters=guyon_params,
            X_train=train_returns,
            y_train=train_y_full,
            setting=GUYON_SETTING,
        )
    else:  # MAE
        import statsmodels.api as sm

        quant_res = sm.QuantReg(train_y, train_X).fit(q=0.5, disp=False)
        beta_vec = np.asarray(quant_res.params, dtype=float)

    stage2_elapsed = time.perf_counter() - t0

    theta_hat = stage1.theta_hat.copy()
    theta_hat[4] = float(beta_vec[0])  # beta_0
    theta_hat[5] = float(beta_vec[1])  # beta_1
    theta_hat[6] = float(beta_vec[2])  # beta_2

    train_pred = train_X @ beta_vec
    test_pred = test_X @ beta_vec
    train_r2, train_rmse = _r2_and_rmse(train_y, train_pred)
    test_r2, test_rmse = _r2_and_rmse(test_y, test_pred)

    return FitResult(
        theta_hat=theta_hat,
        fit_time_sec=stage1.fit_time_sec + stage2_elapsed,
        method_name=f"M3a-{loss}",
        diagnostics={
            "stage1_theta_hat": stage1.theta_hat.copy(),
            "stage1_diagnostics": stage1.diagnostics,
            "stage2_beta": beta_vec.copy(),
            "loss": loss,
            "stage1_fit_time_sec": stage1.fit_time_sec,
            "stage2_fit_time_sec": stage2_elapsed,
            "fast_dt_years_used": _scalar_dt_years(fast_dt_array),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "max_delta": GUYON_MAX_DELTA,
        },
    )


def fit_m3b(
    daily_arrays: dict[str, pd.DataFrame],
    fast_arrays: dict[str, pd.DataFrame],
    fast_dt_array: np.ndarray,
    path_id: int,
) -> FitResult:
    """M3b: tiered observe-only (Theta from daily M2, no Stage-2 refit).

    """
    n_daily = len(daily_arrays["sigma"])
    daily_dt_array = np.full(n_daily, float(DAY_STEPS))
    stage1 = fit_m2(daily_arrays, daily_dt_array, path_id)

    # Build fast-tape design with Stage-1 (lambda, theta) for features, then evaluate
    # the FULL Stage-1 beta vector (no refit).
    _, _, test_X, test_y, _ = _build_fast_design(fast_arrays, path_id, stage1.theta_hat)
    # Stage-1 beta in (beta_0, beta_1, beta_2) order matches our intercept-first design.
    stage1_beta = np.array(
        [stage1.theta_hat[4], stage1.theta_hat[5], stage1.theta_hat[6]], dtype=float
    )
    test_pred = test_X @ stage1_beta
    test_r2, test_rmse = _r2_and_rmse(test_y, test_pred)

    return FitResult(
        theta_hat=stage1.theta_hat.copy(),
        fit_time_sec=stage1.fit_time_sec,
        method_name="M3b",
        diagnostics={
            "stage1_theta_hat": stage1.theta_hat.copy(),
            "stage1_diagnostics": stage1.diagnostics,
            "fast_dt_years_used": _scalar_dt_years(fast_dt_array),
            "test_r2": float(test_r2),
            "test_rmse": float(test_rmse),
            "max_delta": GUYON_MAX_DELTA,
        },
    )



EPS_R2 = 1e-8  


_UKF_ALPHA = 1e-2
_UKF_BETA = 2.0
_UKF_KAPPA = 3 - 5  # = -2


def _m4_propagate_step(
    state: np.ndarray,
    dt_years: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Deterministic core of the simulator's eq-47 recursion (drift terms only).

    """
    sigma = state[0]
    R10, R11, R20, R21 = state[1], state[2], state[3], state[4]
    lam1_0, lam1_1, lam2_0, lam2_1 = theta[0], theta[1], theta[2], theta[3]
    beta_0, beta_1, beta_2 = theta[4], theta[5], theta[6]
    theta_1, theta_2 = theta[7], theta[8]

    d1_0 = np.exp(-lam1_0 * dt_years)
    d1_1 = np.exp(-lam1_1 * dt_years)
    d2_0 = np.exp(-lam2_0 * dt_years)
    d2_1 = np.exp(-lam2_1 * dt_years)

    # Drift-only update for R_{1,j} (diffusion handled by Q).
    R10_new = d1_0 * R10
    R11_new = d1_1 * R11
    # R_{2,j}: simulator uses sigma^2*dt (NOT observed r^2*1). drift = (1-exp)*sigma^2.
    sigma2 = sigma * sigma
    R20_new = d2_0 * (R20 + lam2_0 * sigma2 * dt_years)
    R21_new = d2_1 * (R21 + lam2_1 * sigma2 * dt_years)

    lam1_bar = (1.0 - theta_1) * lam1_0 + theta_1 * lam1_1
    lam2_bar = (1.0 - theta_2) * lam2_0 + theta_2 * lam2_1
    R1_bar = ((1.0 - theta_1) * lam1_0 * R10_new + theta_1 * lam1_1 * R11_new) / lam1_bar
    R2_bar = ((1.0 - theta_2) * lam2_0 * R20_new + theta_2 * lam2_1 * R21_new) / lam2_bar

    sqrt_R2 = np.sqrt(max(R2_bar, EPS_R2))
    sigma_new = beta_0 + beta_1 * R1_bar + beta_2 * sqrt_R2
    return np.array([sigma_new, R10_new, R11_new, R20_new, R21_new])


def _m4_noise_vector(
    state: np.ndarray,
    dt_years: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Rank-1 noise-propagation vector ``g`` at the prior mean state.
    """
    sigma = state[0]
    lam1_0, lam1_1 = theta[0], theta[1]
    beta_1 = theta[5]
    theta_1 = theta[7]
    d1_0 = np.exp(-lam1_0 * dt_years)
    d1_1 = np.exp(-lam1_1 * dt_years)

    sqrt_dt = np.sqrt(max(dt_years, 0.0))
    g_R10 = d1_0 * lam1_0 * sigma * sqrt_dt
    g_R11 = d1_1 * lam1_1 * sigma * sqrt_dt

    # sigma' = beta_0 + beta_1*R_bar_1' + beta_2*sqrt R_bar_2'. Noise enters sigma only through beta_1*R_bar_1'.
    # R_bar_1' = [(1-theta_1) lambda_{1,0} R'_{1,0} + theta_1 lambda_{1,1} R'_{1,1}] / lambda_bar_1, so the
    # Delta W-coefficient on R_bar_1' is the same weighted combination of g_R10, g_R11.
    lam1_bar = (1.0 - theta_1) * lam1_0 + theta_1 * lam1_1
    g_R1bar = ((1.0 - theta_1) * lam1_0 * g_R10 + theta_1 * lam1_1 * g_R11) / lam1_bar
    g_sigma = beta_1 * g_R1bar

    return np.array([g_sigma, g_R10, g_R11, 0.0, 0.0])


def _m4_warmup_R(
    daily_sigma_series: np.ndarray,
    theta: np.ndarray,
    n_warmup_daily: int,
) -> tuple[float, float, float, float]:
    """Bootstrap (R_{1,0}, R_{1,1}, R_{2,0}, R_{2,1}) over GUYON_MAX_DELTA DAILY bars.

    """
    DT_DAILY_YEARS = float(DAY_STEPS) / MINUTES_PER_YEAR  # = 1/252
    R10 = R11 = R20 = R21 = 0.0
    lam1_0, lam1_1, lam2_0, lam2_1 = theta[0], theta[1], theta[2], theta[3]
    d1_0 = np.exp(-lam1_0 * DT_DAILY_YEARS)
    d1_1 = np.exp(-lam1_1 * DT_DAILY_YEARS)
    d2_0 = np.exp(-lam2_0 * DT_DAILY_YEARS)
    d2_1 = np.exp(-lam2_1 * DT_DAILY_YEARS)
    for k in range(n_warmup_daily):
        sigma_k = float(daily_sigma_series[k])
        sigma2_dt = sigma_k * sigma_k * DT_DAILY_YEARS
        R10 = d1_0 * R10
        R11 = d1_1 * R11
        R20 = d2_0 * (R20 + lam2_0 * sigma2_dt)
        R21 = d2_1 * (R21 + lam2_1 * sigma2_dt)
    return R10, R11, R20, R21


def fit_m4(
    daily_arrays: dict[str, pd.DataFrame],
    fast_arrays: dict[str, pd.DataFrame],
    fast_dt_array: np.ndarray,
    path_id: int,
) -> FitResult:
    """M4: Theta-fixed UKF on the 4F-PDV model.
    """
    # ---- Stage 1: M2 on daily tape ----
    n_daily = len(daily_arrays["sigma"])
    daily_dt_array = np.full(n_daily, float(DAY_STEPS))
    stage1 = fit_m2(daily_arrays, daily_dt_array, path_id)
    theta = stage1.theta_hat

    # ---- Stage 2: UKF setup ----
    sigma_obs = fast_arrays["sigma"].iloc[:, path_id].to_numpy(dtype=float)
    n_fast = len(sigma_obs)
    dt_years_per_bar = np.asarray(fast_dt_array, dtype=float) / MINUTES_PER_YEAR
    assert dt_years_per_bar.shape == (
        n_fast,
    ), f"fast_dt_array length ({len(fast_dt_array)}) != fast tape length ({n_fast})"

    daily_sigma = daily_arrays["sigma"].iloc[:, path_id].to_numpy(dtype=float)
    n_warmup_daily = min(GUYON_MAX_DELTA, n_daily)
    R10_0, R11_0, R20_0, R21_0 = _m4_warmup_R(daily_sigma, theta, n_warmup_daily)
    n_warmup = int(round(n_warmup_daily * n_fast / n_daily))
    n_warmup = max(1, min(n_warmup, n_fast - 1))
    # Use the observed sigma at the warmup boundary as our sigma_0.
    sigma_0 = float(sigma_obs[n_warmup])
    x = np.array([sigma_0, R10_0, R11_0, R20_0, R21_0])

    # Initial covariance - diagonal per design memo.
    P = np.diag([1e-3, 1e-4, 1e-4, 1e-5, 1e-5])

    # Measurement noise: empirical from Stage 1 train RMSE, clipped to 1e-6.
    train_rmse = float(stage1.diagnostics.get("train_rmse", 1e-3))
    R_obs_scalar = max(train_rmse * train_rmse, 1e-6)
    R_obs = np.array([[R_obs_scalar]])

    # Tiny diagonal regularizer (distinct from the state-dependent Q).
    Q_reg = np.diag([1e-10, 1e-12, 1e-12, 1e-12, 1e-12])

    # Filter loop over bars n_warmup .. n_fast-1.
    n_filter = n_fast - n_warmup
    state_trace = np.empty((n_filter, 5))
    predicted_sigma = np.empty(n_filter)
    innovation_seq = np.empty(n_filter)
    nis_seq = np.empty(n_filter)
    log_likelihood = 0.0
    stats = None  # KalmanStats; allocated by first call

    # Import locally to avoid circular import risk if ukf gets richer later.
    from code_section7.engine.ukf import KalmanStats, predict_step, update_step  # noqa: PLC0415

    stats = KalmanStats()

    t0 = time.perf_counter()
    for k in range(n_filter):
        bar_idx = n_warmup + k
        dt_year = float(dt_years_per_bar[bar_idx])

        # Predict: closure over (dt_year, theta).
        def fx(state, _dt=dt_year, _theta=theta):
            return _m4_propagate_step(state, _dt, _theta)

        g_vec = _m4_noise_vector(x, dt_year, theta)
        Q_state_dep = np.outer(g_vec, g_vec)

        x_pred, P_pred, stats = predict_step(
            x,
            P,
            fx,
            Q_state_dep,
            Q_reg,
            alpha=_UKF_ALPHA,
            beta=_UKF_BETA,
            kappa=_UKF_KAPPA,
            stats=stats,
        )
        predicted_sigma[k] = x_pred[0]

        # Update: linear h(x) = sigma.
        def hx(state):
            return np.array([state[0]])

        z = np.array([float(sigma_obs[bar_idx])])
        x, P, info, stats = update_step(
            x_pred,
            P_pred,
            hx,
            z,
            R_obs,
            alpha=_UKF_ALPHA,
            beta=_UKF_BETA,
            kappa=_UKF_KAPPA,
            stats=stats,
        )
        # Clamp R-rows non-negative (state-physical, like Guyon's EPS_R2 floor).
        if x[3] < 0:
            x[3] = 0.0
        if x[4] < 0:
            x[4] = 0.0

        state_trace[k] = x
        innovation_seq[k] = float(info["y"][0])
        nis_seq[k] = info["nis"]
        # Gaussian log-likelihood contribution: -0.5 (log(2pi*S) + y^2/S)
        S_scalar = float(info["S"][0, 0])
        if S_scalar > 0:
            log_likelihood += -0.5 * (np.log(2 * np.pi * S_scalar) + (info["y"][0] ** 2) / S_scalar)
    elapsed = time.perf_counter() - t0


    test_size = 50
    if n_filter > test_size:
        sigma_true_test = sigma_obs[n_fast - test_size :]
        sigma_pred_test = predicted_sigma[-test_size:]
        sigma_post_test = state_trace[-test_size:, 0]
        test_r2, test_rmse = _r2_and_rmse(sigma_true_test, sigma_pred_test)
        test_r2_filtered, test_rmse_filtered = _r2_and_rmse(sigma_true_test, sigma_post_test)
    else:
        test_r2 = float("nan")
        test_rmse = float("nan")
        test_r2_filtered = float("nan")
        test_rmse_filtered = float("nan")

    fast_dt_summary = {
        "median": float(np.median(fast_dt_array)),
        "max": float(np.max(fast_dt_array)),
        "n": int(len(fast_dt_array)),
    }

    return FitResult(
        theta_hat=stage1.theta_hat.copy(),
        fit_time_sec=stage1.fit_time_sec + elapsed,
        method_name="M4",
        diagnostics={
            "stage1_theta_hat": stage1.theta_hat.copy(),
            "stage1_diagnostics": stage1.diagnostics,
            "state_trace": state_trace,
            "predicted_sigma": predicted_sigma,
            "innovation_seq": innovation_seq,
            "nis_seq": nis_seq,
            "log_likelihood": float(log_likelihood),
            "cholesky_failures": int(stats.cholesky_failures),
            "test_r2": float(test_r2),
            "test_rmse": float(test_rmse),
            "test_r2_filtered": float(test_r2_filtered),
            "test_rmse_filtered": float(test_rmse_filtered),
            "fast_dt_summary": fast_dt_summary,
            "n_warmup": int(n_warmup),
            "n_warmup_daily": int(n_warmup_daily),
            "stage1_fit_time_sec": stage1.fit_time_sec,
            "stage2_fit_time_sec": float(elapsed),
            "R_obs_scalar": float(R_obs_scalar),
        },
    )
