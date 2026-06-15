"""Calibrate the 4-factor 2EXP PDV model on pre-2022 daily data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ._paths import OUTPUTS_ROOT
from .data import load_daily_yf


from empirical_study.empirical_study_2exp import find_optimal_parameters_exp 

CACHE_FILE = OUTPUTS_ROOT / "_cache" / "intraday_vix_fit_params_v2.json"


def _extract_params_from_solution(sol: dict) -> dict:
    """Pull the 9 calibrated parameters out of `find_optimal_parameters_exp`'s sol dict."""
    x = np.asarray(sol["sol"]["x"], dtype=float)
    return {
        "beta_0": float(x[0]),
        "beta_1": float(x[1]),
        "beta_2": float(x[2]),
        "theta_1": float(x[3]),
        "theta_2": float(x[4]),
        "lam_1_0": float(x[5]),
        "lam_1_1": float(x[6]),
        "lam_2_0": float(x[7]),
        "lam_2_1": float(x[8]),
        "train_rmse": float(sol["train_rmse"]),
        "test_rmse": float(sol["test_rmse"]),
        "train_r2": float(sol["train_r2"]),
        "test_r2": float(sol["test_r2"]),
    }


def calibrate_2exp(
    train_start: str = "1990-01-01",
    train_end: str = "2023-07-18",
    test_end: str = "2025-09-16",
    max_delta: int = 1000,
) -> dict:
    """Fit the 2EXP regression on daily SPX/VIX (yfinance) for the train window."""
    spx = load_daily_yf("^GSPC", train_start, test_end)["Close"]
    vix = load_daily_yf("^VIX", train_start, test_end)["Close"] / 100.0
    test_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sol = find_optimal_parameters_exp(
        vol=vix,
        index=spx,
        p=1,
        setting=[(1, 1), (2, 1 / 2)],
        train_start_date=pd.to_datetime(train_start),
        test_start_date=pd.to_datetime(test_start),
        test_end_date=pd.to_datetime(test_end),
        max_delta=max_delta,
    )
    return _extract_params_from_solution(sol)


def load_or_calibrate(refresh: bool = False, **kwargs) -> dict:
    """Read cached params or run calibration once and cache the result."""
    if CACHE_FILE.exists() and not refresh:
        return json.loads(CACHE_FILE.read_text())
    params = calibrate_2exp(**kwargs)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(params, indent=2))
    return params


def init_R_state(
    daily_returns: pd.Series,
    asof: pd.Timestamp,
    params: dict,
    lookback_years: int = 4,
) -> dict:
    """Closed-form Markov initialization of the 4 R-factors at `asof` (start-of-day)."""
    cutoff = asof - pd.Timedelta(days=int(lookback_years * 365.25))
    r = daily_returns.loc[(daily_returns.index >= cutoff) & (daily_returns.index < asof)].to_numpy()
    n = len(r)
    if n == 0:
        raise ValueError(f"no daily returns available before {asof}")
    dt = 1.0 / 252.0
    ages = np.arange(n - 1, -1, -1) * dt  
    state = {}
    for n_idx, lam_keys in [(1, ("lam_1_0", "lam_1_1")), (2, ("lam_2_0", "lam_2_1"))]:
        x = r if n_idx == 1 else r**2
        for j_idx, lam_key in enumerate(lam_keys):
            lam = params[lam_key]
            kernel = lam * np.exp(-lam * ages)
            state[f"R_{n_idx}_{j_idx}"] = float(np.sum(kernel * x))
    return state


def aggregates(state: dict, params: dict) -> dict:
    """Compute R_bar_n, lambda_n_bar, R_n from the 4 sub-factors."""
    th1, th2 = params["theta_1"], params["theta_2"]
    l10, l11 = params["lam_1_0"], params["lam_1_1"]
    l20, l21 = params["lam_2_0"], params["lam_2_1"]
    R1 = (1 - th1) * state["R_1_0"] + th1 * state["R_1_1"]
    R2 = (1 - th2) * state["R_2_0"] + th2 * state["R_2_1"]
    lam1_bar = (1 - th1) * l10 + th1 * l11
    lam2_bar = (1 - th2) * l20 + th2 * l21
    R1_bar = ((1 - th1) * l10 * state["R_1_0"] + th1 * l11 * state["R_1_1"]) / lam1_bar
    R2_bar = ((1 - th2) * l20 * state["R_2_0"] + th2 * l21 * state["R_2_1"]) / lam2_bar
    return {
        "R1": R1,
        "R2": R2,
        "lam1_bar": lam1_bar,
        "lam2_bar": lam2_bar,
        "R1_bar": R1_bar,
        "R2_bar": R2_bar,
    }
