"""Multi-asset Guyon extension (Section 6.3 model 5)."""

from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd

from ._paths import OUTPUTS_ROOT
from .calibration import init_R_state
from .data import load_daily_foreign_aligned, load_daily_yf

CACHE_FILE = OUTPUTS_ROOT / "_cache" / "intraday_vix_fit_multiasset_v2.json"
FOREIGN_TICKERS = {"sx5e": "^STOXX50E", "nky": "^N225"}


def _state_to_features(state: dict, params: dict) -> tuple[float, float]:
    """Convert a 4-factor R-state into (R_1, sqrt(R_2)) under (theta_1, theta_2) from ``params``."""
    R1 = (1 - params["theta_1"]) * state["R_1_0"] + params["theta_1"] * state["R_1_1"]
    R2 = (1 - params["theta_2"]) * state["R_2_0"] + params["theta_2"] * state["R_2_1"]
    return R1, float(np.sqrt(max(R2, 0.0)))


def _features_for_dates(
    daily_returns: pd.Series, params: dict, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """End-of-day inclusive (R_1, sqrt(R_2)) for each date in ``dates``."""
    rows = []
    for d in dates:
        try:
            state = init_R_state(daily_returns, d + pd.Timedelta(days=1), params)
            R1, sqrt_R2 = _state_to_features(state, params)
        except ValueError:
            R1, sqrt_R2 = float("nan"), float("nan")
        rows.append((R1, sqrt_R2))
    arr = np.asarray(rows, dtype=float)
    return pd.DataFrame({"R_1": arr[:, 0], "sqrt_R_2": arr[:, 1]}, index=dates)


def prepare_foreign_returns(
    start: str, end: str, spx_calendar: pd.DatetimeIndex
) -> Dict[str, pd.Series]:
    """Load + align foreign daily returns to the SPX trading calendar (forward-filled)."""
    out: Dict[str, pd.Series] = {}
    for key, ticker in FOREIGN_TICKERS.items():
        aligned = load_daily_foreign_aligned(ticker, start, end, target_calendar=spx_calendar)
        out[key] = aligned.pct_change().dropna()
    return out


def fit_multiasset(
    spx_params: dict,
    train_start: str = "1990-01-01",
    train_end: str = "2023-07-18",
) -> dict:
    """OLS-fit foreign beta_3..beta_6 on the residual (VIX_t - sigma_t^{SPX-only})."""
    spx = load_daily_yf("^GSPC", "1989-01-01", train_end)
    vix = load_daily_yf("^VIX", train_start, train_end)
    spx_returns = spx["Close"].pct_change().dropna()
    target = vix["Close"] / 100.0

    train_dates = target.index[(target.index >= train_start) & (target.index <= train_end)]

    spx_feats = _features_for_dates(spx_returns, spx_params, train_dates)
    sigma_spx_only = (
        spx_params["beta_0"]
        + spx_params["beta_1"] * spx_feats["R_1"]
        + spx_params["beta_2"] * spx_feats["sqrt_R_2"]
    )

    foreign_returns = prepare_foreign_returns(
        "1989-01-01", train_end, spx_calendar=spx_returns.index
    )
    feat_cols: dict[str, pd.Series] = {}
    for key in FOREIGN_TICKERS:
        feats = _features_for_dates(foreign_returns[key], spx_params, train_dates)
        feat_cols[f"{key}_R_1"] = feats["R_1"]
        feat_cols[f"{key}_sqrt_R_2"] = feats["sqrt_R_2"]

    aligned = pd.concat(
        [
            target.rename("target"),
            sigma_spx_only.rename("sigma_spx_only"),
            pd.DataFrame(feat_cols),
        ],
        axis=1,
    ).dropna()
    y = (aligned["target"] - aligned["sigma_spx_only"]).values
    X = aligned[["sx5e_R_1", "sx5e_sqrt_R_2", "nky_R_1", "nky_sqrt_R_2"]].values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

    return {
        "beta_3_sx5e_R_1": float(coeffs[0]),
        "beta_4_sx5e_sqrt_R_2": float(coeffs[1]),
        "beta_5_nky_R_1": float(coeffs[2]),
        "beta_6_nky_sqrt_R_2": float(coeffs[3]),
        "n_train": int(len(y)),
        "effective_start": str(aligned.index.min().date()),
        "effective_end": str(aligned.index.max().date()),
        "train_start": str(train_start),
        "train_end": str(train_end),
    }


def load_or_fit_multiasset(spx_params: dict, refresh: bool = False, **kwargs) -> dict:
    """Read cached foreign betas, or run :func:`fit_multiasset` once and cache."""
    if CACHE_FILE.exists() and not refresh:
        return json.loads(CACHE_FILE.read_text())
    result = fit_multiasset(spx_params, **kwargs)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(result, indent=2))
    return result


def predict_multiasset_day(
    asof: pd.Timestamp,
    sigma_spx_only: float,
    foreign_returns: Dict[str, pd.Series],
    spx_params: dict,
    multi_params: dict,
) -> float:
    """sigma_hat^{m5}_t = sigma_hat^{SPX-only}_t + sum_f (beta_R1^f * R_1^f + beta_sqrtR2^f * sqrt.R_2^f)."""
    asof_inclusive = asof + pd.Timedelta(days=1)
    sigma = float(sigma_spx_only)
    for key, returns in foreign_returns.items():
        try:
            state = init_R_state(returns, asof_inclusive, spx_params)
        except ValueError:
            return float("nan")
        R1, sqrt_R2 = _state_to_features(state, spx_params)
        if key == "sx5e":
            sigma += multi_params["beta_3_sx5e_R_1"] * R1
            sigma += multi_params["beta_4_sx5e_sqrt_R_2"] * sqrt_R2
        else:
            sigma += multi_params["beta_5_nky_R_1"] * R1
            sigma += multi_params["beta_6_nky_sqrt_R_2"] * sqrt_R2
    return sigma
