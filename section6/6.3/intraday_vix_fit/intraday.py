"""Per-day Euler integration of eq 53 with realized hourly returns."""

from __future__ import annotations

import math

import numpy as np

from .calibration import aggregates


def run_intraday_day(
    state: dict,
    sigma_open: float,
    returns: np.ndarray,
    params: dict,
    dt_per_step: float | None = None,
) -> dict:
    """Euler-integrate eq 53 over a sequence of substeps with realized returns.
    dict with keys:
        sigma_close - sigma after the last substep
        sigma_path - np.ndarray of length len(returns)+1
        state_close - final R-state
        neg_count - number of substeps where sigma went < 0
    """
    n_steps = len(returns)
    if n_steps == 0:
        raise ValueError("returns must be non-empty")
    dt = dt_per_step if dt_per_step is not None else 1.0 / (252.0 * n_steps)
    sigma = float(sigma_open)
    sigma_path = [sigma]
    neg_count = 0
    for r in returns:
        agg = aggregates(state, params)
        sqrt_R2 = math.sqrt(max(agg["R2"], 1e-12))
        # eq 53 - drift uses current sigma^2, R_bar_1, R_bar_2 / sqrt R_2
        drift = (
            -params["beta_1"] * agg["lam1_bar"] * agg["R1_bar"]
            + 0.5 * params["beta_2"] * agg["lam2_bar"] * (sigma**2 - agg["R2_bar"]) / sqrt_R2
        ) * dt
        # diffusion: beta_1 lambda_bar_1 sigma_t dW_t with sigma_t dW_t = r (realized substitution)
        diffusion = params["beta_1"] * agg["lam1_bar"] * float(r)
        sigma = sigma + drift + diffusion
        if sigma < 0:
            neg_count += 1
        sigma_path.append(sigma)
        # Markov R_n update with realized r and r^2 (matches option_pricing_4.ipynb)
        for j in (0, 1):
            lam1 = params[f"lam_1_{j}"]
            lam2 = params[f"lam_2_{j}"]
            state[f"R_1_{j}"] = math.exp(-lam1 * dt) * (state[f"R_1_{j}"] + lam1 * float(r))
            state[f"R_2_{j}"] = math.exp(-lam2 * dt) * (state[f"R_2_{j}"] + lam2 * float(r) ** 2)
    return {
        "sigma_close": sigma,
        "sigma_path": np.asarray(sigma_path),
        "state_close": state,
        "neg_count": neg_count,
    }
