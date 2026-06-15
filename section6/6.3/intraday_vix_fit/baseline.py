"""
Standard 2EXP fit - sigma_t = beta_0 + beta_1 R_1 + beta_2 sqrt.R_2 
"""

from __future__ import annotations

import math


def run_baseline_day(state: dict, params: dict) -> float:
    """sigma from the regression evaluated at the given R-state.

    For Guyon's contemporaneous fit, pass an end-of-day R-state (today's daily
    return included).
    """
    R1 = (1 - params["theta_1"]) * state["R_1_0"] + params["theta_1"] * state["R_1_1"]
    R2 = (1 - params["theta_2"]) * state["R_2_0"] + params["theta_2"] * state["R_2_1"]
    return params["beta_0"] + params["beta_1"] * R1 + params["beta_2"] * math.sqrt(max(R2, 0.0))
