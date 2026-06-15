"""Exact-OU R_{p,j} propagation for the PDV 4-FPDV model.

"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParamSet:
    beta0: float
    beta1: float
    beta2: float
    lam10: float
    lam11: float
    theta1: float
    lam20: float
    lam21: float
    theta2: float

    def asarray(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "ParamSet":
        return cls(**{f.name: float(v) for f, v in zip(fields(cls), x.tolist())})


@dataclass(frozen=True)
class State:
    R10: float
    R11: float
    R20: float
    R21: float


@dataclass(frozen=True)
class BarQuantities:
    lam1: float
    lam2: float
    R1: float
    R2: float
    R1_nobar: float
    R2_nobar: float


def bar_quantities(state: State, theta: ParamSet) -> BarQuantities:
    """Weighted-average lam_bar_n, R_bar_n per thesis eq 53.
    """
    w10, w11 = 1.0 - theta.theta1, theta.theta1
    w20, w21 = 1.0 - theta.theta2, theta.theta2
    lam1 = w10 * theta.lam10 + w11 * theta.lam11
    lam2 = w20 * theta.lam20 + w21 * theta.lam21
    R1 = (w10 * theta.lam10 * state.R10 + w11 * theta.lam11 * state.R11) / lam1
    R2 = (w20 * theta.lam20 * state.R20 + w21 * theta.lam21 * state.R21) / lam2
    R1_nobar = w10 * state.R10 + w11 * state.R11
    R2_nobar = w20 * state.R20 + w21 * state.R21
    return BarQuantities(lam1=lam1, lam2=lam2, R1=R1, R2=R2, R1_nobar=R1_nobar, R2_nobar=R2_nobar)


def init_state(returns: pd.Series, dt: float, theta: ParamSet) -> State:
    """Initialize R_{p,j} from prior returns by exact-OU warmup rollup.

    """
    r = np.asarray(returns.dropna().to_numpy(), dtype=float).ravel()
    R10 = R11 = R20 = R21 = 0.0
    d10 = np.exp(-theta.lam10 * dt)
    d11 = np.exp(-theta.lam11 * dt)
    d20 = np.exp(-theta.lam20 * dt)
    d21 = np.exp(-theta.lam21 * dt)
    for ri in r:
        R10 = d10 * R10 + (1.0 - d10) * ri
        R11 = d11 * R11 + (1.0 - d11) * ri
        R20 = d20 * R20 + (1.0 - d20) * ri * ri
        R21 = d21 * R21 + (1.0 - d21) * ri * ri
    return State(R10=R10, R11=R11, R20=R20, R21=R21)


def propagate_state(state: State, next_return: float, dt: float, theta: ParamSet) -> State:
    """One exact-OU step of R_{p,j} given the next realized return."""
    next_return = float(next_return)
    d10 = np.exp(-theta.lam10 * dt)
    d11 = np.exp(-theta.lam11 * dt)
    d20 = np.exp(-theta.lam20 * dt)
    d21 = np.exp(-theta.lam21 * dt)
    r2 = next_return * next_return
    return State(
        R10=d10 * state.R10 + (1.0 - d10) * next_return,
        R11=d11 * state.R11 + (1.0 - d11) * next_return,
        R20=d20 * state.R20 + (1.0 - d20) * r2,
        R21=d21 * state.R21 + (1.0 - d21) * r2,
    )


def propagate_trajectory(
    returns: pd.Series, dt: float, theta: ParamSet, *, s0: State | None = None
) -> pd.DataFrame:
    """Minute-by-minute propagation across a Series; returns DataFrame of R_{p,j} indexed by returns.index.

    """
    r = returns.to_numpy()
    n = len(r)
    if s0 is None:
        s0 = State(R10=0.0, R11=0.0, R20=0.0, R21=0.0)
    R10 = np.empty(n)
    R11 = np.empty(n)
    R20 = np.empty(n)
    R21 = np.empty(n)
    d10 = np.exp(-theta.lam10 * dt)
    d11 = np.exp(-theta.lam11 * dt)
    d20 = np.exp(-theta.lam20 * dt)
    d21 = np.exp(-theta.lam21 * dt)
    R10[0], R11[0], R20[0], R21[0] = s0.R10, s0.R11, s0.R20, s0.R21
    for i in range(1, n):
        ri = r[i]
        r2 = ri * ri
        R10[i] = d10 * R10[i - 1] + (1.0 - d10) * ri
        R11[i] = d11 * R11[i - 1] + (1.0 - d11) * ri
        R20[i] = d20 * R20[i - 1] + (1.0 - d20) * r2
        R21[i] = d21 * R21[i - 1] + (1.0 - d21) * r2
    return pd.DataFrame({"R10": R10, "R11": R11, "R20": R20, "R21": R21}, index=returns.index)
