"""Section 7.2 trade lifecycle: signal - entry - exit - Trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from code_section7.state import ParamSet, State
from code_section7.backtest.forecast import (
    M4Filter,
    forecast_m2,
    forecast_m4,
    theta_hat_to_param_set,
)


_MINUTES_PER_YEAR = 252 * 1440  # 362_880
_DT_MIN_YEARS = 1.0 / _MINUTES_PER_YEAR  # fallback used only when prev_ts is unknown
_VX_MULTIPLIER = 1000.0  # $ per VX point per contract
_CFE_FEE_ROUND_TRIP = 2.20  # $ per contract per round-trip ($1.10 * 2 sides)

Direction = Literal["long", "short"]


@dataclass
class Trade:
    """One round-trip trade record."""

    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp  # scheduled exit; actual close ts may differ slightly
    direction: Direction
    entry_px: float
    forecast_vx_pts: float = float("nan")
    exit_px: float | None = None
    pnl_pts: float | None = None
    pnl_dollars: float | None = None
    hyperparams: dict = field(default_factory=dict)


def decide_trade(
    forecast_vx: float,
    bid: float,
    ask: float,
    x_threshold: float,
) -> tuple[Direction | None, float | None]:
    """Return ('long', ask) | ('short', bid) | (None, None).

    A trade is opened iff ``|forecast_vx| > x_threshold * (ask - bid)``.
    Non-positive spread (crossed or stale book) -> no trade.
    """
    spread = ask - bid
    if spread <= 0 or not np.isfinite(spread):
        return None, None
    threshold = x_threshold * spread
    if forecast_vx > threshold:
        return "long", ask
    if forecast_vx < -threshold:
        return "short", bid
    return None, None


def close_trade(
    direction: Direction,
    entry_px: float,
    bid: float,
    ask: float,
) -> tuple[float, float]:
    """Compute (exit_px, pnl_pts) by crossing the opposite spread side."""
    if direction == "long":
        exit_px = float(bid)
        pnl_pts = exit_px - entry_px  # entered at ask, exit at bid
    elif direction == "short":
        exit_px = float(ask)
        pnl_pts = entry_px - exit_px  # entered at bid, exit at ask
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    return exit_px, pnl_pts


def _propagate_state_guyon(
    state: State,
    dt_years: float,
    theta: ParamSet,
    realized_return: float,
) -> State:
    """Exact-OU eq 50/51 step driven by realized return r_t (Guyon p. 1240 eq 26).
    """
    d10 = np.exp(-theta.lam10 * dt_years)
    d11 = np.exp(-theta.lam11 * dt_years)
    d20 = np.exp(-theta.lam20 * dt_years)
    d21 = np.exp(-theta.lam21 * dt_years)
    r_sq = realized_return * realized_return
    return State(
        R10=d10 * (state.R10 + theta.lam10 * realized_return),
        R11=d11 * (state.R11 + theta.lam11 * realized_return),
        R20=d20 * (state.R20 + theta.lam20 * r_sq),
        R21=d21 * (state.R21 + theta.lam21 * r_sq),
    )


def initialize_R_kernel_sum(
    past_returns: np.ndarray | pd.Series,
    lam_pair: tuple[float, float],
    dt_years: float,
    transform: str = "identity",
    max_delta: int = 1000,
) -> tuple[float, float]:
    """Guyon's ``initialize_R`` (``torch_montecarlo.py`` lines 18-34) in numpy.
    """
    r = np.asarray(past_returns, dtype=float)
    if hasattr(past_returns, "dropna"):
        r = past_returns.dropna().to_numpy(dtype=float)
    r = r[-max_delta:][::-1]  # most recent first
    if transform == "squared":
        x = r * r
    elif transform == "identity":
        x = r
    else:
        raise ValueError(f"unknown transform: {transform!r}")
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    k = np.arange(n, dtype=float)
    timestamps = k * dt_years
    lam = np.asarray(lam_pair, dtype=float)[:, None]  # shape (2, 1)
    weights = lam * np.exp(-lam * timestamps[None, :])  # shape (2, n)
    R = np.sum(x[None, :] * weights, axis=1)  # shape (2,)
    return float(R[0]), float(R[1])


def initialize_state_guyon(
    past_returns: np.ndarray | pd.Series,
    theta: ParamSet,
    dt_years: float,
    max_delta: int = 1000,
) -> State:
    """Initialize a State via Guyon's kernel-sum on historical returns.
    """
    R10, R11 = initialize_R_kernel_sum(
        past_returns, (theta.lam10, theta.lam11), dt_years, "identity", max_delta
    )
    R20, R21 = initialize_R_kernel_sum(
        past_returns, (theta.lam20, theta.lam21), dt_years, "squared", max_delta
    )
    return State(R10=R10, R11=R11, R20=R20, R21=R21)


def _seed_state_at_steady_R2(sigma_init: float) -> State:
    """Fallback state seed: R_1=0, R_2 ~ sigma_init^2 (skips ~100-trading-day transient).
    """
    sigma2 = float(sigma_init) ** 2
    return State(R10=0.0, R11=0.0, R20=sigma2, R21=sigma2)


def _is_in_trading_window(
    ts: pd.Timestamp,
    start_hour: tuple[int, int],
    close_hour: tuple[int, int],
) -> bool:
    """True if ts.time() is in [start_hour, close_hour] inclusive.
    """
    t = ts.time()
    start = pd.Timestamp(year=2000, month=1, day=1, hour=start_hour[0], minute=start_hour[1]).time()
    close = pd.Timestamp(year=2000, month=1, day=1, hour=close_hour[0], minute=close_hour[1]).time()
    return start <= t <= close


def _is_at_forecast_boundary(
    ts: pd.Timestamp,
    start_hour: tuple[int, int],
    forecast_window_min: int,
) -> bool:
    """True if ts is aligned to start_hour + k*forecast_window_min (k >= 0 integer)."""
    start_minutes = start_hour[0] * 60 + start_hour[1]
    ts_minutes = ts.hour * 60 + ts.minute
    if ts.second != 0 or ts.microsecond != 0:
        return False
    if ts_minutes < start_minutes:
        return False
    return (ts_minutes - start_minutes) % forecast_window_min == 0


def _is_in_settlement_mask(
    ts: pd.Timestamp,
    settlement_dates: pd.DatetimeIndex,
    n_days: int,
) -> bool:
    """True if ts.date() is within +/-n_days **calendar days** of any settlement.
    """
    ts_date = pd.Timestamp(ts.date())
    for settle in settlement_dates:
        if abs((ts_date - pd.Timestamp(settle).normalize()).days) <= n_days:
            return True
    return False


def _init_m4_filter(
    theta_param: ParamSet,
    state: State,
    sigma_init: float,
    R_meas: np.ndarray | None,
) -> M4Filter:
    """Build a fresh M4Filter seeded from a State and an initial sigma."""
    x0 = np.array([sigma_init, state.R10, state.R11, state.R20, state.R21])
    P0 = np.diag([1e-4, 1e-8, 1e-8, 1e-8, 1e-8])
    R = R_meas if R_meas is not None else np.array([[1e-6]])
    return M4Filter(theta_param, x0, P0, R)


def run_strategy(
    theta: dict,
    method: Literal["M2", "M4"],
    data: pd.DataFrame,
    hyperparams: dict,
    settlement_dates: pd.DatetimeIndex,
    *,
    warmup_returns: pd.Series | None = None,
    warmup_dt_years: float = 1.0 / 252.0,
    init_ts: pd.Timestamp | None = None,
    init_sigma_obs: float | None = None,
    R_meas: np.ndarray | None = None,
    exit_policy: Literal["fixed_H", "signal_flip"] = "fixed_H",
) -> list[Trade]:
    """Run the strategy on aligned 1-min ES+VX data; return closed trades.
    """
    if method not in ("M2", "M4"):
        raise ValueError(f"unknown method: {method!r}")

    theta_vec = np.asarray(theta["theta_hat"], dtype=float)
    theta_param = theta_hat_to_param_set(theta_vec)

    first_vx = data["vx_mid"].dropna().iloc[0] if not data["vx_mid"].dropna().empty else np.nan
    if warmup_returns is not None and len(warmup_returns) > 0:
        if init_ts is None or init_sigma_obs is None:
            raise ValueError(
                "warmup_returns requires init_ts and init_sigma_obs so the "
                "first per-bar dt and R_2 drift anchor are well-defined."
            )
        state = initialize_state_guyon(warmup_returns, theta_param, dt_years=warmup_dt_years)
        prev_ts: pd.Timestamp | None = init_ts
        last_sigma_obs = float(init_sigma_obs)
    else:
        if not np.isfinite(first_vx):
            raise ValueError("Need a finite initial vx_mid to seed state; got all-NaN series")
        state = _seed_state_at_steady_R2(first_vx / 100.0)
        prev_ts = None  # legacy path: first bar uses _DT_MIN_YEARS
        last_sigma_obs = float(first_vx) / 100.0

    last_sigma_sq_realized = last_sigma_obs * last_sigma_obs


    filter_obj: M4Filter | None = None
    if method == "M4":
        filter_obj = _init_m4_filter(theta_param, state, last_sigma_obs, R_meas)


    threshold_x = float(hyperparams["threshold_x"])
    forecast_window_min = int(hyperparams["forecast_window_min"])
    start_hour = tuple(hyperparams["start_hour"])
    close_hour = tuple(hyperparams["close_hour"])
    n_mask_days = int(hyperparams["n_mask_days"])

    trades: list[Trade] = []
    open_position: Trade | None = None

    for ts, row in data.iterrows():
        r = row["return_1min"]
        vx_mid = row["vx_mid"]
        vx_bid = row["vx_bid"]
        vx_ask = row["vx_ask"]

        # sigma_obs: prefer the bar's vx_mid; fall back to the most recent observed
        # value (vx_mid is ffilled by the data layer, but a NaN can still appear
        # at the very start of the window before VX comes online).
        if np.isfinite(vx_mid):
            sigma_obs = float(vx_mid) / 100.0
            last_sigma_obs = sigma_obs
        else:
            sigma_obs = last_sigma_obs

        # Actual elapsed dt for this bar (smart propagator - Fix 3).
        if prev_ts is not None:
            dt_minutes = (ts - prev_ts).total_seconds() / 60.0
            dt_years = dt_minutes / _MINUTES_PER_YEAR
        else:
            dt_minutes = 1.0
            dt_years = _DT_MIN_YEARS

        # ---- Step 1: state update (skip if return unusable)
        if pd.notna(r) and np.isfinite(sigma_obs):
            state = _propagate_state_guyon(
                state=state,
                dt_years=dt_years,
                theta=theta_param,
                realized_return=float(r),
            )
            if filter_obj is not None:
                filter_obj.step(float(r), dt_minutes, sigma_obs)

            if dt_years > 0.0:
                last_sigma_sq_realized = float(r) * float(r) / dt_years
        prev_ts = ts

        # Step 2: maybe close open position
        if open_position is not None:
            should_close = False
            if exit_policy == "fixed_H":
                # Original behaviour: close once the scheduled H-minute hold elapses.
                should_close = ts >= open_position.exit_ts
            else:  # signal_flip

                if not _is_in_trading_window(ts, start_hour, close_hour):
                    should_close = True
                elif (
                    _is_at_forecast_boundary(ts, start_hour, forecast_window_min)
                    and pd.notna(r)
                    and np.isfinite(sigma_obs)
                ):
                    if method == "M2":
                        new_forecast = forecast_m2(
                            state, theta_param, last_sigma_sq_realized, forecast_window_min
                        )
                    else:  # M4
                        new_forecast = forecast_m4(filter_obj, forecast_window_min)
                    entry_forecast = open_position.forecast_vx_pts
                    if (
                        np.isfinite(new_forecast)
                        and np.isfinite(entry_forecast)
                        and np.sign(new_forecast) != np.sign(entry_forecast)
                        and new_forecast != 0.0
                    ):
                        should_close = True

            if should_close and np.isfinite(vx_bid) and np.isfinite(vx_ask):
                exit_px, pnl_pts = close_trade(
                    open_position.direction,
                    open_position.entry_px,
                    float(vx_bid),
                    float(vx_ask),
                )
                open_position.exit_ts = ts  # actual close time (may differ from scheduled)
                open_position.exit_px = exit_px
                open_position.pnl_pts = pnl_pts
                open_position.pnl_dollars = pnl_pts * _VX_MULTIPLIER - _CFE_FEE_ROUND_TRIP
                trades.append(open_position)
                open_position = None

        # Step 3: maybe open new position
        can_trade = (
            open_position is None
            and np.isfinite(vx_bid)
            and np.isfinite(vx_ask)
            and np.isfinite(sigma_obs)
            and _is_in_trading_window(ts, start_hour, close_hour)
            and _is_at_forecast_boundary(ts, start_hour, forecast_window_min)
            and not _is_in_settlement_mask(ts, settlement_dates, n_mask_days)
        )
        if can_trade:
            if method == "M2":

                forecast_vx = forecast_m2(
                    state, theta_param, last_sigma_sq_realized, forecast_window_min
                )
            else:  # M4
                forecast_vx = forecast_m4(filter_obj, forecast_window_min)

            direction, entry_px = decide_trade(
                forecast_vx,
                float(vx_bid),
                float(vx_ask),
                threshold_x,
            )
            if direction is not None:
                exit_ts = ts + pd.Timedelta(minutes=forecast_window_min)
                open_position = Trade(
                    entry_ts=ts,
                    exit_ts=exit_ts,
                    direction=direction,
                    entry_px=float(entry_px),
                    forecast_vx_pts=float(forecast_vx),
                    hyperparams=dict(hyperparams),
                )

    return trades
