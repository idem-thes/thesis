"""Regression targets for Section 8: realized Delta VX over a horizon, plus the VX market frame.


"""

from __future__ import annotations

import numpy as np
import pandas as pd

from code_section7.lob.features import _asof, _ns


def _mid(events: pd.DataFrame) -> np.ndarray:
    return (events["bid_px_00"].to_numpy(float) + events["ask_px_00"].to_numpy(float)) / 2.0


def build_market_frame(vx_events: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """vx_mid / vx_bid / vx_ask as-of the last VX event <= each grid timestamp."""
    ts = _ns(vx_events["ts_event"])
    g = _ns(grid)
    bid = _asof(ts, vx_events["bid_px_00"].to_numpy(float), g)
    ask = _asof(ts, vx_events["ask_px_00"].to_numpy(float), g)
    return pd.DataFrame({"vx_mid": (bid + ask) / 2.0, "vx_bid": bid, "vx_ask": ask}, index=grid)


def realized_delta_vx(vx_events: pd.DataFrame, grid: pd.DatetimeIndex, horizon_s: int) -> pd.Series:
    """Delta VX over `horizon_s`: mid as-of (t+h) minus mid as-of t, for each grid t."""
    ts = _ns(vx_events["ts_event"])
    mid_ev = _mid(vx_events)
    g = _ns(grid)
    h = np.timedelta64(int(horizon_s), "s").astype("timedelta64[ns]")
    return pd.Series(_asof(ts, mid_ev, g + h) - _asof(ts, mid_ev, g), index=grid)
