"""
Data loaders - daily SPX/VIX/SX5E/NKY from yfinance, hourly SPX from FirstRateData, frozen VIX session anchors from CSV.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from ._paths import DATA_ROOT, OUTPUTS_ROOT

RTH_HOURS = (10, 11, 12, 13, 14, 15, 16)
SPX_HOURLY_FILE = DATA_ROOT / "SPX" / "SPX_main" / "SPX_full_1hour.txt"
SESSION_ANCHORS_FILE = OUTPUTS_ROOT / "_results" / "intraday_vix_fit" / "vix_session_anchors.csv"


def load_daily_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLC from yfinance, naive date index."""
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.DatetimeIndex(df.index.date)
    return df


def load_hourly_spx_local() -> pd.DataFrame:
    """Local FirstRateData SPX hourly, filtered to 10:00-16:00 ET."""
    df = pd.read_csv(
        SPX_HOURLY_FILE,
        header=None,
        names=["DateTime", "Open", "High", "Low", "Close"],
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize("America/New_York")
    df = df.set_index("DateTime")
    return df[df.index.hour.isin(RTH_HOURS)].copy()


def load_hourly_vix_yf(period: str = "730d") -> pd.DataFrame:
    """Yahoo `^VIX` 1h, tz-converted to ET, filtered to 10:00-16:00."""
    df = yf.Ticker("^VIX").history(period=period, interval="1h", auto_adjust=False)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = df.index.tz_convert("America/New_York")
    return df[df.index.hour.isin(RTH_HOURS)].copy()


def load_session_anchors_csv(path: Path = SESSION_ANCHORS_FILE) -> pd.DataFrame:
    """Frozen 538-session VIX open/close anchors at 10:00 / 16:00 ET (2023-07-19 -> 2025-09-16).
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


def load_daily_foreign_aligned(
    ticker: str,
    start: str,
    end: str,
    target_calendar: pd.DatetimeIndex,
) -> pd.Series:
    """
    Daily foreign-index close, forward-filled and aligned to ``target_calendar``.
    """
    df = load_daily_yf(ticker, start, end)
    return df["Close"].reindex(target_calendar).ffill()


def session_bars(
    date: _dt.date,
    spx_h: pd.DataFrame,
    vix_h: pd.DataFrame,
) -> Optional[dict]:
    """
    Return SPX closes + VIX open/close for a given session, or None if any bar is missing.
    """
    spx_day = spx_h[spx_h.index.date == date]
    vix_day = vix_h[vix_h.index.date == date]
    if len(spx_day) != 7 or len(vix_day) != 7:
        return None
    spx_hours = sorted(spx_day.index.hour.tolist())
    vix_hours = sorted(vix_day.index.hour.tolist())
    if spx_hours != list(RTH_HOURS) or vix_hours != list(RTH_HOURS):
        return None
    return {
        "date": date,
        "spx_closes": spx_day.sort_index()["Close"].to_numpy(),  # length 7
        "vix_open": float(vix_day.sort_index().iloc[0]["Open"]),
        "vix_close": float(vix_day.sort_index().iloc[-1]["Close"]),
    }


def session_bars_from_csv(
    date: _dt.date,
    spx_h: pd.DataFrame,
    anchors_row: pd.Series,
) -> Optional[dict]:
    """
    Same as :func:`session_bars` but with VIX open/close sourced from the cached CSV.
    """
    if anchors_row["status"] != "ok":
        return None
    spx_day = spx_h[spx_h.index.date == date]
    if len(spx_day) != 7:
        return None
    spx_hours = sorted(spx_day.index.hour.tolist())
    if spx_hours != list(RTH_HOURS):
        return None
    return {
        "date": date,
        "spx_closes": spx_day.sort_index()["Close"].to_numpy(),
        "vix_open": float(anchors_row["vix_open"]),
        "vix_close": float(anchors_row["vix_close"]),
    }
