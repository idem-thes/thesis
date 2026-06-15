"""Assemble the Section 7.2 training matrix: (X, y_bidup, y_askdown, block_ts).


"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from code_section7.backtest.data import (
    _ES_ROLL_BRIDGES,
    load_vx_settlement_dates,
)
from code_section7.backtest.rf_baseline import _trading_days_to_next_settlement
from code_section7.data_loader import load_session_events
from code_section7.lob import features, labels

log = logging.getLogger(__name__)

_TZ = "America/New_York"
_GRID_OPEN = (10, 30)  # emit/trade window opens 10:30 ET (features warm from 09:30)
_GRID_CLOSE = (15, 0)  # emit/trade window closes 15:00 ET
_DEFAULT_CACHE_DIR = Path("outputs/_cache/section7_lob")

# Canonical Section 7.1 full-window PDV warmup (mirrors backtest.grid_run.main).
_WARMUP_END = "2025-02-28"
_WARMUP_N_RETURNS = 1000
_WARMUP_DT_YEARS = 1.0 / 252.0


_PDV_ANCHOR_START = "2025-03-02"


def training_grid(session_date: pd.Timestamp | str) -> pd.DatetimeIndex:
    """1-second ET grid 10:30:00-15:00:00 inclusive for ``session_date``.

    Features warm from 09:30 but we only emit/trade on this 10:30-15:00 window.
    """
    day = pd.Timestamp(session_date).normalize()
    start = day + pd.Timedelta(hours=_GRID_OPEN[0], minutes=_GRID_OPEN[1])
    end = day + pd.Timedelta(hours=_GRID_CLOSE[0], minutes=_GRID_CLOSE[1])
    return pd.date_range(start, end, freq="1s", tz=_TZ)


def trade_grid(t0: pd.Timestamp, t1: pd.Timestamp, delta_s: int) -> pd.DatetimeIndex:
    """Non-overlapping delta-block starts ``t0, t0+delta, ...`` strictly ``< t1``; tz preserved."""
    return pd.date_range(t0, t1, freq=f"{int(delta_s)}s", inclusive="left")


def _es_roll_dates() -> set:
    """ES roll calendar dates from ``_ES_ROLL_BRIDGES`` (last-clean + monday-open).

    Derived from the bridge specs so we never call ``load_rolls`` (which triggers
    a full Databento decode). VX rolls sit on the 3rd-Wed and are already covered
    by the settlement mask.
    """
    dates: set = set()
    for bridge in _ES_ROLL_BRIDGES:
        dates.add(bridge["last_clean_ts"].date())
        dates.add(bridge["monday_open_ts"].date())
    return dates


def apply_calendar_masks(df: pd.DataFrame, n_settle_days: int = 3) -> pd.DataFrame:
    """Drop rows near a VX settlement Wednesday or on an ES roll day; return a copy.

    ``df`` is indexed by tz-aware ET timestamps. A row is dropped if its calendar
    date is within +/-``n_settle_days`` calendar days of any VX cash-settlement
    Wednesday (``backtest.data.load_vx_settlement_dates``) OR equals an ES roll
    day (derived from ``backtest.data._ES_ROLL_BRIDGES`` - no Databento decode).
    Column order is preserved.
    """
    out = df.copy()
    if len(out) == 0:
        return out

    idx = out.index
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    bar_dates = idx_naive.normalize()

    keep = np.ones(len(out), dtype=bool)

    # +/-n_settle_days around each VX cash-settlement Wednesday.
    for settlement in load_vx_settlement_dates():
        s_norm = pd.Timestamp(settlement).normalize()
        delta_days = (bar_dates - s_norm).days.values
        keep &= np.abs(delta_days) > n_settle_days

    # ES roll days (exact-date match).
    roll_dates = _es_roll_dates()
    if roll_dates:
        roll_hit = np.array([d.date() in roll_dates for d in bar_dates])
        keep &= ~roll_hit

    return out.loc[keep]


def build_session_dataset(
    date: pd.Timestamp | str,
    delta_s: int,
    k: int,
    pdv_1min: pd.DataFrame,
    *,
    grid_kind: str = "train",
    tick_es: float = 0.25,
    tick_vx: float = 0.05,
) -> dict:
    """Assemble one session's (X, y_bidup, y_askdown, block_ts).

    """
    es = load_session_events("ES", date)
    vx = load_session_events("VX", date)
    if es is None or vx is None:
        return _empty_result()

    if grid_kind == "train":
        grid = training_grid(date)
    else:
        day = pd.Timestamp(date).normalize()
        t0 = day + pd.Timedelta(hours=_GRID_OPEN[0], minutes=_GRID_OPEN[1])
        t1 = day + pd.Timedelta(hours=_GRID_CLOSE[0], minutes=_GRID_CLOSE[1])
        grid = trade_grid(t0.tz_localize(_TZ), t1.tz_localize(_TZ), delta_s)

    feats = features.build_session_features(
        es, vx, pdv_1min, grid, tick_es=tick_es, tick_vx=tick_vx
    )
    feats["tdays_to_settle"] = _trading_days_to_next_settlement(grid, load_vx_settlement_dates())

    lab = labels.make_labels(vx, grid, delta_s=delta_s, tick=tick_vx, k=k)

    joined = feats.join(lab, how="inner")
    joined = apply_calendar_masks(joined)
    joined = joined[joined["is_live"]]
    feat_cols = list(feats.columns)
    joined = joined.dropna(subset=feat_cols)

    X = joined[feat_cols]
    return {
        "X": X,
        "y_bidup": joined["y_bidup"],
        "y_askdown": joined["y_askdown"],
        "block_ts": X.index,
    }


def build_dataset(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    delta_s: int,
    k: int,
    theta: dict,
    *,
    grid_kind: str = "train",
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    tick_es: float = 0.25,
    tick_vx: float = 0.05,
) -> dict:
    """Full-window (X, y_bidup, y_askdown, block_ts), per-session cached + concat.

    """
    from code_section7.data_loader import _check_memory

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdv_full = _build_full_window_pdv(start, end, theta)

    param_tag = _param_tag(theta, tick_es, tick_vx)
    sessions = _trading_sessions(start, end)

    frames: list[dict] = []
    for date in sessions:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        cache_path = cache_dir / f"ds_{grid_kind}_{date_str}_d{delta_s}s_k{k}_{param_tag}.parquet"

        if cache_path.exists():
            frames.append(_read_cache(cache_path))
            continue

        _check_memory(f"build_dataset {date_str}")

        day_pdv = _slice_pdv_for_day(pdv_full, date)
        if day_pdv.empty:
            continue

        result = build_session_dataset(
            date,
            delta_s,
            k,
            day_pdv,
            grid_kind=grid_kind,
            tick_es=tick_es,
            tick_vx=tick_vx,
        )
        if len(result["X"]) == 0:
            del day_pdv
            gc.collect()
            continue

        _write_cache(cache_path, result)
        frames.append(result)

        del day_pdv
        gc.collect()

    if not frames:
  
        return _empty_result()

    return _concat_results(frames)





def _build_full_window_pdv(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    theta: dict,
) -> pd.DataFrame:
    """1-minute PDV block over [anchor, end], continuously propagated (canonical Section 7.1).

    """
    from code_section7.backtest.data import (
        load_databento_1min,
        load_frd_spx_daily,
        load_spx_1min,
        splice_es_roll,
    )
    from code_section7.lob import pdv

    prop_start = min(pd.Timestamp(start), pd.Timestamp(_PDV_ANCHOR_START))
    es1 = load_databento_1min("ES", prop_start, end)
    spx1 = load_spx_1min(prop_start, end)
    return_1min = splice_es_roll(es1, spx1).set_index("ts")["return_1min"]

    spx_daily = load_frd_spx_daily()
    daily_ret = np.log(spx_daily["close"]).diff().dropna()
    warmup_end = pd.Timestamp(_WARMUP_END)
    warmup = daily_ret.loc[daily_ret.index <= warmup_end].tail(_WARMUP_N_RETURNS)

    return pdv.pdv_trajectory(
        return_1min,
        theta,
        warmup_returns=warmup,
        warmup_dt_years=_WARMUP_DT_YEARS,
    )


def _slice_pdv_for_day(pdv_full: pd.DataFrame, date: pd.Timestamp | str) -> pd.DataFrame:
    """Rows of the full PDV trajectory whose calendar date == ``date``."""
    day = pd.Timestamp(date).normalize().date()
    idx = pdv_full.index
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    return pdv_full[idx_naive.normalize().date == day]


def _trading_sessions(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> pd.DatetimeIndex:
    """Business-day session dates in [start, end] inclusive (tz-naive, date-only).

    Uses a business-day range so weekends are skipped; absent-data days are
    handled downstream (``build_session_dataset`` returns an empty result when a
    session file is missing, and the PDV-slice guard skips holidays with no bars).
    """
    return pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())




def _param_tag(theta: dict, tick_es: float, tick_vx: float) -> str:
    """Short stable hash of the feature-affecting params for the cache filename.


    """
    payload = json.dumps(
        {"theta": theta, "tick_es": tick_es, "tick_vx": tick_vx},
        sort_keys=True,
        default=float,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def _write_cache(path: Path, result: dict) -> None:
    """Persist one session result to parquet (features + labels in one frame).


    """
    frame = result["X"].copy()
    frame["__y_bidup"] = result["y_bidup"].to_numpy()
    frame["__y_askdown"] = result["y_askdown"].to_numpy()
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def _read_cache(path: Path) -> dict:
    """Reload a cached session result back into the standard result dict.

    """
    frame = pd.read_parquet(path)
    y_bidup = frame.pop("__y_bidup").rename("y_bidup")
    y_askdown = frame.pop("__y_askdown").rename("y_askdown")
    return {
        "X": frame,
        "y_bidup": y_bidup,
        "y_askdown": y_askdown,
        "block_ts": frame.index,
    }




def _empty_result() -> dict:
    """Empty (X, y_bidup, y_askdown, block_ts) result."""
    empty_idx = pd.DatetimeIndex([], tz=_TZ)
    return {
        "X": pd.DataFrame(index=empty_idx),
        "y_bidup": pd.Series(dtype="uint8", index=empty_idx),
        "y_askdown": pd.Series(dtype="uint8", index=empty_idx),
        "block_ts": empty_idx,
    }


def _concat_results(frames: list[dict]) -> dict:
    """Concatenate per-session results chronologically into one combined result."""
    X = pd.concat([f["X"] for f in frames])
    y_bidup = pd.concat([f["y_bidup"] for f in frames])
    y_askdown = pd.concat([f["y_askdown"] for f in frames])
    return {
        "X": X,
        "y_bidup": y_bidup,
        "y_askdown": y_askdown,
        "block_ts": X.index,
    }
