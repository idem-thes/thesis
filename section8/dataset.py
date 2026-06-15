"""Assemble the Section 8 regression matrix: X (36-col VX features) + market + Delta VX targets.

"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import pandas as pd

from code_section7.data_loader import _check_memory, load_session_events
from code_section7.lob.dataset import (
    _build_full_window_pdv,
    _param_tag,
    _slice_pdv_for_day,
    _trading_sessions,
    apply_calendar_masks,
    training_grid,
)

from code_section8_run2.features import build_vx_session_features
from code_section8_run2.targets import build_market_frame, realized_delta_vx

log = logging.getLogger(__name__)
MARKET_COLS = ["vx_mid", "vx_bid", "vx_ask"]
_DEFAULT_CACHE_DIR = Path("outputs/_cache/section8_run2")
_EMPTY = {
    "X": pd.DataFrame(),
    "market": pd.DataFrame(),
    "targets": {},
    "block_ts": pd.DatetimeIndex([]),
}


def assemble(
    feats: pd.DataFrame,
    market: pd.DataFrame,
    targets: dict[int, pd.Series],
    *,
    n_settle_days: int = 3,
) -> dict:
    """Join features+market+targets, apply calendar masks, drop NaN-FEATURE rows.

    """
    feat_cols = list(feats.columns)
    df = feats.join(market, how="inner")
    for h, y in targets.items():
        df[f"y_{h}s"] = y
    df = apply_calendar_masks(df, n_settle_days=n_settle_days)
    df = df.dropna(subset=feat_cols)
    X = df[feat_cols]
    return {
        "X": X,
        "market": df[MARKET_COLS],
        "targets": {h: df[f"y_{h}s"] for h in targets},
        "block_ts": X.index,
    }


def build_session_dataset(
    date,
    horizons: list[int],
    pdv_1min: pd.DataFrame,
    *,
    tick_es: float = 0.25,
    tick_vx: float = 0.05,
) -> dict:
    """One session's regression dataset on the 1 s 10:30-15:00 grid (or empty)."""
    es = load_session_events("ES", date)
    vx = load_session_events("VX", date)
    if es is None or vx is None:
        return dict(_EMPTY)
    grid = training_grid(date)
    feats = build_vx_session_features(es, vx, pdv_1min, grid, tick_es=tick_es, tick_vx=tick_vx)
    market = build_market_frame(vx, grid)
    targets = {h: realized_delta_vx(vx, grid, h) for h in horizons}
    return assemble(feats, market, targets)


def _htag(horizons: list[int]) -> str:
    return "h" + "-".join(str(h) for h in horizons)


def _to_frame(result: dict) -> pd.DataFrame:
    df = result["X"].copy()
    for c in MARKET_COLS:
        df[c] = result["market"][c]
    for h, y in result["targets"].items():
        df[f"y_{h}s"] = y
    return df


def _from_frame(df: pd.DataFrame, horizons: list[int]) -> dict:
    tcols = {f"y_{h}s" for h in horizons}
    fcols = [c for c in df.columns if c not in MARKET_COLS and c not in tcols]
    return {
        "X": df[fcols],
        "market": df[MARKET_COLS],
        "targets": {h: df[f"y_{h}s"] for h in horizons},
        "block_ts": df.index,
    }


def build_dataset(
    start,
    end,
    horizons: list[int],
    theta: dict,
    *,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    tick_es: float = 0.25,
    tick_vx: float = 0.05,
) -> dict:
    """Full-window (X, market, targets, block_ts), per-session cached + concatenated.

    PDV propagated ONCE over [start, end] (continuous Section 7.1 wiring) and sliced per day.
    `theta` = code_section7.backtest.calibrate.load_theta_hat(...)["theta"].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdv_full = _build_full_window_pdv(start, end, theta)
    tag = _param_tag(theta, tick_es, tick_vx)
    htag = _htag(horizons)

    frames: list[dict] = []
    for date in _trading_sessions(start, end):
        d = pd.Timestamp(date).strftime("%Y-%m-%d")
        path = cache_dir / f"reg_{d}_{htag}_{tag}.parquet"
        if path.exists():
            frames.append(_from_frame(pd.read_parquet(path), horizons))
            continue
        _check_memory(f"build_dataset {d}")
        day_pdv = _slice_pdv_for_day(pdv_full, date)
        if day_pdv.empty:
            continue
        res = build_session_dataset(date, horizons, day_pdv, tick_es=tick_es, tick_vx=tick_vx)
        if len(res["X"]) == 0:
            del day_pdv
            gc.collect()
            continue
        _to_frame(res).to_parquet(path)
        frames.append(res)
        del day_pdv
        gc.collect()

    if not frames:
        return dict(_EMPTY)
    X = pd.concat([f["X"] for f in frames])
    market = pd.concat([f["market"] for f in frames])
    targets = {h: pd.concat([f["targets"][h] for f in frames]) for h in horizons}
    return {"X": X, "market": market, "targets": targets, "block_ts": X.index}
