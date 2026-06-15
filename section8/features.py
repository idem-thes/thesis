"""27-col VX/ES Delta VX-regression feature set (Section 8 run-2).

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from code_section7.lob.features import _asof, _event_frame, _ns


_EPS = 1e-12
_RTH_SECONDS = 23400  # 6.5h * 3600
_MACD_FAST_SPAN = 10
_MACD_SLOW_SPAN = 20

# Build-order column list. ES block (1-12), VX block (13-24), PDV (25-26), Time (27).
VX_FEATURE_COLS = [
    "es_imb_1",
    "es_imb_3",
    "es_imb_5",
    "es_microprice_dev",
    "es_spread_ticks",
    "es_ret_5s",
    "es_ret_10s",
    "es_ret_20s",
    "es_rvol_20s",
    "es_rvol_40s",
    "es_rvol_60s",
    "es_macd",
    "vx_imb_1",
    "vx_imb_3",
    "vx_imb_5",
    "vx_microprice_dev",
    "vx_spread_ticks",
    "vx_ret_5s",
    "vx_ret_10s",
    "vx_ret_20s",
    "vx_rvol_20s",
    "vx_rvol_40s",
    "vx_rvol_60s",
    "vx_macd",
    "drift_R1",
    "drift_R2",
    "progress_rth",
]


def _compute_block(
    events: pd.DataFrame, lattice: pd.DatetimeIndex, tick: float, prefix: str
) -> dict[str, np.ndarray]:
    """Compute the 12 features for one asset block (ES or VX)."""
    ef = _event_frame(events, levels=5, clock="ts_event")
    g_ns = _ns(lattice)
    out: dict[str, np.ndarray] = {}

    # imb_1: from _event_frame's L1 imbalance, masked to NaN when both sides empty
    # (matches the convention of imb_3 / imb_5 below; events.imbalance returns 0.0
    # on an empty book by default).
    l1_denom_ok = (ef["bid_sz0"] + ef["ask_sz0"]) > _EPS
    imb_l1_masked = np.where(l1_denom_ok, ef["imb_l1"], np.nan)
    out[f"{prefix}imb_1"] = _asof(ef["ts"], imb_l1_masked, g_ns)

    # imb_3 / imb_5: _event_frame exposes only L1 + full-N imb_deep, so compute
    # the 3- and 5-level imbalances directly from raw event size columns.
    ev = events.sort_values("ts_event").reset_index(drop=True)
    ts_ns = _ns(pd.DatetimeIndex(ev["ts_event"]))
    bid_n = [ev[f"bid_sz_{i:02d}"].to_numpy(float) for i in range(5)]
    ask_n = [ev[f"ask_sz_{i:02d}"].to_numpy(float) for i in range(5)]
    for n in (3, 5):
        bsum = np.sum(bid_n[:n], axis=0)
        asum = np.sum(ask_n[:n], axis=0)
        denom = np.where((bsum + asum) > _EPS, bsum + asum, np.nan)
        ratio = (bsum - asum) / denom
        out[f"{prefix}imb_{n}"] = _asof(ts_ns, ratio, g_ns)

    # microprice_dev = (mu - m) / tick. spread_ticks = (ask - bid) / tick.
    micro = _asof(ef["ts"], ef["microprice"], g_ns)
    mid_arr = _asof(ef["ts"], ef["mid"], g_ns)
    out[f"{prefix}microprice_dev"] = (micro - mid_arr) / tick
    bid0 = _asof(ef["ts"], ef["bid_px0"], g_ns)
    ask0 = _asof(ef["ts"], ef["ask_px0"], g_ns)
    out[f"{prefix}spread_ticks"] = (ask0 - bid0) / tick

    # Lattice mid as a pandas Series (for rolling + ewm operations on 1s lattice).
    mid_s = pd.Series(mid_arr, index=lattice)

    # Raw log-returns over 5/10/20s (mid_t / mid_{t-Ns}). No sigma-scaling.
    for n in (5, 10, 20):
        prev = mid_s.shift(n)
        out[f"{prefix}ret_{n}s"] = np.log(mid_s / prev).to_numpy()

    # 1s log-returns for RVol windows.
    ret_1s = np.log(mid_s / mid_s.shift(1))
    for n in (20, 40, 60):
        rv = np.sqrt((ret_1s**2).rolling(f"{n}s", min_periods=n).sum())
        out[f"{prefix}rvol_{n}s"] = rv.to_numpy()

    # MACD: EMA(span=10) - EMA(span=20) of mid on the 1s lattice. Raw price units.
    ema_f = mid_s.ewm(span=_MACD_FAST_SPAN, adjust=True).mean()
    ema_s = mid_s.ewm(span=_MACD_SLOW_SPAN, adjust=True).mean()
    out[f"{prefix}macd"] = (ema_f - ema_s).to_numpy()

    return out


def build_vx_session_features(
    es_events: pd.DataFrame,
    vx_events: pd.DataFrame,
    pdv_1min: pd.DataFrame,
    grid: pd.DatetimeIndex,
    *,
    tick_vx: float = 0.05,
    tick_es: float = 0.25,
) -> pd.DataFrame:
    """27-col feature frame on a 1s lattice, as-of sampled onto `grid`."""
    grid_idx = pd.DatetimeIndex(grid)
    cgrid = pd.date_range(grid.min(), grid.max(), freq="1s", tz=grid.tz)
    feat: dict[str, np.ndarray] = {c: np.full(len(cgrid), np.nan, float) for c in VX_FEATURE_COLS}

    feat.update(_compute_block(es_events, cgrid, tick_es, "es_"))
    feat.update(_compute_block(vx_events, cgrid, tick_vx, "vx_"))

    # PDV block - backward as-of on the 1-min pdv_trajectory frame, constant within the minute.
    g_ns = _ns(cgrid)
    pdv_idx = _ns(pdv_1min.index)
    for c in ("drift_R1", "drift_R2"):
        feat[c] = _asof(pdv_idx, pdv_1min[c].to_numpy(float), g_ns)


    open_naive = cgrid.tz_convert(None).normalize() + pd.Timedelta(hours=9, minutes=30)
    open_ts = open_naive.tz_localize(cgrid.tz, ambiguous="raise", nonexistent="raise")
    secs = (cgrid - open_ts).total_seconds().to_numpy(float)
    feat["progress_rth"] = secs / _RTH_SECONDS

    frame_1s = pd.DataFrame(feat, index=cgrid)[VX_FEATURE_COLS].astype(float)
    if len(cgrid) == len(grid_idx) and cgrid.equals(grid_idx):
        return frame_1s
    return frame_1s.reindex(grid_idx, method="ffill")
