"""Per-session ES+VX+PDV feature frame for Section 7.2 LOB forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from code_section7.lob import events

_EPS = 1e-12


_Z_LOOKBACK_S = 300
_VOL_LOOKBACK_S = 300
_QDEPTH_LOOKBACK_S = 300

_ES_OFI_WINDOWS = (5, 10, 30, 60)
_ES_RET_WINDOWS = (5, 10, 30, 60)
_ES_RVOL_WINDOWS = (30, 60)
_VX_RET_WINDOWS = tuple(range(1, 21))  # Degtyar Section 3.2.8 "last 20 deltas"

_MACD_FAST = 12
_MACD_SLOW = 26
_BESTSZ_VOL_S = 60
_VX_RVOL_S = 60


def _ns(times: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """tz-aware stamps -> datetime64[ns]; a uniform UTC shift preserves ordering."""
    return pd.Index(times).to_numpy("datetime64[ns]")


def _asof(ts: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Value of the last event <= each query time (NaN where none precedes it)."""
    snap = np.searchsorted(ts, query, side="right") - 1
    out = np.full(len(query), np.nan)
    valid = snap >= 0
    out[valid] = values[snap[valid]]
    return out


def _window_sum(
    ts: np.ndarray, increments: np.ndarray, query: np.ndarray, window_s: int
) -> np.ndarray:
    """Sum increments over events in the trailing window (t-W, t] for each query t."""
    window = np.timedelta64(window_s, "s").astype("timedelta64[ns]")
    hi = np.searchsorted(ts, query, side="right")  # count of events <= t
    lo = np.searchsorted(ts, query - window, side="right")  # count of events <= t-W
    csum = np.concatenate(([0.0], np.cumsum(increments)))
    return csum[hi] - csum[lo]


def _trailing_z(series: pd.Series, lookback_s: int) -> pd.Series:
    """(x - trailing mean) / trailing std over a full `lookback_s` window."""
    window = f"{lookback_s}s"
    mean = series.rolling(window, min_periods=lookback_s).mean()
    std = series.rolling(window, min_periods=lookback_s).std()
    return (series - mean) / std.where(std > _EPS)


def _per_second_vol(returns_1s: pd.Series, lookback_s: int) -> pd.Series:
    """Trailing per-second realized vol = sqrt (mean r^2) over a full `lookback_s` window."""
    mean_sq = (returns_1s**2).rolling(f"{lookback_s}s", min_periods=lookback_s).mean()
    return np.sqrt(mean_sq)


def _event_frame(ev: pd.DataFrame, levels: int, clock: str = "ts_event") -> dict[str, np.ndarray]:
    """Event-level arrays: timestamps, mid, microprice, imbalances, OFI, best sizes.

    """
    ts_raw = _ns(ev[clock])
    order = np.argsort(ts_raw, kind="stable")
    ev = ev.iloc[order]
    ts = ts_raw[order]
    bid_px0 = ev["bid_px_00"].to_numpy(float)
    ask_px0 = ev["ask_px_00"].to_numpy(float)
    bid_sz0 = ev["bid_sz_00"].to_numpy(float)
    ask_sz0 = ev["ask_sz_00"].to_numpy(float)
    bid_sz_cols = [ev[f"bid_sz_{i:02d}"].to_numpy(float) for i in range(levels)]
    ask_sz_cols = [ev[f"ask_sz_{i:02d}"].to_numpy(float) for i in range(levels)]
    bid_sz_deep = np.sum(bid_sz_cols, axis=0)
    ask_sz_deep = np.sum(ask_sz_cols, axis=0)
    return {
        "ts": ts,
        "mid": (bid_px0 + ask_px0) / 2.0,
        "microprice": events.microprice(bid_px0, bid_sz0, ask_px0, ask_sz0),
        "imb_l1": events.imbalance(bid_sz0, ask_sz0),
        "imb_deep": events.imbalance(bid_sz_deep, ask_sz_deep),
        "ofi": events.ofi_increments(bid_px0, bid_sz0, ask_px0, ask_sz0),
        "bid_px0": bid_px0,
        "ask_px0": ask_px0,
        "bid_sz0": bid_sz0,
        "ask_sz0": ask_sz0,
    }


def _scaled_returns(
    mid_grid: pd.Series, ts: np.ndarray, mid_ev: np.ndarray, g_ns: np.ndarray, windows
) -> dict[str, pd.Series]:
    """log(mid_t / mid_{t-L}) / sigma_hat for each horizon L; sigma_hat = trailing per-sec vol."""
    ret_1s = np.log(mid_grid / mid_grid.shift(1))
    sigma = _per_second_vol(ret_1s, _VOL_LOOKBACK_S)
    out: dict[str, pd.Series] = {}
    for w in windows:
        back = g_ns - np.timedelta64(w, "s").astype("timedelta64[ns]")
        mid_back = _asof(ts, mid_ev, back)
        raw = np.log(mid_grid.to_numpy() / mid_back)
        out[w] = pd.Series(raw, index=mid_grid.index) / sigma
    return out


def _realized_vol_windows(mid_grid: pd.Series, windows) -> dict[int, pd.Series]:
    """sqrt sum r^2 of 1-s mid log-returns over each trailing window."""
    ret_1s = np.log(mid_grid / mid_grid.shift(1))
    sq = ret_1s**2
    out: dict[int, pd.Series] = {}
    for w in windows:
        out[w] = np.sqrt(sq.rolling(f"{w}s", min_periods=w).sum())
    return out


def build_es_features(
    es_events: pd.DataFrame,
    cgrid: pd.DatetimeIndex,
    *,
    tick_es: float = 0.25,
    steps_per_sec: int = 1,
    clock: str = "ts_event",
) -> dict[str, pd.Series | np.ndarray]:
    """ES feature Series on the canonical lattice ``cgrid``.

    """
    levels = sum(1 for c in es_events.columns if c.startswith("bid_sz_"))
    g_ns = _ns(cgrid)
    es = _event_frame(es_events, levels, clock=clock)
    es_mid_grid = pd.Series(_asof(es["ts"], es["mid"], g_ns), index=cgrid)

    out: dict[str, pd.Series | np.ndarray] = {}

    # --- ES instantaneous (as-of last ES event <= t) ---
    es_micro = _asof(es["ts"], es["microprice"], g_ns)
    out["es_imb_l1"] = _asof(es["ts"], es["imb_l1"], g_ns)
    out["es_imb_deep"] = _asof(es["ts"], es["imb_deep"], g_ns)
    out["es_microprice_dev"] = (es_micro - es_mid_grid.to_numpy()) / tick_es

    # --- ES windowed OFI (sum over window, then trailing z) ---
    for w in _ES_OFI_WINDOWS:
        ofi_sum = _window_sum(es["ts"], es["ofi"], g_ns, w)
        out[f"es_ofi_{w}s"] = _trailing_z(pd.Series(ofi_sum, index=cgrid), _Z_LOOKBACK_S)

    # --- ES windowed scaled returns + realized vol ---
    es_rets = _scaled_returns(es_mid_grid, es["ts"], es["mid"], g_ns, _ES_RET_WINDOWS)
    for w in _ES_RET_WINDOWS:
        out[f"es_ret_{w}s"] = es_rets[w]
    es_rvol = _realized_vol_windows(es_mid_grid, _ES_RVOL_WINDOWS)
    for w in _ES_RVOL_WINDOWS:
        out[f"es_rvol_{w}s"] = es_rvol[w]

    # --- ES MACD (EMA12 - EMA26 of es_mid, sigma-scaled; spans wall-clock-preserved) ---
    ema_fast = es_mid_grid.ewm(span=_MACD_FAST * steps_per_sec, adjust=True).mean()
    ema_slow = es_mid_grid.ewm(span=_MACD_SLOW * steps_per_sec, adjust=True).mean()
    mid_std = es_mid_grid.rolling(f"{_Z_LOOKBACK_S}s", min_periods=_Z_LOOKBACK_S).std()
    out["es_macd"] = (ema_fast - ema_slow) / mid_std.where(mid_std > _EPS)

    # --- ES best-size volatility ---
    es_best_sz = pd.Series(_asof(es["ts"], es["bid_sz0"], g_ns), index=cgrid)
    out["es_bestsz_vol_60s"] = es_best_sz.rolling(f"{_BESTSZ_VOL_S}s", min_periods=2).std()
    return out


def build_session_features(
    es_events: pd.DataFrame,
    vx_events: pd.DataFrame,
    pdv_1min: pd.DataFrame,
    grid: pd.DatetimeIndex,
    *,
    tick_es: float = 0.25,
    tick_vx: float = 0.05,
) -> pd.DataFrame:
    """ES+VX+PDV trailing feature frame indexed by `grid`, one row per timestamp.

    """
    levels = sum(1 for c in es_events.columns if c.startswith("bid_sz_"))

    # Canonical 1-second lattice spanning the requested grid. All windows / EMAs /
    # shift(1) below run on this lattice so a coarse `grid` gets identical feature
    # definitions to the 1 s grid (then we sample back onto `grid` at the end).
    cgrid = pd.date_range(grid.min(), grid.max(), freq="1s", tz=grid.tz)
    g_ns = _ns(cgrid)

    vx = _event_frame(vx_events, levels)
    vx_mid_grid = pd.Series(_asof(vx["ts"], vx["mid"], g_ns), index=cgrid)

    feat: dict[str, pd.Series | np.ndarray] = {}

    # --- ES block (extracted, lattice-aware; steps_per_sec=1 -> 1 s pipeline) ---
    es_feat = build_es_features(es_events, cgrid, tick_es=tick_es, steps_per_sec=1)
    feat.update(es_feat)
    es_microprice_dev = es_feat["es_microprice_dev"]  # reused by esvx_div below

    # --- VX instantaneous ---
    vx_micro = _asof(vx["ts"], vx["microprice"], g_ns)
    feat["vx_imb_l1"] = _asof(vx["ts"], vx["imb_l1"], g_ns)
    vx_microprice_dev = (vx_micro - vx_mid_grid.to_numpy()) / tick_vx
    feat["vx_microprice_dev"] = vx_microprice_dev
    vx_bid0 = _asof(vx["ts"], vx["bid_px0"], g_ns)
    vx_ask0 = _asof(vx["ts"], vx["ask_px0"], g_ns)
    feat["vx_spread_ticks"] = (vx_ask0 - vx_bid0) / tick_vx
    vx_bid_sz = pd.Series(_asof(vx["ts"], vx["bid_sz0"], g_ns), index=cgrid)
    vx_ask_sz = pd.Series(_asof(vx["ts"], vx["ask_sz0"], g_ns), index=cgrid)
    bid_med = vx_bid_sz.rolling(f"{_QDEPTH_LOOKBACK_S}s", min_periods=1).median()
    ask_med = vx_ask_sz.rolling(f"{_QDEPTH_LOOKBACK_S}s", min_periods=1).median()
    feat["vx_qdepth_bid"] = vx_bid_sz / bid_med.where(bid_med > _EPS)
    feat["vx_qdepth_ask"] = vx_ask_sz / ask_med.where(ask_med > _EPS)

    # --- ES->VX cross-asset divergence ---
    z_es_dev = _trailing_z(pd.Series(es_microprice_dev, index=cgrid), _Z_LOOKBACK_S)
    z_vx_dev = _trailing_z(pd.Series(vx_microprice_dev, index=cgrid), _Z_LOOKBACK_S)
    feat["esvx_div"] = z_es_dev - z_vx_dev

    # --- VX windowed: 20-delta signed returns + realized vol ---
    vx_rets = _scaled_returns(vx_mid_grid, vx["ts"], vx["mid"], g_ns, _VX_RET_WINDOWS)
    for w in _VX_RET_WINDOWS:
        feat[f"vx_ret_{w}s"] = vx_rets[w]
    vx_rvol = _realized_vol_windows(vx_mid_grid, (_VX_RVOL_S,))
    feat["vx_rvol_60s"] = vx_rvol[_VX_RVOL_S]

    # --- PDV block (backward asof onto grid; constant within the minute) ---
    pdv_cols = ["R_bar_1", "R_bar_2", "sigma_model", "drift_R1", "drift_R2"]
    pdv_idx = _ns(pdv_1min.index)
    for c in pdv_cols:
        feat[c] = _asof(pdv_idx, pdv_1min[c].to_numpy(float), g_ns)

    # --- Time ---
    open_ts = pd.Index(cgrid.normalize()) + pd.Timedelta(hours=9, minutes=30)
    secs = (cgrid - open_ts).total_seconds().to_numpy(float)
    feat["secs_since_open"] = secs
    feat["bucket_15m"] = np.floor(secs / (15 * 60.0))

    # Feature body is fully defined on the 1 s lattice; sample it onto `grid`.
    frame_1s = pd.DataFrame(feat, index=cgrid).astype(float)
    grid_idx = pd.DatetimeIndex(grid)
    if len(cgrid) == len(grid_idx) and cgrid.equals(grid_idx):
        return frame_1s  # grid already is the 1 s lattice - identity
    # Backward/as-of sample: each (integer-second) grid point lies on `cgrid`, so
    # this picks the 1 s feature at that timestamp; ffill is a safe fallback only.
    return frame_1s.reindex(grid_idx, method="ffill")
