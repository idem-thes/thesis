"""Section 7.1 forecast skill + magnitude on a HORIZON GRID ."""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from code_section7.backtest.calibrate import load_theta_hat  # noqa: E402
from code_section7.backtest.data import (  # noqa: E402
    load_frd_spx_daily,
    load_frd_vx_daily,
    load_vx_settlement_dates,
)
from code_section7.backtest.forecast import (  # noqa: E402
    expected_dsigma_drift,
    theta_hat_to_param_set,
)
from code_section7.backtest.run import _load_databento_window  # noqa: E402
from code_section7.backtest.strategy import (  # noqa: E402
    _MINUTES_PER_YEAR,
    _is_at_forecast_boundary,
    _is_in_settlement_mask,
    _is_in_trading_window,
    _propagate_state_guyon,
    initialize_state_guyon,
)
from code_section7.state import bar_quantities  # noqa: E402

_THETA_PATH = REPO_ROOT / "outputs" / "_cache" / "section7_backtest_theta_hat.json"
_OUT = REPO_ROOT / "outputs" / "_cache" / "section7_forecast_grid"
_START_HOUR, _CLOSE_HOUR = (10, 30), (15, 0)
_N_MASK_DAYS = 3
_W2_START = pd.Timestamp("2025-06-01 00:00:00", tz="America/New_York")
_START, _END, _WARMUP_END = "2025-03-02", "2025-08-29", "2025-02-28"
HORIZONS_MIN = [1, 5, 10, 15, 30, 45, 60]  # 45 included for validation vs committed
_QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]


def _replay_drift() -> pd.DataFrame:
    """Per-bar trace with the eq-54 drift (mu) at every bar; mirrors _replay_m2."""
    saved = load_theta_hat(_THETA_PATH)
    theta = saved["theta"]
    theta_param = theta_hat_to_param_set(np.asarray(theta["theta_hat"], dtype=float))

    spx = load_frd_spx_daily()
    vx = load_frd_vx_daily()
    wend = pd.Timestamp(_WARMUP_END)
    ret = np.log(spx["close"]).diff().dropna()
    warmup = ret.loc[ret.index <= wend].tail(1000)
    init_price = float(spx.loc[spx.index <= wend, "close"].iloc[-1])
    init_sigma = float(vx.loc[vx.index <= wend, "close"].iloc[-1]) / 100.0

    state = initialize_state_guyon(warmup, theta_param, dt_years=1.0 / 252.0)
    prev_ts = pd.Timestamp(wend.strftime("%Y-%m-%d") + " 16:00:00", tz="America/New_York")
    last_ssq = init_sigma * init_sigma
    last_sigma = init_sigma

    data = _load_databento_window(_START, _END, init_price=init_price)
    ts_arr = data.index
    r_arr = data["return_1min"].to_numpy()
    vx_mid_arr = data["vx_mid"].to_numpy()
    vx_bid_arr = data["vx_bid"].to_numpy()
    vx_ask_arr = data["vx_ask"].to_numpy()
    n = len(data)

    mu = np.full(n, np.nan)
    in_win = np.zeros(n, bool)
    in_set = np.zeros(n, bool)
    settle = load_vx_settlement_dates()

    for i in range(n):
        ts = ts_arr[i]
        r = r_arr[i]
        vx_mid = vx_mid_arr[i]
        sigma = float(vx_mid) / 100.0 if np.isfinite(vx_mid) else last_sigma
        last_sigma = sigma
        dt_years = (ts - prev_ts).total_seconds() / 60.0 / _MINUTES_PER_YEAR if prev_ts is not None else 1.0 / _MINUTES_PER_YEAR
        if pd.notna(r) and np.isfinite(sigma):
            state = _propagate_state_guyon(state=state, dt_years=dt_years, theta=theta_param, realized_return=float(r))
            if dt_years > 0.0:
                last_ssq = float(r) * float(r) / dt_years
        prev_ts = ts
        iw = _is_in_trading_window(ts, _START_HOUR, _CLOSE_HOUR)
        isz = _is_in_settlement_mask(ts, settle, _N_MASK_DAYS)
        in_win[i] = iw
        in_set[i] = isz
        if iw and not isz and np.isfinite(vx_bid_arr[i]) and np.isfinite(vx_ask_arr[i]):
            mu[i] = expected_dsigma_drift(state, theta_param, last_ssq)

    return pd.DataFrame(
        {"mu": mu, "vx_mid": vx_mid_arr, "vx_bid": vx_bid_arr, "vx_ask": vx_ask_arr,
         "in_window": in_win, "in_settlement": in_set}, index=ts_arr)


def _realized(vx_mid: pd.Series, h_min: int) -> np.ndarray:
    target = vx_mid.reindex(vx_mid.index + pd.Timedelta(minutes=h_min)).to_numpy()
    return target - vx_mid.to_numpy()


def _dm(f: np.ndarray, realized: np.ndarray, lag: int) -> tuple[float, float]:
    m = np.isfinite(f) & np.isfinite(realized)
    if m.sum() < max(2, lag + 1):
        return float("nan"), float("nan")
    d = (realized[m] - f[m]) ** 2 - realized[m] ** 2  # baseline = 0 (no-change)
    T = len(d)
    db = float(np.mean(d))
    var = float(np.mean((d - db) ** 2))
    for k in range(1, lag + 1):
        if k >= T:
            break
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(np.mean((d[k:] - db) * (d[:-k] - db)))
    if var <= 0 or not np.isfinite(var):
        return float("nan"), float("nan")
    from scipy.stats import norm
    stat = db / np.sqrt(var / T)
    return float(stat), float(2.0 * (1.0 - norm.cdf(abs(stat))))


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trace = _replay_drift()
    print(f"[fgrid] replay {len(trace)} bars in {time.time()-t0:.1f}s", flush=True)

    ts_idx = trace.index
    skill_rows, mag_rows = [], []
    for h in HORIZONS_MIN:
        boundary = np.array([_is_at_forecast_boundary(t, _START_HOUR, h) for t in ts_idx])
        qual = trace["in_window"].to_numpy() & ~trace["in_settlement"].to_numpy() & boundary \
            & np.isfinite(trace["vx_bid"].to_numpy()) & np.isfinite(trace["vx_ask"].to_numpy()) \
            & np.isfinite(trace["mu"].to_numpy())
        f = 100.0 * trace["mu"].to_numpy() * h / _MINUTES_PER_YEAR
        realized_all = _realized(trace["vx_mid"], h)
        sub = pd.DataFrame({
            "ts": ts_idx, "forecast_vx_pts": f, "realized": realized_all,
            "vx_bid": trace["vx_bid"].to_numpy(), "vx_ask": trace["vx_ask"].to_numpy()}).loc[qual].copy()
        sub = sub[np.isfinite(sub["forecast_vx_pts"]) & np.isfinite(sub["realized"])]  # scoreable
        sub["direction"] = np.where(sub["forecast_vx_pts"] > 0, "long", "short")
        sub["window"] = np.where(pd.DatetimeIndex(sub["ts"]) < _W2_START, "W1", "W2")
        sub.to_parquet(_OUT / f"forecast_set_h{h}.parquet", index=False)

        fa, ra = sub["forecast_vx_pts"].to_numpy(), sub["realized"].to_numpy()
        nz = (fa != 0) & (ra != 0)
        dir_acc = float(np.mean(np.sign(fa[nz]) == np.sign(ra[nz]))) if nz.any() else float("nan")
        dm_stat, dm_p = _dm(fa, ra, lag=h)
        skill_rows.append({"h_min": h, "MAE1": float(np.mean(np.abs(ra - fa))),
                           "MAE0": float(np.mean(np.abs(ra))), "dir_acc": dir_acc,
                           "dm_stat": dm_stat, "dm_p": dm_p, "n_nonzero": int(nz.sum()), "n": len(sub)})
        q = np.abs(fa)
        mag_rows.append({"h_min": h, **{f"p{int(p*100)}": float(np.quantile(q, p)) for p in _QUANTILES}, "n": len(sub)})

    skill = pd.DataFrame(skill_rows)
    mag = pd.DataFrame(mag_rows)
    skill.to_csv(_OUT / "forecast_skill_grid.csv", index=False)
    mag.to_csv(_OUT / "forecast_magnitude_grid.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== SKILL (all-bars, full window) ===")
    print(skill.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== MAGNITUDE (|forecast| VX pts; spread=0.05) ===")
    print(mag.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Validate H=45 against the committed diagnostic.
    p = REPO_ROOT / "outputs" / "section7" / "backtest" / "sandbox_fine" / "forecast_diagnostic_top2.json"
    if p.exists():
        ref = json.load(open(p))["all_bars"]["full"]
        r45 = skill[skill["h_min"] == 45].iloc[0]
        print(f"\n[validate H=45] mine MAE={r45['MAE1']:.4f} dir={r45['dir_acc']:.3f} DM={r45['dm_stat']:+.3f} n={int(r45['n'])}")
        print(f"[validate H=45] ref  MAE={ref['MAE']:.4f} dir={ref['dir_hit']:.3f} DM={ref['DM']:+.3f} n={ref['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
