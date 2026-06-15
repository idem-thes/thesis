"""Assemble the rebuilt Section 7.1 SDE tables."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from code_section7.backtest.calibrate import load_theta_hat  # noqa: E402
from code_section7.backtest.data import (  # noqa: E402
    load_frd_spx_daily, load_frd_vx_daily, load_vx_settlement_dates)
from code_section7.backtest.forecast import theta_hat_to_param_set  # noqa: E402
from code_section7.backtest.run import _load_databento_window  # noqa: E402
from code_section7.backtest.strategy import (  # noqa: E402
    _MINUTES_PER_YEAR, _is_in_settlement_mask, _is_in_trading_window,
    _propagate_state_guyon, initialize_state_guyon)
from code_section7.state import bar_quantities  # noqa: E402

_FG = REPO_ROOT / "outputs" / "_cache" / "section7_forecast_grid"
_PT = REPO_ROOT / "outputs" / "_cache" / "section7_pct_trading"
_THETA = REPO_ROOT / "outputs" / "_cache" / "section7_backtest_theta_hat.json"
_W2 = pd.Timestamp("2025-06-01 00:00:00", tz="America/New_York")
HORIZONS_MIN = [1, 5, 10, 15, 30, 60]
PCT = [0.05, 0.10, 0.20]
SEC = {1: 60, 5: 300, 10: 600, 15: 900, 30: 1800, 60: 3600}


def _tex_int(v: float) -> str:
    return f"{v:+,.0f}".replace(",", "{,}")


def _drift_components() -> pd.Series:
    """Per-bar (drift_R1, drift_R2) at qualifying bars, indexed by ts."""
    saved = load_theta_hat(_THETA)
    th = saved["theta"]
    tp = theta_hat_to_param_set(np.asarray(th["theta_hat"], float))
    spx, vx = load_frd_spx_daily(), load_frd_vx_daily()
    wend = pd.Timestamp("2025-02-28")
    ret = np.log(spx["close"]).diff().dropna()
    warmup = ret.loc[ret.index <= wend].tail(1000)
    init_price = float(spx.loc[spx.index <= wend, "close"].iloc[-1])
    init_sigma = float(vx.loc[vx.index <= wend, "close"].iloc[-1]) / 100.0
    state = initialize_state_guyon(warmup, tp, dt_years=1.0 / 252.0)
    prev = pd.Timestamp("2025-02-28 16:00:00", tz="America/New_York")
    last_ssq, last_sig = init_sigma ** 2, init_sigma
    settle = load_vx_settlement_dates()

    data = _load_databento_window("2025-03-02", "2025-08-29", init_price=init_price)
    ts = data.index
    r = data["return_1min"].to_numpy()
    vmid, vbid, vask = data["vx_mid"].to_numpy(), data["vx_bid"].to_numpy(), data["vx_ask"].to_numpy()
    dR1 = np.full(len(data), np.nan)
    dR2 = np.full(len(data), np.nan)
    for i in range(len(data)):
        t = ts[i]
        sig = float(vmid[i]) / 100.0 if np.isfinite(vmid[i]) else last_sig
        last_sig = sig
        dty = (t - prev).total_seconds() / 60.0 / _MINUTES_PER_YEAR if prev is not None else 1.0 / _MINUTES_PER_YEAR
        if pd.notna(r[i]) and np.isfinite(sig):
            state = _propagate_state_guyon(state=state, dt_years=dty, theta=tp, realized_return=float(r[i]))
            if dty > 0:
                last_ssq = float(r[i]) ** 2 / dty
        prev = t
        if _is_in_trading_window(t, (10, 30), (15, 0)) and not _is_in_settlement_mask(t, settle, 3) \
                and np.isfinite(vbid[i]) and np.isfinite(vask[i]):
            bq = bar_quantities(state, tp)
            dR1[i] = -tp.beta1 * bq.lam1 * bq.R1
            dR2[i] = (tp.beta2 * bq.lam2 / 2.0) * (last_ssq - bq.R2) / np.sqrt(max(bq.R2_nobar, 1e-8))
    out = pd.DataFrame({"drift_R1": dR1, "drift_R2": dR2}, index=ts)
    return out.dropna()


def _thr(h: int, pct: float) -> float:
    fs = pd.read_parquet(_FG / f"forecast_set_h{h}.parquet")
    return float(np.quantile(fs["forecast_vx_pts"].abs().to_numpy(), 1.0 - pct))


def main() -> int:
    skill = pd.read_csv(_FG / "forecast_skill_grid.csv")
    mag = pd.read_csv(_FG / "forecast_magnitude_grid.csv")
    trd = pd.read_parquet(_PT / "pct_trading.parquet")
    trades = pd.read_parquet(_PT / "pct_trades.parquet")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])

    print("[tables] replaying drift components ...", flush=True)
    drift = _drift_components()

    # ---- 1. forecast skill (h in {1,5,10,15}; 60/300/600/900 s) ----
    print("\n=== LaTeX: tab:section7_forecast_skill ===")
    for h in [1, 5, 10, 15]:
        r = skill[skill["h_min"] == h].iloc[0]
        print(f"${SEC[h]}$ & ${r['MAE1']:.3f}$ & ${r['MAE0']:.3f}$ & ${r['dir_acc']:.3f}$ "
              f"& ${r['dm_stat']:+.2f}\\ ({r['dm_p']:.3f})$ & ${int(r['n_nonzero']):,}$ & ${int(r['n']):,}$ \\\\".replace(",", "{,}"))

    # ---- 2. forecast magnitude ----
    print("\n=== LaTeX: tab:section7_forecast_magnitude ===")
    for h in [1, 5, 10, 15]:
        r = mag[mag["h_min"] == h].iloc[0]
        cells = " & ".join(f"${r[f'p{p}']:.4f}$" for p in [50, 75, 90, 95, 99])
        print(f"${SEC[h]}$ & {cells} & ${int(r['n']):,}$ \\\\".replace(",", "{,}"))

    # ---- 3. trading Table 14 (all 6 horizons x 3 gates) ----
    print("\n=== LaTeX: tab:section7_trading ===")
    for h in HORIZONS_MIN:
        lbl = f"{SEC[h]}\\,s" if h <= 15 else f"{h}\\,min"
        print(f"\\multicolumn{{8}}{{l}}{{\\textit{{$h={lbl}$}}}}\\\\")
        for pct in PCT:
            r = trd[(trd["h_min"] == h) & (np.isclose(trd["top_pct"], pct))].iloc[0]
            pa = f"{r['p_fill_adv']:.2f}" if np.isfinite(r['p_fill_adv']) else "---"
            pf = f"{r['p_fill_fav']:.2f}" if np.isfinite(r['p_fill_fav']) else "---"
            rt = f"{r['ratio']:.2f}" if np.isfinite(r['ratio']) else "---"
            pna = f"{r['pnl_adv']:+.0f}" if np.isfinite(r['pnl_adv']) else "---"
            pnf = f"{r['pnl_fav']:+.0f}" if np.isfinite(r['pnl_fav']) else "---"
            print(f"top ${int(round(pct*100))}\\%$ & ${int(r['n_fire']):,}$ & ${pa}$ & ${pf}$ & ${rt}$ "
                  f"& ${pna}$ & ${pnf}$ & ${_tex_int(r['total'])}$ \\\\".replace(",", "{,}"))

    # ---- 4. W1/W2 split + 5. R1/R2 channel (top 10%) ----
    tol = pd.Timedelta(minutes=2)
    dr1 = drift["drift_R1"]; dr2 = drift["drift_R2"]

    def _chan(sub):
        """R1/R2 channel stats for a subset of filled trades."""
        if sub.empty:
            return dict(n_R1=0, pnl_R1=0.0, mu_R1=np.nan, n_R2=0, pnl_R2=0.0, mu_R2=np.nan)
        a1 = dr1.reindex(pd.DatetimeIndex(sub["entry_ts"]), method="nearest", tolerance=tol).to_numpy()
        a2 = dr2.reindex(pd.DatetimeIndex(sub["entry_ts"]), method="nearest", tolerance=tol).to_numpy()
        v = ~(np.isnan(a1) | np.isnan(a2))
        cv = sub.iloc[np.flatnonzero(v)]
        isR1 = np.abs(a1[v]) >= np.abs(a2[v])
        pnl = cv["pnl"].to_numpy()
        return dict(n_R1=int(isR1.sum()), pnl_R1=float(pnl[isR1].sum()),
                    mu_R1=float(pnl[isR1].mean()) if isR1.any() else np.nan,
                    n_R2=int((~isR1).sum()), pnl_R2=float(pnl[~isR1].sum()),
                    mu_R2=float(pnl[~isR1].mean()) if (~isR1).any() else np.nan)

    wrows, crows, cwrows = [], [], []
    for h in HORIZONS_MIN:
        thr = _thr(h, 0.10)
        cell = trades[(trades["h_min"] == h) & (trades["forecast_vx_pts"].abs() >= thr) & (trades["filled"])].copy()
        w1 = cell[pd.DatetimeIndex(cell["entry_ts"]) < _W2]
        w2 = cell[pd.DatetimeIndex(cell["entry_ts"]) >= _W2]
        wrows.append({"h": h, "n_w1": len(w1), "pnl_w1": w1["pnl"].sum(), "n_w2": len(w2), "pnl_w2": w2["pnl"].sum()})
        crows.append({"h": h, **_chan(cell)})
        cwrows.append({"h": h, "win": "W1", **_chan(w1)})
        cwrows.append({"h": h, "win": "W2", **_chan(w2)})

    print("\n=== LaTeX: tab:section7_wsplit (top 10%) ===")
    for r in wrows:
        lbl = f"{SEC[r['h']]}$\\,$s" if r['h'] <= 15 else f"{r['h']}$\\,$min"
        print(f"{lbl} & ${r['n_w1']}$ & ${_tex_int(r['pnl_w1'])}$ & ${r['n_w2']}$ & ${_tex_int(r['pnl_w2'])}$ \\\\")
    print("\n=== LaTeX: tab:section7_channel_split (top 10%, full window) ===")
    for r in crows:
        lbl = f"{SEC[r['h']]}$\\,$s" if r['h'] <= 15 else f"{r['h']}$\\,$min"
        mu1 = f"{r['mu_R1']:+.0f}" if np.isfinite(r['mu_R1']) else "---"
        mu2 = f"{r['mu_R2']:+.0f}" if np.isfinite(r['mu_R2']) else "---"
        print(f"{lbl} & ${r['n_R1']}$ & ${_tex_int(r['pnl_R1'])}$ & ${mu1}$ & ${r['n_R2']}$ & ${_tex_int(r['pnl_R2'])}$ & ${mu2}$ \\\\")

    print("\n=== LaTeX: tab:section7_channel_split SPLIT BY W1/W2 (top 10%) ===")
    for i, r in enumerate(cwrows):
        lbl = (f"{SEC[r['h']]}$\\,$s" if r['h'] <= 15 else f"{r['h']}$\\,$min") if r['win'] == "W1" else ""
        mu1 = f"{r['mu_R1']:+.0f}" if np.isfinite(r['mu_R1']) else "---"
        mu2 = f"{r['mu_R2']:+.0f}" if np.isfinite(r['mu_R2']) else "---"
        if r['win'] == "W1" and i > 0:
            print("\\midrule")
        print(f"{lbl} & {r['win']} & ${r['n_R1']}$ & ${_tex_int(r['pnl_R1'])}$ & ${mu1}$ & ${r['n_R2']}$ & ${_tex_int(r['pnl_R2'])}$ & ${mu2}$ \\\\")
    print("\nW1/W2 channel:", pd.DataFrame(cwrows).to_string(index=False))

    print("\n=== plain summary ===")
    print("W1/W2:", pd.DataFrame(wrows).to_string(index=False))
    print("channel:", pd.DataFrame(crows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
