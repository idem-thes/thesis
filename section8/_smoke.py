"""Section 8 run-2 smoke: one clean ~2-week window, sliding 7/1/1 + val depth-tuning, all horizons.

"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from code_section7.backtest.calibrate import load_theta_hat
from code_section7.backtest.data import load_vx_settlement_dates
from code_section7.data_loader import load_session_events
from code_section8_run2 import execution as ex
from code_section8_run2.dataset import build_dataset
from code_section8_run2.metrics import forecast_report
from code_section8_run2.walkforward import run_regression_wf

HORIZONS_S = [40, 60, 300, 600, 900]
PCTILE_GRID = [0.05, 0.10, 0.20]  # trade the top 5/10/20% by |E_hat| (per horizon)
PDV_COLS = ["drift_R1", "drift_R2"]
_THETA = "outputs/_cache/section7_backtest_theta_hat.json"


def conviction_pnl(oos: pd.DataFrame, horizon_s: int) -> pd.DataFrame:
    """PnL ($) trading the top-pct |E_hat| blocks per horizon (sign = direction).

    """
    oos = oos.copy()
    oos["date"] = pd.DatetimeIndex(oos["block_ts"]).tz_convert("America/New_York").normalize()
    ev_by_day = {d: load_session_events("VX", pd.Timestamp(d)) for d in oos["date"].unique()}
    absp = oos["y_pred"].abs().to_numpy()
    rows = []
    for pct in PCTILE_GRID:
        thr = float(np.quantile(absp, 1.0 - pct))
        fired = oos[absp >= thr]
        taker = maker = 0.0
        n_fire = n_dir = n_hit = 0
        for _, r in fired.iterrows():
            if float(r["vx_ask"]) <= float(r["vx_bid"]):
                continue
            ev = ev_by_day[r["date"]]
            if ev is None:
                continue
            n_fire += 1
            direction = "long" if r["y_pred"] > 0 else "short"
            if r["y_true"] != 0:
                n_dir += 1
                n_hit += int(np.sign(r["y_pred"]) == np.sign(r["y_true"]))
            entry_px = float(r["vx_ask"]) if direction == "long" else float(r["vx_bid"])
            taker += (
                ex.taker_trade(ev, r["block_ts"], direction, horizon_s, entry_px=entry_px) or 0.0
            )
            maker += ex.maker_block_pnl(ev, r["block_ts"], direction, horizon_s) or 0.0
        rows.append(
            {
                "top_pct": pct,
                "n_fire": n_fire,
                "dir_fired": round(n_hit / n_dir, 3) if n_dir else float("nan"),
                "taker": round(taker, 1),
                "maker": round(maker, 1),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="section8-smoke")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args(argv)

    theta = load_theta_hat(_THETA)["theta"]
    ds = build_dataset(args.start, args.end, HORIZONS_S, theta)
    print(f"[smoke] dataset rows={len(ds['X'])} cols={ds['X'].shape[1]}", flush=True)

    settle = load_vx_settlement_dates()
    oos_full = run_regression_wf(ds, HORIZONS_S, settle)
    ds_pdv = {**ds, "X": ds["X"][[c for c in PDV_COLS if c in ds["X"].columns]]}
    oos_pdv = run_regression_wf(ds_pdv, HORIZONS_S, settle)

    print("\n=== forecast skill (OOS) ===")
    for h in HORIZONS_S:
        o = oos_full[h]
        if len(o) == 0:
            print(f"  h={h:>4}s: no OOS blocks")
            continue
        rep_rw = forecast_report(
            o["y_true"].to_numpy(), o["y_pred"].to_numpy(), baseline=np.zeros(len(o)), lag=1
        )
        pdv_pred = oos_pdv[h].set_index("block_ts")["y_pred"].reindex(o["block_ts"]).to_numpy()
        rep_pdv = forecast_report(
            o["y_true"].to_numpy(), o["y_pred"].to_numpy(), baseline=pdv_pred, lag=1
        )
        print(
            f"  h={h:>4}s: MAE={rep_rw['mae']:.4f} (RW {rep_rw['mae_baseline']:.4f}) "
            f"dir={rep_rw['dir_acc']:.3f} DMvRW={rep_rw['dm_stat']:+.2f}(p{rep_rw['dm_p']:.2f}) "
            f"DMvPDV={rep_pdv['dm_stat']:+.2f}(p{rep_pdv['dm_p']:.2f}) n={rep_rw['n']}"
        )

    print("\n=== conviction-gate PnL ($) - trade top-pct |E_hat| per horizon ===")
    for h in HORIZONS_S:
        if len(oos_full[h]):
            print(f"  horizon {h}s:")
            print(conviction_pnl(oos_full[h], h).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
