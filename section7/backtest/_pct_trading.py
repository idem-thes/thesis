"""Section 7.1 trading Table-14 mirror: top {5,10,20}% gate * cancel-aware maker."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from code_section7.backtest._cancel_aware_passive import cancel_aware_trade  # noqa: E402
from code_section7.data_loader import load_session_events  # noqa: E402

_FG = REPO_ROOT / "outputs" / "_cache" / "section7_forecast_grid"
_OUT = REPO_ROOT / "outputs" / "_cache" / "section7_pct_trading"
HORIZONS_MIN = [1, 5, 10, 15, 30, 60]
PCTILE_GRID = [0.05, 0.10, 0.20]
_CAPITAL, _TRADING_DAYS = 10_000.0, 252
NY = "America/New_York"


def _sharpe(per_trade: pd.DataFrame) -> float:
    if per_trade.empty:
        return float("nan")
    daily = per_trade.groupby("date")["pnl"].sum() / _CAPITAL
    if len(daily) < 2:
        return float("nan")
    sd = float(daily.std(ddof=1))
    return float(daily.mean()) / sd * math.sqrt(_TRADING_DAYS) if sd > 0 else float("nan")


def run(max_dates: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Per horizon: thresholds computed over the FULL scoreable forecast set (as in
    # Section 8 _fill_selection - quantile over all predictions, THEN the ask>bid filter).
    # Execution universe = top-max(pct) fired set with a valid book.
    fired_per_h, thr_per_h = {}, {}
    for h in HORIZONS_MIN:
        fs = pd.read_parquet(_FG / f"forecast_set_h{h}.parquet")
        fs["entry_ts"] = pd.to_datetime(fs["ts"])
        if fs["entry_ts"].dt.tz is None:
            fs["entry_ts"] = fs["entry_ts"].dt.tz_localize(NY)
        absf_all = fs["forecast_vx_pts"].abs().to_numpy()  # full scoreable set
        thr_per_h[h] = {pct: float(np.quantile(absf_all, 1.0 - pct)) for pct in PCTILE_GRID}
        fs = fs[fs["vx_ask"] > fs["vx_bid"]].copy()  # valid book (as in the backtest)
        absf = fs["forecast_vx_pts"].abs().to_numpy()
        fired_per_h[h] = fs[absf >= thr_per_h[h][max(PCTILE_GRID)]].copy()  # universe = widest gate

    # Date-batch the union of all top-20% signals across horizons.
    by_date: dict[pd.Timestamp, list] = defaultdict(list)
    for h, fs in fired_per_h.items():
        for idx, r in fs.iterrows():
            by_date[pd.Timestamp(r["entry_ts"].date())].append((h, idx, r["entry_ts"], r["direction"]))
    dates = sorted(by_date.keys())
    if max_dates is not None:
        dates = dates[:max_dates]

    # Execute each signal once; key (h, idx) -> {filled, pnl}.
    res: dict[tuple, dict] = {}
    for date in dates:
        ev = load_session_events("VX", date, levels=5)
        if ev is None:
            print(f"  [{date.date()}] no VX session - skip {len(by_date[date])}", flush=True)
            continue
        t0 = time.time()
        for (h, idx, ets, direction) in by_date[date]:
            rec = cancel_aware_trade(ev, ets, direction, h)
            res[(h, idx)] = {"filled": rec["filled"], "pnl": rec["pnl_dollars"]}
        print(f"  [{date.date()}] {len(by_date[date]):>5} sigs  ev={len(ev):,}  {time.time()-t0:.1f}s", flush=True)
        del ev

    rows, trade_rows = [], []
    for h in HORIZONS_MIN:
        fs = fired_per_h[h].copy()
        # attach execution result + only keep signals we actually executed (date-limited smoke safe)
        fs["filled"] = [res.get((h, idx), {}).get("filled", np.nan) for idx in fs.index]
        fs["pnl"] = [res.get((h, idx), {}).get("pnl", np.nan) for idx in fs.index]
        fs = fs[fs["filled"].notna()].copy()
        for _, tr in fs.iterrows():
            trade_rows.append({"h_min": h, "entry_ts": tr["entry_ts"], "forecast_vx_pts": tr["forecast_vx_pts"],
                               "realized": tr["realized"], "filled": bool(tr["filled"]), "pnl": tr["pnl"]})
        absf = fs["forecast_vx_pts"].abs().to_numpy()
        for pct in PCTILE_GRID:
            thr = thr_per_h[h][pct]
            cell = fs[absf >= thr].copy()
            sgn_f = np.sign(cell["forecast_vx_pts"]); sgn_r = np.sign(cell["realized"])
            fav = (sgn_f == sgn_r) & (cell["realized"] != 0)
            adv = (sgn_f == -sgn_r) & (cell["realized"] != 0)
            filledm = cell["filled"].astype(bool)
            fcell = cell[filledm]
            tfav = (np.sign(fcell["forecast_vx_pts"]) == np.sign(fcell["realized"])) & (fcell["realized"] != 0)
            tadv = (np.sign(fcell["forecast_vx_pts"]) == -np.sign(fcell["realized"])) & (fcell["realized"] != 0)
            fcell = fcell.assign(date=pd.DatetimeIndex(fcell["entry_ts"]).tz_convert(NY).date)
            rows.append({
                "h_min": h, "top_pct": pct, "n_fire": int(len(cell)),
                "n_filled": int(filledm.sum()),
                "p_fill_adv": float(cell.loc[adv, "filled"].astype(bool).mean()) if adv.any() else float("nan"),
                "p_fill_fav": float(cell.loc[fav, "filled"].astype(bool).mean()) if fav.any() else float("nan"),
                "pnl_adv": float(fcell.loc[tadv, "pnl"].mean()) if tadv.any() else float("nan"),
                "pnl_fav": float(fcell.loc[tfav, "pnl"].mean()) if tfav.any() else float("nan"),
                "total": float(fcell["pnl"].sum()),
                "sharpe": _sharpe(fcell[["date", "pnl"]]),
            })
    df = pd.DataFrame(rows)
    df["ratio"] = df["p_fill_adv"] / df["p_fill_fav"]
    return df, pd.DataFrame(trade_rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dates", type=int, default=None)
    args = ap.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)
    df, trades = run(args.max_dates)
    pd.set_option("display.width", 220)
    cols = ["h_min", "top_pct", "n_fire", "n_filled", "p_fill_adv", "p_fill_fav", "ratio", "pnl_adv", "pnl_fav", "total", "sharpe"]
    print("\n=== Section 7 percentile-gated cancel-aware trading ===")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.max_dates is None:
        df.to_parquet(_OUT / "pct_trading.parquet", index=False)
        trades.to_parquet(_OUT / "pct_trades.parquet", index=False)  # per-signal, for re-aggregation
        print(f"\n[pct-trading] wrote {_OUT / 'pct_trading.parquet'} + pct_trades.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
