"""Section 8 run-2 maker trading performance - recompute from committed OOS predictions.

"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from code_section7.data_loader import load_session_events
from code_section8_run2 import execution as ex
from code_section8_run2._smoke import HORIZONS_S, PCTILE_GRID

_OUT = Path("outputs/_results/section8_run2")
_CAPITAL = 10_000.0  # same $10k margin base as the Section 7 Sharpe table
_TRADING_DAYS = 252


def _sharpe_daily(per_trade: pd.DataFrame) -> float:
    """Section 7-convention annualised Sharpe: daily $ / capital, mean/std(ddof=1)*sqrt 252."""
    if per_trade.empty:
        return float("nan")
    daily = per_trade.groupby("date")["pnl"].sum() / _CAPITAL
    if len(daily) < 2:
        return float("nan")
    sd = float(daily.std(ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return float("nan")
    return float(daily.mean()) / sd * math.sqrt(_TRADING_DAYS)


def _slice_for_trade(ev: pd.DataFrame, entry_ts: pd.Timestamp, horizon_s: int) -> pd.DataFrame:
    """Contiguous [entry-snapshot, deadline] window of the day's events.

    The execution functions address events purely positionally (searchsorted +
    .iloc, no index/.loc), so a contiguous slice that contains the entry snapshot
    row (last ts_event <= entry) through the last row <= deadline yields results
    identical to passing the full 2.5M-row day frame - but ~100x faster, since the
    sim re-extracts every column to numpy on each call. Two extra leading rows are
    harmless (the fill loop starts after the entry snapshot).
    """
    deadline = entry_ts + pd.Timedelta(seconds=int(horizon_s))
    tse = ev["ts_event"]
    start_g = int(tse.searchsorted(entry_ts, side="right"))
    end_g = int(tse.searchsorted(deadline, side="right"))
    return ev.iloc[max(0, start_g - 2) : max(end_g, start_g)]


def maker_stats(
    oos: pd.DataFrame, horizon_s: int, pct: float, ev_by_day: dict
) -> tuple[dict, pd.DataFrame]:
    absp = oos["y_pred"].abs().to_numpy()
    thr = float(np.quantile(absp, 1.0 - pct))
    fired = oos[absp >= thr]

    n_fire = 0
    trades = []  # filled maker trades: {block_ts, date, pnl}
    for _, r in fired.iterrows():
        if float(r["vx_ask"]) <= float(r["vx_bid"]):
            continue
        ev = ev_by_day[r["date"]]
        if ev is None:
            continue
        n_fire += 1
        direction = "long" if r["y_pred"] > 0 else "short"
        ev_slice = _slice_for_trade(ev, r["block_ts"], horizon_s)
        pnl = ex.maker_block_pnl(ev_slice, r["block_ts"], direction, horizon_s)
        if pnl is None:
            continue  # never filled
        trades.append({"block_ts": r["block_ts"], "date": r["date"].date(), "pnl": float(pnl)})

    tdf = pd.DataFrame(trades)
    n_filled = len(tdf)
    n_fav = int((tdf["pnl"] > 0).sum()) if n_filled else 0
    maker_total = float(tdf["pnl"].sum()) if n_filled else 0.0
    stat = {
        "horizon_s": horizon_s,
        "top_pct": pct,
        "n_fire": n_fire,
        "n_filled": n_filled,
        "n_fav": n_fav,
        "p_fill_given_signal": (n_filled / n_fire) if n_fire else float("nan"),
        "p_fav_given_signal": (n_fav / n_fire) if n_fire else float("nan"),
        "p_fav_given_fill": (n_fav / n_filled) if n_filled else float("nan"),
        "maker_total": round(maker_total, 1),
        "sharpe": _sharpe_daily(tdf),
    }
    return stat, tdf


def main() -> int:
    committed = pd.read_parquet(_OUT / "conviction_pnl.parquet")

    oos_by_h = {}
    all_dates = set()
    for h in HORIZONS_S:
        o = pd.read_parquet(_OUT / f"oos_predictions_h{h}s.parquet")
        o["date"] = pd.DatetimeIndex(o["block_ts"]).tz_convert("America/New_York").normalize()
        oos_by_h[h] = o
        all_dates.update(o["date"].unique())

    # Decode each VX session once, share across all (h, pct) recomputes.
    print(f"[maker-trading] loading VX events for {len(all_dates)} sessions...", flush=True)
    ev_by_day = {d: load_session_events("VX", pd.Timestamp(d)) for d in sorted(all_dates)}

    rows = []
    trades_by_key = {}
    for h in HORIZONS_S:
        for pct in PCTILE_GRID:
            stat, tdf = maker_stats(oos_by_h[h], h, pct, ev_by_day)
            rows.append(stat)
            trades_by_key[(h, pct)] = tdf
    df = pd.DataFrame(rows)

    # Faithfulness check vs committed conviction_pnl maker totals.
    chk = committed.merge(
        df[["horizon_s", "top_pct", "maker_total"]], on=["horizon_s", "top_pct"]
    )
    chk["abs_diff"] = (chk["maker"] - chk["maker_total"]).abs()
    max_diff = float(chk["abs_diff"].max())
    print("[maker-trading] recompute vs committed conviction_pnl maker $:")
    print(chk[["horizon_s", "top_pct", "maker", "maker_total", "abs_diff"]].to_string(index=False))
    print(f"[maker-trading] max abs diff = ${max_diff:.2f}")
    assert max_diff < 1.0, "recomputed maker total diverges from committed conviction_pnl"

    df.to_parquet(_OUT / "maker_trading.parquet", index=False)

    trade_frames = []
    for (h, pct), tdf in trades_by_key.items():
        if not tdf.empty:
            t = tdf.copy()
            t["horizon_s"], t["top_pct"] = h, pct
            trade_frames.append(t)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_parquet(
            _OUT / "maker_trades.parquet", index=False
        )

    pd.set_option("display.width", 200)
    print("\n[maker-trading] table:")
    print(
        df[
            [
                "horizon_s",
                "top_pct",
                "n_fire",
                "n_filled",
                "p_fill_given_signal",
                "p_fav_given_signal",
                "p_fav_given_fill",
                "sharpe",
                "maker_total",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print(f"\n[maker-trading] wrote {_OUT / 'maker_trading.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
