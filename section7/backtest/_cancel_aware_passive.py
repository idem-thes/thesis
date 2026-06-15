"""Section 7.1 trading re-run under Section 8's cancel-aware honest maker ."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from code_section7.backtest.passive_sandbox import (  # noqa: E402
    CFE_FEE_RT,
    HS,
    VX_MULTIPLIER,
    XS,
    _snapshot_at,
    find_take_profit_exit,
    taker_exit_px,
)
from code_section7.data_loader import load_session_events  # noqa: E402
from code_section7.lob.backtest import (  # noqa: E402
    simulate_passive_entry_until_cancel_aware,
)

TRADES_DIR = REPO_ROOT / "outputs" / "section7" / "backtest" / "sandbox_fine"
OUT_CACHE = REPO_ROOT / "outputs" / "_cache" / "section7_passive_cancel_aware"
NY = "America/New_York"


def _trade_parquet(x: float, h: int) -> pathlib.Path:
    return TRADES_DIR / f"m2_full_grid_trades_x{x}_h{h}.parquet"


def _slice_for_trade(ev: pd.DataFrame, entry_ts: pd.Timestamp, deadline: pd.Timestamp) -> pd.DataFrame:
    """Contiguous [entry-snapshot, deadline] slice (positional, Section 8 _maker_trading idiom).

    The execution fns address events positionally (searchsorted + .iloc), so a
    contiguous slice containing the entry snapshot row through the last row <=
    deadline is identical to the full-day frame but far faster.
    """
    tse = ev["ts_event"]
    start_g = int(tse.searchsorted(entry_ts, side="right"))
    end_g = int(tse.searchsorted(deadline, side="right"))
    return ev.iloc[max(0, start_g - 2) : max(end_g, start_g) + 1]


def _mid_at(events: pd.DataFrame, ts: pd.Timestamp) -> float:
    snap = _snapshot_at(events, ts)
    if snap is None:
        return float("nan")
    bid, ask = float(snap["bid_px_00"]), float(snap["ask_px_00"])
    if not (np.isfinite(bid) and np.isfinite(ask)):
        return float("nan")
    return 0.5 * (bid + ask)


def cancel_aware_trade(
    events: pd.DataFrame,
    entry_ts: pd.Timestamp,
    direction: Literal["long", "short"],
    h_minutes: int,
) -> dict:
    """One Section 7 signal through Section 8's cancel-aware maker; full per-trade record."""
    deadline = entry_ts + pd.Timedelta(minutes=int(h_minutes))
    ev = _slice_for_trade(events, entry_ts, deadline)

    filled, fill_ts, fill_px, q0 = simulate_passive_entry_until_cancel_aware(
        ev, entry_ts, direction, deadline, queue_position="back", fill_on="trade"
    )
    row = {
        "entry_ts": entry_ts,
        "direction": direction,
        "filled": bool(filled),
        "queue_initial": q0,
        "fill_ts": fill_ts,
        "fill_px": fill_px,
        "exit_ts": None,
        "exit_px": None,
        "exit_reason": None,
        "pnl_dollars": 0.0,
        "realized_move": _mid_at(events, entry_ts + pd.Timedelta(minutes=int(h_minutes)))
        - _mid_at(events, entry_ts),
    }
    if filled and fill_px is not None:
        tp = find_take_profit_exit(ev, fill_ts, deadline, direction, fill_px)
        if tp is not None:
            exit_ts, px, reason = tp
        else:
            exit_ts, px, reason = deadline, taker_exit_px(ev, deadline, direction), "signal_time_force"
        if px is not None:
            pnl_pts = (px - fill_px) if direction == "long" else (fill_px - px)
            row["exit_ts"], row["exit_px"], row["exit_reason"] = exit_ts, px, reason
            row["pnl_dollars"] = pnl_pts * VX_MULTIPLIER - CFE_FEE_RT
        else:
            row["filled"] = False
    return row


def run(cells: list[tuple[float, int]], max_dates: int | None) -> tuple[pd.DataFrame, dict]:
    signals: dict[tuple[float, int], pd.DataFrame] = {}
    for x, h in cells:
        p = _trade_parquet(x, h)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["entry_ts"] = pd.to_datetime(df["entry_ts"])
        if df["entry_ts"].dt.tz is None:
            df["entry_ts"] = df["entry_ts"].dt.tz_localize(NY)
        signals[(x, h)] = df

    by_date: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for (x, h), df in signals.items():
        for pos, r in df.iterrows():
            by_date[pd.Timestamp(r["entry_ts"].date())].append(
                {"x": x, "h": h, "entry_ts": r["entry_ts"], "direction": r["direction"]}
            )

    dates = sorted(by_date.keys())
    if max_dates is not None:
        dates = dates[:max_dates]

    per_cell: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for date in dates:
        events = load_session_events("VX", date, levels=5)
        if events is None:
            print(f"  [{date.date()}] no VX session - skipping {len(by_date[date])} entries", flush=True)
            continue
        t0 = time.time()
        for e in by_date[date]:
            rec = cancel_aware_trade(events, e["entry_ts"], e["direction"], e["h"])
            per_cell[(e["x"], e["h"])].append(rec)
        print(f"  [{date.date()}] {len(by_date[date]):>5} entries  events={len(events):,}  {time.time()-t0:.1f}s", flush=True)
        del events

    summary_rows, trade_dfs = [], {}
    for (x, h), rows in per_cell.items():
        pdf = pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)
        trade_dfs[(x, h)] = pdf
        n_sig, n_fill = len(pdf), int(pdf["filled"].sum())
        summary_rows.append(
            {
                "threshold_x": x,
                "forecast_window_min": h,
                "n_signals": n_sig,
                "n_filled": n_fill,
                "fill_rate": (n_fill / n_sig) if n_sig else np.nan,
                "passive_net": float(pdf.loc[pdf["filled"], "pnl_dollars"].sum()),
            }
        )
    return pd.DataFrame(summary_rows), trade_dfs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="comma list of x:h (default: full XS*HS grid)")
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--write", action="store_true", help="write cache (default on for full run)")
    args = ap.parse_args(argv)

    if args.cells:
        cells = [(float(c.split(":")[0]), int(c.split(":")[1])) for c in args.cells.split(",")]
    else:
        cells = [(x, h) for x in XS for h in HS]

    print(f"[cancel-aware] {len(cells)} cells, max_dates={args.max_dates}", flush=True)
    summary, trade_dfs = run(cells, args.max_dates)
    pd.set_option("display.width", 200)
    print("\n=== cancel-aware summary ===")
    print(summary.sort_values(["forecast_window_min", "threshold_x"]).to_string(index=False))

    if args.write or (not args.cells and args.max_dates is None):
        OUT_CACHE.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUT_CACHE / "summary.csv", index=False)
        for (x, h), df in trade_dfs.items():
            if not df.empty:
                df.to_parquet(OUT_CACHE / f"trades_x{x}_h{h}.parquet")
        print(f"\n[cancel-aware] wrote cache -> {OUT_CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
