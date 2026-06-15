"""Section 7.2 backtest driver.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root (4 levels up from this file: backtest/run.py -> backtest/ -> code_section7/ -> repo)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THETA_PATH = _PROJECT_ROOT / "outputs" / "_cache" / "section7_backtest_theta_hat.json"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "section7" / "backtest"


def _git_sha() -> str:
    """Return current git HEAD short SHA, or 'unknown' on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit theta_hat on FRD pre-Databento window; persist JSON."""
    from code_section7.backtest.calibrate import (
        fit_one_shot_theta,
        persist_theta_hat,
        prepare_calibration_arrays,
    )

    start = args.start
    end = args.end
    n_mask_days = args.n_mask_days
    out_path = Path(args.output) if args.output else _DEFAULT_THETA_PATH

    print(f"[calibrate] preparing arrays {start} -> {end} (n_mask_days={n_mask_days}) ...")
    arrays = prepare_calibration_arrays(start=start, end=end, n_mask_days=n_mask_days)
    print(f"[calibrate] fitting M2 on {len(arrays['S'])} daily bars ... (may take 30-120s)")
    theta_dict = fit_one_shot_theta(arrays)

    metadata = {
        "calibration_start": start,
        "calibration_end": end,
        "n_mask_days": n_mask_days,
        "git_sha": _git_sha(),
        "fit_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    persist_theta_hat(theta_dict, out_path, metadata)
    print(f"[calibrate] wrote theta_hat -> {out_path}")
    print(f"[calibrate]   theta_hat = {theta_dict['theta_hat']}")
    print(
        f"[calibrate]   train_r2 = {theta_dict.get('train_r2'):.4f}, test_r2 = {theta_dict.get('test_r2'):.4f}"
    )
    return 0


def _load_databento_window(
    start: str,
    end: str,
    init_price: float | None = None,
) -> pd.DataFrame:
    """Load ES + VX 1-min Databento data on an ES spine.
    """
    from code_section7.backtest.data import (
        load_databento_1min,
        load_spx_1min,
        splice_es_roll,
    )

    print(f"[data] loading Databento ES 1-min {start} -> {end} ...")
    es = load_databento_1min("ES", start, end)
    print(f"[data] loading Databento VX 1-min {start} -> {end} ...")
    vx = load_databento_1min("VX", start, end)
    print("[data] loading SPX 1-min for ES-roll wall-clock bridge ...")
    spx = load_spx_1min(start, end)

    es_spliced = splice_es_roll(es, spx)

    # ES spine: left-join VX, ffill sigma-anchor, leave trading quotes NaN when VX closed.
    es_idx = es_spliced.set_index("ts")
    vx_idx = vx.set_index("ts")
    merged = es_idx[["return_1min", "mid_close"]].rename(
        columns={"mid_close": "es_mid"}
    ).join(
        vx_idx[["mid_close", "bid_close", "ask_close"]].rename(
            columns={"mid_close": "vx_mid", "bid_close": "vx_bid", "ask_close": "vx_ask"}
        ),
        how="left",
    )
    merged["vx_mid"] = merged["vx_mid"].ffill()

    # Bridge the first bar's return from the warmup-end reference price.
    if init_price is not None and len(merged) > 0:
        first_idx = merged.index[0]
        first_es = float(merged.loc[first_idx, "es_mid"])
        if np.isfinite(first_es) and init_price > 0:
            merged.loc[first_idx, "return_1min"] = float(np.log(first_es) - np.log(init_price))

    return merged


def cmd_smoke(args: argparse.Namespace) -> int:
    """Fast 5-day, single-cell sanity check on real Databento data."""
    from code_section7.backtest.calibrate import load_theta_hat
    from code_section7.backtest.data import load_vx_settlement_dates
    from code_section7.backtest.strategy import run_strategy

    theta_path = Path(args.theta) if args.theta else _DEFAULT_THETA_PATH
    if not theta_path.exists():
        print(f"[smoke] ERROR: theta JSON not found at {theta_path}")
        print("[smoke]        run `python -m code_section7.backtest.run calibrate` first")
        return 2
    saved = load_theta_hat(theta_path)
    theta = saved["theta"]

    start = args.start
    end = args.end
    data = _load_databento_window(start, end)
    print(f"[smoke] merged data shape: {data.shape}")
    if data.empty:
        print("[smoke] ERROR: empty data; window may be outside Databento cache")
        return 3

    settlement_dates = load_vx_settlement_dates()
    hp = {
        "threshold_x": 0.5,
        "forecast_window_min": 30,
        "start_hour": (9, 30),
        "close_hour": (16, 0),
        "n_mask_days": 2,
    }
    print(f"[smoke] running strategy (method={args.method}) with hp={hp} ...")
    trades = run_strategy(
        theta=theta,
        method=args.method,
        data=data,
        hyperparams=hp,
        settlement_dates=settlement_dates,
    )
    print(f"[smoke] produced {len(trades)} trades")
    if trades:
        print(f"[smoke]   first trade: {trades[0]}")
        print(f"[smoke]   last trade: {trades[-1]}")
        total_pnl = sum(t.pnl_dollars for t in trades if t.pnl_dollars is not None)
        print(f"[smoke]   total $PnL = {total_pnl:.2f}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward + final OOS for one method on the full Databento window."""
    from code_section7.backtest.calibrate import load_theta_hat
    from code_section7.backtest.data import load_vx_settlement_dates
    from code_section7.backtest.walkforward import final_oos, walk_forward

    theta_path = Path(args.theta) if args.theta else _DEFAULT_THETA_PATH
    if not theta_path.exists():
        print(f"[backtest] ERROR: theta JSON not found at {theta_path}")
        return 2
    saved = load_theta_hat(theta_path)
    theta = saved["theta"]

    data = _load_databento_window(args.start, args.end)
    if data.empty:
        print("[backtest] ERROR: empty merged data")
        return 3
    settlement_dates = load_vx_settlement_dates()

    print(f"[backtest] walk-forward (method={args.method}) ...")
    wf = walk_forward(
        theta=theta,
        method=args.method,
        data=data,
        settlement_dates=settlement_dates,
        oos_days=args.oos_days,
        initial_capital=args.capital,
    )
    print(f"[backtest] n_folds = {wf['n_folds']}")
    print(
        f"[backtest] chosen top-{wf['top_k_used']} HPs "
        f"(rank-based, top_k_per_fold={wf['top_k_per_fold_used']}, "
        f"min_trades_per_fold={wf['min_trades_per_fold_used']}):"
    )
    for i, hp in enumerate(wf["chosen_hp_list"]):
        # Look up the ranking info
        info = next((r for r in wf["hp_rankings"] if r["hp"] == hp), None)
        print(
            f"  [{i + 1}] {hp}  top_count={info['top_k_per_fold_count']}, "
            f"mean_test_sharpe={info['mean_test_sharpe']:.3f}, "
            f"n_valid_folds={info['n_valid_folds']}"
        )

    if not wf["chosen_hp_list"]:
        print("[backtest] no HP qualified for OOS (insufficient trades across all folds); aborting")
        return 4

    print("[backtest] final OOS (bagged across top-K HPs) ...")
    oos = final_oos(
        theta=theta,
        method=args.method,
        data=data,
        settlement_dates=settlement_dates,
        chosen_hp=wf["chosen_hp_list"],
        oos_days=args.oos_days,
        initial_capital=args.capital,
    )
    summary_json_safe = {
        "summary_flat": oos["summary_flat"],
        "bagged_sharpe": oos["bagged_sharpe"],
        "bagged_total_return": oos["bagged_total_return"],
        "bagged_max_drawdown": oos["bagged_max_drawdown"],
        "n_hps_bagged": oos["n_hps_bagged"],
        "oos_start_date": str(oos["oos_start_date"].date()),
        "oos_end_date": str(oos["oos_end_date"].date()),
        "hps_used": oos["hps_used"],
        "per_hp_summaries": [
            {"hp": r["hp"], "summary": r["summary"]} for r in oos["per_hp_results"]
        ],
    }
    print(f"[backtest] OOS summary: {json.dumps(summary_json_safe, indent=2, default=str)}")

    # Save trades + summary to parquet/json
    out_dir = Path(args.output) if args.output else _DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(
        [
            {
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "direction": t.direction,
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "pnl_pts": t.pnl_pts,
                "pnl_dollars": t.pnl_dollars,
                "forecast_vx_pts": t.forecast_vx_pts,
                "hp_threshold_x": t.hyperparams.get("threshold_x"),
                "hp_forecast_window_min": t.hyperparams.get("forecast_window_min"),
            }
            for t in oos["trades"]
        ]
    )
    trades_df.to_parquet(out_dir / f"{args.method}_oos_trades.parquet")
    with open(out_dir / f"{args.method}_oos_summary.json", "w") as f:
        json.dump(summary_json_safe, f, indent=2, default=str)

    # Walk-forward rankings (full hp_rankings table)
    rankings_df = pd.DataFrame(
        [
            {
                "threshold_x": r["hp"]["threshold_x"],
                "forecast_window_min": r["hp"]["forecast_window_min"],
                "start_hour": str(r["hp"]["start_hour"]),
                "close_hour": str(r["hp"]["close_hour"]),
                "n_mask_days": r["hp"]["n_mask_days"],
                "top_k_per_fold_count": r["top_k_per_fold_count"],
                "mean_test_sharpe": r["mean_test_sharpe"],
                "n_valid_folds": r["n_valid_folds"],
            }
            for r in wf["hp_rankings"]
        ]
    )
    rankings_df.to_csv(out_dir / f"{args.method}_walkforward_rankings.csv", index=False)

    print(
        f"[backtest] wrote {out_dir}/{args.method}_oos_*.parquet|.json + walkforward_rankings.csv"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print summary tables from saved OOS outputs."""
    out_dir = Path(args.output) if args.output else _DEFAULT_OUTPUT_DIR
    for method in ("M2", "M4"):
        summary_path = out_dir / f"{method}_oos_summary.json"
        if not summary_path.exists():
            print(f"[report] {method}: no summary at {summary_path}")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n=== {method} OOS summary ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="section7-backtest", description=__doc__)
    subs = parser.add_subparsers(dest="subcommand", required=True)

    p_cal = subs.add_parser("calibrate", help="Fit one-shot theta_hat on FRD pre-Databento window")
    p_cal.add_argument("--start", default="2008-07-10")
    p_cal.add_argument("--end", default="2025-02-28")
    p_cal.add_argument("--n-mask-days", type=int, default=2)
    p_cal.add_argument(
        "--output",
        help="Output JSON path (default: outputs/_cache/section7_backtest_theta_hat.json)",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_sm = subs.add_parser("smoke", help="Fast single-cell sanity on Databento data")
    p_sm.add_argument("--start", default="2025-03-03")
    p_sm.add_argument("--end", default="2025-03-07")
    p_sm.add_argument("--method", choices=["M2", "M4"], default="M2")
    p_sm.add_argument("--theta", help="Path to theta_hat JSON (default: cache)")
    p_sm.set_defaults(func=cmd_smoke)

    p_bt = subs.add_parser("backtest", help="Walk-forward + final OOS for one method")
    p_bt.add_argument("--start", default="2025-03-01")
    p_bt.add_argument("--end", default="2025-05-31")
    p_bt.add_argument("--method", choices=["M2", "M4"], required=True)
    p_bt.add_argument("--theta", help="Path to theta_hat JSON (default: cache)")
    p_bt.add_argument("--oos-days", type=int, default=10)
    p_bt.add_argument(
        "--capital",
        type=float,
        default=5000.0,
        help="Initial capital ($); placeholder for CME VX margin",
    )
    p_bt.add_argument("--output", help="Output directory (default: outputs/section7/backtest)")
    p_bt.set_defaults(func=cmd_backtest)

    p_rp = subs.add_parser("report", help="Print saved OOS summaries for M2 and M4")
    p_rp.add_argument("--output", help="Backtest output dir (default: outputs/section7/backtest)")
    p_rp.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
