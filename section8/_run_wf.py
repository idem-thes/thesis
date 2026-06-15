"""Section 8 run-2 verdict: full 6-month sliding 7/1/1 walk-forward + leakage control.

"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from code_section7.backtest.calibrate import load_theta_hat
from code_section7.backtest.data import load_vx_settlement_dates
from code_section8_run2._smoke import HORIZONS_S, PCTILE_GRID, PDV_COLS, conviction_pnl
from code_section8_run2.dataset import build_dataset
from code_section8_run2.metrics import directional_accuracy, forecast_report
from code_section8_run2.walkforward import (
    _DEPTH_GRID,
    _fit_select_depth,
    _row_dates,
    _slice_rows,
    run_regression_wf,
    sliding_folds,
)

_THETA = "outputs/_cache/section7_backtest_theta_hat.json"
_BLOCKS = ["ES", "VX", "PDV", "time"]


def _block_of(c: str) -> str:
    if c in ("drift_R1", "drift_R2"):
        return "PDV"
    if c == "progress_rth":
        return "time"
    if c.startswith("es_"):
        return "ES"
    return "VX"  # vx_*


def block_importance(ds: dict, horizons: list[int], settle: pd.DatetimeIndex) -> pd.DataFrame:
    """Per horizon: fold-averaged CatBoost importance grouped into ES/VX/PDV/time (%)."""
    tdays = pd.DatetimeIndex(sorted(set(_row_dates(ds["X"].index))))
    folds = sliding_folds(tdays, settle)
    rows = []
    for h in horizons:
        y = ds["targets"][h]
        per = {b: [] for b in _BLOCKS}
        for fold in folds:
            tr = _slice_rows(ds, fold["train"])
            va = _slice_rows(ds, fold["val"])
            ytr = y.loc[tr].dropna()
            if len(ytr) == 0:
                continue
            Xtr = ds["X"].loc[ytr.index]
            yva = y.loc[va].dropna()
            Xval = ds["X"].loc[yva.index] if len(yva) > 0 else None
            m = _fit_select_depth(
                Xtr,
                ytr,
                Xval,
                yva if len(yva) > 0 else None,
                depths=_DEPTH_GRID,
                early_stopping_rounds=50,
                seed=20260525,
            )
            imp = np.asarray(m.get_feature_importance(), float)
            pb = defaultdict(float)
            for nm, v in zip(Xtr.columns, imp):
                pb[_block_of(nm)] += v
            s = sum(pb.values()) or 1.0
            for b in _BLOCKS:
                per[b].append(100.0 * pb.get(b, 0.0) / s)
        rows.append(
            {"horizon_s": h, **{b: (float(np.mean(per[b])) if per[b] else 0.0) for b in _BLOCKS}}
        )
    return pd.DataFrame(rows)


def feature_importance(ds: dict, horizons: list[int], settle: pd.DatetimeIndex) -> pd.DataFrame:
    """Per horizon: fold-averaged CatBoost importance per feature (raw + pct).

    Mirrors block_importance() but without the ES/VX/PDV/time grouping. Returns
    a long-format DataFrame with columns (horizon_s, feature, importance_raw,
    importance_pct), sorted by (horizon_s asc, importance_raw desc).
    """
    tdays = pd.DatetimeIndex(sorted(set(_row_dates(ds["X"].index))))
    folds = sliding_folds(tdays, settle)
    rows = []
    for h in horizons:
        y = ds["targets"][h]
        sums: dict[str, list[float]] = {c: [] for c in ds["X"].columns}
        for fold in folds:
            tr = _slice_rows(ds, fold["train"])
            va = _slice_rows(ds, fold["val"])
            ytr = y.loc[tr].dropna()
            if len(ytr) == 0:
                continue
            Xtr = ds["X"].loc[ytr.index]
            yva = y.loc[va].dropna()
            Xval = ds["X"].loc[yva.index] if len(yva) > 0 else None
            m = _fit_select_depth(
                Xtr,
                ytr,
                Xval,
                yva if len(yva) > 0 else None,
                depths=_DEPTH_GRID,
                early_stopping_rounds=50,
                seed=20260525,
            )
            imp = np.asarray(m.get_feature_importance(), float)
            for nm, v in zip(Xtr.columns, imp):
                sums[nm].append(float(v))
        per_feat = {c: (float(np.mean(vals)) if vals else 0.0) for c, vals in sums.items()}
        total = sum(per_feat.values()) or 1.0
        for nm, v in per_feat.items():
            rows.append(
                {
                    "horizon_s": h,
                    "feature": nm,
                    "importance_raw": v,
                    "importance_pct": 100.0 * v / total,
                }
            )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["horizon_s", "importance_raw"], ascending=[True, False]).reset_index(
            drop=True
        )
    return df


def _plot(fc: pd.DataFrame, pnl: pd.DataFrame, imp: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    if len(fc):
        ax[0].plot(fc["horizon_s"], fc["dir_acc"], "o-", label="real")
        ax[0].plot(fc["horizon_s"], fc["dir_shuffled"], "s--", label="shuffled")
        ax[0].axhline(0.5, ls=":", c="grey")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("horizon (s)")
        ax[0].set_ylabel("dir-acc")
        ax[0].set_title("Forecast skill + shuffle control")
        ax[0].legend()
    if len(imp):
        bottom = np.zeros(len(imp))
        for b in _BLOCKS:
            ax[1].bar(imp["horizon_s"].astype(str), imp[b], bottom=bottom, label=b)
            bottom += imp[b].to_numpy()
        ax[1].set_xlabel("horizon (s)")
        ax[1].set_ylabel("importance %")
        ax[1].set_title("Block importance")
        ax[1].legend()
    if len(pnl):
        for pct, g in pnl.groupby("top_pct"):
            ax[2].plot(g["horizon_s"], g["maker"], "o-", label=f"maker top{int(pct*100)}%")
        ax[2].axhline(0.0, ls=":", c="grey")
        ax[2].set_xscale("log")
        ax[2].set_xlabel("horizon (s)")
        ax[2].set_ylabel("maker PnL $")
        ax[2].set_title("Conviction-gate PnL (does not monetize)")
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "section8_verdict.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="section8-run2-run-wf")
    p.add_argument("--start", default="2025-03-02")
    p.add_argument("--end", default="2025-08-29")
    p.add_argument("--out", default="outputs/_results/section8_run2")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    theta = load_theta_hat(_THETA)["theta"]
    ds = build_dataset(args.start, args.end, HORIZONS_S, theta)
    print(f"[run_wf] dataset rows={len(ds['X'])} cols={ds['X'].shape[1]}", flush=True)
    settle = load_vx_settlement_dates()

    oos_full = run_regression_wf(ds, HORIZONS_S, settle)
    ds_pdv = {**ds, "X": ds["X"][[c for c in PDV_COLS if c in ds["X"].columns]]}
    oos_pdv = run_regression_wf(ds_pdv, HORIZONS_S, settle)
    oos_shuf = run_regression_wf(ds, HORIZONS_S, settle, shuffle_train_seed=12345)

    fc = []
    for h in HORIZONS_S:
        o = oos_full[h]
        if len(o) == 0:
            continue
        o.to_parquet(out / f"oos_predictions_h{h}s.parquet", index=False)
        rw = forecast_report(
            o["y_true"].to_numpy(), o["y_pred"].to_numpy(), baseline=np.zeros(len(o)), lag=1
        )
        pdvp = oos_pdv[h].set_index("block_ts")["y_pred"].reindex(o["block_ts"]).to_numpy()
        pv = forecast_report(o["y_true"].to_numpy(), o["y_pred"].to_numpy(), baseline=pdvp, lag=1)
        sh = oos_shuf[h]
        sh_dir = (
            directional_accuracy(sh["y_pred"].to_numpy(), sh["y_true"].to_numpy())
            if len(sh)
            else float("nan")
        )
        fc.append(
            {
                "horizon_s": h,
                **rw,
                "dm_vs_pdv": pv["dm_stat"],
                "dm_vs_pdv_p": pv["dm_p"],
                "dir_shuffled": sh_dir,
            }
        )
    fc_df = pd.DataFrame(fc)

    pnl_frames = []
    for h in HORIZONS_S:
        if len(oos_full[h]):
            t = conviction_pnl(oos_full[h], h)
            t["horizon_s"] = h
            pnl_frames.append(t)
    pnl_df = pd.concat(pnl_frames, ignore_index=True) if pnl_frames else pd.DataFrame()
    imp_df = block_importance(ds, HORIZONS_S, settle)
    feat_imp_df = feature_importance(ds, HORIZONS_S, settle)

    fc_df.to_parquet(out / "forecast_skill.parquet", index=False)
    pnl_df.to_parquet(out / "conviction_pnl.parquet", index=False)
    imp_df.to_parquet(out / "block_importance.parquet", index=False)
    feat_imp_df.to_parquet(out / "feature_importance.parquet", index=False)
    with open(out / "verdict.json", "w") as fh:
        json.dump(
            {
                "forecast_skill": fc_df.to_dict("records"),
                "conviction_pnl": pnl_df.to_dict("records"),
                "block_importance": imp_df.to_dict("records"),
                "feature_importance": feat_imp_df.to_dict("records"),
                "window": [args.start, args.end],
                "horizons_s": HORIZONS_S,
                "pctile_grid": PCTILE_GRID,
            },
            fh,
            indent=2,
            default=str,
        )
    _plot(fc_df, pnl_df, imp_df, out)
    print("[run_wf] forecast skill:\n", fc_df.to_string(index=False))
    print("[run_wf] block importance:\n", imp_df.to_string(index=False))
    if len(feat_imp_df):
        for h in HORIZONS_S:
            top = feat_imp_df[feat_imp_df["horizon_s"] == h].head(5)
            if not top.empty:
                print(f"[run_wf] top-5 features h={h}s:")
                print(top[["feature", "importance_pct"]].to_string(index=False))
    print(f"[run_wf] wrote artifacts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
