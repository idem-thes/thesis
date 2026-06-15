"""Driver - calibrate (or load), iterate over the frozen 538-session window, compute all methods.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ._paths import OUTPUTS_ROOT
from .baseline import run_baseline_day
from .calibration import init_R_state, load_or_calibrate
from .data import (
    load_daily_yf,
    load_hourly_spx_local,
    load_session_anchors_csv,
    session_bars_from_csv,
)
from .intraday import run_intraday_day
from .multi_asset import (
    load_or_fit_multiasset,
    predict_multiasset_day,
    prepare_foreign_returns,
)

log = logging.getLogger(__name__)


def main(refresh_calib: bool = False) -> pd.DataFrame:
    """Run the per-day loop. Returns the residuals DataFrame and writes a parquet copy."""
    params = load_or_calibrate(refresh=refresh_calib)
    log.info(
        "params loaded: %s",
        {k: f"{v:.4f}" for k, v in params.items() if isinstance(v, float)},
    )

    spx_d = load_daily_yf("^GSPC", "1989-01-01", str(_dt.date.today()))
    daily_returns = spx_d["Close"].pct_change().dropna()
    spx_h = load_hourly_spx_local()
    anchors = load_session_anchors_csv()
    anchors_by_date = anchors.set_index("date")

    multi_params = load_or_fit_multiasset(params, refresh=refresh_calib)
    log.info(
        "multi-asset betas: %s",
        {k: f"{v:.6f}" for k, v in multi_params.items() if isinstance(v, float)},
    )
    foreign_returns = prepare_foreign_returns(
        "1989-01-01", str(_dt.date.today()), spx_calendar=daily_returns.index
    )

    common_dates = sorted(set(anchors["date"]) & set(d for d in spx_h.index.date))
    log.info(
        "common sessions: %d (range %s -> %s)",
        len(common_dates),
        common_dates[0],
        common_dates[-1],
    )

    # Previous-session anchors come from the CSV (calendar-consistent with today's row).
    vix_close_eod = anchors_by_date["vix_close"]
    spx_close_eod = (
        spx_h[spx_h.index.hour == 16]
        .groupby(spx_h[spx_h.index.hour == 16].index.date)["Close"]
        .first()
    )

    rows = []
    for d in common_dates:
        sb = session_bars_from_csv(d, spx_h, anchors_by_date.loc[d])
        if sb is None:
            rows.append({"date": d, "status": "missing_bars"})
            continue
        try:
            state_pre = init_R_state(daily_returns, pd.Timestamp(d), params)
            # Standard 2EXP fit (Guyon empirical_study.ipynb): R_n through CLOSE of day d.
            # `init_R_state` filters with `< asof`, so passing d+1 day includes day d's return.
            state_today_inclusive = init_R_state(
                daily_returns, pd.Timestamp(d) + pd.Timedelta(days=1), params
            )
        except ValueError:
            rows.append({"date": d, "status": "no_history"})
            continue
        # CSV anchors are already fractional (e.g. 0.1318 for 13.18%) - no /100 needed.
        sigma_open_vix = sb["vix_open"]
        hourly_r = np.diff(sb["spx_closes"]) / sb["spx_closes"][:-1]
        # Method 1: Standard 2EXP fit (Guyon) - end-of-day R_n, no SDE.
        sigma_std = run_baseline_day(state_today_inclusive, params)
        # Model 5: Multi-asset Guyon - sigma^SPX + foreign-index beta*R features.
        sigma_multi = predict_multiasset_day(
            pd.Timestamp(d),
            sigma_spx_only=sigma_std,
            foreign_returns=foreign_returns,
            spx_params=params,
            multi_params=multi_params,
        )
        # Method 2: Intraday fit (VIX-anchored) - sigma_open from VIX 10:00 bar.
        intra_vix = run_intraday_day(dict(state_pre), sigma_open_vix, hourly_r, params)
        # Method 3: Intraday fit (regression-anchored) - sigma_open from baseline regression
        # at start-of-day R_n (no today data in init). Isolates path-dependence from VIX anchor.
        sigma_open_reg = run_baseline_day(state_pre, params)
        intra_reg = run_intraday_day(dict(state_pre), sigma_open_reg, hourly_r, params)

        # Methods that need overnight info - require previous-session SPX/VIX closes.
        prev_dates = [pd_d for pd_d in spx_close_eod.index if pd_d < d]
        intra_overnight = None  # method 3 - yesterday VIX_close anchor + overnight + intraday
        intra_reg_overnight = (
            None  # method 5 - regression sigma_open + overnight + intraday (equal info to standard)
        )
        d_prev = prev_dates[-1] if prev_dates else None
        prev_anchor_ok = (
            d_prev is not None
            and d_prev in vix_close_eod.index
            and pd.notna(vix_close_eod.loc[d_prev])
        )
        # Model 2: VIX random walk - sigma_hat_t = VIX_{t-1}^close. NaN for the first
        # session (no prior anchor in the CSV) and after any missing-bars gap.
        vix_prev_close = float(vix_close_eod.loc[d_prev]) if prev_anchor_ok else float("nan")
        if prev_anchor_ok:
            spx_prev_close = float(spx_close_eod.loc[d_prev])
            r_overnight = (sb["spx_closes"][0] - spx_prev_close) / spx_prev_close
            returns_with_overnight = np.concatenate([[r_overnight], hourly_r])
            # Method 3: anchor to yesterday's VIX_close (already fractional from CSV).
            intra_overnight = run_intraday_day(
                dict(state_pre),
                vix_prev_close,
                returns_with_overnight,
                params,
            )
            # Method 5: anchor to yesterday's regression sigma - pure path-dependence test
            # with same close-to-close information as the standard fit.
            intra_reg_overnight = run_intraday_day(
                dict(state_pre),
                sigma_open_reg,
                returns_with_overnight,
                params,
            )

        rows.append(
            {
                "date": d,
                "vix_open": sb["vix_open"],
                "vix_close": sb["vix_close"],
                "sigma_intra": float(intra_vix["sigma_close"]),
                "sigma_intra_reg": float(intra_reg["sigma_close"]),
                "sigma_intra_overnight": (
                    float(intra_overnight["sigma_close"]) if intra_overnight else float("nan")
                ),
                "sigma_intra_reg_overnight": (
                    float(intra_reg_overnight["sigma_close"])
                    if intra_reg_overnight
                    else float("nan")
                ),
                "sigma_std": float(sigma_std),
                "sigma_rw": vix_prev_close,
                "sigma_multi": float(sigma_multi),
                "neg_count": int(intra_vix["neg_count"])
                + int(intra_reg["neg_count"])
                + (int(intra_overnight["neg_count"]) if intra_overnight else 0)
                + (int(intra_reg_overnight["neg_count"]) if intra_reg_overnight else 0),
                "status": "ok",
            }
        )
    df = pd.DataFrame(rows)
    _persist(df, params)
    return df


def _persist(df: pd.DataFrame, params: dict) -> Path:
    out_dir = OUTPUTS_ROOT / f"{_dt.date.today()}_intraday_vix_fit"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "residuals.parquet")
    import json

    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    log.info("results written to %s", out_dir)
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = main()
    print(df.head())
    print()
    print(f"ok={int((df['status'] == 'ok').sum())}, missing={int((df['status'] != 'ok').sum())}")
    ok = df[df["status"] == "ok"]
    if len(ok):
        for col, label in [
            ("sigma_std", "m1 std         "),
            ("sigma_rw", "m2 vix RW      "),
            ("sigma_intra", "m3 today VIX_op"),
            ("sigma_intra_overnight", "m4 yest VIX+ON "),
            ("sigma_multi", "m5 multi-asset "),
        ]:
            sub = ok.dropna(subset=[col])
            rmse = ((sub[col] - sub["vix_close"]) ** 2).mean() ** 0.5
            print(f"{label} RMSE: {rmse * 100:.3f}% (n={len(sub)})")
        print(f"neg_count total: {int(ok['neg_count'].sum())} (across {len(ok)} sessions)")
