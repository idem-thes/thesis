"""One-off calibration: FRD VX/SPX, longest train, no holdout."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from code_section7.backtest.calibrate import (
    fit_one_shot_theta,
    persist_theta_hat,
    prepare_calibration_arrays,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = _PROJECT_ROOT / "outputs" / "_cache" / "section7_backtest_theta_hat.json"

LOAD_FROM = "2008-07-10"
TRAIN_START = "2012-07-10"
TRAIN_END = "2025-02-28"
N_MASK_DAYS = 2
MAX_DELTA = 1000


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT)
            .decode()
            .strip()[:7]
        )
    except Exception:
        return "unknown"


def main() -> int:
    print(f"[calibrate] preparing arrays {LOAD_FROM} -> {TRAIN_END} (n_mask_days={N_MASK_DAYS}) ...")
    arrays = prepare_calibration_arrays(start=LOAD_FROM, end=TRAIN_END, n_mask_days=N_MASK_DAYS)
    n = len(arrays["S"])
    print(
        f"[calibrate] fitting M2 on {n} daily bars "
        f"(train={TRAIN_START}->{TRAIN_END}, no holdout, max_delta={MAX_DELTA}) ..."
    )
    theta_dict = fit_one_shot_theta(
        arrays,
        train_start_date=TRAIN_START,
        test_start_date=TRAIN_END,
        test_end_date=TRAIN_END,
        max_delta=MAX_DELTA,
    )
    metadata = {
        "source": "FRD VX/SPX daily (calibrate.py path)",
        "load_from": LOAD_FROM,
        "train_start_date": TRAIN_START,
        "train_end_date": TRAIN_END,
        "n_mask_days": N_MASK_DAYS,
        "max_delta": MAX_DELTA,
        "git_sha": _git_sha(),
        "fit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    persist_theta_hat(theta_dict, OUT_PATH, metadata)
    print(f"[calibrate] wrote theta_hat -> {OUT_PATH}")
    print(f"[calibrate]   theta_hat = {theta_dict['theta_hat']}")
    print(f"[calibrate]   param_order = {theta_dict['param_order']}")
    print(
        f"[calibrate]   train_r2 = {theta_dict['train_r2']:.4f}, "
        f"test_r2 = {theta_dict['test_r2']:.4f}"
    )
    print(
        f"[calibrate]   train_rmse = {theta_dict['train_rmse']:.5f}, "
        f"test_rmse = {theta_dict['test_rmse']:.5f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
