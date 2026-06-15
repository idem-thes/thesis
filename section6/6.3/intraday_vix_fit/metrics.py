"""Per-model evaluation metrics for 6.3."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def direction_correct_per_day(
    sigma_hat: pd.Series,
    vix_close: pd.Series,
    vix_prev_close: pd.Series,
) -> pd.Series:
    """1.0 / 0.0 / 0.5 indicator per session (0.5 on either-side-zero ties)."""
    df = pd.concat(
        [
            sigma_hat.rename("sigma_hat"),
            vix_close.rename("vix_close"),
            vix_prev_close.rename("vix_prev_close"),
        ],
        axis=1,
    ).dropna()
    pred_sign = np.sign(df["sigma_hat"] - df["vix_prev_close"])
    actual_sign = np.sign(df["vix_close"] - df["vix_prev_close"])
    only_one_flat = (pred_sign == 0) ^ (actual_sign == 0)  # XOR: exactly one side zero
    is_match = (
        pred_sign == actual_sign
    )  # captures both-zero (correct flat call) and both-equal-non-zero
    return pd.Series(
        np.where(only_one_flat, 0.5, np.where(is_match, 1.0, 0.0)),
        index=df.index,
        name=sigma_hat.name,
    )


def cumulative_direction_correct(per_day: pd.Series) -> pd.Series:
    """Running mean of :func:`direction_correct_per_day` up to each date."""
    counts = np.arange(1, len(per_day) + 1)
    return pd.Series(per_day.cumsum().values / counts, index=per_day.index, name=per_day.name)


def summary_metrics(sigma_hat: pd.Series, vix_close: pd.Series) -> dict:
    """RMSE / MAE / bias / R^2 of sigma_hat vs VIX_close, dropping NaN-aligned rows."""
    df = pd.concat([sigma_hat, vix_close], axis=1).dropna()
    err = df.iloc[:, 0] - df.iloc[:, 1]
    target = df.iloc[:, 1]
    ss_res = float((err**2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(df)),
    }


def all_model_metrics(
    df: pd.DataFrame,
    target_col: str,
    model_cols: Iterable[str],
) -> pd.DataFrame:
    """Stack per-model summary metrics into a tidy DataFrame indexed by model column."""
    rows = []
    for col in model_cols:
        m = summary_metrics(df[col], df[target_col])
        m["model"] = col
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")
