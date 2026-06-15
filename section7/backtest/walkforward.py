"""Section 7.2 walk-forward hyperparameter search and final-OOS evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from code_section7.backtest.metrics import sharpe, summarize, trades_to_daily_returns
from code_section7.backtest.strategy import Trade, run_strategy


_THRESHOLD_GRID: tuple[float, ...] = (0.025, 0.05, 0.1, 0.25, 0.5)
_FORECAST_WINDOW_GRID: tuple[int, ...] = (1, 5, 15, 30)

_START_HOUR_GRID: tuple[tuple[int, int], ...] = ((10, 0),)
_CLOSE_HOUR_GRID: tuple[tuple[int, int], ...] = ((15, 30),)
_N_MASK_DAYS_GRID: tuple[int, ...] = (2,)

GRID_SIZE = (
    len(_THRESHOLD_GRID)
    * len(_FORECAST_WINDOW_GRID)
    * len(_START_HOUR_GRID)
    * len(_CLOSE_HOUR_GRID)
    * len(_N_MASK_DAYS_GRID)
)  # 5*4*1*1*1 = 20


@dataclass
class Window:
    """One walk-forward (train, test) fold expressed in calendar dates."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def hyperparam_grid() -> list[dict]:
    """Return all 20 hyperparam combinations as a list of dicts."""
    return [
        {
            "threshold_x": tx,
            "forecast_window_min": fw,
            "start_hour": sh,
            "close_hour": ch,
            "n_mask_days": nm,
        }
        for tx, fw, sh, ch, nm in product(
            _THRESHOLD_GRID,
            _FORECAST_WINDOW_GRID,
            _START_HOUR_GRID,
            _CLOSE_HOUR_GRID,
            _N_MASK_DAYS_GRID,
        )
    ]


def generate_windows(
    trading_days: pd.DatetimeIndex,
    train_days: int = 10,
    test_days: int = 10,
    settlement_dates: pd.DatetimeIndex | None = None,
    n_mask_days: int = 2,
    slide_days: int = 5,
) -> list[Window]:
    """Slide 1-day (train, test) windows over ``trading_days``.
    """
    if settlement_dates is None:
        settlement_dates = pd.DatetimeIndex([])
    days = pd.DatetimeIndex(trading_days).normalize()
    if days.tz is not None:
        days = days.tz_localize(None)
    settle = pd.DatetimeIndex(settlement_dates).normalize()
    if len(settle) > 0 and settle.tz is not None:
        settle = settle.tz_localize(None)
    n = len(days)
    needed = train_days + test_days
    windows: list[Window] = []
    if n < needed:
        return windows

    for i in range(0, n - needed + 1, slide_days):
        train_start = days[i]
        train_end = days[i + train_days - 1]
        test_start = days[i + train_days]
        test_end = days[i + train_days + test_days - 1]

        lo = train_start - pd.Timedelta(days=n_mask_days)
        hi = test_end + pd.Timedelta(days=n_mask_days)
        overlaps = any((s >= lo) and (s <= hi) for s in settle)
        if overlaps:
            continue
        windows.append(
            Window(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return windows


def _slice_data_by_date(
    data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return rows in [start_date 00:00, end_date 23:59:59] inclusive.
    """
    if data.index.tz is not None:
        index_dates = data.index.tz_convert("America/New_York").normalize().tz_localize(None)
    else:
        index_dates = data.index.normalize()
    mask = (index_dates >= start_date.normalize()) & (index_dates <= end_date.normalize())
    return data.loc[mask]


def _score_trades(trades: list[Trade], initial_capital: float) -> float:
    """Compute Sharpe of the trade list; NaN for empty or singleton series."""
    daily = trades_to_daily_returns(trades, initial_capital)
    if len(daily) < 2:
        return float("nan")
    return sharpe(daily)


def grid_search(
    theta: dict,
    method: Literal["M2", "M4"],
    train_data: pd.DataFrame,
    settlement_dates: pd.DatetimeIndex,
    *,
    grid: list[dict] | None = None,
    initial_capital: float = 5000.0,
    warmup_returns: pd.Series | None = None,
    R_meas: np.ndarray | None = None,
) -> dict:
    """Sweep the hyperparam grid on ``train_data``; return best by Sharpe.
    """
    grid = grid if grid is not None else hyperparam_grid()
    best_score = -np.inf
    best_hp: dict | None = None
    all_results = []

    for hp in grid:
        trades = run_strategy(
            theta=theta,
            method=method,
            data=train_data,
            hyperparams=hp,
            settlement_dates=settlement_dates,
            warmup_returns=warmup_returns,
            R_meas=R_meas,
        )
        sr = _score_trades(trades, initial_capital)
        all_results.append((hp, sr, len(trades)))
        if np.isfinite(sr) and sr > best_score:
            best_score = sr
            best_hp = hp

    return {
        "best_hp": best_hp,
        "best_sharpe": (best_score if np.isfinite(best_score) else float("nan")),
        "all_results": all_results,
    }


def evaluate_test(
    theta: dict,
    method: Literal["M2", "M4"],
    test_data: pd.DataFrame,
    settlement_dates: pd.DatetimeIndex,
    chosen_hp: dict,
    *,
    initial_capital: float = 5000.0,
    warmup_returns: pd.Series | None = None,
    R_meas: np.ndarray | None = None,
) -> dict:
    """Apply chosen_hp to test_data; return (trades, sharpe, n_trades)."""
    trades = run_strategy(
        theta=theta,
        method=method,
        data=test_data,
        hyperparams=chosen_hp,
        settlement_dates=settlement_dates,
        warmup_returns=warmup_returns,
        R_meas=R_meas,
    )
    return {
        "trades": trades,
        "test_sharpe": _score_trades(trades, initial_capital),
        "n_trades": len(trades),
    }


def _hp_key(hp: dict) -> tuple:
    """Canonical hashable key for a hyperparam dict (tuples sorted by name)."""
    return tuple(sorted((k, v) for k, v in hp.items()))


def walk_forward(
    theta: dict,
    method: Literal["M2", "M4"],
    data: pd.DataFrame,
    settlement_dates: pd.DatetimeIndex,
    *,
    train_days: int = 10,
    test_days: int = 10,
    slide_days: int = 5,
    oos_days: int = 10,
    grid: list[dict] | None = None,
    initial_capital: float = 5000.0,
    min_trades_per_fold: int = 5,
    top_k_per_fold: int = 3,
    top_k: int = 5,
    warmup_returns: pd.Series | None = None,
    R_meas: np.ndarray | None = None,
) -> dict:
    """Rolling walk-forward with rank-based top-K HP selection (anti-overfit).
    """
    grid = grid if grid is not None else hyperparam_grid()

    # Trading-day index from data
    if data.index.tz is not None:
        date_index = data.index.tz_convert("America/New_York").normalize().tz_localize(None)
    else:
        date_index = data.index.normalize()
    trading_days = pd.DatetimeIndex(sorted(set(date_index))).normalize()

    if len(trading_days) <= oos_days:
        raise ValueError(f"Not enough trading days ({len(trading_days)}) for oos_days={oos_days}")
    oos_start_date = trading_days[-oos_days]
    walk_days = trading_days[:-oos_days]

    windows = generate_windows(
        walk_days,
        train_days=train_days,
        test_days=test_days,
        settlement_dates=None,
        n_mask_days=0,
        slide_days=slide_days,
    )

    # Per-fold metrics for every HP
    fold_results = []
    for win_idx, win in enumerate(windows):
        train_data = _slice_data_by_date(data, win.train_start, win.train_end)
        test_data = _slice_data_by_date(data, win.test_start, win.test_end)
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        per_hp = []
        for hp in grid:
            train_trades = run_strategy(
                theta=theta,
                method=method,
                data=train_data,
                hyperparams=hp,
                settlement_dates=settlement_dates,
                warmup_returns=warmup_returns,
                R_meas=R_meas,
            )
            test_trades = run_strategy(
                theta=theta,
                method=method,
                data=test_data,
                hyperparams=hp,
                settlement_dates=settlement_dates,
                warmup_returns=warmup_returns,
                R_meas=R_meas,
            )
            per_hp.append(
                {
                    "hp": hp,
                    "n_train_trades": len(train_trades),
                    "n_test_trades": len(test_trades),
                    "train_sharpe": _score_trades(train_trades, initial_capital),
                    "test_sharpe": _score_trades(test_trades, initial_capital),
                }
            )
        fold_results.append({"window": win, "per_hp": per_hp})

    n_folds = len(fold_results)

    # Rank-based aggregation across folds
    n_cells = len(grid)
    hp_top_count = [0] * n_cells
    hp_test_sharpes_valid: list[list[float]] = [[] for _ in range(n_cells)]
    hp_keys = [_hp_key(hp) for hp in grid]
    key_to_idx = {k: i for i, k in enumerate(hp_keys)}

    for fold in fold_results:
        # Filter HPs with sufficient trades on both legs and finite test_sharpe
        valid = []
        for r in fold["per_hp"]:
            if (
                r["n_train_trades"] >= min_trades_per_fold
                and r["n_test_trades"] >= min_trades_per_fold
                and np.isfinite(r["test_sharpe"])
            ):
                valid.append(r)
        # Sort by test_sharpe descending; top_k_per_fold are "fold winners"
        valid.sort(key=lambda r: r["test_sharpe"], reverse=True)
        for r in valid[:top_k_per_fold]:
            hp_top_count[key_to_idx[_hp_key(r["hp"])]] += 1
        # Track mean-test-sharpe stats across all valid appearances
        for r in valid:
            hp_test_sharpes_valid[key_to_idx[_hp_key(r["hp"])]].append(r["test_sharpe"])

    # Build full ranking: (top_count desc, mean_test_sharpe desc, n_valid_folds desc, hp)
    hp_rankings = []
    for i, hp in enumerate(grid):
        sharpes = hp_test_sharpes_valid[i]
        mean_sharpe = float(np.mean(sharpes)) if sharpes else float("nan")
        hp_rankings.append(
            {
                "hp": hp,
                "top_k_per_fold_count": hp_top_count[i],
                "mean_test_sharpe": mean_sharpe,
                "n_valid_folds": len(sharpes),
            }
        )
    hp_rankings.sort(
        key=lambda r: (
            r["top_k_per_fold_count"],
            r["mean_test_sharpe"] if np.isfinite(r["mean_test_sharpe"]) else float("-inf"),
            r["n_valid_folds"],
        ),
        reverse=True,
    )

    chosen_hp_list = [r["hp"] for r in hp_rankings[:top_k] if r["top_k_per_fold_count"] > 0]

    return {
        "chosen_hp_list": chosen_hp_list,
        "fold_results": fold_results,
        "hp_rankings": hp_rankings,
        "oos_start_date": oos_start_date,
        "n_folds": n_folds,
        "top_k_per_fold_used": top_k_per_fold,
        "top_k_used": top_k,
        "min_trades_per_fold_used": min_trades_per_fold,
    }


def final_oos(
    theta: dict,
    method: Literal["M2", "M4"],
    data: pd.DataFrame,
    settlement_dates: pd.DatetimeIndex,
    chosen_hp,  # dict | list[dict]
    *,
    oos_days: int = 10,
    n_trials: int = GRID_SIZE,
    initial_capital: float = 5000.0,
    warmup_returns: pd.Series | None = None,
    R_meas: np.ndarray | None = None,
) -> dict:
    """Apply chosen HP(s) to the last ``oos_days`` of ``data``.
    """
    from code_section7.backtest.metrics import (
        sharpe as _sharpe,
        max_drawdown as _maxdd,
        trades_to_daily_returns as _t2dr,
    )

    if isinstance(chosen_hp, dict):
        hps = [chosen_hp]
    else:
        hps = list(chosen_hp)

    if data.index.tz is not None:
        date_index = data.index.tz_convert("America/New_York").normalize().tz_localize(None)
    else:
        date_index = data.index.normalize()
    trading_days = pd.DatetimeIndex(sorted(set(date_index))).normalize()

    if len(trading_days) <= oos_days:
        raise ValueError(f"Not enough trading days ({len(trading_days)}) for oos_days={oos_days}")
    oos_start_date = trading_days[-oos_days]
    oos_end_date = trading_days[-1]
    oos_data = _slice_data_by_date(data, oos_start_date, oos_end_date)

    # Per-HP run
    per_hp_results = []
    all_trades = []
    per_hp_daily_returns = []
    for hp in hps:
        trades = run_strategy(
            theta=theta,
            method=method,
            data=oos_data,
            hyperparams=hp,
            settlement_dates=settlement_dates,
            warmup_returns=warmup_returns,
            R_meas=R_meas,
        )
        per_hp_summary = summarize(trades, initial_capital=initial_capital, n_trials=n_trials)
        per_hp_results.append({"hp": hp, "trades": trades, "summary": per_hp_summary})
        all_trades.extend(trades)
        if trades:
            per_hp_daily_returns.append(_t2dr(trades, initial_capital))


    bagged_daily = pd.Series(dtype=float)
    if per_hp_daily_returns:
        all_dates = sorted(set().union(*[d.index for d in per_hp_daily_returns]))
        aligned = pd.DataFrame(0.0, index=all_dates, columns=range(len(per_hp_daily_returns)))
        for i, daily in enumerate(per_hp_daily_returns):
            aligned.loc[daily.index, i] = daily.values
        bagged_daily = aligned.mean(axis=1)

    bagged_sharpe = _sharpe(bagged_daily) if len(bagged_daily) > 1 else float("nan")
    bagged_equity = (
        (1.0 + bagged_daily).cumprod() if len(bagged_daily) > 0 else pd.Series(dtype=float)
    )
    bagged_max_dd = _maxdd(bagged_equity) if len(bagged_equity) > 0 else float("nan")
    bagged_total_return = (
        float(bagged_equity.iloc[-1] - 1.0) if len(bagged_equity) > 0 else float("nan")
    )

    K = len(hps)
    summary_flat = summarize(all_trades, initial_capital=K * initial_capital, n_trials=n_trials)

    return {
        "trades": all_trades,
        "per_hp_results": per_hp_results,
        "hps_used": hps,
        "summary_flat": summary_flat,
        "bagged_sharpe": float(bagged_sharpe),
        "bagged_total_return": float(bagged_total_return),
        "bagged_max_drawdown": float(bagged_max_dd),
        "bagged_daily_returns": bagged_daily,
        "n_hps_bagged": K,
        "oos_start_date": oos_start_date,
        "oos_end_date": oos_end_date,
    }
