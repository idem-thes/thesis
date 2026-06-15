"""Section 8 sliding walk-forward - N-train / 1-val / 1-test (default 7/1/1), settlement-aware.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from code_section8_run2.metrics import mae
from code_section8_run2.model import fit_regression, predict

_TZ = "America/New_York"
_OOS_COLS = ["block_ts", "y_pred", "y_true", "vx_mid", "vx_bid", "vx_ask", "fold_id"]
_DEPTH_GRID = (2, 4, 6)


def _row_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(_TZ).tz_localize(None)
    return idx.normalize()


def sliding_folds(
    trading_days: pd.DatetimeIndex,
    settlement_dates: pd.DatetimeIndex,
    *,
    train_days: int = 7,
    val_days: int = 1,
    test_days: int = 1,
    slide_days: int = 1,
) -> list[dict]:
    """Sliding {"train":[...], "val":[...], "test":[...]} folds over trading days.

    """
    td = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_days).normalize())))
    settle = pd.DatetimeIndex(settlement_dates).normalize()
    win = train_days + val_days + test_days
    folds: list[dict] = []
    for s in range(0, len(td) - win + 1, slide_days):
        block = td[s : s + win]
        if ((settle >= block[0]) & (settle <= block[-1])).any():
            continue
        folds.append(
            {
                "train": list(block[:train_days]),
                "val": list(block[train_days : train_days + val_days]),
                "test": list(block[train_days + val_days :]),
            }
        )
    return folds


def nonoverlap_blocks(test_ts: pd.DatetimeIndex, horizon_s: int) -> pd.DatetimeIndex:
    """Greedy non-overlapping subset: first block, then each >= horizon_s after the last."""
    ts = pd.DatetimeIndex(sorted(test_ts))
    if len(ts) == 0:
        return ts
    h = pd.Timedelta(seconds=int(horizon_s))
    keep = [ts[0]]
    for t in ts[1:]:
        if t - keep[-1] >= h:
            keep.append(t)
    return pd.DatetimeIndex(keep)


def _slice_rows(ds: dict, dates) -> pd.Index:
    want = [pd.Timestamp(d).normalize() for d in np.atleast_1d(dates)]
    rd = _row_dates(ds["X"].index)
    return ds["X"].index[rd.isin(want)]


def _fit_select_depth(Xtr, ytr, Xval, yval, *, depths, early_stopping_rounds, seed):
    """Fit at each depth (early-stop on val); return the model with lowest val MAE.

    With no val rows, fall back to a single fit at the default config depth.
    """
    if Xval is None or len(Xval) == 0:
        return fit_regression(Xtr, ytr, seed=seed)
    yval_arr = np.asarray(yval, float)
    best_model, best_mae = None, np.inf
    for d in depths:
        m = fit_regression(
            Xtr,
            ytr,
            eval_set=(Xval, yval),
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            depth=int(d),
        )
        score = mae(yval_arr, predict(m, Xval))
        if score < best_mae:
            best_mae, best_model = score, m
    return best_model


def run_regression_wf(
    ds: dict,
    horizons: list[int],
    settlement_dates: pd.DatetimeIndex,
    *,
    train_days: int = 7,
    val_days: int = 1,
    test_days: int = 1,
    slide_days: int = 1,
    depths=_DEPTH_GRID,
    early_stopping_rounds: int = 50,
    seed: int = 20260525,
    shuffle_train_seed: int | None = None,
) -> dict[int, pd.DataFrame]:
    """Per horizon -> OOS DataFrame [block_ts, y_pred, y_true, vx_mid, vx_bid, vx_ask, fold_id].

    """
    rng = np.random.default_rng(shuffle_train_seed) if shuffle_train_seed is not None else None
    trading_days = pd.DatetimeIndex(sorted(set(_row_dates(ds["X"].index))))
    folds = sliding_folds(
        trading_days,
        settlement_dates,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
        slide_days=slide_days,
    )
    out: dict[int, list[pd.DataFrame]] = {h: [] for h in horizons}

    for fold_id, fold in enumerate(folds):
        tr_idx = _slice_rows(ds, fold["train"])
        va_idx = _slice_rows(ds, fold["val"])
        te_idx = _slice_rows(ds, fold["test"])
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        assert _row_dates(tr_idx).max() < _row_dates(te_idx).min(), f"fold {fold_id} leaks"

        for h in horizons:
            y = ds["targets"][h]
            ytr = y.loc[tr_idx].dropna()
            if len(ytr) == 0:
                continue
            if rng is not None:
                ytr = pd.Series(rng.permutation(ytr.to_numpy()), index=ytr.index)
            Xtr = ds["X"].loc[ytr.index]
            yva = y.loc[va_idx].dropna()
            Xval = ds["X"].loc[yva.index] if len(yva) > 0 else None
            model = _fit_select_depth(
                Xtr,
                ytr,
                Xval,
                yva if len(yva) > 0 else None,
                depths=depths,
                early_stopping_rounds=early_stopping_rounds,
                seed=seed,
            )
            blocks = nonoverlap_blocks(te_idx, h)
            out[h].append(
                pd.DataFrame(
                    {
                        "block_ts": blocks,
                        "y_pred": predict(model, ds["X"].loc[blocks]),
                        "y_true": y.loc[blocks].to_numpy(),
                        "vx_mid": ds["market"]["vx_mid"].loc[blocks].to_numpy(),
                        "vx_bid": ds["market"]["vx_bid"].loc[blocks].to_numpy(),
                        "vx_ask": ds["market"]["vx_ask"].loc[blocks].to_numpy(),
                        "fold_id": fold_id,
                    }
                )
            )

    result: dict[int, pd.DataFrame] = {}
    for h in horizons:
        result[h] = (
            pd.concat(out[h], ignore_index=True).sort_values("block_ts").reset_index(drop=True)
            if out[h]
            else pd.DataFrame({c: pd.Series(dtype="float64") for c in _OOS_COLS})
        )
    return result
