"""Section 8 execution: sigma-gate a signed Delta VX forecast into taker / maker trades.

"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from code_section7.backtest.passive_sandbox import (
    CFE_FEE_RT,
    VX_MULTIPLIER,
    find_take_profit_exit,
    taker_exit_px,
)
from code_section7.lob.backtest import simulate_passive_entry_until_cancel_aware


def gate(pred: float, *, spread: float, alpha: float) -> tuple[bool, bool]:
    """(fire_long, fire_short) for a signed forecast vs the alpha*spread threshold."""
    thr = alpha * spread
    return (pred > thr, pred < -thr)


def _exit_pnl(events, entry_ts, entry_px, direction, deadline) -> float | None:
    """First-profitable-touch exit (1-tick TP) else taker-out at deadline; $ PnL."""
    tp = find_take_profit_exit(events, entry_ts, deadline, direction, entry_px)
    if tp is not None:
        _exit_ts, exit_px, _reason = tp
    else:
        exit_px = taker_exit_px(events, deadline, direction)
    if exit_px is None:
        return None
    pnl_pts = (exit_px - entry_px) if direction == "long" else (entry_px - exit_px)
    return float(pnl_pts * VX_MULTIPLIER - CFE_FEE_RT)


def taker_trade(
    events: pd.DataFrame,
    entry_ts: pd.Timestamp,
    direction: Literal["long", "short"],
    horizon_s: int,
    *,
    entry_px: float,
) -> float | None:
    """Cross in at `entry_px` (ask_t long / bid_t short), close as soon as profitable."""
    entry_ts = pd.Timestamp(entry_ts)
    deadline = entry_ts + pd.Timedelta(seconds=int(horizon_s))
    return _exit_pnl(events, entry_ts, float(entry_px), direction, deadline)


def maker_block_pnl(
    events: pd.DataFrame,
    entry_ts: pd.Timestamp,
    direction: Literal["long", "short"],
    horizon_s: int,
    *,
    queue_position: str = "back",
) -> float | None:
    """Honest passive maker PnL ($) for one gated block, or None if never filled."""
    entry_ts = pd.Timestamp(entry_ts)
    deadline = entry_ts + pd.Timedelta(seconds=int(horizon_s))
    res = simulate_passive_entry_until_cancel_aware(
        events,
        entry_ts,
        direction,
        deadline,
        queue_position=queue_position,
        fill_on="trade",  # FINAL corrected model
    )
    filled, fill_ts, fill_px = res[0], res[1], res[2]
    if not filled or fill_px is None:
        return None
    return _exit_pnl(events, fill_ts, float(fill_px), direction, deadline)
