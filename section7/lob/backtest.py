"""Section 7.2 block-level fill + PnL evaluator (production home of the both-filled fix).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from code_section7.backtest.passive_sandbox import (
    CFE_FEE_RT,
    TICK_TOL,
    VX_MULTIPLIER,
    _CONSUME_SIDE,
    _snapshot_at,
    find_take_profit_exit,
    simulate_passive_entry_until,
    taker_exit_px,
)

__all__ = [
    "LobTrade",
    "VX_MULTIPLIER",
    "CFE_FEE_RT",
    "simulate_passive_entry_until",
    "simulate_passive_entry_until_cancel_aware",
    "evaluate_block",
    "evaluate_session",
]


_CANCEL_SIDE: dict[str, str] = {"long": "B", "short": "A"}

_BID_PX_COLS = [f"bid_px_{i:02d}" for i in range(10)]
_ASK_PX_COLS = [f"ask_px_{i:02d}" for i in range(10)]


@dataclass
class LobTrade:
    """One booked trade from a delta-block.

    ``kind`` is ``"long"`` / ``"short"`` for a single-leg maker-in / taker-out
    directional trade, or ``"spread"`` for a both-filled flat round-trip.
    ``pnl_dollars`` is net of the CFE round-trip fee.
    """

    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    kind: str  # "long" | "short" | "spread"
    pnl_dollars: float



def _level_size_at_price(px_row: np.ndarray, sz_row: np.ndarray, limit_px: float) -> float | None:
    """Displayed size at ``limit_px`` on our resting side, for one book row.

    ``px_row`` / ``sz_row`` are the per-level price / size ndarrays of a single
    event (already restricted to our resting side: bids for a long, asks for a
    short). Returns the size of the level whose price matches ``limit_px`` within
    ``TICK_TOL``, or ``None`` if our limit price is absent from the book row
    (e.g. the touch has moved away from our limit) or its size is not finite.
    """
    match = np.flatnonzero(np.isfinite(px_row) & (np.abs(px_row - limit_px) < TICK_TOL))
    if match.size == 0:
        return None
    sz = sz_row[match[0]]
    return float(sz) if np.isfinite(sz) else None


def simulate_passive_entry_until_cancel_aware(
    events: pd.DataFrame,
    entry_ts: pd.Timestamp,
    direction: Literal["long", "short"],
    deadline: pd.Timestamp,
    *,
    queue_position: str = "back",
    fill_on: Literal["queue", "trade"] = "queue",
    order_size: float | None = None,
) -> (
    tuple[bool, pd.Timestamp | None, float | None, float]
    | tuple[bool, pd.Timestamp | None, float | None, float, float]
):
    """Cancel-aware passive fill in ``(entry_ts, deadline]``.

    """
    if fill_on not in ("queue", "trade"):
        raise ValueError(f"fill_on must be 'queue' or 'trade'; got {fill_on!r}")
    sized = order_size is not None
    if sized and not (float(order_size) > 0):
        raise ValueError(f"order_size must be a positive number or None; got {order_size!r}")

    def _ret(filled, fill_ts, fill_px, q_init, filled_qty):
        """Return the 4-tuple (all-or-nothing) or 5-tuple (size-aware) per mode."""
        if sized:
            return filled, fill_ts, fill_px, q_init, float(filled_qty)
        return filled, fill_ts, fill_px, q_init

    snap = _snapshot_at(events, entry_ts)
    if snap is None:
        return _ret(False, None, None, float("nan"), 0.0)

    if direction == "long":
        limit_px = float(snap["bid_px_00"])
        queue_ahead = float(snap["bid_sz_00"])
    else:
        limit_px = float(snap["ask_px_00"])
        queue_ahead = float(snap["ask_sz_00"])

    if not np.isfinite(limit_px):
        return _ret(False, None, None, queue_ahead, 0.0)
    if queue_position == "front":
        queue_ahead = 0.0  # first in line: fill on the first consuming trade
    elif not (queue_ahead > 0):
        return _ret(False, None, None, queue_ahead, 0.0)

    queue_initial = queue_ahead
    # Back-of-queue fills when the displayed queue ahead is fully consumed (<= 0).
    # Front starts at 0, so a real (positive-size) trade must push it STRICTLY
    # below 0; a cancel subtracts 0 at the front and must NOT trigger a fill.
    fills_at = 0.0 if queue_position == "back" else -1e-12

    start_idx = int(events["ts_event"].searchsorted(entry_ts, side="right"))
    end_idx = int(events["ts_event"].searchsorted(deadline, side="right"))
    if end_idx <= start_idx:
        return _ret(False, None, None, queue_initial, 0.0)

    consume_side = _CONSUME_SIDE[direction]
    cancel_side = _CANCEL_SIDE[direction]

 
    ts = events["ts_event"].to_numpy()
    action = events["action"].to_numpy(str)
    side = events["side"].to_numpy(str)
    price = events["price"].to_numpy(float)
    size = events["size"].to_numpy(float)

    px_cols = _BID_PX_COLS if direction == "long" else _ASK_PX_COLS
    sz_prefix = "bid_sz_" if direction == "long" else "ask_sz_"
    level_px_cols = [c for c in px_cols if c in events.columns]
    level_sz_cols = [f"{sz_prefix}{c[-2:]}" for c in level_px_cols]
    level_px = events[level_px_cols].to_numpy(float)
    level_sz = events[level_sz_cols].to_numpy(float)

    # Last known displayed level size at our limit price; seeded from entry.
    level_size = queue_initial

    # Size-aware partial-fill accumulators (used only when ``sized``).
    filled_qty = 0.0
    last_fill_ts: pd.Timestamp | None = None

    for pos in range(start_idx, end_idx):
        if not np.isfinite(price[pos]) or abs(price[pos] - limit_px) >= TICK_TOL:
            # Refresh the level-size estimate from this off-our-price row's book
            # if it happens to still show our price (keeps the fallback fresh).
            sz = _level_size_at_price(level_px[pos], level_sz[pos], limit_px)
            if sz is not None and sz > 0:
                level_size = sz
            continue

        is_consuming_trade = action[pos] == "T" and side[pos] == consume_side
        if is_consuming_trade:
            queue_ahead -= size[pos]
        elif action[pos] == "C" and side[pos] == cancel_side:
            # Pre-event level size: the prior row's book (post-state of the
            # previous event = pre-state of this one); fall back to last known.
            # ``start_idx >= 1`` is guaranteed (the entry snapshot exists), and
            # ``level_*[start_idx - 1]`` IS that entry snapshot row.
            size_before = _level_size_at_price(level_px[pos - 1], level_sz[pos - 1], limit_px)
            if size_before is None or size_before <= 0:
                size_before = level_size
            frac = 0.0 if size_before <= 0 else queue_ahead / size_before
            frac = min(max(frac, 0.0), 1.0)
            queue_ahead -= size[pos] * frac

        # Update the running level-size estimate from this event's own book.
        sz = _level_size_at_price(level_px[pos], level_sz[pos], limit_px)
        if sz is not None and sz > 0:
            level_size = sz

        if not sized:

            if queue_ahead <= fills_at and (fill_on == "queue" or is_consuming_trade):
                return _ret(True, pd.Timestamp(ts[pos]), limit_px, queue_initial, 0.0)
            continue

 
        if fill_on == "queue":
            if queue_ahead <= fills_at:
                return _ret(True, pd.Timestamp(ts[pos]), limit_px, queue_initial, order_size)
            continue
        if is_consuming_trade and queue_ahead <= fills_at:
            residual = -queue_ahead  # trade size left after clearing the queue ahead
            if residual > 0:
                take = min(float(order_size) - filled_qty, residual)
                if take > 0:
                    filled_qty += take
                    last_fill_ts = pd.Timestamp(ts[pos])
                    # The unfilled remainder stays at the front for the next trade;
                    # reset queue_ahead to 0 so subsequent trades fill fully.
                    queue_ahead = 0.0
                    if filled_qty >= float(order_size) - 1e-9:
                        return _ret(True, last_fill_ts, limit_px, queue_initial, filled_qty)

    # Deadline reached. Size-aware: book whatever partial accumulated (may be 0).
    if sized:
        return _ret(bool(filled_qty > 0.0), last_fill_ts,
                    limit_px if filled_qty > 0 else None, queue_initial, filled_qty)
    return _ret(False, None, None, queue_initial, 0.0)


_ENTRY_SIMS = {
    "trade_only": simulate_passive_entry_until,
    "cancel_aware": simulate_passive_entry_until_cancel_aware,
}



def _single_leg_pnl(
    events: pd.DataFrame,
    fill_ts: pd.Timestamp,
    fill_px: float,
    direction: Literal["long", "short"],
    deadline: pd.Timestamp,
) -> LobTrade | None:
    tp = find_take_profit_exit(events, fill_ts, deadline, direction, fill_px)
    if tp is not None:
        exit_ts, exit_px, _reason = tp
    else:
        exit_ts = deadline
        exit_px = taker_exit_px(events, exit_ts, direction)
    if exit_px is None:
        return None

    pnl_pts = (exit_px - fill_px) if direction == "long" else (fill_px - exit_px)
    pnl = pnl_pts * VX_MULTIPLIER - CFE_FEE_RT
    return LobTrade(
        entry_ts=pd.Timestamp(fill_ts),
        exit_ts=pd.Timestamp(exit_ts),
        kind=direction,
        pnl_dollars=float(pnl),
    )


def evaluate_block(
    events: pd.DataFrame,
    entry_ts: pd.Timestamp,
    fire_long: bool,
    fire_short: bool,
    delta_s: int,
    *,
    fill_model: str = "trade_only",
    queue_position: str = "back",
) -> list[LobTrade]:
    """PnL of one delta-block's signals.

    """
    if fill_model not in _ENTRY_SIMS:
        raise ValueError(f"fill_model must be 'trade_only' or 'cancel_aware'; got {fill_model!r}")
    if queue_position not in ("back", "front"):
        raise ValueError(f"queue_position must be 'back' or 'front'; got {queue_position!r}")
    if queue_position == "front" and fill_model != "cancel_aware":
        # The trade-only sim lives in Section 7.1 passive_sandbox and has no queue knob;
        # front-of-queue (which makes cancels irrelevant anyway) is offered only
        # via the cancel-aware sim.
        raise ValueError("queue_position='front' is only supported for fill_model='cancel_aware'")
    if not (fire_long or fire_short):
        return []

    entry_ts = pd.Timestamp(entry_ts)
    deadline = entry_ts + pd.Timedelta(seconds=delta_s)
    entry_sim = _ENTRY_SIMS[fill_model]
    sim_kw = {"queue_position": queue_position} if fill_model == "cancel_aware" else {}

    long_filled = short_filled = False
    long_ts = short_ts = None
    long_px = short_px = None
    if fire_long:
        long_filled, long_ts, long_px, _q = entry_sim(events, entry_ts, "long", deadline, **sim_kw)
    if fire_short:
        short_filled, short_ts, short_px, _q = entry_sim(events, entry_ts, "short", deadline, **sim_kw)

    # Both legs filled -> spread capture: ONE flat round-trip, no taker exit.
    if fire_long and fire_short and long_filled and short_filled:
        bid_entry = float(long_px)  # long leg posted at the best bid
        ask_entry = float(short_px)  # short leg posted at the best ask
        pnl = (ask_entry - bid_entry) * VX_MULTIPLIER - CFE_FEE_RT
        book_ts = max(pd.Timestamp(long_ts), pd.Timestamp(short_ts))
        return [LobTrade(entry_ts=book_ts, exit_ts=book_ts, kind="spread", pnl_dollars=float(pnl))]

    # Otherwise: exactly one filled leg (if any) is directional via taker-out.
    if long_filled:
        tr = _single_leg_pnl(events, long_ts, long_px, "long", deadline)
        return [tr] if tr is not None else []
    if short_filled:
        tr = _single_leg_pnl(events, short_ts, short_px, "short", deadline)
        return [tr] if tr is not None else []

    return []


def evaluate_session(
    events: pd.DataFrame,
    blocks: Iterable[tuple[pd.Timestamp, bool, bool]],
    delta_s: int,
    *,
    fill_model: str = "trade_only",
    queue_position: str = "back",
) -> list[LobTrade]:
    """Apply :func:`evaluate_block` over a session's blocks.
    """
    trades: list[LobTrade] = []
    for entry_ts, fire_long, fire_short in blocks:
        trades.extend(
            evaluate_block(
                events,
                entry_ts,
                bool(fire_long),
                bool(fire_short),
                delta_s,
                fill_model=fill_model,
                queue_position=queue_position,
            )
        )
    return trades
