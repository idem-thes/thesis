"""Data layer for the Section 7.2 PnL backtest."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)



_FRD_VX_1DAY = Path("data/VX/VX_main/VX_1_DAY/VX_full_1day_continuous_UNadjusted.txt")
_FRD_SPX_1DAY = Path("data/SPX/SPX_main/SPX_full_1day.txt")
_FRD_SPX_1MIN = Path("data/SPX/SPX_main/SPX_full_1min.txt")
_FRD_VX_CONTRACTS = Path("data/VX/VX_all_contracts_dates.txt")




def load_frd_vx_daily() -> pd.DataFrame:
    """Read FirstRateData VX continuous daily file.

    Returns a DataFrame indexed by DatetimeIndex (date-only, tz-naive) with
    columns: open, high, low, close, volume, open_interest.
    Coverage: 2008-07-10 -> 2025-09-16 (~4329 rows).
    """
    df = pd.read_csv(
        _FRD_VX_1DAY,
        header=None,
        names=["date", "open", "high", "low", "close", "volume", "open_interest"],
        parse_dates=["date"],
    )
    df = df.set_index("date")
    df.index.name = "date"
    # Validate
    if not df.index.is_monotonic_increasing:
        raise ValueError("VX daily index is not monotonic increasing")
    for col in ["open", "high", "low", "close"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            raise ValueError(f"VX daily: {n_nan} NaN values in {col}")
    return df



def load_frd_spx_daily() -> pd.DataFrame:
    """Read FirstRateData SPX daily file.

    Returns a DataFrame indexed by DatetimeIndex (date-only, tz-naive) with
    columns: open, high, low, close.
    Coverage: 2000-11-27 -> 2025-09-16.
    """
    df = pd.read_csv(
        _FRD_SPX_1DAY,
        header=None,
        names=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    df = df.set_index("date")
    df.index.name = "date"
    if not df.index.is_monotonic_increasing:
        raise ValueError("SPX daily index is not monotonic increasing")
    for col in ["open", "high", "low", "close"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            raise ValueError(f"SPX daily: {n_nan} NaN values in {col}")
    return df




def load_spx_1min(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Read FirstRateData SPX 1-minute file, filtered to [start, end] inclusive.

    Uses chunksize iteration to avoid loading all 1.76 M rows.
    Returns DataFrame indexed by DatetimeIndex (tz-naive ET, minute resolution),
    columns: open, high, low, close.

    `start` and `end` are inclusive; any ISO date string or pd.Timestamp accepted.
    """
    start_ts = pd.Timestamp(start)
    # end is inclusive up to the end of that day
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        _FRD_SPX_1MIN,
        header=None,
        names=["datetime", "open", "high", "low", "close"],
        parse_dates=["datetime"],
        chunksize=50_000,
    ):
        # Filter the chunk to [start_ts, end_ts)
        mask = (chunk["datetime"] >= start_ts) & (chunk["datetime"] < end_ts)
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)
        # If we have passed end_ts, stop early
        elif not chunk.empty and chunk["datetime"].iloc[-1] >= end_ts:
            break

    if not chunks:
        return pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([], name="datetime"),
        )

    df = pd.concat(chunks, ignore_index=True)
    df = df.set_index("datetime")
    df.index.name = "datetime"
    return df



def _load_quotes_raw(
    product: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freq: str,
) -> pd.DataFrame:
    """Thin wrapper around code_section7.data_loader.load_quotes.

    Extracted as a module-level name so tests can monkeypatch it.
    Returns columns: [ts (tz-aware America/New_York), bid_px, ask_px, mid_px, symbol].
    """
    from code_section7.data_loader import load_quotes

    return load_quotes(product, str(start), str(end), freq=freq)  # type: ignore[arg-type]


def load_databento_1min(
    product: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Load Databento BBO at 1-second resolution then resample to 1-minute bars.
    """
    raw = _load_quotes_raw(product, start, end, freq="1s")

    # Set ts as index for resampling
    raw_indexed = raw.set_index("ts")

    bid = raw_indexed["bid_px"].resample("1min")
    ask = raw_indexed["ask_px"].resample("1min")
    mid = raw_indexed["mid_px"].resample("1min")
    sym = raw_indexed["symbol"].resample("1min")

    bars = pd.DataFrame(
        {
            "bid_open": bid.first(),
            "bid_close": bid.last(),
            "ask_open": ask.first(),
            "ask_close": ask.last(),
            "mid_open": mid.first(),
            "mid_close": mid.last(),
            "mid_high": mid.max(),
            "mid_low": mid.min(),
            "symbol": sym.last(),
        }
    )

    # Drop bars where no data exists (any NaN in the price columns)
    price_cols = ["bid_open", "bid_close", "ask_open", "ask_close", "mid_open", "mid_close"]
    bars = bars.dropna(subset=price_cols)

    bars = bars.reset_index().rename(columns={"ts": "ts"})
    # Ensure ts column is present (index was named 'ts' from raw_indexed)
    if "ts" not in bars.columns:
        bars = bars.rename(columns={bars.columns[0]: "ts"})

    return bars[
        [
            "ts",
            "bid_open",
            "bid_close",
            "ask_open",
            "ask_close",
            "mid_open",
            "mid_close",
            "mid_high",
            "mid_low",
            "symbol",
        ]
    ]



_VX_MONTH_CODES = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def load_vx_settlement_dates() -> pd.DatetimeIndex:
    """Return actual VX cash-settlement Wednesdays per contract in the FRD calendar.
    """
    df = pd.read_csv(
        _FRD_VX_CONTRACTS,
        header=None,
        names=["date", "symbol"],
        parse_dates=["date"],
    )
    out: list[pd.Timestamp] = []
    for sym in df["symbol"].unique():
        if not isinstance(sym, str) or len(sym) != 3:
            continue
        month_code, yr_str = sym[0], sym[1:]
        if month_code not in _VX_MONTH_CODES or not yr_str.isdigit():
            continue
        year = 2000 + int(yr_str)
        month = _VX_MONTH_CODES[month_code]
        # 3rd Wednesday: weekday Mon=0..Sun=6, Wed=2
        first = pd.Timestamp(year=year, month=month, day=1)
        offset_to_wed = (2 - first.weekday()) % 7
        out.append(first + pd.Timedelta(days=offset_to_wed + 14))
    return pd.DatetimeIndex(sorted(set(out)))




def mask_settlement_spikes(
    df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    n_days: int = 2,
) -> pd.DataFrame:
    """Drop rows whose date index falls within +/-n_days trading days of each settlement.
    """
    df = df.copy()
    trading_index = df.index
    keep = pd.Series(True, index=trading_index)

    for settlement in dates:
        # Find position of the settlement date in the trading index
        pos_arr = trading_index.searchsorted(settlement)
        # Clip to valid range
        pos = min(int(pos_arr), len(trading_index) - 1)

        lo = max(0, pos - n_days)
        hi = min(len(trading_index) - 1, pos + n_days)

        window_dates = trading_index[lo : hi + 1]
        keep[window_dates] = False

    return df.loc[keep]

_ES_ROLL_BRIDGES: list[dict] = [
    {
        "name": "ESH5_to_ESM5",
        "last_clean_ts": pd.Timestamp("2025-03-21 09:30:56", tz="America/New_York"),
        "fri_rth_end_ts": pd.Timestamp("2025-03-21 16:00:00", tz="America/New_York"),
        "monday_open_ts": pd.Timestamp("2025-03-24 09:30:00", tz="America/New_York"),
    },
    {
        "name": "ESM5_to_ESU5",
        "last_clean_ts": pd.Timestamp("2025-06-20 09:58:00", tz="America/New_York"),
        "fri_rth_end_ts": pd.Timestamp("2025-06-20 16:00:00", tz="America/New_York"),
        "monday_open_ts": pd.Timestamp("2025-06-23 09:30:00", tz="America/New_York"),
    },
]


def splice_es_roll(es_1min: pd.DataFrame, spx_1min: pd.DataFrame) -> pd.DataFrame:
    """Splice ES across two known quarterly-roll weekends using SPX as a wall-clock bridge."""
    out = es_1min.copy()

    # Baseline ES log-return on the full mid_close series.
    log_mid = np.log(out["mid_close"])
    es_returns = log_mid.diff()
    out["return_1min"] = es_returns.values

    # Strip any tz on the SPX index for safe naive comparisons.
    spx_close = spx_1min["close"]
    if isinstance(spx_close.index, pd.DatetimeIndex) and spx_close.index.tz is not None:
        spx_close = spx_close.copy()
        spx_close.index = spx_close.index.tz_localize(None)

    # Columns that already exist in the ES frame; we'll preserve their order on output.
    es_cols = list(out.columns)

    for bridge in _ES_ROLL_BRIDGES:
        last_clean = bridge["last_clean_ts"]
        fri_end = bridge["fri_rth_end_ts"]
        mon_open = bridge["monday_open_ts"]

        # Skip bridge silently if its window is outside the loaded ES frame.
        ts_series = out["ts"]
        if ts_series.empty:
            continue
        ts_min, ts_max = ts_series.min(), ts_series.max()
        if mon_open < ts_min or last_clean > ts_max:
            continue

   
        # Keep rows at exactly last_clean_ts and exactly monday_open_ts.
        drop_mask = (out["ts"] > last_clean) & (out["ts"] < mon_open)
        out = out.loc[~drop_mask].copy()


        naive_last = last_clean.tz_localize(None)
        naive_fri = fri_end.tz_localize(None)
        bridge_spx = spx_close[(spx_close.index > naive_last) & (spx_close.index <= naive_fri)]
        if len(bridge_spx) > 0:
        
            spx_log = np.log(bridge_spx)
            spx_diffs = spx_log.diff()
            # Anchor the first bridge bar to the SPX close at-or-before last_clean.
            anchor_mask = spx_close.index <= naive_last
            if anchor_mask.any():
                anchor_close = float(spx_close[anchor_mask].iloc[-1])
                if np.isfinite(anchor_close) and anchor_close > 0:
                    spx_diffs.iloc[0] = float(
                        np.log(float(bridge_spx.iloc[0])) - np.log(anchor_close)
                    )

            tz = "America/New_York"
            bridge_ts = bridge_spx.index.tz_localize(tz)

            bridge_rows = pd.DataFrame({col: [np.nan] * len(bridge_spx) for col in es_cols})
            bridge_rows["ts"] = bridge_ts
            if "mid_close" in bridge_rows.columns:
                bridge_rows["mid_close"] = bridge_spx.values
            # mid_open mirrors mid_close to keep downstream plotting consistent.
            if "mid_open" in bridge_rows.columns:
                bridge_rows["mid_open"] = bridge_spx.values
            if "mid_high" in bridge_rows.columns:
                bridge_rows["mid_high"] = bridge_spx.values
            if "mid_low" in bridge_rows.columns:
                bridge_rows["mid_low"] = bridge_spx.values
            # bid_close, ask_close, bid_open, ask_open stay NaN - strategy skips
            # bars without quotes, so no synthetic trading happens here.
            if "return_1min" in bridge_rows.columns:
                bridge_rows["return_1min"] = spx_diffs.values
            if "symbol" in bridge_rows.columns:
                bridge_rows["symbol"] = "SPX_BRIDGE"

            out = pd.concat([out, bridge_rows[es_cols]], ignore_index=True)

      
        mon_open_idx = out.index[out["ts"] == mon_open]
        if (
            len(mon_open_idx) == 1
            and naive_fri in spx_close.index
            and mon_open.tz_localize(None) in spx_close.index
        ):
            fri_close_spx = float(spx_close.loc[naive_fri])
            mon_open_spx = float(spx_close.loc[mon_open.tz_localize(None)])
            if (
                np.isfinite(fri_close_spx)
                and np.isfinite(mon_open_spx)
                and fri_close_spx > 0
                and mon_open_spx > 0
            ):
                weekend_ret = float(np.log(mon_open_spx) - np.log(fri_close_spx))
                out.at[mon_open_idx[0], "return_1min"] = weekend_ret
            else:
                log.warning(
                    "splice_es_roll[%s]: non-finite SPX close at Fri-end or Mon-open; "
                    "monday_open return left as raw ES diff.",
                    bridge["name"],
                )
        elif len(mon_open_idx) == 0:
            log.warning(
                "splice_es_roll[%s]: monday_open_ts %s not in ES frame; "
                "weekend return not patched.",
                bridge["name"],
                mon_open,
            )


    out = out.sort_values("ts").reset_index(drop=True)
    return out
