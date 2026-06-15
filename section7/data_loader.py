"""Data loader - the only module that touches data/databento/.

"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

Product = Literal["ES", "VX"]
Freq = Literal["1s"]

_DATABENTO_ROOT = Path("data/databento")
_CACHE_ROOT = Path("outputs/_cache")

_PRODUCT_DIRS: dict[str, list[Path]] = {
    "ES": [
        # Window 1: 2025-03-02 -> 2025-05-30 (purchased 2026-05-11)
        _DATABENTO_ROOT / "GLBX.MDP3/ES.c.0/GLBX-20260511-ADYL7397MR",
        # Window 2: 2025-06-01 -> 2025-08-29 (purchased 2026-05-16)
        _DATABENTO_ROOT / "GLBX.MDP3/ES.c.0/GLBX-20260516-369KL44UJ3",
    ],
    "VX": [
        _DATABENTO_ROOT / "XCBF.PITCH/VX.c.0/XCBF-20260511-A9P7FK9SV9",
        _DATABENTO_ROOT / "XCBF.PITCH/VX.c.0/XCBF-20260516-LLRNJE8P4B",
    ],
}
_PRODUCT_FILE_PATTERNS = {
    "ES": "glbx-mdp3-*.mbp-10.dbn.zst",
    "VX": "xcbf-pitch-*.mbp-10.dbn.zst",
}


_VIXY_VENUES = [
    "XNAS.ITCH",
    "BATS.PITCH",
    "BATY.PITCH",
    "EDGA.PITCH",
    "EDGX.PITCH",
    "ARCX.PILLAR",
    "XNYS.PILLAR",
    "XASE.PILLAR",
    "XCHI.PILLAR",
    "XBOS.ITCH",
    "XPSX.ITCH",
]


def _vixy_file_prefix(venue: str) -> str:
    """Databento batch filename prefix: dataset code lowercased, '.'->'-'."""
    return venue.lower().replace(".", "-")


def _vixy_session_path(venue: str, date: "pd.Timestamp | str") -> "Path | None":
    """First on-disk .dbn.zst for (venue, date) across that venue's job dirs, else None."""
    d = pd.Timestamp(date)
    fname = f"{_vixy_file_prefix(venue)}-{d.strftime('%Y%m%d')}.mbp-10.dbn.zst"
    venue_root = _DATABENTO_ROOT / venue / "VIXY"
    if not venue_root.exists():
        return None
    for jobdir in sorted(venue_root.glob("*")):
        p = jobdir / fname
        if p.exists():
            return p
    return None


# Memory monitoring thresholds (GB), tuned for 24 GB M5 Pro.
_WARN_RSS_GB = 20.0
_KILL_RSS_GB = 22.0
_WARN_AVAIL_GB = 3.0
_KILL_AVAIL_GB = 1.5

_MONTH_ORDER = {
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
_SYMBOL_RE = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})$")

_IID_TO_RAW_SYMBOL: dict[int, str] = {
    # ES front-month iids (CME Globex MDP 3.0)
    5002: "ESH5",
    4916: "ESM5",
    14160: "ESU5",  # added Window 2 - front from 2025-06-23
    # VX front-month iids (Cboe Futures Exchange)
    150222: "VXH5",
    150654: "VXJ5",
    151270: "VXK5",
    151776: "VXM5",
    152274: "VXN5",  # added Window 2 - front 2025-06-18 -> 2025-07-15
    154785: "VXQ5",  # added Window 2 - front 2025-07-16 -> 2025-08-19
    157108: "VXU5",  # added Window 2 - front from 2025-08-20
}


def _parse_contract(symbol: str) -> tuple[str, int, int]:
    """Parse a futures contract symbol into (root, calendar_year, month_int).


    """
    m = _SYMBOL_RE.match(symbol)
    if m is None:
        raise ValueError(f"unrecognized contract symbol: {symbol!r}")
    root, month_code, yr_str = m.groups()
    year = 2020 + int(yr_str) if len(yr_str) == 1 else 2000 + int(yr_str)
    return root, year, _MONTH_ORDER[month_code]


def _check_memory(
    stage: str,
    _rss_gb: float | None = None,
    _avail_gb: float | None = None,
) -> None:
    """Inspect process RSS + system available memory.



    """
    use_injection = _rss_gb is not None

    if not use_injection:
        import psutil

        proc = psutil.Process()
        _rss_gb = proc.memory_info().rss / (1024**3)
        _avail_gb = psutil.virtual_memory().available / (1024**3)

    log.info(f"[memory:{stage}] RSS={_rss_gb:.1f} GB / avail={_avail_gb:.1f} GB")

    soft = _rss_gb > _WARN_RSS_GB or _avail_gb < _WARN_AVAIL_GB
    if soft:
        log.warning(
            f"[memory:{stage}] soft threshold tripped (RSS={_rss_gb:.1f} GB, avail={_avail_gb:.1f} GB); gc.collect()"
        )
        gc.collect()
        if not use_injection:
            proc = psutil.Process()
            _rss_gb = proc.memory_info().rss / (1024**3)
            _avail_gb = psutil.virtual_memory().available / (1024**3)

    hard = _rss_gb > _KILL_RSS_GB or _avail_gb < _KILL_AVAIL_GB
    if hard:
        log.critical(
            f"[memory:{stage}] hard threshold tripped (RSS={_rss_gb:.1f} GB, avail={_avail_gb:.1f} GB) - aborting"
        )
        raise SystemExit(1)


def _resample_bbo(df_raw: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample raw MBP-10 events to a periodic BBO grid.

    """
    idx = df_raw.index.tz_convert("America/New_York")
    df = df_raw.set_axis(idx)[["bid_px_00", "ask_px_00", "symbol"]]
    resampled = df.resample(freq).last().dropna(subset=["bid_px_00", "ask_px_00"])
    resampled = resampled.rename(columns={"bid_px_00": "bid_px", "ask_px_00": "ask_px"})
    resampled["mid_px"] = (resampled["bid_px"] + resampled["ask_px"]) / 2
    resampled = resampled.reset_index().rename(columns={resampled.index.name or "ts_event": "ts"})
    if "ts" not in resampled.columns:
        # Fallback: the index name may be empty after some pandas versions; rename first col.
        resampled = resampled.rename(columns={resampled.columns[0]: "ts"})
    return resampled[["ts", "bid_px", "ask_px", "mid_px", "symbol"]]


def _rolls_by_symbol_diff(quotes: pd.DataFrame) -> pd.DataFrame:
    """Method A: roll date = first day whose dominant symbol differs from prior day's."""
    dates = quotes["ts"].dt.date
    daily_dominant = (
        pd.Series(quotes["symbol"].values, index=dates)
        .groupby(level=0)
        .agg(lambda s: s.mode().iloc[0])
        .sort_index()
    )
    changed = daily_dominant != daily_dominant.shift(1)
    if len(daily_dominant) > 0:
        changed.iloc[0] = False  # day 0 has no prior; never call it a roll
    roll_dates = list(daily_dominant.index[changed])
    return pd.DataFrame(
        {
            "roll_date": roll_dates,
            "old_symbol": [daily_dominant.shift(1).loc[d] for d in roll_dates],
            "new_symbol": [daily_dominant.loc[d] for d in roll_dates],
        }
    )


def _rolls_by_volume_crossover(quotes: pd.DataFrame) -> pd.DataFrame:
    """Method B: roll date = first day where new_symbol's count exceeds old_symbol's."""
    dates = quotes["ts"].dt.date
    counts = (
        pd.DataFrame({"date": dates, "symbol": quotes["symbol"].values})
        .groupby(["date", "symbol"])
        .size()
        .unstack(fill_value=0)
    )
    symbols_sorted = sorted(counts.columns, key=lambda s: _parse_contract(s)[1:])
    rolls = []
    for old, new in zip(symbols_sorted, symbols_sorted[1:]):
        crossover = counts[new] > counts[old]
        if crossover.any():
            roll_date = crossover.idxmax()
            rolls.append({"roll_date": roll_date, "old_symbol": old, "new_symbol": new})
    return pd.DataFrame(rolls, columns=["roll_date", "old_symbol", "new_symbol"])


def _assert_rolls_agree(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Raise ValueError if methods A and B produce different roll tuples."""
    cols = ["roll_date", "old_symbol", "new_symbol"]
    a_set = set(map(tuple, a[cols].itertuples(index=False)))
    b_set = set(map(tuple, b[cols].itertuples(index=False)))
    if a_set != b_set:
        only_a = a_set - b_set
        only_b = b_set - a_set
        raise ValueError(
            "roll detection methods A and B disagree:\n"
            f"  symbol-diff only:      {sorted(only_a)}\n"
            f"  volume-crossover only: {sorted(only_b)}"
        )


def _add_discontinuity(rolls: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Append discontinuity_pts = mid_px(first new on/after roll_date) - mid_px(last old before)."""
    rolls = rolls.copy()
    dates = quotes["ts"].dt.date
    discont = []
    for _, row in rolls.iterrows():
        roll_d = row["roll_date"]
        prior = quotes.loc[(dates < roll_d) & (quotes["symbol"] == row["old_symbol"]), "mid_px"]
        after = quotes.loc[(dates >= roll_d) & (quotes["symbol"] == row["new_symbol"]), "mid_px"]
        if len(prior) == 0 or len(after) == 0:
            discont.append(float("nan"))
        else:
            discont.append(float(after.iloc[0] - prior.iloc[-1]))
    rolls["discontinuity_pts"] = discont
    return rolls


def _assert_rolls_vs_frd(rolls: pd.DataFrame, frd_calendar: pd.DataFrame, product: str) -> None:
    """VX-only cross-check against FirstRateData VX calendar.
    """
    if product != "VX":
        return

    cal = frd_calendar.sort_values("Date").reset_index(drop=True)
    cal_changed = cal["Symbol"] != cal["Symbol"].shift(1)
    # FRD roll: row where Symbol differs from prior - skip the always-changed first row
    cal_rolls = cal.loc[cal_changed].iloc[1:] if len(cal) > 0 else cal.iloc[0:0]

    def _frd_year_month(sym: str) -> tuple[int, int]:
        """FRD '<MonthLetter><YY>' -> (calendar_year_4digit, month_int)."""
        return 2000 + int(sym[1:]), _MONTH_ORDER[sym[0]]

    for _, db_roll in rolls.iterrows():
        _, db_year, db_month = _parse_contract(db_roll["new_symbol"])
        target = (db_year, db_month)

        match_mask = cal_rolls["Symbol"].apply(lambda s: _frd_year_month(s) == target)
        match = cal_rolls[match_mask]

        if len(match) == 0:
            raise ValueError(
                f"FRD calendar has no entry matching {db_roll['new_symbol']} "
                f"(year={db_year}, month={db_month})"
            )
        frd_date = match.iloc[0]["Date"].date()
        diff_days = abs((frd_date - db_roll["roll_date"]).days)

        prefix = f"[FRD-check] {db_roll['old_symbol']} -> {db_roll['new_symbol']}"
        if diff_days <= 2:
            log.info(f"{prefix}: aligned within {diff_days} days")
        elif diff_days <= 14:
            log.warning(
                f"{prefix}: differs by {diff_days} days (FRD={frd_date}, Databento={db_roll['roll_date']})"
            )
        else:
            raise ValueError(
                f"FRD calendar disagrees with Databento by {diff_days} days for "
                f"{db_roll['old_symbol']} -> {db_roll['new_symbol']} "
                f"(FRD={frd_date}, Databento={db_roll['roll_date']})"
            )


def _session_paths(product: Product) -> list[Path]:
    """Sorted list of per-session .dbn.zst paths on disk for the product.

    Globs across every directory in `_PRODUCT_DIRS[product]` so that successive
    Databento purchases stitch into one chronological session list.
    """
    paths: list[Path] = []
    for d in _PRODUCT_DIRS[product]:
        paths.extend(d.glob(_PRODUCT_FILE_PATTERNS[product]))
    return sorted(paths)


def _date_from_filename(path: Path) -> str:
    """Extract 'YYYYMMDD' date string from a Databento session filename."""
    # e.g. "glbx-mdp3-20250303.mbp-10.dbn.zst" -> "20250303"
    stem = path.name.split(".")[0]  # "glbx-mdp3-20250303"
    return stem.rsplit("-", 1)[-1]


def _resolve_iid_to_raw(mappings) -> dict[int, str]:
    """Build instrument_id (int) -> raw_symbol map for the iids referenced in `mappings`.

    """
    out: dict[int, str] = {}
    for _, intervals in mappings.items():
        for interval in intervals:
            iid_str = (
                interval.get("symbol")
                if isinstance(interval, dict)
                else getattr(interval, "symbol", None)
            )
            if iid_str is None:
                continue
            try:
                iid = int(iid_str)
            except (TypeError, ValueError):
                continue
            raw = _IID_TO_RAW_SYMBOL.get(iid)
            if raw is not None:
                out[iid] = raw
    return out


def _decode_session(path: Path, freq: str) -> pd.DataFrame:
    """Decode one .dbn.zst session, resample to BBO grid, return small df.

    """
    import databento as db  # lazy import to keep module-load cheap

    _check_memory(f"before {path.stem}")
    store = db.DBNStore.from_file(path)
    df_raw = store.to_df()

    # Rewrite the constant continuous-symbol column to per-contract tickers.
    iid_to_raw = _resolve_iid_to_raw(store.metadata.mappings)
    if iid_to_raw:
        df_raw["symbol"] = df_raw["instrument_id"].map(iid_to_raw).fillna(df_raw["symbol"])

    out = _resample_bbo(df_raw, freq)
    del df_raw, store
    gc.collect()
    _check_memory(f"after {path.stem}")
    return out


_PRODUCT_FILE_PREFIX = {"ES": "glbx-mdp3", "VX": "xcbf-pitch"}


def _session_path_for_date(product: Product, date: pd.Timestamp | str) -> Path | None:
    """First on-disk .dbn.zst for (product, date) across all purchase dirs, else None."""
    d = pd.Timestamp(date)
    fname = f"{_PRODUCT_FILE_PREFIX[product]}-{d.strftime('%Y%m%d')}.mbp-10.dbn.zst"
    for base in _PRODUCT_DIRS[product]:
        p = base / fname
        if p.exists():
            return p
    return None


def load_session_events(
    product: Product,
    date: pd.Timestamp | str,
    levels: int = 5,
) -> pd.DataFrame | None:
    """Raw MBP-10 events for one session, top-`levels` depth, tz America/New_York.

    """
    if not 1 <= levels <= 10:
        raise ValueError(f"levels must be in 1..10 for MBP-10; got {levels}")

    import databento as db

    path = _session_path_for_date(product, date)
    if path is None:
        return None

    _check_memory(f"before events {product} {pd.Timestamp(date).date()}")
    store = db.DBNStore.from_file(str(path))
    df = store.to_df()
 
    if "ts_recv" not in df.columns:
        df = df.reset_index()

    iid_to_raw = _resolve_iid_to_raw(store.metadata.mappings)
    if iid_to_raw:
        df["symbol"] = df["instrument_id"].map(iid_to_raw).fillna(df["symbol"])

    px = [f"{s}_{i:02d}" for s in ("bid_px", "ask_px") for i in range(levels)]
    sz = [f"{s}_{i:02d}" for s in ("bid_sz", "ask_sz") for i in range(levels)]
    keep = ["ts_recv", "ts_event", "action", "side", "price", "size", *px, *sz, "symbol"]
    out = df[keep].copy()
    for col in ("ts_recv", "ts_event"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.tz_convert("America/New_York")
    out["action"] = out["action"].astype("category")
    out["side"] = out["side"].astype("category")
    out = out.sort_values("ts_event").reset_index(drop=True)
    del df, store
    gc.collect()
    return out


def load_venue_session_events(
    venue: str,
    date: "pd.Timestamp | str",
    levels: int = 10,
) -> "pd.DataFrame | None":
    """Raw equity MBP-10 events for one venue+session, keyed on ts_recv.

    """
    if not 1 <= levels <= 10:
        raise ValueError(f"levels must be in 1..10 for MBP-10; got {levels}")
    import databento as db

    path = _vixy_session_path(venue, date)
    if path is None:
        return None

    _check_memory(f"before venue {venue} {pd.Timestamp(date).date()}")
    store = db.DBNStore.from_file(str(path))
    df = store.to_df().reset_index()  # ts_recv was the index -> now a column
    px = [f"{s}_{i:02d}" for s in ("bid_px", "ask_px") for i in range(levels)]
    sz = [f"{s}_{i:02d}" for s in ("bid_sz", "ask_sz") for i in range(levels)]
    keep = ["ts_recv", "ts_event", "action", "side", "price", "size", *px, *sz]
    out = df[keep].copy()
    for col in ("ts_recv", "ts_event"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.tz_convert("America/New_York")
    out["action"] = out["action"].astype("category")
    out["side"] = out["side"].astype("category")
    out["venue"] = venue
    out = out.sort_values("ts_recv").reset_index(drop=True)  # SORT BY ts_recv
    del df, store
    gc.collect()
    return out


def load_quotes(
    product: Product,
    start: str | None = None,
    end: str | None = None,
    freq: Freq = "1s",
) -> pd.DataFrame:
    """1-sec BBO snapshots, forward-filled through MBP-10 events.

    """
    import databento as db
    from tqdm import tqdm

    lib_ver = db.__version__.replace(".", "")
    consolidated = _CACHE_ROOT / f"databento_quotes_{product}_{freq}_db{lib_ver}.parquet"

    if not consolidated.exists():
        partial_dir = _CACHE_ROOT / "databento_quotes_partial"
        partial_dir.mkdir(parents=True, exist_ok=True)

        all_sessions = _session_paths(product)
        if not all_sessions:
            dirs = ", ".join(str(d) for d in _PRODUCT_DIRS[product])
            raise FileNotFoundError(f"no .dbn.zst sessions found for {product} under {dirs}")

        def partial_path(p: Path) -> Path:
            return partial_dir / f"{product}_{_date_from_filename(p)}_{freq}_db{lib_ver}.parquet"

        missing = [p for p in all_sessions if not partial_path(p).exists()]

        for path in tqdm(missing, desc=f"decoding {product}", unit="session"):
            out_path = partial_path(path)
            df = _decode_session(path, freq)
            df.to_parquet(out_path, index=False)
            del df
            gc.collect()

        parts = sorted(partial_dir.glob(f"{product}_*_{freq}_db{lib_ver}.parquet"))
        all_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(consolidated, index=False)

    df = pd.read_parquet(consolidated)
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start, tz="America/New_York")]
    if end is not None:
        end_excl = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1)
        df = df[df["ts"] < end_excl]
    return df.reset_index(drop=True)


def load_rolls(product: Product) -> pd.DataFrame:
    """Continuous-symbol roll-date table.

    """
    quotes = load_quotes(product)
    a = _rolls_by_symbol_diff(quotes)
    b = _rolls_by_volume_crossover(quotes)
    _assert_rolls_agree(a, b)

    if product == "VX":
        from code_section6.data_loader import load_vx_calendar

        frd = load_vx_calendar()
        if not quotes.empty:
            date_min = quotes["ts"].dt.date.min()
            date_max = quotes["ts"].dt.date.max()
            frd_window = frd[(frd["Date"].dt.date >= date_min) & (frd["Date"].dt.date <= date_max)]
        else:
            frd_window = frd.iloc[0:0]
        _assert_rolls_vs_frd(a, frd_window, product)

    return _add_discontinuity(a, quotes)


def databento_trading_dates(product: Product) -> pd.DatetimeIndex:
    """Session dates present on disk."""
    raise NotImplementedError("implement when features.py needs the calendar.")
