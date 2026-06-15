"""Section 7.2.1 figure: ES (front-month continuous, yfinance) vs SPX cash
(FirstRateData) over the calibration period.

"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import yfinance as yf  # noqa: E402

SPX_PATH = REPO_ROOT / "data" / "SPX" / "SPX_main" / "SPX_full_1day.txt"
OUT_DIR = REPO_ROOT / "outputs" / "_results" / "section7"
OUT_PDF = OUT_DIR / "es_spx_calibration.pdf"
OUT_HTML = OUT_DIR / "es_spx_calibration.html"

CALIB_START = "2012-07-10"
CALIB_END = "2025-02-28"

SPX_COLOR = "#1f77b4"
ES_COLOR = "#d62728"


def _load_spx_daily() -> pd.Series:
    df = pd.read_csv(
        SPX_PATH,
        header=None,
        names=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    return df.set_index("date")["close"].sort_index()


def _load_es_daily(start: str, end: str) -> pd.Series:
    raw = yf.download("ES=F", start=start, end=end, auto_adjust=False, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def main() -> None:
    print(f"Loading FRD SPX daily from {SPX_PATH.name} ...")
    spx = _load_spx_daily()
    print(f"  SPX raw: {len(spx):,} bars, {spx.index.min().date()} -> {spx.index.max().date()}")

    print(f"Fetching ES=F from yfinance ({CALIB_START} -> {CALIB_END}) ...")
    es = _load_es_daily(CALIB_START, CALIB_END)
    print(f"  ES raw: {len(es):,} bars, {es.index.min().date()} -> {es.index.max().date()}")

    joined = pd.concat({"SPX": spx, "ES": es}, axis=1).dropna()
    joined = joined.loc[CALIB_START:CALIB_END]
    print(
        f"  Inner-joined: {len(joined):,} bars, {joined.index.min().date()} -> {joined.index.max().date()}"
    )
    print(
        f"  mean |ES - SPX| basis: {(joined['ES'] - joined['SPX']).abs().mean():.2f} index points"
    )
    print(f"  Pearson corr(ES, SPX): {joined.corr().iloc[0, 1]:.4f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=joined.index,
            y=joined["SPX"],
            mode="lines",
            name="SPX (cash)",
            line=dict(color=SPX_COLOR, width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=joined.index,
            y=joined["ES"],
            mode="lines",
            name="ES (front-month continuous)",
            line=dict(color=ES_COLOR, width=1.2),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Index points",
        template="plotly_white",
        width=900,
        height=420,
        margin=dict(l=70, r=20, t=20, b=60),
        legend=dict(
            x=0.01,
            y=0.99,
            yanchor="top",
            xanchor="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=0.5,
            font=dict(size=11),
        ),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT_PDF), format="pdf")
    fig.write_html(str(OUT_HTML))
    print(f"\nWrote {OUT_PDF}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
