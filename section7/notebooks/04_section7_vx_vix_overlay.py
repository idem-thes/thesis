"""Section 7.2.1 figure: VX (FRD front-month continuous) vs cash VIX over the
calibration period (train_start = 2012-07-10 through train_end = 2025-02-28).


"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import yfinance as yf  # noqa: E402

VX_PATH = (
    REPO_ROOT / "data" / "VX" / "VX_main" / "VX_1_DAY" / "VX_full_1day_continuous_UNadjusted.txt"
)
OUT_DIR = REPO_ROOT / "outputs" / "_results" / "section7"
OUT_PDF = OUT_DIR / "vx_vix_calibration.pdf"
OUT_HTML = OUT_DIR / "vx_vix_calibration.html"

CALIB_START = "2012-07-10"
CALIB_END = "2025-02-28"

VX_COLOR = "#d62728"
VIX_COLOR = "#1f77b4"


def _load_vx_daily() -> pd.Series:
    df = pd.read_csv(
        VX_PATH,
        header=None,
        names=["date", "open", "high", "low", "close", "volume", "oi"],
        parse_dates=["date"],
    )
    return df.set_index("date")["close"].sort_index()


def _load_vix_daily(start: str, end: str) -> pd.Series:
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def main() -> None:
    print(f"Loading FRD VX daily from {VX_PATH.name} ...")
    vx = _load_vx_daily()
    print(f"  VX raw: {len(vx):,} bars, {vx.index.min().date()} -> {vx.index.max().date()}")

    print(f"Fetching ^VIX from yfinance ({CALIB_START} -> {CALIB_END}) ...")
    vix = _load_vix_daily(CALIB_START, CALIB_END)
    print(f"  VIX raw: {len(vix):,} bars, {vix.index.min().date()} -> {vix.index.max().date()}")

    joined = pd.concat({"VX": vx, "VIX": vix}, axis=1).dropna()
    joined = joined.loc[CALIB_START:CALIB_END]
    print(
        f"  Inner-joined: {len(joined):,} bars, {joined.index.min().date()} -> {joined.index.max().date()}"
    )
    print(f"  mean |VX - VIX| basis: {(joined['VX'] - joined['VIX']).abs().mean():.2f} vol points")
    print(f"  Pearson corr(VX, VIX): {joined.corr().iloc[0, 1]:.4f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=joined.index,
            y=joined["VIX"],
            mode="lines",
            name="VIX (cash)",
            line=dict(color=VIX_COLOR, width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=joined.index,
            y=joined["VX"],
            mode="lines",
            name="VX (front-month continuous)",
            line=dict(color=VX_COLOR, width=1.2),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volatility points",
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
