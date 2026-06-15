"""Section 7.1 figure: ES + VX front-month BBO mid overlay with cash-settlement markers.

"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import code_section7.data_loader as dl  # noqa: E402

ES_COLOR = "#1f77b4"
VX_COLOR = "#d62728"

# Cash-settlement timestamps (Table section7_expiries). tz = America/New_York.
EXPIRIES = [
    {"contract": "VXH5", "ts": "2025-03-19 09:00:00", "color": VX_COLOR},
    {"contract": "ESH5", "ts": "2025-03-21 09:30:00", "color": ES_COLOR},
    {"contract": "VXJ5", "ts": "2025-04-16 09:00:00", "color": VX_COLOR},
    {"contract": "VXK5", "ts": "2025-05-21 09:00:00", "color": VX_COLOR},
    {"contract": "VXM5", "ts": "2025-06-18 09:00:00", "color": VX_COLOR},
    {"contract": "ESM5", "ts": "2025-06-20 09:30:00", "color": ES_COLOR},
    {"contract": "VXN5", "ts": "2025-07-16 09:00:00", "color": VX_COLOR},
    {"contract": "VXQ5", "ts": "2025-08-20 09:00:00", "color": VX_COLOR},
]

WINDOW_START = "2025-03-02"
WINDOW_END = "2025-08-29"
RESAMPLE_FREQ = "5min"

OUT_PDF = REPO_ROOT / "outputs" / "_results" / "section7" / "data_overlay.pdf"


def _load_resampled(product: str) -> pd.Series:
    df = dl.load_quotes(product, start=WINDOW_START, end=WINDOW_END)

    spread = df["ask_px"] - df["bid_px"]
    df = df[(spread >= 0) & (spread <= 0.05 * df["mid_px"])]
    return df.set_index("ts")["mid_px"].resample(RESAMPLE_FREQ).last().dropna()


def main() -> None:
    print(f"Loading ES quotes ({WINDOW_START} -> {WINDOW_END}) ...")
    es = _load_resampled("ES")
    print(f"  ES: {len(es):,} {RESAMPLE_FREQ} bars")

    print(f"Loading VX quotes ({WINDOW_START} -> {WINDOW_END}) ...")
    vx = _load_resampled("VX")
    print(f"  VX: {len(vx):,} {RESAMPLE_FREQ} bars")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=es.index,
            y=es.values,
            name="ES front-month mid (index pt)",
            line=dict(color=ES_COLOR, width=0.9),
            connectgaps=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=vx.index,
            y=vx.values,
            name="VX front-month mid (vol pt)",
            line=dict(color=VX_COLOR, width=0.9),
            connectgaps=False,
        ),
        secondary_y=True,
    )

    for exp in EXPIRIES:
        ts = pd.Timestamp(exp["ts"], tz="America/New_York")
        fig.add_shape(
            type="line",
            x0=ts,
            x1=ts,
            yref="paper",
            y0=0,
            y1=1,
            line=dict(color=exp["color"], width=0.9, dash="dash"),
        )
        # VX and ES expiries land 1-2 days apart (VXH5/ESH5, VXM5/ESM5), so
        # split the labels vertically: VX at the bottom of its line, ES at the
        # top. yanchor="bottom" keeps both clear of the axis frame.
        is_vx = exp["color"] == VX_COLOR
        fig.add_annotation(
            x=ts,
            y=0.015 if is_vx else 1.02,
            yref="paper",
            yanchor="bottom",
            text=exp["contract"],
            showarrow=False,
            font=dict(size=10, color=exp["color"]),
            xanchor="center",
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=70, r=70, t=40, b=60),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        xaxis_title="date",
    )
    fig.update_yaxes(
        title_text="ES BBO mid (S&P 500 index pt)",
        secondary_y=False,
        color=ES_COLOR,
        gridcolor="rgba(31,119,180,0.15)",
    )
    fig.update_yaxes(
        title_text="VX BBO mid (VIX vol pt)",
        secondary_y=True,
        color=VX_COLOR,
        gridcolor="rgba(214,39,40,0.15)",
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT_PDF), format="pdf", width=900, height=520, scale=2)
    print(f"Written: {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
