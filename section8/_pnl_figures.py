"""Section 8 run-2 PnL figures: maker cumulative curves + maker PnL surface.

"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402


pio.kaleido.scope.mathjax = None



OUT_DIR = REPO_ROOT / "outputs" / "_results" / "section8_run2"
MAKER_TRADES = OUT_DIR / "maker_trades.parquet"
CONVICTION_PNL = OUT_DIR / "conviction_pnl.parquet"

OUT_CURVES_PDF = OUT_DIR / "pnl_curves_maker.pdf"
OUT_CURVES_HTML = OUT_DIR / "pnl_curves_maker.html"
OUT_SURFACE_PDF = OUT_DIR / "pnl_surface_maker.pdf"
OUT_SURFACE_HTML = OUT_DIR / "pnl_surface_maker.html"

TZ = "America/New_York"


HORIZONS_S = [40, 60, 300, 600, 900]
GATES = [0.05, 0.10, 0.20]

# Curves figure zooms into the only profitable horizon (900s), one curve per
# conviction gate (matches the fig:section8_pnl_curves caption).
CURVE_HORIZON = 900

# One distinct, print-safe colour per gate (ColorBrewer Dark2, fixed order).
GATE_COLORS = {
    0.05: "#1b9e77",
    0.10: "#d95f02",
    0.20: "#7570b3",
}




def build_curves_figure(trades: pd.DataFrame) -> tuple[go.Figure, dict]:
    """Single-panel cumulative maker PnL at h=900s, one hv-step curve per gate.

    900s is the only horizon in profit; each curve (top 5/10/20% by |E_hat|) is
    anchored at $0 at the earliest entry across the gates, then steps up/down at
    each filled trade.
    """
    sub = trades[trades["horizon_s"] == CURVE_HORIZON].copy()
    sub["block_ts"] = pd.DatetimeIndex(sub["block_ts"]).tz_convert(TZ)
    x0 = sub["block_ts"].min()
    x1 = sub["block_ts"].max()

    fig = go.Figure()
    diagnostics = {}

    for g in GATES:
        cell = sub[np.isclose(sub["top_pct"], g)].sort_values("block_ts")
        if cell.empty:
            continue
        cum = cell["pnl"].cumsum().to_numpy()
        x = pd.DatetimeIndex([x0]).append(pd.DatetimeIndex(cell["block_ts"]))
        y = np.concatenate([[0.0], cum])
        n = len(cell)
        pct = int(round(g * 100))
        diagnostics[g] = dict(n_trades=n, final_pnl=float(cum[-1]))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"top {pct}% (n = {n})",
                line=dict(color=GATE_COLORS[g], width=1.8, shape="hv"),
                hovertemplate=(
                    f"top {pct}%<br>entry: %{{x}}<br>cum PnL: $%{{y:,.0f}}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_xaxes(title_text="Entry date (ET)", range=[x0, x1])
    fig.update_yaxes(
        title_text="Cumulative maker PnL ($)",
        tickformat="$,.0f",
        zeroline=False,
    )
    fig.update_layout(
        template="plotly_white",
        width=1000,
        height=540,
        margin=dict(l=80, r=30, t=40, b=90),
        font=dict(size=12),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.18,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=0.5,
            font=dict(size=11),
        ),
    )
    return fig, diagnostics





def _format_cell(value: float) -> str:
    """Render a dollar value as e.g. ``+1,886`` or ``-31.3k`` (no $).

    For |value| >= 10000 collapse to k-format so the label fits in the cell.
    Mirrors the Section 7 surface notebook helper.
    """
    if np.isnan(value):
        return ""
    if abs(value) >= 10000:
        k = value / 1000.0
        if abs(k) < 100:
            return f"{k:+.1f}k"
        return f"{k:+.0f}k"
    return f"{value:+,.0f}"


def _annotate_heatmap(
    fig: go.Figure,
    mat: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    z_abs_max: float,
) -> None:
    """Per-cell dollar annotations; text flips white on saturated cells."""
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ratio = abs(v) / z_abs_max if z_abs_max > 0 else 0.0
            color = "white" if ratio > 0.55 else "#222222"
            fig.add_annotation(
                x=x,
                y=y,
                xref="x",
                yref="y",
                text=_format_cell(v),
                showarrow=False,
                font=dict(size=11, color=color, family="Arial"),
            )


def build_surface_figure(conviction: pd.DataFrame) -> tuple[go.Figure, dict]:
    """RdYlGn heatmap of maker PnL over (conviction gate, horizon).

    """
    x_labels = [f"{int(round(g * 100))}%" for g in GATES]

    y_labels = [f"{h} s" for h in HORIZONS_S]

    mat = np.full((len(HORIZONS_S), len(GATES)), np.nan, dtype=float)
    for i, h in enumerate(HORIZONS_S):
        for j, g in enumerate(GATES):
            row = conviction[
                (conviction["horizon_s"] == h) & (np.isclose(conviction["top_pct"], g))
            ]
            if not row.empty:
                mat[i, j] = float(row["maker"].iloc[0])

    z_abs_max = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    if z_abs_max == 0:
        z_abs_max = 1.0

    diagnostics = dict(
        total=float(np.nansum(mat)),
        n_profitable=int(np.nansum(mat > 0)),
        z_abs_max=z_abs_max,
    )

    fig = go.Figure(
        go.Heatmap(
            z=mat,
            x=x_labels,
            y=y_labels,
            colorscale="RdYlGn",
            zmid=0.0,
            zmin=-z_abs_max,
            zmax=z_abs_max,
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text="Maker PnL ($)", side="right", font=dict(size=11)),
                tickformat="$,.0f",
                tickfont=dict(size=9),
                thickness=14,
            ),
            hovertemplate="gate = %{x}<br>h = %{y} s<br>PnL = $%{z:,.0f}<extra></extra>",
        )
    )
    _annotate_heatmap(fig, mat, x_labels, y_labels, z_abs_max)

    fig.update_xaxes(title_text="Conviction gate (top %)", type="category")
    fig.update_yaxes(title_text="Horizon h", type="category")
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#dddddd",
        width=620,
        height=520,
        margin=dict(l=70, r=30, t=30, b=70),
        font=dict(size=12),
    )
    return fig, diagnostics



def main() -> int:
    trades = pd.read_parquet(MAKER_TRADES)
    conviction = pd.read_parquet(CONVICTION_PNL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building cumulative maker-PnL curves (h=900s, one per conviction gate) ...")
    curves_fig, curves_diag = build_curves_figure(trades)
    curves_fig.write_image(
        str(OUT_CURVES_PDF),
        format="pdf",
        width=curves_fig.layout.width,
        height=curves_fig.layout.height,
    )
    curves_fig.write_html(str(OUT_CURVES_HTML))
    print(f"  Wrote {OUT_CURVES_PDF}")
    for g in GATES:
        d = curves_diag.get(g)
        if d:
            print(f"    top {int(round(g * 100)):>2}%:  final PnL = ${d['final_pnl']:+,.0f}   n = {d['n_trades']}")

    print("\nBuilding maker-PnL surface heatmap (gate x horizon) ...")
    surface_fig, surface_diag = build_surface_figure(conviction)
    surface_fig.write_image(
        str(OUT_SURFACE_PDF),
        format="pdf",
        width=surface_fig.layout.width,
        height=surface_fig.layout.height,
    )
    surface_fig.write_html(str(OUT_SURFACE_HTML))
    print(f"  Wrote {OUT_SURFACE_PDF}")
    print(
        f"    grid total = ${surface_diag['total']:,.0f}   "
        f"profitable cells = {surface_diag['n_profitable']}/{len(HORIZONS_S) * len(GATES)}   "
        f"|z|max = ${surface_diag['z_abs_max']:,.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
