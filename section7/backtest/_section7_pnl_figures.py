"""Section 7.1 PnL figures under the cancel-aware maker - curves + surface."""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

pio.kaleido.scope.mathjax = None

_PT = REPO_ROOT / "outputs" / "_cache" / "section7_pct_trading"
_FG = REPO_ROOT / "outputs" / "_cache" / "section7_forecast_grid"
_OUT = REPO_ROOT / "outputs" / "_results" / "section7"
TZ = "America/New_York"
GATES = [0.05, 0.10, 0.20]
HORIZONS_MIN = [1, 5, 10, 15, 30, 60]
SEC = {1: 60, 5: 300, 10: 600, 15: 900, 30: 1800, 60: 3600}
CURVE_H_MIN = 15  # 900 s - the Section 8-matched horizon
GATE_COLORS = {0.05: "#1b9e77", 0.10: "#d95f02", 0.20: "#7570b3"}


def _thr(h_min: int, pct: float) -> float:
    fs = pd.read_parquet(_FG / f"forecast_set_h{h_min}.parquet")
    return float(np.quantile(fs["forecast_vx_pts"].abs().to_numpy(), 1.0 - pct))


def build_curves(trades: pd.DataFrame) -> go.Figure:
    sub = trades[(trades["h_min"] == CURVE_H_MIN) & (trades["filled"])].copy()
    sub["entry_ts"] = pd.DatetimeIndex(pd.to_datetime(sub["entry_ts"])).tz_convert(TZ)
    absf = sub["forecast_vx_pts"].abs().to_numpy()
    x0, x1 = sub["entry_ts"].min(), sub["entry_ts"].max()
    fig = go.Figure()
    for g in GATES:
        thr = _thr(CURVE_H_MIN, g)
        cell = sub[absf >= thr].sort_values("entry_ts")
        if cell.empty:
            continue
        cum = cell["pnl"].cumsum().to_numpy()
        x = pd.DatetimeIndex([x0]).append(pd.DatetimeIndex(cell["entry_ts"]))
        y = np.concatenate([[0.0], cum])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=f"top {int(round(g*100))}% (n = {len(cell)})",
            line=dict(color=GATE_COLORS[g], width=1.8, shape="hv"),
            hovertemplate=f"top {int(round(g*100))}%<br>entry: %{{x}}<br>cum PnL: $%{{y:,.0f}}<extra></extra>"))
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_xaxes(title_text="Entry date (ET)", range=[x0, x1])
    fig.update_yaxes(title_text="Cumulative maker PnL ($)", tickformat="$,.0f", zeroline=False)
    fig.update_layout(template="plotly_white", width=1000, height=540,
                      margin=dict(l=80, r=30, t=40, b=90), font=dict(size=12),
                      legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center",
                                  bgcolor="rgba(255,255,255,0.85)", bordercolor="gray",
                                  borderwidth=0.5, font=dict(size=11)))
    return fig


def _fmt(v: float) -> str:
    if np.isnan(v):
        return ""
    if abs(v) >= 10000:
        k = v / 1000.0
        return f"{k:+.1f}k" if abs(k) < 100 else f"{k:+.0f}k"
    return f"{v:+,.0f}"


def build_surface(trd: pd.DataFrame) -> go.Figure:
    x_labels = [f"{int(round(g*100))}%" for g in GATES]
    y_labels = [f"{SEC[h]} s" if h <= 15 else f"{h} min" for h in HORIZONS_MIN]
    mat = np.full((len(HORIZONS_MIN), len(GATES)), np.nan)
    for i, h in enumerate(HORIZONS_MIN):
        for j, g in enumerate(GATES):
            r = trd[(trd["h_min"] == h) & (np.isclose(trd["top_pct"], g))]
            if not r.empty:
                mat[i, j] = float(r["total"].iloc[0])
    zmax = float(np.nanmax(np.abs(mat)))
    fig = go.Figure(go.Heatmap(
        z=mat, x=x_labels, y=y_labels, colorscale="RdYlGn", zmid=0.0, zmin=-zmax, zmax=zmax,
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Maker PnL ($)", side="right", font=dict(size=11)),
                      tickformat="$,.0f", tickfont=dict(size=9), thickness=14),
        hovertemplate="gate = %{x}<br>h = %{y}<br>PnL = $%{z:,.0f}<extra></extra>"))
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) / zmax > 0.55 else "#222222"
            fig.add_annotation(x=x, y=y, text=_fmt(v), showarrow=False,
                               font=dict(size=11, color=color, family="Arial"))
    fig.update_xaxes(title_text="Conviction gate (top %)", type="category")
    fig.update_yaxes(title_text="Horizon h", type="category")
    fig.update_layout(template="plotly_white", plot_bgcolor="#dddddd", width=620, height=560,
                      margin=dict(l=70, r=30, t=30, b=70), font=dict(size=12))
    return fig


def main() -> int:
    trades = pd.read_parquet(_PT / "pct_trades.parquet")
    trd = pd.read_parquet(_PT / "pct_trading.parquet")
    _OUT.mkdir(parents=True, exist_ok=True)
    cur = build_curves(trades)
    cur.write_image(str(_OUT / "pnl_curves_pct.pdf"), format="pdf", width=cur.layout.width, height=cur.layout.height)
    print(f"wrote {_OUT/'pnl_curves_pct.pdf'}")
    srf = build_surface(trd)
    srf.write_image(str(_OUT / "pnl_surface_pct.pdf"), format="pdf", width=srf.layout.width, height=srf.layout.height)
    print(f"wrote {_OUT/'pnl_surface_pct.pdf'}   grid total = ${np.nansum(trd['total']):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
