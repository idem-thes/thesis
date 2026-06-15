"""Static Plotly renderer for 6.3 - six PDV"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._paths import OUTPUTS_ROOT
from .metrics import (
    cumulative_direction_correct,
    direction_correct_per_day,
    summary_metrics,
)

log = logging.getLogger(__name__)

DEFAULT_OUT_DIR = OUTPUTS_ROOT / "_results" / "section63_5model"

# (column, short panel label, full label, color). Order = model number.
MODELS: list[tuple[str, str, str, str]] = [
    ("sigma_std", "m1: Guyon", "Guyon SPX-only", "#888888"),
    ("sigma_rw", "m2: VIX RW", "VIX random walk", "#1f77b4"),
    ("sigma_intra", "m3: today VIX_op", "today VIX_open + intraday SDE", "#ff7f0e"),
    (
        "sigma_intra_overnight",
        "m4: yest VIX+ON",
        "yest VIX_close + ON + intraday SDE",
        "#2ca02c",
    ),
    ("sigma_multi", "m5: multi-asset", "multi-asset Guyon (SPX+SX5E+NKY)", "#9467bd"),
]

VIX_COLOR = "#000000"
WIDTH = 900
HEIGHT_WIDE_SINGLE = 320
HEIGHT_BPANEL_VERTICAL = 820  # 4 panels stacked vertically
HEIGHT_HSTRIP = 320  # 5 panels horizontal


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    ok["date"] = pd.to_datetime(ok["date"])
    ok = ok.set_index("date").sort_index()
    ok["vix_prev_close"] = ok["vix_close"].shift(1)
    return ok


def _base_layout(title: str, **kw) -> dict:
    layout = {
        "template": "plotly_white",
        "title": {"text": title, "font": {"size": 14}, "x": 0.5, "xanchor": "center"},
        "margin": {"l": 60, "r": 20, "t": 70, "b": 50},
        "font": {"size": 11},
    }
    layout.update(kw)
    return layout


def render_b(ok: pd.DataFrame) -> go.Figure:
    """4 panels stacked vertically: VIX + m1 + m_i overlay through time."""
    others = [m for m in MODELS if m[0] != "sigma_std"]
    m1_col, _m1_label, _m1_full, m1_color = MODELS[0]

    fig = make_subplots(
        rows=len(others),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[m[1] for m in others],
        vertical_spacing=0.06,
    )
    for i, (col, panel_label, _full, color) in enumerate(others, start=1):
        showlegend = i == 1
        fig.add_trace(
            go.Scatter(
                x=ok.index,
                y=ok["vix_close"] * 100,
                mode="lines",
                line={"color": VIX_COLOR, "width": 1.0},
                name="VIX close",
                showlegend=showlegend,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ok.index,
                y=ok[m1_col] * 100,
                mode="lines",
                line={"color": m1_color, "width": 1.0},
                name="m1 sigma_hat",
                showlegend=showlegend,
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ok.index,
                y=ok[col] * 100,
                mode="lines",
                line={"color": color, "width": 1.0},
                name=panel_label.split(":")[0] + " sigma_hat",
                showlegend=showlegend,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text="sigma (%)", row=i, col=1)

    fig.update_xaxes(title_text="date", row=len(others), col=1)
    fig.update_layout(
        **_base_layout("(b) m1 baseline vs each more-info model"),
        width=WIDTH,
        height=HEIGHT_BPANEL_VERTICAL,
        legend={"orientation": "h", "x": 0, "y": 1.06, "xanchor": "left", "yanchor": "bottom"},
    )
    return fig


def render_c(ok: pd.DataFrame) -> go.Figure:
    """1 wide panel: VIX_close + all 5 sigma_hat overlaid."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ok.index,
            y=ok["vix_close"] * 100,
            mode="lines",
            line={"color": VIX_COLOR, "width": 1.2},
            name="VIX close",
        )
    )
    for col, panel_label, _full, color in MODELS:
        fig.add_trace(
            go.Scatter(
                x=ok.index,
                y=ok[col] * 100,
                mode="lines",
                line={"color": color, "width": 0.9},
                opacity=0.85,
                name=panel_label,
            )
        )
    fig.update_layout(
        **_base_layout("(c) All 5 models - fit vs actual VIX_close"),
        width=WIDTH,
        height=HEIGHT_WIDE_SINGLE,
        xaxis_title="date",
        yaxis_title="sigma (%)",
        legend={"orientation": "h", "x": 0, "y": 1.06, "xanchor": "left", "yanchor": "bottom"},
    )
    return fig


def render_d(ok: pd.DataFrame) -> go.Figure:
    """1 wide panel: residuals (sigma_hat - VIX_close) overlaid for all 5 models."""
    fig = go.Figure()
    fig.add_hline(y=0, line={"color": "black", "width": 0.5})
    for col, panel_label, _full, color in MODELS:
        res = (ok[col] - ok["vix_close"]) * 100
        fig.add_trace(
            go.Scatter(
                x=ok.index,
                y=res,
                mode="lines",
                line={"color": color, "width": 0.7},
                opacity=0.85,
                name=panel_label,
            )
        )
    fig.update_layout(
        **_base_layout("(d) All 5 residuals through time"),
        width=WIDTH,
        height=HEIGHT_WIDE_SINGLE,
        xaxis_title="date",
        yaxis_title="residual (%)",
        legend={"orientation": "h", "x": 0, "y": 1.06, "xanchor": "left", "yanchor": "bottom"},
    )
    return fig


def render_e(ok: pd.DataFrame) -> go.Figure:
    """5 panels: sigma_hat vs VIX_close scatter, with y=x diagonal."""
    fig = make_subplots(
        rows=1,
        cols=len(MODELS),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[m[1] for m in MODELS],
        horizontal_spacing=0.025,
    )
    target_pct = ok["vix_close"] * 100
    lo = float(target_pct.min()) - 1
    hi = float(target_pct.max()) + 1
    for i, (col, panel_label, _full, color) in enumerate(MODELS, start=1):
        sub = ok[[col, "vix_close"]].dropna()
        fig.add_trace(
            go.Scatter(
                x=sub["vix_close"] * 100,
                y=sub[col] * 100,
                mode="markers",
                marker={"color": color, "size": 3.5, "opacity": 0.7},
                name=panel_label,
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line={"color": "black", "width": 0.6, "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text="VIX close (%)", row=1, col=i)
    fig.update_yaxes(title_text="sigma_hat (%)", row=1, col=1)
    fig.update_layout(
        **_base_layout("(e) Fit vs actual - scatter"),
        width=WIDTH,
        height=HEIGHT_HSTRIP,
    )
    return fig


def render_f(ok: pd.DataFrame) -> go.Figure:
    """5 panels: residual distribution histograms (shared x-axis range)."""
    residuals = {col: ((ok[col] - ok["vix_close"]) * 100).dropna() for col, *_ in MODELS}
    flat = pd.concat(residuals.values())
    lo = float(flat.quantile(0.005)) - 0.5
    hi = float(flat.quantile(0.995)) + 0.5
    bin_size = (hi - lo) / 40

    fig = make_subplots(
        rows=1,
        cols=len(MODELS),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[m[1] for m in MODELS],
        horizontal_spacing=0.025,
    )
    for i, (col, panel_label, _full, color) in enumerate(MODELS, start=1):
        res = residuals[col]
        fig.add_trace(
            go.Histogram(
                x=res,
                autobinx=False,
                xbins={"start": lo, "end": hi, "size": bin_size},
                marker={"color": color, "line": {"color": "white", "width": 0.4}},
                opacity=0.85,
                name=panel_label,
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        fig.add_vline(x=0, line={"color": "black", "width": 0.5}, row=1, col=i)
        fig.add_vline(
            x=float(res.mean()),
            line={"color": color, "width": 1.0, "dash": "dash"},
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text="residual (%)", row=1, col=i)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_layout(
        **_base_layout("(f) Residual distribution"),
        width=WIDTH,
        height=HEIGHT_HSTRIP,
    )
    return fig


def render_g(ok: pd.DataFrame, dc_curves: dict[str, pd.Series]) -> go.Figure:
    """5 panels: cumulative direction-correct rate per model + chance line."""
    fig = make_subplots(
        rows=1,
        cols=len(MODELS),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[m[1] for m in MODELS],
        horizontal_spacing=0.025,
    )
    for i, (col, panel_label, _full, color) in enumerate(MODELS, start=1):
        cum = dc_curves[col]
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                line={"color": color, "width": 1.2},
                name=panel_label,
                showlegend=False,
            ),
            row=1,
            col=i,
        )
        fig.add_hline(y=0.5, line={"color": "grey", "width": 0.6, "dash": "dash"}, row=1, col=i)
    fig.update_yaxes(range=[-0.02, 1.02], row=1, col=1, title_text="cumulative DC")
    for i in range(2, len(MODELS) + 1):
        fig.update_yaxes(range=[-0.02, 1.02], row=1, col=i)
    fig.update_layout(
        **_base_layout("(g) Cumulative direction-correct rate"),
        width=WIDTH,
        height=HEIGHT_HSTRIP,
    )
    return fig


def render_section63(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    """Render the 6 figures + metrics CSV. Returns the output dir."""
    out_dir = out_dir or DEFAULT_OUT_DIR
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ok = _prepare_frame(df)

    metrics_rows: list[dict] = []
    dc_curves: dict[str, pd.Series] = {}
    for col, panel_label, full_label, _color in MODELS:
        if col not in ok.columns:
            log.warning("column %s missing from frame, skipping", col)
            continue
        m = summary_metrics(ok[col], ok["vix_close"])
        dc_per_day = direction_correct_per_day(ok[col], ok["vix_close"], ok["vix_prev_close"])
        cum_dc = cumulative_direction_correct(dc_per_day)
        dc_curves[col] = cum_dc
        metrics_rows.append(
            {
                "model": panel_label.split(":")[0],
                "column": col,
                "label": full_label,
                **m,
                "dc_final": float(cum_dc.iloc[-1]) if len(cum_dc) else float("nan"),
            }
        )

    figures = {
        "b_baseline_overlay": render_b(ok),
        "c_all_models_fit": render_c(ok),
        "d_all_residuals": render_d(ok),
        "e_scatter": render_e(ok),
        "f_residual_dist": render_f(ok),
        "g_directional_accuracy": render_g(ok, dc_curves),
    }

    for name, fig in figures.items():
        out = plots_dir / f"{name}.pdf"
        fig.write_image(
            str(out),
            format="pdf",
            width=fig.layout.width,
            height=fig.layout.height,
        )
        log.info("wrote %s", out.name)

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    metrics_df.to_csv(out_dir / "metrics.csv")
    log.info("wrote metrics.csv (%d models)", len(metrics_rows))

    return out_dir


def main(residuals_path: Path | None = None) -> Path:
    if residuals_path is None:
        runs = sorted(OUTPUTS_ROOT.glob("*_intraday_vix_fit/residuals.parquet"))
        if not runs:
            raise FileNotFoundError("no residuals.parquet under outputs/<date>_intraday_vix_fit/")
        residuals_path = runs[-1]
    log.info("rendering from %s", residuals_path)
    df = pd.read_parquet(residuals_path)
    return render_section63(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = main()
    print(f"output written to {out}")
