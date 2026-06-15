"""Section 8 run-2 adverse-selection decomposition: P(fill|move) and mean PnL by move.

"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from code_section8_run2._smoke import HORIZONS_S, PCTILE_GRID  # noqa: E402

_OUT = REPO_ROOT / "outputs" / "_results" / "section8_run2"


def _tex_int(v: float) -> str:
    """Signed integer with LaTeX thousands separator, e.g. -23{,}032 / +1{,}966."""
    return f"{v:+,.0f}".replace(",", "{,}")


def _cell(oos: pd.DataFrame, trades: pd.DataFrame, pct: float) -> dict:
    absp = oos["y_pred"].abs().to_numpy()
    thr = float(np.quantile(absp, 1.0 - pct))
    fired = oos[absp >= thr].copy()
    fired = fired[fired["vx_ask"] > fired["vx_bid"]]  # valid-book filter (as in the backtest)

    filled_ts = set(trades["block_ts"])
    fired["filled"] = fired["block_ts"].isin(filled_ts)
    fav = (np.sign(fired["y_true"]) == np.sign(fired["y_pred"])) & (fired["y_true"] != 0)
    adv = (np.sign(fired["y_true"]) == -np.sign(fired["y_pred"])) & (fired["y_true"] != 0)
    p_adv = float(fired.loc[adv, "filled"].mean())
    p_fav = float(fired.loc[fav, "filled"].mean())

    tfav = (np.sign(trades["y_true"]) == np.sign(trades["y_pred"])) & (trades["y_true"] != 0)
    tadv = (np.sign(trades["y_true"]) == -np.sign(trades["y_pred"])) & (trades["y_true"] != 0)
    return {
        "n_fire": int(len(fired)),
        "n_filled": int(fired["filled"].sum()),
        "p_fill_adv": p_adv,
        "p_fill_fav": p_fav,
        "ratio": p_adv / p_fav,
        "pnl_adv": float(trades.loc[tadv, "pnl"].mean()),
        "pnl_fav": float(trades.loc[tfav, "pnl"].mean()),
        "total": float(trades["pnl"].sum()),
    }


def main() -> int:
    mt = pd.read_parquet(_OUT / "maker_trades.parquet")
    ref = pd.read_parquet(_OUT / "maker_trading.parquet")  # n_fire / n_filled
    cp = pd.read_parquet(_OUT / "conviction_pnl.parquet")  # maker total

    rows = []
    for h in HORIZONS_S:
        oos = pd.read_parquet(_OUT / f"oos_predictions_h{h}s.parquet")
        ycols = oos[["block_ts", "y_pred", "y_true"]]
        for pct in PCTILE_GRID:
            trades = mt[(mt["horizon_s"] == h) & (np.isclose(mt["top_pct"], pct))].merge(
                ycols, on="block_ts", how="left"
            )
            c = _cell(oos, trades, pct)
            r = ref[(ref["horizon_s"] == h) & (np.isclose(ref["top_pct"], pct))].iloc[0]
            mk = float(cp[(cp["horizon_s"] == h) & (np.isclose(cp["top_pct"], pct))]["maker"].iloc[0])
            assert c["n_fire"] == int(r["n_fire"]), f"n_fire mismatch h={h} pct={pct}"
            assert c["n_filled"] == int(r["n_filled"]), f"n_filled mismatch h={h} pct={pct}"
            assert abs(c["total"] - mk) < 1.0, f"total mismatch h={h} pct={pct}"
            rows.append({"horizon_s": h, "top_pct": pct, **c})
    df = pd.DataFrame(rows)

    a = df[(df["horizon_s"] == 40) & (np.isclose(df["top_pct"], 0.10))].iloc[0]
    assert abs(a["p_fill_adv"] - 0.953) < 0.005 and abs(a["p_fill_fav"] - 0.635) < 0.005
    assert abs(a["ratio"] - 1.50) < 0.02

    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nLaTeX rows for tab:section8_trading (gate & n_fire & adv & fav & ratio & PnL_adv & PnL_fav & total):")
    for _, r in df.iterrows():
        print(
            f"top ${int(round(r['top_pct'] * 100))}\\%$ & ${_tex_int(r['n_fire']).lstrip('+')}$ "
            f"& ${r['p_fill_adv']:.2f}$ & ${r['p_fill_fav']:.2f}$ & ${r['ratio']:.2f}$ "
            f"& ${r['pnl_adv']:+.0f}$ & ${r['pnl_fav']:+.0f}$ & ${_tex_int(r['total'])}$ \\\\"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
