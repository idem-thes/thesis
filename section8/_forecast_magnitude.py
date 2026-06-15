"""Section 8 run-2 forecast-magnitude quantiles: distribution of |E_hat[Delta VX]| by horizon.

"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from code_section8_run2._smoke import HORIZONS_S  # noqa: E402

_OUT = REPO_ROOT / "outputs" / "_results" / "section8_run2"
_QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]
_TICK = 0.05  # VX bid-ask spread = one tick


def main() -> int:
    skill = pd.read_parquet(_OUT / "forecast_skill.parquet")  # canonical n per horizon
    rows = []
    for h in HORIZONS_S:
        o = pd.read_parquet(_OUT / f"oos_predictions_h{h}s.parquet")
        a = o.loc[o["y_true"].notna(), "y_pred"].abs()  # scoreable set
        n_ref = int(skill.loc[skill["horizon_s"] == h, "n"].iloc[0])
        assert len(a) == n_ref, f"n mismatch h={h}: {len(a)} vs forecast_skill {n_ref}"
        q = a.quantile(_QUANTILES)
        rows.append({"horizon_s": h, **{f"p{int(p*100)}": q[p] for p in _QUANTILES}, "n": len(a)})
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    print(f"Quantiles of |y_pred| (VX points); one-tick spread = {_TICK}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nLaTeX rows for tab:section8_forecast_magnitude:")
    for _, r in df.iterrows():
        cells = " & ".join(f"${r[f'p{int(p*100)}']:.4f}$" for p in _QUANTILES)
        n = f"{int(r['n']):,}".replace(",", "{,}")
        print(f"${int(r['horizon_s'])}$ & {cells} & ${n}$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
