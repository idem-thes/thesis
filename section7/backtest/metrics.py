"""Performance metrics for the Section 7.2 real-data PnL backtest.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import norm

# Euler-Mascheroni constant (Bailey & Lopez de Prado 2014 notation: gamma_e).
_EULER_MASCHERONI: float = 0.5772156649015328

# CFE exchange fee: $1.10 per contract per side -> $2.20 per round-trip.
_CFE_FEE_PER_TRADE: float = 2.20

# Trading days per year for annualisation.
_TRADING_DAYS_PER_YEAR: int = 252




def trades_to_daily_returns(trades: list, initial_capital: float) -> pd.Series:
    """Aggregate per-trade PnL into daily return fractions.
    """
    if not trades:
        return pd.Series([], dtype=float)

    daily_pnl: dict = {}
    for trade in trades:
        day = trade.exit_ts.date()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + trade.pnl_dollars

    dates = sorted(daily_pnl.keys())
    pnl_values = [daily_pnl[d] for d in dates]
    daily_returns = [pnl / initial_capital for pnl in pnl_values]

    return pd.Series(daily_returns, index=dates, dtype=float)


def sharpe(daily_returns: pd.Series) -> float:
    """Annualised Sharpe ratio using sample standard deviation.
    """
    n = len(daily_returns)
    if n < 2:
        return float("nan")

    mean_r = float(daily_returns.mean())
    std_r = float(daily_returns.std(ddof=1))

    if std_r == 0.0:
        return float("nan")

    return mean_r / std_r * math.sqrt(_TRADING_DAYS_PER_YEAR)


def deflated_sharpe(sr: float, daily_returns: pd.Series, n_trials: int) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
    """
    t = len(daily_returns)
    if t < 4:
        return float("nan")

    # Sample skewness and excess kurtosis of the return series.
    skew = float(daily_returns.skew())
    # pandas .kurt() returns excess kurtosis (Fisher definition, Normal -> 0).
    kurt = float(daily_returns.kurt())

    # Variance of the Sharpe ratio estimator (Bailey & Lopez de Prado 2014 eq. 10).
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (t - 1)

    # Guard against numerical negatives (can occur for extreme params).
    if var_sr <= 0.0:
        var_sr = 1e-12

    std_sr = math.sqrt(var_sr)

    # Expected maximum Sharpe across N trials (eq. 11).
    gamma_e = _EULER_MASCHERONI
    n = float(n_trials)

    # Phi^-1(1 - 1/N) term - clip argument away from 0 and 1 to avoid +/-inf.
    arg1 = max(min(1.0 - 1.0 / n, 1.0 - 1e-15), 1e-15)
    # Phi^-1(1 - 1/(N*e)) term.
    arg2 = max(min(1.0 - 1.0 / (n * math.e), 1.0 - 1e-15), 1e-15)

    sr_0 = std_sr * ((1.0 - gamma_e) * norm.ppf(arg1) + gamma_e * norm.ppf(arg2))

    # DSR = Phi((SR - SR_0) / std(SR)).
    z = (sr - sr_0) / std_sr
    dsr = float(norm.cdf(z))

    return dsr


def hit_rate(trades: list) -> float:
    """Fraction of trades with strictly positive PnL.
    """
    n = len(trades)
    if n == 0:
        return float("nan")

    n_winners = sum(1 for t in trades if t.pnl_dollars > 0.0)
    return n_winners / n


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a positive fraction of the running peak.
    """
    if len(equity_curve) == 0:
        return float("nan")

    running_max = equity_curve.cummax()
    drawdowns = (running_max - equity_curve) / running_max

    return float(drawdowns.max())


def summarize(trades: list, initial_capital: float, n_trials: int) -> dict:
    """Aggregate all backtest metrics into a single result dict.
    """
    n_trades = len(trades)

    gross_pnl = sum(t.pnl_dollars for t in trades) if trades else 0.0
    total_fee = n_trades * _CFE_FEE_PER_TRADE
    net_pnl = gross_pnl - total_fee

    hr = hit_rate(trades)

    daily_returns = trades_to_daily_returns(trades, initial_capital)
    sr = sharpe(daily_returns)

    # DSR requires at least 4 observations; fall back to nan for very short series.
    if len(daily_returns) >= 4 and not math.isnan(sr):
        dsr = deflated_sharpe(sr=sr, daily_returns=daily_returns, n_trials=n_trials)
    else:
        dsr = float("nan")

    # Equity curve: start at initial_capital, then add each day's PnL.
    if len(daily_returns) > 0:
        cumulative_pnl = (daily_returns * initial_capital).cumsum()
        equity_curve = initial_capital + cumulative_pnl
        mdd = max_drawdown(equity_curve)
    else:
        mdd = float("nan")

    final_equity = initial_capital + net_pnl

    return {
        "n_trades": int(n_trades),
        "gross_pnl_dollars": float(gross_pnl),
        "net_pnl_dollars": float(net_pnl),
        "hit_rate": float(hr),
        "sharpe": float(sr),
        "deflated_sharpe": float(dsr),
        "max_drawdown_pct": float(mdd),
        "final_equity": float(final_equity),
    }
