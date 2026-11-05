"""Performance metrics with unit-tested CAPM α / β formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(r: pd.Series, m: pd.Series) -> pd.DataFrame:
    df = pd.concat([r, m], axis=1, join="inner").dropna()
    df.columns = ["r", "m"]
    return df


def capm_alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
    annualize: bool = True,
) -> tuple[float, float]:
    """CAPM regression on excess returns:

        r_p - r_f = α + β (r_m - r_f) + ε

    When ``annualize`` is True, α is returned as annualized simple rate
    ``α_period * periods_per_year`` (not a bogus percentage scale).
    β is dimensionless.
    """
    df = _aligned(portfolio_returns, benchmark_returns)
    if len(df) < 3:
        return float("nan"), float("nan")
    # risk_free is annualized; convert to per-period
    rf_period = risk_free / periods_per_year
    y = df["r"].to_numpy(dtype=float) - rf_period
    x = df["m"].to_numpy(dtype=float) - rf_period
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = np.sum((x - x_mean) ** 2)
    if var_x <= 0:
        return float("nan"), float("nan")
    beta = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
    alpha_period = float(y_mean - beta * x_mean)
    alpha = alpha_period * periods_per_year if annualize else alpha_period
    return alpha, beta


def sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 2:
        return float("nan")
    rf = risk_free / periods_per_year
    excess = r - rf
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / sd)


def sortino_ratio(
    returns: pd.Series,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 2:
        return float("nan")
    rf = risk_free / periods_per_year
    excess = r - rf
    downside = excess[excess < 0]
    dd = downside.std(ddof=1) if len(downside) else 0.0
    if dd == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / dd)


def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    """Return (max_dd, duration_in_bars)."""
    r = returns.fillna(0.0)
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else float("nan")
    # duration: longest consecutive period under water
    under = dd < 0
    dur = 0
    best = 0
    for flag in under:
        if flag:
            dur += 1
            best = max(best, dur)
        else:
            dur = 0
    return max_dd, best


def cagr(returns: pd.Series, periods_per_year: int = 365) -> float:
    r = returns.fillna(0.0)
    if len(r) == 0:
        return float("nan")
    equity = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or equity <= 0:
        return float("nan")
    return float(equity ** (1.0 / years) - 1.0)


def summarize_returns(
    returns: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
    label: str = "strategy",
) -> dict:
    vol = float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year)) if returns.dropna().size else float("nan")
    mdd, mdd_dur = max_drawdown(returns)
    out = {
        "label": label,
        "n_bars": int(returns.dropna().shape[0]),
        "cagr": cagr(returns, periods_per_year),
        "vol_ann": vol,
        "sharpe": sharpe_ratio(returns, risk_free=risk_free, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, risk_free=risk_free, periods_per_year=periods_per_year),
        "max_drawdown": mdd,
        "max_dd_duration_bars": mdd_dur,
        "calmar": (cagr(returns, periods_per_year) / abs(mdd)) if mdd and mdd < 0 else float("nan"),
        "hit_rate": float((returns.dropna() > 0).mean()) if returns.dropna().size else float("nan"),
    }
    if benchmark is not None:
        a, b = capm_alpha_beta(
            returns, benchmark, risk_free=risk_free, periods_per_year=periods_per_year, annualize=True
        )
        out["alpha_ann"] = a
        out["beta"] = b
    return out
