"""Benchmark beta monitor (BTC for crypto)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.backtest.metrics import capm_alpha_beta


def rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Causal rolling CAPM β of portfolio vs benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["r", "m"]
    betas = []
    idx = []
    for i in range(len(aligned)):
        start = max(0, i - window + 1)
        if i - start + 1 < max(10, window // 3):
            betas.append(np.nan)
            idx.append(aligned.index[i])
            continue
        sub = aligned.iloc[start : i + 1]
        _, b = capm_alpha_beta(sub["r"], sub["m"], annualize=False)
        betas.append(b)
        idx.append(aligned.index[i])
    return pd.Series(betas, index=idx, name="rolling_beta")


def beta_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int = 60,
    target_abs_beta: float = 0.25,
) -> dict:
    """Summary stats for book-level BTC β monitoring."""
    path = rolling_beta(portfolio_returns, benchmark_returns, window=window)
    finite = path.dropna()
    alpha, beta = capm_alpha_beta(portfolio_returns, benchmark_returns, annualize=True)
    within = float((finite.abs() <= target_abs_beta).mean()) if len(finite) else float("nan")
    return {
        "full_sample_alpha_ann": alpha,
        "full_sample_beta": beta,
        "rolling_beta_mean": float(finite.mean()) if len(finite) else float("nan"),
        "rolling_beta_std": float(finite.std(ddof=1)) if len(finite) else float("nan"),
        "pct_days_abs_beta_within_target": within,
        "target_abs_beta": target_abs_beta,
        "rolling_beta": path,
    }
