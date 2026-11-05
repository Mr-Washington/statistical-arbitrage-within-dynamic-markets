"""Static / expanding OLS hedge ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.cointegration.engle_granger import ols_hedge


def fit_ols_hedge(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    alpha, beta, _ = ols_hedge(aligned.iloc[:, 0].to_numpy(), aligned.iloc[:, 1].to_numpy())
    return alpha, beta


def expanding_ols_beta(y: pd.Series, x: pd.Series, min_obs: int = 30) -> pd.DataFrame:
    """Causal expanding-window OLS; beta at t uses observations ≤ t only."""
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    alphas = np.full(len(aligned), np.nan)
    betas = np.full(len(aligned), np.nan)
    for i in range(min_obs - 1, len(aligned)):
        a, b, _ = ols_hedge(aligned["y"].iloc[: i + 1].to_numpy(), aligned["x"].iloc[: i + 1].to_numpy())
        alphas[i] = a
        betas[i] = b
    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=aligned.index)
