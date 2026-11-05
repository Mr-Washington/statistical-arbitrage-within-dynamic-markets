"""Engle–Granger cointegration test and residual spread."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class EngleGrangerResult:
    y: str
    x: str
    alpha: float
    beta: float
    adf_stat: float
    adf_pvalue: float
    n_obs: int
    residual_std: float

    @property
    def passes(self) -> bool:
        return self.adf_pvalue <= 0.05 and np.isfinite(self.beta)


def ols_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS of y on [1, x]. Returns alpha, beta, residuals."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if len(y) < 10:
        raise ValueError("need at least 10 observations for OLS hedge")
    X = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - (alpha + beta * x)
    return alpha, beta, resid


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    *,
    y_name: str | None = None,
    x_name: str | None = None,
    maxlag: int | None = None,
) -> EngleGrangerResult:
    """ADF on OLS residual of y ~ alpha + beta x."""
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    alpha, beta, resid = ols_hedge(aligned["y"].to_numpy(), aligned["x"].to_numpy())
    # autolag AIC; regression='c' for residual mean
    try:
        adf = adfuller(resid, maxlag=maxlag, autolag="AIC", result_object=True)
        adf_stat = float(adf.statistic)
        adf_pvalue = float(adf.pvalue)
    except TypeError:
        adf = adfuller(resid, maxlag=maxlag, autolag="AIC")
        adf_stat = float(adf[0])
        adf_pvalue = float(adf[1])
    return EngleGrangerResult(
        y=y_name or str(y.name or "y"),
        x=x_name or str(x.name or "x"),
        alpha=alpha,
        beta=beta,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        n_obs=len(resid),
        residual_std=float(np.std(resid, ddof=1)),
    )


def residual_spread(
    y: pd.Series,
    x: pd.Series,
    alpha: float,
    beta: float,
) -> pd.Series:
    """s_t = y_t - alpha - beta x_t (level residual)."""
    aligned = pd.concat([y, x], axis=1, join="inner")
    aligned.columns = ["y", "x"]
    s = aligned["y"] - alpha - beta * aligned["x"]
    s.name = "spread"
    return s
