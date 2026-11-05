"""Kalman-filter dynamic hedge ratio (random-walk intercept & slope).

State: θ_t = [α_t, β_t]'
Observation: y_t = α_t + β_t x_t + ε_t
Transition: θ_t = θ_{t-1} + η_t
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge(
    y: pd.Series,
    x: pd.Series,
    *,
    delta: float = 1e-4,
    obs_cov: float = 1e-3,
) -> pd.DataFrame:
    """Causal Kalman filter; parameters at t use observations ≤ t only.

    Parameters
    ----------
    delta :
        Process-noise scale for the random-walk state (larger → faster adaptation).
    obs_cov :
        Observation noise variance R.
    """
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    n = len(aligned)
    alphas = np.zeros(n)
    betas = np.zeros(n)
    spreads = np.zeros(n)

    # State covariance prior
    P = np.zeros((2, 2))
    theta = np.zeros(2)  # [alpha, beta]
    R = float(obs_cov)
    # Process noise Q = delta / (1 - delta) * I  (common pairs-trading choice)
    q = delta / (1.0 - delta) if 0 < delta < 1 else float(delta)
    Q = q * np.eye(2)

    yv = aligned["y"].to_numpy(dtype=float)
    xv = aligned["x"].to_numpy(dtype=float)

    for t in range(n):
        # Predict
        theta_pred = theta
        P_pred = P + Q
        # Observation matrix
        F = np.array([1.0, xv[t]])
        y_hat = float(F @ theta_pred)
        # Innovation
        err = yv[t] - y_hat
        S = float(F @ P_pred @ F.T + R)
        K = (P_pred @ F.T) / S
        # Update
        theta = theta_pred + K * err
        P = P_pred - np.outer(K, F) @ P_pred
        alphas[t] = theta[0]
        betas[t] = theta[1]
        spreads[t] = err  # residual using pre-update prediction (causal)

    return pd.DataFrame(
        {"alpha": alphas, "beta": betas, "spread": spreads},
        index=aligned.index,
    )


def fit_kalman_hedge(
    y: pd.Series,
    x: pd.Series,
    *,
    delta: float = 1e-4,
    obs_cov: float = 1e-3,
) -> tuple[float, float, pd.DataFrame]:
    """Return terminal (alpha, beta) and full causal path."""
    path = kalman_hedge(y, x, delta=delta, obs_cov=obs_cov)
    if path.empty:
        raise ValueError("empty series for Kalman hedge")
    return float(path["alpha"].iloc[-1]), float(path["beta"].iloc[-1]), path
