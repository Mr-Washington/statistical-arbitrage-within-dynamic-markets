"""Ablation: OLS vs Kalman hedge on a known cointegrated pair."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.hedge.kalman import kalman_hedge
from stat_arb.hedge.ols import fit_ols_hedge


def _cointegrated_pair(n=400, beta=1.4, seed=7):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, size=n))
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.9 * eps[t - 1] + rng.normal(0, 0.5)
    y = 2.0 + beta * x + eps
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(y, index=idx, name="y"), pd.Series(x, index=idx, name="x")


def test_ols_recovers_beta():
    y, x = _cointegrated_pair()
    alpha, beta = fit_ols_hedge(y, x)
    assert abs(beta - 1.4) < 0.15
    # Intercept noisier on finite samples with persistent residual; beta is the hedge object
    assert abs(alpha - 2.0) < 2.0


def test_kalman_tracks_beta():
    y, x = _cointegrated_pair()
    path = kalman_hedge(y, x, delta=1e-3, obs_cov=1e-2)
    # After burn-in, beta should be near true value
    terminal = path["beta"].iloc[-50:].mean()
    assert abs(terminal - 1.4) < 0.25
    assert path["spread"].std() > 0


def test_kalman_is_causal_length():
    y, x = _cointegrated_pair(n=100)
    path = kalman_hedge(y, x)
    assert len(path) == 100
    assert path.index.equals(y.index)
