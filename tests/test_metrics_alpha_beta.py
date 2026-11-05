"""Unit tests for CAPM α/β — guard against scale bugs like 4-digit α."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.backtest.metrics import capm_alpha_beta, sharpe_ratio, summarize_returns


def test_capm_alpha_beta_on_toy_series():
    # Construct r = 0.001 + 0.5 * m + noise (daily), so ann α ≈ 0.001*365, β≈0.5
    rng = np.random.default_rng(0)
    n = 2000
    m = rng.normal(0.0004, 0.01, size=n)
    eps = rng.normal(0.0, 0.001, size=n)
    r = 0.001 + 0.5 * m + eps
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    alpha, beta = capm_alpha_beta(
        pd.Series(r, index=idx),
        pd.Series(m, index=idx),
        risk_free=0.0,
        periods_per_year=365,
        annualize=True,
    )
    assert 0.4 < beta < 0.6
    # Annualized alpha should be near 0.001 * 365 = 0.365 (± noise)
    assert 0.2 < alpha < 0.55
    # Definitely not thousands of percent
    assert abs(alpha) < 2.0


def test_capm_period_alpha_not_annualized():
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    m = pd.Series(np.linspace(-0.01, 0.01, 100), index=idx)
    r = 0.002 + 1.0 * m
    alpha_p, beta = capm_alpha_beta(r, m, annualize=False)
    assert abs(beta - 1.0) < 1e-9
    assert abs(alpha_p - 0.002) < 1e-9


def test_sharpe_and_summary_finite():
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    r = pd.Series(0.001 + 0.01 * np.random.default_rng(1).normal(size=100), index=idx)
    s = sharpe_ratio(r)
    assert np.isfinite(s)
    summary = summarize_returns(r, benchmark=r * 0.5)
    assert "alpha_ann" in summary
    assert abs(summary["alpha_ann"]) < 5  # sanity
