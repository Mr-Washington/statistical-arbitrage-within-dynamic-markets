"""Ensure execution lag prevents same-bar look-ahead fills."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.backtest.engine import run_pairs_backtest
from stat_arb.cointegration.engle_granger import EngleGrangerResult
from stat_arb.config import BacktestConfig, PairScreenConfig, StrategyConfig, ZScoreConfig
from stat_arb.data.adapters.synthetic import SyntheticCryptoAdapter
from stat_arb.data.schema import bars_to_close_panel


def test_execution_lag_shifts_weights():
    adapter = SyntheticCryptoAdapter(n_days=300, seed=42)
    bars = adapter.get_bars(["AAAUSDT", "BBBUSDT", "BTCUSDT"], adapter.start, adapter.start + pd.Timedelta(days=299))
    closes = bars_to_close_panel(bars, ffill_limit=3)
    pair = EngleGrangerResult(
        y="BBBUSDT",
        x="AAAUSDT",
        alpha=2.0,
        beta=1.4,
        adf_stat=-5.0,
        adf_pvalue=0.001,
        n_obs=200,
        residual_std=1.0,
    )
    cfg = StrategyConfig(
        backtest=BacktestConfig(hedge_method="ols", execution_lag=1),
        zscore=ZScoreConfig(window=40, z_entry=1.5, z_exit=0.5, z_stop=5.0, max_holding_days=30),
        screen=PairScreenConfig(use_log_prices=False),
    )
    result = run_pairs_backtest(closes, ["AAAUSDT", "BBBUSDT"], cfg, freeze_pairs=[pair])
    # First bar weights must be flat (no look-ahead): lag means first decision cannot earn bar-0
    assert (result.weights.iloc[0].abs() == 0).all()


def test_ols_vs_kalman_ablation_runs():
    adapter = SyntheticCryptoAdapter(n_days=350, seed=3)
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "BTCUSDT"]
    bars = adapter.get_bars(symbols, adapter.start, adapter.start + pd.Timedelta(days=349))
    closes = bars_to_close_panel(bars, ffill_limit=3)
    pair = EngleGrangerResult(
        y="BBBUSDT",
        x="AAAUSDT",
        alpha=2.0,
        beta=1.4,
        adf_stat=-5.0,
        adf_pvalue=0.001,
        n_obs=200,
        residual_std=1.0,
    )
    out = {}
    for method in ("ols", "kalman"):
        cfg = StrategyConfig(
            backtest=BacktestConfig(hedge_method=method, execution_lag=1),
            screen=PairScreenConfig(use_log_prices=False),
        )
        res = run_pairs_backtest(closes, symbols, cfg, freeze_pairs=[pair])
        out[method] = float(res.net_returns.sum())
        assert np.isfinite(out[method])
    # Both produce a defined path (values may differ — that is the ablation)
    assert "ols" in out and "kalman" in out


def test_walk_forward_smoke():
    from stat_arb.backtest.walkforward import run_walk_forward
    from stat_arb.config import UniverseConfig, WalkForwardConfig, PairScreenConfig

    adapter = SyntheticCryptoAdapter(n_days=400, seed=42)
    bars = adapter.get_bars(adapter.all_symbols(), adapter.start, adapter.start + pd.Timedelta(days=399))
    closes = bars_to_close_panel(bars, ffill_limit=3)
    cfg = StrategyConfig(
        universe=UniverseConfig(min_listing_days=40, min_adv_usd=1.0, max_zero_volume_frac=0.5),
        screen=PairScreenConfig(lookback=80, min_corr=0.3, adf_pvalue_max=0.1, max_pairs=5),
        walk_forward=WalkForwardConfig(train_bars=150, test_bars=40, step_bars=40, embargo_bars=1),
        backtest=BacktestConfig(hedge_method="ols"),
    )
    wf = run_walk_forward(closes, bars, benchmark=closes["BTCUSDT"].pct_change(), config=cfg)
    assert len(wf.folds) >= 1
    assert wf.summary_net.get("n_bars", 0) > 0
