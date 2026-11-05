"""CLI entry: run a reproducible synthetic walk-forward demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stat_arb.backtest.walkforward import run_walk_forward
from stat_arb.config import (
    BacktestConfig,
    CostConfig,
    PairScreenConfig,
    StrategyConfig,
    UniverseConfig,
    WalkForwardConfig,
    ZScoreConfig,
)
from stat_arb.data.adapters.synthetic import SyntheticCryptoAdapter
from stat_arb.data.schema import bars_to_close_panel
from stat_arb.portfolio.beta import beta_report


def _demo_config(hedge_method: str) -> StrategyConfig:
    """Frozen P0 defaults tuned for the synthetic demo panel."""
    return StrategyConfig(
        universe=UniverseConfig(
            min_listing_days=40,
            min_adv_usd=1.0,
            max_zero_volume_frac=0.5,
            exclude_stables=True,
        ),
        screen=PairScreenConfig(lookback=120, min_corr=0.3, adf_pvalue_max=0.05, max_pairs=5),
        zscore=ZScoreConfig(window=40, z_entry=1.75, z_exit=0.4, z_stop=4.0, max_holding_days=25),
        costs=CostConfig(fee_bps_per_side=5.0, slippage_bps_per_side=2.0, funding_bps_per_day=0.0),
        backtest=BacktestConfig(hedge_method=hedge_method, execution_lag=1),  # type: ignore[arg-type]
        walk_forward=WalkForwardConfig(train_bars=180, test_bars=60, step_bars=60, embargo_bars=1),
    )


def run_demo(hedge_method: str = "ols", seed: int = 42, out_dir: str = "Research/artifacts") -> dict:
    adapter = SyntheticCryptoAdapter(n_days=500, seed=seed)
    symbols = adapter.all_symbols()
    bars = adapter.get_bars(symbols, adapter.start, adapter.start + pd.Timedelta(days=499))
    closes = bars_to_close_panel(bars, ffill_limit=3, allow_bfill=False)
    btc = closes["BTCUSDT"].pct_change()

    funding = (
        bars.pivot(index="timestamp", columns="instrument_id", values="funding_rate")
        .sort_index()
        .astype(float)
    )

    cfg_ols = _demo_config("ols")
    cfg_kalman = _demo_config("kalman")

    reports = {}
    for name, cfg in (("ols", cfg_ols), ("kalman", cfg_kalman)):
        if hedge_method not in ("both", name):
            continue
        wf = run_walk_forward(closes, bars, benchmark=btc, config=cfg, funding=funding)
        br = beta_report(wf.combined_net, btc) if len(wf.combined_net) else {}
        reports[name] = {
            "n_folds": len(wf.folds),
            "pairs_per_fold": [len(f.pairs) for f in wf.folds],
            "summary_gross": {k: v for k, v in wf.summary_gross.items() if k != "label"},
            "summary_net": {k: v for k, v in wf.summary_net.items() if k != "label"},
            "beta_monitor": {
                k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else None)
                for k, v in br.items()
                if k != "rolling_beta"
            },
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "demo_walkforward_results.json"
    path.write_text(json.dumps(reports, indent=2, default=str))
    print(json.dumps(reports, indent=2, default=str))
    print(f"\nWrote {path}")
    return reports


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Statistical arbitrage demo backtest")
    p.add_argument("--hedge", choices=["ols", "kalman", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="Research/artifacts")
    args = p.parse_args(argv)
    run_demo(hedge_method=args.hedge, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
