"""Walk-forward validation with parameter freeze and embargo."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stat_arb.backtest.engine import BacktestResult, run_pairs_backtest
from stat_arb.backtest.metrics import summarize_returns
from stat_arb.cointegration.screening import screen_pairs
from stat_arb.config import StrategyConfig
from stat_arb.data.universe import point_in_time_universe


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    pairs: list
    result: BacktestResult
    metrics_gross: dict
    metrics_net: dict


@dataclass
class WalkForwardReport:
    folds: list[WalkForwardFold]
    combined_gross: pd.Series
    combined_net: pd.Series
    summary_gross: dict
    summary_net: dict


def iter_folds(
    index: pd.DatetimeIndex,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    embargo_bars: int = 1,
):
    """Yield (train_slice, test_slice) indexers with embargo gap."""
    i = 0
    n = len(index)
    fold_id = 0
    while True:
        train_start = i
        train_end = i + train_bars  # exclusive
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars
        if test_end > n:
            break
        yield fold_id, index[train_start:train_end], index[test_start:test_end]
        fold_id += 1
        i += step_bars


def run_walk_forward(
    closes: pd.DataFrame,
    bars: pd.DataFrame,
    benchmark: pd.Series | None = None,
    config: StrategyConfig | None = None,
    funding: pd.DataFrame | None = None,
) -> WalkForwardReport:
    """Train → freeze pairs + hedge params → OOS test → roll."""
    cfg = config or StrategyConfig()
    wf = cfg.walk_forward
    folds: list[WalkForwardFold] = []
    gross_parts: list[pd.Series] = []
    net_parts: list[pd.Series] = []

    for fold_id, train_idx, test_idx in iter_folds(
        closes.index,
        wf.train_bars,
        wf.test_bars,
        wf.step_bars,
        wf.embargo_bars,
    ):
        train_end = train_idx[-1]
        universe = point_in_time_universe(bars, train_end, cfg.universe)
        train_closes = closes.loc[train_idx]
        pairs = screen_pairs(train_closes, universe, cfg.screen)
        # Freeze: pairs (y,x,alpha,beta) from train only
        test_closes = closes.loc[test_idx]
        # Extend test window left with train lookback for z-score warm-start,
        # but only keep OOS returns for metrics.
        warm = cfg.zscore.window + cfg.screen.lookback
        warm_start_pos = max(0, closes.index.get_loc(test_idx[0]) - warm)
        warm_idx = closes.index[warm_start_pos : closes.index.get_loc(test_idx[-1]) + 1]
        warm_closes = closes.loc[warm_idx]

        fr = None
        if funding is not None:
            fr = funding.reindex(warm_closes.index)

        result = run_pairs_backtest(
            warm_closes,
            universe,
            cfg,
            funding=fr,
            freeze_pairs=pairs,
        )
        # Restrict to OOS
        g = result.gross_returns.reindex(test_idx).fillna(0.0)
        n = result.net_returns.reindex(test_idx).fillna(0.0)
        w = result.weights.reindex(test_idx).fillna(0.0)
        oos = BacktestResult(
            gross_returns=g,
            net_returns=n,
            weights=w,
            pair_meta=result.pair_meta,
            equity_gross=(1.0 + g).cumprod(),
            equity_net=(1.0 + n).cumprod(),
            turnover=result.turnover.reindex(test_idx).fillna(0.0),
        )
        bm = benchmark.reindex(test_idx) if benchmark is not None else None
        mg = summarize_returns(g, benchmark=bm, periods_per_year=cfg.backtest.annualization, label=f"fold{fold_id}_gross")
        mn = summarize_returns(n, benchmark=bm, periods_per_year=cfg.backtest.annualization, label=f"fold{fold_id}_net")
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                pairs=pairs,
                result=oos,
                metrics_gross=mg,
                metrics_net=mn,
            )
        )
        gross_parts.append(g)
        net_parts.append(n)

    if not folds:
        empty = pd.Series(dtype=float)
        return WalkForwardReport(
            folds=[],
            combined_gross=empty,
            combined_net=empty,
            summary_gross={},
            summary_net={},
        )

    combined_gross = pd.concat(gross_parts).sort_index()
    combined_gross = combined_gross[~combined_gross.index.duplicated(keep="first")]
    combined_net = pd.concat(net_parts).sort_index()
    combined_net = combined_net[~combined_net.index.duplicated(keep="first")]
    bm_all = benchmark.reindex(combined_gross.index) if benchmark is not None else None
    summary_gross = summarize_returns(
        combined_gross, benchmark=bm_all, periods_per_year=cfg.backtest.annualization, label="wf_gross"
    )
    summary_net = summarize_returns(
        combined_net, benchmark=bm_all, periods_per_year=cfg.backtest.annualization, label="wf_net"
    )
    return WalkForwardReport(
        folds=folds,
        combined_gross=combined_gross,
        combined_net=combined_net,
        summary_gross=summary_gross,
        summary_net=summary_net,
    )
