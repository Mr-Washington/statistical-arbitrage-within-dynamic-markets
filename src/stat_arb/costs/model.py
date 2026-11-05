"""Explicit transaction-cost and funding model (bps)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stat_arb.config import CostConfig


@dataclass(frozen=True)
class CostBreakdown:
    trade_cost: pd.Series
    funding_cost: pd.Series
    total: pd.Series


def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover = 0.5 * L1 change in weights (per bar)."""
    dw = weights.fillna(0.0).diff().abs().sum(axis=1)
    return 0.5 * dw


def apply_costs(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
    config: CostConfig | None = None,
    funding_rates: pd.DataFrame | None = None,
) -> tuple[pd.Series, CostBreakdown]:
    """Subtract trading + funding costs from gross portfolio returns.

    Trading cost per bar ≈ turnover * (fee + slippage) bps * 1e-4 * 2 sides handled
    via per-side bps on each traded notional unit (turnover already one-way).
    """
    cfg = config or CostConfig()
    w = weights.reindex(index=gross_returns.index).fillna(0.0)
    turn = turnover_from_weights(w)
    bps = cfg.fee_bps_per_side + cfg.slippage_bps_per_side
    # Each unit of one-way turnover pays bps on the traded notional (one side).
    # Enter+exit both appear in turnover over the life of the trade.
    trade_cost = turn * (bps * 1e-4)

    funding_cost = pd.Series(0.0, index=gross_returns.index)
    if funding_rates is not None and not funding_rates.empty:
        # funding PnL ≈ -position_weight * funding_rate (long pays when rate > 0)
        fr = funding_rates.reindex(index=gross_returns.index).fillna(0.0)
        common = [c for c in w.columns if c in fr.columns]
        if common:
            funding_cost = -(w[common] * fr[common]).sum(axis=1)
    elif cfg.funding_bps_per_day != 0.0:
        # Flat research default when adapter marks funding unavailable
        gross_exposure = w.abs().sum(axis=1)
        funding_cost = gross_exposure * (cfg.funding_bps_per_day * 1e-4)

    total = trade_cost + funding_cost
    net = gross_returns.fillna(0.0) - total
    return net, CostBreakdown(trade_cost=trade_cost, funding_cost=funding_cost, total=total)
