from __future__ import annotations

import pandas as pd

from stat_arb.config import CostConfig
from stat_arb.costs.model import apply_costs, turnover_from_weights


def test_costs_reduce_returns():
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    gross = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=idx)
    weights = pd.DataFrame({"A": [0, 1, 1, 0, 0], "B": [0, -1, -1, 0, 0]}, index=idx)
    net, breakdown = apply_costs(gross, weights, CostConfig(fee_bps_per_side=10, slippage_bps_per_side=0))
    assert (net <= gross + 1e-12).all()
    assert breakdown.total.sum() > 0
    turn = turnover_from_weights(weights)
    assert turn.iloc[1] > 0
