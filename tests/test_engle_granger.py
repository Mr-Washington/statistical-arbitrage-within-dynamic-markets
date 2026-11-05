from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.cointegration.engle_granger import engle_granger, residual_spread
from stat_arb.cointegration.screening import screen_pairs
from stat_arb.data.adapters.synthetic import SyntheticCryptoAdapter
from stat_arb.data.schema import bars_to_close_panel


def test_eg_detects_cointegrated_pair():
    rng = np.random.default_rng(1)
    n = 300
    x = np.cumsum(rng.normal(0, 1, n))
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.85 * eps[t - 1] + rng.normal()
    y = 1.0 + 1.2 * x + eps
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    res = engle_granger(pd.Series(y, index=idx), pd.Series(x, index=idx), y_name="Y", x_name="X")
    assert res.adf_pvalue < 0.05
    s = residual_spread(pd.Series(y, index=idx), pd.Series(x, index=idx), res.alpha, res.beta)
    assert len(s) == n


def test_screen_finds_synthetic_pairs():
    adapter = SyntheticCryptoAdapter(n_days=400, seed=42)
    bars = adapter.get_bars(adapter.all_symbols(), adapter.start, adapter.start + pd.Timedelta(days=399))
    closes = bars_to_close_panel(bars, ffill_limit=3)
    # Lower ADV threshold for synthetic volumes
    from stat_arb.config import PairScreenConfig, UniverseConfig
    from stat_arb.data.universe import point_in_time_universe

    uni = point_in_time_universe(
        bars,
        closes.index[-1],
        UniverseConfig(min_listing_days=40, min_adv_usd=1.0, max_zero_volume_frac=0.5),
    )
    pairs = screen_pairs(closes, uni, PairScreenConfig(lookback=120, min_corr=0.3, adf_pvalue_max=0.1))
    names = {(p.y, p.x) for p in pairs} | {(p.x, p.y) for p in pairs}
    # Expect at least one of the planted cointegrated pairs
    assert ("AAAUSDT", "BBBUSDT") in names or ("CCCUSDT", "DDDUSDT") in names
