from __future__ import annotations

import pandas as pd
import pytest

from stat_arb.config import UniverseConfig
from stat_arb.data.adapters.synthetic import SyntheticCryptoAdapter
from stat_arb.data.schema import bars_to_close_panel
from stat_arb.data.universe import point_in_time_universe


def test_excludes_stables_and_uses_trailing_only():
    adapter = SyntheticCryptoAdapter(n_days=200, seed=0, include_stables=True)
    bars = adapter.get_bars(adapter.all_symbols(), adapter.start, adapter.start + pd.Timedelta(days=199))
    as_of = bars["timestamp"].min() + pd.Timedelta(days=100)
    uni = point_in_time_universe(
        bars,
        as_of,
        UniverseConfig(min_listing_days=30, min_adv_usd=1.0, max_zero_volume_frac=0.5, exclude_stables=True),
    )
    assert "USDCUSDT" not in uni
    assert "BTCUSDT" in uni
    # Point-in-time: names listed after as_of should not appear
    late_only = bars.loc[bars["timestamp"] > as_of, "instrument_id"].unique()
    early = set(bars.loc[bars["timestamp"] <= as_of, "instrument_id"])
    for sym in late_only:
        if sym not in early:
            assert sym not in uni


def test_no_bfill_in_panel():
    adapter = SyntheticCryptoAdapter(n_days=120, seed=1)
    # Include BTC (lists day 0) and FFF (lists day 80) so the wide panel has leading NaNs for FFF
    bars = adapter.get_bars(
        ["BTCUSDT", "FFFUSDT"], adapter.start, adapter.start + pd.Timedelta(days=119)
    )
    closes = bars_to_close_panel(bars, ffill_limit=3, allow_bfill=False)
    assert closes["FFFUSDT"].isna().sum() > 0
    with pytest.raises(ValueError, match="bfill"):
        bars_to_close_panel(bars, allow_bfill=True)
