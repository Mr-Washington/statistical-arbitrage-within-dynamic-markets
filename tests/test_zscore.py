from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.config import ZScoreConfig
from stat_arb.signals.zscore import generate_positions, rolling_zscore


def test_zscore_entry_exit_stop():
    idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    # Craft z path: flat → spike to 2.5 → back to 0.2 → spike to 4.5 stop
    zvals = np.zeros(30)
    zvals[5] = 2.5
    zvals[6:10] = 2.2
    zvals[10] = 0.2
    zvals[15] = -2.5
    zvals[16:18] = -2.1
    zvals[18] = -4.5
    z = pd.Series(zvals, index=idx)
    pos = generate_positions(z, ZScoreConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0, max_holding_days=50))
    assert pos.iloc[5] == -1  # short spread on +z
    assert pos.iloc[10] == 0  # exited near mean
    assert pos.iloc[15] == 1  # long spread on -z
    assert pos.iloc[18] == 0  # stopped out


def test_rolling_zscore_past_only():
    idx = pd.date_range("2020-01-01", periods=80, freq="D", tz="UTC")
    s = pd.Series(np.arange(80, dtype=float), index=idx)
    z = rolling_zscore(s, window=20)
    # First values should be NaN until min periods
    assert z.isna().sum() > 0
    assert z.dropna().std() > 0
