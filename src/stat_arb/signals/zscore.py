"""Causal rolling z-score and mean-reversion trade rules."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from stat_arb.config import ZScoreConfig


class Signal(Enum):
    FLAT = 0
    LONG_SPREAD = 1   # long y / short β x  when z is low
    SHORT_SPREAD = -1  # short y / long β x when z is high


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """z_t uses mean/std estimated on (t-window, t] — past-only via rolling."""
    mu = spread.rolling(window=window, min_periods=max(10, window // 3)).mean()
    sd = spread.rolling(window=window, min_periods=max(10, window // 3)).std(ddof=1)
    z = (spread - mu) / sd.replace(0.0, np.nan)
    z.name = "zscore"
    return z


def generate_positions(
    z: pd.Series,
    config: ZScoreConfig | None = None,
) -> pd.Series:
    """State machine: enter at |z|>=z_in, exit at |z|<=z_out, stop at |z|>=z_stop.

    Position is the intended spread side known at the close of bar t
    (execution lag applied by the backtester).
    """
    cfg = config or ZScoreConfig()
    pos = np.zeros(len(z), dtype=float)
    state = 0
    hold = 0
    values = z.to_numpy(dtype=float)
    for i, zi in enumerate(values):
        if not np.isfinite(zi):
            pos[i] = 0.0
            state = 0
            hold = 0
            continue
        if state == 0:
            if zi >= cfg.z_entry:
                state = -1  # short spread
                hold = 1
            elif zi <= -cfg.z_entry:
                state = 1  # long spread
                hold = 1
        else:
            hold += 1
            stop = abs(zi) >= cfg.z_stop
            mean_revert_exit = abs(zi) <= cfg.z_exit
            time_stop = hold > cfg.max_holding_days
            if stop or mean_revert_exit or time_stop:
                state = 0
                hold = 0
        pos[i] = float(state)
    return pd.Series(pos, index=z.index, name="position")
