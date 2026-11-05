"""Market data adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import pandas as pd

from stat_arb.data.schema import Instrument


class MarketDataAdapter(ABC):
    """Pluggable market-data source. Auth / rate limits stay inside the adapter."""

    name: str = "base"

    @abstractmethod
    def get_bars(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """Return long-form bars matching the unified schema."""

    def get_funding(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        return None

    def get_borrow(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        return None

    def get_universe(self, as_of: str | pd.Timestamp) -> list[Instrument]:
        raise NotImplementedError

    def get_benchmark_bars(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        freq: str = "1d",
        symbol: str = "BTCUSDT",
    ) -> pd.DataFrame:
        return self.get_bars([symbol], start, end, freq=freq)
