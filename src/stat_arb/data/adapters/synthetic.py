"""Synthetic cointegrated crypto-like panel for reproducible research / tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from stat_arb.data.adapters.base import MarketDataAdapter
from stat_arb.data.schema import Instrument, validate_bars


STABLE_BASES = {"USDT", "USDC", "BUSD", "DAI"}


class SyntheticCryptoAdapter(MarketDataAdapter):
    """Generate cointegrated pairs + noise names with a shared BTC factor."""

    name = "synthetic_crypto"

    def __init__(
        self,
        *,
        n_days: int = 500,
        seed: int = 42,
        start: str = "2022-01-01",
        include_stables: bool = True,
    ) -> None:
        self.n_days = n_days
        self.seed = seed
        self.start = pd.Timestamp(start, tz="UTC")
        self.include_stables = include_stables
        self._panel = self._build_panel()

    def _build_panel(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        idx = pd.date_range(self.start, periods=self.n_days, freq="D", tz="UTC")
        # Common crypto factor (BTC-like)
        btc_ret = 0.0005 + 0.02 * rng.standard_normal(self.n_days)
        btc = 20_000 * np.cumprod(1.0 + btc_ret)
        eth = 0.06 * btc * np.cumprod(1.0 + 0.005 * rng.standard_normal(self.n_days))
        # Cointegrated pair: Y = alpha + beta X + AR(1) residual
        x = 50.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal(self.n_days) + 0.3 * btc_ret)
        eps = np.zeros(self.n_days)
        shock = 0.5 * rng.standard_normal(self.n_days)
        for t in range(1, self.n_days):
            eps[t] = 0.85 * eps[t - 1] + shock[t]
        y = 2.0 + 1.4 * x + eps
        # Second cointegrated pair
        a = 10.0 * np.cumprod(1.0 + 0.012 * rng.standard_normal(self.n_days) + 0.25 * btc_ret)
        eps2 = np.zeros(self.n_days)
        shock2 = 0.4 * rng.standard_normal(self.n_days)
        for t in range(1, self.n_days):
            eps2[t] = 0.8 * eps2[t - 1] + shock2[t]
        b = -1.0 + 0.9 * a + eps2
        # Non-cointegrated noise names
        noise1 = 5.0 * np.cumprod(1.0 + 0.03 * rng.standard_normal(self.n_days))
        noise2 = 8.0 * np.cumprod(1.0 + 0.025 * rng.standard_normal(self.n_days) + 0.1 * btc_ret)

        series = {
            "BTCUSDT": btc,
            "ETHUSDT": eth,
            "AAAUSDT": x,
            "BBBUSDT": y,
            "CCCUSDT": a,
            "DDDUSDT": b,
            "EEEUSDT": noise1,
            "FFFUSDT": noise2,
        }
        if self.include_stables:
            series["USDCUSDT"] = np.ones(self.n_days) + 1e-4 * rng.standard_normal(self.n_days)

        rows: list[dict] = []
        for sym, closes in series.items():
            vol = rng.uniform(1e5, 5e6, size=self.n_days)
            # Stagger listing ages for point-in-time universe
            list_offset = {
                "BTCUSDT": 0,
                "ETHUSDT": 0,
                "AAAUSDT": 10,
                "BBBUSDT": 10,
                "CCCUSDT": 30,
                "DDDUSDT": 30,
                "EEEUSDT": 50,
                "FFFUSDT": 80,
                "USDCUSDT": 0,
            }.get(sym, 0)
            for i, ts in enumerate(idx):
                if i < list_offset:
                    continue
                c = float(closes[i])
                o = c * (1.0 + 0.001 * rng.standard_normal())
                h = max(o, c) * (1.0 + abs(0.002 * rng.standard_normal()))
                l = min(o, c) * (1.0 - abs(0.002 * rng.standard_normal()))
                v = float(vol[i])
                rows.append(
                    {
                        "timestamp": ts,
                        "instrument_id": sym,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": v,
                        "quote_volume": v * c,
                        "funding_rate": 0.0001 * rng.standard_normal(),
                        "borrow_rate": pd.NA,
                        "adjusted": False,
                    }
                )
        return validate_bars(pd.DataFrame(rows))

    @staticmethod
    def _as_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            return t.tz_localize("UTC")
        return t.tz_convert("UTC")

    def get_bars(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        freq: str = "1d",
    ) -> pd.DataFrame:
        del freq  # synthetic is daily
        start_ts = self._as_utc(start)
        end_ts = self._as_utc(end)
        df = self._panel
        mask = (
            df["instrument_id"].isin(list(symbols))
            & (df["timestamp"] >= start_ts)
            & (df["timestamp"] <= end_ts)
        )
        return df.loc[mask].copy()

    def get_funding(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame | None:
        bars = self.get_bars(symbols, start, end)
        if bars.empty:
            return None
        return bars[["timestamp", "instrument_id", "funding_rate"]].copy()

    def get_universe(self, as_of: str | pd.Timestamp) -> list[Instrument]:
        as_of_ts = self._as_utc(as_of)
        hist = self._panel.loc[self._panel["timestamp"] <= as_of_ts]
        out: list[Instrument] = []
        for sym, g in hist.groupby("instrument_id"):
            base = str(sym).replace("USDT", "")
            out.append(
                Instrument(
                    instrument_id=str(sym),
                    asset_class="crypto",
                    quote_currency="USDT",
                    base_symbol=base,
                    is_stable=base in STABLE_BASES,
                )
            )
        return out

    def all_symbols(self) -> list[str]:
        return sorted(self._panel["instrument_id"].unique().tolist())
