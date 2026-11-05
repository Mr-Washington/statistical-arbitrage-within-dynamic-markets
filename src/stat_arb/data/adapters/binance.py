"""Binance public klines adapter (no API secrets on argv).

Uses the public REST market-data endpoints. Responses are cached to CSV under
``cache_dir`` so research is reproducible without re-hitting the network.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

from stat_arb.data.adapters.base import MarketDataAdapter
from stat_arb.data.schema import Instrument, validate_bars

STABLE_BASES = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "USDE"}

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class BinancePublicKlinesAdapter(MarketDataAdapter):
    """Fetch OHLCV from Binance public `/api/v3/klines` (spot)."""

    name = "binance_public"

    def __init__(
        self,
        *,
        base_url: str = "https://api.binance.com",
        cache_dir: str | Path = "data/cache/binance",
        session: requests.Session | None = None,
        request_pause_s: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.request_pause_s = request_pause_s

    def _cache_path(self, symbol: str, start: str, end: str, freq: str) -> Path:
        key = hashlib.sha256(f"{symbol}|{start}|{end}|{freq}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{symbol}_{freq}_{key}.csv"

    def _fetch_klines(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> list[list]:
        url = f"{self.base_url}/api/v3/klines"
        out: list[list] = []
        cursor = start_ms
        step = INTERVAL_MS[interval]
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            out.extend(batch)
            last_open = int(batch[-1][0])
            nxt = last_open + step
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(self.request_pause_s)
        return out

    @staticmethod
    def _klines_to_frame(symbol: str, raw: list[list]) -> pd.DataFrame:
        rows = []
        for k in raw:
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                    "instrument_id": symbol,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[7]),
                    "funding_rate": pd.NA,
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
        if freq not in INTERVAL_MS:
            raise ValueError(f"unsupported freq {freq}; choose from {sorted(INTERVAL_MS)}")
        start_ts = self._as_utc(start)
        end_ts = self._as_utc(end)
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            cache = self._cache_path(symbol, str(start_ts.date()), str(end_ts.date()), freq)
            if cache.exists():
                frames.append(validate_bars(pd.read_csv(cache, parse_dates=["timestamp"])))
                continue
            raw = self._fetch_klines(
                symbol,
                int(start_ts.timestamp() * 1000),
                int(end_ts.timestamp() * 1000),
                freq,
            )
            frame = self._klines_to_frame(symbol, raw)
            if not frame.empty:
                frame.to_csv(cache, index=False)
                meta = cache.with_suffix(".json")
                meta.write_text(
                    json.dumps(
                        {
                            "symbol": symbol,
                            "start": str(start_ts),
                            "end": str(end_ts),
                            "freq": freq,
                            "rows": len(frame),
                        },
                        indent=2,
                    )
                )
            frames.append(frame)
        if not frames:
            return validate_bars(pd.DataFrame())
        return validate_bars(pd.concat(frames, ignore_index=True))

    def get_universe(self, as_of: str | pd.Timestamp) -> list[Instrument]:
        """Best-effort liquid USDT spot set from exchangeInfo (point-in-time approx)."""
        del as_of  # listing history not in public exchangeInfo; filter later via bars
        url = f"{self.base_url}/api/v3/exchangeInfo"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        info = resp.json()
        out: list[Instrument] = []
        for s in info.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            base = s.get("baseAsset", "")
            out.append(
                Instrument(
                    instrument_id=s["symbol"],
                    asset_class="crypto",
                    quote_currency="USDT",
                    base_symbol=base,
                    is_stable=base in STABLE_BASES,
                )
            )
        return out
