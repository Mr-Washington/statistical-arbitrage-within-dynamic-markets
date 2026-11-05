"""Unified bar schema for multi-asset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

BAR_COLUMNS = (
    "timestamp",
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "funding_rate",
    "borrow_rate",
    "adjusted",
)


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    asset_class: str = "crypto"
    quote_currency: str = "USDT"
    base_symbol: str = ""
    is_stable: bool = False
    tick_size: float = 0.0
    lot_size: float = 0.0


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=list(BAR_COLUMNS))


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate a long-form OHLCV panel."""
    if df.empty:
        return empty_bars()
    out = df.copy()
    missing = [c for c in ("timestamp", "instrument_id", "close") if c not in out.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "funding_rate", "borrow_rate"):
        if col not in out.columns:
            out[col] = pd.NA if col in ("funding_rate", "borrow_rate") else 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "adjusted" not in out.columns:
        out["adjusted"] = False
    out = out.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
    return out[list(BAR_COLUMNS)]


def bars_to_close_panel(
    bars: pd.DataFrame,
    *,
    ffill_limit: int = 3,
    allow_bfill: bool = False,
) -> pd.DataFrame:
    """Wide close panel; capped ffill only (no bfill in research paths)."""
    if allow_bfill:
        raise ValueError("bfill is forbidden in research paths")
    validated = validate_bars(bars)
    wide = validated.pivot(index="timestamp", columns="instrument_id", values="close").sort_index()
    if ffill_limit and ffill_limit > 0:
        wide = wide.ffill(limit=ffill_limit)
    return wide


def bars_to_volume_panel(bars: pd.DataFrame) -> pd.DataFrame:
    validated = validate_bars(bars)
    return validated.pivot(index="timestamp", columns="instrument_id", values="volume").sort_index()


def filter_as_of(bars: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Drop any bar after the as-of clock (no future leakage)."""
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    validated = validate_bars(bars)
    return validated.loc[validated["timestamp"] <= as_of].copy()


def instrument_ids(bars: pd.DataFrame) -> list[str]:
    return sorted(validate_bars(bars)["instrument_id"].unique().tolist())


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
