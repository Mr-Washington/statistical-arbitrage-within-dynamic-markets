"""Point-in-time liquid universe construction (no end-of-sample top-N)."""

from __future__ import annotations

import pandas as pd

from stat_arb.config import UniverseConfig
from stat_arb.data.schema import Instrument, validate_bars


def _base_is_stable(instrument_id: str, stables: tuple[str, ...]) -> bool:
    base = instrument_id.replace("USDT", "").replace("USD", "")
    return any(base == s or instrument_id.startswith(s) for s in stables)


def point_in_time_universe(
    bars: pd.DataFrame,
    as_of: str | pd.Timestamp,
    config: UniverseConfig | None = None,
    instruments: list[Instrument] | None = None,
) -> list[str]:
    """Eligibility at day ``as_of`` using only trailing information.

    Filters:
    - listing age (bars observed before as_of)
    - trailing ADV$
    - zero-volume fraction
    - min price
    - exclude stables as tradable legs
    """
    cfg = config or UniverseConfig()
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")

    validated = validate_bars(bars)
    hist = validated.loc[validated["timestamp"] <= as_of_ts]
    if hist.empty:
        return []

    stable_ids: set[str] = set()
    if instruments:
        stable_ids = {i.instrument_id for i in instruments if i.is_stable}
    if cfg.exclude_stables:
        for sym in hist["instrument_id"].unique():
            if _base_is_stable(str(sym), cfg.stables):
                stable_ids.add(str(sym))

    eligible: list[str] = []
    for sym, g in hist.groupby("instrument_id"):
        sym = str(sym)
        if cfg.exclude_stables and sym in stable_ids:
            continue
        g = g.sort_values("timestamp")
        if len(g) < cfg.min_listing_days:
            continue
        trailing = g.tail(cfg.min_listing_days)
        adv = float((trailing["quote_volume"]).mean())
        if adv < cfg.min_adv_usd:
            continue
        zero_frac = float((trailing["volume"].fillna(0.0) <= 0).mean())
        if zero_frac > cfg.max_zero_volume_frac:
            continue
        last_px = float(trailing["close"].iloc[-1])
        if last_px < cfg.min_price:
            continue
        eligible.append(sym)
    return sorted(eligible)
