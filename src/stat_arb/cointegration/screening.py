"""Pair screening: correlation pre-gate → Engle–Granger."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from stat_arb.cointegration.engle_granger import EngleGrangerResult, engle_granger
from stat_arb.config import PairScreenConfig


def screen_pairs(
    closes: pd.DataFrame,
    universe: list[str],
    config: PairScreenConfig | None = None,
) -> list[EngleGrangerResult]:
    """Screen candidate pairs on the trailing window ending at the last row.

    Uses only the provided ``closes`` window (caller must slice train data).
    Correlation is a pre-gate only — cointegration is the eligibility proof.
    Log prices are used when ``use_log_prices`` so hedge β stays scale-stable.
    """
    cfg = config or PairScreenConfig()
    cols = [c for c in universe if c in closes.columns]
    if len(cols) < 2:
        return []
    raw = closes[cols].tail(cfg.lookback).dropna(how="any", axis=0)
    if len(raw) < max(30, cfg.lookback // 3):
        return []
    window = np.log(raw.clip(lower=1e-12)) if cfg.use_log_prices else raw

    corr = raw.pct_change().dropna().corr()
    results: list[EngleGrangerResult] = []
    for a, b in combinations(cols, 2):
        if abs(float(corr.loc[a, b])) < cfg.min_corr:
            continue
        med_a = float(raw[a].median())
        med_b = float(raw[b].median())
        if min(med_a, med_b) <= 0:
            continue
        scale_ratio = max(med_a, med_b) / min(med_a, med_b)
        if scale_ratio > cfg.max_price_scale_ratio:
            continue
        # Primary regression: higher-vol name as y (dependent)
        vol_a = float(raw[a].pct_change().std())
        vol_b = float(raw[b].pct_change().std())
        if vol_a >= vol_b:
            y_name, x_name = a, b
        else:
            y_name, x_name = b, a
        try:
            primary = engle_granger(window[y_name], window[x_name], y_name=y_name, x_name=x_name)
        except Exception:
            continue
        chosen = primary
        if cfg.reverse_regression:
            try:
                reverse = engle_granger(window[x_name], window[y_name], y_name=x_name, x_name=y_name)
                # Prefer lower ADF p-value (stronger residual stationarity)
                if reverse.adf_pvalue < chosen.adf_pvalue:
                    chosen = reverse
            except Exception:
                pass
        if not (chosen.adf_pvalue <= cfg.adf_pvalue_max and np.isfinite(chosen.beta)):
            continue
        if not (cfg.min_abs_beta <= abs(chosen.beta) <= cfg.max_abs_beta):
            continue
        results.append(chosen)

    results.sort(key=lambda r: r.adf_pvalue)
    return results[: cfg.max_pairs]
