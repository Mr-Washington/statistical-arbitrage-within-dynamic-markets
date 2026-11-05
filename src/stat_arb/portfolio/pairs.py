"""Dollar-neutral pair weights from hedge ratio and spread side."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PairSpec:
    y: str
    x: str
    alpha: float
    beta: float
    hedge_method: str


def dollar_neutral_weights(
    side: float,
    beta: float,
    *,
    y: str,
    x: str,
    target_notional: float = 1.0,
) -> dict[str, float]:
    """Map spread side to asset weights with long notional ≈ |β| short notional.

    ``side > 0`` → long spread (long y, short β x)
    ``side < 0`` → short spread (short y, long β x)

    Dollar neutrality: |w_y| * P_y ≈ |β| * |w_x| * P_x in notional terms when
    weights are applied to returns of equal notional units. We set:
      w_y = side * target
      w_x = -side * beta * target
    so the β-scaled short matches the hedge ratio in price-space residual.
    """
    if side == 0 or not pd.notna(beta):
        return {y: 0.0, x: 0.0}
    s = 1.0 if side > 0 else -1.0
    return {
        y: s * target_notional,
        x: -s * float(beta) * target_notional,
    }


def aggregate_book_weights(
    pair_weights: list[dict[str, float]],
    *,
    max_gross: float | None = None,
) -> dict[str, float]:
    book: dict[str, float] = {}
    for w in pair_weights:
        for k, v in w.items():
            book[k] = book.get(k, 0.0) + v
    if max_gross is not None:
        gross = sum(abs(v) for v in book.values())
        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            book = {k: v * scale for k, v in book.items()}
    return book
