"""Event-style vectorized pairs backtest with next-bar execution (no look-ahead)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stat_arb.cointegration.engle_granger import residual_spread
from stat_arb.cointegration.screening import screen_pairs
from stat_arb.config import StrategyConfig
from stat_arb.costs.model import apply_costs
from stat_arb.hedge.kalman import kalman_hedge
from stat_arb.hedge.ols import fit_ols_hedge
from stat_arb.portfolio.pairs import dollar_neutral_weights
from stat_arb.signals.zscore import generate_positions, rolling_zscore


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    weights: pd.DataFrame
    pair_meta: list[dict]
    equity_gross: pd.Series
    equity_net: pd.Series
    turnover: pd.Series


def _pair_returns(
    closes: pd.DataFrame,
    y: str,
    x: str,
    side: pd.Series,
    beta: float | pd.Series,
    *,
    lag: int,
    target_notional: float,
    log_hedge: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply lagged positions (dollar-neutral) with next-bar execution.

    When ``log_hedge`` is True (default, matching log-price EG/Kalman), PnL is
    return-space: ``side * (r_y − β r_x) / (1+|β|)``.
    Otherwise price-space residual ``Δy − β Δx`` normalized by gross notional.
    Execution: signal at close t fills at close t+lag (no same-bar look-ahead).
    """
    lagged_side = side.shift(lag)
    y_px = closes[y]
    x_px = closes[x]
    r_y = y_px.pct_change()
    r_x = x_px.pct_change()
    dy = y_px.diff()
    dx = x_px.diff()
    weights = pd.DataFrame(0.0, index=closes.index, columns=[y, x])
    pnl = pd.Series(0.0, index=closes.index, dtype=float)

    for ts in closes.index:
        s = lagged_side.loc[ts] if ts in lagged_side.index else np.nan
        if not np.isfinite(s) or s == 0:
            continue
        b = float(beta.loc[ts]) if isinstance(beta, pd.Series) else float(beta)
        if not np.isfinite(b):
            continue
        w = dollar_neutral_weights(float(s), b, y=y, x=x, target_notional=target_notional)
        weights.loc[ts, y] = w[y]
        weights.loc[ts, x] = w[x]
        if log_hedge:
            ry = r_y.loc[ts]
            rx = r_x.loc[ts]
            if not (np.isfinite(ry) and np.isfinite(rx)):
                continue
            denom = 1.0 + abs(b)
            pnl.loc[ts] = float(s) * target_notional * (float(ry) - b * float(rx)) / denom
        else:
            yp = y_px.shift(1).loc[ts]
            xp = x_px.shift(1).loc[ts]
            if not (np.isfinite(yp) and np.isfinite(xp) and yp != 0):
                continue
            gross_notional = abs(float(yp)) + abs(b) * abs(float(xp))
            if gross_notional <= 0:
                continue
            dollar = float(s) * target_notional * (float(dy.loc[ts]) - b * float(dx.loc[ts]))
            pnl.loc[ts] = dollar / gross_notional

    return pnl.fillna(0.0), weights.fillna(0.0)


def run_pairs_backtest(
    closes: pd.DataFrame,
    universe: list[str],
    config: StrategyConfig | None = None,
    *,
    funding: pd.DataFrame | None = None,
    freeze_pairs: list | None = None,
) -> BacktestResult:
    """Screen pairs on the full provided window then trade with causal z-scores.

    For walk-forward, pass only the OOS closes and ``freeze_pairs`` selected on train.
    """
    cfg = config or StrategyConfig()
    lag = cfg.backtest.execution_lag
    if freeze_pairs is None:
        pairs = screen_pairs(closes, universe, cfg.screen)
    else:
        pairs = freeze_pairs

    if not pairs:
        idx = closes.index
        empty_w = pd.DataFrame(0.0, index=idx, columns=universe[:1] or ["EMPTY"])
        z = pd.Series(0.0, index=idx)
        return BacktestResult(
            gross_returns=z,
            net_returns=z,
            weights=empty_w,
            pair_meta=[],
            equity_gross=pd.Series(1.0, index=idx),
            equity_net=pd.Series(1.0, index=idx),
            turnover=z,
        )

    all_weights: list[pd.DataFrame] = []
    meta: list[dict] = []
    gross_parts: list[pd.Series] = []

    log_hedge = cfg.screen.use_log_prices
    for i, pair in enumerate(pairs[: cfg.backtest.max_concurrent_pairs]):
        y, x = pair.y, pair.x
        if y not in closes.columns or x not in closes.columns:
            continue
        y_series = np.log(closes[y].clip(lower=1e-12)) if log_hedge else closes[y]
        x_series = np.log(closes[x].clip(lower=1e-12)) if log_hedge else closes[x]
        method = cfg.backtest.hedge_method
        if method == "kalman":
            path = kalman_hedge(
                y_series,
                x_series,
                delta=cfg.backtest.kalman_delta,
                obs_cov=cfg.backtest.kalman_obs_cov,
            )
            spread = path["spread"]
            # Freeze β for weights within the run to avoid daily turnover thrash;
            # spread/z still use the dynamic residual.
            beta_t = float(path["beta"].iloc[min(60, len(path) - 1)])
            alpha_t = float(path["alpha"].iloc[min(60, len(path) - 1)])
            if freeze_pairs is not None:
                alpha_t, beta_t = pair.alpha, pair.beta
            beta_for_weights: float | pd.Series = beta_t
        else:
            # Frozen train α/β when walk-forward; else fit on provided window.
            alpha_t, beta_t = pair.alpha, pair.beta
            if freeze_pairs is None:
                alpha_t, beta_t = fit_ols_hedge(y_series, x_series)
            spread = residual_spread(y_series, x_series, alpha_t, beta_t)
            beta_for_weights = beta_t

        z = rolling_zscore(spread, cfg.zscore.window)
        side = generate_positions(z, cfg.zscore)
        pr, w = _pair_returns(
            closes,
            y,
            x,
            side,
            beta_for_weights,
            lag=lag,
            target_notional=cfg.backtest.target_pair_notional,
            log_hedge=log_hedge,
        )
        gross_parts.append(pr)
        all_weights.append(w)
        meta.append(
            {
                "y": y,
                "x": x,
                "alpha": alpha_t,
                "beta": beta_t if not isinstance(beta_for_weights, pd.Series) else float(pd.Series(beta_for_weights).dropna().iloc[-1]),
                "hedge_method": method,
                "adf_pvalue": getattr(pair, "adf_pvalue", float("nan")),
            }
        )

    if not gross_parts:
        idx = closes.index
        z = pd.Series(0.0, index=idx)
        return BacktestResult(
            gross_returns=z,
            net_returns=z,
            weights=pd.DataFrame(0.0, index=idx),
            pair_meta=[],
            equity_gross=pd.Series(1.0, index=idx),
            equity_net=pd.Series(1.0, index=idx),
            turnover=z,
        )

    # Equal-weight across active pairs (gross book)
    gross = sum(gross_parts) / len(gross_parts)
    # Align weights into a single book
    cols = sorted({c for w in all_weights for c in w.columns})
    book = pd.DataFrame(0.0, index=closes.index, columns=cols)
    for w in all_weights:
        book[w.columns] = book[w.columns].add(w, fill_value=0.0)
    book = book / len(all_weights)

    net, breakdown = apply_costs(gross, book, cfg.costs, funding_rates=funding)
    equity_gross = (1.0 + gross.fillna(0.0)).cumprod()
    equity_net = (1.0 + net.fillna(0.0)).cumprod()
    turn = book.fillna(0.0).diff().abs().sum(axis=1) * 0.5

    return BacktestResult(
        gross_returns=gross.rename("gross"),
        net_returns=net.rename("net"),
        weights=book,
        pair_meta=meta,
        equity_gross=equity_gross.rename("equity_gross"),
        equity_net=equity_net.rename("equity_net"),
        turnover=turn.rename("turnover"),
    )
