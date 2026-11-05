"""Default research parameters frozen for P0 walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


HedgeMethod = Literal["ols", "kalman"]


@dataclass(frozen=True)
class UniverseConfig:
    min_listing_days: int = 60
    min_adv_usd: float = 1_000_000.0
    max_zero_volume_frac: float = 0.20
    min_price: float = 0.01
    exclude_stables: bool = True
    stables: tuple[str, ...] = (
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "TUSD",
        "FDUSD",
        "USDP",
        "USDE",
    )


@dataclass(frozen=True)
class PairScreenConfig:
    lookback: int = 120
    adf_pvalue_max: float = 0.05
    min_corr: float = 0.5
    max_pairs: int = 10
    reverse_regression: bool = True
    use_log_prices: bool = True
    min_abs_beta: float = 0.05
    max_abs_beta: float = 5.0
    max_price_scale_ratio: float = 50.0


@dataclass(frozen=True)
class ZScoreConfig:
    window: int = 60
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    max_holding_days: int = 20


@dataclass(frozen=True)
class CostConfig:
    fee_bps_per_side: float = 10.0
    funding_bps_per_day: float = 0.0  # set when perp funding available
    slippage_bps_per_side: float = 2.0


@dataclass(frozen=True)
class BacktestConfig:
    hedge_method: HedgeMethod = "ols"
    kalman_delta: float = 1e-4
    kalman_obs_cov: float = 1e-3
    execution_lag: int = 1  # decision at t → fill at t+lag
    annualization: int = 365
    risk_free: float = 0.0
    target_pair_notional: float = 1.0
    max_concurrent_pairs: int = 5
    ffill_limit: int = 3


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 180
    test_bars: int = 60
    step_bars: int = 60
    embargo_bars: int = 1


@dataclass(frozen=True)
class StrategyConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    screen: PairScreenConfig = field(default_factory=PairScreenConfig)
    zscore: ZScoreConfig = field(default_factory=ZScoreConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
