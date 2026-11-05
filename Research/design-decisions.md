# Design Decisions (P0 + Kalman)

Decisions below close gaps called out in the baseline strategy review (title/engine mismatch, opaque costs, survivorship, broken α, no cointegration).

## 1. True stat-arb engine, not rank momentum

**Decision:** Cointegration → hedge → residual z-score only. No cross-sectional rank demean path in the research core.  
**Why:** Aligns code with claim language; avoids repeating the baseline’s branding mismatch.  
**Code:** `cointegration/`, `hedge/`, `signals/zscore.py`, `backtest/engine.py`.

## 2. Kalman is in P0 (user override)

**Decision:** OLS and Kalman are both first-class; demos and tests ablate both.  
**Why:** Abstract listed Kalman under P2; product owner pulled it into P0.  
**Detail:** Kalman updates the residual used for z-scores; portfolio **weights** use a frozen/band β so daily β jitter does not create spurious turnover.  
**Code:** `hedge/kalman.py`, ablation in `tests/test_hedge_ols_kalman.py` and `tests/test_backtest_no_lookahead.py`.

## 3. Log-price Engle–Granger + β sanity filters

**Decision:** Default screen uses \(\log P\) regressions; reject pairs with \(|\beta|\) outside `[0.05, 5]` or extreme median price-scale ratios.  
**Why:** Level-price EG on BTC vs small alts produced \(|\beta|\) ~ 10³, exploding notionals and fake “cost” drag.  
**Code:** `cointegration/screening.py`, `PairScreenConfig`.

## 4. Point-in-time universe; no stables as legs

**Decision:** Eligibility at \(t\) from trailing bars only; USDT/USDC/… bases excluded.  
**Why:** Kills end-of-sample top-N survivorship and removes stable “pairs” that are not tradable residual relationships.  
**Code:** `data/universe.py`.

## 5. No `bfill`; capped `ffill`; lagged fills

**Decision:** `bars_to_close_panel(..., allow_bfill=False)` raises if bfill requested; default `ffill_limit=3`; `execution_lag=1`.  
**Why:** Baseline hygiene failures. Decision time ≠ fill time.  
**Code:** `data/schema.py`, `backtest/engine.py`.

## 6. Explicit cost model; always pre/post

**Decision:** Fee + slippage bps per side; optional funding series from adapter; every summary publishes gross and net.  
**Why:** Baseline asserted “with transaction costs” without a model.  
**Code:** `costs/model.py`, demo JSON under `Research/artifacts/`.

## 7. Walk-forward with freeze + embargo

**Decision:** Train → freeze pairs & hedge params → embargo → OOS test → roll. Grid search not run on full sample in P0.  
**Why:** Stops window shopping; matches abstract validation plan.  
**Code:** `backtest/walkforward.py`.

## 8. CAPM α/β formulas unit-tested

**Decision:** Excess-return regression; annualize α as \(\alpha_{\text{period}} \times\) periods/year (not a percent-scale bug).  
**Why:** Baseline reported incoherent 4-digit α.  
**Code:** `backtest/metrics.py`, `tests/test_metrics_alpha_beta.py`.

## 9. Pluggable adapters

**Decision:** `MarketDataAdapter` interface; synthetic (tests/demo) + Binance public klines with on-disk CSV cache; **no API secrets on argv**.  
**Why:** Multi-asset extensibility without rewriting the research core.  
**Code:** `data/adapters/`.

## 10. Dollar neutrality now; β sleeve designed via monitor

**Decision:** Per-pair dollar-neutral weights from β; book-level BTC β **monitored and reported** (hedge sleeve optional later).  
**Why:** P0 abstract asks for dollar-neutral pairs + β monitor/report.  
**Code:** `portfolio/pairs.py`, `portfolio/beta.py`.
