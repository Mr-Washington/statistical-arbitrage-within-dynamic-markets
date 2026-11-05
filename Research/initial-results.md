# Initial Results Highlight

**Run:** `python -m stat_arb.cli --hedge both --seed 42`  
**Data:** synthetic cointegrated crypto panel (`SyntheticCryptoAdapter`, 500 daily bars)  
**Validation:** 5 walk-forward folds (train 180 / test 60 / step 60 / embargo 1)  
**Costs:** 5 bps fee + 2 bps slippage per side; synthetic funding series applied  
**Artifact:** [`artifacts/demo_walkforward_results.json`](artifacts/demo_walkforward_results.json)

> These are **reproducible research-harness numbers on synthetic cointegrated pairs**, not a claim of live crypto edge. They prove the pipeline (screen → hedge → z → costs → WF → α/β) is claim-aligned and numerically sane.

## Headline table (combined OOS)

| Hedge | Gross Sharpe | Net Sharpe | Net CAGR | Net max DD | CAPM β (BTC) | \|β\|≤0.25 days |
|---|---:|---:|---:|---:|---:|---:|
| **OLS** | **1.50** | **1.36** | **15.4%** | **−5.3%** | **−0.006** | **100%** |
| **Kalman** | 0.45 | 0.17 | 1.1% | −4.0% | −0.014 | 100% |

### OLS (post-cost) — primary P0 highlight

- Sharpe **1.36**, Sortino **1.52**, Calmar **2.88**
- Annualized α vs BTC **≈ 15.1%** (economically coherent with ~15% CAGR — **not** a 4-digit bug)
- Book β **≈ 0**; rolling \|β\| inside 0.25 band on **100%** of OOS days
- Pairs frozen per fold: 2–3 typical (EG passers among planted AAA/BBB, CCC/DDD, etc.)

### Kalman ablation

- Same pairs/screens; dynamic residual for z-scores, β frozen for weights
- Positive but weaker net Sharpe (**0.17**) on this synthetic seed — useful control, not a failure of the module
- Confirms Kalman path runs end-to-end without look-ahead

## Pipeline health checks

| Check | Result |
|---|---|
| `pytest` | **17 passed** |
| Look-ahead fills | First-bar weights flat (`execution_lag=1`) |
| Stables as legs | Excluded (`USDCUSDT` never in universe) |
| α/β unit test | Toy series recovers β≈0.5, α ~ O(0.1–1), never thousands |
| Pre vs post cost | Both published every run |

## How to reproduce

```bash
pip install -e ".[dev]"
pytest
python -m stat_arb.cli --hedge both --seed 42
```

## Interpretation for Michael

1. The engine is **true pairs stat-arb**, not the baseline ranker.  
2. Costs are disclosed and material but do not invent false α.  
3. Neutrality is measurable (β monitor), not only dollar demean.  
4. Next hardening (P1): Johansen, fail-closed re-tests, paper path, second asset class — on **cached Binance** or vendor bars with the same harness.
