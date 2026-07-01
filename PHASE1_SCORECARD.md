# Phase 1 — Honest Walk-Forward Scorecard

*The deliverable that Phase 1 gates the project on: does any pair show a real,
out-of-sample, cost-aware edge? Run on your actual `oanda_h1_ba_live/` CSVs.*

Date: 2026-07-01

---

## Bottom line

**No pair shows a statistically significant positive edge after honest evaluation.**
Across 18 pairs, 12 produced enough out-of-sample trades to measure; **0 of those 12
had an expectancy 95% confidence interval above zero.** Mean out-of-sample AUC was
**0.507** — indistinguishable from a coin flip. Three pairs were significantly
*negative* (GBP/CHF, GBP/USD, CHF/JPY).

This is the same conclusion the audit predicted, now confirmed with numbers on your
data. It is the cheapest possible moment to learn it — before the account, not after.

## What "honest" means here (vs the current trainer)

| Dimension | Current `03_train…` | This evaluation |
|---|---|---|
| Split | single in-sample holdout | 5-fold **walk-forward** (expanding) |
| Label leakage | none removed | **purge** last `HORIZON_BARS` train rows each fold |
| Gate selection | grid-search precision **on the reported set** | gate chosen on **train**, applied to untouched **test** |
| Success metric | AUC + precision on ~30 in-sample trades | **cost-aware expectancy** (spread deducted) + bootstrap 95% CI |
| Neutral bars | dropped silently | dropped **and reported** (they're scored live anyway) |

Expectancy is in **R units**: a win = +`TP_ATR` (1.3), a loss = −`SL_ATR` (1.0), minus
one spread (in ATR units) per trade. "Edge" = the bootstrapped 95% CI lower bound of
mean expectancy is above 0.

## Results (12 pairs with ≥10 out-of-sample trades)

Sorted by CI lower bound (best first). `IS_AUC` is the trainer's in-sample AUC; `OOS_AUC`
is this walk-forward's out-of-sample AUC.

| Pair | IS_AUC | OOS_AUC | Trades | Precision | Expectancy (R) | 95% CI | Edge? |
|------|:------:|:-------:|:------:|:---------:|:--------------:|:------:|:-----:|
| USDCAD | 0.526 | 0.528 | 25 | 0.720 | +0.202 | [−0.273, +0.664] | no |
| AUDCAD | 0.570 | 0.552 | 25 | 0.680 | +0.171 | [−0.257, +0.560] | no |
| EURUSD | 0.475 | 0.506 | 61 | 0.623 | +0.205 | [−0.078, +0.478] | no |
| USDJPY | 0.551 | 0.497 | 30 | 0.467 | +0.003 | [−0.385, +0.396] | no |
| AUDUSD | 0.513 | 0.494 | 51 | 0.628 | −0.013 | [−0.361, +0.306] | no |
| EURJPY | 0.537 | 0.496 | 187 | 0.503 | −0.037 | [−0.208, +0.132] | no |
| AUDNZD | 0.516 | 0.513 | 149 | 0.550 | −0.046 | [−0.229, +0.132] | no |
| NZDUSD | 0.485 | 0.490 | 32 | 0.531 | −0.323 | [−0.852, +0.191] | no |
| CHFJPY | 0.518 | 0.498 | 69 | 0.420 | −0.319 | [−0.569, −0.050] | **negative** |
| USDCHF | 0.529 | 0.504 | 23 | 0.391 | −0.453 | [−0.979, +0.092] | no |
| GBPUSD | 0.528 | 0.478 | 20 | 0.300 | −0.561 | [−1.039, −0.056] | **negative** |
| GBPCHF | 0.556 | 0.547 | 16 | 0.500 | −1.225 | [−1.840, −0.637] | **negative** |

Six further pairs (AUD/JPY, CAD/JPY, EUR/CHF, EUR/GBP, GBP/JPY, NZD/JPY) fired **0–8**
trades out-of-sample at their selected gate — too few to evaluate, which is itself a
finding: the operating point that looked good in-sample barely triggers out-of-sample.

## How to read this

- **The three "best" pairs are noise, not edge.** USD/CAD, AUD/CAD and EUR/USD show
  positive point estimates (+0.17 to +0.20 R) but every one of their confidence
  intervals crosses zero, on 25–61 trades. That is exactly what an edgeless strategy
  looks like after you pick the best-looking gate. EUR/USD is especially instructive:
  the trainer labels it `tradable: false` with in-sample AUC 0.475, yet it is deployed.
- **In-sample AUC overstates out-of-sample AUC** almost everywhere (USD/JPY 0.551→0.497,
  AUD/JPY 0.548→0.482, GBP/JPY 0.534→0.500) — the signature of the leakage/optimism the
  methodology fixes remove.
- **20–30% of bars are dropped as "neutral"** during training but scored live (finding
  C3). The live probability distribution is therefore not the one the model was
  validated on.
- **Costs matter at this signal strength.** With an edge this thin, the spread deduction
  alone is enough to push borderline-positive point estimates negative.

## What this does *not* say

It does not say the market is unbeatable or the engineering is wasted. It says **this
specific feature set + ATR-barrier label + H1 horizon, evaluated honestly, has not
demonstrated an edge that survives costs.** That is a targeted, fixable result.

## Recommended next steps (in order)

1. **Do not trade live capital on the current models.** Run practice/paper only until a
   pair clears the bar below.
2. **Set the bar explicitly:** a pair is tradable when its walk-forward expectancy CI
   lower bound is **> 0 after costs**, across folds, with a usable trade count
   (target ≥100 OOS trades). Wire this definition into `create_h1_registry.py`'s
   `tradable` flag so deployment can't outrun the evidence.
3. **Attack the label, not just the model.** The biggest lever is usually the target:
   test a longer horizon, a meta-labeling layer (predict *whether to take* a
   trend/mean-reversion signal rather than raw direction), or session/regime filters —
   re-evaluating each change through `tools/walk_forward_eval.py`.
4. **Fix C3 directly:** either model "neutral" as a third class or restrict live scoring
   to the regime the labels represent, so train and live distributions match.
5. **Only after a pair clears the bar,** proceed to Phases 2–5 (concurrency, package
   split, tests, model store) for that subset.

## Reproduce / extend

This scorecard was produced by a numpy logistic-regression **proxy** of the deployed LR
pipeline (same features, same labels), because the sandbox couldn't install the tree/
boosting libraries. To reproduce with the **full** model set (logistic_regression,
extra_trees, lightgbm, xgboost, catboost) in your environment:

```bash
python tools/walk_forward_eval.py                 # all pairs, all classical models
python tools/walk_forward_eval.py --pairs EURUSD USDCAD --folds 6 --cost-mult 1.5
```

It writes `tools/eval/walk_forward_results.json` and prints a scorecard with the best
model per pair. Expect the tree/boosting models to move individual numbers, but given
the in-sample AUCs already sit near 0.5, the aggregate conclusion is unlikely to change
without a change to the **label or features** — which is where the effort should go.

*Methodology parameters: 5 folds, last 40% walk-forward, purge = `HORIZON_BARS` (8),
`TP_ATR`=1.3, `SL_ATR`=1.0, cost = 1× spread (ATR units), 2000-sample bootstrap CIs,
seed 42.*
