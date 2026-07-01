# Label Experiment — does a more realistic label improve edge?

Date: 2026-07-01

## What was tested

The current model learns a symmetric ATR triple-barrier direction label (long TP at
1.3×ATR vs short TP, first-to-hit, neutral bars dropped). To see whether a *more
realistic* target creates a tradeable edge, five label definitions were trained (LR
proxy, walk-forward, purged split, out-of-sample gate) and scored under **one realistic
P&L engine** so the comparison is fair:

> enter the model's predicted direction, exit at the **first of** {TP = 1.3 ATR,
> SL = 1.0 ATR, horizon end}, realised return in ATR-R units, **minus one spread**.

Run it with `python tools/eval_labels.py <variant>` (results in `tools/eval/eval_labels_results.json`).

## Result

| Label variant | Mean expectancy (R) | Pairs with edge (CI-lo > 0) |
|---|:---:|:---:|
| horizon_end_h8  | **−0.177** | **0 / 12** |
| baseline_h16    | −0.202 | 0 / 13 |
| fwd_return_h8   | −0.216 | 0 / 11 |
| baseline_h8 (current) | −0.264 | 0 / 12 |
| horizon_end_h16 | −0.300 | 0 / 14 |

**No label variant produced a single pair with a statistically significant positive edge
after costs.** Every mean expectancy is negative. Changing the label — resolving neutral
bars at the horizon end, using a pure forward-return sign, or lengthening the horizon —
does **not** turn this feature set into a profitable strategy.

## What we adopted anyway (and why)

`horizon_end_h8` is the least-negative variant **and** it fixes a real correctness bug
(audit **C3**): the old label drops ~20–30% of bars as "neutral" during training but the
live server scores every bar, so train and live distributions differ. The horizon-end
label gives every bar a target, aligning the two. So it is now the trainer default
(`LABEL_MODE=horizon_end`) — a realism/correctness win, **not** a profitability claim.

## The honest takeaway

The edge problem is **not in the label** — it's in the **features / signal**. Across the
walk-forward (Phase 1) and this label sweep, H1 direction on these pairs looks close to
unpredictable with the current inputs. The remaining levers, in rough order of promise:

1. **Different / richer features** — order-flow or volume proxies, cross-pair and
   cross-asset context, volatility regime, session structure, higher-timeframe trend
   state as first-class inputs.
2. **Meta-labeling** — stop predicting raw direction; predict whether a *specific* rule
   (trend-follow, mean-revert) will be profitable, and size by that.
3. **A different horizon/instrument** entirely, or accept that H1 direction here has no
   exploitable edge.

Each idea should be run through `tools/walk_forward_eval.py` / `tools/eval_labels.py`
**before** any of it reaches live trading. The infrastructure to test honestly now
exists; the search for signal is the open problem.

*Caveat: this sweep uses a linear (logistic-regression) proxy because the sandbox can't
install the tree/boosting libraries. Run `tools/walk_forward_eval.py` in your venv to
confirm with the full model set — but given in-sample tree AUCs already sit near 0.5, the
conclusion is unlikely to change without new features.*
