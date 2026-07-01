# ML-Assisted-Cloud — Code & Strategy Audit + Refactor Plan

*Prepared as an experienced quant developer would review a live FX system before signing off on it. Scope: assessment first, no logic changed. Genuine bugs and risk gaps are flagged, not silently fixed.*

Date: 2026-06-30

---

## 1. Executive summary

This is a per-pair, H1 machine-learning FX decision service. A FastAPI app (`fx_api_sniper_CLperpair.py`, 3,916 lines) receives TradingView-style payloads, runs a per-pair classifier through a stack of guards (noise, news, staleness, direction consensus, entry reversal, technical review, optional LLM review), and returns a `BUY`/`SELL`/`NONE` decision with position size and SL/TP. A separate script (`03_train_h1_auto_models.py`, 1,430 lines) trains the models against OANDA H1 candles using triple-barrier-style ATR labels.

The engineering is more disciplined than most retail bots: chronological train/valid split, per-pair gates, daily trade caps, duplicate-signal suppression, an SQLite audit trail, dockerised deploy. That is a real foundation.

**But two categories of problem stand out and both matter more than the folder layout:**

1. **The measured statistical edge is at or below coin-flip, and the way the edge is selected is overfit.** Validation AUCs range roughly **0.47–0.57** across pairs; EUR/USD's own `metrics.json` reports AUC **0.475** with `"tradable": false`, yet the pair is deployed. "Best gate" precision is chosen by maximising precision on **30–45 validation trades** — far too few to be reliable. This is the single most important finding: the code quality almost doesn't matter until the edge is real.

2. **There is train/validation label leakage and a live/train distribution mismatch** that inflate the already-marginal backtest numbers, so the true out-of-sample edge is very likely *worse* than the 0.47–0.57 shown.

On the software side, the highest-impact issues are **unsynchronised shared state accessed from multiple threads** (real race conditions on trade counters and the daily caps that are supposed to protect capital), a **hardcoded `avg_auc = 0.56` for every pair** that feeds live position sizing, and **~430 MB of unused model artifacts committed to git**.

Recommended path: **do not expand features. Fix the evaluation methodology first, re-establish whether any edge survives honest validation, and in parallel harden the concurrency and config issues.** Detailed, phased plan in §6.

---

## 2. What the system does (as-built)

```
TradingView / caller ──POST /predict──> FastAPI service
                                         ├─ normalise payload, compute spread/ATR ratios
                                         ├─ load per-pair bundle (model + gates + feature order)
                                         ├─ model.predict_proba -> p_up -> side, conf, margin
                                         ├─ optional probability calibrator
                                         ├─ guards (fail-closed -> NONE):
                                         │    staleness → noise → news → direction consensus
                                         │    → entry reversal → hint-disagreement → pair-score
                                         │    → sniper gate (conf/margin) → technical review
                                         │    → duplicate → daily caps → open-trade cap → AI review
                                         ├─ compute SL/TP from ATR, size from equity risk %
                                         └─ return decision + units + SL/TP  (does NOT place the order)

Background thread: auto_close_worker — closes broker positions older than MAX_HOLD_MINUTES
Training: 03_train_h1_auto_models.py — ATR triple-barrier labels, 5 classical models + optional TCN,
          picks "best" per pair, writes models/<PAIR>/…  ; create_h1_registry.py assembles registry.json
```

Key architectural fact worth stating plainly: **`/predict` is advisory** — it returns `would_order` and sizing but never calls OANDA to open a trade. Actual execution happens in the caller. That has a consequence the current design misses (see H2 below): the service's own daily-cap counters increment on *recommendation*, not on *fill*, so its risk limits can drift out of sync with the real account.

---

## 3. Findings by severity

Severity key: **S1 Critical** (can lose money or corrupt risk controls) · **S2 High** (correctness / robustness) · **S3 Medium** (maintainability / hygiene) · **S4 Low** (polish).

### S1 — Critical

**C1. The deployed edge is not statistically established.**
`models/EURUSD/metrics.json`: `best.auc = 0.475`, `accuracy = 0.495`, `tradable = false`. Several pairs sit at 0.47–0.55 AUC — indistinguishable from random, some below random. `precision_at_gate` (e.g. 0.656) is computed on `trades_at_gate` of **32** samples (EURUSD) / **44** (AUDCAD). A precision estimate on 32 trades has a ~±17% 95% confidence interval; selecting the gate that maximises it is fitting to noise. **Consequence:** live gates and `pair_score` are chosen by in-sample optimisation over a tiny slice; expected live precision is materially lower than reported. This must be resolved before capital is at risk.

**C2. Label leakage across the train/valid boundary (no purge/embargo).**
`train_pair()` (fx `03_train_h1_auto_models.py:1100`) splits chronologically — good — but labels from `build_atr_direction_labels()` look forward `HORIZON_BARS = 8` bars. The last ~8 training rows resolve their labels *using validation-period price action*, and there is no gap between the two sets. This is the classic López de Prado purging/embargo omission; it leaks future information into training and inflates validation AUC. Fix: drop the final `HORIZON_BARS` rows of the training set (purge) and optionally embargo a few bars after the split.

**C3. Selection bias: trained on decisive bars, deployed on all bars.**
Labeling assigns `NaN` to "neutral/ambiguous" bars (no barrier hit within the horizon) and `train_pair()` drops them (`:1075`). So the model learns the conditional distribution *"given a clean TP-first outcome occurred"*, but at inference `/predict` scores **every** incoming bar, including the many that would have been neutral. Train and live input distributions differ. This biases probabilities and, combined with C1/C2, means backtest precision does not transfer. Fix: model neutral as a third outcome, or restrict live scoring to the same regime the labels represent, or re-label so every bar has a defined target.

**C4. Unsynchronised shared mutable state across threads (race conditions on risk controls).**
Globals `_trade_count_today`, `_open_trade_ids`, `_open_trade_meta`, `_recent_signals`, `_bar_history` (`:254–260`) are mutated from request threads (FastAPI runs sync endpoints in a threadpool) **and** from the `auto_close_worker` background thread, with no locks. Two concrete failures: (a) the daily-cap check is check-then-increment (`trades_today_total() >= CAP` … later `inc_trade()`), so concurrent `/predict` calls can both pass the cap and both increment — the cap that protects capital is not atomic; (b) `auto_close_worker` iterates `_open_trade_meta` while `/trade_event` writes to it, risking `RuntimeError: dictionary changed size during iteration`. Fix: a single `threading.Lock` (or `RLock`) around all shared-state reads/writes, or move counters into SQLite with atomic transactions.

**C5. `avg_auc` is hardcoded to 0.56 for every pair and feeds live position sizing.**
`create_h1_registry.py:` `"avg_auc": metrics.get("avg_auc") or metrics.get("auc") or 0.56`. `metrics.json` has no `avg_auc` key and its `auc` lives under `best.auc`, so `.get("auc")` returns `None` → **every pair gets 0.56**. `compute_units_dynamic()` then bumps size by 1.10× when `avg_auc >= 0.57` and cuts 0.90× when `< 0.54` — logic that can now never fire, and worse, a genuinely weak pair (AUC 0.47) is sized as if it were 0.56. Live sizing is decoupled from measured model quality. Fix: read the real per-pair AUC (`summary.best.auc`) into the registry.

### S2 — High

**H1. Duplicated function name with load-order-dependent aliasing.**
`build_runtime_feature_row` is defined twice (`:680` and `:2494`); `:2492` captures the first into `ORIGINAL_ALERT_BUILD_RUNTIME_FEATURE_ROW` one line before the redefinition. It works only because of exact top-to-bottom execution order — any reordering, or a linter "deduplicating" it, silently changes behaviour. Rename to `build_alert_feature_row` / `build_runtime_feature_row_v2` and make the fallback explicit.

**H2. Risk counters track recommendations, not fills.**
Because `/predict` is advisory, `inc_trade()` / `remember_signal()` fire on `would_order` regardless of whether the caller actually filled (or the fill was rejected/partial). Daily caps, open-trade caps and duplicate suppression therefore diverge from the real OANDA account over time. Fix: drive counters from confirmed fills via `/trade_event`, and/or reconcile against `GET /v3/accounts/{id}/openTrades` on a timer.

**H3. Auto-close granularity can hold positions far past the intended max.**
`MAX_HOLD_MINUTES = 60` but `AUTO_CLOSE_CHECK_SECONDS = 1800` (30 min). A position opened right after a sweep isn't re-checked for 30 min, so effective max hold is up to ~90 min — 50% over the intended limit, unbounded if a sweep throws. For a time-boxed "sniper" strategy this is a real risk-control gap. Fix: check interval ≤ ¼ of max-hold, and make the worker resilient so one bad iteration can't stall the loop.

**H4. Broad `except Exception` swallowing throughout.**
The `/predict` body, model-loading, news parsing and the auto-close worker catch bare `Exception` and continue (often returning `NONE`, which is fail-safe for trading) but log little or nothing structured. Silent failure hides model-load errors, feature drift and OANDA outages. Fix: narrow exception types, log with stack traces to a structured logger, and surface a health signal (see H6).

**H5. Live/training threshold inconsistency.**
Training/registry recommend `default_margin` ~0.02; live `DEFAULT_GATE` margin is 0.04 and `PAIR_GATES` use 0.04–0.06. The gate that was validated is not the gate that trades. That may be intentional conservatism, but it's undocumented and means the deployed operating point was never actually evaluated. Fix: make the live gate the evaluated gate, or document the deliberate shift and re-measure precision at the live gate.

**H6. No automated tests, no health/liveness depth, no CI.**
For a system that moves money there is zero test coverage. `/health` exists but a regression in feature ordering, label logic or gate parsing would ship undetected. Fix: unit tests for the pure functions (labeling, feature row, SL/TP, sizing, gate logic) with golden values; a smoke test that loads every bundle and scores a synthetic bar; a CI workflow running lint + tests.

### S3 — Medium

**M1. ~430 MB of unused artifacts committed to git.** 201 files under `models/` are tracked (612 MB total); each pair's `candidate_models/` (~24 MB each × 18 ≈ 430 MB) holds the *losing* candidate models that runtime never loads — only `best_model.pkl` is used. This bloats every clone and CI run. Fix: stop tracking `candidate_models/` (or all of `models/`) in git; use Git LFS or an artifact store / model registry; keep only `best_model.pkl` + metadata in the image.

**M2. `.DS_Store` tracked in git** (root and `models/`). Remove and rely on `.gitignore`.

**M3. Empty `app/` directory** (only `.DS_Store`) — dead scaffolding that implies an intended package split never happened. Either build the package here (see §5) or delete it.

**M4. `volume` is a dead feature.** CSVs contain only bid/ask OHLC; `read_pair_csv()` fills `volume = 0` when missing, so `volume` is constant zero for every row — a zero-variance feature carried through training and required at inference. Drop it or source real tick-volume.

**M5. Monolithic modules.** 3,916 lines in one file mixing config, indicators, DB, broker I/O, guards, LLM review, HTML dashboard and routes. Hard to test, review, or reason about blast radius. Package split proposed in §5.

**M6. Deprecated FastAPI `@app.on_event("startup")`** (`:3135`) — removed in newer FastAPI. Migrate to the `lifespan` context manager before the next dependency bump.

**M7. Config sprawl via `os.getenv` scattered across both files.** Dozens of env vars read inline with string defaults; no single validated settings object, no `.env.example`, and defaults silently diverge between the two files. Fix: one `pydantic-settings` `Settings` model, imported everywhere; ship `.env.example`.

**M8. Missing pipeline stages `01`/`02`.** The numeric prefix on `03_train_…` implies data-download and feature-prep steps that aren't in the repo — reproducibility gap. Document or commit them.

### S4 — Low

**S4-1.** No `README`, no run/deploy docs, no `LICENSE`. **S4-2.** No dependency pinning (`requirements.txt` is unversioned — non-reproducible builds; pin or use a lockfile). **S4-3.** `anthropic` (LLM review) and `torch` (TCN) are imported but absent from `requirements.txt` — the AI/TCN paths depend on packages the documented environment doesn't install. **S4-4.** Version string `"8.1-h1-auto-registry-m15-safety-ai"` embedded in code; move to a single source of truth. **S4-5.** Commit history is dominated by "Fix duplicate…/Fix missing…" hotfixes — a symptom of the no-tests problem (H6).

---

## 4. The strategy reality check (read this before any refactor)

A clean way to see the core issue, straight from the committed metrics:

| Pair | Best model | Valid AUC | Acc | Gate | Trades@gate | Prec@gate | `tradable` |
|------|-----------|:---------:|:---:|:----:|:-----------:|:---------:|:----------:|
| EURUSD | logistic_regression | **0.475** | 0.495 | 0.66 | 32 | 0.656 | **false** |
| AUDCAD | extra_trees | 0.570 | 0.539 | 0.64 | 44 | 0.727 | true |

Two honest observations from an experienced desk:

- **AUC ≈ 0.5 means the classifier has essentially no ranking power.** EURUSD is below 0.5 on its own validation set. A high `precision_at_gate` on ~30 trades is not evidence of edge — it's what noise looks like after you search across gate/margin combinations for the best-looking number.
- **Every methodological bias found (C2 leakage, C3 selection bias, H5 gate shift, C1 tiny-sample gate search) points the same direction: the real out-of-sample number is lower than shown.** So the true picture is "no demonstrated edge yet," not "small edge to be tuned."

This is not a reason to abandon the project — it's a reason to fix *how the edge is measured* before touching anything else, so that when you do have signal, you can trust it. Concretely, the evaluation upgrade (Phase 1) is: purged split → walk-forward across multiple folds → report AUC, precision **and** a cost-aware P&L (spread + slippage) with confidence intervals → only then set gates. If no configuration clears costs across folds, that is the finding, and it's far cheaper to learn now than in the account.

---

## 5. Target architecture

Keep the behaviour; split the monolith into a testable package so each piece can be verified in isolation.

```
ml_assisted_cloud/
├── pyproject.toml            # packaging + pinned deps + tool config (ruff, mypy, pytest)
├── .env.example             # every env var, documented (no secrets)
├── README.md
├── config/
│   └── settings.py          # single pydantic-settings Settings (replaces scattered os.getenv)
├── core/
│   ├── instruments.py       # pair/instrument maps, pip size, precision, rounding
│   ├── indicators.py        # rsi/atr/adx/ema — ONE implementation shared by train + serve
│   └── features.py          # ONE feature builder shared by train + serve (kills H1 + drift)
├── data/
│   ├── candles.py           # OANDA candle fetch + dataframe
│   └── history.py           # rolling bar history / CSV seeding
├── models/
│   ├── registry.py          # load bundles, real per-pair avg_auc (fixes C5)
│   └── wrappers.py          # LightGBM / TCN wrappers
├── strategy/
│   ├── guards.py            # staleness, noise, news, direction, entry-reversal
│   ├── gating.py            # conf/margin/pair-score gates, hint disagreement
│   ├── sizing.py            # compute_units_dynamic, compute_sl_tp_prices
│   └── review.py            # technical + LLM review
├── execution/
│   ├── broker.py            # OANDA REST client (headers, request, positions, close)
│   ├── state.py             # trade counters + open-trade tracking — behind ONE lock (fixes C4)
│   └── auto_close.py        # worker with proper interval + resilience (fixes H3)
├── persistence/
│   ├── db.py                # SQLite schema + typed accessors
│   └── audit.py             # audit/trade CSV writers
├── api/
│   ├── schemas.py           # Pydantic payload models
│   ├── routes.py            # thin FastAPI endpoints -> call strategy/execution
│   └── app.py               # lifespan startup (fixes M6)
├── training/
│   ├── labels.py            # ATR barriers + purge/embargo (fixes C2)
│   ├── evaluate.py          # walk-forward + cost-aware P&L + CIs (fixes C1)
│   └── train.py             # orchestration
└── tests/
    ├── test_features.py test_labels.py test_sizing.py test_gating.py test_registry.py
    └── test_smoke_predict.py
```

The single most valuable structural change is **one shared `core/features.py` and `core/indicators.py` used by both training and serving.** Today the runtime feature row and the training features are built by separate code, which is exactly how live/train feature drift (a silent P&L killer) creeps in.

---

## 6. Prioritised refactor roadmap

Ordered by return on risk. Each phase is behaviour-preserving unless it's explicitly a methodology fix, and each ends in a state you can verify.

**Phase 0 — Safety net & hygiene (0.5–1 day).** No behaviour change.
Add `.gitignore` for `.DS_Store`; `git rm --cached` the tracked `.DS_Store` and `candidate_models/` (M1, M2). Pin dependencies and add the missing `anthropic`/`torch` (S4-2/3). Write `README.md` + `.env.example` (M7, S4-1). Snapshot current `/predict` outputs on a fixed set of payloads as **golden files** — this is what lets every later refactor prove it changed nothing.

**Phase 1 — Trust the numbers (2–4 days). Highest priority.**
Purge + embargo the split (C2). Handle neutral bars honestly (C3). Replace the single holdout with **walk-forward** evaluation and report AUC + precision + cost-aware P&L with confidence intervals; set gates *out-of-sample* (C1, H5). Fix the real per-pair `avg_auc` in the registry (C5). Deliverable: a one-page per-pair scorecard that says, honestly, which pairs (if any) show edge after costs. **Gate the whole project on this.**

**Phase 2 — Concurrency & risk integrity (1–2 days).**
Put all shared state behind one lock or move it into SQLite transactions (C4). Make daily/open-trade caps atomic (check-and-increment in one critical section). Drive counters from confirmed fills and reconcile against OANDA open trades (H2). Tighten `auto_close` interval and make the worker crash-resilient (H3). Verify against Phase 0 goldens.

**Phase 3 — Package split (3–5 days).**
Carve the monolith into §5's layout, moving code verbatim; unify indicators/features into `core/` (H1, M5). Migrate to FastAPI `lifespan` (M6) and `pydantic-settings` (M7). Re-run goldens after each move.

**Phase 4 — Tests & CI (2–3 days).**
Unit tests with golden values for labeling, features, SL/TP, sizing, gates; a smoke test that loads every bundle and scores a synthetic bar; GitHub Actions running ruff + mypy + pytest on every push (H6). Now the "Fix duplicate…" hotfix cycle (S4-5) stops.

**Phase 5 — Model store & observability (2–3 days).**
Move model artifacts out of git to LFS/registry (M1); keep only `best_model.pkl` + metadata in the image. Add structured logging with stack traces and a deeper `/health` that reports bundle load status, last-scored time and OANDA reachability (H4).

Rough total: ~2–3 focused weeks, but **Phases 0–2 (≈1 week) capture most of the risk reduction.** Phase 1 may end the project early if no edge survives honest testing — and that is the cheapest possible outcome.

---

## 7. What is already good (keep it)

Chronological (not shuffled) split and `shuffle=False` for the TCN — the right instinct for time series. `StandardScaler` fit inside pipelines on training data only — no scaler leakage. Fail-closed guard design (any error/uncertainty → `NONE`). Per-pair gates and daily caps as a concept. SQLite audit trail + CSV export. Dockerised deploy with a sane `.dockerignore`. Secrets kept out of git (`.env` correctly ignored). This is a serious starting point; the plan above is about making its numbers trustworthy and its risk controls thread-safe — not starting over.
