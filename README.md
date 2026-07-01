# ML-Assisted-Cloud — H1 FX decision service

A per-pair, H1 machine-learning decision service for spot FX. A FastAPI app scores
incoming market payloads through a trained per-pair classifier and a stack of safety
guards, returning a `BUY` / `SELL` / `NONE` decision with position size and SL/TP.
A companion script trains the models from OANDA H1 candles.

> **Status / honest disclaimer.** This is research software that can place capital at
> risk. As of the latest audit the deployed models show validation AUCs of roughly
> 0.47–0.57 (i.e. at or below coin-flip for several pairs) and the evaluation method
> has known biases. **Do not run this with real money before working through
> `AUDIT_AND_REFACTOR_PLAN.md`, especially Phase 1 (trustworthy evaluation).**

## How it works

```
caller ──POST /predict──▶ FastAPI service
                          ├─ load per-pair bundle (model + gates + feature order)
                          ├─ model.predict_proba → side, confidence, margin
                          ├─ guards (any failure ⇒ NONE):
                          │    staleness → noise → news → direction consensus →
                          │    entry reversal → hint disagreement → pair-score →
                          │    conf/margin gate → technical review → duplicate →
                          │    daily caps → open-trade cap → optional AI review
                          ├─ size from equity risk %, SL/TP from ATR
                          └─ return decision + units + SL/TP
Background thread: auto-closes broker positions older than MAX_HOLD_MINUTES.
```

**Important:** `/predict` is *advisory* — it returns a recommendation and sizing but
does **not** place the order itself. Execution is the caller's responsibility. (See
audit finding H2: internal trade counters therefore track recommendations, not fills.)

## Repository layout

```
fx_api_sniper_CLperpair.py   # the FastAPI decision service (monolith — see plan §5)
03_train_h1_auto_models.py   # training: ATR-barrier labels, 5 classical models + TCN
create_h1_registry.py        # assembles models/registry.json from per-pair outputs
models/<PAIR>/               # best_model.pkl + metadata per pair (candidate_models are unused)
oanda_h1_ba_live/*.csv       # per-pair H1 candle history (bid/ask OHLC)
logs/                        # SQLite audit DB + CSV audit/trade logs (gitignored)
AUDIT_AND_REFACTOR_PLAN.md   # full code + strategy review and phased roadmap
.env.example                 # documented environment template
```

## Setup

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in OANDA_TOKEN / OANDA_ACCOUNT_ID
```

Point `OANDA_BASE_URL` at the practice endpoint
(`https://api-fxpractice.oanda.com`) until you have completed the audit plan.

## Run the service

```bash
uvicorn fx_api_sniper_CLperpair:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path                          | Purpose                                  |
|--------|-------------------------------|------------------------------------------|
| POST   | `/predict`                    | Score a payload → decision + size + SL/TP |
| POST   | `/trade_event`                | Record an open/close event               |
| GET    | `/health`                     | Liveness + basic status                  |
| GET    | `/stats`, `/pnl_stats`, `/pair_stats` | Aggregated performance           |
| GET    | `/news_events`                | Current news blackout events             |
| POST   | `/news_event`, `/reload-news` | Add / reload news events                 |
| GET    | `/export/closed_trades.xlsx`  | Export closed trades                     |
| GET    | `/dashboard`                  | HTML dashboard                           |

## Train models

```bash
# Put per-pair H1 candle CSVs in DATA_DIR, then:
python 03_train_h1_auto_models.py
python create_h1_registry.py        # rebuild models/registry.json
```

Labels use an ATR triple-barrier scheme (`SL_ATR`, `TP_ATR`, `HORIZON_BARS`). These
**must** stay consistent with the live SL/TP settings — see `.env.example`.

## Docker

```bash
docker build -t fx-sniper .
docker run --env-file .env -p 8000:8000 fx-sniper
```

## Verifying changes safely

Before/after any refactor, snapshot `/predict` outputs and diff them:

```bash
python tools/golden_snapshot.py --write      # capture current behaviour as golden
# ...make changes...
python tools/golden_snapshot.py --check       # fail if any output changed
```

## Development (package, tests, tools)

Pure, testable logic is being extracted from the monolith into a dependency-light
package `mlac/` (only needs numpy + pandas):

```
mlac/instruments.py   pair maps, pip size/precision, price + units helpers
mlac/indicators.py    rsi / atr / adx / ema (matches the trainer)
mlac/labeling.py      ATR triple-barrier labels + purged-split helper
mlac/sizing.py        compute_units_dynamic, compute_sl_tp_prices
mlac/gating.py        side/margin/gate decision predicates
mlac/reservation.py   ReservationBook — fill-aware caps (confirmed + pending + TTL)
```

These are the tested reference implementations. The live entrypoint has **not** yet been
rewired to import from `mlac/` — that migration must be verified with
`tools/golden_snapshot.py` before deploying (the entrypoint auto-deploys to live trading).

Run the tests:

```bash
pip install pytest numpy pandas
pytest                      # 30 unit tests over mlac/
```

CI runs the same suite on every push/PR (`.github/workflows/ci.yml`).

Tools:

```bash
python tools/model_store.py report            # artifact sizes, tradable flags
python tools/model_store.py verify            # every pair has required files
python tools/model_store.py prune-candidates --yes   # free ~428 MB of unused candidates
python tools/walk_forward_eval.py             # honest out-of-sample evaluation
python tools/golden_snapshot.py --write|--check      # behaviour baseline for refactors
```

Observability: `GET /health` dumps config; `GET /health/deep` reports live runtime state
(fill-aware trade counters, pending reservations, tracked vs broker open trades).

## Known issues

See `AUDIT_AND_REFACTOR_PLAN.md` for the full severity-ranked list. The critical ones:
model edge is not statistically established (C1), train/valid label leakage (C2),
train-vs-live distribution mismatch (C3), unsynchronised shared state across threads
(C4), and a hardcoded `avg_auc = 0.56` feeding live sizing (C5).
