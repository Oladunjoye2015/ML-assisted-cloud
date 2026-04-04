# fx_api_sniper_CLperpair.py
# Fully upgraded FX sniper API
#
# Features
# - Accepts EURUSD / EUR_USD / EUR-USD
# - Returns OANDA instrument format: EUR_USD
# - Pair-specific gates
# - Correct SELL confidence logic
# - Auto equity sizing (phase 2) with fallback equity
# - OANDA-safe SL/TP precision
# - FORCE buy/sell support for testing
# - Daily trade cap
# - Duplicate signal blocking
# - Payload sanity checks
# - Open trade cap
# - Trade-event logging
# - hint_side support from TradingView
# - Per-instrument disagreement override thresholds
# - Performance tracking per pair
# - Drop weak pairs automatically
# - Scale units based on AUC + live win rate
# - /health, /predict, /stats, /dashboard, /trade_event, /pnl_stats, /pair_stats, /weak_pairs
#
# Run locally:
#   uvicorn fx_api_sniper_CLperpair:app --host 0.0.0.0 --port 8000

import os
import csv
import glob
import math
import json
import hashlib
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
from collections import deque
from typing import Any, Dict, Optional, Literal, List, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# -------------------- env --------------------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

AUDIT_CSV = os.path.join(LOG_DIR, "audit.csv")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")

DEFAULT_GATE = {
    "conf": float(os.getenv("CONF_GATE", "0.65")),
    "margin": float(os.getenv("MARGIN_GATE", "0.10")),
}

# Keys must be OANDA instrument format
PAIR_GATES: Dict[str, Dict[str, float]] = {
    "USD_CHF": {"conf": 0.55, "margin": 0.10},
    "EUR_GBP": {"conf": 0.65, "margin": 0.12},
    "USD_CAD": {"conf": 0.55, "margin": 0.10},
    "CAD_JPY": {"conf": 0.70, "margin": 0.10},
    "AUD_JPY": {"conf": 0.70, "margin": 0.08},
    "USD_JPY": {"conf": 0.50, "margin": 0.10},
    "EUR_JPY": {"conf": 0.70, "margin": 0.15},
    "GBP_JPY": {"conf": 0.70, "margin": 0.12},
    "EUR_USD": {"conf": 0.55, "margin": 0.10},
    "NZD_USD": {"conf": 0.65, "margin": 0.10},
    "GBP_USD": {"conf": 0.55, "margin": 0.10},
    "AUD_USD": {"conf": 0.65, "margin": 0.12},
}

PAIR_DISAGREE_CONF: Dict[str, float] = {
    "EUR_GBP": 0.75,
    "CAD_JPY": 0.80,
    "AUD_JPY": 0.80,
    "GBP_JPY": 0.80,
    "USD_CHF": 0.65,
    "USD_CAD": 0.65,
    "EUR_USD": 0.65,
    "NZD_USD": 0.75,
    "GBP_USD": 0.65,
    "USD_JPY": 0.50,
    "AUD_USD": 0.75,
    "EUR_JPY": 0.80,
}
DEFAULT_DISAGREE_CONF = float(os.getenv("DEFAULT_DISAGREE_CONF", "0.70"))

UNITS_JPY = int(os.getenv("UNITS_JPY", "1000"))
UNITS_NON_JPY = int(os.getenv("UNITS_NON_JPY", "2000"))

MAX_TRADES_PER_DAY_TOTAL = int(os.getenv("MAX_TRADES_PER_DAY_TOTAL", "6"))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "2"))

# -------------------- fail-safe config --------------------
DUP_WINDOW_SECONDS = int(os.getenv("DUP_WINDOW_SECONDS", "300"))
MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "3.5"))
MIN_ATR_NON_JPY = float(os.getenv("MIN_ATR_NON_JPY", "0.00005"))
MIN_ATR_JPY = float(os.getenv("MIN_ATR_JPY", "0.005"))

# -------------------- phase 2 sizing + pair performance --------------------
USE_EQUITY_SIZING = os.getenv("USE_EQUITY_SIZING", "true").lower() == "true"
DEFAULT_EQUITY = float(os.getenv("DEFAULT_EQUITY", "200"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))  # 0.5%

MIN_PAIR_SCORE_TO_TRADE = float(os.getenv("MIN_PAIR_SCORE_TO_TRADE", "0.48"))
MIN_TRADES_FOR_PAIR_SCORING = int(os.getenv("MIN_TRADES_FOR_PAIR_SCORING", "8"))

AUC_WEIGHT = float(os.getenv("AUC_WEIGHT", "0.55"))
WINRATE_WEIGHT = float(os.getenv("WINRATE_WEIGHT", "0.45"))

MIN_UNITS_JPY = int(os.getenv("MIN_UNITS_JPY", "100"))
MIN_UNITS_NON_JPY = int(os.getenv("MIN_UNITS_NON_JPY", "100"))
MAX_UNITS_JPY = int(os.getenv("MAX_UNITS_JPY", "3000"))
MAX_UNITS_NON_JPY = int(os.getenv("MAX_UNITS_NON_JPY", "5000"))

_recent_signals: Dict[str, deque] = {}

# -------------------- state --------------------
_trade_count_today: Dict[str, int] = {}
_trade_day = dt.datetime.now(dt.timezone.utc).date()
_open_trade_ids: set[str] = set()

# -------------------- helpers --------------------
def utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def now_unix() -> int:
    return int(now_utc().timestamp())

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def normalize_pair(symbol: str) -> Optional[str]:
    if not symbol:
        return None
    s = symbol.strip().upper().replace("-", "_")
    if "_" in s:
        parts = s.split("_")
        if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
            return parts[0] + parts[1]
    if len(s) == 6 and s.isalpha():
        return s
    return None

def pair_to_instrument(pair6: str) -> str:
    return pair6[:3] + "_" + pair6[3:]

def instrument_is_jpy(instrument: str) -> bool:
    return instrument.upper().endswith("_JPY")

def instrument_precision(instrument: str) -> int:
    return 3 if instrument_is_jpy(instrument) else 5

def instrument_pip_size(instrument: str) -> float:
    return 0.01 if instrument_is_jpy(instrument) else 0.0001

def min_atr_for_instrument(instrument: str) -> float:
    return MIN_ATR_JPY if instrument_is_jpy(instrument) else MIN_ATR_NON_JPY

def format_oanda_price(price: float, instrument: str) -> str:
    precision = instrument_precision(instrument)
    q = Decimal("1." + ("0" * precision))
    val = Decimal(str(price)).quantize(q, rounding=ROUND_HALF_UP)
    return f"{val:.{precision}f}"

def base_units_for_instrument(instrument: str) -> int:
    return UNITS_JPY if instrument_is_jpy(instrument) else UNITS_NON_JPY

def min_units_for_instrument(instrument: str) -> int:
    return MIN_UNITS_JPY if instrument_is_jpy(instrument) else MIN_UNITS_NON_JPY

def max_units_for_instrument(instrument: str) -> int:
    return MAX_UNITS_JPY if instrument_is_jpy(instrument) else MAX_UNITS_NON_JPY

def pip_value_per_1000(instrument: str) -> float:
    return 0.10

def get_equity_used(payload_obj: Any) -> float:
    eq = safe_float(getattr(payload_obj, "equity", None), 0.0)
    nav = safe_float(getattr(payload_obj, "nav", None), 0.0)
    if eq > 0:
        return eq
    if nav > 0:
        return nav
    return DEFAULT_EQUITY

def _round_down_to_pip(price: float, pip: float) -> float:
    return math.floor(price / pip) * pip

def _round_up_to_pip(price: float, pip: float) -> float:
    return math.ceil(price / pip) * pip

def compute_sl_tp_prices(
    side: str,
    mid_c: float,
    atr14: float,
    instrument: str,
    labeling: Dict[str, Any],
    min_dist_pips: float = 5.0,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if side not in ("BUY", "SELL"):
        return (None, None, None, None)

    pip = instrument_pip_size(instrument)
    sl_atr = safe_float(labeling.get("sl_atr"), 1.0)
    tp_atr = safe_float(labeling.get("tp_atr"), 1.3)
    atr = max(float(atr14), pip)

    sl_dist = max(sl_atr * atr, min_dist_pips * pip)
    tp_dist = max(tp_atr * atr, min_dist_pips * pip)

    if side == "BUY":
        sl_raw = mid_c - sl_dist
        tp_raw = mid_c + tp_dist
        sl_price = _round_down_to_pip(sl_raw, pip)
        tp_price = _round_up_to_pip(tp_raw, pip)
        if sl_price >= mid_c:
            sl_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)
        if tp_price <= mid_c:
            tp_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
    else:
        sl_raw = mid_c + sl_dist
        tp_raw = mid_c - tp_dist
        sl_price = _round_up_to_pip(sl_raw, pip)
        tp_price = _round_down_to_pip(tp_raw, pip)
        if sl_price <= mid_c:
            sl_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
        if tp_price >= mid_c:
            tp_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)

    if abs(sl_price - tp_price) < pip:
        tp_price = tp_price + pip if side == "BUY" else tp_price - pip

    sl_str = format_oanda_price(sl_price, instrument)
    tp_str = format_oanda_price(tp_price, instrument)
    mid_str = format_oanda_price(mid_c, instrument)

    sl_price_f = float(sl_str)
    tp_price_f = float(tp_str)
    mid_c_f = float(mid_str)

    sl_pips = abs(mid_c_f - sl_price_f) / pip
    tp_pips = abs(tp_price_f - mid_c_f) / pip

    return float(sl_pips), float(tp_pips), sl_str, tp_str

def make_signal_fingerprint(
    instrument: str,
    side: str,
    bar_time: int,
    mid_c: float,
    tf: Optional[str],
) -> str:
    raw = {
        "instrument": instrument,
        "side": side,
        "bar_time": int(bar_time),
        "mid_c": round(float(mid_c), instrument_precision(instrument)),
        "tf": tf or "",
    }
    s = json.dumps(raw, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def is_duplicate_signal(pair6: str, fingerprint: str) -> bool:
    tnow = now_unix()
    q = _recent_signals.setdefault(pair6, deque())
    while q and (tnow - q[0][0] > DUP_WINDOW_SECONDS):
        q.popleft()
    return any(fp == fingerprint for _, fp in q)

def remember_signal(pair6: str, fingerprint: str) -> None:
    q = _recent_signals.setdefault(pair6, deque())
    q.append((now_unix(), fingerprint))

def payload_sanity_checks(p: "TVPayload", instrument: str) -> Optional[str]:
    if safe_float(p.spread_pips, 9999.0) > MAX_SPREAD_PIPS:
        return f"Spread too high: {p.spread_pips} pips > {MAX_SPREAD_PIPS}"
    if safe_float(p.atr14, 0.0) < min_atr_for_instrument(instrument):
        return f"ATR too small: {p.atr14}"
    if not (p.mid_l <= p.mid_c <= p.mid_h):
        return "Bad payload: mid_c not between mid_l and mid_h"
    if not (p.mid_l <= p.mid_o <= p.mid_h):
        return "Bad payload: mid_o not between mid_l and mid_h"
    if safe_float(p.spread_pips, -1.0) < 0:
        return "Bad payload: negative spread_pips"
    if safe_float(p.spread_atr, -1.0) < 0:
        return "Bad payload: negative spread_atr"
    if p.mid_h < p.mid_l:
        return "Bad payload: mid_h < mid_l"
    return None

def write_csv_row(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def write_audit_row(out: Dict[str, Any]) -> None:
    write_csv_row(AUDIT_CSV, out)

def write_trade_row(row: Dict[str, Any]) -> None:
    write_csv_row(TRADES_CSV, row)

def read_csv_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

def read_audit_df() -> pd.DataFrame:
    return read_csv_df(AUDIT_CSV)

def read_trades_df() -> pd.DataFrame:
    return read_csv_df(TRADES_CSV)

def read_closed_trades_df() -> pd.DataFrame:
    df = read_trades_df()
    if df.empty:
        return pd.DataFrame()
    return df[df["status"].isin(["CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"])].copy()

def safe_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=bool)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])

def _check_daily_reset() -> None:
    global _trade_day
    today = dt.datetime.now(dt.timezone.utc).date()
    if today != _trade_day:
        _trade_day = today
        _trade_count_today.clear()

def trades_today(pair6: str) -> int:
    _check_daily_reset()
    return _trade_count_today.get(pair6, 0)

def inc_trade(pair6: str) -> None:
    _check_daily_reset()
    _trade_count_today[pair6] = _trade_count_today.get(pair6, 0) + 1

def current_open_trade_count() -> int:
    return len(_open_trade_ids)

def can_open_trade() -> bool:
    return current_open_trade_count() < MAX_OPEN_TRADES

def note_trade_opened(order_id: Optional[str]) -> None:
    if order_id:
        _open_trade_ids.add(order_id)

def note_trade_closed(order_id: Optional[str]) -> None:
    if order_id and order_id in _open_trade_ids:
        _open_trade_ids.remove(order_id)

def pair_live_stats(instrument: str) -> Dict[str, Any]:
    df = read_closed_trades_df()
    if df.empty or "instrument" not in df.columns:
        return {"n": 0, "win_rate": None, "avg_pnl": None, "net_pnl": 0.0}

    sub = df[df["instrument"] == instrument].copy()
    if sub.empty:
        return {"n": 0, "win_rate": None, "avg_pnl": None, "net_pnl": 0.0}

    pnl = pd.to_numeric(sub.get("pnl"), errors="coerce").fillna(0.0)
    n = int(len(sub))
    wins = int((pnl > 0).sum())
    win_rate = wins / n if n else None

    return {
        "n": n,
        "win_rate": win_rate,
        "avg_pnl": float(pnl.mean()) if n else None,
        "net_pnl": float(pnl.sum()),
    }

def compute_pair_score(instrument: str, avg_auc: float) -> float:
    live = pair_live_stats(instrument)
    n = int(live["n"])
    auc_norm = max(0.0, min(1.0, (avg_auc - 0.50) / 0.10))

    if n < MIN_TRADES_FOR_PAIR_SCORING or live["win_rate"] is None:
        return auc_norm

    wr = max(0.0, min(1.0, float(live["win_rate"])))
    score = (AUC_WEIGHT * auc_norm) + (WINRATE_WEIGHT * wr)
    return max(0.0, min(1.0, score))

def compute_units_dynamic(
    instrument: str,
    sl_pips: float,
    avg_auc: float,
    pair_score: float,
    equity_used: float,
    force_units_abs: Optional[int] = None,
) -> int:
    if force_units_abs is not None:
        return max(1, abs(int(force_units_abs)))

    if sl_pips is None or sl_pips <= 0:
        return 0

    base = base_units_for_instrument(instrument)

    if USE_EQUITY_SIZING:
        risk_cap = equity_used * RISK_PCT
        risk_per_1000 = float(sl_pips) * pip_value_per_1000(instrument)
        sized = int((risk_cap / risk_per_1000) * 1000) if risk_per_1000 > 0 else base
        base = sized

    if pair_score >= 0.80:
        base = int(base * 1.35)
    elif pair_score >= 0.65:
        base = int(base * 1.15)
    elif pair_score >= 0.50:
        base = int(base * 1.00)
    else:
        base = int(base * 0.70)

    if avg_auc >= 0.57:
        base = int(base * 1.10)
    elif avg_auc < 0.54:
        base = int(base * 0.90)

    base = max(min_units_for_instrument(instrument), base)
    base = min(max_units_for_instrument(instrument), base)
    return base

# -------------------- load bundles --------------------
def load_bundles(models_dir: str) -> Dict[str, Dict[str, Any]]:
    bundles: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(models_dir, "*_bundle.joblib"))):
        b = joblib.load(path)
        pair = str(b.get("pair", "")).upper().replace("_", "")
        if pair:
            b["_bundle_path"] = path
            bundles[pair] = b
    return bundles

BUNDLES = load_bundles(MODELS_DIR)

# -------------------- API models --------------------
class TVPayload(BaseModel):
    type: Optional[str] = "fx"
    symbol: str
    tf: Optional[str] = None
    t: int

    mid_o: float
    mid_h: float
    mid_l: float
    mid_c: float

    ema20: float
    ema50: float
    ema200: float
    rsi14: float
    adx14: float
    atr14: float
    macdh: float

    ret1: float
    ret3: float
    ret6: float
    ret12: float

    d20: float
    d50: float
    d200: float
    s20: float
    s50: float
    s200: float

    bbw: float
    spread_c: float
    spread_atr: float
    spread_pips: float

    trend_regime: int
    vol_regime: int
    hr: int
    dow: int

    session: Optional[int] = None
    ema50_h4: Optional[float] = None
    adx14_h4: Optional[float] = None
    ema20_d1: Optional[float] = None
    rsi14_d1: Optional[float] = None

    equity: Optional[float] = None
    nav: Optional[float] = None

    hint_side: Optional[str] = None

    force_decision: Optional[Literal["BUY", "SELL"]] = None
    force_units_abs: Optional[int] = None

class TradeEvent(BaseModel):
    instrument: str
    side: Literal["BUY", "SELL"]
    units_signed: int
    entry_price: float
    sl_price: float
    tp_price: float
    status: Literal["OPEN", "CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"]
    pnl: Optional[float] = None
    order_id: Optional[str] = None
    reason: Optional[str] = None
    pair_score: Optional[float] = None
    ts: Optional[str] = None

def make_out(**kwargs) -> Dict[str, Any]:
    return kwargs

app = FastAPI(title="FX Sniper Per Pair", version="5.0")

@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": utc_ts(),
        "pairs_loaded": len(BUNDLES),
        "pairs": sorted([pair_to_instrument(p) for p in BUNDLES.keys()]),
        "dup_window_seconds": DUP_WINDOW_SECONDS,
        "max_spread_pips": MAX_SPREAD_PIPS,
        "units_jpy": UNITS_JPY,
        "units_non_jpy": UNITS_NON_JPY,
        "max_open_trades": MAX_OPEN_TRADES,
        "default_disagree_conf": DEFAULT_DISAGREE_CONF,
        "use_equity_sizing": USE_EQUITY_SIZING,
        "default_equity": DEFAULT_EQUITY,
        "risk_pct": RISK_PCT,
        "min_pair_score_to_trade": MIN_PAIR_SCORE_TO_TRADE,
        "current_open_trades": current_open_trade_count(),
    }

@app.get("/stats")
def stats():
    df = read_audit_df()
    if df.empty:
        return {
            "ok": True,
            "rows": 0,
            "would_order_count": 0,
            "decision_counts": {},
            "pair_counts": {},
            "last_ts": None,
        }

    decision_counts = df["decision"].value_counts(dropna=False).to_dict() if "decision" in df.columns else {}
    pair_counts = df["instrument"].value_counts(dropna=False).to_dict() if "instrument" in df.columns else {}
    would_count = int(safe_bool_series(df, "would_order").sum())

    last_ts = None
    if "ts" in df.columns and not df["ts"].dropna().empty:
        last_ts = df["ts"].dropna().max().isoformat()

    return {
        "ok": True,
        "rows": int(len(df)),
        "would_order_count": would_count,
        "decision_counts": decision_counts,
        "pair_counts": pair_counts,
        "last_ts": last_ts,
    }

@app.get("/pnl_stats")
def pnl_stats():
    df = read_trades_df()
    if df.empty:
        return {
            "ok": True,
            "trades": 0,
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "net_pnl": 0.0,
            "avg_pnl": None,
            "open_trades": current_open_trade_count(),
        }

    closed = df[df["status"].isin(["CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"])].copy()
    if closed.empty or "pnl" not in closed.columns:
        return {
            "ok": True,
            "trades": int(len(df)),
            "closed_trades": int(len(closed)),
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "net_pnl": 0.0,
            "avg_pnl": None,
            "open_trades": current_open_trade_count(),
        }

    pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    closed_n = int(len(closed))
    win_rate = wins / closed_n if closed_n else None

    return {
        "ok": True,
        "trades": int(len(df)),
        "closed_trades": closed_n,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()) if closed_n else None,
        "open_trades": current_open_trade_count(),
    }

@app.get("/pair_stats")
def pair_stats():
    df = read_closed_trades_df()
    if df.empty:
        return {"ok": True, "pairs": []}

    df["pnl"] = pd.to_numeric(df.get("pnl"), errors="coerce").fillna(0.0)
    rows = []

    for instrument, sub in df.groupby("instrument"):
        n = int(len(sub))
        wins = int((sub["pnl"] > 0).sum())
        losses = int((sub["pnl"] < 0).sum())
        win_rate = wins / n if n else None
        net_pnl = float(sub["pnl"].sum())
        avg_pnl = float(sub["pnl"].mean()) if n else None

        avg_auc = 0.0
        pair6 = instrument.replace("_", "")
        if pair6 in BUNDLES:
            avg_auc = safe_float((BUNDLES[pair6].get("train_meta") or {}).get("avg_auc"), 0.0)

        pair_score = compute_pair_score(instrument, avg_auc)

        rows.append({
            "instrument": instrument,
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "avg_pnl": avg_pnl,
            "avg_auc": avg_auc,
            "pair_score": pair_score,
            "is_tradeable": pair_score >= MIN_PAIR_SCORE_TO_TRADE,
        })

    rows = sorted(rows, key=lambda x: (x["pair_score"], x["net_pnl"]), reverse=True)
    return {"ok": True, "pairs": rows}

@app.get("/weak_pairs")
def weak_pairs():
    df = read_closed_trades_df()
    if df.empty:
        return {"ok": True, "weak_pairs": []}

    df["pnl"] = pd.to_numeric(df.get("pnl"), errors="coerce").fillna(0.0)
    weak = []

    for instrument, sub in df.groupby("instrument"):
        n = int(len(sub))
        if n < MIN_TRADES_FOR_PAIR_SCORING:
            continue

        wins = int((sub["pnl"] > 0).sum())
        win_rate = wins / n if n else 0.0

        pair6 = instrument.replace("_", "")
        avg_auc = 0.0
        if pair6 in BUNDLES:
            avg_auc = safe_float((BUNDLES[pair6].get("train_meta") or {}).get("avg_auc"), 0.0)

        pair_score = compute_pair_score(instrument, avg_auc)

        if pair_score < MIN_PAIR_SCORE_TO_TRADE:
            weak.append({
                "instrument": instrument,
                "trades": n,
                "win_rate": win_rate,
                "avg_auc": avg_auc,
                "pair_score": pair_score,
            })

    weak = sorted(weak, key=lambda x: x["pair_score"])
    return {"ok": True, "weak_pairs": weak}

@app.get("/dashboard")
def dashboard():
    audit_df = read_audit_df()
    trades_df = read_trades_df()

    total_rows = len(audit_df)
    would_count = int(safe_bool_series(audit_df, "would_order").sum()) if not audit_df.empty and "would_order" in audit_df.columns else 0
    none_count = int((audit_df["decision"] == "NONE").sum()) if not audit_df.empty and "decision" in audit_df.columns else 0

    if not audit_df.empty and "ts" in audit_df.columns:
        latest_rows = audit_df.sort_values("ts", ascending=False).head(50).copy()
    elif not audit_df.empty:
        latest_rows = audit_df.tail(50).copy()
    else:
        latest_rows = pd.DataFrame()

    cols = [
        c for c in [
            "ts", "instrument", "symbol", "hint_side", "decision", "confidence",
            "side_prob", "p_up", "margin", "pair_score", "equity_used",
            "units_signed", "sl_price", "tp_price", "would_order", "why",
        ] if c in latest_rows.columns
    ]

    table_html = latest_rows[cols].to_html(index=False, escape=False) if not latest_rows.empty else "<p>No audit data yet.</p>"

    by_pair_html = "<p>No audit data yet.</p>"
    if not audit_df.empty and "instrument" in audit_df.columns:
        by_pair = (
            audit_df["instrument"]
            .value_counts()
            .rename_axis("instrument")
            .reset_index(name="count")
        )
        by_pair_html = by_pair.to_html(index=False)

    pnl_html = "<p>No trade data yet.</p>"
    if not trades_df.empty:
        pnl_cols = [c for c in ["ts", "instrument", "side", "units_signed", "status", "pnl", "reason", "order_id"] if c in trades_df.columns]
        pnl_html = trades_df.sort_values("ts", ascending=False).head(50)[pnl_cols].to_html(index=False, escape=False)

    html = f"""
    <html>
      <head>
        <title>FX Sniper Dashboard</title>
        <meta http-equiv="refresh" content="15">
      </head>
      <body style="font-family: Arial; padding: 24px;">
        <h1>FX Sniper Dashboard</h1>

        <div style="display:flex; gap:24px; margin-bottom:24px; flex-wrap:wrap;">
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;">
            <h3>Total predictions</h3>
            <div style="font-size:28px;">{total_rows}</div>
          </div>
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;">
            <h3>Would order</h3>
            <div style="font-size:28px;">{would_count}</div>
          </div>
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;">
            <h3>Blocked / NONE</h3>
            <div style="font-size:28px;">{none_count}</div>
          </div>
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;">
            <h3>Open trades tracked</h3>
            <div style="font-size:28px;">{current_open_trade_count()}</div>
          </div>
        </div>

        <h2>By pair</h2>
        {by_pair_html}

        <h2>Latest predictions</h2>
        {table_html}

        <h2>Latest trade events</h2>
        {pnl_html}

        <p style="margin-top:24px;">
          JSON endpoints:
          <a href="/health">/health</a> |
          <a href="/stats">/stats</a> |
          <a href="/pnl_stats">/pnl_stats</a> |
          <a href="/pair_stats">/pair_stats</a> |
          <a href="/weak_pairs">/weak_pairs</a>
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/trade_event")
def trade_event(t: TradeEvent):
    row = t.model_dump()
    if not row.get("ts"):
        row["ts"] = utc_ts()
    if row.get("pair_score") is None:
        row["pair_score"] = None

    write_trade_row(row)

    if t.status == "OPEN":
        note_trade_opened(t.order_id)
    if t.status in ("CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"):
        note_trade_closed(t.order_id)

    return {
        "ok": True,
        "open_trades": current_open_trade_count(),
        "status": t.status,
        "order_id": t.order_id,
    }

@app.post("/predict")
def predict(p: TVPayload):
    pair6 = normalize_pair(p.symbol)

    if pair6 is None or pair6 not in BUNDLES:
        out = make_out(
            decision="NONE",
            confidence=0.0,
            side_prob=0.0,
            p_up=0.0,
            margin=0.0,
            why="Symbol not allowed",
            ts=utc_ts(),
            pair="",
            instrument="",
            symbol=p.symbol,
            hint_side=str(getattr(p, "hint_side", "") or "").upper(),
            model_version="",
            avg_auc=0.0,
            pair_score=None,
            equity_used=get_equity_used(p),
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            trend_regime=int(getattr(p, "trend_regime", 0) or 0),
            vol_regime=int(getattr(p, "vol_regime", 0) or 0),
            spread_pips=float(getattr(p, "spread_pips", 0.0) or 0.0),
            spread_atr=float(getattr(p, "spread_atr", 0.0) or 0.0),
            would_order=False,
        )
        write_audit_row(out)
        return out

    b = BUNDLES[pair6]
    instrument = pair_to_instrument(pair6)
    model_version = str(b.get("model_version", ""))
    labeling = b.get("labeling", {}) or {}
    train_meta = b.get("train_meta", {}) or {}
    avg_auc = safe_float(train_meta.get("avg_auc"), 0.0)
    hint_side = str(getattr(p, "hint_side", "") or "").upper()
    equity_used = get_equity_used(p)
    pair_score = compute_pair_score(instrument, avg_auc)

    gate = PAIR_GATES.get(instrument, DEFAULT_GATE)
    conf_gate = float(gate["conf"])
    margin_gate = float(gate["margin"])

    if pair_score < MIN_PAIR_SCORE_TO_TRADE:
        out = make_out(
            decision="NONE",
            confidence=0.0,
            side_prob=0.0,
            p_up=0.0,
            margin=0.0,
            why=f"Pair blocked: {instrument} score {pair_score:.2f} < {MIN_PAIR_SCORE_TO_TRADE:.2f}",
            ts=utc_ts(),
            pair=instrument,
            instrument=instrument,
            symbol=p.symbol,
            hint_side=hint_side,
            model_version=model_version,
            avg_auc=avg_auc,
            pair_score=pair_score,
            equity_used=equity_used,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            trend_regime=int(getattr(p, "trend_regime", 0) or 0),
            vol_regime=int(getattr(p, "vol_regime", 0) or 0),
            spread_pips=float(getattr(p, "spread_pips", 0.0) or 0.0),
            spread_atr=float(getattr(p, "spread_atr", 0.0) or 0.0),
            would_order=False,
        )
        write_audit_row(out)
        return out

    bad_payload_reason = payload_sanity_checks(p, instrument)
    if bad_payload_reason:
        out = make_out(
            decision="NONE",
            confidence=0.0,
            side_prob=0.0,
            p_up=0.0,
            margin=0.0,
            why=bad_payload_reason,
            ts=utc_ts(),
            pair=instrument,
            instrument=instrument,
            symbol=p.symbol,
            hint_side=hint_side,
            model_version=model_version,
            avg_auc=avg_auc,
            pair_score=pair_score,
            equity_used=equity_used,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            trend_regime=int(getattr(p, "trend_regime", 0) or 0),
            vol_regime=int(getattr(p, "vol_regime", 0) or 0),
            spread_pips=float(getattr(p, "spread_pips", 0.0) or 0.0),
            spread_atr=float(getattr(p, "spread_atr", 0.0) or 0.0),
            would_order=False,
        )
        write_audit_row(out)
        return out

    if p.force_decision in ("BUY", "SELL"):
        side = p.force_decision
        fingerprint = make_signal_fingerprint(
            instrument=instrument,
            side=side,
            bar_time=p.t,
            mid_c=float(p.mid_c),
            tf=p.tf,
        )

        if is_duplicate_signal(pair6, fingerprint):
            out = make_out(
                decision="NONE",
                confidence=1.0,
                side_prob=1.0,
                p_up=1.0 if side == "BUY" else 0.0,
                margin=1.0,
                why=f"Duplicate signal blocked for {instrument}",
                ts=utc_ts(),
                pair=instrument,
                instrument=instrument,
                symbol=p.symbol,
                hint_side=hint_side,
                model_version=model_version,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                trend_regime=int(p.trend_regime),
                vol_regime=int(p.vol_regime),
                spread_pips=float(p.spread_pips),
                spread_atr=float(p.spread_atr),
                would_order=False,
            )
            write_audit_row(out)
            return out

        if not can_open_trade():
            out = make_out(
                decision="NONE",
                confidence=1.0,
                side_prob=1.0,
                p_up=1.0 if side == "BUY" else 0.0,
                margin=1.0,
                why=f"Open trade cap reached ({MAX_OPEN_TRADES})",
                ts=utc_ts(),
                pair=instrument,
                instrument=instrument,
                symbol=p.symbol,
                hint_side=hint_side,
                model_version=model_version,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                trend_regime=int(p.trend_regime),
                vol_regime=int(p.vol_regime),
                spread_pips=float(p.spread_pips),
                spread_atr=float(p.spread_atr),
                would_order=False,
            )
            write_audit_row(out)
            return out

        sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(
            side=side,
            mid_c=float(p.mid_c),
            atr14=float(p.atr14),
            instrument=instrument,
            labeling=labeling,
        )

        units_abs = compute_units_dynamic(
            instrument=instrument,
            sl_pips=sl_pips,
            avg_auc=avg_auc,
            pair_score=pair_score,
            equity_used=equity_used,
            force_units_abs=p.force_units_abs,
        )
        units_signed = units_abs if side == "BUY" else -units_abs

        out = make_out(
            decision=side,
            confidence=1.0,
            side_prob=1.0,
            p_up=1.0 if side == "BUY" else 0.0,
            margin=1.0,
            why="FORCED decision (bypassed model/gates)",
            ts=utc_ts(),
            pair=instrument,
            instrument=instrument,
            symbol=p.symbol,
            hint_side=hint_side,
            model_version=model_version,
            avg_auc=avg_auc,
            pair_score=pair_score,
            equity_used=equity_used,
            units=units_abs,
            units_signed=units_signed,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            sl_price=sl_price,
            tp_price=tp_price,
            trend_regime=int(p.trend_regime),
            vol_regime=int(p.vol_regime),
            spread_pips=float(p.spread_pips),
            spread_atr=float(p.spread_atr),
            would_order=True,
        )

        if trades_today(pair6) >= MAX_TRADES_PER_DAY_TOTAL:
            out["decision"] = "NONE"
            out["would_order"] = False
            out["units"] = None
            out["units_signed"] = None
            out["sl_pips"] = None
            out["tp_pips"] = None
            out["sl_price"] = None
            out["tp_price"] = None
            out["why"] = f"Daily lock: max trades for {instrument} reached"
            write_audit_row(out)
            return out

        remember_signal(pair6, fingerprint)
        inc_trade(pair6)
        write_audit_row(out)
        return out

    try:
        feat_order: List[str] = list(b.get("feature_order") or [])
        data = p.model_dump()

        missing = [f for f in feat_order if f not in data or data[f] is None]
        if missing:
            out = make_out(
                decision="NONE",
                confidence=0.0,
                side_prob=0.0,
                p_up=0.0,
                margin=0.0,
                why=f"Missing feature in payload: '{missing[0]}'",
                ts=utc_ts(),
                pair=instrument,
                instrument=instrument,
                symbol=p.symbol,
                hint_side=hint_side,
                model_version=model_version,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                trend_regime=int(p.trend_regime),
                vol_regime=int(p.vol_regime),
                spread_pips=float(p.spread_pips),
                spread_atr=float(p.spread_atr),
                would_order=False,
            )
            write_audit_row(out)
            return out

        X = pd.DataFrame([{f: data[f] for f in feat_order}], columns=feat_order)

        model = b["model"]
        calibrator = b.get("calibrator", None)

        proba = model.predict_proba(X)[0]
        p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])

        side = "BUY" if p_up >= 0.5 else "SELL"
        side_prob = p_up if side == "BUY" else (1.0 - p_up)

        conf = side_prob
        if calibrator is not None:
            try:
                conf = float(calibrator.predict([side_prob])[0])
            except Exception:
                conf = side_prob

        conf = max(0.0, min(1.0, conf))
        side_prob = max(0.0, min(1.0, side_prob))
        p_up = max(0.0, min(1.0, p_up))
        margin = float(abs(p_up - 0.5) * 2.0)

        disagree_conf_gate = PAIR_DISAGREE_CONF.get(instrument, DEFAULT_DISAGREE_CONF)
        hint_disagrees = hint_side in ("BUY", "SELL") and side != hint_side

        if hint_disagrees and conf < disagree_conf_gate:
            out = make_out(
                decision="NONE",
                confidence=float(conf),
                side_prob=float(side_prob),
                p_up=float(p_up),
                margin=float(margin),
                why=(
                    f"Blocked disagreement: ML {side} vs hint {hint_side} "
                    f"(conf {conf:.2f} < {disagree_conf_gate:.2f})"
                ),
                ts=utc_ts(),
                pair=instrument,
                instrument=instrument,
                symbol=p.symbol,
                hint_side=hint_side,
                model_version=model_version,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                trend_regime=int(p.trend_regime),
                vol_regime=int(p.vol_regime),
                spread_pips=float(p.spread_pips),
                spread_atr=float(p.spread_atr),
                would_order=False,
            )
            write_audit_row(out)
            return out

        would_order = (conf >= conf_gate) and (margin >= margin_gate)

        fingerprint = make_signal_fingerprint(
            instrument=instrument,
            side=side,
            bar_time=p.t,
            mid_c=float(p.mid_c),
            tf=p.tf,
        )

        if would_order and is_duplicate_signal(pair6, fingerprint):
            out = make_out(
                decision="NONE",
                confidence=float(conf),
                side_prob=float(side_prob),
                p_up=float(p_up),
                margin=float(margin),
                why=f"Duplicate signal blocked for {instrument}",
                ts=utc_ts(),
                pair=instrument,
                instrument=instrument,
                symbol=p.symbol,
                hint_side=hint_side,
                model_version=model_version,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                trend_regime=int(p.trend_regime),
                vol_regime=int(p.vol_regime),
                spread_pips=float(p.spread_pips),
                spread_atr=float(p.spread_atr),
                would_order=False,
            )
            write_audit_row(out)
            return out

        if would_order and trades_today(pair6) >= MAX_TRADES_PER_DAY_TOTAL:
            out = make_out(
                decision="NONE",
                confidence=float(conf),
                side_prob=float(side_prob),
                p_up=float(p_up),
                margin=float(margin),
                why=f"Daily lock: max trades for {instrument} reached",
                ts=utc_ts(),
                pair=instrument,
                instrument=instrument,
                symbol=p.symbol,
                hint_side=hint_side,
                model_version=model_version,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                trend_regime=int(p.trend_regime),
                vol_regime=int(p.vol_regime),
                spread_pips=float(p.spread_pips),
                spread_atr=float(p.spread_atr),
                would_order=False,
            )
            write_audit_row(out)
            return out

        if would_order and not can_open_trade():
            out = make_out(
                decision="NONE",
                confidence=float(conf),
                side_prob=float(side_prob),
                p_up=float(p_up),
                margin=float(margin),
                why=f"Open trade cap reached ({MAX_OPEN_TRADES})",
                ts=utc_ts(),
                pair=instrument,
                instrument=instrument,
                symbol=p.symbol,
                hint_side=hint_side,
                model_version=model_version,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                trend_regime=int(p.trend_regime),
                vol_regime=int(p.vol_regime),
                spread_pips=float(p.spread_pips),
                spread_atr=float(p.spread_atr),
                would_order=False,
            )
            write_audit_row(out)
            return out

        units_abs = None
        units_signed = None
        sl_pips = None
        tp_pips = None
        sl_price = None
        tp_price = None

        if would_order:
            sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(
                side=side,
                mid_c=float(p.mid_c),
                atr14=float(p.atr14),
                instrument=instrument,
                labeling=labeling,
            )

            units_abs = compute_units_dynamic(
                instrument=instrument,
                sl_pips=sl_pips,
                avg_auc=avg_auc,
                pair_score=pair_score,
                equity_used=equity_used,
                force_units_abs=p.force_units_abs,
            )
            units_signed = units_abs if side == "BUY" else -units_abs

        out = make_out(
            decision=side if would_order else "NONE",
            confidence=float(conf),
            side_prob=float(side_prob),
            p_up=float(p_up),
            margin=float(margin),
            why=(
                f"OK: {side} passed | conf={conf:.2f}/{conf_gate:.2f}, "
                f"margin={margin:.2f}/{margin_gate:.2f}, "
                f"hint={hint_side or 'NONE'}, "
                f"pair_score={pair_score:.2f}, "
                f"equity_used={equity_used:.2f}"
                if would_order
                else (
                    f"Below sniper gate | conf={conf:.2f}/{conf_gate:.2f}, "
                    f"margin={margin:.2f}/{margin_gate:.2f}, "
                    f"hint={hint_side or 'NONE'}, "
                    f"pair_score={pair_score:.2f}"
                )
            ),
            ts=utc_ts(),
            pair=instrument,
            instrument=instrument,
            symbol=p.symbol,
            hint_side=hint_side,
            model_version=model_version,
            avg_auc=avg_auc,
            pair_score=pair_score,
            equity_used=equity_used,
            units=units_abs,
            units_signed=units_signed,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            sl_price=sl_price,
            tp_price=tp_price,
            trend_regime=int(p.trend_regime),
            vol_regime=int(p.vol_regime),
            spread_pips=float(p.spread_pips),
            spread_atr=float(p.spread_atr),
            would_order=bool(would_order),
        )

        if would_order:
            remember_signal(pair6, fingerprint)
            inc_trade(pair6)

        write_audit_row(out)
        return out

    except Exception as e:
        out = make_out(
            decision="NONE",
            confidence=0.0,
            side_prob=0.0,
            p_up=0.0,
            margin=0.0,
            why=f"Prediction error: {repr(e)}",
            ts=utc_ts(),
            pair=instrument,
            instrument=instrument,
            symbol=p.symbol,
            hint_side=hint_side,
            model_version=model_version,
            avg_auc=avg_auc,
            pair_score=pair_score,
            equity_used=equity_used,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            trend_regime=int(getattr(p, "trend_regime", 0) or 0),
            vol_regime=int(getattr(p, "vol_regime", 0) or 0),
            spread_pips=float(getattr(p, "spread_pips", 0.0) or 0.0),
            spread_atr=float(getattr(p, "spread_atr", 0.0) or 0.0),
            would_order=False,
        )
        write_audit_row(out)
        return out