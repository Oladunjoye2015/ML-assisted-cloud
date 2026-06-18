from __future__ import annotations
import os, csv, glob, math, json, hashlib, datetime as dt, sqlite3, threading, time
from collections import deque
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Literal, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

# Optional model engines for the new H1 auto-model registry.
# Server still boots with the older joblib bundles if one of these is unavailable.
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, ConfigDict

# ====================================================
# ENV
# ====================================================
MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOG_DIR = os.getenv("LOG_DIR", "logs")
DATA_DIR = os.getenv("DATA_DIR", "oanda_h1_ba_live")
DB_PATH = os.getenv("DB_PATH", os.path.join(LOG_DIR, "fx_trading.db"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

AUDIT_CSV = os.path.join(LOG_DIR, "audit.csv")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")

MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "60"))
AUTO_CLOSE_ENABLED = os.getenv("AUTO_CLOSE_ENABLED", "true").lower() == "true"
AUTO_CLOSE_CHECK_SECONDS = int(os.getenv("AUTO_CLOSE_CHECK_SECONDS", "1800"))
AUTO_CLOSE_ALLOW_POSITION_FALLBACK = os.getenv("AUTO_CLOSE_ALLOW_POSITION_FALLBACK", "false").lower() == "true"

OANDA_TOKEN = os.getenv("OANDA_TOKEN", "").strip()
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_BASE_URL = os.getenv("OANDA_BASE_URL", "https://api-fxtrade.oanda.com").strip().rstrip("/")

DEFAULT_GATE = {"conf": float(os.getenv("CONF_GATE", "0.54")), "margin": float(os.getenv("MARGIN_GATE", "0.04"))}

PAIR_GATES: Dict[str, Dict[str, float]] = {
    "EUR_GBP": {"conf": 0.57, "margin": 0.05},
    "USD_CAD": {"conf": 0.54, "margin": 0.04},
    "CAD_JPY": {"conf": 0.58, "margin": 0.05},
    "AUD_JPY": {"conf": 0.58, "margin": 0.05},
    "USD_JPY": {"conf": 0.58, "margin": 0.05},
    "EUR_JPY": {"conf": 0.60, "margin": 0.06},
    "GBP_JPY": {"conf": 0.60, "margin": 0.06},
    "NZD_JPY": {"conf": 0.57, "margin": 0.05},
    "GBP_CHF": {"conf": 0.59, "margin": 0.06},
    "AUD_CAD": {"conf": 0.56, "margin": 0.05},
    "USD_CHF": {"conf": 0.54, "margin": 0.04},
}

PAIR_DISAGREE_CONF: Dict[str, float] = {
    "EUR_GBP": 0.64,
    "CAD_JPY": 0.66,
    "AUD_JPY": 0.66,
    "GBP_JPY": 0.66,
    "USD_CHF": 0.60,
    "USD_CAD": 0.60,
    "USD_JPY": 0.65,
    "EUR_JPY": 0.66,
    "NZD_JPY": 0.64,
    "GBP_CHF": 0.66,
    "AUD_CAD": 0.63,
}

DEFAULT_DISAGREE_CONF = float(os.getenv("DEFAULT_DISAGREE_CONF", "0.62"))
UNITS_JPY = int(os.getenv("UNITS_JPY", "1000"))
UNITS_NON_JPY = int(os.getenv("UNITS_NON_JPY", "2000"))
MAX_TRADES_PER_DAY_TOTAL = int(os.getenv("MAX_TRADES_PER_DAY_TOTAL", "3"))  # H1: reduce daily exposure
MAX_TRADES_PER_DAY_PER_PAIR = int(os.getenv("MAX_TRADES_PER_DAY_PER_PAIR", "1"))  # H1: avoid repeat losses on same pair
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "1"))  # H1: one open trade only
DUP_WINDOW_SECONDS = int(os.getenv("DUP_WINDOW_SECONDS", "300"))
MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "3.5"))
MIN_ATR_NON_JPY = float(os.getenv("MIN_ATR_NON_JPY", "0.00005"))
MIN_ATR_JPY = float(os.getenv("MIN_ATR_JPY", "0.005"))
USE_EQUITY_SIZING = os.getenv("USE_EQUITY_SIZING", "true").lower() == "true"
DEFAULT_EQUITY = float(os.getenv("DEFAULT_EQUITY", "200"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.0015"))  # H1: smaller risk because losses are larger
MIN_PAIR_SCORE_TO_TRADE = float(os.getenv("MIN_PAIR_SCORE_TO_TRADE", "0.25"))
MIN_TRADES_FOR_PAIR_SCORING = int(os.getenv("MIN_TRADES_FOR_PAIR_SCORING", "20"))
AUC_WEIGHT = float(os.getenv("AUC_WEIGHT", "0.80"))
WINRATE_WEIGHT = float(os.getenv("WINRATE_WEIGHT", "0.20"))
MIN_UNITS_JPY = int(os.getenv("MIN_UNITS_JPY", "100"))
MIN_UNITS_NON_JPY = int(os.getenv("MIN_UNITS_NON_JPY", "100"))
MAX_UNITS_JPY = int(os.getenv("MAX_UNITS_JPY", "3000"))
MAX_UNITS_NON_JPY = int(os.getenv("MAX_UNITS_NON_JPY", "5000"))
DEFAULT_SL_ATR = float(os.getenv("DEFAULT_SL_ATR", "0.90"))  # H1: tighter initial risk unless model label overrides
DEFAULT_TP_ATR = float(os.getenv("DEFAULT_TP_ATR", "1.40"))  # H1: improve reward/risk
BAR_HISTORY_LEN = int(os.getenv("BAR_HISTORY_LEN", "300"))

# ====================================================
# H1 HYBRID MODEL FEATURE SOURCE
# TradingView/Make triggers the signal, but OANDA candles can provide the
# actual model features before order approval. This mirrors your M15 hybrid flow.
# ====================================================
MARKET_CONTEXT_ENABLED = os.getenv("MARKET_CONTEXT_ENABLED", "true").lower() == "true"
MARKET_CONTEXT_GRANULARITIES = [
    x.strip().upper()
    for x in os.getenv("MARKET_CONTEXT_GRANULARITIES", "H1,H4,D").split(",")
    if x.strip()
]
MARKET_CONTEXT_CANDLE_COUNT = int(os.getenv("MARKET_CONTEXT_CANDLE_COUNT", "160"))
MARKET_CONTEXT_REQUIRED = os.getenv("MARKET_CONTEXT_REQUIRED", "false").lower() == "true"
MARKET_CONTEXT_MAX_FETCH_SECONDS = int(os.getenv("MARKET_CONTEXT_MAX_FETCH_SECONDS", "20"))

MODEL_FEATURE_SOURCE = os.getenv("MODEL_FEATURE_SOURCE", "hybrid").strip().lower()
if MODEL_FEATURE_SOURCE not in {"alert", "oanda", "hybrid"}:
    MODEL_FEATURE_SOURCE = "hybrid"

MODEL_FEATURE_OANDA_GRANULARITY = os.getenv("MODEL_FEATURE_OANDA_GRANULARITY", "H1").strip().upper()
MODEL_FEATURE_OANDA_CANDLE_COUNT = int(os.getenv("MODEL_FEATURE_OANDA_CANDLE_COUNT", "240"))
MODEL_FEATURE_OANDA_MIN_CANDLES = int(os.getenv("MODEL_FEATURE_OANDA_MIN_CANDLES", "80"))
MODEL_FEATURE_FALLBACK_TO_ALERT = os.getenv("MODEL_FEATURE_FALLBACK_TO_ALERT", "true").lower() == "true"


# New H1 auto-model registry settings.
# Supports CatBoost, LightGBM, and Neural TCN challenger models trained per pair.
AUTO_MODEL_REGISTRY_ENABLED = os.getenv("AUTO_MODEL_REGISTRY_ENABLED", "true").lower() == "true"
AUTO_REGISTRY_PATH = os.getenv("AUTO_REGISTRY_PATH", os.path.join(MODELS_DIR, "registry.json"))
TCN_LOOKBACK = int(os.getenv("TCN_LOOKBACK", "48"))
AUTO_REGISTRY_OVERRIDES_JOBLIB = os.getenv("AUTO_REGISTRY_OVERRIDES_JOBLIB", "true").lower() == "true"
TORCH_DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


# ====================================================
# H1 SAFETY / CONTEXT / AI REVIEW LAYERS
# Adapted from the M15 server while keeping the H1 model registry architecture.
# Defaults are conservative but less tight than M15 because H1 candles are larger.
# ====================================================
NOISE_FILTER_ENABLED = os.getenv("NOISE_FILTER_ENABLED", "true").lower() == "true"
MIN_NOISE_RANGE_PIPS = float(os.getenv("MIN_NOISE_RANGE_PIPS", "3.0"))
MIN_NOISE_ATR_PIPS = float(os.getenv("MIN_NOISE_ATR_PIPS", "5.0"))
MIN_BODY_RANGE_RATIO = float(os.getenv("MIN_BODY_RANGE_RATIO", "0.12"))
MIN_RANGE_ATR_RATIO = float(os.getenv("MIN_RANGE_ATR_RATIO", "0.20"))
MAX_SPREAD_RANGE_RATIO = float(os.getenv("MAX_SPREAD_RANGE_RATIO", "0.35"))
MAX_WICK_BODY_RATIO = float(os.getenv("MAX_WICK_BODY_RATIO", "8.0"))
NOISE_FILTER_REQUIRE_TREND_ALIGNMENT = os.getenv("NOISE_FILTER_REQUIRE_TREND_ALIGNMENT", "false").lower() == "true"
MIN_TREND_DIST_ABS = float(os.getenv("MIN_TREND_DIST_ABS", "0.00005"))

NEWS_FILTER_ENABLED = os.getenv("NEWS_FILTER_ENABLED", "true").lower() == "true"
NEWS_BLOCK_BEFORE_MIN = int(os.getenv("NEWS_BLOCK_BEFORE_MIN", "60"))
NEWS_BLOCK_AFTER_MIN = int(os.getenv("NEWS_BLOCK_AFTER_MIN", "30"))
NEWS_BLOCK_IMPACTS = {x.strip().upper() for x in os.getenv("NEWS_BLOCK_IMPACTS", "HIGH,RED").split(",") if x.strip()}
NEWS_EVENTS_FILE = os.getenv("NEWS_EVENTS_FILE", os.path.join(DATA_DIR, "news_events.json"))
NEWS_EVENTS_JSON = os.getenv("NEWS_EVENTS_JSON", "").strip()
NEWS_MANUAL_BLACKOUT_UTC = os.getenv("NEWS_MANUAL_BLACKOUT_UTC", "").strip()
NEWS_BLOCK_ALL_CURRENCIES = os.getenv("NEWS_BLOCK_ALL_CURRENCIES", "false").lower() == "true"
NEWS_BLOCK_UNKNOWN_CURRENCY = os.getenv("NEWS_BLOCK_UNKNOWN_CURRENCY", "false").lower() == "true"
NEWS_KEEP_PAST_HOURS = int(os.getenv("NEWS_KEEP_PAST_HOURS", "24"))
NEWS_DEFAULT_TITLE = os.getenv("NEWS_DEFAULT_TITLE", "economic_news")

SIGNAL_STALENESS_GUARD_ENABLED = os.getenv("SIGNAL_STALENESS_GUARD_ENABLED", "true").lower() == "true"
SIGNAL_MAX_AGE_SECONDS = int(os.getenv("SIGNAL_MAX_AGE_SECONDS", "7200"))

DIRECTION_CONFIRMATION_ENABLED = os.getenv("DIRECTION_CONFIRMATION_ENABLED", "true").lower() == "true"
DIRECTION_CONFIRMATION_REQUIRED = os.getenv("DIRECTION_CONFIRMATION_REQUIRED", "false").lower() == "true"
DIRECTION_CONFIRM_MIN_SCORE = int(os.getenv("DIRECTION_CONFIRM_MIN_SCORE", "3"))
DIRECTION_CONFIRM_EMA_BUFFER_PIPS = float(os.getenv("DIRECTION_CONFIRM_EMA_BUFFER_PIPS", "0.5"))
DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA50 = os.getenv("DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA50", "false").lower() == "true"
DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA200 = os.getenv("DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA200", "false").lower() == "true"
DIRECTION_CONFIRM_BLOCK_STRONG_OPPOSITE_CANDLE = os.getenv("DIRECTION_CONFIRM_BLOCK_STRONG_OPPOSITE_CANDLE", "true").lower() == "true"
DIRECTION_CONFIRM_STRONG_BODY_RATIO = float(os.getenv("DIRECTION_CONFIRM_STRONG_BODY_RATIO", "0.45"))
DIRECTION_CONFIRM_MIN_BODY_PIPS = float(os.getenv("DIRECTION_CONFIRM_MIN_BODY_PIPS", "2.0"))

ENTRY_REVERSAL_GUARD_ENABLED = os.getenv("ENTRY_REVERSAL_GUARD_ENABLED", "true").lower() == "true"
ENTRY_REVERSAL_GUARD_REQUIRED = os.getenv("ENTRY_REVERSAL_GUARD_REQUIRED", "false").lower() == "true"
ENTRY_REVERSAL_MAX_ADVERSE_PIPS = float(os.getenv("ENTRY_REVERSAL_MAX_ADVERSE_PIPS", "4.0"))
ENTRY_REVERSAL_MAX_SPREAD_PIPS = float(os.getenv("ENTRY_REVERSAL_MAX_SPREAD_PIPS", "3.2"))
LIVE_PRICE_MAX_AGE_SECONDS = int(os.getenv("LIVE_PRICE_MAX_AGE_SECONDS", "20"))

AI_REVIEW_ENABLED = os.getenv("AI_REVIEW_ENABLED", "false").lower() == "true"
AI_REVIEW_PROVIDER = os.getenv("AI_REVIEW_PROVIDER", "openai").strip().lower()
AI_REVIEW_MODEL = os.getenv("AI_REVIEW_MODEL", "gpt-4o-mini").strip()
AI_REVIEW_MAX_RISK_SCORE = int(os.getenv("AI_REVIEW_MAX_RISK_SCORE", "60"))
# Side-aware AI review gates copied from M15 v16 and adapted to H1.
# Normal allow: risk <= AI_REVIEW_MAX_RISK_SCORE
# Conditional allow: risk <= AI_REVIEW_CONDITIONAL_RISK_SCORE only when model probability is strong
# Hard block: risk >= AI_REVIEW_HARD_BLOCK_SCORE
AI_REVIEW_CONDITIONAL_RISK_SCORE = int(os.getenv("AI_REVIEW_CONDITIONAL_RISK_SCORE", "75"))
AI_REVIEW_HARD_BLOCK_SCORE = int(os.getenv("AI_REVIEW_HARD_BLOCK_SCORE", "85"))
AI_REVIEW_MIN_MODEL_PROB = float(os.getenv("AI_REVIEW_MIN_MODEL_PROB", "0.52"))
AI_REVIEW_STRONG_MODEL_PROB = float(os.getenv("AI_REVIEW_STRONG_MODEL_PROB", "0.58"))
AI_REVIEW_MAX_SPREAD_ATR = float(os.getenv("AI_REVIEW_MAX_SPREAD_ATR", "0.18"))
# When API keys are missing, use deterministic side-aware reviewer instead of blocking all trades.
AI_REVIEW_FALLBACK_TO_RULES = os.getenv("AI_REVIEW_FALLBACK_TO_RULES", "true").lower() == "true"
AI_REVIEW_TIMEOUT_SECONDS = int(os.getenv("AI_REVIEW_TIMEOUT_SECONDS", "25"))
AI_REVIEW_REQUIRE_APPROVAL = os.getenv("AI_REVIEW_REQUIRE_APPROVAL", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

# ====================================================
# H1 FOREX TECHNICAL REVIEW / LOSS CONTROL
# Adapted from your M15 v17 technical-review layer for H1/H4/D.
# It is a deterministic final confirmation after the ML gate, designed
# to reduce large H1 losses by blocking trades fighting H1/H4/D structure.
# ====================================================
TECHNICAL_REVIEW_ENABLED = os.getenv("TECHNICAL_REVIEW_ENABLED", "true").lower() == "true"
TECHNICAL_REVIEW_REQUIRED = os.getenv("TECHNICAL_REVIEW_REQUIRED", "true").lower() == "true"
TECH_MIN_SCORE_FOR_BUY = float(os.getenv("TECH_MIN_SCORE_FOR_BUY", "62"))
TECH_MIN_SCORE_FOR_SELL = float(os.getenv("TECH_MIN_SCORE_FOR_SELL", "62"))
TECH_STRONG_SCORE = float(os.getenv("TECH_STRONG_SCORE", "72"))
TECH_MIN_ALIGNED_TIMEFRAMES = int(os.getenv("TECH_MIN_ALIGNED_TIMEFRAMES", "2"))
TECH_HARD_BLOCK_OPPOSITE_H1 = os.getenv("TECH_HARD_BLOCK_OPPOSITE_H1", "true").lower() == "true"
TECH_HARD_BLOCK_OPPOSITE_H4 = os.getenv("TECH_HARD_BLOCK_OPPOSITE_H4", "false").lower() == "true"
TECH_REQUIRE_H1_ALIGNMENT = os.getenv("TECH_REQUIRE_H1_ALIGNMENT", "true").lower() == "true"
TECH_REQUIRE_H4_OR_D_ALIGNMENT = os.getenv("TECH_REQUIRE_H4_OR_D_ALIGNMENT", "true").lower() == "true"
TECH_BLOCK_HIGH_SPREAD_ATR = os.getenv("TECH_BLOCK_HIGH_SPREAD_ATR", "true").lower() == "true"
TECH_MAX_SPREAD_ATR = float(os.getenv("TECH_MAX_SPREAD_ATR", "0.18"))
TECH_BLOCK_NEAR_SR = os.getenv("TECH_BLOCK_NEAR_SR", "true").lower() == "true"
TECH_NEAR_SR_ATR_MULT = float(os.getenv("TECH_NEAR_SR_ATR_MULT", "0.35"))

NEWS_EVENTS: List[Dict[str, Any]] = []

PAIR_MAP: Dict[str, str] = {
    "EURGBP": "EUR_GBP",
    "USDCAD": "USD_CAD",
    "CADJPY": "CAD_JPY",
    "AUDJPY": "AUD_JPY",
    "USDJPY": "USD_JPY",
    "EURJPY": "EUR_JPY",
    "GBPJPY": "GBP_JPY",
    "NZDJPY": "NZD_JPY",
    "GBPCHF": "GBP_CHF",
    "AUDCAD": "AUD_CAD",
    "USDCHF": "USD_CHF",
}

INSTRUMENT_TO_PAIR6 = {v: k for k, v in PAIR_MAP.items()}
JPY_INSTRUMENTS = {v for v in PAIR_MAP.values() if v.endswith("_JPY")}

_recent_signals: Dict[str, deque] = {}
_bar_history: Dict[str, deque] = {pair6: deque(maxlen=BAR_HISTORY_LEN) for pair6 in PAIR_MAP}
_tcn_feature_history: Dict[str, deque] = {}
_trade_count_today: Dict[str, int] = {}
_trade_day = dt.datetime.now(dt.timezone.utc).date()
_open_trade_ids: set[str] = set()
_open_trade_meta: Dict[str, Dict[str, Any]] = {}

# ====================================================
# UTILS
# ====================================================
def utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def now_unix() -> int:
    return int(now_utc().timestamp())

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default

def normalize_pair(symbol: str) -> Optional[str]:
    if not symbol:
        return None
    s = str(symbol).strip().upper().replace("-", "").replace("_", "").replace("/", "")
    if len(s) == 6 and s.isalpha() and s in PAIR_MAP:
        return s
    return None

def pair_to_instrument(pair6: str) -> str:
    return PAIR_MAP[pair6]

def instrument_to_symbol(instrument: str) -> str:
    return INSTRUMENT_TO_PAIR6.get(str(instrument).upper(), str(instrument).replace("_", ""))

def instrument_is_jpy(instrument: str) -> bool:
    return instrument in JPY_INSTRUMENTS

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
    return eq if eq > 0 else (nav if nav > 0 else DEFAULT_EQUITY)

def _round_down_to_pip(price: float, pip: float) -> float:
    return math.floor(price / pip) * pip

def _round_up_to_pip(price: float, pip: float) -> float:
    return math.ceil(price / pip) * pip

def infer_session_code(ts_utc: dt.datetime) -> int:
    h = ts_utc.hour
    if 0 <= h < 7:
        return 0
    if 7 <= h < 13:
        return 1
    if 13 <= h < 22:
        return 2
    return 3

def normalize_side(side: Any) -> str:
    s = str(side or "").strip().upper()
    if s in ("BUY", "LONG"):
        return "BUY"
    if s in ("SELL", "SHORT"):
        return "SELL"
    return s

def make_tracking_key(
    order_id: Optional[str],
    broker_trade_id: Optional[str],
    client_trade_id: Optional[str],
    instrument: str,
    side: str,
    ts: Optional[str],
) -> str:
    for candidate in (broker_trade_id, client_trade_id, order_id):
        if candidate not in (None, ""):
            return str(candidate)
    return f"{instrument}:{side}:{ts or utc_ts()}"

# ====================================================
# SQLITE
# ====================================================
def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            instrument TEXT,
            side TEXT,
            units_signed INTEGER,
            entry_price REAL,
            sl_price REAL,
            tp_price REAL,
            status TEXT,
            pnl REAL,
            order_id TEXT,
            reason TEXT,
            pair_score REAL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_order_id ON trade_events(order_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_status ON trade_events(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_ts ON trade_events(ts)")
    conn.commit()
    conn.close()

def insert_trade_event_db(row: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trade_events
        (ts, instrument, side, units_signed, entry_price, sl_price, tp_price, status, pnl, order_id, reason, pair_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("ts"),
        row.get("instrument"),
        row.get("side"),
        row.get("units_signed"),
        row.get("entry_price"),
        row.get("sl_price"),
        row.get("tp_price"),
        row.get("status"),
        row.get("pnl"),
        row.get("order_id"),
        row.get("reason"),
        row.get("pair_score"),
    ))
    conn.commit()
    conn.close()

def read_trade_events_db() -> pd.DataFrame:
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM trade_events ORDER BY ts DESC", conn)
        if not df.empty and "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        return df
    finally:
        conn.close()

# ====================================================
# CSV
# ====================================================
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
    insert_trade_event_db(row)

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
    db_df = read_trade_events_db()
    if not db_df.empty:
        return db_df
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

# ====================================================
# TRADE STATE
# ====================================================
def _check_daily_reset() -> None:
    global _trade_day
    today = dt.datetime.now(dt.timezone.utc).date()
    if today != _trade_day:
        _trade_day = today
        _trade_count_today.clear()

def trades_today(pair6: str) -> int:
    _check_daily_reset()
    return _trade_count_today.get(pair6, 0)

def trades_today_total() -> int:
    _check_daily_reset()
    return sum(_trade_count_today.values())

def inc_trade(pair6: str) -> None:
    _check_daily_reset()
    _trade_count_today[pair6] = _trade_count_today.get(pair6, 0) + 1

def current_open_trade_count() -> int:
    return len(_open_trade_ids)

def can_open_trade() -> bool:
    return current_open_trade_count() < MAX_OPEN_TRADES

def note_trade_opened(tracking_key: Optional[str]) -> None:
    if tracking_key:
        _open_trade_ids.add(str(tracking_key))

def note_trade_closed(tracking_key: Optional[str]) -> None:
    if tracking_key and str(tracking_key) in _open_trade_ids:
        _open_trade_ids.remove(str(tracking_key))

def tracking_keys_for_close_event(t: "TradeEvent") -> List[str]:
    keys: List[str] = []
    for candidate in (t.broker_trade_id, t.client_trade_id, t.order_id):
        if candidate not in (None, ""):
            keys.append(str(candidate))
    for tracking_key, meta in _open_trade_meta.items():
        if str(meta.get("order_id", "")) and str(meta.get("order_id")) == str(t.order_id):
            keys.append(tracking_key)
        if str(meta.get("broker_trade_id", "")) and str(meta.get("broker_trade_id")) == str(t.broker_trade_id):
            keys.append(tracking_key)
        if str(meta.get("client_trade_id", "")) and str(meta.get("client_trade_id")) == str(t.client_trade_id):
            keys.append(tracking_key)
    seen = set()
    deduped = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped

# ====================================================
# SIGNAL DUP CHECK
# ====================================================
def make_signal_fingerprint(instrument: str, side: str, bar_time: int, mid_c: float, tf: Optional[str]) -> str:
    raw = {
        "instrument": instrument,
        "side": side,
        "bar_time": int(bar_time),
        "mid_c": round(float(mid_c), instrument_precision(instrument)),
        "tf": tf or "",
    }
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()

def is_duplicate_signal(pair6: str, fingerprint: str) -> bool:
    tnow = now_unix()
    q = _recent_signals.setdefault(pair6, deque())
    while q and (tnow - q[0][0] > DUP_WINDOW_SECONDS):
        q.popleft()
    return any(fp == fingerprint for _, fp in q)

def remember_signal(pair6: str, fingerprint: str) -> None:
    _recent_signals.setdefault(pair6, deque()).append((now_unix(), fingerprint))

# ====================================================
# TECHNICALS
# ====================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_sm = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_sm
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_sm
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val, plus_di, minus_di

def update_bar_history(pair6: str, payload: Dict[str, Any]) -> pd.DataFrame:
    q = _bar_history.setdefault(pair6, deque(maxlen=BAR_HISTORY_LEN))
    row = {
        "t": safe_int(payload.get("t")),
        "time": pd.to_datetime(safe_int(payload.get("t")), unit="s", utc=True, errors="coerce"),
        "mid_o": safe_float(payload.get("mid_o")),
        "mid_h": safe_float(payload.get("mid_h")),
        "mid_l": safe_float(payload.get("mid_l")),
        "mid_c": safe_float(payload.get("mid_c")),
    }
    if q and q[-1]["t"] == row["t"]:
        q[-1] = row
    else:
        q.append(row)
    return pd.DataFrame(list(q))

def seed_history_from_csv(data_dir: str) -> None:
    root = Path(data_dir)
    if not root.exists():
        return
    for pair6 in PAIR_MAP:
        path = root / f"{pair6}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path).tail(BAR_HISTORY_LEN).copy()
            req = {"time", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"}
            if not req.issubset(df.columns):
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"])
            df["mid_o"] = (pd.to_numeric(df["bid_o"], errors="coerce") + pd.to_numeric(df["ask_o"], errors="coerce")) / 2.0
            df["mid_h"] = (pd.to_numeric(df["bid_h"], errors="coerce") + pd.to_numeric(df["ask_h"], errors="coerce")) / 2.0
            df["mid_l"] = (pd.to_numeric(df["bid_l"], errors="coerce") + pd.to_numeric(df["ask_l"], errors="coerce")) / 2.0
            df["mid_c"] = (pd.to_numeric(df["bid_c"], errors="coerce") + pd.to_numeric(df["ask_c"], errors="coerce")) / 2.0
            q = deque(maxlen=BAR_HISTORY_LEN)
            for _, r in df.iterrows():
                q.append({
                    "t": int(pd.Timestamp(r["time"]).timestamp()),
                    "time": r["time"],
                    "mid_o": float(r["mid_o"]),
                    "mid_h": float(r["mid_h"]),
                    "mid_l": float(r["mid_l"]),
                    "mid_c": float(r["mid_c"]),
                })
            _bar_history[pair6] = q
        except Exception:
            continue

# ====================================================
# FEATURES / GATES / SIZING
# ====================================================
def build_runtime_feature_row(payload: Dict[str, Any], pair6: str, instrument: str, feat_order: List[str]) -> Dict[str, Any]:
    ps = instrument_pip_size(instrument)
    hist = update_bar_history(pair6, payload)
    for c in ["mid_o", "mid_h", "mid_l", "mid_c"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    ts = pd.to_datetime(safe_int(payload.get("t")), unit="s", utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.utcnow()

    mid_o = safe_float(payload.get("mid_o"))
    mid_h = safe_float(payload.get("mid_h"))
    mid_l = safe_float(payload.get("mid_l"))
    mid_c = safe_float(payload.get("mid_c"))
    atr14 = safe_float(payload.get("atr14"))

    spread_c = safe_float(payload.get("spread_c"), np.nan)
    bid_c = safe_float(payload.get("bid_c"), np.nan)
    ask_c = safe_float(payload.get("ask_c"), np.nan)
    if not np.isfinite(spread_c) or spread_c <= 0:
        if np.isfinite(bid_c) and np.isfinite(ask_c) and ask_c >= bid_c:
            spread_c = ask_c - bid_c
        else:
            spread_c = 0.0

    spread_pips = safe_float(payload.get("spread_pips"), np.nan)
    if not np.isfinite(spread_pips) or spread_pips < 0:
        spread_pips = spread_c / ps if ps > 0 else 0.0

    spread_atr = safe_float(payload.get("spread_atr"), np.nan)
    if (not np.isfinite(spread_atr) or spread_atr < 0) and atr14 > 0:
        spread_atr = spread_c / atr14
    elif not np.isfinite(spread_atr):
        spread_atr = 0.0

    roll_high_12 = float(hist["mid_h"].tail(12).max()) if not hist.empty else mid_h
    roll_low_12 = float(hist["mid_l"].tail(12).min()) if not hist.empty else mid_l
    roll_high_24 = float(hist["mid_h"].tail(24).max()) if not hist.empty else mid_h
    roll_low_24 = float(hist["mid_l"].tail(24).min()) if not hist.empty else mid_l

    if len(hist) >= 3:
        adx_series, plus_di_series, minus_di_series = adx(hist["mid_h"], hist["mid_l"], hist["mid_c"], 14)
    else:
        adx_series = pd.Series([safe_float(payload.get("adx14"))])
        plus_di_series = pd.Series([safe_float(payload.get("plus_di14"))])
        minus_di_series = pd.Series([safe_float(payload.get("minus_di14"))])

    plus_di14 = safe_float(payload.get("plus_di14"), float(plus_di_series.iloc[-1]) if len(plus_di_series) else 0.0)
    minus_di14 = safe_float(payload.get("minus_di14"), float(minus_di_series.iloc[-1]) if len(minus_di_series) else 0.0)

    ret24 = safe_float(payload.get("ret24"), np.nan)
    if not np.isfinite(ret24):
        if len(hist) >= 25 and hist["mid_c"].iloc[-25] != 0:
            ret24 = (mid_c / float(hist["mid_c"].iloc[-25])) - 1.0
        else:
            ret24 = 0.0

    row = {
        "spread_pips": spread_pips,
        "spread_atr": spread_atr,
        "ema20": safe_float(payload.get("ema20")),
        "ema50": safe_float(payload.get("ema50")),
        "ema200": safe_float(payload.get("ema200")),
        "rsi14": safe_float(payload.get("rsi14")),
        "macdh": safe_float(payload.get("macdh")),
        "adx14": safe_float(payload.get("adx14"), float(adx_series.iloc[-1]) if len(adx_series) else 0.0),
        "plus_di14": plus_di14,
        "minus_di14": minus_di14,
        "atr14": atr14,
        "atr_pct": atr14 / mid_c if mid_c > 0 else 0.0,
        "bbw": safe_float(payload.get("bbw")),
        "ret1": safe_float(payload.get("ret1")),
        "ret3": safe_float(payload.get("ret3")),
        "ret6": safe_float(payload.get("ret6")),
        "ret12": safe_float(payload.get("ret12")),
        "ret24": ret24,
        "d20": safe_float(payload.get("d20")),
        "d50": safe_float(payload.get("d50")),
        "d200": safe_float(payload.get("d200")),
        "s20": safe_float(payload.get("s20")),
        "s50": safe_float(payload.get("s50")),
        "s200": safe_float(payload.get("s200")),
        "dist_high_12": (roll_high_12 - mid_c) / ps if ps > 0 else 0.0,
        "dist_low_12": (mid_c - roll_low_12) / ps if ps > 0 else 0.0,
        "dist_high_24": (roll_high_24 - mid_c) / ps if ps > 0 else 0.0,
        "dist_low_24": (mid_c - roll_low_24) / ps if ps > 0 else 0.0,
        "range_pips": (mid_h - mid_l) / ps if ps > 0 else 0.0,
        "body_pips": (mid_c - mid_o) / ps if ps > 0 else 0.0,
        "upper_wick_pips": (mid_h - max(mid_o, mid_c)) / ps if ps > 0 else 0.0,
        "lower_wick_pips": (min(mid_o, mid_c) - mid_l) / ps if ps > 0 else 0.0,
        "hour": safe_float(payload.get("hour"), safe_float(payload.get("hr"), float(ts.hour))),
        "dow": safe_float(payload.get("dow"), float(ts.dayofweek)),
        "month": safe_float(payload.get("month"), float(ts.month)),
        "session": safe_float(payload.get("session"), float(infer_session_code(ts.to_pydatetime()))),
        "ema50_h4": safe_float(payload.get("ema50_h4"), safe_float(payload.get("ema50"))),
        "adx14_h4": safe_float(payload.get("adx14_h4"), safe_float(payload.get("adx14"))),
        "ema20_d1": safe_float(payload.get("ema20_d1"), safe_float(payload.get("ema20"))),
        "rsi14_d1": safe_float(payload.get("rsi14_d1"), safe_float(payload.get("rsi14"))),
        "trend_regime": safe_float(payload.get("trend_regime")),
        "vol_regime": safe_float(payload.get("vol_regime")),
        "mid_o": mid_o,
        "mid_h": mid_h,
        "mid_l": mid_l,
        "mid_c": mid_c,
        "spread_c": spread_c,
        "hr": safe_float(payload.get("hr"), float(ts.hour)),
        "t": safe_int(payload.get("t")),
    }

    out: Dict[str, Any] = {}
    for f in feat_order:
        out[f] = row.get(f, safe_float(payload.get(f), 0.0))
    return out

def payload_sanity_checks(payload: Dict[str, Any], instrument: str) -> Optional[str]:
    spread_pips = safe_float(payload.get("spread_pips"), np.nan)
    if np.isfinite(spread_pips) and spread_pips > MAX_SPREAD_PIPS:
        return f"Spread too high: {spread_pips} pips > {MAX_SPREAD_PIPS}"

    atr14 = safe_float(payload.get("atr14"), 0.0)
    if atr14 < min_atr_for_instrument(instrument):
        return f"ATR too small: {atr14}"

    mid_l = safe_float(payload.get("mid_l"))
    mid_c = safe_float(payload.get("mid_c"))
    mid_h = safe_float(payload.get("mid_h"))
    mid_o = safe_float(payload.get("mid_o"))

    if not (mid_l <= mid_c <= mid_h):
        return "Bad payload: mid_c not between mid_l and mid_h"
    if not (mid_l <= mid_o <= mid_h):
        return "Bad payload: mid_o not between mid_l and mid_h"
    if mid_h < mid_l:
        return "Bad payload: mid_h < mid_l"
    if safe_float(payload.get("spread_pips"), 0.0) < 0:
        return "Bad payload: negative spread_pips"
    if safe_float(payload.get("spread_atr"), 0.0) < 0:
        return "Bad payload: negative spread_atr"
    return None

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
    return {
        "n": n,
        "win_rate": (wins / n if n else None),
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
    return max(0.0, min(1.0, (AUC_WEIGHT * auc_norm) + (WINRATE_WEIGHT * wr)))

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
        base = int((risk_cap / risk_per_1000) * 1000) if risk_per_1000 > 0 else base

    if pair_score >= 0.80:
        base = int(base * 1.35)
    elif pair_score >= 0.65:
        base = int(base * 1.15)
    elif pair_score < 0.50:
        base = int(base * 0.70)

    if avg_auc >= 0.57:
        base = int(base * 1.10)
    elif avg_auc < 0.54:
        base = int(base * 0.90)

    return min(max_units_for_instrument(instrument), max(min_units_for_instrument(instrument), base))

def compute_sl_tp_prices(
    side: str,
    mid_c: float,
    atr14: float,
    instrument: str,
    sl_atr: float,
    tp_atr: float,
    min_dist_pips: float = 5.0,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if side not in ("BUY", "SELL"):
        return (None, None, None, None)

    pip = instrument_pip_size(instrument)
    atrv = max(float(atr14), pip)
    sl_dist = max(sl_atr * atrv, min_dist_pips * pip)
    tp_dist = max(tp_atr * atrv, min_dist_pips * pip)

    if side == "BUY":
        sl_price = _round_down_to_pip(mid_c - sl_dist, pip)
        tp_price = _round_up_to_pip(mid_c + tp_dist, pip)
        if sl_price >= mid_c:
            sl_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)
        if tp_price <= mid_c:
            tp_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
    else:
        sl_price = _round_up_to_pip(mid_c + sl_dist, pip)
        tp_price = _round_down_to_pip(mid_c - tp_dist, pip)
        if sl_price <= mid_c:
            sl_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
        if tp_price >= mid_c:
            tp_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)

    sl_str = format_oanda_price(sl_price, instrument)
    tp_str = format_oanda_price(tp_price, instrument)
    mid_str = format_oanda_price(mid_c, instrument)

    sl_price_f = float(sl_str)
    tp_price_f = float(tp_str)
    mid_c_f = float(mid_str)

    sl_pips = abs(mid_c_f - sl_price_f) / pip
    tp_pips = abs(tp_price_f - mid_c_f) / pip
    return float(sl_pips), float(tp_pips), sl_str, tp_str


# ====================================================
# H1 NEWS / NOISE / DIRECTION / AI HELPERS
# ====================================================
def parse_utc_datetime(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
        txt = str(value).strip()
        if txt.isdigit():
            return dt.datetime.fromtimestamp(float(txt), tz=dt.timezone.utc)
        txt = txt.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(txt)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None


def pair_currencies(pair6: str) -> List[str]:
    pair6 = str(pair6 or "").upper().replace("_", "")
    return [pair6[:3], pair6[3:]] if len(pair6) == 6 else []


def normalize_news_currency(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    raw = value if isinstance(value, list) else str(value).replace("/", ",").replace("|", ",").split(",")
    return [str(x).strip().upper() for x in raw if str(x).strip()]


def normalize_impact(value: Any) -> str:
    impact = str(value or "").strip().upper()
    if impact in ("3", "HIGH", "RED", "H"):
        return "HIGH"
    if impact in ("2", "MED", "MEDIUM", "ORANGE", "M"):
        return "MEDIUM"
    if impact in ("1", "LOW", "YELLOW", "L"):
        return "LOW"
    return impact or "UNKNOWN"


def event_impact_is_blocked(impact: str) -> bool:
    impact = normalize_impact(impact)
    return impact in NEWS_BLOCK_IMPACTS or (impact == "HIGH" and "RED" in NEWS_BLOCK_IMPACTS)


def normalize_news_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or raw.get("event") or raw.get("name") or NEWS_DEFAULT_TITLE).strip()
    impact = normalize_impact(raw.get("impact") or raw.get("importance") or raw.get("level") or "HIGH")
    currencies = normalize_news_currency(raw.get("currency") or raw.get("currencies") or raw.get("ccy"))
    start = parse_utc_datetime(raw.get("start") or raw.get("start_utc") or raw.get("blackout_start"))
    end = parse_utc_datetime(raw.get("end") or raw.get("end_utc") or raw.get("blackout_end"))
    event_time = parse_utc_datetime(raw.get("time") or raw.get("time_utc") or raw.get("datetime") or raw.get("timestamp"))
    before_min = safe_int(raw.get("before_min"), NEWS_BLOCK_BEFORE_MIN)
    after_min = safe_int(raw.get("after_min"), NEWS_BLOCK_AFTER_MIN)
    if event_time and not start:
        start = event_time - dt.timedelta(minutes=before_min)
    if event_time and not end:
        end = event_time + dt.timedelta(minutes=after_min)
    if start and not end:
        end = start + dt.timedelta(minutes=before_min + after_min)
    if end and not start:
        start = end - dt.timedelta(minutes=before_min + after_min)
    if not start or not end:
        return None
    if end < start:
        start, end = end, start
    if not currencies and NEWS_BLOCK_UNKNOWN_CURRENCY:
        currencies = ["ALL"]
    return {
        "title": title,
        "impact": impact,
        "currencies": currencies,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "time_utc": event_time.isoformat() if event_time else None,
        "before_min": before_min,
        "after_min": after_min,
        "source": str(raw.get("source") or "manual"),
    }


def parse_manual_blackout_events() -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not NEWS_MANUAL_BLACKOUT_UTC:
        return events
    for chunk in NEWS_MANUAL_BLACKOUT_UTC.split(";"):
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) < 3:
            continue
        ev = normalize_news_event({
            "start": parts[0],
            "end": parts[1],
            "currency": parts[2],
            "impact": parts[3] if len(parts) >= 4 else "HIGH",
            "title": parts[4] if len(parts) >= 5 else NEWS_DEFAULT_TITLE,
            "source": "manual_env_blackout",
        })
        if ev:
            events.append(ev)
    return events


def load_news_events() -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if NEWS_EVENTS_JSON:
        try:
            parsed = json.loads(NEWS_EVENTS_JSON)
            if isinstance(parsed, dict):
                parsed = parsed.get("events", [])
            for raw in parsed if isinstance(parsed, list) else []:
                ev = normalize_news_event(raw)
                if ev:
                    events.append(ev)
        except Exception as e:
            print(f"WARNING: failed to parse NEWS_EVENTS_JSON: {e}")
    events.extend(parse_manual_blackout_events())
    file_path = Path(NEWS_EVENTS_FILE)
    if file_path.exists():
        try:
            parsed = json.loads(file_path.read_text())
            if isinstance(parsed, dict):
                parsed = parsed.get("events", [])
            for raw in parsed if isinstance(parsed, list) else []:
                ev = normalize_news_event(raw)
                if ev:
                    events.append(ev)
        except Exception as e:
            print(f"WARNING: failed to parse NEWS_EVENTS_FILE={NEWS_EVENTS_FILE}: {e}")
    cutoff = now_utc() - dt.timedelta(hours=NEWS_KEEP_PAST_HOURS)
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for ev in events:
        end = parse_utc_datetime(ev.get("end_utc"))
        if end and end < cutoff:
            continue
        key = (ev.get("title"), tuple(ev.get("currencies") or []), ev.get("start_utc"), ev.get("end_utc"))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(ev)
    return cleaned


def save_news_events_to_file(events: List[Dict[str, Any]]) -> None:
    try:
        file_path = Path(NEWS_EVENTS_FILE)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({"events": events}, indent=2))
    except Exception as e:
        print(f"WARNING: failed to save news events to {NEWS_EVENTS_FILE}: {e}")


def runtime_news_filter(pair6: str, payload: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not NEWS_FILTER_ENABLED:
        return True, "news_filter_disabled", {"news_filter_enabled": False, "news_filter_passed": True}
    ts = parse_utc_datetime(payload.get("t") or payload.get("bar_time") or payload.get("ts")) or now_utc()
    pair_ccys = pair_currencies(pair6)
    relevant = set(pair_ccys)
    metrics: Dict[str, Any] = {
        "news_filter_enabled": True,
        "news_filter_passed": True,
        "news_pair_currencies": ",".join(pair_ccys),
        "news_events_loaded": len(NEWS_EVENTS),
    }
    nearest = None
    nearest_delta = None
    for ev in NEWS_EVENTS:
        impact = normalize_impact(ev.get("impact"))
        if not event_impact_is_blocked(impact):
            continue
        event_ccys = set(normalize_news_currency(ev.get("currencies")))
        currency_matches = NEWS_BLOCK_ALL_CURRENCIES or "ALL" in event_ccys or bool(relevant.intersection(event_ccys))
        if not currency_matches:
            continue
        start = parse_utc_datetime(ev.get("start_utc"))
        end = parse_utc_datetime(ev.get("end_utc"))
        event_time = parse_utc_datetime(ev.get("time_utc"))
        if not start or not end:
            continue
        if event_time:
            delta = abs((event_time - ts).total_seconds()) / 60.0
            if nearest_delta is None or delta < nearest_delta:
                nearest_delta = delta
                nearest = ev
        if start <= ts <= end:
            metrics.update({
                "news_filter_passed": False,
                "news_block_title": ev.get("title"),
                "news_block_impact": impact,
                "news_block_currencies": ",".join(ev.get("currencies") or []),
                "news_blackout_start_utc": start.isoformat(),
                "news_blackout_end_utc": end.isoformat(),
                "news_event_time_utc": event_time.isoformat() if event_time else None,
            })
            return False, f"News filter blocked: {impact} {','.join(ev.get('currencies') or [])} {ev.get('title')} blackout {start.isoformat()} to {end.isoformat()}", metrics
    if nearest:
        metrics.update({
            "nearest_news_title": nearest.get("title"),
            "nearest_news_impact": normalize_impact(nearest.get("impact")),
            "nearest_news_currencies": ",".join(nearest.get("currencies") or []),
            "nearest_news_minutes": round(float(nearest_delta or 0.0), 2),
        })
    return True, "news_filter_passed", metrics


def signal_staleness_guard(payload: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not SIGNAL_STALENESS_GUARD_ENABLED:
        return True, "signal_staleness_guard_disabled", {"signal_staleness_guard_enabled": False}
    signal_time = parse_utc_datetime(payload.get("t") or payload.get("bar_time") or payload.get("ts"))
    if signal_time is None:
        return False, "Signal staleness guard blocked: missing_or_invalid_signal_time", {"signal_staleness_guard_enabled": True, "signal_staleness_guard_passed": False}
    age_seconds = max(0.0, (now_utc() - signal_time).total_seconds())
    passed = age_seconds <= SIGNAL_MAX_AGE_SECONDS
    metrics = {
        "signal_staleness_guard_enabled": True,
        "signal_staleness_guard_passed": passed,
        "signal_time_utc": signal_time.isoformat(),
        "signal_age_seconds": round(float(age_seconds), 2),
        "signal_max_age_seconds": SIGNAL_MAX_AGE_SECONDS,
    }
    if not passed:
        return False, f"Signal staleness guard blocked: signal too old {age_seconds:.1f}s > {SIGNAL_MAX_AGE_SECONDS}s", metrics
    return True, "signal_staleness_guard_passed", metrics


def runtime_noise_filter(payload: Dict[str, Any], feature_row: Dict[str, Any], instrument: str, side: str) -> Tuple[bool, str, Dict[str, Any]]:
    if not NOISE_FILTER_ENABLED:
        return True, "noise_filter_disabled", {"noise_filter_enabled": False, "noise_filter_passed": True}
    pip = instrument_pip_size(instrument)
    mid_o = safe_float(payload.get("mid_o"), 0.0)
    mid_h = safe_float(payload.get("mid_h"), 0.0)
    mid_l = safe_float(payload.get("mid_l"), 0.0)
    mid_c = safe_float(payload.get("mid_c"), 0.0)
    range_pips = abs(mid_h - mid_l) / pip if pip > 0 else 0.0
    body_pips = abs(mid_c - mid_o) / pip if pip > 0 else 0.0
    upper_wick_pips = max(0.0, (mid_h - max(mid_o, mid_c)) / pip) if pip > 0 else 0.0
    lower_wick_pips = max(0.0, (min(mid_o, mid_c) - mid_l) / pip) if pip > 0 else 0.0
    wick_total_pips = upper_wick_pips + lower_wick_pips
    atr14 = safe_float(payload.get("atr14"), safe_float(feature_row.get("atr14"), 0.0))
    atr_pips = atr14 / pip if atr14 > 0 and pip > 0 else 0.0
    spread_pips = safe_float(payload.get("spread_pips"), safe_float(feature_row.get("spread_pips"), 0.0))
    body_range_ratio = body_pips / range_pips if range_pips > 0 else 0.0
    range_atr_ratio = range_pips / atr_pips if atr_pips > 0 else 0.0
    spread_range_ratio = spread_pips / range_pips if range_pips > 0 else 999.0
    wick_body_ratio = wick_total_pips / max(body_pips, 0.1)
    ema20_val = safe_float(payload.get("ema20"), 0.0)
    ema50_val = safe_float(payload.get("ema50"), 0.0)
    momentum = safe_float(feature_row.get("ret1"), 0.0) + safe_float(feature_row.get("ret3"), 0.0) + safe_float(feature_row.get("ret6"), 0.0)
    trend_aligned = True
    if NOISE_FILTER_REQUIRE_TREND_ALIGNMENT:
        if side == "BUY":
            trend_aligned = momentum >= 0 or (mid_c >= ema20_val if ema20_val > 0 else True)
        elif side == "SELL":
            trend_aligned = momentum <= 0 or (mid_c <= ema20_val if ema20_val > 0 else True)
    metrics = {
        "noise_filter_enabled": True,
        "noise_filter_passed": True,
        "range_pips": round(float(range_pips), 4),
        "body_pips": round(float(body_pips), 4),
        "atr_pips": round(float(atr_pips), 4),
        "spread_pips_runtime": round(float(spread_pips), 4),
        "body_range_ratio": round(float(body_range_ratio), 4),
        "range_atr_ratio": round(float(range_atr_ratio), 4),
        "spread_range_ratio": round(float(spread_range_ratio), 4),
        "wick_body_ratio": round(float(wick_body_ratio), 4),
        "momentum_sum": round(float(momentum), 8),
        "trend_aligned": bool(trend_aligned),
    }
    reasons = []
    if range_pips < MIN_NOISE_RANGE_PIPS:
        reasons.append(f"range_too_small:{range_pips:.2f}<{MIN_NOISE_RANGE_PIPS:.2f}pips")
    if atr_pips > 0 and atr_pips < MIN_NOISE_ATR_PIPS:
        reasons.append(f"atr_too_small:{atr_pips:.2f}<{MIN_NOISE_ATR_PIPS:.2f}pips")
    if atr_pips > 0 and range_atr_ratio < MIN_RANGE_ATR_RATIO:
        reasons.append(f"range_vs_atr_too_small:{range_atr_ratio:.2f}<{MIN_RANGE_ATR_RATIO:.2f}")
    if range_pips > 0 and spread_range_ratio > MAX_SPREAD_RANGE_RATIO:
        reasons.append(f"spread_too_large_vs_range:{spread_range_ratio:.2f}>{MAX_SPREAD_RANGE_RATIO:.2f}")
    if body_range_ratio < MIN_BODY_RANGE_RATIO and wick_body_ratio > MAX_WICK_BODY_RATIO:
        reasons.append(f"doji_wick_noise:body_ratio={body_range_ratio:.2f},wick_body={wick_body_ratio:.2f}")
    if NOISE_FILTER_REQUIRE_TREND_ALIGNMENT and not trend_aligned:
        reasons.append("trend_not_aligned_with_model_side")
    if reasons:
        metrics["noise_filter_passed"] = False
        return False, "Noise filter blocked: " + "; ".join(reasons), metrics
    return True, "noise_filter_passed", metrics


def _sign_for_direction(value: float, eps: float = 0.0) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def direction_consensus_guard(payload: Dict[str, Any], feature_row: Dict[str, Any], instrument: str, side: str) -> Tuple[bool, str, Dict[str, Any]]:
    if not DIRECTION_CONFIRMATION_ENABLED:
        return True, "direction_confirmation_disabled", {"direction_confirmation_enabled": False}
    side_sign = 1 if side == "BUY" else -1 if side == "SELL" else 0
    if side_sign == 0:
        return False, "Direction confirmation blocked: invalid side", {"direction_confirmation_enabled": True, "direction_confirmation_passed": False}
    pip = instrument_pip_size(instrument)
    mid_o = safe_float(payload.get("mid_o"), 0.0)
    mid_c = safe_float(payload.get("mid_c"), 0.0)
    ema20 = safe_float(payload.get("ema20"), 0.0)
    ema50 = safe_float(payload.get("ema50"), 0.0)
    ema200 = safe_float(payload.get("ema200"), 0.0)
    rsi14 = safe_float(payload.get("rsi14"), safe_float(feature_row.get("rsi14"), 50.0))
    macdh = safe_float(payload.get("macdh"), safe_float(feature_row.get("macdh"), 0.0))
    ret1 = safe_float(feature_row.get("ret1"), safe_float(payload.get("ret1"), 0.0))
    ret3 = safe_float(feature_row.get("ret3"), safe_float(payload.get("ret3"), 0.0))
    ret6 = safe_float(feature_row.get("ret6"), safe_float(payload.get("ret6"), 0.0))
    body_pips_signed = ((mid_c - mid_o) / pip) if pip > 0 else 0.0
    score = 0
    aligned, conflicts, required_failures = [], [], []
    candle_sign = _sign_for_direction(body_pips_signed, DIRECTION_CONFIRM_MIN_BODY_PIPS)
    if candle_sign == side_sign:
        score += 1; aligned.append("candle_body")
    elif candle_sign == -side_sign:
        conflicts.append("candle_body")
        rng = abs(safe_float(payload.get("mid_h"), 0.0) - safe_float(payload.get("mid_l"), 0.0)) / pip if pip > 0 else 0.0
        body_ratio = abs(body_pips_signed) / rng if rng > 0 else 0.0
        if DIRECTION_CONFIRM_BLOCK_STRONG_OPPOSITE_CANDLE and body_ratio >= DIRECTION_CONFIRM_STRONG_BODY_RATIO:
            required_failures.append("strong_opposite_candle")
    for label, val in [("ret1", ret1), ("ret3", ret3), ("ret6", ret6), ("macdh", macdh)]:
        sig = _sign_for_direction(val, 0.0)
        if sig == side_sign:
            score += 1; aligned.append(label)
        elif sig == -side_sign:
            conflicts.append(label)
    ema_buffer = DIRECTION_CONFIRM_EMA_BUFFER_PIPS * pip
    for label, ema_val, reject in [("ema20", ema20, False), ("ema50", ema50, DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA50), ("ema200", ema200, DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA200)]:
        if ema_val <= 0 or mid_c <= 0:
            continue
        sig = _sign_for_direction(mid_c - ema_val, ema_buffer)
        if sig == side_sign:
            score += 1; aligned.append(label)
        elif sig == -side_sign:
            conflicts.append(label)
            if reject:
                required_failures.append(f"{label}_against_side")
    if side == "BUY" and rsi14 >= 50:
        score += 1; aligned.append("rsi_ge_50")
    elif side == "SELL" and rsi14 <= 50:
        score += 1; aligned.append("rsi_le_50")
    else:
        conflicts.append("rsi_side")
    passed = score >= DIRECTION_CONFIRM_MIN_SCORE and not required_failures
    metrics = {
        "direction_confirmation_enabled": True,
        "direction_confirmation_passed": passed,
        "direction_confirmation_required": DIRECTION_CONFIRMATION_REQUIRED,
        "direction_confirmation_score": int(score),
        "direction_confirmation_min_score": DIRECTION_CONFIRM_MIN_SCORE,
        "direction_confirmation_aligned": aligned,
        "direction_confirmation_conflicts": conflicts,
        "direction_confirmation_required_failures": required_failures,
    }
    if not passed:
        reason = f"Direction confirmation blocked: side={side}, score={score}/{DIRECTION_CONFIRM_MIN_SCORE}, conflicts={','.join(conflicts) or 'none'}, required_failures={','.join(required_failures) or 'none'}"
        if DIRECTION_CONFIRMATION_REQUIRED:
            return False, reason, metrics
        return True, "Direction confirmation optional: " + reason, metrics
    return True, f"direction_confirmation_passed:score={score}/{DIRECTION_CONFIRM_MIN_SCORE}", metrics


def fetch_live_oanda_quote(instrument: str) -> Dict[str, Any]:
    if not broker_can_close():
        return {"ok": False, "error": "Missing OANDA env vars"}
    result = oanda_request("GET", f"/v3/accounts/{OANDA_ACCOUNT_ID}/pricing?instruments={instrument}", timeout=15)
    if not result.get("ok"):
        return result
    data = result.get("data") or {}
    prices = data.get("prices") or []
    if not prices:
        return {"ok": False, "error": "No live prices returned", "data": data}
    price = prices[0]
    bids = price.get("bids") or []
    asks = price.get("asks") or []
    if not bids or not asks:
        return {"ok": False, "error": "Live price missing bid/ask", "data": price}
    bid = safe_float(bids[0].get("price"), np.nan)
    ask = safe_float(asks[0].get("price"), np.nan)
    quote_time = parse_utc_datetime(price.get("time"))
    age_seconds = max(0.0, (now_utc() - quote_time).total_seconds()) if quote_time else None
    if not np.isfinite(bid) or not np.isfinite(ask) or ask < bid:
        return {"ok": False, "error": "Invalid live bid/ask", "data": price}
    if age_seconds is not None and age_seconds > LIVE_PRICE_MAX_AGE_SECONDS:
        return {"ok": False, "error": f"Live quote too old: {age_seconds:.2f}s", "data": price}
    return {"ok": True, "instrument": instrument, "bid": bid, "ask": ask, "time": price.get("time"), "quote_age_seconds": age_seconds, "raw": price}


def entry_reversal_guard(payload: Dict[str, Any], feature_row: Dict[str, Any], instrument: str, side: str) -> Tuple[bool, str, Dict[str, Any]]:
    if not ENTRY_REVERSAL_GUARD_ENABLED:
        return True, "entry_reversal_guard_disabled", {"entry_reversal_guard_enabled": False}
    quote = fetch_live_oanda_quote(instrument)
    if not quote.get("ok"):
        metrics = {"entry_reversal_guard_enabled": True, "entry_reversal_guard_passed": not ENTRY_REVERSAL_GUARD_REQUIRED, "entry_reversal_quote_error": quote.get("error")}
        if ENTRY_REVERSAL_GUARD_REQUIRED:
            return False, f"Entry reversal guard blocked: live quote unavailable: {quote.get('error')}", metrics
        return True, f"Entry reversal guard optional: live quote unavailable: {quote.get('error')}", metrics
    pip = instrument_pip_size(instrument)
    bid = safe_float(quote.get("bid"), 0.0)
    ask = safe_float(quote.get("ask"), 0.0)
    live_mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    alert_mid = safe_float(payload.get("mid_c"), 0.0)
    spread_pips = (ask - bid) / pip if pip > 0 and ask >= bid else 999.0
    if side == "BUY":
        adverse_pips = (alert_mid - live_mid) / pip if alert_mid > 0 and live_mid > 0 else 0.0
    elif side == "SELL":
        adverse_pips = (live_mid - alert_mid) / pip if alert_mid > 0 and live_mid > 0 else 0.0
    else:
        adverse_pips = 0.0
    reasons = []
    if spread_pips > ENTRY_REVERSAL_MAX_SPREAD_PIPS:
        reasons.append(f"spread_too_high_for_entry:{spread_pips:.2f}>{ENTRY_REVERSAL_MAX_SPREAD_PIPS:.2f}")
    if adverse_pips >= ENTRY_REVERSAL_MAX_ADVERSE_PIPS:
        reasons.append(f"live_price_reversed_against_{side.lower()}:{adverse_pips:.2f}>={ENTRY_REVERSAL_MAX_ADVERSE_PIPS:.2f}pips")
    metrics = {
        "entry_reversal_guard_enabled": True,
        "entry_reversal_guard_passed": len(reasons) == 0,
        "entry_reversal_live_bid": bid,
        "entry_reversal_live_ask": ask,
        "entry_reversal_live_mid": live_mid,
        "entry_reversal_alert_mid": alert_mid,
        "entry_reversal_adverse_pips": round(float(adverse_pips), 4),
        "entry_reversal_spread_pips": round(float(spread_pips), 4),
        "entry_reversal_quote_age_seconds": quote.get("quote_age_seconds"),
    }
    if reasons:
        return False, "Entry reversal guard blocked: " + "; ".join(reasons), metrics
    return True, "entry_reversal_guard_passed", metrics



# ====================================================
# SIDE-AWARE AI REVIEWER HELPERS — H1 PATCH FROM M15 v16
# ====================================================
def _parse_json_object_from_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def ai_side_norm(side: Any) -> str:
    side_text = str(side or "").strip().upper()
    if side_text in {"BUY", "LONG", "BULL", "BULLISH"}:
        return "BUY"
    if side_text in {"SELL", "SHORT", "BEAR", "BEARISH"}:
        return "SELL"
    return side_text


def ai_trend_norm(trend: Any) -> str:
    text = str(trend or "").strip().lower()
    if text in {"bull", "bullish", "up", "uptrend", "buy", "long"}:
        return "bullish"
    if text in {"bear", "bearish", "down", "downtrend", "sell", "short"}:
        return "bearish"
    return "neutral"


def ai_trend_supports_side(side: Any, trend: Any) -> bool:
    side_text = ai_side_norm(side)
    trend_text = ai_trend_norm(trend)
    return (side_text == "BUY" and trend_text == "bullish") or (side_text == "SELL" and trend_text == "bearish")


def ai_trend_conflicts_with_side(side: Any, trend: Any) -> bool:
    side_text = ai_side_norm(side)
    trend_text = ai_trend_norm(trend)
    return (side_text == "BUY" and trend_text == "bearish") or (side_text == "SELL" and trend_text == "bullish")


def ai_direction_score(value: Any, eps: float = 0.0) -> int:
    value_float = safe_float(value, 0.0)
    if value_float > eps:
        return 1
    if value_float < -eps:
        return -1
    return 0


def ai_tf_trend_from_summary(summary: Dict[str, Any]) -> str:
    """Infer independent bullish/bearish trend from a market-context timeframe summary.

    This is intentionally independent of the alert side. It fixes the old behavior
    where bearish indicators could be treated as bad even for a SELL alert.
    """
    if not isinstance(summary, dict) or not summary.get("ok"):
        return "neutral"

    score = 0
    for key in ("ret1", "ret3", "ret5", "ema20_dist", "ema50_dist", "ema200_dist", "macd_hist", "body_pips_signed"):
        score += ai_direction_score(summary.get(key), 0.0)

    rsi14 = safe_float(summary.get("rsi14"), 50.0)
    if rsi14 >= 52:
        score += 1
    elif rsi14 <= 48:
        score -= 1

    price_vs_ema50 = ai_trend_norm(summary.get("price_vs_ema50"))
    price_vs_ema200 = ai_trend_norm(summary.get("price_vs_ema200"))
    if price_vs_ema50 == "bullish":
        score += 1
    elif price_vs_ema50 == "bearish":
        score -= 1
    if price_vs_ema200 == "bullish":
        score += 1
    elif price_vs_ema200 == "bearish":
        score -= 1

    if score >= 3:
        return "bullish"
    if score <= -3:
        return "bearish"
    return "neutral"


def ai_pattern_bias_from_context(context: Dict[str, Any]) -> str:
    candle_ctx = context.get("latest_candlestick_context") or {}
    if isinstance(candle_ctx, dict):
        return ai_side_norm(candle_ctx.get("candle_bias") or context.get("latest_h1_candle_bias") or context.get("latest_m15_candle_bias"))
    return ai_side_norm(context.get("latest_h1_candle_bias") or context.get("latest_m15_candle_bias"))


def ai_side_aware_rule_review(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic side-aware review adapted for H1.

    Key fix inherited from M15 v16:
    - bearish trend supports SELL
    - bullish trend supports BUY
    """
    side = ai_side_norm(context.get("side"))
    prob = safe_float(context.get("model_approval_probability"), safe_float(context.get("confidence"), 0.0))
    market_context = context.get("external_market_context") or context.get("market_context") or {}
    summaries = market_context.get("summaries") or {}
    risk_context = context.get("risk_context") or context.get("guards") or {}
    model_features = context.get("model_features") or context.get("features") or {}

    risk = 0
    supports: List[str] = []
    conflicts: List[str] = []
    reasons: List[str] = []

    # 1. Model probability
    if prob < 0.40:
        risk += 30
        reasons.append(f"model_probability_very_low:{prob:.3f}")
    elif prob < AI_REVIEW_MIN_MODEL_PROB:
        risk += 20
        reasons.append(f"model_probability_low:{prob:.3f}<{AI_REVIEW_MIN_MODEL_PROB:.3f}")
    elif prob >= AI_REVIEW_STRONG_MODEL_PROB:
        risk -= 10
        supports.append(f"model_probability_strong:{prob:.3f}")
    else:
        supports.append(f"model_probability_acceptable:{prob:.3f}")

    # 2. H1-style multi-timeframe trend alignment. Use available contexts.
    tf_order = [tf for tf in ("H1", "H4", "D", "M15") if tf in summaries]
    if not tf_order:
        tf_order = list(summaries.keys())[:4]

    trends: Dict[str, str] = {}
    aligned_count = 0
    conflict_count = 0
    for tf in tf_order:
        trend = ai_tf_trend_from_summary((summaries or {}).get(tf, {}))
        trends[tf] = trend
        if trend == "neutral":
            continue
        if ai_trend_supports_side(side, trend):
            aligned_count += 1
            supports.append(f"{tf}_trend_supports_{side.lower()}:{trend}")
        elif ai_trend_conflicts_with_side(side, trend):
            conflict_count += 1
            conflicts.append(f"{tf}_trend_conflicts_with_{side.lower()}:{trend}")

    if conflict_count >= 3:
        risk += 35
        reasons.append("all_timeframes_conflict")
    elif conflict_count == 2:
        risk += 25
        reasons.append("two_timeframes_conflict")
    elif conflict_count == 1:
        risk += 10
        reasons.append("one_timeframe_conflict")

    if aligned_count >= 3:
        risk -= 20
        supports.append("all_timeframes_aligned")
    elif aligned_count == 2:
        risk -= 12
        supports.append("two_timeframes_aligned")
    elif aligned_count == 1:
        risk -= 5
        supports.append("one_timeframe_aligned")

    # 3. Latest candle/pattern side-awareness, when available.
    pattern_bias = ai_pattern_bias_from_context(context)
    candle_ctx = context.get("latest_candlestick_context") or {}
    pattern_name = str(candle_ctx.get("pattern") or context.get("latest_h1_candle_pattern") or context.get("latest_m15_candle_pattern") or "unknown")
    if pattern_bias in {"BUY", "SELL"}:
        if pattern_bias == side:
            risk -= 8
            supports.append(f"latest_candle_supports_{side.lower()}:{pattern_name}")
        else:
            risk += 15
            conflicts.append(f"latest_candle_conflicts_with_{side.lower()}:{pattern_name}:{pattern_bias}")
            reasons.append("latest_candle_conflicts_with_side")
    elif pattern_name.lower() in {"doji", "inside_bar"}:
        risk += 8
        reasons.append(f"indecision_candle:{pattern_name}")

    # 4. Existing guard context
    if risk_context.get("direction_confirmation_passed") is False:
        risk += 20
        reasons.append("direction_confirmation_failed")
    if risk_context.get("live_quote_guard_passed") is False or risk_context.get("entry_reversal_guard_passed") is False:
        risk += 15
        reasons.append("live_quote_or_entry_reversal_guard_failed")
    if market_context.get("higher_timeframe_conflict") and conflict_count > aligned_count:
        risk += 10
        reasons.append("higher_timeframe_conflict")
    if risk_context.get("noise_filter_passed") is False:
        risk += 15
        reasons.append("noise_filter_failed")
    if risk_context.get("news_filter_passed") is False:
        risk += 30
        reasons.append("news_filter_failed")

    # 4B. Technical review should complement AI, not compete with it.
    # The deterministic technical layer runs first and is passed here as structured context.
    # AI/rules add risk when the technical score is weak and reduce risk when H1/H4/D agree.
    tech = risk_context.get("technical_review") or context.get("technical_review") or {}
    if isinstance(tech, dict) and tech.get("enabled") is not False:
        tech_decision = str(tech.get("decision") or "").upper()
        tech_score = safe_float(tech.get("technical_score"), 0.0)
        tech_min_score = safe_float(tech.get("minimum_required_score"), 0.0)
        tech_strong_score = safe_float(tech.get("strong_score"), 68.0)
        tech_aligned = safe_int(tech.get("aligned_timeframes"), 0)
        tech_conflicting = safe_int(tech.get("conflicting_timeframes"), 0)
        tech_hard_failures = tech.get("hard_failures") or []
        if tech_decision == "BLOCK" or bool(tech_hard_failures):
            add = 30 if bool(tech_hard_failures) else 20
            risk += add
            reasons.append(f"technical_review_blocked:score={tech_score:.2f},hard_failures={','.join(map(str, tech_hard_failures)) or 'none'}")
            conflicts.append("technical_review_block")
        elif tech_score >= tech_strong_score and tech_aligned >= 2 and tech_conflicting == 0:
            risk -= 12
            supports.append(f"technical_review_strong:score={tech_score:.2f},aligned={tech_aligned}")
        elif tech_score >= tech_min_score and tech_aligned >= 1:
            risk -= 6
            supports.append(f"technical_review_passed:score={tech_score:.2f},aligned={tech_aligned}")
        elif tech_min_score > 0 and tech_score < tech_min_score:
            risk += 12
            reasons.append(f"technical_review_weak:score={tech_score:.2f}<{tech_min_score:.2f}")
        if tech_conflicting >= 2:
            risk += 12
            reasons.append(f"technical_review_multiple_tf_conflicts:{tech_conflicting}")

    # 5. Spread/ATR
    spread_pips = safe_float(risk_context.get("spread_pips"), safe_float(model_features.get("spread_pips"), 0.0))
    atr_pips = safe_float(model_features.get("atr_pips"), safe_float(risk_context.get("atr_pips"), 0.0))
    spread_atr = safe_float(risk_context.get("spread_atr"), 0.0)
    if spread_atr <= 0 and atr_pips > 0:
        spread_atr = spread_pips / atr_pips

    if spread_atr > 0:
        if spread_atr > AI_REVIEW_MAX_SPREAD_ATR:
            risk += 20
            reasons.append(f"spread_atr_too_high:{spread_atr:.3f}>{AI_REVIEW_MAX_SPREAD_ATR:.3f}")
        elif spread_atr > AI_REVIEW_MAX_SPREAD_ATR * 0.75:
            risk += 8
            reasons.append(f"spread_atr_elevated:{spread_atr:.3f}")
        else:
            supports.append(f"spread_atr_ok:{spread_atr:.3f}")

    # 6. Pair score for H1 registry system.
    pair_score = safe_float(context.get("pair_score"), 1.0)
    min_pair_score = safe_float(context.get("min_pair_score"), 0.0)
    if min_pair_score > 0 and pair_score < min_pair_score:
        risk += 25
        reasons.append(f"pair_score_low:{pair_score:.3f}<{min_pair_score:.3f}")
    elif pair_score >= max(min_pair_score, 0.5):
        supports.append(f"pair_score_ok:{pair_score:.3f}")

    risk = int(max(0, min(100, round(risk))))

    # Final decision: tiered, not one hard 60 block.
    if risk >= AI_REVIEW_HARD_BLOCK_SCORE:
        verdict = "REJECT"
        decision = "block"
        reason_code = f"ai_hard_block_risk_score:{risk}>={AI_REVIEW_HARD_BLOCK_SCORE}"
    elif prob < AI_REVIEW_MIN_MODEL_PROB:
        verdict = "REJECT"
        decision = "block"
        reason_code = f"ai_block_low_probability:{prob:.3f}<{AI_REVIEW_MIN_MODEL_PROB:.3f}"
    elif risk <= AI_REVIEW_MAX_RISK_SCORE:
        verdict = "APPROVE"
        decision = "allow"
        reason_code = f"ai_review_passed:risk={risk}<={AI_REVIEW_MAX_RISK_SCORE}"
    elif risk <= AI_REVIEW_CONDITIONAL_RISK_SCORE and prob >= AI_REVIEW_STRONG_MODEL_PROB:
        verdict = "APPROVE"
        decision = "allow_conditional"
        reason_code = f"ai_conditional_allow:risk={risk},prob={prob:.3f}"
    else:
        verdict = "REJECT"
        decision = "block"
        reason_code = f"ai_block_risk_score:{risk}>{AI_REVIEW_MAX_RISK_SCORE}"

    trend_text = ", ".join([f"{tf}={trends.get(tf)}" for tf in tf_order]) or "none"
    explanation = (
        f"Side-aware H1 review for {side}: risk={risk}, probability={prob * 100:.1f}%. "
        f"Trends {trend_text}. Supports={supports[:6]}. Conflicts={conflicts[:6]}. Reasons={reasons[:6]}."
    )

    return {
        "enabled": True,
        "provider": "side_aware_rules",
        "model": "deterministic_h1",
        "ai_verdict": verdict,
        "decision": decision,
        "risk_score": risk,
        "reason": f"{reason_code}. {explanation}",
        "side": side,
        "model_probability": prob,
        "timeframe_trends": trends,
        "aligned_timeframes": aligned_count,
        "conflicting_timeframes": conflict_count,
        "supports": supports,
        "conflicts": conflicts,
        "risk_reasons": reasons,
        "spread_atr": spread_atr,
    }


def ai_merge_llm_and_rule_reviews(rule_review: Dict[str, Any], llm_review: Dict[str, Any]) -> Dict[str, Any]:
    """Keep LLM context, but never let wording bugs reverse side logic."""
    if not isinstance(llm_review, dict):
        return rule_review

    merged = dict(llm_review)
    merged["side_aware_rule_review"] = rule_review

    # Hard safety: deterministic hard blocks always stay rejected.
    if rule_review.get("ai_verdict") == "REJECT" and safe_int(rule_review.get("risk_score"), 100) >= AI_REVIEW_HARD_BLOCK_SCORE:
        return rule_review

    # Low probability should remain rejected.
    if rule_review.get("ai_verdict") == "REJECT" and safe_float(rule_review.get("model_probability"), 0.0) < AI_REVIEW_MIN_MODEL_PROB:
        return rule_review

    llm_verdict = str(llm_review.get("ai_verdict", "REJECT")).upper()
    llm_risk = safe_int(llm_review.get("risk_score"), 100)

    if llm_verdict == "APPROVE" and llm_risk < AI_REVIEW_HARD_BLOCK_SCORE:
        return merged

    # Fix for SELL + bearish trend or BUY + bullish trend being incorrectly treated as conflict.
    if rule_review.get("ai_verdict") == "APPROVE" and safe_int(rule_review.get("risk_score"), 100) <= AI_REVIEW_CONDITIONAL_RISK_SCORE:
        merged["ai_verdict"] = "APPROVE"
        merged["risk_score"] = min(llm_risk, safe_int(rule_review.get("risk_score"), 0))
        merged["reason"] = (
            "Side-aware rule override approved. "
            f"Rule review: {rule_review.get('reason')} "
            f"Original AI reason: {llm_review.get('reason')}"
        )[:900]
        return merged

    return merged


def review_signal_with_ai(context: Dict[str, Any]) -> Dict[str, Any]:
    """AI compares H1 signal with fresh market context, then approves or rejects.

    v6 H1 patch from M15 v16:
    - deterministic side-aware review runs first
    - bearish trend supports SELL
    - bullish trend supports BUY
    - tiered risk approval instead of one hard 60 block
    - if API key is missing and fallback is enabled, use side-aware rules instead of blocking all trades
    """
    if not AI_REVIEW_ENABLED:
        return {"enabled": False, "ai_verdict": "SKIPPED", "risk_score": 0, "reason": "AI review disabled"}

    rule_review = ai_side_aware_rule_review(context)

    # Optional deterministic mode for testing:
    # AI_REVIEW_PROVIDER=rules
    if AI_REVIEW_PROVIDER in {"rules", "rule", "deterministic", "none"}:
        return rule_review

    system_prompt = (
        "You are a conservative forex H1 trade risk reviewer. You do not place trades. "
        "Return JSON only with ai_verdict APPROVE or REJECT, risk_score integer 0-100, and reason. "
        "CRITICAL SIDE-AWARE RULES: bullish trend supports BUY; bearish trend supports SELL. "
        "Do not reject a SELL because EMA/momentum/trend is bearish. That supports SELL. "
        "Do not reject a BUY because EMA/momentum/trend is bullish. That supports BUY. "
        "Reject when the alert side conflicts with the trend: SELL conflicts with bullish; BUY conflicts with bearish. "
        "Use the provided side_aware_rule_review as a safety reference. "
        "Approve only when model probability, pair score, trend, EMA/momentum, spread, noise/news guards, and risk limits mostly agree. "
        "Reject genuine conflicts: side disagrees with H1/H4/D trend, high spread vs ATR, stale/noisy signal, news blackout, live reversal risk, weak edge, or poor risk/reward."
    )

    context_with_rules = dict(context)
    context_with_rules["side_aware_rule_review"] = rule_review
    user_payload = json.dumps(context_with_rules, default=str, separators=(",", ":"))[:24000]

    try:
        if AI_REVIEW_PROVIDER == "anthropic":
            if not ANTHROPIC_API_KEY:
                if AI_REVIEW_FALLBACK_TO_RULES:
                    return {**rule_review, "provider": "side_aware_rules_fallback", "reason": "ANTHROPIC_API_KEY missing; used side-aware rules fallback. " + str(rule_review.get("reason", ""))}
                return {"enabled": True, "ai_verdict": "REJECT", "risk_score": 100, "reason": "ANTHROPIC_API_KEY missing"}
            from anthropic import Anthropic
            client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=AI_REVIEW_TIMEOUT_SECONDS)
            response = client.messages.create(
                model=AI_REVIEW_MODEL,
                max_tokens=350,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_payload}],
            )
            text = "".join([getattr(block, "text", "") for block in response.content])
            result = _parse_json_object_from_text(text)
        else:
            if not OPENAI_API_KEY:
                if AI_REVIEW_FALLBACK_TO_RULES:
                    return {**rule_review, "provider": "side_aware_rules_fallback", "reason": "OPENAI_API_KEY missing; used side-aware rules fallback. " + str(rule_review.get("reason", ""))}
                return {"enabled": True, "ai_verdict": "REJECT", "risk_score": 100, "reason": "OPENAI_API_KEY missing"}
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, timeout=AI_REVIEW_TIMEOUT_SECONDS)
            response = client.chat.completions.create(
                model=AI_REVIEW_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            )
            text = response.choices[0].message.content or "{}"
            result = _parse_json_object_from_text(text)

        verdict = str(result.get("ai_verdict", "REJECT")).strip().upper()
        risk_score = int(max(0, min(100, safe_int(result.get("risk_score"), 100))))
        reason = str(result.get("reason") or "AI review returned no reason")[:700]

        prob = safe_float(context.get("model_approval_probability"), safe_float(context.get("confidence"), 0.0))
        if risk_score >= AI_REVIEW_HARD_BLOCK_SCORE:
            verdict = "REJECT"
            reason = f"Hard risk score {risk_score}>={AI_REVIEW_HARD_BLOCK_SCORE}. {reason}"
        elif prob < AI_REVIEW_MIN_MODEL_PROB:
            verdict = "REJECT"
            reason = f"Model probability {prob:.3f}<{AI_REVIEW_MIN_MODEL_PROB:.3f}. {reason}"
        elif risk_score > AI_REVIEW_MAX_RISK_SCORE:
            if risk_score <= AI_REVIEW_CONDITIONAL_RISK_SCORE and prob >= AI_REVIEW_STRONG_MODEL_PROB:
                verdict = "APPROVE"
                reason = f"Conditional AI approval: risk {risk_score}<={AI_REVIEW_CONDITIONAL_RISK_SCORE} and probability {prob:.3f}>={AI_REVIEW_STRONG_MODEL_PROB:.3f}. {reason}"
            else:
                verdict = "REJECT"
                reason = f"Risk score {risk_score}>{AI_REVIEW_MAX_RISK_SCORE}. {reason}"

        if verdict not in {"APPROVE", "REJECT"}:
            verdict = "REJECT"
            risk_score = max(risk_score, 100)
            reason = f"Invalid AI verdict. {reason}"

        llm_review = {
            "enabled": True,
            "provider": AI_REVIEW_PROVIDER,
            "model": AI_REVIEW_MODEL,
            "ai_verdict": verdict,
            "risk_score": risk_score,
            "reason": reason,
        }
        return ai_merge_llm_and_rule_reviews(rule_review, llm_review)

    except Exception as exc:
        if AI_REVIEW_FALLBACK_TO_RULES:
            return {**rule_review, "provider": "side_aware_rules_fallback", "reason": f"AI review error; used side-aware rules fallback: {repr(exc)}. {rule_review.get('reason', '')}"}
        return {
            "enabled": True,
            "provider": AI_REVIEW_PROVIDER,
            "model": AI_REVIEW_MODEL,
            "ai_verdict": "REJECT",
            "risk_score": 100,
            "reason": f"AI review error: {repr(exc)}",
        }

# ====================================================
# MODEL LOADING
# ====================================================
def normalize_bundle(path: str, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pair_raw = str(raw.get("pair", "")).upper().replace("_", "")
    if not pair_raw:
        stem = Path(path).stem.upper().replace("_BUNDLE", "")
        pair_raw = stem if stem in PAIR_MAP else ""

    if not pair_raw or pair_raw not in PAIR_MAP:
        return None

    feat_order = list(raw.get("feature_order") or raw.get("features") or [])
    if not feat_order:
        return None

    train_meta = raw.get("train_meta") or {}
    labeling = raw.get("labeling") or {}
    avg_auc = safe_float(raw.get("avg_auc"), safe_float(train_meta.get("avg_auc"), 0.0))
    model_version = str(raw.get("model_version") or Path(path).name)

    return {
        "pair6": pair_raw,
        "instrument": pair_to_instrument(pair_raw),
        "model": raw["model"],
        "calibrator": raw.get("calibrator"),
        "feature_order": feat_order,
        "avg_auc": avg_auc,
        "labeling": {
            "sl_atr": safe_float(labeling.get("sl_atr"), DEFAULT_SL_ATR),
            "tp_atr": safe_float(labeling.get("tp_atr"), DEFAULT_TP_ATR),
        },
        "model_version": model_version,
        "_bundle_path": path,
    }

def load_bundles(models_dir: str) -> Dict[str, Dict[str, Any]]:
    bundles: Dict[str, Dict[str, Any]] = {}
    seen = set()
    patterns = ["*.joblib", "*_bundle.joblib"]
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(models_dir, pat))):
            if path in seen:
                continue
            seen.add(path)
            try:
                raw = joblib.load(path)
                if not isinstance(raw, dict) or "model" not in raw:
                    continue
                b = normalize_bundle(path, raw)
                if b:
                    bundles[b["pair6"]] = b
            except Exception:
                continue
    return bundles

# ====================================================
# AUTO MODEL REGISTRY LOADING
# CatBoost auto model + LightGBM auto model + Neural TCN challenger.
# The training script writes models_h1/registry.json. This server reads it and
# converts the winning model for each pair into the same bundle shape used by
# the existing /predict route.
# ====================================================
class LightGBMBoosterWrapper:
    def __init__(self, booster: Any):
        self.booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self.booster.predict(X), dtype=float).reshape(-1)
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


if nn is not None:
    class Chomp1d(nn.Module):
        def __init__(self, chomp_size: int):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            if self.chomp_size == 0:
                return x
            return x[:, :, :-self.chomp_size].contiguous()


    class TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            residual = x if self.downsample is None else self.downsample(x)
            return self.relu(out + residual)


    class TCNClassifier(nn.Module):
        def __init__(self, n_features: int, channels=(32, 32, 32), kernel_size: int = 3, dropout: float = 0.15):
            super().__init__()
            layers = []
            in_ch = n_features
            for i, out_ch in enumerate(channels):
                layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))
                in_ch = out_ch
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_ch, 1))

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.tcn(x)
            return self.head(x).squeeze(1)
else:
    TCNClassifier = None


class TCNRuntimeWrapper:
    def __init__(self, pair6: str, model: Any, scaler: Any, features: List[str], lookback: int):
        self.pair6 = pair6
        self.model = model
        self.scaler = scaler
        self.features = features
        self.lookback = int(lookback)
        _tcn_feature_history.setdefault(pair6, deque(maxlen=max(self.lookback, BAR_HISTORY_LEN)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # The TCN needs a rolling sequence of feature rows. Until enough H1 bars
        # arrive after startup, return neutral probability so the gate blocks trading.
        row = [safe_float(X.iloc[0][f]) for f in self.features]
        q = _tcn_feature_history.setdefault(self.pair6, deque(maxlen=max(self.lookback, BAR_HISTORY_LEN)))
        q.append(row)

        if len(q) < self.lookback or torch is None:
            return np.array([[0.5, 0.5]], dtype=float)

        seq = np.asarray(list(q)[-self.lookback:], dtype=np.float32)
        if self.scaler is not None:
            seq = self.scaler.transform(seq).astype(np.float32)

        with torch.no_grad():
            xt = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(TORCH_DEVICE)
            logits = self.model(xt)
            p_up = float(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0])

        p_up = max(0.0, min(1.0, p_up))
        return np.array([[1.0 - p_up, p_up]], dtype=float)


def _registry_pair_to_pair6(pair_value: Any) -> Optional[str]:
    pair6 = str(pair_value or "").upper().replace("_", "").replace("/", "").strip()
    return pair6 if pair6 in PAIR_MAP else None


def _avg_auc_from_auto_meta(meta: Dict[str, Any]) -> float:
    summary = meta.get("summary") or {}
    best = str(meta.get("best_model") or "").lower()
    if best in summary:
        return safe_float(summary.get(best, {}).get("avg_auc"), 0.0)
    for key in ("catboost", "lightgbm", "tcn"):
        if key in summary:
            return safe_float(summary.get(key, {}).get("avg_auc"), 0.0)
    return safe_float(meta.get("avg_auc"), 0.0)


def _load_one_auto_registry_model(pair6: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best_model = str(meta.get("best_model") or "").lower().strip()
    summary = meta.get("summary") or {}
    features = list(
        meta.get("features")
        or meta.get("feature_order")
        or summary.get("feature_columns")
        or []
    )
    if not features:
        print(f"WARNING: auto registry has no feature columns for {pair6}")
        return None

    model_path_raw = meta.get("model_path")
    if not model_path_raw:
        return None
    model_path = Path(str(model_path_raw))
    if not model_path.is_absolute():
        candidates = [
            model_path,
            Path(MODELS_DIR) / model_path,
            Path(MODELS_DIR) / pair6 / model_path.name,
            Path(MODELS_DIR) / pair6 / "best_model.pkl",
        ]
        model_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    if not model_path.exists():
        print(f"WARNING: auto registry model missing for {pair6}: {model_path}")
        return None

    try:
        # The H1 registry saves ExtraTrees, XGBoost, LogisticRegression,
        # LightGBM, and CatBoost as joblib .pkl files.
        if model_path.suffix.lower() in {".pkl", ".joblib"}:
            model = joblib.load(model_path)

        elif best_model == "catboost":
            if CatBoostClassifier is None:
                print(f"WARNING: CatBoost not installed; skipping {pair6}")
                return None
            model = CatBoostClassifier()
            model.load_model(str(model_path))

        elif best_model == "lightgbm":
            if lgb is None:
                print(f"WARNING: LightGBM not installed; skipping {pair6}")
                return None
            booster = lgb.Booster(model_file=str(model_path))
            model = LightGBMBoosterWrapper(booster)

        elif best_model == "tcn":
            if torch is None or TCNClassifier is None:
                print(f"WARNING: PyTorch not installed; skipping TCN for {pair6}")
                return None
            tcn = TCNClassifier(n_features=len(features)).to(TORCH_DEVICE)
            tcn.load_state_dict(torch.load(str(model_path), map_location=TORCH_DEVICE))
            tcn.eval()

            scaler = None
            scaler_path_raw = meta.get("scaler_path")
            if scaler_path_raw:
                scaler_path = Path(str(scaler_path_raw))
                if not scaler_path.is_absolute() and not scaler_path.exists():
                    scaler_path = Path(MODELS_DIR) / scaler_path.name
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
            model = TCNRuntimeWrapper(pair6, tcn, scaler, features, TCN_LOOKBACK)

        else:
            try:
                model = joblib.load(model_path)
            except Exception as load_exc:
                print(f"WARNING: unknown/unsupported best_model for {pair6}: {best_model}; {repr(load_exc)}")
                return None

        avg_auc = _avg_auc_from_auto_meta(meta)
        pair_score = safe_float(meta.get("pair_score"), 0.50)
        default_gate = safe_float(meta.get("default_gate"), DEFAULT_GATE["conf"])
        default_margin = safe_float(meta.get("default_margin"), DEFAULT_GATE["margin"])

        return {
            "pair6": pair6,
            "instrument": pair_to_instrument(pair6),
            "model": model,
            "calibrator": None,
            "feature_order": features,
            "avg_auc": avg_auc,
            "pair_score": pair_score,
            "labeling": {
                "sl_atr": safe_float(meta.get("sl_atr"), DEFAULT_SL_ATR),
                "tp_atr": safe_float(meta.get("tp_atr"), DEFAULT_TP_ATR),
            },
            "model_version": f"auto_registry:{best_model}:{Path(model_path).name}",
            "best_model": best_model,
            "model_source": "auto_registry",
            "gate_override": {"conf": default_gate, "margin": default_margin},
            "raw_meta": meta,
            "_bundle_path": str(model_path),
        }
    except Exception as e:
        print(f"ERROR loading auto registry model for {pair6}: {repr(e)}")
        return None


def load_auto_registry_bundles(registry_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not AUTO_MODEL_REGISTRY_ENABLED:
        return out
    path = Path(registry_path)
    if not path.exists():
        print(f"Auto registry not found at {path}; using older joblib bundles only.")
        return out

    try:
        with open(path, "r") as f:
            registry = json.load(f)
    except Exception as e:
        print(f"ERROR reading auto registry {path}: {repr(e)}")
        return out

    pairs = registry.get("pairs") or {}
    for pair_key, meta in pairs.items():
        if not isinstance(meta, dict):
            continue
        pair6 = _registry_pair_to_pair6(meta.get("pair") or pair_key)
        if not pair6:
            continue
        bundle = _load_one_auto_registry_model(pair6, meta)
        if bundle:
            out[pair6] = bundle
            print(f"Loaded auto registry {pair6}: {bundle.get('best_model')} | pair_score={bundle.get('pair_score')} | gate={bundle.get('gate_override')}")
    return out


JOBLIB_BUNDLES = load_bundles(MODELS_DIR)
AUTO_REGISTRY_BUNDLES = load_auto_registry_bundles(AUTO_REGISTRY_PATH)

if AUTO_REGISTRY_OVERRIDES_JOBLIB:
    BUNDLES = {**JOBLIB_BUNDLES, **AUTO_REGISTRY_BUNDLES}
else:
    BUNDLES = {**AUTO_REGISTRY_BUNDLES, **JOBLIB_BUNDLES}



# ====================================================
# H1 HYBRID OANDA CANDLE FEATURE HELPERS
# ====================================================
def fetch_oanda_candles(instrument: str, granularity: str, count: int) -> Dict[str, Any]:
    safe_count = max(20, min(int(count or 120), 500))
    gran = str(granularity or "H1").upper()
    return oanda_request(
        "GET",
        f"/v3/instruments/{instrument}/candles?price=M&granularity={gran}&count={safe_count}",
        timeout=MARKET_CONTEXT_MAX_FETCH_SECONDS,
    )


def candles_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    if not result.get("ok"):
        return pd.DataFrame()
    data = result.get("data") or {}
    candles = data.get("candles") or []
    rows = []
    for candle in candles:
        mid = candle.get("mid") or {}
        if not mid:
            continue
        rows.append(
            {
                "time": pd.to_datetime(candle.get("time"), utc=True, errors="coerce"),
                "complete": bool(candle.get("complete", False)),
                "volume": safe_float(candle.get("volume"), 0.0),
                "mid_o": safe_float(mid.get("o"), 0.0),
                "mid_h": safe_float(mid.get("h"), 0.0),
                "mid_l": safe_float(mid.get("l"), 0.0),
                "mid_c": safe_float(mid.get("c"), 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def generic_h1_training_features(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    work = df.copy()
    for col in ["mid_o", "mid_h", "mid_l", "mid_c", "volume", "spread_c"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")
    close = work["mid_c"]
    pip = instrument_pip_size(instrument)
    work["ret1"] = close.pct_change(1)
    work["ret3"] = close.pct_change(3)
    work["ret5"] = close.pct_change(5)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    work["ema20_dist"] = (close - ema20) / close.replace(0, np.nan)
    work["ema50_dist"] = (close - ema50) / close.replace(0, np.nan)
    work["ema200_dist"] = (close - ema200) / close.replace(0, np.nan)
    work["rsi14"] = rsi_runtime(close, 14)
    work["atr14"] = atr_runtime(work, 14)
    work["atr14_pct"] = work["atr14"] / close.replace(0, np.nan) * 100.0
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    work["bb_width"] = ((ma20 + 2 * std20) - (ma20 - 2 * std20)) / close.replace(0, np.nan)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    work["macd_hist"] = macd - signal
    volume_mean = work["volume"].rolling(50).mean()
    volume_std = work["volume"].rolling(50).std()
    work["vol_z"] = (work["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    work["spread_pips"] = work["spread_c"].fillna(0.0) / pip if pip > 0 else 0.0
    dt_series = pd.to_datetime(work["time"], utc=True, errors="coerce")
    work["hour_utc"] = dt_series.dt.hour.fillna(0)
    work["dayofweek"] = dt_series.dt.dayofweek.fillna(0)
    work["range_pips"] = (work["mid_h"] - work["mid_l"]).abs() / pip if pip > 0 else 0.0
    work["body_pips"] = (work["mid_c"] - work["mid_o"]).abs() / pip if pip > 0 else 0.0
    work["body_range_ratio"] = work["body_pips"] / work["range_pips"].replace(0, np.nan)
    return work


def build_oanda_h1_feature_row(payload: Dict[str, Any], pair6: str, instrument: str, feature_order: list[str]) -> Dict[str, Any]:
    meta = {
        "_model_feature_source_requested": MODEL_FEATURE_SOURCE,
        "_model_feature_source_used": "oanda",
        "_model_feature_source_ok": False,
        "_model_feature_source_reason": "not_started",
        "_model_feature_granularity": MODEL_FEATURE_OANDA_GRANULARITY,
    }
    if not broker_can_close():
        meta["_model_feature_source_used"] = "alert"
        meta["_model_feature_source_reason"] = "broker_not_ready"
        return meta

    try:
        count = max(MODEL_FEATURE_OANDA_MIN_CANDLES, min(MODEL_FEATURE_OANDA_CANDLE_COUNT, 500))
        result = fetch_oanda_candles(instrument, MODEL_FEATURE_OANDA_GRANULARITY, count)
        if not result.get("ok"):
            meta["_model_feature_source_used"] = "alert"
            meta["_model_feature_source_reason"] = f"oanda_fetch_failed:{result.get('error') or result.get('status_code')}"
            return meta

        df = candles_to_dataframe(result)
        if df.empty:
            meta["_model_feature_source_used"] = "alert"
            meta["_model_feature_source_reason"] = "oanda_no_candles"
            return meta

        if "complete" in df.columns:
            df = df[df["complete"] == True].copy()
        df = df.tail(count).copy()

        if len(df) < MODEL_FEATURE_OANDA_MIN_CANDLES:
            meta["_model_feature_source_used"] = "alert"
            meta["_model_feature_source_reason"] = f"oanda_not_enough_completed_candles:{len(df)}"
            return meta

        pip = instrument_pip_size(instrument)
        spread_c = safe_float(payload.get("spread_c"), np.nan)
        if not np.isfinite(spread_c) or spread_c <= 0:
            spread_pips = safe_float(payload.get("spread_pips"), 0.0)
            spread_c = spread_pips * pip if pip > 0 else 0.0
        if not np.isfinite(spread_c):
            spread_c = 0.0
        df["spread_c"] = spread_c

        if "add_h1_training_features" in globals():
            feat_df = add_h1_training_features(df, instrument)
        elif "add_training_features" in globals():
            feat_df = add_training_features(df, instrument)
        elif "add_runtime_features" in globals():
            feat_df = add_runtime_features(df, instrument)
        else:
            feat_df = generic_h1_training_features(df, instrument)

        last = feat_df.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict()
        row = {f: safe_float(last.get(f), safe_float(payload.get(f), 0.0)) for f in feature_order}
        row.update(meta)
        row["_model_feature_source_ok"] = True
        row["_model_feature_source_used"] = "oanda_latest_closed"
        row["_model_feature_source_reason"] = "oanda_latest_completed_h1_candle_features"
        row["_model_feature_time"] = str(last.get("time", ""))
        row["_model_feature_last_close"] = safe_float(last.get("mid_c"), 0.0)
        row["_model_feature_candles"] = int(len(df))
        return row
    except Exception as exc:
        meta["_model_feature_source_used"] = "alert"
        meta["_model_feature_source_reason"] = f"oanda_feature_error:{repr(exc)}"
        return meta



def _direction_label_h1(value: float, eps: float = 0.0) -> str:
    if value > eps:
        return "bullish"
    if value < -eps:
        return "bearish"
    return "neutral"


def classify_latest_h1_candle_pattern(df: pd.DataFrame, instrument: str, lookback_trend_bars: int = 5) -> Dict[str, Any]:
    """Conservative candlestick context for H1 technical review."""
    if df.empty:
        return {"pattern": "unknown", "candle_bias": "NEUTRAL", "pattern_confidence": 0, "reason": "no_candles"}
    pip = instrument_pip_size(instrument)
    work = df.copy()
    last = work.iloc[-1]
    prev = work.iloc[-2] if len(work) >= 2 else None
    o = safe_float(last.get("mid_o"), 0.0)
    h = safe_float(last.get("mid_h"), 0.0)
    l = safe_float(last.get("mid_l"), 0.0)
    c = safe_float(last.get("mid_c"), 0.0)
    rng = max(h - l, 0.0)
    body = abs(c - o)
    upper = max(h - max(o, c), 0.0)
    lower = max(min(o, c) - l, 0.0)
    signed = c - o
    range_pips = rng / pip if pip > 0 else 0.0
    body_pips = body / pip if pip > 0 else 0.0
    body_signed_pips = signed / pip if pip > 0 else 0.0
    body_ratio = body / rng if rng > 0 else 0.0
    upper_ratio = upper / rng if rng > 0 else 0.0
    lower_ratio = lower / rng if rng > 0 else 0.0
    wick_body = max(upper, lower) / max(body, 1e-12)

    trend_bias = "NEUTRAL"
    if len(work) >= lookback_trend_bars + 1:
        prior = safe_float(work["mid_c"].iloc[-lookback_trend_bars-1], 0.0)
        if prior > 0:
            change = c / prior - 1.0
            trend_bias = "BUY" if change > 0 else "SELL" if change < 0 else "NEUTRAL"

    pattern, bias, confidence, reason = "neutral", "NEUTRAL", 25, "small_or_mixed_candle"
    if prev is not None:
        po = safe_float(prev.get("mid_o"), 0.0)
        pc = safe_float(prev.get("mid_c"), 0.0)
        curr_bull, curr_bear = c > o, c < o
        prev_bull, prev_bear = pc > po, pc < po
        engulfs = min(o, c) <= min(po, pc) and max(o, c) >= max(po, pc)
        if curr_bull and prev_bear and engulfs and body_ratio >= 0.35:
            pattern, bias, confidence, reason = "bullish_engulfing", "BUY", 78, "bullish_body_engulfed_prior_bearish_body"
        elif curr_bear and prev_bull and engulfs and body_ratio >= 0.35:
            pattern, bias, confidence, reason = "bearish_engulfing", "SELL", 78, "bearish_body_engulfed_prior_bullish_body"

    if pattern == "neutral":
        if rng <= 0 or body_ratio <= 0.10 or body_pips <= 0.5:
            pattern, bias, confidence, reason = "doji", "NEUTRAL", 65, "very_small_body_relative_to_range"
        elif lower >= 2.0 * max(body, 1e-12) and upper_ratio <= 0.30 and lower_ratio >= 0.45:
            pattern = "hammer" if trend_bias in {"SELL", "NEUTRAL"} else "hanging_man"
            bias = "BUY" if pattern == "hammer" else "SELL"
            confidence, reason = 70, "long_lower_wick_rejection"
        elif upper >= 2.0 * max(body, 1e-12) and lower_ratio <= 0.30 and upper_ratio >= 0.45:
            pattern = "shooting_star" if trend_bias in {"BUY", "NEUTRAL"} else "inverted_hammer"
            bias = "SELL" if pattern == "shooting_star" else "BUY"
            confidence, reason = 70, "long_upper_wick_rejection"
        elif body_ratio >= 0.65 and signed > 0:
            pattern, bias, confidence, reason = "strong_bull", "BUY", 70, "large_bullish_body"
        elif body_ratio >= 0.65 and signed < 0:
            pattern, bias, confidence, reason = "strong_bear", "SELL", 70, "large_bearish_body"
        elif prev is not None:
            ph = safe_float(prev.get("mid_h"), 0.0); pl = safe_float(prev.get("mid_l"), 0.0)
            if h <= ph and l >= pl:
                pattern, bias, confidence, reason = "inside_bar", "NEUTRAL", 55, "inside_previous_candle_range"
            elif h >= ph and l <= pl:
                pattern, bias, confidence, reason = "outside_bar", ("BUY" if signed > 0 else "SELL" if signed < 0 else "NEUTRAL"), 60, "outside_previous_candle_range"
    return {
        "pattern": pattern,
        "candle_bias": bias,
        "pattern_confidence": int(confidence),
        "reason": reason,
        "trend_bias_last5": trend_bias,
        "body_pips": float(body_pips),
        "body_pips_signed": float(body_signed_pips),
        "range_pips": float(range_pips),
        "body_range_ratio": float(body_ratio),
        "upper_wick_range_ratio": float(upper_ratio),
        "lower_wick_range_ratio": float(lower_ratio),
        "wick_body_ratio": float(wick_body),
    }


def build_external_market_context(pair6: str, instrument: str, hint_side: str, feature_row: Dict[str, Any]) -> Dict[str, Any]:
    if not MARKET_CONTEXT_ENABLED:
        return {"enabled": False, "ok": True, "reason": "market_context_disabled"}
    if not broker_can_close():
        return {"enabled": True, "ok": not MARKET_CONTEXT_REQUIRED, "reason": "broker_not_ready_for_market_context"}

    summaries: Dict[str, Any] = {}
    errors = []
    for granularity in MARKET_CONTEXT_GRANULARITIES:
        result = fetch_oanda_candles(instrument, granularity, MARKET_CONTEXT_CANDLE_COUNT)
        if not result.get("ok"):
            errors.append(f"{granularity}:fetch_failed:{result.get('error') or result.get('status_code')}")
            summaries[granularity] = {"ok": False, "granularity": granularity, "reason": str(result.get("error") or result.get("status_code") or "fetch_failed")}
            continue

        df = candles_to_dataframe(result)
        if "complete" in df.columns and not df.empty:
            df = df[df["complete"] == True].copy()
        if df.empty or len(df) < 20:
            summaries[granularity] = {"ok": False, "granularity": granularity, "reason": f"not_enough_candles:{len(df)}"}
            continue

        df = df.tail(MARKET_CONTEXT_CANDLE_COUNT).copy()
        close = pd.to_numeric(df["mid_c"], errors="coerce")
        pip = instrument_pip_size(instrument)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        atr = atr_runtime(df, 14)
        rsi = rsi_runtime(close, 14)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        recent_high_20 = float(pd.to_numeric(df["mid_h"], errors="coerce").tail(20).max())
        recent_low_20 = float(pd.to_numeric(df["mid_l"], errors="coerce").tail(20).min())
        candle_pattern = classify_latest_h1_candle_pattern(df, instrument)
        last = df.iloc[-1]
        last_close = safe_float(last.get("mid_c"), 0.0)
        prev_close = safe_float(close.iloc[-2], last_close) if len(close) >= 2 else last_close
        ret1 = (last_close / prev_close - 1.0) if prev_close > 0 else 0.0
        ret3 = close.pct_change(3).iloc[-1] if len(close) >= 4 else 0.0
        ret5 = close.pct_change(5).iloc[-1] if len(close) >= 6 else 0.0
        ema20_dist = (last_close - float(ema20.iloc[-1])) / last_close if last_close > 0 else 0.0
        ema50_dist = (last_close - float(ema50.iloc[-1])) / last_close if last_close > 0 else 0.0
        ema200_dist = (last_close - float(ema200.iloc[-1])) / last_close if last_close > 0 else 0.0
        body_pips_signed = (safe_float(last.get("mid_c"), 0.0) - safe_float(last.get("mid_o"), 0.0)) / pip if pip > 0 else 0.0
        range_pips = (safe_float(last.get("mid_h"), 0.0) - safe_float(last.get("mid_l"), 0.0)) / pip if pip > 0 else 0.0

        side = 1 if hint_side == "BUY" else -1 if hint_side == "SELL" else 0
        aligned_votes = 0
        for v in [ret1, ret3, ret5, ema20_dist, ema50_dist, body_pips_signed]:
            value = safe_float(v, 0.0)
            if side == 1 and value > 0:
                aligned_votes += 1
            elif side == -1 and value < 0:
                aligned_votes += 1

        summaries[granularity] = {
            "ok": True,
            "granularity": granularity,
            "candles": int(len(df)),
            "last_time": str(last.get("time", "")),
            "last_close": float(last_close),
            "ret1": float(ret1) if np.isfinite(ret1) else 0.0,
            "ret3": float(ret3) if np.isfinite(ret3) else 0.0,
            "ret5": float(ret5) if np.isfinite(ret5) else 0.0,
            "ema20_dist": float(ema20_dist),
            "ema50_dist": float(ema50_dist),
            "ema200_dist": float(ema200_dist),
            "rsi14": float(rsi.iloc[-1]) if np.isfinite(rsi.iloc[-1]) else 50.0,
            "atr_pips": float(atr.iloc[-1] / pip) if pip > 0 and np.isfinite(atr.iloc[-1]) else 0.0,
            "macd_hist": float(macd_hist.iloc[-1]) if np.isfinite(macd_hist.iloc[-1]) else 0.0,
            "body_pips_signed": float(body_pips_signed),
            "range_pips": float(range_pips),
            "recent_high_20": recent_high_20,
            "recent_low_20": recent_low_20,
            "distance_to_recent_high_pips": float((recent_high_20 - last_close) / pip) if pip > 0 and recent_high_20 > 0 and last_close > 0 else 0.0,
            "distance_to_recent_low_pips": float((last_close - recent_low_20) / pip) if pip > 0 and recent_low_20 > 0 and last_close > 0 else 0.0,
            "price_vs_ema20": _direction_label_h1(ema20_dist),
            "price_vs_ema50": _direction_label_h1(ema50_dist),
            "price_vs_ema200": _direction_label_h1(ema200_dist),
            "candle_pattern": candle_pattern,
            "last_candle_pattern": candle_pattern.get("pattern"),
            "last_candle_bias": candle_pattern.get("candle_bias"),
            "hint_side_alignment_score": int(aligned_votes),
            "hint_side_aligned": bool(aligned_votes >= 4),
        }

    ok_summaries = [s for s in summaries.values() if s.get("ok")]
    return {
        "enabled": True,
        "ok": bool(ok_summaries) and (not MARKET_CONTEXT_REQUIRED or not errors),
        "instrument": instrument,
        "pair": pair6,
        "hint_side": hint_side,
        "granularities": MARKET_CONTEXT_GRANULARITIES,
        "summaries": summaries,
        "available_timeframes": len(ok_summaries),
        "aligned_timeframes": sum(1 for s in ok_summaries if s.get("hint_side_aligned")),
        "errors": errors,
    }


# Preserve original alert/runtime feature builder, then wrap it with H1 hybrid OANDA features.
ORIGINAL_ALERT_BUILD_RUNTIME_FEATURE_ROW = build_runtime_feature_row

def build_runtime_feature_row(payload: Dict[str, Any], pair6: str, instrument: str, feature_order: list[str]) -> Dict[str, Any]:
    if MODEL_FEATURE_SOURCE in {"oanda", "hybrid"}:
        oanda_row = build_oanda_h1_feature_row(payload, pair6, instrument, feature_order)
        if bool(oanda_row.get("_model_feature_source_ok", False)):
            return oanda_row
        if MODEL_FEATURE_SOURCE == "oanda" and not MODEL_FEATURE_FALLBACK_TO_ALERT:
            strict_row = {f: 0.0 for f in feature_order}
            strict_row.update(oanda_row)
            strict_row["_model_feature_source_used"] = "oanda_failed_no_alert_fallback"
            return strict_row

        fallback_row = ORIGINAL_ALERT_BUILD_RUNTIME_FEATURE_ROW(payload, pair6, instrument, feature_order)
        fallback_row.update({
            "_model_feature_source_requested": MODEL_FEATURE_SOURCE,
            "_model_feature_source_used": "alert_fallback",
            "_model_feature_source_ok": False,
            "_model_feature_source_reason": oanda_row.get("_model_feature_source_reason", "oanda_unavailable"),
            "_model_feature_granularity": MODEL_FEATURE_OANDA_GRANULARITY,
        })
        return fallback_row

    alert_row = ORIGINAL_ALERT_BUILD_RUNTIME_FEATURE_ROW(payload, pair6, instrument, feature_order)
    alert_row.update({
        "_model_feature_source_requested": MODEL_FEATURE_SOURCE,
        "_model_feature_source_used": "alert",
        "_model_feature_source_ok": True,
        "_model_feature_source_reason": "alert_feature_source_selected",
    })
    return alert_row


# ====================================================
# H1 FOREX TECHNICAL REVIEW
# ====================================================
def _side_to_sign_h1(side: str) -> int:
    side = normalize_side(side)
    if side == "BUY":
        return 1
    if side == "SELL":
        return -1
    return 0


def _summary_bias_for_side_h1(summary: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Score one H1/H4/D summary against BUY/SELL direction."""
    side = normalize_side(side)
    side_sign = _side_to_sign_h1(side)
    if not summary or not summary.get("ok") or side_sign == 0:
        return {"ok": False, "trend": "unknown", "score": 0, "supports": [], "conflicts": ["summary_unavailable"]}

    score = 0
    supports: List[str] = []
    conflicts: List[str] = []
    checks = [
        ("ema20", safe_float(summary.get("ema20_dist"), 0.0)),
        ("ema50", safe_float(summary.get("ema50_dist"), 0.0)),
        ("ema200", safe_float(summary.get("ema200_dist"), 0.0)),
        ("ret1", safe_float(summary.get("ret1"), 0.0)),
        ("ret3", safe_float(summary.get("ret3"), 0.0)),
        ("ret5", safe_float(summary.get("ret5"), 0.0)),
        ("macd_hist", safe_float(summary.get("macd_hist"), 0.0)),
        ("candle_body", safe_float(summary.get("body_pips_signed"), 0.0)),
    ]
    for label, value in checks:
        sig = _sign_for_direction(float(value), 0.0)
        if sig == side_sign:
            score += 1
            supports.append(label)
        elif sig == -side_sign:
            score -= 1
            conflicts.append(label)

    rsi14 = safe_float(summary.get("rsi14"), 50.0)
    if side == "BUY":
        if 45 <= rsi14 <= 68:
            score += 1; supports.append("rsi_constructive_buy")
        elif rsi14 < 42:
            score -= 1; conflicts.append("rsi_weak_for_buy")
        elif rsi14 > 75:
            score -= 1; conflicts.append("rsi_overextended_buy")
    elif side == "SELL":
        if 32 <= rsi14 <= 55:
            score += 1; supports.append("rsi_constructive_sell")
        elif rsi14 > 60:
            score -= 1; conflicts.append("rsi_strong_against_sell")
        elif rsi14 < 25:
            score -= 1; conflicts.append("rsi_oversold_sell_chase")

    if score >= 3:
        trend = "bullish" if side == "BUY" else "bearish"
    elif score <= -3:
        trend = "bearish" if side == "BUY" else "bullish"
    else:
        trend = "mixed"
    return {"ok": True, "trend": trend, "score": int(score), "supports": supports, "conflicts": conflicts}


def _nearest_h1_market_levels(summary: Dict[str, Any], instrument: str, side: str) -> Dict[str, Any]:
    pip = instrument_pip_size(instrument)
    last_close = safe_float(summary.get("last_close"), 0.0)
    recent_high = safe_float(summary.get("recent_high_20"), 0.0)
    recent_low = safe_float(summary.get("recent_low_20"), 0.0)
    atr_pips = safe_float(summary.get("atr_pips"), 0.0)
    distance_to_resistance = ((recent_high - last_close) / pip) if pip > 0 and recent_high > 0 and last_close > 0 else 0.0
    distance_to_support = ((last_close - recent_low) / pip) if pip > 0 and recent_low > 0 and last_close > 0 else 0.0
    risks: List[str] = []
    supports: List[str] = []
    near_mult = TECH_NEAR_SR_ATR_MULT
    if side == "BUY":
        if atr_pips > 0 and 0 < distance_to_resistance < near_mult * atr_pips:
            risks.append(f"buy_near_h1_resistance:{distance_to_resistance:.2f}pips")
        if atr_pips > 0 and distance_to_support > 0:
            supports.append(f"h1_support_buffer:{distance_to_support:.2f}pips")
    elif side == "SELL":
        if atr_pips > 0 and 0 < distance_to_support < near_mult * atr_pips:
            risks.append(f"sell_near_h1_support:{distance_to_support:.2f}pips")
        if atr_pips > 0 and distance_to_resistance > 0:
            supports.append(f"h1_resistance_buffer:{distance_to_resistance:.2f}pips")
    return {
        "last_close": last_close,
        "recent_high_20": recent_high,
        "recent_low_20": recent_low,
        "distance_to_resistance_pips": round(float(distance_to_resistance), 4),
        "distance_to_support_pips": round(float(distance_to_support), 4),
        "level_supports": supports,
        "level_risks": risks,
    }


def run_h1_forex_technical_review(
    pair6: str,
    instrument: str,
    hint_side: str,
    decision_prob: float,
    feature_row: Dict[str, Any],
    market_context: Dict[str, Any],
    spread_atr: float = 0.0,
) -> Dict[str, Any]:
    """Institutional-style H1/H4/D technical confirmation after ML approval."""
    if not TECHNICAL_REVIEW_ENABLED:
        return {"enabled": False, "decision": "SKIPPED", "allow_trade": True, "technical_score": 100.0, "reason": "technical_review_disabled"}

    side = normalize_side(hint_side)
    if side not in {"BUY", "SELL"}:
        return {"enabled": True, "decision": "BLOCK", "allow_trade": False, "technical_score": 0.0, "reason": "technical_review_blocked:invalid_side"}

    summaries = (market_context or {}).get("summaries") or {}
    h1 = summaries.get("H1") or {}
    h4 = summaries.get("H4") or {}
    d1 = summaries.get("D") or summaries.get("D1") or {}
    tf_results = {
        "H1": _summary_bias_for_side_h1(h1, side),
        "H4": _summary_bias_for_side_h1(h4, side),
        "D": _summary_bias_for_side_h1(d1, side),
    }

    score = 50.0
    supports: List[str] = []
    conflicts: List[str] = []
    reasons: List[str] = []

    aligned_timeframes = 0
    conflicting_timeframes = 0
    for tf, result in tf_results.items():
        tf_score = safe_int(result.get("score"), 0)
        if tf_score >= 3:
            aligned_timeframes += 1
            supports.append(f"{tf}_technical_alignment")
        elif tf_score <= -3:
            conflicting_timeframes += 1
            conflicts.append(f"{tf}_technical_conflict")

    h1_score = safe_int(tf_results["H1"].get("score"), 0)
    h4_score = safe_int(tf_results["H4"].get("score"), 0)
    d_score = safe_int(tf_results["D"].get("score"), 0)
    score += h1_score * 3.5
    score += h4_score * 2.5
    score += d_score * 1.5

    if aligned_timeframes >= 3:
        score += 10; supports.append("all_h1_h4_d_aligned")
    elif aligned_timeframes >= 2:
        score += 6; supports.append("two_timeframes_aligned")
    elif aligned_timeframes >= 1:
        score += 2; supports.append("one_timeframe_aligned")

    if conflicting_timeframes >= 3:
        score -= 20; reasons.append("all_h1_h4_d_conflict")
    elif conflicting_timeframes == 2:
        score -= 13; reasons.append("two_timeframes_conflict")
    elif conflicting_timeframes == 1:
        score -= 6; reasons.append("one_timeframe_conflicts")

    prob = safe_float(decision_prob, 0.0)
    if prob >= AI_REVIEW_STRONG_MODEL_PROB:
        score += 6; supports.append(f"model_probability_strong:{prob:.3f}")
    elif prob >= AI_REVIEW_MIN_MODEL_PROB:
        score += 2; supports.append(f"model_probability_acceptable:{prob:.3f}")
    elif prob < 0.40:
        score -= 12; reasons.append(f"model_probability_very_low:{prob:.3f}")
    else:
        score -= 6; reasons.append(f"model_probability_low:{prob:.3f}")

    candle = (h1.get("candle_pattern") or {}) if isinstance(h1, dict) else {}
    candle_bias = normalize_side(candle.get("candle_bias") or "")
    candle_pattern = str(candle.get("pattern") or h1.get("last_candle_pattern") or "unknown")
    body_ratio = safe_float(candle.get("body_range_ratio"), safe_float(feature_row.get("body_range_ratio"), 0.0))
    if candle_bias == side:
        score += 5; supports.append(f"h1_candle_supports_{side.lower()}:{candle_pattern}")
    elif candle_bias in {"BUY", "SELL"} and candle_bias != side:
        score -= 8; conflicts.append(f"h1_candle_conflicts:{candle_pattern}:{candle_bias}")
    elif candle_pattern.lower() in {"doji", "inside_bar"} or body_ratio < MIN_BODY_RANGE_RATIO:
        score -= 4; reasons.append(f"h1_indecision_or_small_body:{candle_pattern}")

    atr_pips = safe_float(h1.get("atr_pips"), safe_float(feature_row.get("atr_pips"), 0.0))
    spread_pips = safe_float(feature_row.get("spread_pips"), 0.0)
    if spread_atr <= 0 and atr_pips > 0:
        spread_atr = spread_pips / atr_pips
    if spread_atr > 0:
        if spread_atr > TECH_MAX_SPREAD_ATR:
            score -= 12; reasons.append(f"spread_atr_too_high:{spread_atr:.3f}>{TECH_MAX_SPREAD_ATR:.3f}")
        elif spread_atr > TECH_MAX_SPREAD_ATR * 0.75:
            score -= 5; reasons.append(f"spread_atr_elevated:{spread_atr:.3f}")
        else:
            score += 3; supports.append(f"spread_atr_ok:{spread_atr:.3f}")

    levels = _nearest_h1_market_levels(h1, instrument, side)
    if levels.get("level_supports"):
        score += 2; supports.extend(levels.get("level_supports") or [])
    if TECH_BLOCK_NEAR_SR and levels.get("level_risks"):
        score -= 7; reasons.extend(levels.get("level_risks") or [])

    min_score = TECH_MIN_SCORE_FOR_BUY if side == "BUY" else TECH_MIN_SCORE_FOR_SELL
    hard_failures: List[str] = []
    if TECH_REQUIRE_H1_ALIGNMENT and h1_score <= -3:
        hard_failures.append("h1_technical_conflict")
    if TECH_HARD_BLOCK_OPPOSITE_H1 and h1_score <= -5:
        hard_failures.append("strong_h1_opposite_technical_structure")
    if TECH_HARD_BLOCK_OPPOSITE_H4 and h4_score <= -5:
        hard_failures.append("strong_h4_opposite_technical_structure")
    if TECH_REQUIRE_H4_OR_D_ALIGNMENT and h4_score < 3 and d_score < 3:
        hard_failures.append("neither_h4_nor_daily_aligns")
    if aligned_timeframes < TECH_MIN_ALIGNED_TIMEFRAMES:
        hard_failures.append(f"not_enough_aligned_timeframes:{aligned_timeframes}<{TECH_MIN_ALIGNED_TIMEFRAMES}")
    if TECH_BLOCK_HIGH_SPREAD_ATR and spread_atr > TECH_MAX_SPREAD_ATR:
        hard_failures.append(f"spread_atr_hard_block:{spread_atr:.3f}>{TECH_MAX_SPREAD_ATR:.3f}")

    score = round(max(0.0, min(100.0, score)), 2)
    allow_trade = bool(score >= min_score and not hard_failures)
    decision = "PASS" if allow_trade else "BLOCK"
    if allow_trade:
        reason = "technical_review_passed"
    elif score < min_score:
        reason = f"technical_review_blocked:score={score:.2f}<{min_score:.2f}"
    else:
        reason = "technical_review_blocked:" + ";".join(hard_failures)

    return {
        "enabled": True,
        "decision": decision,
        "allow_trade": allow_trade,
        "technical_score": score,
        "minimum_required_score": min_score,
        "strong_score": TECH_STRONG_SCORE,
        "reason": reason,
        "pair": pair6,
        "instrument": instrument,
        "side_reviewed": side,
        "timeframes": tf_results,
        "aligned_timeframes": aligned_timeframes,
        "conflicting_timeframes": conflicting_timeframes,
        "h1_score_raw": h1_score,
        "h4_score_raw": h4_score,
        "daily_score_raw": d_score,
        "spread_atr": round(float(spread_atr), 4) if spread_atr else 0.0,
        "atr_pips": atr_pips,
        "spread_pips": spread_pips,
        "support_resistance_context": levels,
        "latest_candle_pattern": candle_pattern,
        "latest_candle_bias": candle_bias or "NEUTRAL",
        "supports": supports[:12],
        "conflicts": conflicts[:12],
        "risk_reasons": reasons[:12],
        "hard_failures": hard_failures,
    }


# ====================================================
# PAYLOAD MODELS
# ====================================================
class TVPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = "fx"
    symbol: str
    tf: Optional[str] = None
    t: int
    mid_o: float
    mid_h: float
    mid_l: float
    mid_c: float
    ema20: Optional[float] = 0.0
    ema50: Optional[float] = 0.0
    ema200: Optional[float] = 0.0
    rsi14: Optional[float] = 0.0
    adx14: Optional[float] = 0.0
    atr14: Optional[float] = 0.0
    macdh: Optional[float] = 0.0
    ret1: Optional[float] = 0.0
    ret3: Optional[float] = 0.0
    ret6: Optional[float] = 0.0
    ret12: Optional[float] = 0.0
    ret24: Optional[float] = None
    d20: Optional[float] = 0.0
    d50: Optional[float] = 0.0
    d200: Optional[float] = 0.0
    s20: Optional[float] = 0.0
    s50: Optional[float] = 0.0
    s200: Optional[float] = 0.0
    bbw: Optional[float] = 0.0
    spread_c: Optional[float] = None
    spread_atr: Optional[float] = None
    spread_pips: Optional[float] = None
    trend_regime: Optional[int] = 0
    vol_regime: Optional[int] = 0
    hr: Optional[int] = None
    hour: Optional[int] = None
    dow: Optional[int] = None
    month: Optional[int] = None
    session: Optional[int] = None
    plus_di14: Optional[float] = None
    minus_di14: Optional[float] = None
    ema50_h4: Optional[float] = None
    adx14_h4: Optional[float] = None
    ema20_d1: Optional[float] = None
    rsi14_d1: Optional[float] = None
    bid_c: Optional[float] = None
    ask_c: Optional[float] = None
    equity: Optional[float] = None
    nav: Optional[float] = None
    hint_side: Optional[str] = None
    force_decision: Optional[Literal["BUY", "SELL"]] = None
    force_units_abs: Optional[int] = None

class NewsEventPayload(BaseModel):
    title: Optional[str] = None
    currency: Optional[Any] = None
    currencies: Optional[Any] = None
    impact: Optional[str] = "HIGH"
    time_utc: Optional[Any] = None
    time: Optional[Any] = None
    start_utc: Optional[Any] = None
    end_utc: Optional[Any] = None
    start: Optional[Any] = None
    end: Optional[Any] = None
    before_min: Optional[int] = None
    after_min: Optional[int] = None
    source: Optional[str] = "api"

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
    symbol: Optional[str] = None
    broker_trade_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    client_trade_id: Optional[str] = None

def make_out(**kwargs) -> Dict[str, Any]:
    return kwargs

# ====================================================
# OANDA HELPERS
# ====================================================
def broker_can_close() -> bool:
    return bool(OANDA_TOKEN and OANDA_ACCOUNT_ID and OANDA_BASE_URL)

def oanda_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OANDA_TOKEN}",
        "Content-Type": "application/json",
    }

def oanda_request(method: str, path: str, json_body: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Dict[str, Any]:
    if not broker_can_close():
        return {"ok": False, "error": "Missing OANDA env vars"}

    url = f"{OANDA_BASE_URL}{path}"
    try:
        r = requests.request(method=method.upper(), url=url, headers=oanda_headers(), json=json_body, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = r.text
        if r.status_code in (200, 201):
            return {"ok": True, "status_code": r.status_code, "data": body}
        return {"ok": False, "status_code": r.status_code, "error": body}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

def get_oanda_position(instrument: str) -> Dict[str, Any]:
    return oanda_request("GET", f"/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}")

def get_position_side_trade_ids(instrument: str, side: str) -> Dict[str, Any]:
    side = normalize_side(side)
    res = get_oanda_position(instrument)
    if not res.get("ok"):
        return res

    body = res.get("data") or {}
    position = body.get("position") or {}
    branch = position.get("long") if side == "BUY" else position.get("short")
    trade_ids = []
    units = "0"
    if isinstance(branch, dict):
        trade_ids = [str(x) for x in (branch.get("tradeIDs") or []) if str(x).strip()]
        units = str(branch.get("units", "0"))
    return {
        "ok": True,
        "status_code": res.get("status_code"),
        "trade_ids": trade_ids,
        "units": units,
        "data": body,
    }

def close_oanda_trade_by_specifier(trade_specifier: str) -> Dict[str, Any]:
    if not trade_specifier:
        return {"ok": False, "error": "Missing trade_specifier"}
    return oanda_request(
        "PUT",
        f"/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_specifier}/close",
        json_body={"units": "ALL"},
    )

def close_oanda_position_side(instrument: str, side: str) -> Dict[str, Any]:
    side = normalize_side(side)
    payload = {"longUnits": "NONE", "shortUnits": "NONE"}
    if side == "BUY":
        payload["longUnits"] = "ALL"
    elif side == "SELL":
        payload["shortUnits"] = "ALL"
    else:
        return {"ok": False, "error": f"Unsupported side for position close: {side}"}

    return oanda_request(
        "PUT",
        f"/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}/close",
        json_body=payload,
    )

def tracked_open_count_for_instrument_side(instrument: str, side: str) -> int:
    side = normalize_side(side)
    count = 0
    for meta in _open_trade_meta.values():
        if meta.get("instrument") == instrument and normalize_side(meta.get("side")) == side:
            count += 1
    return count

def close_open_trade_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    instrument = str(meta.get("instrument", "")).upper()
    side = normalize_side(meta.get("side"))
    broker_trade_id = str(meta.get("broker_trade_id") or "").strip()
    client_trade_id = str(meta.get("client_trade_id") or "").strip()

    attempts: List[Dict[str, Any]] = []

    if broker_trade_id:
        res = close_oanda_trade_by_specifier(broker_trade_id)
        attempts.append({"method": "trade_id", "specifier": broker_trade_id, "result": res})
        if res.get("ok"):
            return {"ok": True, "method": "trade_id", "specifier": broker_trade_id, "data": res.get("data"), "attempts": attempts}

    if client_trade_id:
        spec = client_trade_id if client_trade_id.startswith("@") else f"@{client_trade_id}"
        res = close_oanda_trade_by_specifier(spec)
        attempts.append({"method": "client_trade_id", "specifier": spec, "result": res})
        if res.get("ok"):
            return {"ok": True, "method": "client_trade_id", "specifier": spec, "data": res.get("data"), "attempts": attempts}

    position_res = get_position_side_trade_ids(instrument, side)
    attempts.append({"method": "inspect_position", "instrument": instrument, "side": side, "result": position_res})

    if position_res.get("ok"):
        trade_ids = position_res.get("trade_ids") or []
        if len(trade_ids) == 1:
            spec = str(trade_ids[0])
            res = close_oanda_trade_by_specifier(spec)
            attempts.append({"method": "single_trade_from_position", "specifier": spec, "result": res})
            if res.get("ok"):
                return {"ok": True, "method": "single_trade_from_position", "specifier": spec, "data": res.get("data"), "attempts": attempts}

        if len(trade_ids) == 0:
            return {
                "ok": False,
                "error": "No open trade found on broker for instrument/side",
                "attempts": attempts,
            }

        if len(trade_ids) > 1:
            if AUTO_CLOSE_ALLOW_POSITION_FALLBACK and tracked_open_count_for_instrument_side(instrument, side) == 1:
                res = close_oanda_position_side(instrument, side)
                attempts.append({"method": "position_side_close", "instrument": instrument, "side": side, "result": res})
                if res.get("ok"):
                    return {"ok": True, "method": "position_side_close", "specifier": instrument, "data": res.get("data"), "attempts": attempts}
            return {
                "ok": False,
                "error": f"Ambiguous broker state: {len(trade_ids)} open trades found for {instrument} {side}",
                "attempts": attempts,
            }

    return {
        "ok": False,
        "error": "Could not close trade",
        "attempts": attempts,
    }

# ====================================================
# AUTO CLOSE
# ====================================================
def auto_close_worker() -> None:
    while True:
        try:
            if not AUTO_CLOSE_ENABLED:
                time.sleep(AUTO_CLOSE_CHECK_SECONDS)
                continue

            if not _open_trade_meta:
                time.sleep(AUTO_CLOSE_CHECK_SECONDS)
                continue

            if not broker_can_close():
                time.sleep(AUTO_CLOSE_CHECK_SECONDS)
                continue

            now = now_utc()

            for tracking_key, meta in list(_open_trade_meta.items()):
                opened_at = meta.get("opened_at_dt")
                if opened_at is None:
                    continue

                age_minutes = (now - opened_at).total_seconds() / 60.0
                if age_minutes < MAX_HOLD_MINUTES:
                    continue

                close_result = close_open_trade_meta(meta)

                if not close_result.get("ok"):
                    write_audit_row({
                        "ts": utc_ts(),
                        "pair": instrument_to_symbol(meta["instrument"]),
                        "instrument": meta["instrument"],
                        "symbol": instrument_to_symbol(meta["instrument"]),
                        "hint_side": meta["side"],
                        "model_version": "auto_close",
                        "avg_auc": None,
                        "pair_score": meta.get("pair_score"),
                        "equity_used": None,
                        "trend_regime": None,
                        "vol_regime": None,
                        "spread_pips": None,
                        "spread_atr": None,
                        "confidence": 0,
                        "side_prob": 0,
                        "p_up": 0,
                        "margin": 0,
                        "decision": "NONE",
                        "would_order": False,
                        "units": abs(int(meta["units_signed"])),
                        "units_signed": meta["units_signed"],
                        "sl_pips": None,
                        "tp_pips": None,
                        "sl_price": meta["sl_price"],
                        "tp_price": meta["tp_price"],
                        "why": f"AUTO_CLOSE_FAILED | tracking_key={tracking_key} | error={json.dumps(close_result, default=str)[:1500]}",
                    })
                    continue

                row = {
                    "instrument": meta["instrument"],
                    "side": meta["side"],
                    "units_signed": meta["units_signed"],
                    "entry_price": meta["entry_price"],
                    "sl_price": meta["sl_price"],
                    "tp_price": meta["tp_price"],
                    "status": "MANUAL",
                    "pnl": None,
                    "order_id": meta.get("order_id"),
                    "reason": f"Max hold time reached ({MAX_HOLD_MINUTES}m)",
                    "pair_score": meta.get("pair_score"),
                    "ts": utc_ts(),
                }
                write_trade_row(row)
                note_trade_closed(tracking_key)
                _open_trade_meta.pop(tracking_key, None)

        except Exception as e:
            write_audit_row({
                "ts": utc_ts(),
                "pair": "",
                "instrument": "",
                "symbol": "",
                "hint_side": "",
                "model_version": "auto_close",
                "avg_auc": None,
                "pair_score": None,
                "equity_used": None,
                "trend_regime": None,
                "vol_regime": None,
                "spread_pips": None,
                "spread_atr": None,
                "confidence": 0,
                "side_prob": 0,
                "p_up": 0,
                "margin": 0,
                "decision": "NONE",
                "would_order": False,
                "units": None,
                "units_signed": None,
                "sl_pips": None,
                "tp_pips": None,
                "sl_price": None,
                "tp_price": None,
                "why": f"AUTO_CLOSE_WORKER_EXCEPTION | {repr(e)}",
            })

        time.sleep(AUTO_CLOSE_CHECK_SECONDS)

# ====================================================
# APP
# ====================================================
app = FastAPI(title="FX Sniper Per Pair", version="8.1-h1-auto-registry-m15-safety-ai")

@app.on_event("startup")
def _startup() -> None:
    global NEWS_EVENTS
    init_db()
    seed_history_from_csv(DATA_DIR)
    NEWS_EVENTS = load_news_events()
    print(f"Loaded {len(NEWS_EVENTS)} H1 news blackout events")
    if AUTO_CLOSE_ENABLED:
        t = threading.Thread(target=auto_close_worker, daemon=True)
        t.start()

# ====================================================
# PREDICT
# ====================================================
def build_response_base(
    p: TVPayload,
    pair6: str,
    instrument: str,
    model_version: str,
    avg_auc: float,
    pair_score: Optional[float],
    equity_used: float,
    hint_side: str,
    conf: float = 0.0,
    side_prob: float = 0.0,
    p_up: float = 0.0,
    margin: float = 0.0,
) -> Dict[str, Any]:
    clean_symbol = pair6 or normalize_pair(p.symbol) or str(p.symbol).upper().replace("_", "")
    return {
        "ts": utc_ts(),
        "pair": pair6,
        "instrument": instrument,
        "symbol": clean_symbol,
        "raw_symbol": p.symbol,
        "hint_side": hint_side,
        "model_version": model_version,
        "avg_auc": avg_auc,
        "pair_score": pair_score,
        "equity_used": equity_used,
        "trend_regime": int(getattr(p, "trend_regime", 0) or 0),
        "vol_regime": int(getattr(p, "vol_regime", 0) or 0),
        "spread_pips": float(getattr(p, "spread_pips", 0.0) or 0.0),
        "spread_atr": float(getattr(p, "spread_atr", 0.0) or 0.0),
        "confidence": float(conf),
        "side_prob": float(side_prob),
        "p_up": float(p_up),
        "margin": float(margin),
    }

@app.post("/predict")
def predict(p: TVPayload):
    pair6 = normalize_pair(p.symbol)
    hint_side = normalize_side(getattr(p, "hint_side", "") or "")
    equity_used = get_equity_used(p)

    if pair6 is None or pair6 not in PAIR_MAP:
        out = make_out(
            decision="NONE",
            why="Symbol not allowed",
            would_order=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            **build_response_base(p, "", "", "", 0.0, None, equity_used, hint_side),
        )
        write_audit_row(out)
        return out

    instrument = pair_to_instrument(pair6)
    payload = p.model_dump()

    if payload.get("spread_pips") in (None, ""):
        ps = instrument_pip_size(instrument)
        bid_c = safe_float(payload.get("bid_c"), np.nan)
        ask_c = safe_float(payload.get("ask_c"), np.nan)
        spread_c = safe_float(payload.get("spread_c"), np.nan)
        if (not np.isfinite(spread_c) or spread_c <= 0) and np.isfinite(bid_c) and np.isfinite(ask_c):
            spread_c = ask_c - bid_c
            payload["spread_c"] = spread_c
        payload["spread_pips"] = (spread_c / ps) if np.isfinite(spread_c) and spread_c >= 0 else 0.0

    if payload.get("spread_atr") in (None, ""):
        atr14 = safe_float(payload.get("atr14"), 0.0)
        spread_c = safe_float(payload.get("spread_c"), 0.0)
        payload["spread_atr"] = (spread_c / atr14) if atr14 > 0 else 0.0

    b = BUNDLES.get(pair6)
    if not b:
        out = make_out(
            decision="NONE",
            why="Model not loaded for symbol",
            would_order=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            **build_response_base(p, pair6, instrument, "", 0.0, None, equity_used, hint_side),
        )
        write_audit_row(out)
        return out

    model_version = b["model_version"]
    avg_auc = safe_float(b.get("avg_auc"), 0.0)
    pair_score = safe_float(b.get("pair_score"), -1.0)
    if pair_score < 0:
        pair_score = safe_float(BUNDLES.get(pair6, {}).get("pair_score"), -1.0)
        if pair_score < 0:
            pair_score = compute_pair_score(instrument, avg_auc)

    bad_payload_reason = payload_sanity_checks(payload, instrument)
    if bad_payload_reason:
        out = make_out(
            decision="NONE",
            why=bad_payload_reason,
            would_order=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            **build_response_base(p, pair6, instrument, model_version, avg_auc, pair_score, equity_used, hint_side),
        )
        write_audit_row(out)
        return out

    gate = b.get("gate_override") or PAIR_GATES.get(instrument, DEFAULT_GATE)
    conf_gate = float(gate["conf"])
    margin_gate = float(gate["margin"])

    try:
        feat_order = b["feature_order"]
        feature_row = build_runtime_feature_row(payload, pair6, instrument, feat_order)
        market_context = build_external_market_context(pair6, instrument, hint_side, feature_row)
        X = pd.DataFrame([{f: feature_row.get(f, 0.0) for f in feat_order}], columns=feat_order)

        model = b["model"]
        proba = model.predict_proba(X)[0]
        p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])

        side = "BUY" if p_up >= 0.5 else "SELL"
        if p.force_decision in ("BUY", "SELL"):
            side = p.force_decision

        side_prob = p_up if side == "BUY" else (1.0 - p_up)
        conf = side_prob

        calibrator = b.get("calibrator")
        if calibrator is not None:
            try:
                conf = float(calibrator.predict([side_prob])[0])
            except Exception:
                conf = side_prob

        conf = max(0.0, min(1.0, conf))
        side_prob = max(0.0, min(1.0, side_prob))
        p_up = max(0.0, min(1.0, p_up))
        margin = float(abs(p_up - 0.5) * 2.0)

        base = build_response_base(
            p, pair6, instrument, model_version, avg_auc, pair_score,
            equity_used, hint_side, conf=conf, side_prob=side_prob, p_up=p_up, margin=margin
        )
        base["best_model"] = b.get("best_model", "joblib")
        base["model_source"] = b.get("model_source", "joblib")

        fingerprint = make_signal_fingerprint(instrument, side, p.t, float(p.mid_c), p.tf)

        stale_passed, stale_reason, stale_metrics = signal_staleness_guard(payload)
        if not stale_passed:
            out = make_out(decision="NONE", why=stale_reason, would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, decision_source="signal_staleness_guard_block", signal_id=fingerprint, **stale_metrics, **base)
            write_audit_row(out)
            return out

        noise_passed, noise_reason, noise_metrics = runtime_noise_filter(payload, feature_row, instrument, side)
        if not noise_passed:
            out = make_out(decision="NONE", why=noise_reason, would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, decision_source="noise_filter_block", signal_id=fingerprint, noise_filter_passed=False, noise_filter_reason=noise_reason, **stale_metrics, **noise_metrics, **base)
            write_audit_row(out)
            return out

        news_passed, news_reason, news_metrics = runtime_news_filter(pair6, payload)
        if not news_passed:
            out = make_out(decision="NONE", why=news_reason, would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, decision_source="news_filter_block", signal_id=fingerprint, noise_filter_passed=True, noise_filter_reason=noise_reason, **stale_metrics, **noise_metrics, **news_metrics, **base)
            write_audit_row(out)
            return out

        direction_passed, direction_reason, direction_metrics = direction_consensus_guard(payload, feature_row, instrument, side)
        if not direction_passed:
            out = make_out(decision="NONE", why=direction_reason, would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, decision_source="direction_confirmation_block", signal_id=fingerprint, noise_filter_passed=True, noise_filter_reason=noise_reason, news_filter_reason=news_reason, **stale_metrics, **noise_metrics, **news_metrics, **direction_metrics, **base)
            write_audit_row(out)
            return out

        entry_passed, entry_reason, entry_metrics = entry_reversal_guard(payload, feature_row, instrument, side)
        if not entry_passed:
            out = make_out(decision="NONE", why=entry_reason, would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, decision_source="entry_reversal_guard_block", signal_id=fingerprint, noise_filter_passed=True, noise_filter_reason=noise_reason, news_filter_reason=news_reason, **stale_metrics, **noise_metrics, **news_metrics, **direction_metrics, **entry_metrics, **base)
            write_audit_row(out)
            return out

        guard_metrics = {**stale_metrics, **noise_metrics, **news_metrics, **direction_metrics, **entry_metrics}

        disagree_conf_gate = PAIR_DISAGREE_CONF.get(instrument, DEFAULT_DISAGREE_CONF)
        hint_disagrees = hint_side in ("BUY", "SELL") and side != hint_side

        if hint_disagrees and conf < disagree_conf_gate:
            out = make_out(decision="NONE", why=f"Blocked disagreement: ML {side} vs hint {hint_side} (conf {conf:.2f} < {disagree_conf_gate:.2f})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out)
            return out

        if pair_score < MIN_PAIR_SCORE_TO_TRADE:
            out = make_out(decision="NONE", why=f"Pair blocked: {instrument} score {pair_score:.2f} < {MIN_PAIR_SCORE_TO_TRADE:.2f}", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out)
            return out

        would_order = (conf >= conf_gate) and (margin >= margin_gate)

        technical_review = {
            "enabled": TECHNICAL_REVIEW_ENABLED,
            "decision": "SKIPPED",
            "allow_trade": True,
            "technical_score": 0.0,
            "reason": "technical_review_not_run_below_model_gate" if not would_order else "technical_review_not_run",
        }
        if would_order and TECHNICAL_REVIEW_ENABLED:
            spread_atr_runtime = safe_float(payload.get("spread_atr"), 0.0)
            technical_review = run_h1_forex_technical_review(
                pair6=pair6,
                instrument=instrument,
                hint_side=side,
                decision_prob=conf,
                feature_row=feature_row,
                market_context=market_context,
                spread_atr=spread_atr_runtime,
            )
            if TECHNICAL_REVIEW_REQUIRED and not bool(technical_review.get("allow_trade", False)):
                out = make_out(
                    decision="NONE",
                    why=f"Technical review blocked: {technical_review.get('reason')}",
                    would_order=False,
                    units=None,
                    units_signed=None,
                    sl_pips=None,
                    tp_pips=None,
                    sl_price=None,
                    tp_price=None,
                    decision_source="technical_review_block",
                    signal_id=fingerprint,
                    noise_filter_passed=True,
                    noise_filter_reason=noise_reason,
                    news_filter_reason=news_reason,
                    technical_review_enabled=TECHNICAL_REVIEW_ENABLED,
                    technical_review_decision=technical_review.get("decision"),
                    technical_review_score=technical_review.get("technical_score"),
                    technical_review_min_score=technical_review.get("minimum_required_score"),
                    technical_review_reason=technical_review.get("reason"),
                    technical_aligned_timeframes=technical_review.get("aligned_timeframes"),
                    technical_conflicting_timeframes=technical_review.get("conflicting_timeframes"),
                    technical_h1_score=technical_review.get("h1_score_raw"),
                    technical_h4_score=technical_review.get("h4_score_raw"),
                    technical_daily_score=technical_review.get("daily_score_raw"),
                    technical_hard_failures=technical_review.get("hard_failures"),
                    technical_supports=technical_review.get("supports"),
                    technical_conflicts=technical_review.get("conflicts"),
                    technical_risk_reasons=technical_review.get("risk_reasons"),
                    **guard_metrics,
                    **base,
                )
                write_audit_row(out)
                return out

        if would_order and is_duplicate_signal(pair6, fingerprint):
            out = make_out(decision="NONE", why=f"Duplicate signal blocked for {instrument}", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out)
            return out

        if would_order and trades_today_total() >= MAX_TRADES_PER_DAY_TOTAL:
            out = make_out(decision="NONE", why=f"Daily lock: total max trades reached ({MAX_TRADES_PER_DAY_TOTAL})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out)
            return out

        if would_order and trades_today(pair6) >= MAX_TRADES_PER_DAY_PER_PAIR:
            out = make_out(decision="NONE", why=f"Daily lock: max trades for {instrument} reached", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out)
            return out

        if would_order and not can_open_trade():
            out = make_out(decision="NONE", why=f"Open trade cap reached ({MAX_OPEN_TRADES})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out)
            return out

        units_abs = units_signed = sl_pips = tp_pips = sl_price = tp_price = None
        why = f"Below sniper gate | conf={conf:.2f}/{conf_gate:.2f}, margin={margin:.2f}/{margin_gate:.2f}, hint={hint_side or 'NONE'}, pair_score={pair_score:.2f}"
        decision = "NONE"
        ai_review = {"enabled": False, "ai_verdict": "SKIPPED", "risk_score": 0, "reason": "AI review not run"}

        if would_order and AI_REVIEW_ENABLED:
            ai_context = {
                "pair": pair6,
                "instrument": instrument,
                "side": side,
                "model_version": model_version,
                "best_model": b.get("best_model", "joblib"),
                "model_source": b.get("model_source", "joblib"),
                "probability_up": p_up,
                "confidence": conf,
                "model_approval_probability": conf,
                "side_probability": side_prob,
                "margin": margin,
                "conf_gate": conf_gate,
                "margin_gate": margin_gate,
                "avg_auc": avg_auc,
                "pair_score": pair_score,
                "min_pair_score": MIN_PAIR_SCORE_TO_TRADE,
                "model_feature_source": {
                    "requested": MODEL_FEATURE_SOURCE,
                    "used": feature_row.get("_model_feature_source_used", "unknown"),
                    "ok": bool(feature_row.get("_model_feature_source_ok", False)),
                    "reason": feature_row.get("_model_feature_source_reason", ""),
                    "granularity": feature_row.get("_model_feature_granularity", MODEL_FEATURE_OANDA_GRANULARITY),
                    "time": feature_row.get("_model_feature_time", ""),
                    "last_close": feature_row.get("_model_feature_last_close", 0.0),
                    "candles": feature_row.get("_model_feature_candles", 0),
                },
                "model_features": {k: feature_row.get(k) for k in list(feature_row)[:100]},
                "features": {k: feature_row.get(k) for k in list(feature_row)[:100]},
                "external_market_context": market_context,
                "market_context": market_context,
                "payload_market": {k: payload.get(k) for k in ["mid_o","mid_h","mid_l","mid_c","ema20","ema50","ema200","rsi14","adx14","atr14","macdh","spread_pips","spread_atr","trend_regime","vol_regime"]},
                "risk_context": {**guard_metrics, "spread_pips": payload.get("spread_pips"), "spread_atr": payload.get("spread_atr"), "atr_pips": feature_row.get("atr_pips"), "technical_review": technical_review},
                "guards": guard_metrics,
                "risk": {"max_open_trades": MAX_OPEN_TRADES, "max_trades_day_total": MAX_TRADES_PER_DAY_TOTAL, "max_trades_day_pair": MAX_TRADES_PER_DAY_PER_PAIR, "equity_used": equity_used},
            }
            ai_review = review_signal_with_ai(ai_context)
            if AI_REVIEW_REQUIRE_APPROVAL and ai_review.get("ai_verdict") != "APPROVE":
                would_order = False
                why = f"AI review blocked: {ai_review.get('ai_verdict')} risk={ai_review.get('risk_score')} reason={ai_review.get('reason')}"

        if would_order:
            sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(
                side, float(p.mid_c), float(p.atr14), instrument,
                b["labeling"]["sl_atr"], b["labeling"]["tp_atr"]
            )
            units_abs = compute_units_dynamic(instrument, sl_pips, avg_auc, pair_score, equity_used, p.force_units_abs)
            units_signed = units_abs if side == "BUY" else -units_abs
            why = f"OK: {side} passed | conf={conf:.2f}/{conf_gate:.2f}, margin={margin:.2f}/{margin_gate:.2f}, hint={hint_side or 'NONE'}, pair_score={pair_score:.2f}, equity_used={equity_used:.2f}"
            decision = side

        out = make_out(
            decision=decision,
            why=why,
            would_order=bool(would_order),
            units=units_abs,
            units_signed=units_signed,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            sl_price=sl_price,
            tp_price=tp_price,
            signal_id=fingerprint,
            noise_filter_passed=True,
            noise_filter_reason=noise_reason,
            news_filter_reason=news_reason,
            ai_review_enabled=AI_REVIEW_ENABLED,
            ai_verdict=ai_review.get("ai_verdict"),
            ai_risk_score=ai_review.get("risk_score"),
            ai_reason=ai_review.get("reason"),
            ai_side_aware_provider=ai_review.get("provider"),
            ai_side_aware_decision=ai_review.get("decision"),
            ai_aligned_timeframes=ai_review.get("aligned_timeframes"),
            ai_conflicting_timeframes=ai_review.get("conflicting_timeframes"),
            ai_timeframe_trends=ai_review.get("timeframe_trends"),
            technical_review_enabled=TECHNICAL_REVIEW_ENABLED,
            technical_review_decision=technical_review.get("decision"),
            technical_review_score=technical_review.get("technical_score"),
            technical_review_min_score=technical_review.get("minimum_required_score"),
            technical_review_reason=technical_review.get("reason"),
            technical_aligned_timeframes=technical_review.get("aligned_timeframes"),
            technical_conflicting_timeframes=technical_review.get("conflicting_timeframes"),
            technical_h1_score=technical_review.get("h1_score_raw"),
            technical_h4_score=technical_review.get("h4_score_raw"),
            technical_daily_score=technical_review.get("daily_score_raw"),
            technical_hard_failures=technical_review.get("hard_failures"),
            technical_supports=technical_review.get("supports"),
            technical_conflicts=technical_review.get("conflicts"),
            technical_risk_reasons=technical_review.get("risk_reasons"),
            **guard_metrics,
            **base,
        )

        if would_order:
            remember_signal(pair6, fingerprint)
            inc_trade(pair6)

        write_audit_row(out)
        return out

    except Exception as e:
        out = make_out(
            decision="NONE",
            why=f"Prediction error: {repr(e)}",
            would_order=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            **build_response_base(p, pair6, instrument, model_version, avg_auc, pair_score, equity_used, hint_side),
        )
        write_audit_row(out)
        return out
    

# ====================================================
# TRADE EVENT
# ====================================================
@app.post("/trade_event")
def trade_event(t: TradeEvent):
    row = t.model_dump()
    if not row.get("ts"):
        row["ts"] = utc_ts()

    t.instrument = str(t.instrument).upper()
    t.side = normalize_side(t.side)
    if not row.get("symbol"):
        row["symbol"] = instrument_to_symbol(t.instrument)

    write_trade_row({
        "instrument": t.instrument,
        "side": t.side,
        "units_signed": t.units_signed,
        "entry_price": t.entry_price,
        "sl_price": t.sl_price,
        "tp_price": t.tp_price,
        "status": t.status,
        "pnl": t.pnl,
        "order_id": t.order_id,
        "reason": t.reason,
        "pair_score": t.pair_score,
        "ts": row["ts"],
    })

    tracking_key = make_tracking_key(t.order_id, t.broker_trade_id, t.client_trade_id, t.instrument, t.side, row["ts"])

    if t.status == "OPEN":
        note_trade_opened(tracking_key)
        opened_at_dt = dt.datetime.now(dt.timezone.utc)
        if row.get("ts"):
            try:
                opened_at_dt = pd.to_datetime(row["ts"], utc=True).to_pydatetime()
            except Exception:
                pass

        _open_trade_meta[str(tracking_key)] = {
            "tracking_key": str(tracking_key),
            "instrument": t.instrument,
            "symbol": row["symbol"],
            "side": t.side,
            "units_signed": t.units_signed,
            "entry_price": t.entry_price,
            "sl_price": t.sl_price,
            "tp_price": t.tp_price,
            "pair_score": t.pair_score,
            "opened_at_dt": opened_at_dt,
            "order_id": t.order_id,
            "broker_trade_id": t.broker_trade_id,
            "broker_order_id": t.broker_order_id,
            "client_trade_id": t.client_trade_id,
            "ts": row["ts"],
        }

    if t.status in ("CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"):
        keys = tracking_keys_for_close_event(t)
        if not keys:
            keys = [tracking_key]
        for key in keys:
            note_trade_closed(key)
            _open_trade_meta.pop(str(key), None)

    return {
        "ok": True,
        "open_trades": current_open_trade_count(),
        "status": t.status,
        "order_id": t.order_id,
        "tracking_key": tracking_key,
        "broker_trade_id": t.broker_trade_id,

    }



@app.get("/news_events")
def news_events():
    return {"ok": True, "ts": utc_ts(), "news_filter_enabled": NEWS_FILTER_ENABLED, "events_loaded": len(NEWS_EVENTS), "events": NEWS_EVENTS}

@app.post("/reload-news")
def reload_news():
    global NEWS_EVENTS
    NEWS_EVENTS = load_news_events()
    return {"ok": True, "message": "H1 news blackout events reloaded.", "events_loaded": len(NEWS_EVENTS), "events": NEWS_EVENTS, "ts": utc_ts()}

@app.post("/news_event")
def add_news_event(event: NewsEventPayload):
    global NEWS_EVENTS
    raw = event.model_dump(exclude_none=True)
    ev = normalize_news_event(raw)
    if not ev:
        return {"ok": False, "error": "invalid_news_event", "received": raw}
    NEWS_EVENTS.append(ev)
    deduped = []
    seen = set()
    for item in NEWS_EVENTS:
        key = (item.get("title"), tuple(item.get("currencies") or []), item.get("start_utc"), item.get("end_utc"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    NEWS_EVENTS = deduped
    save_news_events_to_file(NEWS_EVENTS)
    return {"ok": True, "event": ev, "events_loaded": len(NEWS_EVENTS), "news_events_file": NEWS_EVENTS_FILE}

# ====================================================
# STATS / EXPORT
# ====================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": utc_ts(),
        "pairs_loaded": len(BUNDLES),
        "joblib_pairs_loaded": len(JOBLIB_BUNDLES),
        "auto_registry_pairs_loaded": len(AUTO_REGISTRY_BUNDLES),
        "auto_registry_enabled": AUTO_MODEL_REGISTRY_ENABLED,
        "auto_registry_path": AUTO_REGISTRY_PATH,
        "auto_registry_overrides_joblib": AUTO_REGISTRY_OVERRIDES_JOBLIB,
        "pairs": sorted([pair_to_instrument(p) for p in BUNDLES.keys()]),
        "auto_registry_pairs": sorted([pair_to_instrument(p) for p in AUTO_REGISTRY_BUNDLES.keys()]),
        "db_path": DB_PATH,
        "auto_close_enabled": AUTO_CLOSE_ENABLED,
        "max_hold_minutes": MAX_HOLD_MINUTES,
        "auto_close_check_seconds": AUTO_CLOSE_CHECK_SECONDS,
        "auto_close_allow_position_fallback": AUTO_CLOSE_ALLOW_POSITION_FALLBACK,
        "market_context_enabled": MARKET_CONTEXT_ENABLED,
        "market_context_granularities": MARKET_CONTEXT_GRANULARITIES,
        "market_context_required": MARKET_CONTEXT_REQUIRED,
        "model_feature_source": MODEL_FEATURE_SOURCE,
        "model_feature_oanda_granularity": MODEL_FEATURE_OANDA_GRANULARITY,
        "model_feature_oanda_candle_count": MODEL_FEATURE_OANDA_CANDLE_COUNT,
        "model_feature_oanda_min_candles": MODEL_FEATURE_OANDA_MIN_CANDLES,
        "model_feature_fallback_to_alert": MODEL_FEATURE_FALLBACK_TO_ALERT,
        "auto_close_only_checks_oanda_when_open_trade_exists": True,
        "current_open_trades": current_open_trade_count(),
        "noise_filter_enabled": NOISE_FILTER_ENABLED,
        "min_noise_range_pips": MIN_NOISE_RANGE_PIPS,
        "min_noise_atr_pips": MIN_NOISE_ATR_PIPS,
        "min_body_range_ratio": MIN_BODY_RANGE_RATIO,
        "min_range_atr_ratio": MIN_RANGE_ATR_RATIO,
        "max_spread_range_ratio": MAX_SPREAD_RANGE_RATIO,
        "max_wick_body_ratio": MAX_WICK_BODY_RATIO,
        "news_filter_enabled": NEWS_FILTER_ENABLED,
        "news_block_before_min": NEWS_BLOCK_BEFORE_MIN,
        "news_block_after_min": NEWS_BLOCK_AFTER_MIN,
        "news_block_impacts": sorted(NEWS_BLOCK_IMPACTS),
        "news_events_file": NEWS_EVENTS_FILE,
        "news_events_loaded": len(NEWS_EVENTS),
        "signal_staleness_guard_enabled": SIGNAL_STALENESS_GUARD_ENABLED,
        "signal_max_age_seconds": SIGNAL_MAX_AGE_SECONDS,
        "direction_confirmation_enabled": DIRECTION_CONFIRMATION_ENABLED,
        "direction_confirmation_required": DIRECTION_CONFIRMATION_REQUIRED,
        "entry_reversal_guard_enabled": ENTRY_REVERSAL_GUARD_ENABLED,
        "ai_review_enabled": AI_REVIEW_ENABLED,
        "ai_review_provider": AI_REVIEW_PROVIDER,
        "ai_review_model": AI_REVIEW_MODEL,
        "ai_review_require_approval": AI_REVIEW_REQUIRE_APPROVAL,
        "ai_review_conditional_risk_score": AI_REVIEW_CONDITIONAL_RISK_SCORE,
        "ai_review_hard_block_score": AI_REVIEW_HARD_BLOCK_SCORE,
        "ai_review_min_model_prob": AI_REVIEW_MIN_MODEL_PROB,
        "ai_review_strong_model_prob": AI_REVIEW_STRONG_MODEL_PROB,
        "ai_review_max_spread_atr": AI_REVIEW_MAX_SPREAD_ATR,
        "ai_review_fallback_to_rules": AI_REVIEW_FALLBACK_TO_RULES,
        "technical_review_enabled": TECHNICAL_REVIEW_ENABLED,
        "technical_review_required": TECHNICAL_REVIEW_REQUIRED,
        "tech_min_score_for_buy": TECH_MIN_SCORE_FOR_BUY,
        "tech_min_score_for_sell": TECH_MIN_SCORE_FOR_SELL,
        "tech_min_aligned_timeframes": TECH_MIN_ALIGNED_TIMEFRAMES,
        "tech_require_h1_alignment": TECH_REQUIRE_H1_ALIGNMENT,
        "tech_require_h4_or_d_alignment": TECH_REQUIRE_H4_OR_D_ALIGNMENT,
        "tech_max_spread_atr": TECH_MAX_SPREAD_ATR,
        "risk_pct": RISK_PCT,
        "max_open_trades": MAX_OPEN_TRADES,
        "max_trades_per_day_total": MAX_TRADES_PER_DAY_TOTAL,
        "max_trades_per_day_per_pair": MAX_TRADES_PER_DAY_PER_PAIR,
    }

@app.get("/stats")
def stats():
    df = read_audit_df()
    if df.empty:
        return {"ok": True, "rows": 0, "would_order_count": 0, "decision_counts": {}, "pair_counts": {}, "last_ts": None}

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
        return {"ok": True, "trades": 0, "closed_trades": 0, "wins": 0, "losses": 0, "win_rate": None, "net_pnl": 0.0, "avg_pnl": None, "open_trades": current_open_trade_count()}

    closed = df[df["status"].isin(["CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"])].copy()
    if closed.empty or "pnl" not in closed.columns:
        return {"ok": True, "trades": int(len(df)), "closed_trades": int(len(closed)), "wins": 0, "losses": 0, "win_rate": None, "net_pnl": 0.0, "avg_pnl": None, "open_trades": current_open_trade_count()}

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
        pair6 = instrument_to_symbol(instrument)
        avg_auc = safe_float(BUNDLES.get(pair6, {}).get("avg_auc"), 0.0)
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

@app.get("/export/closed_trades.xlsx")
def export_closed_trades_xlsx():
    df = read_closed_trades_df().copy()
    out_path = os.path.join(LOG_DIR, "closed_trades_export.xlsx")

    if df.empty:
        blank = pd.DataFrame(columns=["ts", "instrument", "side", "units_signed", "entry_price", "sl_price", "tp_price", "status", "pnl", "order_id", "reason", "pair_score"])
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            blank.to_excel(writer, index=False, sheet_name="closed_trades")
        return FileResponse(out_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="closed_trades_export.xlsx")

    df["pnl"] = pd.to_numeric(df.get("pnl"), errors="coerce").fillna(0.0)
    summary = pd.DataFrame([{
        "closed_trades": int(len(df)),
        "wins": int((df["pnl"] > 0).sum()),
        "losses": int((df["pnl"] < 0).sum()),
        "win_rate": float((df["pnl"] > 0).sum() / len(df)) if len(df) else None,
        "net_pnl": float(df["pnl"].sum()),
        "avg_pnl": float(df["pnl"].mean()) if len(df) else None,
    }])

    by_pair = df.groupby("instrument", dropna=False).agg(
        trades=("instrument", "count"),
        wins=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).sum())),
        losses=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) < 0).sum())),
        net_pnl=("pnl", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())),
        avg_pnl=("pnl", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).mean())),
    ).reset_index()
    by_pair["win_rate"] = by_pair["wins"] / by_pair["trades"]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="closed_trades")
        summary.to_excel(writer, index=False, sheet_name="summary")
        by_pair.to_excel(writer, index=False, sheet_name="by_pair")

    return FileResponse(out_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="closed_trades_export.xlsx")

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

    cols = [c for c in ["ts", "pair", "instrument", "symbol", "raw_symbol", "hint_side", "decision", "confidence", "side_prob", "p_up", "margin", "pair_score", "equity_used", "units_signed", "sl_price", "tp_price", "would_order", "why"] if c in latest_rows.columns]
    table_html = latest_rows[cols].to_html(index=False, escape=False) if not latest_rows.empty else "<p>No audit data yet.</p>"

    by_pair_html = "<p>No audit data yet.</p>"
    if not audit_df.empty and "instrument" in audit_df.columns:
        by_pair = audit_df["instrument"].value_counts().rename_axis("instrument").reset_index(name="count")
        by_pair_html = by_pair.to_html(index=False)

    pnl_html = "<p>No trade data yet.</p>"
    if not trades_df.empty:
        pnl_cols = [c for c in ["ts", "instrument", "side", "units_signed", "status", "pnl", "reason", "order_id"] if c in trades_df.columns]
        pnl_html = trades_df.sort_values("ts", ascending=False).head(50)[pnl_cols].to_html(index=False, escape=False)

    closed = read_closed_trades_df()
    win_rate_txt = "N/A"
    if not closed.empty and "pnl" in closed.columns:
        pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
        win_rate_txt = f"{((pnl > 0).sum() / len(closed)):.2%}"

    html = f"""
    <html>
      <head>
        <title>FX Sniper Dashboard</title>
        <meta http-equiv="refresh" content="15">
      </head>
      <body style="font-family: Arial; padding: 24px;">
        <h1>FX Sniper Dashboard</h1>
        <div style="display:flex; gap:24px; margin-bottom:24px; flex-wrap:wrap;">
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Total predictions</h3><div style="font-size:28px;">{total_rows}</div></div>
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Would order</h3><div style="font-size:28px;">{would_count}</div></div>
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Blocked / NONE</h3><div style="font-size:28px;">{none_count}</div></div>
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Open trades tracked</h3><div style="font-size:28px;">{current_open_trade_count()}</div></div>
          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Win rate</h3><div style="font-size:28px;">{win_rate_txt}</div></div>
        </div>

        <h2>By pair</h2>{by_pair_html}
        <h2>Latest predictions</h2>{table_html}
        <h2>Latest trade events</h2>{pnl_html}

        <p style="margin-top:24px;">
          JSON endpoints:
          <a href="/health">/health</a> |
          <a href="/stats">/stats</a> |
          <a href="/pnl_stats">/pnl_stats</a> |
          <a href="/pair_stats">/pair_stats</a> |
          <a href="/export/closed_trades.xlsx">/export/closed_trades.xlsx</a>
        </p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)