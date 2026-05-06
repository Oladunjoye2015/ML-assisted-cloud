from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from collections import deque
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict

# ====================================================
# ENV
# ====================================================
MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOG_DIR = os.getenv("LOG_DIR", "logs")
DATA_DIR = os.getenv("DATA_DIR", "data/oanda_h1_ba_live")
DB_PATH = os.getenv("DB_PATH", os.path.join(LOG_DIR, "fx_trading.db"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

AUDIT_CSV = os.path.join(LOG_DIR, "audit.csv")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")

MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "360"))
AUTO_CLOSE_ENABLED = os.getenv("AUTO_CLOSE_ENABLED", "true").lower() == "true"
AUTO_CLOSE_CHECK_SECONDS = int(os.getenv("AUTO_CLOSE_CHECK_SECONDS", "1800"))
AUTO_CLOSE_ALLOW_POSITION_FALLBACK = os.getenv("AUTO_CLOSE_ALLOW_POSITION_FALLBACK", "false").lower() == "true"

OANDA_TOKEN = os.getenv("OANDA_TOKEN", "").strip()
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_BASE_URL = os.getenv("OANDA_BASE_URL", "https://api-fxtrade.oanda.com").strip().rstrip("/")

DEFAULT_GATE = {
    "conf": float(os.getenv("CONF_GATE", "0.56")),
    "margin": float(os.getenv("MARGIN_GATE", "0.04")),
}
DEFAULT_DISAGREE_CONF = float(os.getenv("DEFAULT_DISAGREE_CONF", "0.62"))

UNITS_JPY = int(os.getenv("UNITS_JPY", "1000"))
UNITS_NON_JPY = int(os.getenv("UNITS_NON_JPY", "2000"))
MIN_UNITS_JPY = int(os.getenv("MIN_UNITS_JPY", "100"))
MIN_UNITS_NON_JPY = int(os.getenv("MIN_UNITS_NON_JPY", "100"))
MAX_UNITS_JPY = int(os.getenv("MAX_UNITS_JPY", "3000"))
MAX_UNITS_NON_JPY = int(os.getenv("MAX_UNITS_NON_JPY", "5000"))

MAX_TRADES_PER_DAY_TOTAL = int(os.getenv("MAX_TRADES_PER_DAY_TOTAL", "5"))
MAX_TRADES_PER_DAY_PER_PAIR = int(os.getenv("MAX_TRADES_PER_DAY_PER_PAIR", "2"))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "2"))
DUP_WINDOW_SECONDS = int(os.getenv("DUP_WINDOW_SECONDS", "300"))

MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "3.5"))
MIN_ATR_NON_JPY = float(os.getenv("MIN_ATR_NON_JPY", "0.00005"))
MIN_ATR_JPY = float(os.getenv("MIN_ATR_JPY", "0.005"))

USE_EQUITY_SIZING = os.getenv("USE_EQUITY_SIZING", "true").lower() == "true"
DEFAULT_EQUITY = float(os.getenv("DEFAULT_EQUITY", "200"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))
DEFAULT_SL_ATR = float(os.getenv("DEFAULT_SL_ATR", "1.0"))
DEFAULT_TP_ATR = float(os.getenv("DEFAULT_TP_ATR", "1.3"))
BAR_HISTORY_LEN = int(os.getenv("BAR_HISTORY_LEN", "300"))

# ====================================================
# STRICT LIVE FILTERS
# ====================================================
STRICT_MODEL_FILTER_ENABLED = os.getenv("STRICT_MODEL_FILTER_ENABLED", "true").lower() == "true"
LIVE_MIN_AUC = float(os.getenv("LIVE_MIN_AUC", "0.52"))
LIVE_MIN_PRECISION = float(os.getenv("LIVE_MIN_PRECISION", "0.58"))
LIVE_MIN_TRADES_AT_GATE = int(os.getenv("LIVE_MIN_TRADES_AT_GATE", "50"))
LIVE_MIN_PAIR_SCORE = float(os.getenv("LIVE_MIN_PAIR_SCORE", "0.50"))

PRIMARY_LIVE_PAIRS = {x.strip().upper() for x in os.getenv("PRIMARY_LIVE_PAIRS", "GBPCHF,EURGBP,GBPJPY").split(",") if x.strip()}
EXPERIMENTAL_LIVE_PAIRS = {x.strip().upper() for x in os.getenv("EXPERIMENTAL_LIVE_PAIRS", "AUDCAD,EURCHF,USDCAD,USDJPY").split(",") if x.strip()}
ALLOW_EXPERIMENTAL_PAIRS = os.getenv("ALLOW_EXPERIMENTAL_PAIRS", "false").lower() == "true"
EXPERIMENTAL_SIZE_MULTIPLIER = float(os.getenv("EXPERIMENTAL_SIZE_MULTIPLIER", "0.50"))

# ====================================================
# FALLBACK / SHADOW MODEL SETTINGS
# ====================================================
FALLBACK_MODE_ENABLED = os.getenv("FALLBACK_MODE_ENABLED", "true").lower() == "true"
FALLBACK_CONF_EDGE = float(os.getenv("FALLBACK_CONF_EDGE", "0.04"))
FALLBACK_MARGIN_EDGE = float(os.getenv("FALLBACK_MARGIN_EDGE", "0.03"))
FALLBACK_MIN_AUC = float(os.getenv("FALLBACK_MIN_AUC", "0.53"))
FALLBACK_MIN_PRECISION = float(os.getenv("FALLBACK_MIN_PRECISION", "0.58"))
FALLBACK_MIN_TRADES_AT_GATE = int(os.getenv("FALLBACK_MIN_TRADES_AT_GATE", "100"))
FALLBACK_REQUIRE_HINT_AGREE = os.getenv("FALLBACK_REQUIRE_HINT_AGREE", "true").lower() == "true"

# ====================================================
# PAIRS
# ====================================================
PAIR_MAP: Dict[str, str] = {
    "AUDCAD": "AUD_CAD", "AUDJPY": "AUD_JPY", "AUDNZD": "AUD_NZD", "AUDUSD": "AUD_USD",
    "CADJPY": "CAD_JPY", "CHFJPY": "CHF_JPY",
    "EURCHF": "EUR_CHF", "EURGBP": "EUR_GBP", "EURJPY": "EUR_JPY", "EURUSD": "EUR_USD",
    "GBPCHF": "GBP_CHF", "GBPJPY": "GBP_JPY", "GBPUSD": "GBP_USD",
    "NZDJPY": "NZD_JPY", "NZDUSD": "NZD_USD",
    "USDCAD": "USD_CAD", "USDCHF": "USD_CHF", "USDJPY": "USD_JPY",
}
INSTRUMENT_TO_PAIR6 = {v: k for k, v in PAIR_MAP.items()}
JPY_INSTRUMENTS = {v for v in PAIR_MAP.values() if v.endswith("_JPY")}

_recent_signals: Dict[str, deque] = {}
_bar_history: Dict[str, deque] = {pair6: deque(maxlen=BAR_HISTORY_LEN) for pair6 in PAIR_MAP}
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
    return s if len(s) == 6 and s.isalpha() and s in PAIR_MAP else None

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

def normalize_side(side: Any) -> str:
    s = str(side or "").strip().upper()
    if s in ("BUY", "LONG"):
        return "BUY"
    if s in ("SELL", "SHORT"):
        return "SELL"
    return s

def make_tracking_key(order_id: Optional[str], broker_trade_id: Optional[str], client_trade_id: Optional[str], instrument: str, side: str, ts: Optional[str]) -> str:
    for candidate in (broker_trade_id, client_trade_id, order_id):
        if candidate not in (None, ""):
            return str(candidate)
    return f"{instrument}:{side}:{ts or utc_ts()}"

# ====================================================
# DB / CSV
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
            ts TEXT, instrument TEXT, side TEXT, units_signed INTEGER,
            entry_price REAL, sl_price REAL, tp_price REAL, status TEXT,
            pnl REAL, order_id TEXT, reason TEXT, pair_score REAL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_order_id ON trade_events(order_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_status ON trade_events(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_ts ON trade_events(ts)")
    conn.commit(); conn.close()

def insert_trade_event_db(row: Dict[str, Any]) -> None:
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO trade_events
        (ts, instrument, side, units_signed, entry_price, sl_price, tp_price, status, pnl, order_id, reason, pair_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (row.get("ts"), row.get("instrument"), row.get("side"), row.get("units_signed"), row.get("entry_price"), row.get("sl_price"), row.get("tp_price"), row.get("status"), row.get("pnl"), row.get("order_id"), row.get("reason"), row.get("pair_score")))
    conn.commit(); conn.close()

def read_trade_events_db() -> pd.DataFrame:
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM trade_events ORDER BY ts DESC", conn)
        if not df.empty and "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        return df
    finally:
        conn.close()

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
    return db_df if not db_df.empty else read_csv_df(TRADES_CSV)

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
    _check_daily_reset(); return _trade_count_today.get(pair6, 0)

def trades_today_total() -> int:
    _check_daily_reset(); return sum(_trade_count_today.values())

def inc_trade(pair6: str) -> None:
    _check_daily_reset(); _trade_count_today[pair6] = _trade_count_today.get(pair6, 0) + 1

def current_open_trade_count() -> int:
    return len(_open_trade_ids)

def can_open_trade() -> bool:
    return current_open_trade_count() < MAX_OPEN_TRADES

def note_trade_opened(tracking_key: Optional[str]) -> None:
    if tracking_key: _open_trade_ids.add(str(tracking_key))

def note_trade_closed(tracking_key: Optional[str]) -> None:
    if tracking_key and str(tracking_key) in _open_trade_ids:
        _open_trade_ids.remove(str(tracking_key))

# ====================================================
# DUP CHECK
# ====================================================
def make_signal_fingerprint(instrument: str, side: str, bar_time: int, mid_c: float, tf: Optional[str]) -> str:
    raw = {"instrument": instrument, "side": side, "bar_time": int(bar_time), "mid_c": round(float(mid_c), instrument_precision(instrument)), "tf": tf or ""}
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()

def is_duplicate_signal(pair6: str, fingerprint: str) -> bool:
    tnow = now_unix(); q = _recent_signals.setdefault(pair6, deque())
    while q and (tnow - q[0][0] > DUP_WINDOW_SECONDS): q.popleft()
    return any(fp == fingerprint for _, fp in q)

def remember_signal(pair6: str, fingerprint: str) -> None:
    _recent_signals.setdefault(pair6, deque()).append((now_unix(), fingerprint))

# ====================================================
# TECHNICALS / RUNTIME FEATURES
# ====================================================
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_sm = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_sm
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_sm
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean(), plus_di, minus_di

def ema_runtime(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi_runtime(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def update_bar_history(pair6: str, payload: Dict[str, Any]) -> pd.DataFrame:
    q = _bar_history.setdefault(pair6, deque(maxlen=BAR_HISTORY_LEN))
    t_val = safe_int(payload.get("t"))
    ts = pd.to_datetime(t_val, unit="s", utc=True, errors="coerce")
    if pd.isna(ts): ts = pd.Timestamp.utcnow()
    row = {"t": t_val, "time": ts, "mid_o": safe_float(payload.get("mid_o")), "mid_h": safe_float(payload.get("mid_h")), "mid_l": safe_float(payload.get("mid_l")), "mid_c": safe_float(payload.get("mid_c")), "volume": safe_float(payload.get("volume"), 0.0), "spread_c": safe_float(payload.get("spread_c"), 0.0)}
    if q and q[-1]["t"] == row["t"]: q[-1] = row
    else: q.append(row)
    return pd.DataFrame(list(q))

def seed_history_from_csv(data_dir: str) -> None:
    root = Path(data_dir)
    if not root.exists():
        print(f"WARNING: DATA_DIR not found for history seed: {data_dir}")
        return
    for pair6 in PAIR_MAP:
        path = root / f"{pair6}.csv"
        if not path.exists(): continue
        try:
            df = pd.read_csv(path).tail(BAR_HISTORY_LEN).copy()
            if "time" not in df.columns: continue
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"])
            has_mid = all(c in df.columns for c in ["mid_o", "mid_h", "mid_l", "mid_c"])
            has_bid_ask = all(c in df.columns for c in ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"])
            if not has_mid and not has_bid_ask: continue
            if not has_mid:
                df["mid_o"] = (pd.to_numeric(df["bid_o"], errors="coerce") + pd.to_numeric(df["ask_o"], errors="coerce")) / 2.0
                df["mid_h"] = (pd.to_numeric(df["bid_h"], errors="coerce") + pd.to_numeric(df["ask_h"], errors="coerce")) / 2.0
                df["mid_l"] = (pd.to_numeric(df["bid_l"], errors="coerce") + pd.to_numeric(df["ask_l"], errors="coerce")) / 2.0
                df["mid_c"] = (pd.to_numeric(df["bid_c"], errors="coerce") + pd.to_numeric(df["ask_c"], errors="coerce")) / 2.0
            if "spread_c" not in df.columns and has_bid_ask:
                df["spread_c"] = pd.to_numeric(df["ask_c"], errors="coerce") - pd.to_numeric(df["bid_c"], errors="coerce")
            if "spread_c" not in df.columns: df["spread_c"] = 0.0
            if "volume" not in df.columns: df["volume"] = 0.0
            q = deque(maxlen=BAR_HISTORY_LEN)
            for _, r in df.iterrows():
                q.append({"t": int(pd.Timestamp(r["time"]).timestamp()), "time": r["time"], "mid_o": float(r["mid_o"]), "mid_h": float(r["mid_h"]), "mid_l": float(r["mid_l"]), "mid_c": float(r["mid_c"]), "volume": safe_float(r.get("volume"), 0.0), "spread_c": safe_float(r.get("spread_c"), 0.0)})
            _bar_history[pair6] = q
            print(f"Seeded {pair6} history with {len(q)} bars.")
        except Exception as e:
            print(f"WARNING: failed to seed history for {pair6}: {e}")

def add_runtime_training_features(hist: pd.DataFrame, pair6: str, instrument: str) -> pd.DataFrame:
    df = hist.copy()
    for col in ["mid_o", "mid_h", "mid_l", "mid_c"]: df[col] = pd.to_numeric(df[col], errors="coerce")
    ps = instrument_pip_size(instrument); c = df["mid_c"]
    df["ret1"] = c.pct_change(1); df["ret2"] = c.pct_change(2); df["ret3"] = c.pct_change(3); df["ret6"] = c.pct_change(6); df["ret12"] = c.pct_change(12); df["ret24"] = c.pct_change(24)
    df["range_pips"] = (df["mid_h"] - df["mid_l"]) / ps; df["body_pips"] = (df["mid_c"] - df["mid_o"]) / ps
    df["upper_wick_pips"] = (df["mid_h"] - df[["mid_o", "mid_c"]].max(axis=1)) / ps
    df["lower_wick_pips"] = (df[["mid_o", "mid_c"]].min(axis=1) - df["mid_l"]) / ps
    df["ema20"] = ema_runtime(c, 20); df["ema50"] = ema_runtime(c, 50); df["ema100"] = ema_runtime(c, 100); df["ema200"] = ema_runtime(c, 200)
    df["dist_ema20_pips"] = (c - df["ema20"]) / ps; df["dist_ema50_pips"] = (c - df["ema50"]) / ps; df["dist_ema100_pips"] = (c - df["ema100"]) / ps; df["dist_ema200_pips"] = (c - df["ema200"]) / ps
    df["ema20_slope"] = df["ema20"].diff(3) / ps; df["ema50_slope"] = df["ema50"].diff(6) / ps; df["ema200_slope"] = df["ema200"].diff(12) / ps
    df["rsi14"] = rsi_runtime(c, 14); df["rsi7"] = rsi_runtime(c, 7)
    df["atr14"] = atr(df["mid_h"], df["mid_l"], df["mid_c"], 14); df["atr14_pips"] = df["atr14"] / ps
    adx_series, _, _ = adx(df["mid_h"], df["mid_l"], df["mid_c"], 14); df["adx14"] = adx_series.fillna(20)
    ema12 = ema_runtime(c, 12); ema26 = ema_runtime(c, 26); df["macd"] = ema12 - ema26; df["macd_signal"] = ema_runtime(df["macd"], 9); df["macdh"] = df["macd"] - df["macd_signal"]; df["macdh_pips"] = df["macdh"] / ps
    roll20 = c.rolling(20); df["bb_mid"] = roll20.mean(); df["bb_std"] = roll20.std(); df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]; df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width_pips"] = (df["bb_upper"] - df["bb_lower"]) / ps; df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    if "spread_c" in df.columns:
        df["spread_c"] = pd.to_numeric(df["spread_c"], errors="coerce").fillna(0.0); df["spread_pips"] = df["spread_c"] / ps
    else: df["spread_pips"] = 0.0
    df["spread_atr"] = df["spread_pips"] / df["atr14_pips"].replace(0, np.nan)
    dt_series = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["hour_utc"] = dt_series.dt.hour.fillna(0); df["day_of_week"] = dt_series.dt.dayofweek.fillna(0); df["month"] = dt_series.dt.month.fillna(0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_utc"] / 24); df["hour_cos"] = np.cos(2 * np.pi * df["hour_utc"] / 24); df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7); df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["trend_up"] = (df["ema20"] > df["ema50"]).astype(int); df["trend_down"] = (df["ema20"] < df["ema50"]).astype(int); df["price_above_ema200"] = (c > df["ema200"]).astype(int)
    if "volume" not in df.columns: df["volume"] = 0.0
    return df

def build_runtime_feature_row(payload: Dict[str, Any], pair6: str, instrument: str, feat_order: List[str]) -> Dict[str, Any]:
    ps = instrument_pip_size(instrument)
    bid_c = safe_float(payload.get("bid_c"), np.nan); ask_c = safe_float(payload.get("ask_c"), np.nan); spread_c = safe_float(payload.get("spread_c"), np.nan)
    if (not np.isfinite(spread_c) or spread_c <= 0) and np.isfinite(bid_c) and np.isfinite(ask_c) and ask_c >= bid_c:
        spread_c = ask_c - bid_c
    if not np.isfinite(spread_c): spread_c = 0.0
    payload["spread_c"] = spread_c
    if safe_float(payload.get("spread_pips"), np.nan) != safe_float(payload.get("spread_pips"), np.nan):
        payload["spread_pips"] = spread_c / ps if ps > 0 else 0.0
    hist = update_bar_history(pair6, payload)
    if "spread_c" not in hist.columns: hist["spread_c"] = 0.0
    if "volume" not in hist.columns: hist["volume"] = 0.0
    if len(hist) > 0:
        hist.loc[hist.index[-1], "spread_c"] = spread_c
        hist.loc[hist.index[-1], "volume"] = safe_float(payload.get("volume"), 0.0)
    feat_df = add_runtime_training_features(hist, pair6, instrument)
    last = feat_df.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict()
    return {f: safe_float(last.get(f), safe_float(payload.get(f), 0.0)) for f in feat_order}

# ====================================================
# SIZING / SL TP / SANITY
# ====================================================
def payload_sanity_checks(payload: Dict[str, Any], instrument: str) -> Optional[str]:
    ps = instrument_pip_size(instrument)
    spread_pips = safe_float(payload.get("spread_pips"), np.nan)
    if not np.isfinite(spread_pips):
        spread_c = safe_float(payload.get("spread_c"), 0.0); spread_pips = spread_c / ps if ps > 0 else 0.0; payload["spread_pips"] = spread_pips
    if spread_pips > MAX_SPREAD_PIPS: return f"Spread too high: {spread_pips} pips > {MAX_SPREAD_PIPS}"
    atr14 = safe_float(payload.get("atr14"), 0.0)
    if atr14 > 0 and atr14 < min_atr_for_instrument(instrument): return f"ATR too small: {atr14}"
    mid_l = safe_float(payload.get("mid_l")); mid_c = safe_float(payload.get("mid_c")); mid_h = safe_float(payload.get("mid_h")); mid_o = safe_float(payload.get("mid_o"))
    if not (mid_l <= mid_c <= mid_h): return "Bad payload: mid_c not between mid_l and mid_h"
    if not (mid_l <= mid_o <= mid_h): return "Bad payload: mid_o not between mid_l and mid_h"
    if mid_h < mid_l: return "Bad payload: mid_h < mid_l"
    if safe_float(payload.get("spread_pips"), 0.0) < 0: return "Bad payload: negative spread_pips"
    return None

def compute_units_dynamic(instrument: str, sl_pips: float, avg_auc: float, pair_score: float, equity_used: float, force_units_abs: Optional[int] = None) -> int:
    if force_units_abs is not None: return max(1, abs(int(force_units_abs)))
    if sl_pips is None or sl_pips <= 0: return 0
    base = base_units_for_instrument(instrument)
    if USE_EQUITY_SIZING:
        risk_cap = equity_used * RISK_PCT; risk_per_1000 = float(sl_pips) * pip_value_per_1000(instrument)
        if risk_per_1000 > 0: base = int((risk_cap / risk_per_1000) * 1000)
    if pair_score >= 0.80: base = int(base * 1.20)
    elif pair_score >= 0.65: base = int(base * 1.10)
    elif pair_score < 0.50: base = int(base * 0.70)
    if avg_auc >= 0.57: base = int(base * 1.05)
    elif avg_auc < 0.54: base = int(base * 0.90)
    return min(max_units_for_instrument(instrument), max(min_units_for_instrument(instrument), base))

def _round_down_to_pip(price: float, pip: float) -> float: return math.floor(price / pip) * pip

def _round_up_to_pip(price: float, pip: float) -> float: return math.ceil(price / pip) * pip

def compute_sl_tp_prices(side: str, mid_c: float, atr14: float, instrument: str, sl_atr: float, tp_atr: float, min_dist_pips: float = 5.0) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if side not in ("BUY", "SELL"): return None, None, None, None
    pip = instrument_pip_size(instrument); atrv = max(float(atr14), pip)
    sl_dist = max(sl_atr * atrv, min_dist_pips * pip); tp_dist = max(tp_atr * atrv, min_dist_pips * pip)
    if side == "BUY":
        sl_price = _round_down_to_pip(mid_c - sl_dist, pip); tp_price = _round_up_to_pip(mid_c + tp_dist, pip)
        if sl_price >= mid_c: sl_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)
        if tp_price <= mid_c: tp_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
    else:
        sl_price = _round_up_to_pip(mid_c + sl_dist, pip); tp_price = _round_down_to_pip(mid_c - tp_dist, pip)
        if sl_price <= mid_c: sl_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
        if tp_price >= mid_c: tp_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)
    sl_str = format_oanda_price(sl_price, instrument); tp_str = format_oanda_price(tp_price, instrument); mid_str = format_oanda_price(mid_c, instrument)
    return abs(float(mid_str) - float(sl_str)) / pip, abs(float(tp_str) - float(mid_str)) / pip, sl_str, tp_str

# ====================================================
# MODEL LOADING + FALLBACK
# ====================================================
def load_json_safe(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists(): return default
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return default

def get_best_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    best = metrics.get("best") or {}
    return best if isinstance(best, dict) else {}

def pair_passes_static_training_filter(pair6: str, metrics: Dict[str, Any]) -> Tuple[bool, str]:
    if not STRICT_MODEL_FILTER_ENABLED: return True, "strict_filter_disabled"
    best = get_best_metrics(metrics)
    model_tradable = bool(metrics.get("tradable", False)) and bool(best.get("tradable", False))
    auc = safe_float(best.get("auc"), 0.0); precision = safe_float(best.get("precision_at_gate"), 0.0); trades = safe_int(best.get("trades_at_gate"), 0); pair_score = safe_float(best.get("pair_score"), 0.0)
    reasons = []
    if not model_tradable: reasons.append("training_marked_not_tradable")
    if auc < LIVE_MIN_AUC: reasons.append(f"auc_too_low:{auc:.4f}<{LIVE_MIN_AUC:.4f}")
    if precision < LIVE_MIN_PRECISION: reasons.append(f"precision_too_low:{precision:.4f}<{LIVE_MIN_PRECISION:.4f}")
    if trades < LIVE_MIN_TRADES_AT_GATE: reasons.append(f"trades_too_low:{trades}<{LIVE_MIN_TRADES_AT_GATE}")
    if pair_score < LIVE_MIN_PAIR_SCORE: reasons.append(f"pair_score_too_low:{pair_score:.4f}<{LIVE_MIN_PAIR_SCORE:.4f}")
    if reasons: return False, "; ".join(reasons)
    if pair6 in PRIMARY_LIVE_PAIRS: return True, "primary_live_pair_passed"
    if pair6 in EXPERIMENTAL_LIVE_PAIRS:
        return (True, "experimental_pair_allowed") if ALLOW_EXPERIMENTAL_PAIRS else (False, "experimental_pair_blocked_until_enabled")
    return False, "pair_not_in_primary_or_experimental_live_list"

def load_new_model_bundle(pair_dir: Path) -> Optional[Dict[str, Any]]:
    pair6 = pair_dir.name.upper().replace("_", "")
    if pair6 not in PAIR_MAP: return None
    feature_columns = load_json_safe(pair_dir / "feature_columns.json", [])
    thresholds = load_json_safe(pair_dir / "thresholds.json", {})
    model_type_json = load_json_safe(pair_dir / "best_model_type.json", {})
    metrics = load_json_safe(pair_dir / "metrics.json", {})
    if not feature_columns: return None
    model_type = str(model_type_json.get("model_type") or thresholds.get("model_name") or "").strip()
    if not model_type or model_type == "neural_tcn": return None
    model_path = pair_dir / "best_model.pkl"
    if not model_path.exists(): return None
    try: model = joblib.load(model_path)
    except Exception as e:
        print(f"WARNING: could not load model for {pair6}: {e}"); return None
    candidate_models: Dict[str, Any] = {}
    candidate_metrics_raw = load_json_safe(pair_dir / "candidate_metrics.json", {})
    candidate_metrics = candidate_metrics_raw if isinstance(candidate_metrics_raw, dict) else {}
    candidate_dir = pair_dir / "candidate_models"
    if candidate_dir.exists():
        for model_file in sorted(candidate_dir.glob("*.pkl")):
            try: candidate_models[model_file.stem] = joblib.load(model_file)
            except Exception as e: print(f"WARNING: could not load candidate {pair6}/{model_file.stem}: {e}")
    best = get_best_metrics(metrics)
    passed_filter, filter_reason = pair_passes_static_training_filter(pair6, metrics)
    return {
        "pair6": pair6, "instrument": pair_to_instrument(pair6), "model": model, "model_type": model_type,
        "candidate_models": candidate_models, "candidate_metrics": candidate_metrics,
        "feature_order": list(feature_columns), "thresholds": thresholds, "metrics": metrics, "best": best,
        "avg_auc": safe_float(best.get("auc"), 0.0), "precision_at_gate": safe_float(best.get("precision_at_gate"), 0.0), "trades_at_gate": safe_int(best.get("trades_at_gate"), 0), "training_pair_score": safe_float(best.get("pair_score"), 0.0), "training_tradable": bool(metrics.get("tradable", False)),
        "static_filter_passed": passed_filter, "static_filter_reason": filter_reason,
        "labeling": {"sl_atr": safe_float(thresholds.get("sl_atr"), DEFAULT_SL_ATR), "tp_atr": safe_float(thresholds.get("tp_atr"), DEFAULT_TP_ATR)},
        "gate": safe_float(thresholds.get("gate"), DEFAULT_GATE["conf"]), "margin_gate": safe_float(thresholds.get("margin_gate"), DEFAULT_GATE["margin"]),
        "model_version": f"{pair6}:{model_type}", "_bundle_path": str(pair_dir),
    }

def load_bundles(models_dir: str) -> Dict[str, Dict[str, Any]]:
    bundles: Dict[str, Dict[str, Any]] = {}; root = Path(models_dir)
    if not root.exists(): print(f"WARNING: MODELS_DIR does not exist: {models_dir}"); return bundles
    for pair_dir in sorted(root.iterdir()):
        if pair_dir.is_dir():
            b = load_new_model_bundle(pair_dir)
            if b: bundles[b["pair6"]] = b
    print(f"Loaded {len(bundles)} model bundles from {models_dir}")
    return bundles

def candidate_metric_is_good_for_fallback(metric: Dict[str, Any]) -> Tuple[bool, str]:
    auc = safe_float(metric.get("auc"), 0.0); precision = safe_float(metric.get("precision_at_gate"), 0.0); trades = safe_int(metric.get("trades_at_gate"), 0); tradable = bool(metric.get("tradable", False))
    reasons = []
    if not tradable: reasons.append("candidate_not_tradable")
    if auc < FALLBACK_MIN_AUC: reasons.append(f"auc_too_low:{auc:.4f}<{FALLBACK_MIN_AUC:.4f}")
    if precision < FALLBACK_MIN_PRECISION: reasons.append(f"precision_too_low:{precision:.4f}<{FALLBACK_MIN_PRECISION:.4f}")
    if trades < FALLBACK_MIN_TRADES_AT_GATE: reasons.append(f"trades_too_low:{trades}<{FALLBACK_MIN_TRADES_AT_GATE}")
    return (False, "; ".join(reasons)) if reasons else (True, "candidate_passed_fallback_quality")

def predict_model_probability(model: Any, X: pd.DataFrame) -> float:
    proba = model.predict_proba(X)[0]
    return float(proba[1]) if len(proba) > 1 else float(proba[0])

def evaluate_candidate_models_for_fallback(b: Dict[str, Any], X: pd.DataFrame, primary_model_type: str, primary_conf_gate: float, primary_margin_gate: float, hint_side: str) -> Dict[str, Any]:
    if not FALLBACK_MODE_ENABLED: return {"fallback_allowed": False, "fallback_used": False, "fallback_reason": "fallback_mode_disabled", "candidate_votes": {}}
    candidate_models = b.get("candidate_models") or {}; candidate_metrics = b.get("candidate_metrics") or {}
    if not candidate_models: return {"fallback_allowed": False, "fallback_used": False, "fallback_reason": "no_candidate_models_loaded", "candidate_votes": {}}
    fallback_conf_gate = primary_conf_gate + FALLBACK_CONF_EDGE; fallback_margin_gate = primary_margin_gate + FALLBACK_MARGIN_EDGE
    candidate_votes = {}; best_candidate = None
    for model_name, model in candidate_models.items():
        if model_name == primary_model_type: continue
        metric = candidate_metrics.get(model_name, {})
        quality_ok, quality_reason = candidate_metric_is_good_for_fallback(metric)
        try: p_up = predict_model_probability(model, X)
        except Exception as e:
            candidate_votes[model_name] = {"ok": False, "reason": f"prediction_error:{repr(e)}"}; continue
        side = "BUY" if p_up >= 0.5 else "SELL"; side_prob = p_up if side == "BUY" else 1.0 - p_up; margin = abs(p_up - 0.5) * 2.0
        direction_ok = not (FALLBACK_REQUIRE_HINT_AGREE and hint_side in ("BUY", "SELL")) or side == hint_side
        gate_ok = side_prob >= fallback_conf_gate and margin >= fallback_margin_gate
        vote = {"ok": bool(quality_ok and direction_ok and gate_ok), "side": side, "p_up": float(p_up), "confidence": float(side_prob), "margin": float(margin), "quality_ok": quality_ok, "quality_reason": quality_reason, "direction_ok": direction_ok, "gate_ok": gate_ok, "fallback_conf_gate": fallback_conf_gate, "fallback_margin_gate": fallback_margin_gate, "auc": safe_float(metric.get("auc"), 0.0), "precision_at_gate": safe_float(metric.get("precision_at_gate"), 0.0), "trades_at_gate": safe_int(metric.get("trades_at_gate"), 0)}
        candidate_votes[model_name] = vote
        if vote["ok"]:
            rec = {"model_name": model_name, "side": side, "p_up": float(p_up), "confidence": float(side_prob), "margin": float(margin), "metric": metric}
            if best_candidate is None or rec["confidence"] > best_candidate["confidence"] or (math.isclose(rec["confidence"], best_candidate["confidence"]) and safe_float(metric.get("precision_at_gate"), 0.0) > safe_float(best_candidate["metric"].get("precision_at_gate"), 0.0)):
                best_candidate = rec
    if best_candidate is None:
        return {"fallback_allowed": False, "fallback_used": False, "fallback_reason": "no_candidate_passed_strict_fallback_rules", "candidate_votes": candidate_votes}
    return {"fallback_allowed": True, "fallback_used": True, "fallback_reason": "fallback_candidate_passed_strict_rules", "fallback_model": best_candidate["model_name"], "fallback_side": best_candidate["side"], "fallback_p_up": best_candidate["p_up"], "fallback_confidence": best_candidate["confidence"], "fallback_margin": best_candidate["margin"], "candidate_votes": candidate_votes}

BUNDLES = load_bundles(MODELS_DIR)

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
    bid_c: Optional[float] = None
    ask_c: Optional[float] = None
    spread_c: Optional[float] = None
    spread_atr: Optional[float] = None
    spread_pips: Optional[float] = None
    volume: Optional[float] = 0.0
    atr14: Optional[float] = 0.0
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
    symbol: Optional[str] = None
    broker_trade_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    client_trade_id: Optional[str] = None

def make_out(**kwargs) -> Dict[str, Any]: return kwargs

# ====================================================
# OANDA AUTO-CLOSE HELPERS
# ====================================================
def broker_can_close() -> bool: return bool(OANDA_TOKEN and OANDA_ACCOUNT_ID and OANDA_BASE_URL)
def oanda_headers() -> Dict[str, str]: return {"Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"}
def oanda_request(method: str, path: str, json_body: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Dict[str, Any]:
    if not broker_can_close(): return {"ok": False, "error": "Missing OANDA env vars"}
    try:
        r = requests.request(method=method.upper(), url=f"{OANDA_BASE_URL}{path}", headers=oanda_headers(), json=json_body, timeout=timeout)
        try: body = r.json()
        except Exception: body = r.text
        return {"ok": r.status_code in (200, 201), "status_code": r.status_code, "data" if r.status_code in (200, 201) else "error": body}
    except Exception as e:
        return {"ok": False, "error": repr(e)}
def get_oanda_position(instrument: str) -> Dict[str, Any]: return oanda_request("GET", f"/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}")
def close_oanda_trade_by_specifier(trade_specifier: str) -> Dict[str, Any]:
    return {"ok": False, "error": "Missing trade_specifier"} if not trade_specifier else oanda_request("PUT", f"/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_specifier}/close", {"units": "ALL"})
def close_oanda_position_side(instrument: str, side: str) -> Dict[str, Any]:
    side = normalize_side(side); payload = {"longUnits": "NONE", "shortUnits": "NONE"}
    if side == "BUY": payload["longUnits"] = "ALL"
    elif side == "SELL": payload["shortUnits"] = "ALL"
    else: return {"ok": False, "error": f"Unsupported side: {side}"}
    return oanda_request("PUT", f"/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}/close", payload)

def auto_close_worker() -> None:
    while True:
        try:
            if not AUTO_CLOSE_ENABLED or not broker_can_close() or not _open_trade_meta:
                time.sleep(AUTO_CLOSE_CHECK_SECONDS); continue
            now = now_utc()
            for tracking_key, meta in list(_open_trade_meta.items()):
                opened_at = meta.get("opened_at_dt")
                if opened_at is None or (now - opened_at).total_seconds() / 60.0 < MAX_HOLD_MINUTES: continue
                spec = meta.get("broker_trade_id") or (f"@{meta.get('client_trade_id')}" if meta.get("client_trade_id") else None)
                res = close_oanda_trade_by_specifier(spec) if spec else close_oanda_position_side(meta["instrument"], meta["side"])
                if res.get("ok"):
                    row = {"instrument": meta["instrument"], "side": meta["side"], "units_signed": meta["units_signed"], "entry_price": meta["entry_price"], "sl_price": meta["sl_price"], "tp_price": meta["tp_price"], "status": "MANUAL", "pnl": None, "order_id": meta.get("order_id"), "reason": f"Max hold time reached ({MAX_HOLD_MINUTES}m)", "pair_score": meta.get("pair_score"), "ts": utc_ts()}
                    write_trade_row(row); note_trade_closed(tracking_key); _open_trade_meta.pop(tracking_key, None)
                else:
                    write_audit_row({"ts": utc_ts(), "pair": instrument_to_symbol(meta["instrument"]), "instrument": meta["instrument"], "symbol": instrument_to_symbol(meta["instrument"]), "hint_side": meta["side"], "model_version": "auto_close", "model_type": "auto_close", "avg_auc": None, "pair_score": meta.get("pair_score"), "equity_used": None, "spread_pips": None, "spread_atr": None, "confidence": 0, "side_prob": 0, "p_up": 0, "margin": 0, "decision": "NONE", "would_order": False, "order_allowed": False, "units": abs(int(meta["units_signed"])), "units_signed": meta["units_signed"], "sl_pips": None, "tp_pips": None, "sl_price": meta["sl_price"], "tp_price": meta["tp_price"], "why": f"AUTO_CLOSE_FAILED | tracking_key={tracking_key} | error={json.dumps(res, default=str)[:1500]}"})
        except Exception as e:
            print(f"AUTO_CLOSE_WORKER_EXCEPTION: {e}")
        time.sleep(AUTO_CLOSE_CHECK_SECONDS)

# ====================================================
# APP
# ====================================================
app = FastAPI(title="FX 1H Auto Model Server", version="9.0-fallback-candidate-models")

@app.on_event("startup")
def _startup() -> None:
    init_db(); seed_history_from_csv(DATA_DIR)
    if AUTO_CLOSE_ENABLED:
        threading.Thread(target=auto_close_worker, daemon=True).start()

def build_response_base(p: TVPayload, pair6: str, instrument: str, model_version: str, avg_auc: float, pair_score: Optional[float], equity_used: float, hint_side: str, conf: float = 0.0, side_prob: float = 0.0, p_up: float = 0.0, margin: float = 0.0) -> Dict[str, Any]:
    clean_symbol = pair6 or normalize_pair(p.symbol) or str(p.symbol).upper().replace("_", "")
    return {"ts": utc_ts(), "pair": pair6, "instrument": instrument, "symbol": clean_symbol, "raw_symbol": p.symbol, "hint_side": hint_side, "model_version": model_version, "avg_auc": avg_auc, "pair_score": pair_score, "equity_used": equity_used, "spread_pips": float(getattr(p, "spread_pips", 0.0) or 0.0), "spread_atr": float(getattr(p, "spread_atr", 0.0) or 0.0), "confidence": float(conf), "side_prob": float(side_prob), "p_up": float(p_up), "margin": float(margin)}

@app.post("/predict")
def predict(p: TVPayload):
    pair6 = normalize_pair(p.symbol); hint_side = normalize_side(getattr(p, "hint_side", "") or ""); equity_used = get_equity_used(p)
    if pair6 is None or pair6 not in PAIR_MAP:
        out = make_out(decision="NONE", why="Symbol not allowed", would_order=False, order_allowed=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, model_type="none", **build_response_base(p, "", "", "", 0.0, None, equity_used, hint_side)); write_audit_row(out); return out
    instrument = pair_to_instrument(pair6); payload = p.model_dump(); ps = instrument_pip_size(instrument)
    if payload.get("spread_pips") in (None, ""):
        bid_c = safe_float(payload.get("bid_c"), np.nan); ask_c = safe_float(payload.get("ask_c"), np.nan); spread_c = safe_float(payload.get("spread_c"), np.nan)
        if (not np.isfinite(spread_c) or spread_c <= 0) and np.isfinite(bid_c) and np.isfinite(ask_c) and ask_c >= bid_c:
            spread_c = ask_c - bid_c; payload["spread_c"] = spread_c
        payload["spread_pips"] = spread_c / ps if np.isfinite(spread_c) and spread_c >= 0 else 0.0
    if payload.get("spread_atr") in (None, ""):
        atr14 = safe_float(payload.get("atr14"), 0.0); spread_c = safe_float(payload.get("spread_c"), 0.0); payload["spread_atr"] = (spread_c / atr14) if atr14 > 0 else 0.0
    b = BUNDLES.get(pair6)
    if not b:
        out = make_out(decision="NONE", why="Model not loaded for symbol", would_order=False, order_allowed=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, model_type="none", **build_response_base(p, pair6, instrument, "", 0.0, None, equity_used, hint_side)); write_audit_row(out); return out
    model_version = b["model_version"]; model_type = b.get("model_type", "unknown")
    avg_auc = safe_float(b.get("avg_auc"), 0.0); precision_at_gate = safe_float(b.get("precision_at_gate"), 0.0); trades_at_gate = safe_int(b.get("trades_at_gate"), 0); pair_score = safe_float(b.get("training_pair_score"), 0.0)
    static_filter_passed = bool(b.get("static_filter_passed", False)); static_filter_reason = str(b.get("static_filter_reason", ""))
    bad_payload_reason = payload_sanity_checks(payload, instrument)
    if bad_payload_reason or not static_filter_passed:
        why = bad_payload_reason if bad_payload_reason else f"Static training filter blocked: {static_filter_reason}"
        out = make_out(decision="NONE", why=why, would_order=False, order_allowed=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, model_type=model_type, precision_at_gate=precision_at_gate, trades_at_gate=trades_at_gate, static_filter_passed=static_filter_passed, static_filter_reason=static_filter_reason, **build_response_base(p, pair6, instrument, model_version, avg_auc, pair_score, equity_used, hint_side)); write_audit_row(out); return out
    conf_gate = safe_float(b.get("gate"), DEFAULT_GATE["conf"]); margin_gate = safe_float(b.get("margin_gate"), DEFAULT_GATE["margin"])
    try:
        feat_order = b["feature_order"]; feature_row = build_runtime_feature_row(payload, pair6, instrument, feat_order)
        X = pd.DataFrame([{f: feature_row.get(f, 0.0) for f in feat_order}], columns=feat_order)
        p_up = predict_model_probability(b["model"], X)
        side = "BUY" if p_up >= 0.5 else "SELL"
        if p.force_decision in ("BUY", "SELL"): side = p.force_decision
        side_prob = p_up if side == "BUY" else 1.0 - p_up; conf = max(0.0, min(1.0, side_prob)); p_up = max(0.0, min(1.0, p_up)); margin = float(abs(p_up - 0.5) * 2.0)
        primary_model_type = model_type; primary_side = side; primary_p_up = p_up; primary_confidence = conf; primary_margin = margin; decision_source = "primary"
        fallback_result = {"fallback_used": False, "fallback_allowed": False, "fallback_reason": "primary_not_evaluated_yet", "candidate_votes": {}}
        base = build_response_base(p, pair6, instrument, model_version, avg_auc, pair_score, equity_used, hint_side, conf=conf, side_prob=conf, p_up=p_up, margin=margin)
        hint_disagrees = hint_side in ("BUY", "SELL") and side != hint_side
        if hint_disagrees and conf < DEFAULT_DISAGREE_CONF:
            out = make_out(decision="NONE", why=f"Blocked disagreement: ML {side} vs hint {hint_side} (conf {conf:.2f} < {DEFAULT_DISAGREE_CONF:.2f})", would_order=False, order_allowed=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, model_type=model_type, precision_at_gate=precision_at_gate, trades_at_gate=trades_at_gate, static_filter_passed=static_filter_passed, static_filter_reason=static_filter_reason, conf_gate=conf_gate, margin_gate=margin_gate, decision_source="blocked_disagreement", primary_model_type=primary_model_type, primary_side=primary_side, primary_p_up=primary_p_up, primary_confidence=primary_confidence, primary_margin=primary_margin, fallback_used=False, fallback_allowed=False, fallback_reason="not_checked_due_to_disagreement", candidate_votes={}, **base); write_audit_row(out); return out
        primary_would_order = conf >= conf_gate and margin >= margin_gate
        would_order = primary_would_order
        if not primary_would_order:
            fallback_result = evaluate_candidate_models_for_fallback(b, X, primary_model_type, conf_gate, margin_gate, hint_side)
            if fallback_result.get("fallback_allowed"):
                decision_source = "fallback"; model_type = str(fallback_result["fallback_model"]); side = str(fallback_result["fallback_side"]); p_up = float(fallback_result["fallback_p_up"]); conf = float(fallback_result["fallback_confidence"]); side_prob = conf; margin = float(fallback_result["fallback_margin"]); would_order = True
        fingerprint = make_signal_fingerprint(instrument, side, p.t, float(p.mid_c), p.tf)
        if would_order and is_duplicate_signal(pair6, fingerprint):
            out = make_out(decision="NONE", why=f"Duplicate signal blocked for {instrument}", would_order=False, order_allowed=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, model_type=model_type, precision_at_gate=precision_at_gate, trades_at_gate=trades_at_gate, static_filter_passed=static_filter_passed, static_filter_reason=static_filter_reason, conf_gate=conf_gate, margin_gate=margin_gate, decision_source=decision_source, primary_model_type=primary_model_type, primary_side=primary_side, primary_p_up=primary_p_up, primary_confidence=primary_confidence, primary_margin=primary_margin, fallback_used=bool(fallback_result.get("fallback_used", False)), fallback_allowed=bool(fallback_result.get("fallback_allowed", False)), fallback_reason=fallback_result.get("fallback_reason"), candidate_votes=fallback_result.get("candidate_votes", {}), **base); write_audit_row(out); return out
        if would_order and trades_today_total() >= MAX_TRADES_PER_DAY_TOTAL: would_order = False; block_reason = f"Daily lock: total max trades reached ({MAX_TRADES_PER_DAY_TOTAL})"
        elif would_order and trades_today(pair6) >= MAX_TRADES_PER_DAY_PER_PAIR: would_order = False; block_reason = f"Daily lock: max trades for {instrument} reached"
        elif would_order and not can_open_trade(): would_order = False; block_reason = f"Open trade cap reached ({MAX_OPEN_TRADES})"
        else: block_reason = None
        units_abs = units_signed = sl_pips = tp_pips = sl_price = tp_price = None
        why = block_reason or (f"Below trained gate | primary_model={primary_model_type}, primary_conf={primary_confidence:.2f}/{conf_gate:.2f}, primary_margin={primary_margin:.2f}/{margin_gate:.2f}, fallback_used={fallback_result.get('fallback_used')}, fallback_reason={fallback_result.get('fallback_reason')}, pair_score={pair_score:.2f}")
        decision = "NONE"
        if would_order:
            runtime_atr = safe_float(payload.get("atr14"), 0.0)
            if runtime_atr <= 0: runtime_atr = safe_float(feature_row.get("atr14_pips"), 0.0) * instrument_pip_size(instrument)
            sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(side, float(p.mid_c), float(runtime_atr), instrument, b["labeling"]["sl_atr"], b["labeling"]["tp_atr"])
            units_abs = compute_units_dynamic(instrument, sl_pips, avg_auc, pair_score, equity_used, p.force_units_abs)
            if pair6 in EXPERIMENTAL_LIVE_PAIRS and pair6 not in PRIMARY_LIVE_PAIRS:
                units_abs = max(min_units_for_instrument(instrument), int(units_abs * EXPERIMENTAL_SIZE_MULTIPLIER))
            units_signed = units_abs if side == "BUY" else -units_abs
            why = f"OK: {side} passed | decision_source={decision_source}, model={model_type}, conf={conf:.2f}, margin={margin:.2f}, primary_model={primary_model_type}, primary_conf={primary_confidence:.2f}, primary_margin={primary_margin:.2f}, precision_at_gate={precision_at_gate:.2f}, trades_at_gate={trades_at_gate}, pair_score={pair_score:.2f}, equity_used={equity_used:.2f}"
            decision = side
        out = make_out(decision=decision, why=why, would_order=bool(would_order), order_allowed=bool(would_order), units=units_abs, units_signed=units_signed, sl_pips=sl_pips, tp_pips=tp_pips, sl_price=sl_price, tp_price=tp_price, model_type=model_type, precision_at_gate=precision_at_gate, trades_at_gate=trades_at_gate, static_filter_passed=static_filter_passed, static_filter_reason=static_filter_reason, conf_gate=conf_gate, margin_gate=margin_gate, decision_source=decision_source, primary_model_type=primary_model_type, primary_side=primary_side, primary_p_up=primary_p_up, primary_confidence=primary_confidence, primary_margin=primary_margin, fallback_used=bool(fallback_result.get("fallback_used", False)), fallback_allowed=bool(fallback_result.get("fallback_allowed", False)), fallback_reason=fallback_result.get("fallback_reason"), fallback_model=fallback_result.get("fallback_model"), fallback_confidence=fallback_result.get("fallback_confidence"), fallback_margin=fallback_result.get("fallback_margin"), candidate_votes=fallback_result.get("candidate_votes", {}), **base)
        if would_order:
            remember_signal(pair6, fingerprint); inc_trade(pair6)
        write_audit_row(out); return out
    except Exception as e:
        out = make_out(decision="NONE", why=f"Prediction error: {repr(e)}", would_order=False, order_allowed=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, model_type=model_type, precision_at_gate=precision_at_gate, trades_at_gate=trades_at_gate, static_filter_passed=static_filter_passed, static_filter_reason=static_filter_reason, **build_response_base(p, pair6, instrument, model_version, avg_auc, pair_score, equity_used, hint_side)); write_audit_row(out); return out

@app.post("/trade_event")
def trade_event(t: TradeEvent):
    row = t.model_dump()
    if not row.get("ts"): row["ts"] = utc_ts()
    t.instrument = str(t.instrument).upper(); t.side = normalize_side(t.side)
    if not row.get("symbol"): row["symbol"] = instrument_to_symbol(t.instrument)
    write_trade_row({"instrument": t.instrument, "side": t.side, "units_signed": t.units_signed, "entry_price": t.entry_price, "sl_price": t.sl_price, "tp_price": t.tp_price, "status": t.status, "pnl": t.pnl, "order_id": t.order_id, "reason": t.reason, "pair_score": t.pair_score, "ts": row["ts"]})
    tracking_key = make_tracking_key(t.order_id, t.broker_trade_id, t.client_trade_id, t.instrument, t.side, row["ts"])
    if t.status == "OPEN":
        note_trade_opened(tracking_key)
        opened_at_dt = dt.datetime.now(dt.timezone.utc)
        try: opened_at_dt = pd.to_datetime(row["ts"], utc=True).to_pydatetime()
        except Exception: pass
        _open_trade_meta[str(tracking_key)] = {"tracking_key": str(tracking_key), "instrument": t.instrument, "symbol": row["symbol"], "side": t.side, "units_signed": t.units_signed, "entry_price": t.entry_price, "sl_price": t.sl_price, "tp_price": t.tp_price, "pair_score": t.pair_score, "opened_at_dt": opened_at_dt, "order_id": t.order_id, "broker_trade_id": t.broker_trade_id, "broker_order_id": t.broker_order_id, "client_trade_id": t.client_trade_id, "ts": row["ts"]}
    if t.status in ("CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"):
        for key in [tracking_key, t.broker_trade_id, t.client_trade_id, t.order_id]:
            if key: note_trade_closed(key); _open_trade_meta.pop(str(key), None)
    return {"ok": True, "open_trades": current_open_trade_count(), "status": t.status, "order_id": t.order_id, "tracking_key": tracking_key, "broker_trade_id": t.broker_trade_id}

@app.get("/health")
def health():
    return {"ok": True, "ts": utc_ts(), "model_format": "new_per_pair_best_model_with_fallback_candidates", "pairs_loaded": len(BUNDLES), "pairs": sorted([pair_to_instrument(p) for p in BUNDLES.keys()]), "pair_details": {pair: {"instrument": b.get("instrument"), "model_type": b.get("model_type"), "candidate_models_loaded": sorted(list((b.get("candidate_models") or {}).keys())), "auc": b.get("avg_auc"), "precision_at_gate": b.get("precision_at_gate"), "trades_at_gate": b.get("trades_at_gate"), "training_pair_score": b.get("training_pair_score"), "static_filter_passed": b.get("static_filter_passed"), "static_filter_reason": b.get("static_filter_reason"), "gate": b.get("gate"), "margin_gate": b.get("margin_gate")} for pair, b in BUNDLES.items()}, "strict_model_filter_enabled": STRICT_MODEL_FILTER_ENABLED, "live_min_auc": LIVE_MIN_AUC, "live_min_precision": LIVE_MIN_PRECISION, "live_min_trades_at_gate": LIVE_MIN_TRADES_AT_GATE, "live_min_pair_score": LIVE_MIN_PAIR_SCORE, "primary_live_pairs": sorted(PRIMARY_LIVE_PAIRS), "experimental_live_pairs": sorted(EXPERIMENTAL_LIVE_PAIRS), "allow_experimental_pairs": ALLOW_EXPERIMENTAL_PAIRS, "fallback_mode_enabled": FALLBACK_MODE_ENABLED, "fallback_conf_edge": FALLBACK_CONF_EDGE, "fallback_margin_edge": FALLBACK_MARGIN_EDGE, "fallback_min_auc": FALLBACK_MIN_AUC, "fallback_min_precision": FALLBACK_MIN_PRECISION, "fallback_min_trades_at_gate": FALLBACK_MIN_TRADES_AT_GATE, "fallback_require_hint_agree": FALLBACK_REQUIRE_HINT_AGREE, "db_path": DB_PATH, "data_dir": DATA_DIR, "models_dir": MODELS_DIR, "auto_close_enabled": AUTO_CLOSE_ENABLED, "max_hold_minutes": MAX_HOLD_MINUTES, "current_open_trades": current_open_trade_count()}

@app.get("/stats")
def stats():
    df = read_audit_df()
    if df.empty: return {"ok": True, "rows": 0, "would_order_count": 0, "decision_counts": {}, "pair_counts": {}, "last_ts": None}
    return {"ok": True, "rows": int(len(df)), "would_order_count": int(safe_bool_series(df, "would_order").sum()), "decision_counts": df["decision"].value_counts(dropna=False).to_dict() if "decision" in df.columns else {}, "pair_counts": df["instrument"].value_counts(dropna=False).to_dict() if "instrument" in df.columns else {}, "last_ts": pd.to_datetime(df["ts"], errors="coerce").dropna().max().isoformat() if "ts" in df.columns and not pd.to_datetime(df["ts"], errors="coerce").dropna().empty else None}

@app.get("/pnl_stats")
def pnl_stats():
    df = read_trades_df()
    if df.empty: return {"ok": True, "trades": 0, "closed_trades": 0, "wins": 0, "losses": 0, "win_rate": None, "net_pnl": 0.0, "avg_pnl": None, "open_trades": current_open_trade_count()}
    closed = df[df["status"].isin(["CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"])].copy()
    pnl = pd.to_numeric(closed.get("pnl"), errors="coerce").fillna(0.0) if not closed.empty else pd.Series(dtype=float)
    return {"ok": True, "trades": int(len(df)), "closed_trades": int(len(closed)), "wins": int((pnl > 0).sum()), "losses": int((pnl < 0).sum()), "win_rate": float((pnl > 0).sum() / len(closed)) if len(closed) else None, "net_pnl": float(pnl.sum()) if len(pnl) else 0.0, "avg_pnl": float(pnl.mean()) if len(pnl) else None, "open_trades": current_open_trade_count()}

@app.get("/export/closed_trades.xlsx")
def export_closed_trades_xlsx():
    df = read_closed_trades_df().copy(); out_path = os.path.join(LOG_DIR, "closed_trades_export.xlsx")
    if df.empty: df = pd.DataFrame(columns=["ts", "instrument", "side", "units_signed", "entry_price", "sl_price", "tp_price", "status", "pnl", "order_id", "reason", "pair_score"])
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="closed_trades")
    return FileResponse(out_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="closed_trades_export.xlsx")

@app.get("/dashboard")
def dashboard():
    audit_df = read_audit_df(); trades_df = read_trades_df(); total_rows = len(audit_df)
    would_count = int(safe_bool_series(audit_df, "would_order").sum()) if not audit_df.empty else 0
    latest = audit_df.tail(50).to_html(index=False, escape=False) if not audit_df.empty else "<p>No audit data yet.</p>"
    trades = trades_df.tail(50).to_html(index=False, escape=False) if not trades_df.empty else "<p>No trade data yet.</p>"
    html = f"""
    <html><head><title>FX 1H Fallback Dashboard</title><meta http-equiv=\"refresh\" content=\"15\"></head>
    <body style=\"font-family:Arial;padding:24px;\"><h1>FX 1H Fallback Dashboard</h1>
    <p>Total predictions: <b>{total_rows}</b> | Would order: <b>{would_count}</b> | Open trades: <b>{current_open_trade_count()}</b></p>
    <p><a href=\"/health\">/health</a> | <a href=\"/stats\">/stats</a> | <a href=\"/pnl_stats\">/pnl_stats</a> | <a href=\"/export/closed_trades.xlsx\">Export Excel</a></p>
    <h2>Latest Predictions</h2>{latest}<h2>Latest Trade Events</h2>{trades}</body></html>
    """
    return HTMLResponse(content=html)
