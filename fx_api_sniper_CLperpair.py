from __future__ import annotations

import os
import csv
import math
import json
import hashlib
import datetime as dt
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Literal, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
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
# STRICT LIVE MODEL FILTERS
# ====================================================

STRICT_MODEL_FILTER_ENABLED = os.getenv("STRICT_MODEL_FILTER_ENABLED", "true").lower() == "true"

LIVE_MIN_AUC = float(os.getenv("LIVE_MIN_AUC", "0.52"))
LIVE_MIN_PRECISION = float(os.getenv("LIVE_MIN_PRECISION", "0.58"))
LIVE_MIN_TRADES_AT_GATE = int(os.getenv("LIVE_MIN_TRADES_AT_GATE", "50"))
LIVE_MIN_PAIR_SCORE = float(os.getenv("LIVE_MIN_PAIR_SCORE", "0.50"))

PRIMARY_LIVE_PAIRS = {
    x.strip().upper()
    for x in os.getenv("PRIMARY_LIVE_PAIRS", "GBPCHF,EURGBP,GBPJPY").split(",")
    if x.strip()
}

EXPERIMENTAL_LIVE_PAIRS = {
    x.strip().upper()
    for x in os.getenv("EXPERIMENTAL_LIVE_PAIRS", "AUDCAD,EURCHF,USDCAD,USDJPY").split(",")
    if x.strip()
}

ALLOW_EXPERIMENTAL_PAIRS = os.getenv("ALLOW_EXPERIMENTAL_PAIRS", "false").lower() == "true"
EXPERIMENTAL_SIZE_MULTIPLIER = float(os.getenv("EXPERIMENTAL_SIZE_MULTIPLIER", "0.50"))


# ====================================================
# PAIRS
# ====================================================

PAIR_MAP: Dict[str, str] = {
    "AUDCAD": "AUD_CAD",
    "AUDJPY": "AUD_JPY",
    "AUDNZD": "AUD_NZD",
    "AUDUSD": "AUD_USD",

    "CADJPY": "CAD_JPY",
    "CHFJPY": "CHF_JPY",

    "EURCHF": "EUR_CHF",
    "EURGBP": "EUR_GBP",
    "EURJPY": "EUR_JPY",
    "EURUSD": "EUR_USD",

    "GBPCHF": "GBP_CHF",
    "GBPJPY": "GBP_JPY",
    "GBPUSD": "GBP_USD",

    "NZDJPY": "NZD_JPY",
    "NZDUSD": "NZD_USD",

    "USDCAD": "USD_CAD",
    "USDCHF": "USD_CHF",
    "USDJPY": "USD_JPY",
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

    if len(s) == 6 and s.isalpha() and s in PAIR_MAP:
        return s

    return None


def pair_to_instrument(pair6: str) -> str:
    return PAIR_MAP[pair6]


def instrument_to_symbol(instrument: str) -> str:
    return INSTRUMENT_TO_PAIR6.get(
        str(instrument).upper(),
        str(instrument).replace("_", ""),
    )


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

    cur.execute(
        """
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
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_order_id ON trade_events(order_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_status ON trade_events(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_ts ON trade_events(ts)")

    conn.commit()
    conn.close()


def insert_trade_event_db(row: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO trade_events
        (ts, instrument, side, units_signed, entry_price, sl_price, tp_price, status, pnl, order_id, reason, pair_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
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
        ),
    )

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

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_sm = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)

    plus_di = (
        100
        * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr_sm
    )

    minus_di = (
        100
        * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr_sm
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_val, plus_di, minus_di


def ema_runtime(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi_runtime(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    return out.fillna(50)


def update_bar_history(pair6: str, payload: Dict[str, Any]) -> pd.DataFrame:
    q = _bar_history.setdefault(pair6, deque(maxlen=BAR_HISTORY_LEN))

    t_val = safe_int(payload.get("t"))
    ts = pd.to_datetime(t_val, unit="s", utc=True, errors="coerce")

    if pd.isna(ts):
        ts = pd.Timestamp.utcnow()

    row = {
        "t": t_val,
        "time": ts,
        "mid_o": safe_float(payload.get("mid_o")),
        "mid_h": safe_float(payload.get("mid_h")),
        "mid_l": safe_float(payload.get("mid_l")),
        "mid_c": safe_float(payload.get("mid_c")),
        "volume": safe_float(payload.get("volume"), 0.0),
        "spread_c": safe_float(payload.get("spread_c"), 0.0),
    }

    if q and q[-1]["t"] == row["t"]:
        q[-1] = row
    else:
        q.append(row)

    return pd.DataFrame(list(q))


def seed_history_from_csv(data_dir: str) -> None:
    root = Path(data_dir)

    if not root.exists():
        print(f"WARNING: DATA_DIR not found for history seed: {data_dir}")
        return

    for pair6 in PAIR_MAP:
        path = root / f"{pair6}.csv"

        if not path.exists():
            continue

        try:
            df = pd.read_csv(path).tail(BAR_HISTORY_LEN).copy()

            if "time" not in df.columns:
                continue

            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"])

            has_mid = all(c in df.columns for c in ["mid_o", "mid_h", "mid_l", "mid_c"])

            has_bid_ask = all(
                c in df.columns
                for c in [
                    "bid_o", "bid_h", "bid_l", "bid_c",
                    "ask_o", "ask_h", "ask_l", "ask_c",
                ]
            )

            if not has_mid and not has_bid_ask:
                continue

            if not has_mid:
                df["mid_o"] = (
                    pd.to_numeric(df["bid_o"], errors="coerce")
                    + pd.to_numeric(df["ask_o"], errors="coerce")
                ) / 2.0

                df["mid_h"] = (
                    pd.to_numeric(df["bid_h"], errors="coerce")
                    + pd.to_numeric(df["ask_h"], errors="coerce")
                ) / 2.0

                df["mid_l"] = (
                    pd.to_numeric(df["bid_l"], errors="coerce")
                    + pd.to_numeric(df["ask_l"], errors="coerce")
                ) / 2.0

                df["mid_c"] = (
                    pd.to_numeric(df["bid_c"], errors="coerce")
                    + pd.to_numeric(df["ask_c"], errors="coerce")
                ) / 2.0

            if "spread_c" not in df.columns and has_bid_ask:
                df["spread_c"] = (
                    pd.to_numeric(df["ask_c"], errors="coerce")
                    - pd.to_numeric(df["bid_c"], errors="coerce")
                )

            if "spread_c" not in df.columns:
                df["spread_c"] = 0.0

            if "volume" not in df.columns:
                df["volume"] = 0.0

            q = deque(maxlen=BAR_HISTORY_LEN)

            for _, r in df.iterrows():
                q.append(
                    {
                        "t": int(pd.Timestamp(r["time"]).timestamp()),
                        "time": r["time"],
                        "mid_o": float(r["mid_o"]),
                        "mid_h": float(r["mid_h"]),
                        "mid_l": float(r["mid_l"]),
                        "mid_c": float(r["mid_c"]),
                        "volume": safe_float(r.get("volume"), 0.0),
                        "spread_c": safe_float(r.get("spread_c"), 0.0),
                    }
                )

            _bar_history[pair6] = q
            print(f"Seeded {pair6} history with {len(q)} bars.")

        except Exception as e:
            print(f"WARNING: failed to seed history for {pair6}: {e}")
            continue


# ====================================================
# FEATURE BUILDER - MATCHES NEW TRAINING SCRIPT
# ====================================================

def add_runtime_training_features(
    hist: pd.DataFrame,
    pair6: str,
    instrument: str,
) -> pd.DataFrame:
    df = hist.copy()

    for c in ["mid_o", "mid_h", "mid_l", "mid_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ps = instrument_pip_size(instrument)
    c = df["mid_c"]

    df["ret1"] = c.pct_change(1)
    df["ret2"] = c.pct_change(2)
    df["ret3"] = c.pct_change(3)
    df["ret6"] = c.pct_change(6)
    df["ret12"] = c.pct_change(12)
    df["ret24"] = c.pct_change(24)

    df["range_pips"] = (df["mid_h"] - df["mid_l"]) / ps
    df["body_pips"] = (df["mid_c"] - df["mid_o"]) / ps
    df["upper_wick_pips"] = (df["mid_h"] - df[["mid_o", "mid_c"]].max(axis=1)) / ps
    df["lower_wick_pips"] = (df[["mid_o", "mid_c"]].min(axis=1) - df["mid_l"]) / ps

    df["ema20"] = ema_runtime(c, 20)
    df["ema50"] = ema_runtime(c, 50)
    df["ema100"] = ema_runtime(c, 100)
    df["ema200"] = ema_runtime(c, 200)

    df["dist_ema20_pips"] = (c - df["ema20"]) / ps
    df["dist_ema50_pips"] = (c - df["ema50"]) / ps
    df["dist_ema100_pips"] = (c - df["ema100"]) / ps
    df["dist_ema200_pips"] = (c - df["ema200"]) / ps

    df["ema20_slope"] = df["ema20"].diff(3) / ps
    df["ema50_slope"] = df["ema50"].diff(6) / ps
    df["ema200_slope"] = df["ema200"].diff(12) / ps

    df["rsi14"] = rsi_runtime(c, 14)
    df["rsi7"] = rsi_runtime(c, 7)

    df["atr14"] = atr(df["mid_h"], df["mid_l"], df["mid_c"], 14)
    df["atr14_pips"] = df["atr14"] / ps

    adx_series, _, _ = adx(df["mid_h"], df["mid_l"], df["mid_c"], 14)
    df["adx14"] = adx_series.fillna(20)

    ema12 = ema_runtime(c, 12)
    ema26 = ema_runtime(c, 26)

    df["macd"] = ema12 - ema26
    df["macd_signal"] = ema_runtime(df["macd"], 9)
    df["macdh"] = df["macd"] - df["macd_signal"]
    df["macdh_pips"] = df["macdh"] / ps

    roll20 = c.rolling(20)
    df["bb_mid"] = roll20.mean()
    df["bb_std"] = roll20.std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    df["bb_width_pips"] = (df["bb_upper"] - df["bb_lower"]) / ps
    df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    if "spread_c" in df.columns:
        df["spread_c"] = pd.to_numeric(df["spread_c"], errors="coerce").fillna(0.0)
        df["spread_pips"] = df["spread_c"] / ps
    else:
        df["spread_pips"] = 0.0

    df["spread_atr"] = df["spread_pips"] / df["atr14_pips"].replace(0, np.nan)

    dt_series = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["hour_utc"] = dt_series.dt.hour.fillna(0)
    df["day_of_week"] = dt_series.dt.dayofweek.fillna(0)
    df["month"] = dt_series.dt.month.fillna(0)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_utc"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_utc"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["trend_up"] = (df["ema20"] > df["ema50"]).astype(int)
    df["trend_down"] = (df["ema20"] < df["ema50"]).astype(int)
    df["price_above_ema200"] = (c > df["ema200"]).astype(int)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def build_runtime_feature_row(
    payload: Dict[str, Any],
    pair6: str,
    instrument: str,
    feat_order: List[str],
) -> Dict[str, Any]:
    bid_c = safe_float(payload.get("bid_c"), np.nan)
    ask_c = safe_float(payload.get("ask_c"), np.nan)
    spread_c = safe_float(payload.get("spread_c"), np.nan)

    if (
        (not np.isfinite(spread_c) or spread_c <= 0)
        and np.isfinite(bid_c)
        and np.isfinite(ask_c)
        and ask_c >= bid_c
    ):
        spread_c = ask_c - bid_c

    if not np.isfinite(spread_c):
        spread_c = 0.0

    payload["spread_c"] = spread_c

    hist = update_bar_history(pair6, payload)

    if "spread_c" not in hist.columns:
        hist["spread_c"] = 0.0

    if "volume" not in hist.columns:
        hist["volume"] = 0.0

    if len(hist) > 0:
        hist.loc[hist.index[-1], "spread_c"] = spread_c
        hist.loc[hist.index[-1], "volume"] = safe_float(payload.get("volume"), 0.0)

    feat_df = add_runtime_training_features(hist, pair6, instrument)

    last = (
        feat_df.iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_dict()
    )

    out: Dict[str, Any] = {}

    for f in feat_order:
        out[f] = safe_float(last.get(f), safe_float(payload.get(f), 0.0))

    return out


# ====================================================
# PAYLOAD SANITY / SIZING / SL TP
# ====================================================

def payload_sanity_checks(payload: Dict[str, Any], instrument: str) -> Optional[str]:
    ps = instrument_pip_size(instrument)

    spread_pips = safe_float(payload.get("spread_pips"), np.nan)

    if not np.isfinite(spread_pips):
        spread_c = safe_float(payload.get("spread_c"), 0.0)
        spread_pips = spread_c / ps if ps > 0 else 0.0
        payload["spread_pips"] = spread_pips

    if spread_pips > MAX_SPREAD_PIPS:
        return f"Spread too high: {spread_pips} pips > {MAX_SPREAD_PIPS}"

    atr14 = safe_float(payload.get("atr14"), 0.0)

    if atr14 <= 0:
        # Use a soft pass because runtime features can compute ATR from history.
        atr14 = 0.0

    if atr14 > 0 and atr14 < min_atr_for_instrument(instrument):
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

        if risk_per_1000 > 0:
            base = int((risk_cap / risk_per_1000) * 1000)

    if pair_score >= 0.80:
        base = int(base * 1.20)
    elif pair_score >= 0.65:
        base = int(base * 1.10)
    elif pair_score < 0.50:
        base = int(base * 0.70)

    if avg_auc >= 0.57:
        base = int(base * 1.05)
    elif avg_auc < 0.54:
        base = int(base * 0.90)

    return min(
        max_units_for_instrument(instrument),
        max(min_units_for_instrument(instrument), base),
    )


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
        return None, None, None, None

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
# MODEL LOADING - NEW PER-PAIR TRAINER FORMAT
# ====================================================

def load_json_safe(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return default


def get_best_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    best = metrics.get("best") or {}
    return best if isinstance(best, dict) else {}


def pair_passes_static_training_filter(
    pair6: str,
    metrics: Dict[str, Any],
) -> Tuple[bool, str]:
    if not STRICT_MODEL_FILTER_ENABLED:
        return True, "strict_filter_disabled"

    best = get_best_metrics(metrics)

    model_tradable = bool(metrics.get("tradable", False)) and bool(best.get("tradable", False))
    auc = safe_float(best.get("auc"), 0.0)
    precision = safe_float(best.get("precision_at_gate"), 0.0)
    trades = safe_int(best.get("trades_at_gate"), 0)
    pair_score = safe_float(best.get("pair_score"), 0.0)

    reasons = []

    if not model_tradable:
        reasons.append("training_marked_not_tradable")

    if auc < LIVE_MIN_AUC:
        reasons.append(f"auc_too_low:{auc:.4f}<{LIVE_MIN_AUC:.4f}")

    if precision < LIVE_MIN_PRECISION:
        reasons.append(f"precision_too_low:{precision:.4f}<{LIVE_MIN_PRECISION:.4f}")

    if trades < LIVE_MIN_TRADES_AT_GATE:
        reasons.append(f"trades_too_low:{trades}<{LIVE_MIN_TRADES_AT_GATE}")

    if pair_score < LIVE_MIN_PAIR_SCORE:
        reasons.append(f"pair_score_too_low:{pair_score:.4f}<{LIVE_MIN_PAIR_SCORE:.4f}")

    if reasons:
        return False, "; ".join(reasons)

    if pair6 in PRIMARY_LIVE_PAIRS:
        return True, "primary_live_pair_passed"

    if pair6 in EXPERIMENTAL_LIVE_PAIRS:
        if ALLOW_EXPERIMENTAL_PAIRS:
            return True, "experimental_pair_allowed"

        return False, "experimental_pair_blocked_until_enabled"

    return False, "pair_not_in_primary_or_experimental_live_list"


def load_new_model_bundle(pair_dir: Path) -> Optional[Dict[str, Any]]:
    pair6 = pair_dir.name.upper().replace("_", "")

    if pair6 not in PAIR_MAP:
        return None

    feature_columns = load_json_safe(pair_dir / "feature_columns.json", [])
    thresholds = load_json_safe(pair_dir / "thresholds.json", {})
    model_type_json = load_json_safe(pair_dir / "best_model_type.json", {})
    metrics = load_json_safe(pair_dir / "metrics.json", {})

    if not feature_columns:
        return None

    model_type = str(
        model_type_json.get("model_type")
        or thresholds.get("model_name")
        or ""
    ).strip()

    if not model_type:
        return None

    if model_type == "neural_tcn":
        print(
            f"WARNING: {pair6} winner is neural_tcn, "
            f"but live TCN inference is not enabled in this server."
        )
        return None

    model_path = pair_dir / "best_model.pkl"

    if not model_path.exists():
        return None

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"WARNING: could not load model for {pair6}: {e}")
        return None

    best = get_best_metrics(metrics)

    passed_filter, filter_reason = pair_passes_static_training_filter(pair6, metrics)

    return {
        "pair6": pair6,
        "instrument": pair_to_instrument(pair6),
        "model": model,
        "model_type": model_type,
        "feature_order": list(feature_columns),
        "thresholds": thresholds,
        "metrics": metrics,
        "best": best,

        "avg_auc": safe_float(best.get("auc"), 0.0),
        "precision_at_gate": safe_float(best.get("precision_at_gate"), 0.0),
        "trades_at_gate": safe_int(best.get("trades_at_gate"), 0),
        "training_pair_score": safe_float(best.get("pair_score"), 0.0),
        "training_tradable": bool(metrics.get("tradable", False)),

        "static_filter_passed": passed_filter,
        "static_filter_reason": filter_reason,

        "labeling": {
            "sl_atr": safe_float(thresholds.get("sl_atr"), DEFAULT_SL_ATR),
            "tp_atr": safe_float(thresholds.get("tp_atr"), DEFAULT_TP_ATR),
        },

        "gate": safe_float(thresholds.get("gate"), DEFAULT_GATE["conf"]),
        "margin_gate": safe_float(thresholds.get("margin_gate"), DEFAULT_GATE["margin"]),

        "model_version": f"{pair6}:{model_type}",
        "_bundle_path": str(pair_dir),
    }


def load_bundles(models_dir: str) -> Dict[str, Dict[str, Any]]:
    bundles: Dict[str, Dict[str, Any]] = {}

    root = Path(models_dir)

    if not root.exists():
        print(f"WARNING: MODELS_DIR does not exist: {models_dir}")
        return bundles

    for pair_dir in sorted(root.iterdir()):
        if not pair_dir.is_dir():
            continue

        b = load_new_model_bundle(pair_dir)

        if b:
            bundles[b["pair6"]] = b

    print(f"Loaded {len(bundles)} model bundles from {models_dir}")
    return bundles


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


def oanda_request(
    method: str,
    path: str,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    if not broker_can_close():
        return {"ok": False, "error": "Missing OANDA env vars"}

    url = f"{OANDA_BASE_URL}{path}"

    try:
        r = requests.request(
            method=method.upper(),
            url=url,
            headers=oanda_headers(),
            json=json_body,
            timeout=timeout,
        )

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

    payload = {
        "longUnits": "NONE",
        "shortUnits": "NONE",
    }

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
            return {
                "ok": True,
                "method": "trade_id",
                "specifier": broker_trade_id,
                "data": res.get("data"),
                "attempts": attempts,
            }

    if client_trade_id:
        spec = client_trade_id if client_trade_id.startswith("@") else f"@{client_trade_id}"
        res = close_oanda_trade_by_specifier(spec)
        attempts.append({"method": "client_trade_id", "specifier": spec, "result": res})

        if res.get("ok"):
            return {
                "ok": True,
                "method": "client_trade_id",
                "specifier": spec,
                "data": res.get("data"),
                "attempts": attempts,
            }

    position_res = get_position_side_trade_ids(instrument, side)
    attempts.append(
        {
            "method": "inspect_position",
            "instrument": instrument,
            "side": side,
            "result": position_res,
        }
    )

    if position_res.get("ok"):
        trade_ids = position_res.get("trade_ids") or []

        if len(trade_ids) == 1:
            spec = str(trade_ids[0])
            res = close_oanda_trade_by_specifier(spec)
            attempts.append({"method": "single_trade_from_position", "specifier": spec, "result": res})

            if res.get("ok"):
                return {
                    "ok": True,
                    "method": "single_trade_from_position",
                    "specifier": spec,
                    "data": res.get("data"),
                    "attempts": attempts,
                }

        if len(trade_ids) == 0:
            return {
                "ok": False,
                "error": "No open trade found on broker for instrument/side",
                "attempts": attempts,
            }

        if len(trade_ids) > 1:
            if AUTO_CLOSE_ALLOW_POSITION_FALLBACK and tracked_open_count_for_instrument_side(instrument, side) == 1:
                res = close_oanda_position_side(instrument, side)
                attempts.append(
                    {
                        "method": "position_side_close",
                        "instrument": instrument,
                        "side": side,
                        "result": res,
                    }
                )

                if res.get("ok"):
                    return {
                        "ok": True,
                        "method": "position_side_close",
                        "specifier": instrument,
                        "data": res.get("data"),
                        "attempts": attempts,
                    }

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
                    write_audit_row(
                        {
                            "ts": utc_ts(),
                            "pair": instrument_to_symbol(meta["instrument"]),
                            "instrument": meta["instrument"],
                            "symbol": instrument_to_symbol(meta["instrument"]),
                            "hint_side": meta["side"],
                            "model_version": "auto_close",
                            "model_type": "auto_close",
                            "avg_auc": None,
                            "pair_score": meta.get("pair_score"),
                            "equity_used": None,
                            "spread_pips": None,
                            "spread_atr": None,
                            "confidence": 0,
                            "side_prob": 0,
                            "p_up": 0,
                            "margin": 0,
                            "decision": "NONE",
                            "would_order": False,
                            "order_allowed": False,
                            "units": abs(int(meta["units_signed"])),
                            "units_signed": meta["units_signed"],
                            "sl_pips": None,
                            "tp_pips": None,
                            "sl_price": meta["sl_price"],
                            "tp_price": meta["tp_price"],
                            "why": f"AUTO_CLOSE_FAILED | tracking_key={tracking_key} | error={json.dumps(close_result, default=str)[:1500]}",
                        }
                    )
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
            write_audit_row(
                {
                    "ts": utc_ts(),
                    "pair": "",
                    "instrument": "",
                    "symbol": "",
                    "hint_side": "",
                    "model_version": "auto_close",
                    "model_type": "auto_close",
                    "avg_auc": None,
                    "pair_score": None,
                    "equity_used": None,
                    "spread_pips": None,
                    "spread_atr": None,
                    "confidence": 0,
                    "side_prob": 0,
                    "p_up": 0,
                    "margin": 0,
                    "decision": "NONE",
                    "would_order": False,
                    "order_allowed": False,
                    "units": None,
                    "units_signed": None,
                    "sl_pips": None,
                    "tp_pips": None,
                    "sl_price": None,
                    "tp_price": None,
                    "why": f"AUTO_CLOSE_WORKER_EXCEPTION | {repr(e)}",
                }
            )

        time.sleep(AUTO_CLOSE_CHECK_SECONDS)


# ====================================================
# APP
# ====================================================

app = FastAPI(
    title="FX 1H Auto Model Server",
    version="8.0-new-per-pair-model-format",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    seed_history_from_csv(DATA_DIR)

    if AUTO_CLOSE_ENABLED:
        t = threading.Thread(target=auto_close_worker, daemon=True)
        t.start()


# ====================================================
# RESPONSE BASE
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
        "spread_pips": float(getattr(p, "spread_pips", 0.0) or 0.0),
        "spread_atr": float(getattr(p, "spread_atr", 0.0) or 0.0),
        "confidence": float(conf),
        "side_prob": float(side_prob),
        "p_up": float(p_up),
        "margin": float(margin),
    }


# ====================================================
# PREDICT
# ====================================================

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
            order_allowed=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            model_type="none",
            **build_response_base(p, "", "", "", 0.0, None, equity_used, hint_side),
        )
        write_audit_row(out)
        return out

    instrument = pair_to_instrument(pair6)
    payload = p.model_dump()

    ps = instrument_pip_size(instrument)

    if payload.get("spread_pips") in (None, ""):
        bid_c = safe_float(payload.get("bid_c"), np.nan)
        ask_c = safe_float(payload.get("ask_c"), np.nan)
        spread_c = safe_float(payload.get("spread_c"), np.nan)

        if (
            (not np.isfinite(spread_c) or spread_c <= 0)
            and np.isfinite(bid_c)
            and np.isfinite(ask_c)
            and ask_c >= bid_c
        ):
            spread_c = ask_c - bid_c
            payload["spread_c"] = spread_c

        payload["spread_pips"] = (
            spread_c / ps
            if np.isfinite(spread_c) and spread_c >= 0
            else 0.0
        )

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
            order_allowed=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            model_type="none",
            **build_response_base(p, pair6, instrument, "", 0.0, None, equity_used, hint_side),
        )
        write_audit_row(out)
        return out

    model_version = b["model_version"]
    model_type = b.get("model_type", "unknown")

    avg_auc = safe_float(b.get("avg_auc"), 0.0)
    precision_at_gate = safe_float(b.get("precision_at_gate"), 0.0)
    trades_at_gate = safe_int(b.get("trades_at_gate"), 0)
    pair_score = safe_float(b.get("training_pair_score"), 0.0)

    static_filter_passed = bool(b.get("static_filter_passed", False))
    static_filter_reason = str(b.get("static_filter_reason", ""))

    bad_payload_reason = payload_sanity_checks(payload, instrument)

    if bad_payload_reason:
        out = make_out(
            decision="NONE",
            why=bad_payload_reason,
            would_order=False,
            order_allowed=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            model_type=model_type,
            precision_at_gate=precision_at_gate,
            trades_at_gate=trades_at_gate,
            static_filter_passed=static_filter_passed,
            static_filter_reason=static_filter_reason,
            **build_response_base(
                p,
                pair6,
                instrument,
                model_version,
                avg_auc,
                pair_score,
                equity_used,
                hint_side,
            ),
        )
        write_audit_row(out)
        return out

    if not static_filter_passed:
        out = make_out(
            decision="NONE",
            why=f"Static training filter blocked: {static_filter_reason}",
            would_order=False,
            order_allowed=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            model_type=model_type,
            precision_at_gate=precision_at_gate,
            trades_at_gate=trades_at_gate,
            static_filter_passed=static_filter_passed,
            static_filter_reason=static_filter_reason,
            **build_response_base(
                p,
                pair6,
                instrument,
                model_version,
                avg_auc,
                pair_score,
                equity_used,
                hint_side,
            ),
        )
        write_audit_row(out)
        return out

    conf_gate = safe_float(b.get("gate"), DEFAULT_GATE["conf"])
    margin_gate = safe_float(b.get("margin_gate"), DEFAULT_GATE["margin"])

    try:
        feat_order = b["feature_order"]
        feature_row = build_runtime_feature_row(payload, pair6, instrument, feat_order)

        X = pd.DataFrame(
            [{f: feature_row.get(f, 0.0) for f in feat_order}],
            columns=feat_order,
        )

        model = b["model"]

        proba = model.predict_proba(X)[0]
        p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])

        side = "BUY" if p_up >= 0.5 else "SELL"

        if p.force_decision in ("BUY", "SELL"):
            side = p.force_decision

        side_prob = p_up if side == "BUY" else 1.0 - p_up
        conf = side_prob

        conf = max(0.0, min(1.0, conf))
        side_prob = max(0.0, min(1.0, side_prob))
        p_up = max(0.0, min(1.0, p_up))
        margin = float(abs(p_up - 0.5) * 2.0)

        base = build_response_base(
            p,
            pair6,
            instrument,
            model_version,
            avg_auc,
            pair_score,
            equity_used,
            hint_side,
            conf=conf,
            side_prob=side_prob,
            p_up=p_up,
            margin=margin,
        )

        hint_disagrees = hint_side in ("BUY", "SELL") and side != hint_side

        if hint_disagrees and conf < DEFAULT_DISAGREE_CONF:
            out = make_out(
                decision="NONE",
                why=(
                    f"Blocked disagreement: ML {side} vs hint {hint_side} "
                    f"(conf {conf:.2f} < {DEFAULT_DISAGREE_CONF:.2f})"
                ),
                would_order=False,
                order_allowed=False,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                model_type=model_type,
                precision_at_gate=precision_at_gate,
                trades_at_gate=trades_at_gate,
                static_filter_passed=static_filter_passed,
                static_filter_reason=static_filter_reason,
                conf_gate=conf_gate,
                margin_gate=margin_gate,
                **base,
            )
            write_audit_row(out)
            return out

        would_order = conf >= conf_gate and margin >= margin_gate
        fingerprint = make_signal_fingerprint(instrument, side, p.t, float(p.mid_c), p.tf)

        if would_order and is_duplicate_signal(pair6, fingerprint):
            out = make_out(
                decision="NONE",
                why=f"Duplicate signal blocked for {instrument}",
                would_order=False,
                order_allowed=False,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                model_type=model_type,
                precision_at_gate=precision_at_gate,
                trades_at_gate=trades_at_gate,
                static_filter_passed=static_filter_passed,
                static_filter_reason=static_filter_reason,
                conf_gate=conf_gate,
                margin_gate=margin_gate,
                **base,
            )
            write_audit_row(out)
            return out

        if would_order and trades_today_total() >= MAX_TRADES_PER_DAY_TOTAL:
            out = make_out(
                decision="NONE",
                why=f"Daily lock: total max trades reached ({MAX_TRADES_PER_DAY_TOTAL})",
                would_order=False,
                order_allowed=False,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                model_type=model_type,
                precision_at_gate=precision_at_gate,
                trades_at_gate=trades_at_gate,
                static_filter_passed=static_filter_passed,
                static_filter_reason=static_filter_reason,
                conf_gate=conf_gate,
                margin_gate=margin_gate,
                **base,
            )
            write_audit_row(out)
            return out

        if would_order and trades_today(pair6) >= MAX_TRADES_PER_DAY_PER_PAIR:
            out = make_out(
                decision="NONE",
                why=f"Daily lock: max trades for {instrument} reached",
                would_order=False,
                order_allowed=False,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                model_type=model_type,
                precision_at_gate=precision_at_gate,
                trades_at_gate=trades_at_gate,
                static_filter_passed=static_filter_passed,
                static_filter_reason=static_filter_reason,
                conf_gate=conf_gate,
                margin_gate=margin_gate,
                **base,
            )
            write_audit_row(out)
            return out

        if would_order and not can_open_trade():
            out = make_out(
                decision="NONE",
                why=f"Open trade cap reached ({MAX_OPEN_TRADES})",
                would_order=False,
                order_allowed=False,
                units=None,
                units_signed=None,
                sl_pips=None,
                tp_pips=None,
                sl_price=None,
                tp_price=None,
                model_type=model_type,
                precision_at_gate=precision_at_gate,
                trades_at_gate=trades_at_gate,
                static_filter_passed=static_filter_passed,
                static_filter_reason=static_filter_reason,
                conf_gate=conf_gate,
                margin_gate=margin_gate,
                **base,
            )
            write_audit_row(out)
            return out

        units_abs = None
        units_signed = None
        sl_pips = None
        tp_pips = None
        sl_price = None
        tp_price = None

        why = (
            f"Below trained gate | conf={conf:.2f}/{conf_gate:.2f}, "
            f"margin={margin:.2f}/{margin_gate:.2f}, "
            f"model={model_type}, pair_score={pair_score:.2f}"
        )
        decision = "NONE"

        if would_order:
            runtime_atr = safe_float(payload.get("atr14"), 0.0)

            if runtime_atr <= 0:
                # fallback from runtime features
                runtime_atr = safe_float(feature_row.get("atr14_pips"), 0.0) * instrument_pip_size(instrument)

            sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(
                side,
                float(p.mid_c),
                float(runtime_atr),
                instrument,
                b["labeling"]["sl_atr"],
                b["labeling"]["tp_atr"],
            )

            units_abs = compute_units_dynamic(
                instrument,
                sl_pips,
                avg_auc,
                pair_score,
                equity_used,
                p.force_units_abs,
            )

            if pair6 in EXPERIMENTAL_LIVE_PAIRS and pair6 not in PRIMARY_LIVE_PAIRS:
                units_abs = max(
                    min_units_for_instrument(instrument),
                    int(units_abs * EXPERIMENTAL_SIZE_MULTIPLIER),
                )

            units_signed = units_abs if side == "BUY" else -units_abs

            why = (
                f"OK: {side} passed trained gate | "
                f"conf={conf:.2f}/{conf_gate:.2f}, "
                f"margin={margin:.2f}/{margin_gate:.2f}, "
                f"model={model_type}, "
                f"precision_at_gate={precision_at_gate:.2f}, "
                f"trades_at_gate={trades_at_gate}, "
                f"pair_score={pair_score:.2f}, "
                f"equity_used={equity_used:.2f}"
            )

            decision = side

        out = make_out(
            decision=decision,
            why=why,
            would_order=bool(would_order),
            order_allowed=bool(would_order),
            units=units_abs,
            units_signed=units_signed,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            sl_price=sl_price,
            tp_price=tp_price,

            model_type=model_type,
            precision_at_gate=precision_at_gate,
            trades_at_gate=trades_at_gate,
            static_filter_passed=static_filter_passed,
            static_filter_reason=static_filter_reason,
            conf_gate=conf_gate,
            margin_gate=margin_gate,

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
            order_allowed=False,
            units=None,
            units_signed=None,
            sl_pips=None,
            tp_pips=None,
            sl_price=None,
            tp_price=None,
            model_type=model_type,
            precision_at_gate=precision_at_gate,
            trades_at_gate=trades_at_gate,
            static_filter_passed=static_filter_passed,
            static_filter_reason=static_filter_reason,
            **build_response_base(
                p,
                pair6,
                instrument,
                model_version,
                avg_auc,
                pair_score,
                equity_used,
                hint_side,
            ),
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

    write_trade_row(
        {
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
        }
    )

    tracking_key = make_tracking_key(
        t.order_id,
        t.broker_trade_id,
        t.client_trade_id,
        t.instrument,
        t.side,
        row["ts"],
    )

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


# ====================================================
# HEALTH / STATS / EXPORT
# ====================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": utc_ts(),
        "model_format": "new_per_pair_best_model",
        "pairs_loaded": len(BUNDLES),
        "pairs": sorted([pair_to_instrument(p) for p in BUNDLES.keys()]),
        "pair_details": {
            pair: {
                "instrument": b.get("instrument"),
                "model_type": b.get("model_type"),
                "auc": b.get("avg_auc"),
                "precision_at_gate": b.get("precision_at_gate"),
                "trades_at_gate": b.get("trades_at_gate"),
                "training_pair_score": b.get("training_pair_score"),
                "static_filter_passed": b.get("static_filter_passed"),
                "static_filter_reason": b.get("static_filter_reason"),
                "gate": b.get("gate"),
                "margin_gate": b.get("margin_gate"),
            }
            for pair, b in BUNDLES.items()
        },
        "strict_model_filter_enabled": STRICT_MODEL_FILTER_ENABLED,
        "live_min_auc": LIVE_MIN_AUC,
        "live_min_precision": LIVE_MIN_PRECISION,
        "live_min_trades_at_gate": LIVE_MIN_TRADES_AT_GATE,
        "live_min_pair_score": LIVE_MIN_PAIR_SCORE,
        "primary_live_pairs": sorted(PRIMARY_LIVE_PAIRS),
        "experimental_live_pairs": sorted(EXPERIMENTAL_LIVE_PAIRS),
        "allow_experimental_pairs": ALLOW_EXPERIMENTAL_PAIRS,
        "experimental_size_multiplier": EXPERIMENTAL_SIZE_MULTIPLIER,
        "db_path": DB_PATH,
        "data_dir": DATA_DIR,
        "models_dir": MODELS_DIR,
        "auto_close_enabled": AUTO_CLOSE_ENABLED,
        "max_hold_minutes": MAX_HOLD_MINUTES,
        "auto_close_check_seconds": AUTO_CLOSE_CHECK_SECONDS,
        "auto_close_allow_position_fallback": AUTO_CLOSE_ALLOW_POSITION_FALLBACK,
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
        last_ts = pd.to_datetime(df["ts"], errors="coerce").dropna().max().isoformat()

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
        pair6 = instrument_to_symbol(instrument)

        bundle = BUNDLES.get(pair6, {})
        training_pair_score = safe_float(bundle.get("training_pair_score"), 0.0)

        rows.append(
            {
                "instrument": instrument,
                "pair": pair6,
                "trades": n,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "net_pnl": net_pnl,
                "avg_pnl": avg_pnl,
                "training_pair_score": training_pair_score,
                "static_filter_passed": bundle.get("static_filter_passed"),
                "static_filter_reason": bundle.get("static_filter_reason"),
                "model_type": bundle.get("model_type"),
            }
        )

    rows = sorted(rows, key=lambda x: (x["training_pair_score"], x["net_pnl"]), reverse=True)

    return {"ok": True, "pairs": rows}


@app.get("/export/closed_trades.xlsx")
def export_closed_trades_xlsx():
    df = read_closed_trades_df().copy()
    out_path = os.path.join(LOG_DIR, "closed_trades_export.xlsx")

    if df.empty:
        blank = pd.DataFrame(
            columns=[
                "ts",
                "instrument",
                "side",
                "units_signed",
                "entry_price",
                "sl_price",
                "tp_price",
                "status",
                "pnl",
                "order_id",
                "reason",
                "pair_score",
            ]
        )

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            blank.to_excel(writer, index=False, sheet_name="closed_trades")

        return FileResponse(
            out_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="closed_trades_export.xlsx",
        )

    df["pnl"] = pd.to_numeric(df.get("pnl"), errors="coerce").fillna(0.0)

    summary = pd.DataFrame(
        [
            {
                "closed_trades": int(len(df)),
                "wins": int((df["pnl"] > 0).sum()),
                "losses": int((df["pnl"] < 0).sum()),
                "win_rate": float((df["pnl"] > 0).sum() / len(df)) if len(df) else None,
                "net_pnl": float(df["pnl"].sum()),
                "avg_pnl": float(df["pnl"].mean()) if len(df) else None,
            }
        ]
    )

    by_pair = (
        df.groupby("instrument", dropna=False)
        .agg(
            trades=("instrument", "count"),
            wins=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).sum())),
            losses=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) < 0).sum())),
            net_pnl=("pnl", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())),
            avg_pnl=("pnl", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).mean())),
        )
        .reset_index()
    )

    by_pair["win_rate"] = by_pair["wins"] / by_pair["trades"]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="closed_trades")
        summary.to_excel(writer, index=False, sheet_name="summary")
        by_pair.to_excel(writer, index=False, sheet_name="by_pair")

    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="closed_trades_export.xlsx",
    )


@app.get("/dashboard")
def dashboard():
    audit_df = read_audit_df()
    trades_df = read_trades_df()

    total_rows = len(audit_df)

    would_count = (
        int(safe_bool_series(audit_df, "would_order").sum())
        if not audit_df.empty and "would_order" in audit_df.columns
        else 0
    )

    none_count = (
        int((audit_df["decision"] == "NONE").sum())
        if not audit_df.empty and "decision" in audit_df.columns
        else 0
    )

    if not audit_df.empty and "ts" in audit_df.columns:
        latest_rows = audit_df.sort_values("ts", ascending=False).head(50).copy()
    elif not audit_df.empty:
        latest_rows = audit_df.tail(50).copy()
    else:
        latest_rows = pd.DataFrame()

    cols = [
        c
        for c in [
            "ts",
            "pair",
            "instrument",
            "symbol",
            "raw_symbol",
            "hint_side",
            "decision",
            "confidence",
            "side_prob",
            "p_up",
            "margin",
            "conf_gate",
            "margin_gate",
            "model_type",
            "precision_at_gate",
            "trades_at_gate",
            "pair_score",
            "equity_used",
            "units_signed",
            "sl_price",
            "tp_price",
            "would_order",
            "order_allowed",
            "why",
        ]
        if c in latest_rows.columns
    ]

    table_html = latest_rows[cols].to_html(index=False, escape=False) if not latest_rows.empty else "<p>No audit data yet.</p>"

    by_pair_html = "<p>No audit data yet.</p>"

    if not audit_df.empty and "instrument" in audit_df.columns:
        by_pair = audit_df["instrument"].value_counts().rename_axis("instrument").reset_index(name="count")
        by_pair_html = by_pair.to_html(index=False)

    pnl_html = "<p>No trade data yet.</p>"

    if not trades_df.empty:
        pnl_cols = [
            c
            for c in [
                "ts",
                "instrument",
                "side",
                "units_signed",
                "status",
                "pnl",
                "reason",
                "order_id",
            ]
            if c in trades_df.columns
        ]

        pnl_html = trades_df.sort_values("ts", ascending=False).head(50)[pnl_cols].to_html(index=False, escape=False)

    closed = read_closed_trades_df()
    win_rate_txt = "N/A"

    if not closed.empty and "pnl" in closed.columns:
        pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
        win_rate_txt = f"{((pnl > 0).sum() / len(closed)):.2%}"

    html = f"""
    <html>
      <head>
        <title>FX 1H Auto Model Dashboard</title>
        <meta http-equiv="refresh" content="15">
      </head>
      <body style="font-family: Arial; padding: 24px;">
        <h1>FX 1H Auto Model Dashboard</h1>

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

          <div style="padding:16px; border:1px solid #ccc; border-radius:8px;">
            <h3>Win rate</h3>
            <div style="font-size:28px;">{win_rate_txt}</div>
          </div>
        </div>

        <h2>Model Health</h2>
        <p><a href="/health">/health</a></p>

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
          <a href="/export/closed_trades.xlsx">/export/closed_trades.xlsx</a>
        </p>
      </body>
    </html>
    """

    return HTMLResponse(content=html)