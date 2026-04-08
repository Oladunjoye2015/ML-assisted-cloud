from __future__ import annotations
import os, csv, glob, math, json, hashlib, datetime as dt
from collections import deque
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Literal, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

# env
MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOG_DIR = os.getenv("LOG_DIR", "logs")
DATA_DIR = os.getenv("DATA_DIR", "oanda_h1_ba_live")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

AUDIT_CSV = os.path.join(LOG_DIR, "audit.csv")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")

DEFAULT_GATE = {"conf": float(os.getenv("CONF_GATE", "0.54")), "margin": float(os.getenv("MARGIN_GATE", "0.04"))}
PAIR_GATES: Dict[str, Dict[str, float]] = {
    "USD_CHF": {"conf": 0.54, "margin": 0.04},
    "EUR_GBP": {"conf": 0.57, "margin": 0.05},
    "USD_CAD": {"conf": 0.54, "margin": 0.04},
    "CAD_JPY": {"conf": 0.58, "margin": 0.05},
    "AUD_JPY": {"conf": 0.58, "margin": 0.05},
    "USD_JPY": {"conf": 0.58, "margin": 0.05},
    "EUR_JPY": {"conf": 0.60, "margin": 0.06},
    "GBP_JPY": {"conf": 0.60, "margin": 0.06},
    "EUR_USD": {"conf": 0.53, "margin": 0.04},
    "NZD_USD": {"conf": 0.56, "margin": 0.05},
    "GBP_USD": {"conf": 0.53, "margin": 0.04},
    "AUD_USD": {"conf": 0.56, "margin": 0.05},
    "EUR_CHF": {"conf": 0.56, "margin": 0.04},
    "NZD_JPY": {"conf": 0.57, "margin": 0.05},
    "GBP_CHF": {"conf": 0.59, "margin": 0.06},
    "CHF_JPY": {"conf": 0.58, "margin": 0.05},
    "AUD_CAD": {"conf": 0.56, "margin": 0.05},
    "AUD_NZD": {"conf": 0.57, "margin": 0.04},
}
PAIR_DISAGREE_CONF: Dict[str, float] = {
    "EUR_GBP": 0.64,
    "CAD_JPY": 0.66,
    "AUD_JPY": 0.66,
    "GBP_JPY": 0.66,
    "USD_CHF": 0.60,
    "USD_CAD": 0.60,
    "EUR_USD": 0.60,
    "NZD_USD": 0.64,
    "GBP_USD": 0.60,
    "USD_JPY": 0.65,
    "AUD_USD": 0.64,
    "EUR_JPY": 0.66,
    "EUR_CHF": 0.62,
    "NZD_JPY": 0.64,
    "GBP_CHF": 0.66,
    "CHF_JPY": 0.65,
    "AUD_CAD": 0.63,
    "AUD_NZD": 0.63,
}
DEFAULT_DISAGREE_CONF = float(os.getenv("DEFAULT_DISAGREE_CONF", "0.62"))
UNITS_JPY = int(os.getenv("UNITS_JPY", "1000"))
UNITS_NON_JPY = int(os.getenv("UNITS_NON_JPY", "2000"))
MAX_TRADES_PER_DAY_TOTAL = int(os.getenv("MAX_TRADES_PER_DAY_TOTAL", "6"))
MAX_TRADES_PER_DAY_PER_PAIR = int(os.getenv("MAX_TRADES_PER_DAY_PER_PAIR", "3"))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "2"))
DUP_WINDOW_SECONDS = int(os.getenv("DUP_WINDOW_SECONDS", "300"))
MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "3.5"))
MIN_ATR_NON_JPY = float(os.getenv("MIN_ATR_NON_JPY", "0.00005"))
MIN_ATR_JPY = float(os.getenv("MIN_ATR_JPY", "0.005"))
USE_EQUITY_SIZING = os.getenv("USE_EQUITY_SIZING", "true").lower() == "true"
DEFAULT_EQUITY = float(os.getenv("DEFAULT_EQUITY", "200"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))
MIN_PAIR_SCORE_TO_TRADE = float(os.getenv("MIN_PAIR_SCORE_TO_TRADE", "0.45"))
MIN_TRADES_FOR_PAIR_SCORING = int(os.getenv("MIN_TRADES_FOR_PAIR_SCORING", "8"))
AUC_WEIGHT = float(os.getenv("AUC_WEIGHT", "0.55"))
WINRATE_WEIGHT = float(os.getenv("WINRATE_WEIGHT", "0.45"))
MIN_UNITS_JPY = int(os.getenv("MIN_UNITS_JPY", "100"))
MIN_UNITS_NON_JPY = int(os.getenv("MIN_UNITS_NON_JPY", "100"))
MAX_UNITS_JPY = int(os.getenv("MAX_UNITS_JPY", "3000"))
MAX_UNITS_NON_JPY = int(os.getenv("MAX_UNITS_NON_JPY", "5000"))
DEFAULT_SL_ATR = float(os.getenv("DEFAULT_SL_ATR", "1.0"))
DEFAULT_TP_ATR = float(os.getenv("DEFAULT_TP_ATR", "1.3"))
BAR_HISTORY_LEN = int(os.getenv("BAR_HISTORY_LEN", "300"))

PAIR_MAP: Dict[str, str] = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDCAD": "USD_CAD", "USDJPY": "USD_JPY",
    "USDCHF": "USD_CHF", "AUDUSD": "AUD_USD", "NZDUSD": "NZD_USD", "EURJPY": "EUR_JPY",
    "EURGBP": "EUR_GBP", "AUDJPY": "AUD_JPY", "CADJPY": "CAD_JPY", "GBPJPY": "GBP_JPY",
    "EURCHF": "EUR_CHF", "NZDJPY": "NZD_JPY", "GBPCHF": "GBP_CHF", "CHFJPY": "CHF_JPY",
    "AUDCAD": "AUD_CAD", "AUDNZD": "AUD_NZD",
}
INSTRUMENT_TO_PAIR6 = {v: k for k, v in PAIR_MAP.items()}
JPY_INSTRUMENTS = {v for v in PAIR_MAP.values() if v.endswith("_JPY")}

_recent_signals: Dict[str, deque] = {}
_bar_history: Dict[str, deque] = {pair6: deque(maxlen=BAR_HISTORY_LEN) for pair6 in PAIR_MAP}
_trade_count_today: Dict[str, int] = {}
_trade_day = dt.datetime.now(dt.timezone.utc).date()
_open_trade_ids: set[str] = set()

def utc_ts() -> str: return dt.datetime.now(dt.timezone.utc).isoformat()
def now_utc() -> dt.datetime: return dt.datetime.now(dt.timezone.utc)
def now_unix() -> int: return int(now_utc().timestamp())

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

def read_audit_df() -> pd.DataFrame: return read_csv_df(AUDIT_CSV)
def read_trades_df() -> pd.DataFrame: return read_csv_df(TRADES_CSV)

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

def note_trade_opened(order_id: Optional[str]) -> None:
    if order_id:
        _open_trade_ids.add(order_id)

def note_trade_closed(order_id: Optional[str]) -> None:
    if order_id and order_id in _open_trade_ids:
        _open_trade_ids.remove(order_id)

def make_signal_fingerprint(instrument: str, side: str, bar_time: int, mid_c: float, tf: Optional[str]) -> str:
    raw = {"instrument": instrument, "side": side, "bar_time": int(bar_time), "mid_c": round(float(mid_c), instrument_precision(instrument)), "tf": tf or ""}
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()

def is_duplicate_signal(pair6: str, fingerprint: str) -> bool:
    tnow = now_unix()
    q = _recent_signals.setdefault(pair6, deque())
    while q and (tnow - q[0][0] > DUP_WINDOW_SECONDS):
        q.popleft()
    return any(fp == fingerprint for _, fp in q)

def remember_signal(pair6: str, fingerprint: str) -> None:
    _recent_signals.setdefault(pair6, deque()).append((now_unix(), fingerprint))

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
                q.append({"t": int(pd.Timestamp(r["time"]).timestamp()), "time": r["time"], "mid_o": float(r["mid_o"]), "mid_h": float(r["mid_h"]), "mid_l": float(r["mid_l"]), "mid_c": float(r["mid_c"])})
            _bar_history[pair6] = q
        except Exception:
            continue

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
        "mid_o": mid_o, "mid_h": mid_h, "mid_l": mid_l, "mid_c": mid_c,
        "spread_c": spread_c, "hr": safe_float(payload.get("hr"), float(ts.hour)), "t": safe_int(payload.get("t")),
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
    mid_l = safe_float(payload.get("mid_l")); mid_c = safe_float(payload.get("mid_c")); mid_h = safe_float(payload.get("mid_h")); mid_o = safe_float(payload.get("mid_o"))
    if not (mid_l <= mid_c <= mid_h): return "Bad payload: mid_c not between mid_l and mid_h"
    if not (mid_l <= mid_o <= mid_h): return "Bad payload: mid_o not between mid_l and mid_h"
    if mid_h < mid_l: return "Bad payload: mid_h < mid_l"
    if safe_float(payload.get("spread_pips"), 0.0) < 0: return "Bad payload: negative spread_pips"
    if safe_float(payload.get("spread_atr"), 0.0) < 0: return "Bad payload: negative spread_atr"
    return None

def pair_live_stats(instrument: str) -> Dict[str, Any]:
    df = read_closed_trades_df()
    if df.empty or "instrument" not in df.columns:
        return {"n": 0, "win_rate": None, "avg_pnl": None, "net_pnl": 0.0}
    sub = df[df["instrument"] == instrument].copy()
    if sub.empty:
        return {"n": 0, "win_rate": None, "avg_pnl": None, "net_pnl": 0.0}
    pnl = pd.to_numeric(sub.get("pnl"), errors="coerce").fillna(0.0)
    n = int(len(sub)); wins = int((pnl > 0).sum())
    return {"n": n, "win_rate": (wins / n if n else None), "avg_pnl": float(pnl.mean()) if n else None, "net_pnl": float(pnl.sum())}

def compute_pair_score(instrument: str, avg_auc: float) -> float:
    live = pair_live_stats(instrument)
    n = int(live["n"])
    auc_norm = max(0.0, min(1.0, (avg_auc - 0.50) / 0.10))
    if n < MIN_TRADES_FOR_PAIR_SCORING or live["win_rate"] is None:
        return auc_norm
    wr = max(0.0, min(1.0, float(live["win_rate"])))
    return max(0.0, min(1.0, (AUC_WEIGHT * auc_norm) + (WINRATE_WEIGHT * wr)))

def compute_units_dynamic(instrument: str, sl_pips: float, avg_auc: float, pair_score: float, equity_used: float, force_units_abs: Optional[int] = None) -> int:
    if force_units_abs is not None:
        return max(1, abs(int(force_units_abs)))
    if sl_pips is None or sl_pips <= 0:
        return 0
    base = base_units_for_instrument(instrument)
    if USE_EQUITY_SIZING:
        risk_cap = equity_used * RISK_PCT
        risk_per_1000 = float(sl_pips) * pip_value_per_1000(instrument)
        base = int((risk_cap / risk_per_1000) * 1000) if risk_per_1000 > 0 else base
    if pair_score >= 0.80: base = int(base * 1.35)
    elif pair_score >= 0.65: base = int(base * 1.15)
    elif pair_score < 0.50: base = int(base * 0.70)
    if avg_auc >= 0.57: base = int(base * 1.10)
    elif avg_auc < 0.54: base = int(base * 0.90)
    return min(max_units_for_instrument(instrument), max(min_units_for_instrument(instrument), base))

def compute_sl_tp_prices(side: str, mid_c: float, atr14: float, instrument: str, sl_atr: float, tp_atr: float, min_dist_pips: float = 5.0) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if side not in ("BUY", "SELL"): return (None, None, None, None)
    pip = instrument_pip_size(instrument)
    atrv = max(float(atr14), pip)
    sl_dist = max(sl_atr * atrv, min_dist_pips * pip)
    tp_dist = max(tp_atr * atrv, min_dist_pips * pip)
    if side == "BUY":
        sl_price = _round_down_to_pip(mid_c - sl_dist, pip)
        tp_price = _round_up_to_pip(mid_c + tp_dist, pip)
        if sl_price >= mid_c: sl_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)
        if tp_price <= mid_c: tp_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
    else:
        sl_price = _round_up_to_pip(mid_c + sl_dist, pip)
        tp_price = _round_down_to_pip(mid_c - tp_dist, pip)
        if sl_price <= mid_c: sl_price = _round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
        if tp_price >= mid_c: tp_price = _round_down_to_pip(mid_c - (min_dist_pips * pip), pip)
    sl_str = format_oanda_price(sl_price, instrument)
    tp_str = format_oanda_price(tp_price, instrument)
    mid_str = format_oanda_price(mid_c, instrument)
    sl_price_f = float(sl_str); tp_price_f = float(tp_str); mid_c_f = float(mid_str)
    sl_pips = abs(mid_c_f - sl_price_f) / pip
    tp_pips = abs(tp_price_f - mid_c_f) / pip
    return float(sl_pips), float(tp_pips), sl_str, tp_str

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

BUNDLES = load_bundles(MODELS_DIR)

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

app = FastAPI(title="FX Sniper Per Pair", version="6.0")

@app.on_event("startup")
def _startup() -> None:
    seed_history_from_csv(DATA_DIR)

def build_response_base(p: TVPayload, pair: str, instrument: str, model_version: str, avg_auc: float, pair_score: Optional[float], equity_used: float, hint_side: str, conf: float = 0.0, side_prob: float = 0.0, p_up: float = 0.0, margin: float = 0.0) -> Dict[str, Any]:
    return {
        "ts": utc_ts(), "pair": pair, "instrument": instrument, "symbol": p.symbol, "hint_side": hint_side,
        "model_version": model_version, "avg_auc": avg_auc, "pair_score": pair_score, "equity_used": equity_used,
        "trend_regime": int(getattr(p, "trend_regime", 0) or 0), "vol_regime": int(getattr(p, "vol_regime", 0) or 0),
        "spread_pips": float(getattr(p, "spread_pips", 0.0) or 0.0), "spread_atr": float(getattr(p, "spread_atr", 0.0) or 0.0),
        "confidence": float(conf), "side_prob": float(side_prob), "p_up": float(p_up), "margin": float(margin),
    }

@app.post("/predict")
def predict(p: TVPayload):
    pair6 = normalize_pair(p.symbol)
    hint_side = str(getattr(p, "hint_side", "") or "").upper()
    equity_used = get_equity_used(p)
    if pair6 is None or pair6 not in PAIR_MAP:
        out = make_out(decision="NONE", why="Symbol not allowed", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **build_response_base(p, "", "", "", 0.0, None, equity_used, hint_side))
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
        out = make_out(decision="NONE", why="Model not loaded for symbol", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **build_response_base(p, instrument, instrument, "", 0.0, None, equity_used, hint_side))
        write_audit_row(out)
        return out
    model_version = b["model_version"]
    avg_auc = safe_float(b.get("avg_auc"), 0.0)
    pair_score = compute_pair_score(instrument, avg_auc)
    bad_payload_reason = payload_sanity_checks(payload, instrument)
    if bad_payload_reason:
        out = make_out(decision="NONE", why=bad_payload_reason, would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **build_response_base(p, instrument, instrument, model_version, avg_auc, pair_score, equity_used, hint_side))
        write_audit_row(out)
        return out
    gate = PAIR_GATES.get(instrument, DEFAULT_GATE)
    conf_gate = float(gate["conf"])
    margin_gate = float(gate["margin"])
    if p.force_decision in ("BUY", "SELL"):
        side = p.force_decision
        fingerprint = make_signal_fingerprint(instrument, side, p.t, float(p.mid_c), p.tf)
        base = build_response_base(p, instrument, instrument, model_version, avg_auc, pair_score, equity_used, hint_side, conf=1.0, side_prob=1.0, p_up=(1.0 if side == "BUY" else 0.0), margin=1.0)
        if is_duplicate_signal(pair6, fingerprint):
            out = make_out(decision="NONE", why=f"Duplicate signal blocked for {instrument}", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        if trades_today_total() >= MAX_TRADES_PER_DAY_TOTAL or trades_today(pair6) >= MAX_TRADES_PER_DAY_PER_PAIR:
            out = make_out(decision="NONE", why=f"Daily lock: max trades reached ({instrument})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        if not can_open_trade():
            out = make_out(decision="NONE", why=f"Open trade cap reached ({MAX_OPEN_TRADES})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(side, float(p.mid_c), float(p.atr14), instrument, b["labeling"]["sl_atr"], b["labeling"]["tp_atr"])
        units_abs = compute_units_dynamic(instrument, sl_pips, avg_auc, pair_score, equity_used, p.force_units_abs)
        units_signed = units_abs if side == "BUY" else -units_abs
        out = make_out(decision=side, why="FORCED decision (bypassed model/gates)", would_order=True, units=units_abs, units_signed=units_signed, sl_pips=sl_pips, tp_pips=tp_pips, sl_price=sl_price, tp_price=tp_price, **base)
        remember_signal(pair6, fingerprint); inc_trade(pair6); write_audit_row(out); return out
    try:
        feat_order = b["feature_order"]
        feature_row = build_runtime_feature_row(payload, pair6, instrument, feat_order)
        X = pd.DataFrame([{f: feature_row.get(f, 0.0) for f in feat_order}], columns=feat_order)
        model = b["model"]
        proba = model.predict_proba(X)[0]
        p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
        side = "BUY" if p_up >= 0.5 else "SELL"
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
        base = build_response_base(p, instrument, instrument, model_version, avg_auc, pair_score, equity_used, hint_side, conf=conf, side_prob=side_prob, p_up=p_up, margin=margin)
        disagree_conf_gate = PAIR_DISAGREE_CONF.get(instrument, DEFAULT_DISAGREE_CONF)
        hint_disagrees = hint_side in ("BUY", "SELL") and side != hint_side
        if hint_disagrees and conf < disagree_conf_gate:
            out = make_out(decision="NONE", why=f"Blocked disagreement: ML {side} vs hint {hint_side} (conf {conf:.2f} < {disagree_conf_gate:.2f})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        if pair_score < MIN_PAIR_SCORE_TO_TRADE:
            out = make_out(decision="NONE", why=f"Pair blocked: {instrument} score {pair_score:.2f} < {MIN_PAIR_SCORE_TO_TRADE:.2f}", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        would_order = (conf >= conf_gate) and (margin >= margin_gate)
        fingerprint = make_signal_fingerprint(instrument, side, p.t, float(p.mid_c), p.tf)
        if would_order and is_duplicate_signal(pair6, fingerprint):
            out = make_out(decision="NONE", why=f"Duplicate signal blocked for {instrument}", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        if would_order and trades_today_total() >= MAX_TRADES_PER_DAY_TOTAL:
            out = make_out(decision="NONE", why=f"Daily lock: total max trades reached ({MAX_TRADES_PER_DAY_TOTAL})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        if would_order and trades_today(pair6) >= MAX_TRADES_PER_DAY_PER_PAIR:
            out = make_out(decision="NONE", why=f"Daily lock: max trades for {instrument} reached", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        if would_order and not can_open_trade():
            out = make_out(decision="NONE", why=f"Open trade cap reached ({MAX_OPEN_TRADES})", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **base)
            write_audit_row(out); return out
        units_abs = units_signed = sl_pips = tp_pips = sl_price = tp_price = None
        why = f"Below sniper gate | conf={conf:.2f}/{conf_gate:.2f}, margin={margin:.2f}/{margin_gate:.2f}, hint={hint_side or 'NONE'}, pair_score={pair_score:.2f}"
        decision = "NONE"
        if would_order:
            sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(side, float(p.mid_c), float(p.atr14), instrument, b["labeling"]["sl_atr"], b["labeling"]["tp_atr"])
            units_abs = compute_units_dynamic(instrument, sl_pips, avg_auc, pair_score, equity_used, p.force_units_abs)
            units_signed = units_abs if side == "BUY" else -units_abs
            why = f"OK: {side} passed | conf={conf:.2f}/{conf_gate:.2f}, margin={margin:.2f}/{margin_gate:.2f}, hint={hint_side or 'NONE'}, pair_score={pair_score:.2f}, equity_used={equity_used:.2f}"
            decision = side
        out = make_out(decision=decision, why=why, would_order=bool(would_order), units=units_abs, units_signed=units_signed, sl_pips=sl_pips, tp_pips=tp_pips, sl_price=sl_price, tp_price=tp_price, **base)
        if would_order:
            remember_signal(pair6, fingerprint); inc_trade(pair6)
        write_audit_row(out); return out
    except Exception as e:
        out = make_out(decision="NONE", why=f"Prediction error: {repr(e)}", would_order=False, units=None, units_signed=None, sl_pips=None, tp_pips=None, sl_price=None, tp_price=None, **build_response_base(p, instrument, instrument, model_version, avg_auc, pair_score, equity_used, hint_side))
        write_audit_row(out); return out


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
        return {"ok": True, "rows": 0, "would_order_count": 0, "decision_counts": {}, "pair_counts": {}, "last_ts": None}
    decision_counts = df["decision"].value_counts(dropna=False).to_dict() if "decision" in df.columns else {}
    pair_counts = df["instrument"].value_counts(dropna=False).to_dict() if "instrument" in df.columns else {}
    would_count = int(safe_bool_series(df, "would_order").sum())
    last_ts = None
    if "ts" in df.columns and not df["ts"].dropna().empty:
        last_ts = df["ts"].dropna().max().isoformat()
    return {"ok": True, "rows": int(len(df)), "would_order_count": would_count, "decision_counts": decision_counts, "pair_counts": pair_counts, "last_ts": last_ts}

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
    return {"ok": True, "trades": int(len(df)), "closed_trades": closed_n, "wins": wins, "losses": losses, "win_rate": win_rate, "net_pnl": float(pnl.sum()), "avg_pnl": float(pnl.mean()) if closed_n else None, "open_trades": current_open_trade_count()}

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
        pair6 = instrument.replace("_", "")
        avg_auc = safe_float(BUNDLES.get(pair6, {}).get("avg_auc"), 0.0)
        pair_score = compute_pair_score(instrument, avg_auc)
        rows.append({"instrument": instrument, "trades": n, "wins": wins, "losses": losses, "win_rate": win_rate, "net_pnl": net_pnl, "avg_pnl": avg_pnl, "avg_auc": avg_auc, "pair_score": pair_score, "is_tradeable": pair_score >= MIN_PAIR_SCORE_TO_TRADE})
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
        avg_auc = safe_float(BUNDLES.get(pair6, {}).get("avg_auc"), 0.0)
        pair_score = compute_pair_score(instrument, avg_auc)
        if pair_score < MIN_PAIR_SCORE_TO_TRADE:
            weak.append({"instrument": instrument, "trades": n, "win_rate": win_rate, "avg_auc": avg_auc, "pair_score": pair_score})
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
    cols = [c for c in ["ts", "instrument", "symbol", "hint_side", "decision", "confidence", "side_prob", "p_up", "margin", "pair_score", "equity_used", "units_signed", "sl_price", "tp_price", "would_order", "why"] if c in latest_rows.columns]
    table_html = latest_rows[cols].to_html(index=False, escape=False) if not latest_rows.empty else "<p>No audit data yet.</p>"
    by_pair_html = "<p>No audit data yet.</p>"
    if not audit_df.empty and "instrument" in audit_df.columns:
        by_pair = audit_df["instrument"].value_counts().rename_axis("instrument").reset_index(name="count")
        by_pair_html = by_pair.to_html(index=False)
    pnl_html = "<p>No trade data yet.</p>"
    if not trades_df.empty:
        pnl_cols = [c for c in ["ts", "instrument", "side", "units_signed", "status", "pnl", "reason", "order_id"] if c in trades_df.columns]
        pnl_html = trades_df.sort_values("ts", ascending=False).head(50)[pnl_cols].to_html(index=False, escape=False)
    html = f"""
    <html><head><title>FX Sniper Dashboard</title><meta http-equiv="refresh" content="15"></head>
    <body style="font-family: Arial; padding: 24px;">
      <h1>FX Sniper Dashboard</h1>
      <div style="display:flex; gap:24px; margin-bottom:24px; flex-wrap:wrap;">
        <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Total predictions</h3><div style="font-size:28px;">{total_rows}</div></div>
        <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Would order</h3><div style="font-size:28px;">{would_count}</div></div>
        <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Blocked / NONE</h3><div style="font-size:28px;">{none_count}</div></div>
        <div style="padding:16px; border:1px solid #ccc; border-radius:8px;"><h3>Open trades tracked</h3><div style="font-size:28px;">{current_open_trade_count()}</div></div>
      </div>
      <h2>By pair</h2>{by_pair_html}
      <h2>Latest predictions</h2>{table_html}
      <h2>Latest trade events</h2>{pnl_html}
      <p style="margin-top:24px;">JSON endpoints: <a href="/health">/health</a> | <a href="/stats">/stats</a> | <a href="/pnl_stats">/pnl_stats</a> | <a href="/pair_stats">/pair_stats</a> | <a href="/weak_pairs">/weak_pairs</a></p>
    </body></html>
    """
    return HTMLResponse(content=html)

@app.post("/trade_event")
def trade_event(t: TradeEvent):
    row = t.model_dump()
    if not row.get("ts"):
        row["ts"] = utc_ts()
    write_trade_row(row)
    if t.status == "OPEN":
        note_trade_opened(t.order_id)
    if t.status in ("CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"):
        note_trade_closed(t.order_id)
    return {"ok": True, "open_trades": current_open_trade_count(), "status": t.status, "order_id": t.order_id}
