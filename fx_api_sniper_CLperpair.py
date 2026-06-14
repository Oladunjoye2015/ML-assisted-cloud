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

# ============================================================
# APP / PATHS
# ============================================================
APP_VERSION = "fx-m15-v16-side-aware-ai-review"

MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOG_DIR = os.getenv("LOG_DIR", "logs")
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.getenv("DB_PATH", os.path.join(LOG_DIR, "fx_m15_signal_approval.db"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

AUDIT_CSV = os.path.join(LOG_DIR, "audit_m15_signal_approval.csv")
TRADES_CSV = os.path.join(LOG_DIR, "trades_m15_signal_approval.csv")
TRADE_MANAGEMENT_CSV = os.path.join(LOG_DIR, "trade_management_m15_signal_approval.csv")

# ============================================================
# MODEL APPROVAL / TRADING SETTINGS
# ============================================================
APPROVAL_GATE = float(os.getenv("CONF_GATE", os.getenv("APPROVAL_GATE", "0.60")))
APPROVAL_MARGIN_GATE = float(os.getenv("MARGIN_GATE", os.getenv("APPROVAL_MARGIN_GATE", "0.10")))

LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "120"))
AUTO_CLOSE_ENABLED = os.getenv("AUTO_CLOSE_ENABLED", "false").lower() == "true"
AUTO_CLOSE_CHECK_SECONDS = int(os.getenv("AUTO_CLOSE_CHECK_SECONDS", "21600"))
AUTO_CLOSE_ALLOW_POSITION_FALLBACK = os.getenv("AUTO_CLOSE_ALLOW_POSITION_FALLBACK", "false").lower() == "true"

OANDA_TOKEN = os.getenv("OANDA_TOKEN", "").strip()
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "").strip()
OANDA_BASE_URL = os.getenv("OANDA_BASE_URL", "https://api-fxpractice.oanda.com").strip().rstrip("/")

UNITS_JPY = int(os.getenv("UNITS_JPY", "1000"))
UNITS_NON_JPY = int(os.getenv("UNITS_NON_JPY", "1000"))
MIN_UNITS_JPY = int(os.getenv("MIN_UNITS_JPY", "100"))
MIN_UNITS_NON_JPY = int(os.getenv("MIN_UNITS_NON_JPY", "100"))
MAX_UNITS_JPY = int(os.getenv("MAX_UNITS_JPY", "3000"))
MAX_UNITS_NON_JPY = int(os.getenv("MAX_UNITS_NON_JPY", "5000"))

MAX_TRADES_PER_DAY_TOTAL = int(os.getenv("MAX_TRADES_PER_DAY_TOTAL", "5"))
MAX_TRADES_PER_DAY_PER_PAIR = int(os.getenv("MAX_TRADES_PER_DAY_PER_PAIR", "2"))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "1"))
# Prevent two trades from being placed at the same time.
# PENDING_TRADE_LOCK_ENABLED reserves a slot as soon as /predict returns would_order=true,
# even before Make sends the OPEN /trade_event after OANDA fill. This closes the race window
# where two TradingView alerts could both pass before the first trade is registered as OPEN.
PENDING_TRADE_LOCK_ENABLED = os.getenv("PENDING_TRADE_LOCK_ENABLED", "true").lower() == "true"
PENDING_TRADE_TIMEOUT_SECONDS = int(os.getenv("PENDING_TRADE_TIMEOUT_SECONDS", "600"))
DUP_WINDOW_SECONDS = int(os.getenv("DUP_WINDOW_SECONDS", "300"))

MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "3.0"))
MIN_ATR_NON_JPY = float(os.getenv("MIN_ATR_NON_JPY", "0.00003"))
MIN_ATR_JPY = float(os.getenv("MIN_ATR_JPY", "0.003"))

USE_EQUITY_SIZING = os.getenv("USE_EQUITY_SIZING", "true").lower() == "true"
DEFAULT_EQUITY = float(os.getenv("DEFAULT_EQUITY", "200"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.005"))
DEFAULT_SL_ATR = float(os.getenv("DEFAULT_SL_ATR", "1.0"))
DEFAULT_TP_ATR = float(os.getenv("DEFAULT_TP_ATR", "1.3"))
BAR_HISTORY_LEN = int(os.getenv("BAR_HISTORY_LEN", "300"))

# ============================================================
# SIGNAL-CONDITIONED APPROVAL SETTINGS
# ============================================================
REQUIRE_HINT_SIDE = os.getenv("REQUIRE_HINT_SIDE", "true").lower() == "true"

# ============================================================
# STRICT LOAD FILTERS + EARLY BLOCK CONTROL
# ============================================================
STRICT_MODEL_FILTER_ENABLED = os.getenv("STRICT_MODEL_FILTER_ENABLED", "false").lower() == "true"
# When false (recommended), weak selected-primary metrics do NOT block the whole pair early.
# The server still records static_filter_passed/reason, but continues to score the primary,
# score all candidates, allow fallback rescue, and collect data for auto-switch.
STATIC_FILTER_EARLY_BLOCK_ENABLED = os.getenv("STATIC_FILTER_EARLY_BLOCK_ENABLED", "false").lower() == "true"
LIVE_MIN_AUC = float(os.getenv("LIVE_MIN_AUC", "0.55"))
LIVE_MIN_PRECISION = float(os.getenv("LIVE_MIN_PRECISION", "0.35"))
LIVE_MIN_TRADES_AT_GATE = int(os.getenv("LIVE_MIN_TRADES_AT_GATE", "300"))

PRIMARY_LIVE_PAIRS = {
    x.strip().upper().replace("_", "")
    for x in os.getenv(
        "PRIMARY_LIVE_PAIRS",
        "AUDUSD,EURGBP,EURJPY,EURUSD,GBPCHF,GBPJPY,GBPUSD,NZDUSD,USDCAD,USDCHF,USDJPY",
    ).split(",")
    if x.strip()
}

PRIMARY_MIN_AUC_FOR_ORDER = float(os.getenv("PRIMARY_MIN_AUC_FOR_ORDER", "0.51"))
PRIMARY_MIN_PRECISION_FOR_ORDER = float(os.getenv("PRIMARY_MIN_PRECISION_FOR_ORDER", "0.31"))
PRIMARY_MIN_TRADES_AT_GATE_FOR_ORDER = int(os.getenv("PRIMARY_MIN_TRADES_AT_GATE_FOR_ORDER", "200"))

# ============================================================
# FALLBACK / SHADOW MODEL SETTINGS
# ============================================================
FALLBACK_MODE_ENABLED = os.getenv("FALLBACK_MODE_ENABLED", "true").lower() == "true"
FALLBACK_CONF_EDGE = float(os.getenv("FALLBACK_CONF_EDGE", "0.04"))
FALLBACK_MARGIN_EDGE = float(os.getenv("FALLBACK_MARGIN_EDGE", "0.03"))
FALLBACK_MIN_AUC = float(os.getenv("FALLBACK_MIN_AUC", "0.55"))
FALLBACK_MIN_PRECISION = float(os.getenv("FALLBACK_MIN_PRECISION", "0.35"))
FALLBACK_MIN_TRADES_AT_GATE = int(os.getenv("FALLBACK_MIN_TRADES_AT_GATE", "300"))

# ============================================================
# AUTO MODEL SWITCH SETTINGS
# ============================================================
AUTO_MODEL_SWITCH_ENABLED = os.getenv("AUTO_MODEL_SWITCH_ENABLED", "true").lower() == "true"
SWITCH_MIN_ALERTS = int(os.getenv("SWITCH_MIN_ALERTS", "25"))
SWITCH_MIN_CANDIDATE_WOULD_ORDERS = int(os.getenv("SWITCH_MIN_CANDIDATE_WOULD_ORDERS", "3"))
SWITCH_MIN_CONF_EDGE = float(os.getenv("SWITCH_MIN_CONF_EDGE", "0.03"))
SWITCH_MIN_AUC = float(os.getenv("SWITCH_MIN_AUC", "0.55"))
SWITCH_MIN_PRECISION = float(os.getenv("SWITCH_MIN_PRECISION", "0.35"))
SWITCH_MIN_TRADES_AT_GATE = int(os.getenv("SWITCH_MIN_TRADES_AT_GATE", "300"))
SWITCH_MAX_PRIMARY_WOULD_ORDERS = int(os.getenv("SWITCH_MAX_PRIMARY_WOULD_ORDERS", "0"))
SWITCH_COOLDOWN_MINUTES = int(os.getenv("SWITCH_COOLDOWN_MINUTES", "360"))
SWITCH_LOOKBACK_EVENTS = int(os.getenv("SWITCH_LOOKBACK_EVENTS", "250"))

# ============================================================
# M15 NOISE FILTER
# ============================================================
NOISE_FILTER_ENABLED = os.getenv("NOISE_FILTER_ENABLED", "true").lower() == "true"
MIN_NOISE_RANGE_PIPS = float(os.getenv("MIN_NOISE_RANGE_PIPS", "1.5"))
MIN_NOISE_ATR_PIPS = float(os.getenv("MIN_NOISE_ATR_PIPS", "3.0"))
MIN_BODY_RANGE_RATIO = float(os.getenv("MIN_BODY_RANGE_RATIO", "0.18"))
MIN_RANGE_ATR_RATIO = float(os.getenv("MIN_RANGE_ATR_RATIO", "0.25"))
MAX_SPREAD_RANGE_RATIO = float(os.getenv("MAX_SPREAD_RANGE_RATIO", "0.40"))
MAX_WICK_BODY_RATIO = float(os.getenv("MAX_WICK_BODY_RATIO", "6.0"))
NOISE_FILTER_REQUIRE_SIGNAL_MOMENTUM_ALIGNMENT = os.getenv("NOISE_FILTER_REQUIRE_SIGNAL_MOMENTUM_ALIGNMENT", "false").lower() == "true"

# ============================================================
# NEWS FILTER
# ============================================================
NEWS_FILTER_ENABLED = os.getenv("NEWS_FILTER_ENABLED", "true").lower() == "true"
NEWS_BLOCK_BEFORE_MIN = int(os.getenv("NEWS_BLOCK_BEFORE_MIN", "45"))
NEWS_BLOCK_AFTER_MIN = int(os.getenv("NEWS_BLOCK_AFTER_MIN", "15"))
NEWS_BLOCK_IMPACTS = {
    x.strip().upper()
    for x in os.getenv("NEWS_BLOCK_IMPACTS", "HIGH,RED").split(",")
    if x.strip()
}
NEWS_EVENTS_FILE = os.getenv("NEWS_EVENTS_FILE", os.path.join(DATA_DIR, "news_events.json"))
NEWS_EVENTS_JSON = os.getenv("NEWS_EVENTS_JSON", "").strip()
NEWS_MANUAL_BLACKOUT_UTC = os.getenv("NEWS_MANUAL_BLACKOUT_UTC", "").strip()
NEWS_BLOCK_ALL_CURRENCIES = os.getenv("NEWS_BLOCK_ALL_CURRENCIES", "false").lower() == "true"
NEWS_BLOCK_UNKNOWN_CURRENCY = os.getenv("NEWS_BLOCK_UNKNOWN_CURRENCY", "false").lower() == "true"
NEWS_KEEP_PAST_HOURS = int(os.getenv("NEWS_KEEP_PAST_HOURS", "24"))
NEWS_DEFAULT_TITLE = os.getenv("NEWS_DEFAULT_TITLE", "economic_news")

# ============================================================
# LIVE OANDA PRICE GUARD FOR TP/SL VALIDATION
# ============================================================
LIVE_PRICE_GUARD_ENABLED = os.getenv("LIVE_PRICE_GUARD_ENABLED", "true").lower() == "true"
LIVE_PRICE_GUARD_REQUIRED = os.getenv("LIVE_PRICE_GUARD_REQUIRED", "true").lower() == "true"
LIVE_PRICE_REPRICE_SLTP = os.getenv("LIVE_PRICE_REPRICE_SLTP", "true").lower() == "true"
LIVE_PRICE_BUFFER_PIPS = float(os.getenv("LIVE_PRICE_BUFFER_PIPS", "1.0"))
LIVE_PRICE_MAX_AGE_SECONDS = int(os.getenv("LIVE_PRICE_MAX_AGE_SECONDS", "15"))

# ============================================================
# STALE SIGNAL + ENTRY REVERSAL GUARD
# Blocks old/replayed TradingView alerts and blocks fresh alerts if
# live price has already reversed against the hinted direction before order submission.
# IMPORTANT: current Pine sends t = bar open time, so M15 bar-close alerts can be ~900s old.
# Default 1200s blocks old Make replays while allowing normal M15 close alerts.
# ============================================================
SIGNAL_STALENESS_GUARD_ENABLED = os.getenv("SIGNAL_STALENESS_GUARD_ENABLED", "true").lower() == "true"
SIGNAL_MAX_AGE_SECONDS = int(os.getenv("SIGNAL_MAX_AGE_SECONDS", "1200"))

ENTRY_REVERSAL_GUARD_ENABLED = os.getenv("ENTRY_REVERSAL_GUARD_ENABLED", "true").lower() == "true"
ENTRY_REVERSAL_GUARD_REQUIRED = os.getenv("ENTRY_REVERSAL_GUARD_REQUIRED", "true").lower() == "true"
ENTRY_REVERSAL_MAX_ADVERSE_PIPS = float(os.getenv("ENTRY_REVERSAL_MAX_ADVERSE_PIPS", "1.5"))
ENTRY_REVERSAL_MAX_SPREAD_PIPS = float(os.getenv("ENTRY_REVERSAL_MAX_SPREAD_PIPS", "2.5"))
ENTRY_REVERSAL_REQUIRE_MOMENTUM_ALIGN = os.getenv("ENTRY_REVERSAL_REQUIRE_MOMENTUM_ALIGN", "false").lower() == "true"
ENTRY_REVERSAL_EMA20_SIDE_ENABLED = os.getenv("ENTRY_REVERSAL_EMA20_SIDE_ENABLED", "false").lower() == "true"
ENTRY_REVERSAL_EMA20_BUFFER = float(os.getenv("ENTRY_REVERSAL_EMA20_BUFFER", "0.0"))

# ============================================================
# DIRECTION CONSENSUS GUARD
# Blocks trades when the TradingView hint is fighting the actual M15 candle/momentum/trend context.
# This prevents SELL alerts during a strong BUY push and BUY alerts during a strong SELL dump.
# ============================================================
DIRECTION_CONFIRMATION_ENABLED = os.getenv("DIRECTION_CONFIRMATION_ENABLED", "true").lower() == "true"
DIRECTION_CONFIRMATION_REQUIRED = os.getenv("DIRECTION_CONFIRMATION_REQUIRED", "true").lower() == "true"
DIRECTION_CONFIRM_MIN_SCORE = int(os.getenv("DIRECTION_CONFIRM_MIN_SCORE", "4"))
DIRECTION_CONFIRM_REQUIRE_EMA20_SIDE = os.getenv("DIRECTION_CONFIRM_REQUIRE_EMA20_SIDE", "true").lower() == "true"
DIRECTION_CONFIRM_REQUIRE_RET3_ALIGN = os.getenv("DIRECTION_CONFIRM_REQUIRE_RET3_ALIGN", "true").lower() == "true"
DIRECTION_CONFIRM_REQUIRE_CANDLE_ALIGN = os.getenv("DIRECTION_CONFIRM_REQUIRE_CANDLE_ALIGN", "false").lower() == "true"
DIRECTION_CONFIRM_REQUIRE_MACD_ALIGN = os.getenv("DIRECTION_CONFIRM_REQUIRE_MACD_ALIGN", "false").lower() == "true"
DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA50 = os.getenv("DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA50", "true").lower() == "true"
DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA200 = os.getenv("DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA200", "false").lower() == "true"
DIRECTION_CONFIRM_EMA_BUFFER_PIPS = float(os.getenv("DIRECTION_CONFIRM_EMA_BUFFER_PIPS", "0.2"))
DIRECTION_CONFIRM_BLOCK_STRONG_OPPOSITE_CANDLE = os.getenv("DIRECTION_CONFIRM_BLOCK_STRONG_OPPOSITE_CANDLE", "true").lower() == "true"
DIRECTION_CONFIRM_STRONG_BODY_RATIO = float(os.getenv("DIRECTION_CONFIRM_STRONG_BODY_RATIO", "0.35"))
DIRECTION_CONFIRM_MIN_BODY_PIPS = float(os.getenv("DIRECTION_CONFIRM_MIN_BODY_PIPS", "1.0"))
DIRECTION_CONFIRM_MIN_RET3_ABS = float(os.getenv("DIRECTION_CONFIRM_MIN_RET3_ABS", "0.0"))
DIRECTION_CONFIRM_MIN_RET5_ABS = float(os.getenv("DIRECTION_CONFIRM_MIN_RET5_ABS", "0.0"))

# ============================================================
# OPEN TRADE PROFIT PROTECTION
# Server-managed protective stop updates using actual live OANDA prices.
# Uses R-multiples based on the original stop distance:
#   R = favorable_move_pips / initial_stop_distance_pips
# ============================================================
PROFIT_PROTECTION_ENABLED = os.getenv("PROFIT_PROTECTION_ENABLED", "true").lower() == "true"
OPEN_TRADE_MANAGER_CHECK_SECONDS = int(os.getenv("OPEN_TRADE_MANAGER_CHECK_SECONDS", "60"))
BREAKEVEN_TRIGGER_R = float(os.getenv("BREAKEVEN_TRIGGER_R", "0.75"))
BREAKEVEN_BUFFER_PIPS = float(os.getenv("BREAKEVEN_BUFFER_PIPS", "0.5"))
TRAILING_TRIGGER_R = float(os.getenv("TRAILING_TRIGGER_R", "1.0"))
TRAILING_DISTANCE_R = float(os.getenv("TRAILING_DISTANCE_R", "0.80"))
TRAILING_MIN_IMPROVEMENT_PIPS = float(os.getenv("TRAILING_MIN_IMPROVEMENT_PIPS", "0.5"))
STOP_UPDATE_LIVE_BUFFER_PIPS = float(os.getenv("STOP_UPDATE_LIVE_BUFFER_PIPS", "0.3"))
PROFIT_PROTECTION_REQUIRE_TRADE_SPECIFIER = os.getenv("PROFIT_PROTECTION_REQUIRE_TRADE_SPECIFIER", "true").lower() == "true"

# Early reversal profit lock:
# Closes a winning trade at market when it starts giving back profit
# before the normal breakeven/trailing thresholds can protect it.
REVERSAL_EXIT_ENABLED = os.getenv("REVERSAL_EXIT_ENABLED", "true").lower() == "true"
REVERSAL_EXIT_MIN_PROFIT_R = float(os.getenv("REVERSAL_EXIT_MIN_PROFIT_R", "0.35"))
REVERSAL_EXIT_MIN_PROFIT_PIPS = float(os.getenv("REVERSAL_EXIT_MIN_PROFIT_PIPS", "3.0"))
REVERSAL_EXIT_GIVEBACK_R = float(os.getenv("REVERSAL_EXIT_GIVEBACK_R", "0.20"))
REVERSAL_EXIT_GIVEBACK_PIPS = float(os.getenv("REVERSAL_EXIT_GIVEBACK_PIPS", "2.0"))
REVERSAL_EXIT_MIN_CURRENT_PROFIT_PIPS = float(os.getenv("REVERSAL_EXIT_MIN_CURRENT_PROFIT_PIPS", "0.2"))

# Early adverse-start exit:
# Closes a trade that starts losing and stays weak instead of waiting for the full SL.
ADVERSE_EXIT_ENABLED = os.getenv("ADVERSE_EXIT_ENABLED", "true").lower() == "true"
ADVERSE_EXIT_AFTER_MINUTES = int(os.getenv("ADVERSE_EXIT_AFTER_MINUTES", "15"))
ADVERSE_EXIT_MIN_LOSS_R = float(os.getenv("ADVERSE_EXIT_MIN_LOSS_R", "0.35"))
ADVERSE_EXIT_MIN_LOSS_PIPS = float(os.getenv("ADVERSE_EXIT_MIN_LOSS_PIPS", "3.0"))
ADVERSE_EXIT_REQUIRE_NO_RECOVERY = os.getenv("ADVERSE_EXIT_REQUIRE_NO_RECOVERY", "true").lower() == "true"
ADVERSE_EXIT_MAX_PEAK_PROFIT_R = float(os.getenv("ADVERSE_EXIT_MAX_PEAK_PROFIT_R", "0.20"))
ADVERSE_EXIT_MAX_PEAK_PROFIT_PIPS = float(os.getenv("ADVERSE_EXIT_MAX_PEAK_PROFIT_PIPS", "2.0"))

# ============================================================
# CLOSED TRADE RECONCILIATION
# Registers TP/SL/manual broker closures that happen outside /trade_event.
# The server checks OANDA only while it is already tracking open trades.
# ============================================================
CLOSED_TRADE_SYNC_ENABLED = os.getenv("CLOSED_TRADE_SYNC_ENABLED", "true").lower() == "true"
CLOSED_TRADE_SYNC_CHECK_SECONDS = int(os.getenv("CLOSED_TRADE_SYNC_CHECK_SECONDS", "300"))
CLOSED_TRADE_SYNC_REQUIRE_TRADE_SPECIFIER = os.getenv("CLOSED_TRADE_SYNC_REQUIRE_TRADE_SPECIFIER", "true").lower() == "true"

# Closed-trade classification:
# When OANDA reports a trade as CLOSED, inspect the closing transaction(s)
# and store a specific status when possible:
#   TAKE_PROFIT -> Take Profit order filled
#   STOPPED     -> Stop Loss / Trailing Stop Loss / Guaranteed Stop Loss filled
#   MANUAL      -> Explicit client trade/position close
#   CLOSED      -> Closed but transaction reason was unknown/unavailable
CLOSED_TRADE_CLASSIFICATION_ENABLED = os.getenv("CLOSED_TRADE_CLASSIFICATION_ENABLED", "true").lower() == "true"
CLOSED_TRADE_CLASSIFICATION_MAX_TRANSACTIONS = int(os.getenv("CLOSED_TRADE_CLASSIFICATION_MAX_TRANSACTIONS", "5"))


# ============================================================
# EXTERNAL MARKET CONTEXT + AI REVIEW
# ============================================================
# The alert still provides the immediate signal/candle features.
# When enabled, the server also fetches fresh OANDA candles and compares
# alert data against market context before the final order is allowed.
MARKET_CONTEXT_ENABLED = os.getenv("MARKET_CONTEXT_ENABLED", "true").lower() == "true"
MARKET_CONTEXT_GRANULARITIES = [
    x.strip().upper()
    for x in os.getenv("MARKET_CONTEXT_GRANULARITIES", "M15,H1,H4").split(",")
    if x.strip()
]
MARKET_CONTEXT_CANDLE_COUNT = int(os.getenv("MARKET_CONTEXT_CANDLE_COUNT", "120"))
MARKET_CONTEXT_REQUIRED = os.getenv("MARKET_CONTEXT_REQUIRED", "false").lower() == "true"
MARKET_CONTEXT_MAX_FETCH_SECONDS = int(os.getenv("MARKET_CONTEXT_MAX_FETCH_SECONDS", "20"))

# ============================================================
# MODEL FEATURE SOURCE
# ============================================================
# alert  = model scores the TradingView/Make alert features.
# oanda  = model scores features calculated from latest completed OANDA candles.
# hybrid = TradingView provides the BUY/SELL hint, OANDA provides model features.
MODEL_FEATURE_SOURCE = os.getenv("MODEL_FEATURE_SOURCE", "hybrid").strip().lower()
if MODEL_FEATURE_SOURCE not in {"alert", "oanda", "hybrid"}:
    MODEL_FEATURE_SOURCE = "alert"
MODEL_FEATURE_OANDA_GRANULARITY = os.getenv("MODEL_FEATURE_OANDA_GRANULARITY", "M15").strip().upper()
MODEL_FEATURE_OANDA_CANDLE_COUNT = int(os.getenv("MODEL_FEATURE_OANDA_CANDLE_COUNT", "240"))
MODEL_FEATURE_OANDA_MIN_CANDLES = int(os.getenv("MODEL_FEATURE_OANDA_MIN_CANDLES", "80"))
MODEL_FEATURE_FALLBACK_TO_ALERT = os.getenv("MODEL_FEATURE_FALLBACK_TO_ALERT", "true").lower() == "true"
CANDLE_PATTERN_CONTEXT_ENABLED = os.getenv("CANDLE_PATTERN_CONTEXT_ENABLED", "true").lower() == "true"

# ============================================================
# PATTERN HISTORY / LEARNING MEMORY
# ============================================================
# Stores every signal pattern + AI/model decision, then updates outcomes when
# /trade_event reports OPEN/CLOSED. This does not make the model learn online;
# it creates the dataset and performance summary used by AI review and later retraining.
PATTERN_HISTORY_ENABLED = os.getenv("PATTERN_HISTORY_ENABLED", "true").lower() == "true"
PATTERN_HISTORY_INCLUDE_AI_CONTEXT = os.getenv("PATTERN_HISTORY_INCLUDE_AI_CONTEXT", "true").lower() == "true"
PATTERN_STATS_LOOKBACK = int(os.getenv("PATTERN_STATS_LOOKBACK", "500"))
PATTERN_STATS_MIN_CLOSED = int(os.getenv("PATTERN_STATS_MIN_CLOSED", "5"))
PATTERN_STATS_FOR_AI_ENABLED = os.getenv("PATTERN_STATS_FOR_AI_ENABLED", "true").lower() == "true"

AI_REVIEW_ENABLED = os.getenv("AI_REVIEW_ENABLED", "false").lower() == "true"
AI_REVIEW_PROVIDER = os.getenv("AI_REVIEW_PROVIDER", "openai").strip().lower()
AI_REVIEW_MODEL = os.getenv("AI_REVIEW_MODEL", "gpt-4o-mini").strip()

# Side-aware AI review gates.
# Normal allow: risk <= AI_REVIEW_MAX_RISK_SCORE
# Conditional allow: risk <= AI_REVIEW_CONDITIONAL_RISK_SCORE only when model probability is strong
# Hard block: risk >= AI_REVIEW_HARD_BLOCK_SCORE
AI_REVIEW_MAX_RISK_SCORE = int(os.getenv("AI_REVIEW_MAX_RISK_SCORE", "60"))
AI_REVIEW_CONDITIONAL_RISK_SCORE = int(os.getenv("AI_REVIEW_CONDITIONAL_RISK_SCORE", "75"))
AI_REVIEW_HARD_BLOCK_SCORE = int(os.getenv("AI_REVIEW_HARD_BLOCK_SCORE", "85"))
AI_REVIEW_MIN_MODEL_PROB = float(os.getenv("AI_REVIEW_MIN_MODEL_PROB", "0.52"))
AI_REVIEW_STRONG_MODEL_PROB = float(os.getenv("AI_REVIEW_STRONG_MODEL_PROB", "0.58"))
AI_REVIEW_MAX_SPREAD_ATR = float(os.getenv("AI_REVIEW_MAX_SPREAD_ATR", "0.18"))

# When API keys are missing, use the deterministic side-aware reviewer instead of blocking all trades.
AI_REVIEW_FALLBACK_TO_RULES = os.getenv("AI_REVIEW_FALLBACK_TO_RULES", "true").lower() == "true"
AI_REVIEW_TIMEOUT_SECONDS = int(os.getenv("AI_REVIEW_TIMEOUT_SECONDS", "25"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()


# ============================================================
# SIGNAL APPROVAL FEATURE ORDER
# ============================================================
FEATURE_COLS = [
    "side_sign",
    "setup_pullback",
    "setup_ema_cross",
    "ret1",
    "ret3",
    "ret5",
    "ema20_dist",
    "ema50_dist",
    "ema200_dist",
    "rsi14",
    "atr14_pct",
    "bb_width",
    "macd_hist",
    "vol_z",
    "spread_pips",
    "hour_utc",
    "dayofweek",
    "range_pips",
    "body_pips",
    "body_range_ratio",
    "atr_pips",
    "signal_momentum_aligned",
]

# ============================================================
# PAIRS
# ============================================================
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
_pending_trade_ids: set[str] = set()
_pending_trade_meta: Dict[str, Dict[str, Any]] = {}
BUNDLES: Dict[str, Dict[str, Any]] = {}
NEWS_EVENTS: List[Dict[str, Any]] = []

# ============================================================
# BASIC UTILS
# ============================================================
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
        value = float(x)
        if not np.isfinite(value):
            return default
        return value
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
    pair = str(symbol).strip().upper().replace("-", "").replace("_", "").replace("/", "")
    return pair if len(pair) == 6 and pair.isalpha() and pair in PAIR_MAP else None


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
    quantizer = Decimal("1." + ("0" * precision))
    value = Decimal(str(price)).quantize(quantizer, rounding=ROUND_HALF_UP)
    return f"{value:.{precision}f}"


def base_units_for_instrument(instrument: str) -> int:
    return UNITS_JPY if instrument_is_jpy(instrument) else UNITS_NON_JPY


def min_units_for_instrument(instrument: str) -> int:
    return MIN_UNITS_JPY if instrument_is_jpy(instrument) else MIN_UNITS_NON_JPY


def max_units_for_instrument(instrument: str) -> int:
    return MAX_UNITS_JPY if instrument_is_jpy(instrument) else MAX_UNITS_NON_JPY


def pip_value_per_1000(instrument: str) -> float:
    return 0.10


def normalize_side(side: Any) -> str:
    normalized = str(side or "").strip().upper()
    if normalized in ("BUY", "LONG", "BULL", "BULLISH"):
        return "BUY"
    if normalized in ("SELL", "SHORT", "BEAR", "BEARISH"):
        return "SELL"
    return normalized


def side_sign_from_hint(hint_side: str) -> float:
    return 1.0 if hint_side == "BUY" else -1.0 if hint_side == "SELL" else 0.0


def get_equity_used(payload_obj: Any) -> float:
    equity = safe_float(getattr(payload_obj, "equity", None), 0.0)
    nav = safe_float(getattr(payload_obj, "nav", None), 0.0)
    return equity if equity > 0 else nav if nav > 0 else DEFAULT_EQUITY


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

# ============================================================
# DB / CSV
# ============================================================
def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_sqlite_column(cur: sqlite3.Cursor, table_name: str, column_name: str, column_type: str) -> None:
    existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in existing:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


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
            pair_score REAL,
            tracking_key TEXT,
            broker_trade_id TEXT,
            broker_order_id TEXT,
            client_trade_id TEXT
        )
        """
    )
    # Backward-compatible DB migration for Railway volumes that already have the old schema.
    ensure_sqlite_column(cur, "trade_events", "tracking_key", "TEXT")
    ensure_sqlite_column(cur, "trade_events", "broker_trade_id", "TEXT")
    ensure_sqlite_column(cur, "trade_events", "broker_order_id", "TEXT")
    ensure_sqlite_column(cur, "trade_events", "client_trade_id", "TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_order_id ON trade_events(order_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_status ON trade_events(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_ts ON trade_events(ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_tracking_key ON trade_events(tracking_key)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            signal_id TEXT,
            pair TEXT,
            instrument TEXT,
            model_type TEXT,
            hint_side TEXT,
            decision TEXT,
            approval_probability REAL,
            margin REAL,
            would_order INTEGER,
            order_allowed INTEGER,
            reason TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_pair_ts ON prediction_events(pair, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_signal_id ON prediction_events(signal_id)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_signal_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            signal_id TEXT,
            pair TEXT,
            instrument TEXT,
            role TEXT,
            model_name TEXT,
            hint_side TEXT,
            approval_probability REAL,
            margin REAL,
            model_would_order INTEGER,
            actual_order_sent INTEGER,
            decision_source TEXT,
            conf_gate REAL,
            margin_gate REAL,
            reason TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_signal_pair_ts ON model_signal_events(pair, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_signal_pair_model ON model_signal_events(pair, model_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_signal_signal_id ON model_signal_events(signal_id)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_pattern_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            signal_id TEXT UNIQUE,
            pair TEXT,
            instrument TEXT,
            hint_side TEXT,
            model_type TEXT,
            model_feature_source_requested TEXT,
            model_feature_source_used TEXT,
            model_probability REAL,
            conf_gate REAL,
            decision TEXT,
            decision_source TEXT,
            would_order INTEGER,
            order_allowed INTEGER,
            reason TEXT,
            ai_enabled INTEGER,
            ai_verdict TEXT,
            ai_risk_score REAL,
            ai_reason TEXT,
            candle_pattern TEXT,
            candle_bias TEXT,
            pattern_confidence REAL,
            body_pips REAL,
            range_pips REAL,
            upper_wick_pips REAL,
            lower_wick_pips REAL,
            body_range_ratio REAL,
            wick_body_ratio REAL,
            trend_bias_last5 TEXT,
            m15_hint_aligned INTEGER,
            h1_hint_aligned INTEGER,
            h4_hint_aligned INTEGER,
            htf_conflict INTEGER,
            spread_pips REAL,
            atr_pips REAL,
            entry_price REAL,
            sl_price REAL,
            tp_price REAL,
            units_signed INTEGER,
            order_id TEXT,
            tracking_key TEXT,
            broker_trade_id TEXT,
            client_trade_id TEXT,
            trade_status TEXT,
            trade_outcome TEXT,
            pnl REAL,
            pnl_pips REAL,
            close_reason TEXT,
            opened_at TEXT,
            closed_at TEXT,
            raw_context_json TEXT
        )
        """
    )
    for col, col_type in [
        ("model_feature_source_requested", "TEXT"),
        ("model_feature_source_used", "TEXT"),
        ("ai_enabled", "INTEGER"),
        ("ai_verdict", "TEXT"),
        ("ai_risk_score", "REAL"),
        ("ai_reason", "TEXT"),
        ("candle_pattern", "TEXT"),
        ("candle_bias", "TEXT"),
        ("pattern_confidence", "REAL"),
        ("body_pips", "REAL"),
        ("range_pips", "REAL"),
        ("upper_wick_pips", "REAL"),
        ("lower_wick_pips", "REAL"),
        ("body_range_ratio", "REAL"),
        ("wick_body_ratio", "REAL"),
        ("trend_bias_last5", "TEXT"),
        ("m15_hint_aligned", "INTEGER"),
        ("h1_hint_aligned", "INTEGER"),
        ("h4_hint_aligned", "INTEGER"),
        ("htf_conflict", "INTEGER"),
        ("spread_pips", "REAL"),
        ("atr_pips", "REAL"),
        ("entry_price", "REAL"),
        ("sl_price", "REAL"),
        ("tp_price", "REAL"),
        ("units_signed", "INTEGER"),
        ("order_id", "TEXT"),
        ("tracking_key", "TEXT"),
        ("broker_trade_id", "TEXT"),
        ("client_trade_id", "TEXT"),
        ("trade_status", "TEXT"),
        ("trade_outcome", "TEXT"),
        ("pnl", "REAL"),
        ("pnl_pips", "REAL"),
        ("close_reason", "TEXT"),
        ("opened_at", "TEXT"),
        ("closed_at", "TEXT"),
        ("raw_context_json", "TEXT"),
    ]:
        ensure_sqlite_column(cur, "signal_pattern_history", col, col_type)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pattern_history_pair_ts ON signal_pattern_history(pair, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pattern_history_signal_id ON signal_pattern_history(signal_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pattern_history_pattern ON signal_pattern_history(pair, hint_side, candle_pattern, candle_bias)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pattern_history_tracking ON signal_pattern_history(tracking_key, order_id, broker_trade_id, client_trade_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pattern_history_outcome ON signal_pattern_history(trade_status, trade_outcome)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_active_overrides (
            pair TEXT PRIMARY KEY,
            active_model TEXT,
            previous_model TEXT,
            reason TEXT,
            updated_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_management_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            tracking_key TEXT,
            instrument TEXT,
            side TEXT,
            action TEXT,
            trade_specifier TEXT,
            entry_price REAL,
            live_bid REAL,
            live_ask REAL,
            favorable_pips REAL,
            initial_risk_pips REAL,
            current_r REAL,
            previous_sl_price REAL,
            requested_sl_price REAL,
            updated_sl_price REAL,
            success INTEGER,
            reason TEXT,
            broker_response TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_mgmt_ts ON trade_management_events(ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_mgmt_tracking ON trade_management_events(tracking_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_mgmt_instrument ON trade_management_events(instrument)")

    conn.commit()
    conn.close()


def write_csv_row(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_audit_row(out: Dict[str, Any]) -> None:
    write_csv_row(AUDIT_CSV, out)


def insert_trade_event_db(row: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trade_events
        (ts, instrument, side, units_signed, entry_price, sl_price, tp_price,
         status, pnl, order_id, reason, pair_score, tracking_key,
         broker_trade_id, broker_order_id, client_trade_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            row.get("tracking_key"),
            row.get("broker_trade_id"),
            row.get("broker_order_id"),
            row.get("client_trade_id"),
        ),
    )
    conn.commit()
    conn.close()


def write_trade_row(row: Dict[str, Any]) -> None:
    write_csv_row(TRADES_CSV, row)
    insert_trade_event_db(row)


def insert_trade_management_event_db(row: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trade_management_events
        (ts, tracking_key, instrument, side, action, trade_specifier,
         entry_price, live_bid, live_ask, favorable_pips, initial_risk_pips,
         current_r, previous_sl_price, requested_sl_price, updated_sl_price,
         success, reason, broker_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("ts"),
            row.get("tracking_key"),
            row.get("instrument"),
            row.get("side"),
            row.get("action"),
            row.get("trade_specifier"),
            row.get("entry_price"),
            row.get("live_bid"),
            row.get("live_ask"),
            row.get("favorable_pips"),
            row.get("initial_risk_pips"),
            row.get("current_r"),
            row.get("previous_sl_price"),
            row.get("requested_sl_price"),
            row.get("updated_sl_price"),
            int(bool(row.get("success"))),
            row.get("reason"),
            json.dumps(row.get("broker_response"), default=str) if row.get("broker_response") is not None else None,
        ),
    )
    conn.commit()
    conn.close()


def write_trade_management_event(row: Dict[str, Any]) -> None:
    write_csv_row(TRADE_MANAGEMENT_CSV, row)
    insert_trade_management_event_db(row)

def insert_prediction_event(row: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO prediction_events
        (ts, signal_id, pair, instrument, model_type, hint_side, decision,
         approval_probability, margin, would_order, order_allowed, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("ts"),
            row.get("signal_id"),
            row.get("pair"),
            row.get("instrument"),
            row.get("model_type"),
            row.get("hint_side"),
            row.get("decision"),
            row.get("approval_probability"),
            row.get("margin"),
            int(bool(row.get("would_order"))),
            int(bool(row.get("order_allowed"))),
            row.get("reason"),
        ),
    )
    conn.commit()
    conn.close()


def insert_model_signal_event(row: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_signal_events
        (ts, signal_id, pair, instrument, role, model_name, hint_side,
         approval_probability, margin, model_would_order, actual_order_sent,
         decision_source, conf_gate, margin_gate, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("ts"),
            row.get("signal_id"),
            row.get("pair"),
            row.get("instrument"),
            row.get("role"),
            row.get("model_name"),
            row.get("hint_side"),
            row.get("approval_probability"),
            row.get("margin"),
            int(bool(row.get("model_would_order"))),
            int(bool(row.get("actual_order_sent"))),
            row.get("decision_source"),
            row.get("conf_gate"),
            row.get("margin_gate"),
            row.get("reason"),
        ),
    )
    conn.commit()
    conn.close()




def _market_summary_for_tf(market_context: Dict[str, Any], tf: str) -> Dict[str, Any]:
    return ((market_context or {}).get("summaries") or {}).get(tf, {}) or {}


def _latest_candle_pattern_from_market(market_context: Dict[str, Any]) -> Dict[str, Any]:
    m15 = _market_summary_for_tf(market_context, "M15")
    pattern = m15.get("candle_pattern") or {}
    return pattern if isinstance(pattern, dict) else {}


def _trade_outcome_from_pnl(pnl: Any) -> str:
    value = safe_float(pnl, 0.0)
    if value > 0:
        return "win"
    if value < 0:
        return "loss"
    return "breakeven"


def insert_or_update_pattern_history_from_prediction(out: Dict[str, Any], feature_row: Dict[str, Any], market_context: Dict[str, Any], ai_review: Dict[str, Any]) -> None:
    if not PATTERN_HISTORY_ENABLED:
        return
    pattern = _latest_candle_pattern_from_market(market_context)
    m15 = _market_summary_for_tf(market_context, "M15")
    h1 = _market_summary_for_tf(market_context, "H1")
    h4 = _market_summary_for_tf(market_context, "H4")
    raw_context = None
    if PATTERN_HISTORY_INCLUDE_AI_CONTEXT:
        try:
            raw_context = json.dumps({
                "ai_review": ai_review,
                "external_market_context": market_context,
                "model_feature_source": {
                    "requested": feature_row.get("_model_feature_source_requested"),
                    "used": feature_row.get("_model_feature_source_used"),
                    "reason": feature_row.get("_model_feature_source_reason"),
                },
            }, default=str)[:20000]
        except Exception:
            raw_context = None
    row = {
        "ts": out.get("ts") or utc_ts(),
        "signal_id": out.get("signal_id"),
        "pair": out.get("pair"),
        "instrument": out.get("instrument"),
        "hint_side": out.get("hint_side"),
        "model_type": out.get("model_type"),
        "model_feature_source_requested": feature_row.get("_model_feature_source_requested") or out.get("model_feature_source_requested"),
        "model_feature_source_used": feature_row.get("_model_feature_source_used") or out.get("model_feature_source_used"),
        "model_probability": out.get("approval_probability"),
        "conf_gate": out.get("conf_gate"),
        "decision": out.get("decision"),
        "decision_source": out.get("decision_source"),
        "would_order": int(bool(out.get("would_order"))),
        "order_allowed": int(bool(out.get("order_allowed"))),
        "reason": out.get("why"),
        "ai_enabled": int(bool(out.get("ai_review_enabled"))),
        "ai_verdict": (ai_review or {}).get("ai_verdict"),
        "ai_risk_score": (ai_review or {}).get("risk_score"),
        "ai_reason": (ai_review or {}).get("reason"),
        "candle_pattern": pattern.get("pattern") or m15.get("last_candle_pattern"),
        "candle_bias": pattern.get("candle_bias") or m15.get("last_candle_bias"),
        "pattern_confidence": pattern.get("pattern_confidence"),
        "body_pips": pattern.get("body_pips") or out.get("body_pips"),
        "range_pips": pattern.get("range_pips") or out.get("range_pips"),
        "upper_wick_pips": pattern.get("upper_wick_pips"),
        "lower_wick_pips": pattern.get("lower_wick_pips"),
        "body_range_ratio": pattern.get("body_range_ratio") or out.get("body_range_ratio"),
        "wick_body_ratio": pattern.get("wick_body_ratio") or out.get("wick_body_ratio"),
        "trend_bias_last5": m15.get("trend_bias_last5") or pattern.get("trend_bias_last5"),
        "m15_hint_aligned": int(bool(m15.get("hint_side_aligned"))) if m15 else None,
        "h1_hint_aligned": int(bool(h1.get("hint_side_aligned"))) if h1 else None,
        "h4_hint_aligned": int(bool(h4.get("hint_side_aligned"))) if h4 else None,
        "htf_conflict": int(bool((market_context or {}).get("higher_timeframe_conflict"))),
        "spread_pips": out.get("spread_pips"),
        "atr_pips": out.get("atr_pips") or safe_float(feature_row.get("atr_pips"), 0.0),
        "entry_price": out.get("live_reprice_reference_price") or out.get("entry_reversal_live_mid") or out.get("mid_c"),
        "sl_price": out.get("sl_price"),
        "tp_price": out.get("tp_price"),
        "units_signed": out.get("units_signed"),
        "order_id": ((out.get("order_result") or {}).get("order_id") if isinstance(out.get("order_result"), dict) else None),
        "tracking_key": out.get("pending_trade_id"),
        "broker_trade_id": None,
        "client_trade_id": None,
        "trade_status": "PENDING" if out.get("would_order") else "NO_TRADE",
        "raw_context_json": raw_context,
    }
    columns = list(row.keys())
    placeholders = ",".join(["?"] * len(columns))
    update_clause = ",".join([f"{c}=excluded.{c}" for c in columns if c != "signal_id"])
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT INTO signal_pattern_history ({','.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(signal_id) DO UPDATE SET {update_clause}
        """,
        [row.get(c) for c in columns],
    )
    conn.commit()
    conn.close()


def update_pattern_history_from_trade_event(row: Dict[str, Any], tracking_key: str) -> None:
    if not PATTERN_HISTORY_ENABLED:
        return
    instrument = str(row.get("instrument") or "").upper()
    side = normalize_side(row.get("side") or "")
    status = str(row.get("status") or "").upper()
    ts = row.get("ts") or utc_ts()
    order_id = row.get("order_id")
    broker_trade_id = row.get("broker_trade_id")
    client_trade_id = row.get("client_trade_id")
    pnl = row.get("pnl")
    outcome = _trade_outcome_from_pnl(pnl) if status in {"CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"} else None
    conn = db_conn()
    cur = conn.cursor()
    # Try direct linking first.
    identifiers = [str(x) for x in [tracking_key, order_id, broker_trade_id, client_trade_id] if x]
    matched = 0
    if identifiers:
        where = " OR ".join(["tracking_key=?", "order_id=?", "broker_trade_id=?", "client_trade_id=?"])
        params = []
        for ident in identifiers:
            params.extend([ident, ident, ident, ident])
        cur.execute(
            f"""
            UPDATE signal_pattern_history
            SET trade_status=?, trade_outcome=COALESCE(?, trade_outcome), pnl=COALESCE(?, pnl),
                order_id=COALESCE(?, order_id), tracking_key=COALESCE(?, tracking_key),
                broker_trade_id=COALESCE(?, broker_trade_id), client_trade_id=COALESCE(?, client_trade_id),
                close_reason=COALESCE(?, close_reason),
                opened_at=CASE WHEN ?='OPEN' THEN COALESCE(opened_at, ?) ELSE opened_at END,
                closed_at=CASE WHEN ? IN ('CLOSED','STOPPED','TAKE_PROFIT','MANUAL') THEN ? ELSE closed_at END
            WHERE {where}
            """,
            [status, outcome, pnl, order_id, str(tracking_key), broker_trade_id, client_trade_id, row.get("reason"), status, ts, status, ts] + params,
        )
        matched = cur.rowcount
    # If Make did not pass IDs, attach to latest pending/approved same instrument+side.
    if matched == 0 and instrument and side:
        cur.execute(
            """
            UPDATE signal_pattern_history
            SET trade_status=?, trade_outcome=COALESCE(?, trade_outcome), pnl=COALESCE(?, pnl),
                order_id=COALESCE(?, order_id), tracking_key=COALESCE(?, tracking_key),
                broker_trade_id=COALESCE(?, broker_trade_id), client_trade_id=COALESCE(?, client_trade_id),
                close_reason=COALESCE(?, close_reason),
                opened_at=CASE WHEN ?='OPEN' THEN COALESCE(opened_at, ?) ELSE opened_at END,
                closed_at=CASE WHEN ? IN ('CLOSED','STOPPED','TAKE_PROFIT','MANUAL') THEN ? ELSE closed_at END
            WHERE id = (
                SELECT id FROM signal_pattern_history
                WHERE instrument=? AND hint_side=? AND trade_status IN ('PENDING','OPEN')
                ORDER BY ts DESC LIMIT 1
            )
            """,
            [status, outcome, pnl, order_id, str(tracking_key), broker_trade_id, client_trade_id, row.get("reason"), status, ts, status, ts, instrument, side],
        )
    conn.commit()
    conn.close()


def get_pattern_performance_summary(pair6: str, hint_side: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
    if not (PATTERN_HISTORY_ENABLED and PATTERN_STATS_FOR_AI_ENABLED):
        return {"enabled": False, "reason": "pattern_stats_disabled"}
    pattern = _latest_candle_pattern_from_market(market_context)
    candle_pattern = pattern.get("pattern") or _market_summary_for_tf(market_context, "M15").get("last_candle_pattern") or "unknown"
    candle_bias = pattern.get("candle_bias") or _market_summary_for_tf(market_context, "M15").get("last_candle_bias") or "NEUTRAL"
    conn = db_conn()
    try:
        rows = conn.execute(
            """
            SELECT * FROM signal_pattern_history
            WHERE pair=? AND hint_side=? AND candle_pattern=?
              AND trade_outcome IN ('win','loss','breakeven')
            ORDER BY ts DESC LIMIT ?
            """,
            (pair6, hint_side, candle_pattern, PATTERN_STATS_LOOKBACK),
        ).fetchall()
        broader = conn.execute(
            """
            SELECT * FROM signal_pattern_history
            WHERE hint_side=? AND candle_pattern=?
              AND trade_outcome IN ('win','loss','breakeven')
            ORDER BY ts DESC LIMIT ?
            """,
            (hint_side, candle_pattern, PATTERN_STATS_LOOKBACK),
        ).fetchall()
    finally:
        conn.close()

    def summarize(rows) -> Dict[str, Any]:
        total = len(rows)
        wins = sum(1 for r in rows if r["trade_outcome"] == "win")
        losses = sum(1 for r in rows if r["trade_outcome"] == "loss")
        breakeven = sum(1 for r in rows if r["trade_outcome"] == "breakeven")
        pnls = [safe_float(r["pnl"], 0.0) for r in rows if r["pnl"] is not None]
        approved = sum(1 for r in rows if r["ai_verdict"] == "APPROVE")
        rejected = sum(1 for r in rows if r["ai_verdict"] == "REJECT")
        return {
            "closed_samples": total,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": round(wins / total, 4) if total else None,
            "loss_rate": round(losses / total, 4) if total else None,
            "avg_pnl": round(sum(pnls) / len(pnls), 6) if pnls else None,
            "ai_approved_samples": approved,
            "ai_rejected_samples": rejected,
        }

    pair_stats = summarize(rows)
    global_stats = summarize(broader)
    return {
        "enabled": True,
        "pair": pair6,
        "hint_side": hint_side,
        "candle_pattern": candle_pattern,
        "candle_bias": candle_bias,
        "min_closed_for_confidence": PATTERN_STATS_MIN_CLOSED,
        "pair_pattern_stats": pair_stats,
        "global_pattern_stats": global_stats,
        "confidence": "usable" if pair_stats["closed_samples"] >= PATTERN_STATS_MIN_CLOSED else "low_sample",
    }

def read_trade_events_db() -> pd.DataFrame:
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM trade_events ORDER BY ts DESC", conn)
        if not df.empty and "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        return df
    finally:
        conn.close()


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


def restore_open_trades_from_db() -> None:
    """Best-effort restore of tracked open trades after Railway restart.

    This only restores trades that were sent to /trade_event with status=OPEN
    and have not later been followed by a CLOSED/STOPPED/TAKE_PROFIT/MANUAL event
    with the same tracking key. Stop modification still requires broker_trade_id
    or client_trade_id to have been included by Make.
    """
    try:
        conn = db_conn()
        rows = conn.execute(
            """
            SELECT * FROM trade_events
            WHERE tracking_key IS NOT NULL AND tracking_key != ''
            ORDER BY id ASC
            """
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"WARNING: could not restore open trades from DB: {exc}")
        return

    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        record = dict(row)
        tracking_key = str(record.get("tracking_key") or "").strip()
        if tracking_key:
            latest[tracking_key] = record

    restored = 0
    for tracking_key, record in latest.items():
        status = str(record.get("status") or "").upper()
        if status != "OPEN":
            continue
        instrument = str(record.get("instrument") or "").upper()
        side = normalize_side(record.get("side"))
        entry_price = safe_float(record.get("entry_price"), 0.0)
        sl_price = safe_float(record.get("sl_price"), 0.0)
        tp_price = safe_float(record.get("tp_price"), 0.0)
        if not instrument or side not in {"BUY", "SELL"} or entry_price <= 0:
            continue
        opened_at_dt = dt.datetime.now(dt.timezone.utc)
        try:
            opened_at_dt = pd.to_datetime(record.get("ts"), utc=True).to_pydatetime()
        except Exception:
            pass
        pip = instrument_pip_size(instrument)
        initial_risk_pips = abs(entry_price - sl_price) / pip if pip > 0 and sl_price > 0 else 0.0
        _open_trade_ids.add(tracking_key)
        _open_trade_meta[tracking_key] = {
            "tracking_key": tracking_key,
            "instrument": instrument,
            "symbol": instrument_to_symbol(instrument),
            "side": side,
            "units_signed": safe_int(record.get("units_signed"), 0),
            "entry_price": entry_price,
            "sl_price": sl_price,
            "original_sl_price": sl_price,
            "tp_price": tp_price,
            "pair_score": safe_float(record.get("pair_score"), 0.0),
            "opened_at_dt": opened_at_dt,
            "order_id": record.get("order_id"),
            "broker_trade_id": record.get("broker_trade_id"),
            "broker_order_id": record.get("broker_order_id"),
            "client_trade_id": record.get("client_trade_id"),
            "initial_risk_pips": initial_risk_pips,
            "breakeven_done": False,
            "trailing_active": False,
            "last_stop_update_ts": None,
            "last_favorable_pips": None,
            "last_current_r": None,
            "ts": record.get("ts"),
            "restored_from_db": True,
        }
        restored += 1
    if restored:
        print(f"Restored {restored} open trade(s) from SQLite trade_events.")



def safe_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=bool)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def read_active_override_from_db(pair6: str) -> Optional[Dict[str, Any]]:
    conn = db_conn()
    try:
        row = conn.execute(
            "SELECT * FROM model_active_overrides WHERE pair = ?",
            (pair6,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def write_active_model_override(
    pair6: str,
    new_model: str,
    previous_model: str,
    reason: str,
    bundle: Optional[Dict[str, Any]] = None,
) -> None:
    updated_at = utc_ts()
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_active_overrides
        (pair, active_model, previous_model, reason, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(pair) DO UPDATE SET
            active_model = excluded.active_model,
            previous_model = excluded.previous_model,
            reason = excluded.reason,
            updated_at = excluded.updated_at
        """,
        (pair6, new_model, previous_model, reason, updated_at),
    )
    conn.commit()
    conn.close()

    if bundle is not None:
        candidate_models = bundle.get("candidate_models") or {}
        candidate_metrics = bundle.get("candidate_metrics") or {}
        if new_model in candidate_models:
            bundle["model"] = candidate_models[new_model]
            bundle["model_type"] = new_model
            metric = candidate_metrics.get(new_model, {})
            bundle["avg_auc"] = safe_float(metric.get("auc"), bundle.get("avg_auc", 0.0))
            bundle["precision_at_gate"] = safe_float(metric.get("precision_at_gate"), bundle.get("precision_at_gate", 0.0))
            bundle["trades_at_gate"] = safe_int(metric.get("trades_at_gate"), bundle.get("trades_at_gate", 0))
            bundle["model_version"] = f"{pair6}:M15:signal_approval:{new_model}:override"
            bundle["active_model_override"] = new_model
            bundle["active_override_previous_model"] = previous_model
            bundle["active_override_reason"] = reason
            bundle["active_override_updated_at"] = updated_at

# ============================================================
# TRADE STATE / DUPLICATE CHECKS
# ============================================================
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


def cleanup_expired_pending_trades() -> None:
    if not _pending_trade_meta:
        return
    now_ts = now_unix()
    expired = []
    for pending_id, meta in list(_pending_trade_meta.items()):
        created = safe_int(meta.get("created_unix"), now_ts)
        if now_ts - created >= PENDING_TRADE_TIMEOUT_SECONDS:
            expired.append(pending_id)
    for pending_id in expired:
        _pending_trade_ids.discard(str(pending_id))
        _pending_trade_meta.pop(str(pending_id), None)


def current_open_trade_count() -> int:
    return len(_open_trade_ids)


def current_pending_trade_count() -> int:
    cleanup_expired_pending_trades()
    return len(_pending_trade_ids)


def current_active_trade_count() -> int:
    cleanup_expired_pending_trades()
    return current_open_trade_count() + current_pending_trade_count()


def can_open_trade() -> bool:
    return current_active_trade_count() < MAX_OPEN_TRADES


def reserve_pending_trade(
    signal_id: str,
    pair6: str,
    instrument: str,
    side: str,
    model_type: str,
    decision_source: str,
) -> Optional[str]:
    if not PENDING_TRADE_LOCK_ENABLED:
        return None
    cleanup_expired_pending_trades()
    pending_id = f"pending_{signal_id}"
    _pending_trade_ids.add(pending_id)
    _pending_trade_meta[pending_id] = {
        "pending_id": pending_id,
        "signal_id": signal_id,
        "pair": pair6,
        "instrument": instrument,
        "side": side,
        "model_type": model_type,
        "decision_source": decision_source,
        "created_ts": utc_ts(),
        "created_unix": now_unix(),
        "expires_after_seconds": PENDING_TRADE_TIMEOUT_SECONDS,
    }
    return pending_id


def clear_pending_trades(reason: str = "cleared") -> None:
    if _pending_trade_ids or _pending_trade_meta:
        print(f"Clearing pending trade locks: {reason}")
    _pending_trade_ids.clear()
    _pending_trade_meta.clear()


def note_trade_opened(tracking_key: Optional[str]) -> None:
    if tracking_key:
        _open_trade_ids.add(str(tracking_key))
    # Once Make registers the filled OANDA trade as OPEN, the pending reservation
    # has served its purpose. With MAX_OPEN_TRADES=1 we clear all pending locks.
    clear_pending_trades("open_trade_registered")


def note_trade_closed(tracking_key: Optional[str]) -> None:
    if tracking_key and str(tracking_key) in _open_trade_ids:
        _open_trade_ids.remove(str(tracking_key))
    # A closed trade frees the one-trade lock.
    cleanup_expired_pending_trades()


def make_signal_fingerprint(instrument: str, hint_side: str, bar_time: int, mid_c: float, tf: Optional[str]) -> str:
    raw = {
        "instrument": instrument,
        "hint_side": hint_side,
        "bar_time": int(bar_time),
        "mid_c": round(float(mid_c), instrument_precision(instrument)),
        "tf": tf or "M15",
    }
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()


def is_duplicate_signal(pair6: str, fingerprint: str) -> bool:
    now = now_unix()
    queue = _recent_signals.setdefault(pair6, deque())
    while queue and (now - queue[0][0] > DUP_WINDOW_SECONDS):
        queue.popleft()
    return any(stored == fingerprint for _, stored in queue)


def remember_signal(pair6: str, fingerprint: str) -> None:
    _recent_signals.setdefault(pair6, deque()).append((now_unix(), fingerprint))

# ============================================================
# RUNTIME FEATURES
# ============================================================
def rsi_runtime(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def atr_runtime(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["mid_h"]
    low = df["mid_l"]
    close = df["mid_c"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def update_bar_history(pair6: str, payload: Dict[str, Any]) -> pd.DataFrame:
    queue = _bar_history.setdefault(pair6, deque(maxlen=BAR_HISTORY_LEN))
    t_val = safe_int(payload.get("t") or payload.get("bar_time") or payload.get("ts"), now_unix())
    timestamp = pd.to_datetime(t_val, unit="s", utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.utcnow()
    row = {
        "t": t_val,
        "time": timestamp,
        "mid_o": safe_float(payload.get("mid_o"), safe_float(payload.get("mid_c"), 0.0)),
        "mid_h": safe_float(payload.get("mid_h"), safe_float(payload.get("mid_c"), 0.0)),
        "mid_l": safe_float(payload.get("mid_l"), safe_float(payload.get("mid_c"), 0.0)),
        "mid_c": safe_float(payload.get("mid_c"), 0.0),
        "volume": safe_float(payload.get("volume"), 0.0),
        "spread_c": safe_float(payload.get("spread_c"), 0.0),
    }
    if queue and queue[-1]["t"] == row["t"]:
        queue[-1] = row
    else:
        queue.append(row)
    return pd.DataFrame(list(queue))


def seed_history_from_csv(data_dir: str) -> None:
    root = Path(data_dir)
    if not root.exists():
        print(f"WARNING: DATA_DIR not found for history seed: {data_dir}")
        return
    for pair6, instrument in PAIR_MAP.items():
        candidates = [
            root / f"{instrument}_M15.csv",
            root / f"{pair6}_M15.csv",
            root / f"{instrument}.csv",
            root / f"{pair6}.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if not path:
            continue
        try:
            df = pd.read_csv(path).tail(BAR_HISTORY_LEN).copy()
            if "time" not in df.columns:
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"])
            has_mid = all(col in df.columns for col in ["mid_o", "mid_h", "mid_l", "mid_c"])
            has_bid_ask = all(
                col in df.columns
                for col in ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
            )
            if not has_mid and not has_bid_ask:
                continue
            if not has_mid:
                df["mid_o"] = (pd.to_numeric(df["bid_o"], errors="coerce") + pd.to_numeric(df["ask_o"], errors="coerce")) / 2.0
                df["mid_h"] = (pd.to_numeric(df["bid_h"], errors="coerce") + pd.to_numeric(df["ask_h"], errors="coerce")) / 2.0
                df["mid_l"] = (pd.to_numeric(df["bid_l"], errors="coerce") + pd.to_numeric(df["ask_l"], errors="coerce")) / 2.0
                df["mid_c"] = (pd.to_numeric(df["bid_c"], errors="coerce") + pd.to_numeric(df["ask_c"], errors="coerce")) / 2.0
            if "spread_c" not in df.columns and has_bid_ask:
                df["spread_c"] = pd.to_numeric(df["ask_c"], errors="coerce") - pd.to_numeric(df["bid_c"], errors="coerce")
            if "spread_c" not in df.columns:
                df["spread_c"] = 0.0
            if "volume" not in df.columns:
                df["volume"] = 0.0
            queue = deque(maxlen=BAR_HISTORY_LEN)
            for _, row in df.iterrows():
                queue.append(
                    {
                        "t": int(pd.Timestamp(row["time"]).timestamp()),
                        "time": row["time"],
                        "mid_o": safe_float(row.get("mid_o"), 0.0),
                        "mid_h": safe_float(row.get("mid_h"), 0.0),
                        "mid_l": safe_float(row.get("mid_l"), 0.0),
                        "mid_c": safe_float(row.get("mid_c"), 0.0),
                        "volume": safe_float(row.get("volume"), 0.0),
                        "spread_c": safe_float(row.get("spread_c"), 0.0),
                    }
                )
            _bar_history[pair6] = queue
            print(f"Seeded {pair6} M15 history with {len(queue)} bars from {path}")
        except Exception as exc:
            print(f"WARNING: failed to seed M15 history for {pair6}: {exc}")


def add_signal_approval_runtime_features(hist: pd.DataFrame, instrument: str, hint_side: str, payload: Dict[str, Any]) -> pd.DataFrame:
    df = hist.copy()
    for col in ["mid_o", "mid_h", "mid_l", "mid_c", "volume", "spread_c"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    close = df["mid_c"]
    pip = instrument_pip_size(instrument)
    df["ret1"] = close.pct_change(1)
    df["ret3"] = close.pct_change(3)
    df["ret5"] = close.pct_change(5)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["ema20_dist"] = (close - ema20) / close.replace(0, np.nan)
    df["ema50_dist"] = (close - ema50) / close.replace(0, np.nan)
    df["ema200_dist"] = (close - ema200) / close.replace(0, np.nan)
    df["rsi14"] = rsi_runtime(close, 14)
    df["atr14"] = atr_runtime(df, 14)
    df["atr14_pct"] = df["atr14"] / close.replace(0, np.nan) * 100.0
    df["atr_pips"] = df["atr14"] / pip
    basis = close.rolling(20).mean()
    dev = close.rolling(20).std()
    df["bb_width"] = ((basis + 2.0 * dev) - (basis - 2.0 * dev)) / close.replace(0, np.nan)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal
    volume_mean = df["volume"].rolling(50).mean()
    volume_std = df["volume"].rolling(50).std()
    df["vol_z"] = (df["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    df["spread_pips"] = df["spread_c"].fillna(0.0) / pip
    dt_series = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["hour_utc"] = dt_series.dt.hour.fillna(0)
    df["dayofweek"] = dt_series.dt.dayofweek.fillna(0)
    df["range_pips"] = (df["mid_h"] - df["mid_l"]).abs() / pip
    df["body_pips"] = (df["mid_c"] - df["mid_o"]).abs() / pip
    df["body_range_ratio"] = df["body_pips"] / df["range_pips"].replace(0, np.nan)
    side_sign = side_sign_from_hint(hint_side)
    df["side_sign"] = side_sign
    df["setup_pullback"] = safe_float(payload.get("setup_pullback"), 0.0)
    df["setup_ema_cross"] = safe_float(payload.get("setup_ema_cross"), 0.0)
    momentum = df["ret1"] + df["ret3"] + df["ret5"]
    df["signal_momentum_aligned"] = ((momentum * side_sign) >= 0).astype(float)
    return df


def build_oanda_model_feature_row(
    payload: Dict[str, Any],
    pair6: str,
    instrument: str,
    hint_side: str,
    feature_order: List[str],
) -> Dict[str, Any]:
    """Build the model feature row from latest completed OANDA candles.

    Hybrid mode uses TradingView/Make only as the signal trigger and hint_side.
    The ML model then scores features calculated from broker candles, which keeps
    model input aligned with the market that will actually receive the order.
    """
    meta = {
        "_model_feature_source_requested": MODEL_FEATURE_SOURCE,
        "_model_feature_source_used": "oanda",
        "_model_feature_source_ok": False,
        "_model_feature_source_reason": "not_started",
        "_model_feature_granularity": MODEL_FEATURE_OANDA_GRANULARITY,
    }
    if not broker_ready():
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
        if "volume" not in df.columns:
            df["volume"] = 0.0
        feat_df = add_signal_approval_runtime_features(df, instrument, hint_side, payload)
        last = feat_df.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict()
        row = {feature: safe_float(last.get(feature), safe_float(payload.get(feature), 0.0)) for feature in feature_order}
        row.update(meta)
        row["_model_feature_source_ok"] = True
        row["_model_feature_source_used"] = "oanda_latest_closed"
        row["_model_feature_source_reason"] = "oanda_latest_completed_candle_features"
        row["_model_feature_time"] = str(last.get("time", ""))
        row["_model_feature_last_close"] = safe_float(last.get("mid_c"), 0.0)
        row["_model_feature_candles"] = int(len(df))
        row["_model_feature_ret1"] = safe_float(row.get("ret1"), 0.0)
        row["_model_feature_ema20_dist"] = safe_float(row.get("ema20_dist"), 0.0)
        row["_model_feature_ema50_dist"] = safe_float(row.get("ema50_dist"), 0.0)
        return row
    except Exception as exc:
        meta["_model_feature_source_used"] = "alert"
        meta["_model_feature_source_reason"] = f"oanda_feature_error:{repr(exc)}"
        return meta


def build_runtime_feature_row(payload: Dict[str, Any], pair6: str, instrument: str, hint_side: str, feature_order: List[str]) -> Dict[str, Any]:
    if MODEL_FEATURE_SOURCE in {"oanda", "hybrid"}:
        oanda_row = build_oanda_model_feature_row(payload, pair6, instrument, hint_side, feature_order)
        if bool(oanda_row.get("_model_feature_source_ok", False)):
            return oanda_row
        if MODEL_FEATURE_SOURCE == "oanda" and not MODEL_FEATURE_FALLBACK_TO_ALERT:
            # Keep the model safe by returning zeros plus explicit metadata instead of
            # silently using stale alert features when strict OANDA mode is requested.
            strict_row = {feature: 0.0 for feature in feature_order}
            strict_row.update(oanda_row)
            strict_row["_model_feature_source_used"] = "oanda_failed_no_alert_fallback"
            return strict_row
        # Hybrid fallback: still allow alert features if OANDA candles are temporarily unavailable.
        oanda_meta = dict(oanda_row)
    else:
        oanda_meta = {
            "_model_feature_source_requested": MODEL_FEATURE_SOURCE,
            "_model_feature_source_used": "alert",
            "_model_feature_source_ok": True,
            "_model_feature_source_reason": "alert_feature_source_selected",
        }

    direct_payload = dict(payload)
    direct_payload["side_sign"] = side_sign_from_hint(hint_side)
    mandatory_direct = [
        "ret1", "ret3", "ret5", "ema20_dist", "ema50_dist", "ema200_dist",
        "rsi14", "atr14_pct", "bb_width", "macd_hist", "vol_z", "spread_pips",
        "hour_utc", "dayofweek",
    ]
    if all(direct_payload.get(feature) not in (None, "") for feature in mandatory_direct):
        mid_o = safe_float(direct_payload.get("mid_o"), 0.0)
        mid_h = safe_float(direct_payload.get("mid_h"), 0.0)
        mid_l = safe_float(direct_payload.get("mid_l"), 0.0)
        mid_c = safe_float(direct_payload.get("mid_c"), 0.0)
        pip = instrument_pip_size(instrument)
        range_pips = abs(mid_h - mid_l) / pip if pip > 0 else 0.0
        body_pips = abs(mid_c - mid_o) / pip if pip > 0 else 0.0
        atr14 = safe_float(direct_payload.get("atr14"), 0.0)
        atr_pips = atr14 / pip if atr14 > 0 and pip > 0 else (
            ((safe_float(direct_payload.get("atr14_pct"), 0.0) / 100.0) * mid_c) / pip if pip > 0 and mid_c > 0 else 0.0
        )
        momentum = safe_float(direct_payload.get("ret1"), 0.0) + safe_float(direct_payload.get("ret3"), 0.0) + safe_float(direct_payload.get("ret5"), 0.0)
        derived = {
            "side_sign": side_sign_from_hint(hint_side),
            "setup_pullback": safe_float(direct_payload.get("setup_pullback"), 0.0),
            "setup_ema_cross": safe_float(direct_payload.get("setup_ema_cross"), 0.0),
            "range_pips": range_pips,
            "body_pips": body_pips,
            "body_range_ratio": body_pips / range_pips if range_pips > 0 else 0.0,
            "atr_pips": atr_pips,
            "signal_momentum_aligned": 1.0 if momentum * side_sign_from_hint(hint_side) >= 0 else 0.0,
        }
        merged = {**direct_payload, **derived}
        row = {feature: safe_float(merged.get(feature), 0.0) for feature in feature_order}
        row.update(oanda_meta)
        if row.get("_model_feature_source_used") == "alert" and row.get("_model_feature_source_reason") != "alert_feature_source_selected":
            row["_model_feature_source_used"] = "alert_fallback"
        return row
    pip = instrument_pip_size(instrument)
    bid_c = safe_float(payload.get("bid_c"), np.nan)
    ask_c = safe_float(payload.get("ask_c"), np.nan)
    spread_c = safe_float(payload.get("spread_c"), np.nan)
    if (not np.isfinite(spread_c) or spread_c <= 0) and np.isfinite(bid_c) and np.isfinite(ask_c) and ask_c >= bid_c:
        spread_c = ask_c - bid_c
    if not np.isfinite(spread_c):
        spread_c = 0.0
    payload["spread_c"] = spread_c
    payload["spread_pips"] = spread_c / pip if pip > 0 else 0.0
    hist = update_bar_history(pair6, payload)
    feat_df = add_signal_approval_runtime_features(hist, instrument, hint_side, payload)
    last = feat_df.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict()
    row = {feature: safe_float(last.get(feature), safe_float(payload.get(feature), 0.0)) for feature in feature_order}
    row.update(oanda_meta)
    if row.get("_model_feature_source_used") == "alert" and row.get("_model_feature_source_reason") != "alert_feature_source_selected":
        row["_model_feature_source_used"] = "alert_history_fallback"
    return row

# ============================================================
# NEWS FILTER HELPERS
# ============================================================
def parse_utc_datetime(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
        text = str(value).strip()
        if text.isdigit():
            return dt.datetime.fromtimestamp(float(text), tz=dt.timezone.utc)
        text = text.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None


def pair_currencies(pair6: str) -> List[str]:
    normalized = str(pair6 or "").upper().replace("_", "")
    return [normalized[:3], normalized[3:]] if len(normalized) == 6 else []


def normalize_news_currency(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    raw = value if isinstance(value, list) else str(value).replace("/", ",").replace("|", ",").split(",")
    return [str(item).strip().upper() for item in raw if str(item).strip()]


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
    normalized = normalize_impact(impact)
    return normalized in NEWS_BLOCK_IMPACTS or (normalized == "HIGH" and "RED" in NEWS_BLOCK_IMPACTS)


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
        parts = [part.strip() for part in chunk.split("|")]
        if len(parts) < 3:
            continue
        raw = {
            "start": parts[0],
            "end": parts[1],
            "currency": parts[2],
            "impact": parts[3] if len(parts) >= 4 else "HIGH",
            "title": parts[4] if len(parts) >= 5 else NEWS_DEFAULT_TITLE,
            "source": "manual_env_blackout",
        }
        event = normalize_news_event(raw)
        if event:
            events.append(event)
    return events


def load_news_events() -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if NEWS_EVENTS_JSON:
        try:
            parsed = json.loads(NEWS_EVENTS_JSON)
            if isinstance(parsed, dict):
                parsed = parsed.get("events", [])
            for raw in parsed if isinstance(parsed, list) else []:
                event = normalize_news_event(raw)
                if event:
                    events.append(event)
        except Exception as exc:
            print(f"WARNING: failed to parse NEWS_EVENTS_JSON: {exc}")
    events.extend(parse_manual_blackout_events())
    file_path = Path(NEWS_EVENTS_FILE)
    if file_path.exists():
        try:
            parsed = json.loads(file_path.read_text())
            if isinstance(parsed, dict):
                parsed = parsed.get("events", [])
            for raw in parsed if isinstance(parsed, list) else []:
                event = normalize_news_event(raw)
                if event:
                    events.append(event)
        except Exception as exc:
            print(f"WARNING: failed to parse NEWS_EVENTS_FILE={NEWS_EVENTS_FILE}: {exc}")
    cutoff = now_utc() - dt.timedelta(hours=NEWS_KEEP_PAST_HOURS)
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for event in events:
        end = parse_utc_datetime(event.get("end_utc"))
        if end and end < cutoff:
            continue
        key = (event.get("title"), tuple(event.get("currencies") or []), event.get("start_utc"), event.get("end_utc"))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(event)
    return cleaned


def save_news_events_to_file(events: List[Dict[str, Any]]) -> None:
    try:
        file_path = Path(NEWS_EVENTS_FILE)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({"events": events}, indent=2))
    except Exception as exc:
        print(f"WARNING: failed to save news events to {NEWS_EVENTS_FILE}: {exc}")


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
    for event in NEWS_EVENTS:
        impact = normalize_impact(event.get("impact"))
        if not event_impact_is_blocked(impact):
            continue
        event_ccys = set(normalize_news_currency(event.get("currencies")))
        currency_matches = NEWS_BLOCK_ALL_CURRENCIES or "ALL" in event_ccys or bool(relevant.intersection(event_ccys))
        if not currency_matches:
            continue
        start = parse_utc_datetime(event.get("start_utc"))
        end = parse_utc_datetime(event.get("end_utc"))
        event_time = parse_utc_datetime(event.get("time_utc"))
        if not start or not end:
            continue
        if event_time:
            delta = abs((event_time - ts).total_seconds()) / 60.0
            if nearest_delta is None or delta < nearest_delta:
                nearest_delta = delta
                nearest = event
        if start <= ts <= end:
            metrics.update(
                {
                    "news_filter_passed": False,
                    "news_block_title": event.get("title"),
                    "news_block_impact": impact,
                    "news_block_currencies": ",".join(event.get("currencies") or []),
                    "news_blackout_start_utc": start.isoformat(),
                    "news_blackout_end_utc": end.isoformat(),
                    "news_event_time_utc": event_time.isoformat() if event_time else None,
                }
            )
            return False, f"News filter blocked: {impact} {','.join(event.get('currencies') or [])} {event.get('title')} blackout {start.isoformat()} to {end.isoformat()}", metrics
    if nearest:
        metrics.update(
            {
                "nearest_news_title": nearest.get("title"),
                "nearest_news_impact": normalize_impact(nearest.get("impact")),
                "nearest_news_currencies": ",".join(nearest.get("currencies") or []),
                "nearest_news_minutes": round(float(nearest_delta or 0.0), 2),
            }
        )
    return True, "news_filter_passed", metrics

# ============================================================
# SANITY / NOISE / RISK / PRICES
# ============================================================
def payload_sanity_checks(payload: Dict[str, Any], instrument: str) -> Optional[str]:
    pip = instrument_pip_size(instrument)
    spread_pips = safe_float(payload.get("spread_pips"), np.nan)
    if not np.isfinite(spread_pips):
        spread_c = safe_float(payload.get("spread_c"), 0.0)
        spread_pips = spread_c / pip if pip > 0 else 0.0
        payload["spread_pips"] = spread_pips
    if spread_pips > MAX_SPREAD_PIPS:
        return f"Spread too high: {spread_pips:.2f} pips > {MAX_SPREAD_PIPS:.2f}"
    atr14 = safe_float(payload.get("atr14"), 0.0)
    if atr14 > 0 and atr14 < min_atr_for_instrument(instrument):
        return f"ATR too small: {atr14}"
    mid_l = safe_float(payload.get("mid_l"), 0.0)
    mid_c = safe_float(payload.get("mid_c"), 0.0)
    mid_h = safe_float(payload.get("mid_h"), 0.0)
    mid_o = safe_float(payload.get("mid_o"), mid_c)
    if mid_c <= 0:
        return "Bad payload: mid_c missing or <= 0"
    if mid_h < mid_l:
        return "Bad payload: mid_h < mid_l"
    if mid_l > 0 and mid_h > 0 and not (mid_l <= mid_c <= mid_h):
        return "Bad payload: mid_c not between mid_l and mid_h"
    if mid_l > 0 and mid_h > 0 and not (mid_l <= mid_o <= mid_h):
        return "Bad payload: mid_o not between mid_l and mid_h"
    if safe_float(payload.get("spread_pips"), 0.0) < 0:
        return "Bad payload: negative spread_pips"
    return None



def signal_staleness_guard(payload: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not SIGNAL_STALENESS_GUARD_ENABLED:
        return True, "signal_staleness_guard_disabled", {"signal_staleness_guard_enabled": False}
    signal_time = parse_utc_datetime(payload.get("t") or payload.get("bar_time") or payload.get("ts"))
    if signal_time is None:
        return False, "Signal staleness guard blocked: missing_or_invalid_signal_time", {
            "signal_staleness_guard_enabled": True,
            "signal_staleness_guard_passed": False,
            "signal_age_seconds": None,
        }
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




def _sign_for_direction(value: float, eps: float = 0.0) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def direction_consensus_guard(
    payload: Dict[str, Any],
    feature_row: Dict[str, Any],
    instrument: str,
    hint_side: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Confirm that the hinted side agrees with current M15 direction before ML approval can order.

    This is a hard pre-order guard. The ML model can approve only the TradingView hint; it cannot flip it.
    This guard rejects the hint when the candle, short momentum, and EMA context disagree.
    """
    if not DIRECTION_CONFIRMATION_ENABLED:
        return True, "direction_confirmation_disabled", {"direction_confirmation_enabled": False}

    side = 1 if hint_side == "BUY" else -1 if hint_side == "SELL" else 0
    if side == 0:
        return False, "Direction confirmation blocked: invalid hint_side", {
            "direction_confirmation_enabled": True,
            "direction_confirmation_passed": False,
            "direction_confirmation_score": 0,
        }

    pip = instrument_pip_size(instrument)
    mid_o = safe_float(payload.get("mid_o"), 0.0)
    mid_c = safe_float(payload.get("mid_c"), 0.0)
    body_pips_signed = ((mid_c - mid_o) / pip) if pip > 0 and mid_o > 0 and mid_c > 0 else 0.0
    body_pips_abs = abs(body_pips_signed)
    body_range_ratio = safe_float(feature_row.get("body_range_ratio"), 0.0)

    ret1 = safe_float(feature_row.get("ret1"), safe_float(payload.get("ret1"), 0.0))
    ret3 = safe_float(feature_row.get("ret3"), safe_float(payload.get("ret3"), 0.0))
    ret5 = safe_float(feature_row.get("ret5"), safe_float(payload.get("ret5"), 0.0))
    ema20_dist = safe_float(feature_row.get("ema20_dist"), safe_float(payload.get("ema20_dist"), 0.0))
    ema50_dist = safe_float(feature_row.get("ema50_dist"), safe_float(payload.get("ema50_dist"), 0.0))
    ema200_dist = safe_float(feature_row.get("ema200_dist"), safe_float(payload.get("ema200_dist"), 0.0))
    macd_hist = safe_float(feature_row.get("macd_hist"), safe_float(payload.get("macd_hist"), 0.0))
    rsi14 = safe_float(feature_row.get("rsi14"), safe_float(payload.get("rsi14"), 50.0))
    momentum_aligned = bool(safe_float(feature_row.get("signal_momentum_aligned"), 0.0) >= 0.5)

    ema_eps = DIRECTION_CONFIRM_EMA_BUFFER_PIPS * pip
    score = 0
    aligned = []
    conflicts = []
    required_failures = []

    candle_sign = _sign_for_direction(body_pips_signed, DIRECTION_CONFIRM_MIN_BODY_PIPS)
    if candle_sign == side:
        score += 1
        aligned.append("candle_body")
    elif candle_sign == -side:
        conflicts.append("candle_body")
        if DIRECTION_CONFIRM_REQUIRE_CANDLE_ALIGN:
            required_failures.append("candle_body_not_aligned")
        if DIRECTION_CONFIRM_BLOCK_STRONG_OPPOSITE_CANDLE and body_range_ratio >= DIRECTION_CONFIRM_STRONG_BODY_RATIO:
            required_failures.append("strong_opposite_candle")

    ret1_sign = _sign_for_direction(ret1, 0.0)
    ret3_sign = _sign_for_direction(ret3, DIRECTION_CONFIRM_MIN_RET3_ABS)
    ret5_sign = _sign_for_direction(ret5, DIRECTION_CONFIRM_MIN_RET5_ABS)
    for label, sig in (("ret1", ret1_sign), ("ret3", ret3_sign), ("ret5", ret5_sign)):
        if sig == side:
            score += 1
            aligned.append(label)
        elif sig == -side:
            conflicts.append(label)
    if DIRECTION_CONFIRM_REQUIRE_RET3_ALIGN and ret3_sign == -side:
        required_failures.append("ret3_against_hint")

    ema20_sign = _sign_for_direction(ema20_dist, ema_eps)
    ema50_sign = _sign_for_direction(ema50_dist, ema_eps)
    ema200_sign = _sign_for_direction(ema200_dist, ema_eps)
    if ema20_sign == side:
        score += 1
        aligned.append("ema20_side")
    elif ema20_sign == -side:
        conflicts.append("ema20_side")
        if DIRECTION_CONFIRM_REQUIRE_EMA20_SIDE:
            required_failures.append("ema20_against_hint")
    elif DIRECTION_CONFIRM_REQUIRE_EMA20_SIDE:
        required_failures.append("ema20_not_confirmed")

    if ema50_sign == side:
        score += 1
        aligned.append("ema50_side")
    elif ema50_sign == -side:
        conflicts.append("ema50_side")
        if DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA50:
            required_failures.append("ema50_against_hint")

    if ema200_sign == side:
        score += 1
        aligned.append("ema200_side")
    elif ema200_sign == -side:
        conflicts.append("ema200_side")
        if DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA200:
            required_failures.append("ema200_against_hint")

    macd_sign = _sign_for_direction(macd_hist, 0.0)
    if macd_sign == side:
        score += 1
        aligned.append("macd_hist")
    elif macd_sign == -side:
        conflicts.append("macd_hist")
        if DIRECTION_CONFIRM_REQUIRE_MACD_ALIGN:
            required_failures.append("macd_against_hint")

    if hint_side == "BUY" and rsi14 >= 50.0:
        score += 1
        aligned.append("rsi_ge_50")
    elif hint_side == "SELL" and rsi14 <= 50.0:
        score += 1
        aligned.append("rsi_le_50")
    else:
        conflicts.append("rsi_side")

    if momentum_aligned:
        score += 1
        aligned.append("signal_momentum_aligned")
    else:
        conflicts.append("signal_momentum_not_aligned")

    passed = (score >= DIRECTION_CONFIRM_MIN_SCORE) and not required_failures
    metrics = {
        "direction_confirmation_enabled": True,
        "direction_confirmation_passed": passed,
        "direction_confirmation_required": DIRECTION_CONFIRMATION_REQUIRED,
        "direction_confirmation_score": int(score),
        "direction_confirmation_min_score": DIRECTION_CONFIRM_MIN_SCORE,
        "direction_confirmation_aligned": aligned,
        "direction_confirmation_conflicts": conflicts,
        "direction_confirmation_required_failures": required_failures,
        "direction_confirmation_body_pips_signed": round(float(body_pips_signed), 3),
        "direction_confirmation_body_range_ratio": round(float(body_range_ratio), 4),
        "direction_confirmation_ret1": float(ret1),
        "direction_confirmation_ret3": float(ret3),
        "direction_confirmation_ret5": float(ret5),
        "direction_confirmation_ema20_dist": float(ema20_dist),
        "direction_confirmation_ema50_dist": float(ema50_dist),
        "direction_confirmation_ema200_dist": float(ema200_dist),
        "direction_confirmation_macd_hist": float(macd_hist),
        "direction_confirmation_rsi14": float(rsi14),
    }
    if not passed:
        reason = (
            f"Direction confirmation blocked: hint={hint_side}, score={score}/{DIRECTION_CONFIRM_MIN_SCORE}, "
            f"aligned={','.join(aligned) or 'none'}, conflicts={','.join(conflicts) or 'none'}, "
            f"required_failures={','.join(required_failures) or 'none'}"
        )
        if DIRECTION_CONFIRMATION_REQUIRED:
            return False, reason, metrics
        return True, "Direction confirmation optional: " + reason, metrics
    return True, f"direction_confirmation_passed:score={score}/{DIRECTION_CONFIRM_MIN_SCORE}", metrics

def entry_reversal_guard(
    payload: Dict[str, Any],
    feature_row: Dict[str, Any],
    instrument: str,
    hint_side: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not ENTRY_REVERSAL_GUARD_ENABLED:
        return True, "entry_reversal_guard_disabled", {"entry_reversal_guard_enabled": False}

    quote = fetch_live_oanda_quote(instrument)
    if not quote.get("ok"):
        metrics = {
            "entry_reversal_guard_enabled": True,
            "entry_reversal_guard_passed": not ENTRY_REVERSAL_GUARD_REQUIRED,
            "entry_reversal_quote_error": quote.get("error"),
        }
        if ENTRY_REVERSAL_GUARD_REQUIRED:
            return False, f"Entry reversal guard blocked: live quote unavailable: {quote.get('error')}", metrics
        return True, f"Entry reversal guard optional: live quote unavailable: {quote.get('error')}", metrics

    pip = instrument_pip_size(instrument)
    bid = safe_float(quote.get("bid"), 0.0)
    ask = safe_float(quote.get("ask"), 0.0)
    live_mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    alert_mid = safe_float(payload.get("mid_c"), 0.0)
    spread_pips = (ask - bid) / pip if pip > 0 and ask >= bid else 999.0

    if hint_side == "BUY":
        adverse_pips = (alert_mid - live_mid) / pip if alert_mid > 0 and live_mid > 0 else 0.0
    elif hint_side == "SELL":
        adverse_pips = (live_mid - alert_mid) / pip if alert_mid > 0 and live_mid > 0 else 0.0
    else:
        adverse_pips = 0.0

    momentum_aligned = bool(safe_float(feature_row.get("signal_momentum_aligned"), 0.0) >= 0.5)
    ema20_dist = safe_float(feature_row.get("ema20_dist"), 0.0)
    ema20_side_bad = False
    if ENTRY_REVERSAL_EMA20_SIDE_ENABLED:
        if hint_side == "BUY" and ema20_dist < -ENTRY_REVERSAL_EMA20_BUFFER:
            ema20_side_bad = True
        if hint_side == "SELL" and ema20_dist > ENTRY_REVERSAL_EMA20_BUFFER:
            ema20_side_bad = True

    reasons = []
    if spread_pips > ENTRY_REVERSAL_MAX_SPREAD_PIPS:
        reasons.append(f"spread_too_high_for_entry:{spread_pips:.2f}>{ENTRY_REVERSAL_MAX_SPREAD_PIPS:.2f}")
    if adverse_pips >= ENTRY_REVERSAL_MAX_ADVERSE_PIPS:
        reasons.append(f"live_price_reversed_against_{hint_side.lower()}:{adverse_pips:.2f}>={ENTRY_REVERSAL_MAX_ADVERSE_PIPS:.2f}pips")
    if ENTRY_REVERSAL_REQUIRE_MOMENTUM_ALIGN and not momentum_aligned:
        reasons.append("signal_momentum_not_aligned")
    if ema20_side_bad:
        reasons.append("ema20_side_invalid_for_hint")

    metrics = {
        "entry_reversal_guard_enabled": True,
        "entry_reversal_guard_passed": len(reasons) == 0,
        "entry_reversal_live_bid": bid,
        "entry_reversal_live_ask": ask,
        "entry_reversal_live_mid": live_mid,
        "entry_reversal_alert_mid": alert_mid,
        "entry_reversal_adverse_pips": round(float(adverse_pips), 4),
        "entry_reversal_spread_pips": round(float(spread_pips), 4),
        "entry_reversal_momentum_aligned": momentum_aligned,
        "entry_reversal_ema20_dist": ema20_dist,
        "entry_reversal_quote_age_seconds": quote.get("quote_age_seconds"),
    }
    if reasons:
        return False, "Entry reversal guard blocked: " + "; ".join(reasons), metrics
    return True, "entry_reversal_guard_passed", metrics

def runtime_noise_filter(payload: Dict[str, Any], feature_row: Dict[str, Any], instrument: str) -> Tuple[bool, str, Dict[str, Any]]:
    if not NOISE_FILTER_ENABLED:
        return True, "noise_filter_disabled", {"noise_filter_enabled": False}
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
    atr_pips = safe_float(feature_row.get("atr_pips"), 0.0)
    spread_pips = safe_float(feature_row.get("spread_pips"), safe_float(payload.get("spread_pips"), 0.0))
    body_range_ratio = body_pips / range_pips if range_pips > 0 else 0.0
    range_atr_ratio = range_pips / atr_pips if atr_pips > 0 else 0.0
    spread_range_ratio = spread_pips / range_pips if range_pips > 0 else 999.0
    wick_body_ratio = wick_total_pips / max(body_pips, 0.1)
    momentum_aligned = bool(safe_float(feature_row.get("signal_momentum_aligned"), 0.0) >= 0.5)
    metrics = {
        "noise_filter_enabled": True,
        "range_pips": round(float(range_pips), 4),
        "body_pips": round(float(body_pips), 4),
        "atr_pips": round(float(atr_pips), 4),
        "spread_pips_runtime": round(float(spread_pips), 4),
        "body_range_ratio": round(float(body_range_ratio), 4),
        "range_atr_ratio": round(float(range_atr_ratio), 4),
        "spread_range_ratio": round(float(spread_range_ratio), 4),
        "wick_body_ratio": round(float(wick_body_ratio), 4),
        "signal_momentum_aligned": momentum_aligned,
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
    if NOISE_FILTER_REQUIRE_SIGNAL_MOMENTUM_ALIGNMENT and not momentum_aligned:
        reasons.append("signal_momentum_not_aligned")
    if reasons:
        return False, "Noise filter blocked: " + "; ".join(reasons), metrics
    return True, "noise_filter_passed", metrics


def compute_units_dynamic(
    instrument: str,
    sl_pips: float,
    avg_auc: float,
    precision_at_gate: float,
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
    if avg_auc >= 0.65 and precision_at_gate >= 0.45:
        base = int(base * 1.15)
    elif avg_auc >= 0.60 and precision_at_gate >= 0.40:
        base = int(base * 1.05)
    elif avg_auc < 0.56:
        base = int(base * 0.80)
    return min(max_units_for_instrument(instrument), max(min_units_for_instrument(instrument), base))


def _round_down_to_pip(price: float, pip: float) -> float:
    return math.floor(price / pip) * pip


def _round_up_to_pip(price: float, pip: float) -> float:
    return math.ceil(price / pip) * pip


def compute_sl_tp_prices(
    side: str,
    reference_price: float,
    atr14: float,
    instrument: str,
    sl_atr: float,
    tp_atr: float,
    min_dist_pips: float = 4.0,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    if side not in ("BUY", "SELL"):
        return None, None, None, None
    pip = instrument_pip_size(instrument)
    atr_value = max(float(atr14), pip)
    sl_dist = max(sl_atr * atr_value, min_dist_pips * pip)
    tp_dist = max(tp_atr * atr_value, min_dist_pips * pip)
    if side == "BUY":
        sl_price = _round_down_to_pip(reference_price - sl_dist, pip)
        tp_price = _round_up_to_pip(reference_price + tp_dist, pip)
        if sl_price >= reference_price:
            sl_price = _round_down_to_pip(reference_price - min_dist_pips * pip, pip)
        if tp_price <= reference_price:
            tp_price = _round_up_to_pip(reference_price + min_dist_pips * pip, pip)
    else:
        sl_price = _round_up_to_pip(reference_price + sl_dist, pip)
        tp_price = _round_down_to_pip(reference_price - tp_dist, pip)
        if sl_price <= reference_price:
            sl_price = _round_up_to_pip(reference_price + min_dist_pips * pip, pip)
        if tp_price >= reference_price:
            tp_price = _round_down_to_pip(reference_price - min_dist_pips * pip, pip)
    sl_str = format_oanda_price(sl_price, instrument)
    tp_str = format_oanda_price(tp_price, instrument)
    ref_str = format_oanda_price(reference_price, instrument)
    return abs(float(ref_str) - float(sl_str)) / pip, abs(float(tp_str) - float(ref_str)) / pip, sl_str, tp_str

# ============================================================
# OANDA LIVE PRICE GUARD
# ============================================================
def broker_ready() -> bool:
    return bool(OANDA_TOKEN and OANDA_ACCOUNT_ID and OANDA_BASE_URL)


def oanda_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OANDA_TOKEN}", "Content-Type": "application/json"}


def oanda_request(method: str, path: str, json_body: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Dict[str, Any]:
    if not broker_ready():
        return {"ok": False, "error": "Missing OANDA env vars"}
    try:
        response = requests.request(
            method=method.upper(),
            url=f"{OANDA_BASE_URL}{path}",
            headers=oanda_headers(),
            json=json_body,
            timeout=timeout,
        )
        try:
            body = response.json()
        except Exception:
            body = response.text
        return {
            "ok": response.status_code in (200, 201),
            "status_code": response.status_code,
            "data" if response.status_code in (200, 201) else "error": body,
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def fetch_live_oanda_quote(instrument: str) -> Dict[str, Any]:
    if not broker_ready():
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
    age_seconds = None
    if quote_time:
        age_seconds = max(0.0, (now_utc() - quote_time).total_seconds())
    if not np.isfinite(bid) or not np.isfinite(ask) or ask < bid:
        return {"ok": False, "error": "Invalid live bid/ask", "data": price}
    if age_seconds is not None and age_seconds > LIVE_PRICE_MAX_AGE_SECONDS:
        return {"ok": False, "error": f"Live quote too old: {age_seconds:.2f}s", "data": price}
    return {
        "ok": True,
        "instrument": instrument,
        "bid": bid,
        "ask": ask,
        "time": price.get("time"),
        "quote_age_seconds": age_seconds,
        "raw": price,
    }




# ============================================================
# EXTERNAL MARKET CONTEXT + AI REVIEW HELPERS
# ============================================================
def fetch_oanda_candles(instrument: str, granularity: str, count: int) -> Dict[str, Any]:
    """Fetch recent OANDA candles outside the TradingView alert payload."""
    safe_count = max(20, min(int(count or 120), 500))
    gran = str(granularity or "M15").upper()
    path = f"/v3/instruments/{instrument}/candles?price=M&granularity={gran}&count={safe_count}"
    return oanda_request("GET", path, timeout=MARKET_CONTEXT_MAX_FETCH_SECONDS)


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
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def _direction_label(value: float, eps: float = 0.0) -> str:
    if value > eps:
        return "bullish"
    if value < -eps:
        return "bearish"
    return "neutral"



def classify_latest_candle_pattern(work: pd.DataFrame, instrument: str, lookback_trend_bars: int = 5) -> Dict[str, Any]:
    """Classify the latest completed candle for AI review context.

    This is intentionally rule-based and conservative. It does not place trades;
    it gives the AI reviewer structured context such as doji, hammer, shooting
    star, engulfing, strong bull/bear, inside bar, or outside bar.
    """
    if work.empty or len(work) < 1:
        return {
            "enabled": CANDLE_PATTERN_CONTEXT_ENABLED,
            "pattern": "unknown",
            "candle_bias": "NEUTRAL",
            "pattern_confidence": 0,
            "reason": "no_candles",
        }

    pip = instrument_pip_size(instrument)
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
    body_signed = c - o

    range_pips = rng / pip if pip > 0 else 0.0
    body_pips = body / pip if pip > 0 else 0.0
    body_pips_signed = body_signed / pip if pip > 0 else 0.0
    upper_wick_pips = upper / pip if pip > 0 else 0.0
    lower_wick_pips = lower / pip if pip > 0 else 0.0
    body_range_ratio = body / rng if rng > 0 else 0.0
    upper_wick_range_ratio = upper / rng if rng > 0 else 0.0
    lower_wick_range_ratio = lower / rng if rng > 0 else 0.0
    wick_body_ratio = max(upper, lower) / body if body > 0 else 999.0

    pattern = "neutral"
    bias = "NEUTRAL"
    confidence = 25
    reason = "small_or_mixed_candle"

    # Prev-candle relationship.
    prev_o = prev_c = prev_h = prev_l = None
    if prev is not None:
        prev_o = safe_float(prev.get("mid_o"), 0.0)
        prev_h = safe_float(prev.get("mid_h"), 0.0)
        prev_l = safe_float(prev.get("mid_l"), 0.0)
        prev_c = safe_float(prev.get("mid_c"), 0.0)

    # Light recent trend label for interpreting hammer / shooting star context.
    trend_bias = "NEUTRAL"
    if len(work) >= max(3, lookback_trend_bars + 1):
        prior = safe_float(work["mid_c"].iloc[-lookback_trend_bars-1], 0.0)
        if prior > 0:
            recent_change = (c / prior) - 1.0
            if recent_change > 0:
                trend_bias = "BUY"
            elif recent_change < 0:
                trend_bias = "SELL"

    # Engulfing first because it uses the prior candle body and is useful context.
    if prev is not None and prev_o is not None and prev_c is not None:
        prev_bear = prev_c < prev_o
        prev_bull = prev_c > prev_o
        curr_bull = c > o
        curr_bear = c < o
        body_engulfs_prev = min(o, c) <= min(prev_o, prev_c) and max(o, c) >= max(prev_o, prev_c)
        if curr_bull and prev_bear and body_engulfs_prev and body_range_ratio >= 0.35:
            pattern = "bullish_engulfing"
            bias = "BUY"
            confidence = 78
            reason = "bullish_body_engulfed_prior_bearish_body"
        elif curr_bear and prev_bull and body_engulfs_prev and body_range_ratio >= 0.35:
            pattern = "bearish_engulfing"
            bias = "SELL"
            confidence = 78
            reason = "bearish_body_engulfed_prior_bullish_body"

    if pattern == "neutral":
        if rng <= 0 or body_range_ratio <= 0.10 or body_pips <= 0.2:
            pattern = "doji"
            bias = "NEUTRAL"
            confidence = 65
            reason = "very_small_body_relative_to_range"
        elif lower >= 2.0 * max(body, 1e-12) and upper <= 0.8 * max(body, 1e-12) and lower_wick_range_ratio >= 0.45:
            pattern = "hammer" if trend_bias in {"SELL", "NEUTRAL"} else "hanging_man"
            bias = "BUY" if pattern == "hammer" else "SELL"
            confidence = 72 if trend_bias != "NEUTRAL" else 62
            reason = "long_lower_wick_rejection"
        elif upper >= 2.0 * max(body, 1e-12) and lower <= 0.8 * max(body, 1e-12) and upper_wick_range_ratio >= 0.45:
            pattern = "shooting_star" if trend_bias in {"BUY", "NEUTRAL"} else "inverted_hammer"
            bias = "SELL" if pattern == "shooting_star" else "BUY"
            confidence = 72 if trend_bias != "NEUTRAL" else 62
            reason = "long_upper_wick_rejection"
        elif body_range_ratio >= 0.65 and body_signed > 0:
            pattern = "strong_bull"
            bias = "BUY"
            confidence = 70
            reason = "large_bullish_body_close_near_high"
        elif body_range_ratio >= 0.65 and body_signed < 0:
            pattern = "strong_bear"
            bias = "SELL"
            confidence = 70
            reason = "large_bearish_body_close_near_low"
        elif prev is not None and prev_h is not None and prev_l is not None and h <= prev_h and l >= prev_l:
            pattern = "inside_bar"
            bias = "NEUTRAL"
            confidence = 55
            reason = "inside_previous_candle_range"
        elif prev is not None and prev_h is not None and prev_l is not None and h >= prev_h and l <= prev_l:
            pattern = "outside_bar"
            bias = "BUY" if body_signed > 0 else "SELL" if body_signed < 0 else "NEUTRAL"
            confidence = 60
            reason = "outside_previous_candle_range"

    return {
        "enabled": CANDLE_PATTERN_CONTEXT_ENABLED,
        "pattern": pattern,
        "candle_bias": bias,
        "pattern_confidence": int(confidence),
        "reason": reason,
        "trend_bias_last5": trend_bias,
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "body_pips": float(body_pips),
        "body_pips_signed": float(body_pips_signed),
        "range_pips": float(range_pips),
        "upper_wick_pips": float(upper_wick_pips),
        "lower_wick_pips": float(lower_wick_pips),
        "body_range_ratio": float(body_range_ratio),
        "upper_wick_range_ratio": float(upper_wick_range_ratio),
        "lower_wick_range_ratio": float(lower_wick_range_ratio),
        "wick_body_ratio": float(wick_body_ratio),
    }

def summarize_market_dataframe(df: pd.DataFrame, instrument: str, granularity: str, hint_side: str) -> Dict[str, Any]:
    """Convert raw candles into compact fields for rule checks and AI review."""
    if df.empty or len(df) < 20:
        return {"ok": False, "granularity": granularity, "reason": f"not_enough_candles:{len(df)}"}

    work = df.copy()
    for col in ["mid_o", "mid_h", "mid_l", "mid_c", "volume"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["mid_o", "mid_h", "mid_l", "mid_c"])
    if len(work) < 20:
        return {"ok": False, "granularity": granularity, "reason": f"not_enough_valid_candles:{len(work)}"}

    pip = instrument_pip_size(instrument)
    close = work["mid_c"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    atr = atr_runtime(work, 14)
    rsi = rsi_runtime(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    last = work.iloc[-1]
    prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
    last_close = safe_float(last.get("mid_c"), 0.0)
    body_pips_signed = (safe_float(last.get("mid_c"), 0.0) - safe_float(last.get("mid_o"), 0.0)) / pip if pip > 0 else 0.0
    range_pips = (safe_float(last.get("mid_h"), 0.0) - safe_float(last.get("mid_l"), 0.0)) / pip if pip > 0 else 0.0
    recent_high = float(work["mid_h"].tail(20).max())
    recent_low = float(work["mid_l"].tail(20).min())
    ret1 = (last_close / prev_close - 1.0) if prev_close and prev_close > 0 else 0.0
    ret3 = close.pct_change(3).iloc[-1] if len(close) >= 4 else 0.0
    ret5 = close.pct_change(5).iloc[-1] if len(close) >= 6 else 0.0
    ema20_dist = (last_close - float(ema20.iloc[-1])) / last_close if last_close > 0 else 0.0
    ema50_dist = (last_close - float(ema50.iloc[-1])) / last_close if last_close > 0 else 0.0
    ema200_dist = (last_close - float(ema200.iloc[-1])) / last_close if last_close > 0 else 0.0
    trend_score = 0
    for value in [ret1, ret3, ret5, ema20_dist, ema50_dist, float(macd_hist.iloc[-1]), body_pips_signed]:
        sign = _sign_for_direction(float(value), 0.0)
        if hint_side == "BUY" and sign > 0:
            trend_score += 1
        elif hint_side == "SELL" and sign < 0:
            trend_score += 1
    side_aligned = trend_score >= 4
    bullish_count_5 = int(((work["mid_c"].tail(5) - work["mid_o"].tail(5)) > 0).sum())
    bearish_count_5 = int(((work["mid_c"].tail(5) - work["mid_o"].tail(5)) < 0).sum())
    candle_pattern = classify_latest_candle_pattern(work, instrument) if CANDLE_PATTERN_CONTEXT_ENABLED else {"enabled": False, "pattern": "disabled", "candle_bias": "NEUTRAL", "pattern_confidence": 0}

    return {
        "ok": True,
        "granularity": granularity,
        "candles": int(len(work)),
        "last_time": work.iloc[-1]["time"].isoformat() if hasattr(work.iloc[-1]["time"], "isoformat") else str(work.iloc[-1]["time"]),
        "last_close": float(last_close),
        "ret1": float(ret1),
        "ret3": float(ret3) if np.isfinite(ret3) else 0.0,
        "ret5": float(ret5) if np.isfinite(ret5) else 0.0,
        "ema20_dist": float(ema20_dist),
        "ema50_dist": float(ema50_dist),
        "ema200_dist": float(ema200_dist),
        "price_vs_ema20": _direction_label(ema20_dist),
        "price_vs_ema50": _direction_label(ema50_dist),
        "price_vs_ema200": _direction_label(ema200_dist),
        "rsi14": float(rsi.iloc[-1]) if np.isfinite(rsi.iloc[-1]) else 50.0,
        "atr_pips": float(atr.iloc[-1] / pip) if pip > 0 and np.isfinite(atr.iloc[-1]) else 0.0,
        "macd_hist": float(macd_hist.iloc[-1]) if np.isfinite(macd_hist.iloc[-1]) else 0.0,
        "body_pips_signed": float(body_pips_signed),
        "range_pips": float(range_pips),
        "recent_high_20": recent_high,
        "recent_low_20": recent_low,
        "distance_to_recent_high_pips": float((recent_high - last_close) / pip) if pip > 0 else 0.0,
        "distance_to_recent_low_pips": float((last_close - recent_low) / pip) if pip > 0 else 0.0,
        "bullish_candles_last5": bullish_count_5,
        "bearish_candles_last5": bearish_count_5,
        "hint_side_alignment_score": int(trend_score),
        "hint_side_aligned": bool(side_aligned),
    }


def compare_alert_to_market_context(
    feature_row: Dict[str, Any],
    market_summary: Dict[str, Any],
    hint_side: str,
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {"ok": True, "warnings": []}
    if not market_summary.get("ok"):
        comparisons["ok"] = False
        comparisons["warnings"].append(str(market_summary.get("reason") or "market_summary_unavailable"))
        return comparisons

    alert_ema20 = safe_float(feature_row.get("ema20_dist"), 0.0)
    alert_ema50 = safe_float(feature_row.get("ema50_dist"), 0.0)
    alert_ret3 = safe_float(feature_row.get("ret3"), 0.0)
    market_ema20 = safe_float(market_summary.get("ema20_dist"), 0.0)
    market_ema50 = safe_float(market_summary.get("ema50_dist"), 0.0)
    market_ret3 = safe_float(market_summary.get("ret3"), 0.0)

    side = 1 if hint_side == "BUY" else -1 if hint_side == "SELL" else 0
    for label, alert_value, market_value in [
        ("ema20_dist", alert_ema20, market_ema20),
        ("ema50_dist", alert_ema50, market_ema50),
        ("ret3", alert_ret3, market_ret3),
    ]:
        alert_sign = _sign_for_direction(alert_value, 0.0)
        market_sign = _sign_for_direction(market_value, 0.0)
        if alert_sign and market_sign and alert_sign != market_sign:
            comparisons["warnings"].append(f"alert_market_{label}_sign_mismatch:alert={alert_value:.8f},market={market_value:.8f}")

    if side and not bool(market_summary.get("hint_side_aligned", False)):
        comparisons["warnings"].append(f"market_context_not_aligned_with_{hint_side.lower()}")
    candle_bias = str(market_summary.get("last_candle_bias") or "NEUTRAL").upper()
    candle_pattern = str(market_summary.get("last_candle_pattern") or "unknown")
    if side and candle_bias in {"BUY", "SELL"} and candle_bias != hint_side:
        comparisons["warnings"].append(f"last_candle_pattern_conflicts_with_{hint_side.lower()}:pattern={candle_pattern},bias={candle_bias}")
    comparisons["ok"] = len(comparisons["warnings"]) == 0
    return comparisons


def build_external_market_context(
    pair6: str,
    instrument: str,
    hint_side: str,
    feature_row: Dict[str, Any],
) -> Dict[str, Any]:
    if not MARKET_CONTEXT_ENABLED:
        return {"enabled": False, "ok": True, "reason": "market_context_disabled"}
    if not broker_ready():
        return {"enabled": True, "ok": not MARKET_CONTEXT_REQUIRED, "reason": "broker_not_ready_for_market_context"}

    summaries: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    errors: List[str] = []
    for granularity in MARKET_CONTEXT_GRANULARITIES:
        result = fetch_oanda_candles(instrument, granularity, MARKET_CONTEXT_CANDLE_COUNT)
        if not result.get("ok"):
            errors.append(f"{granularity}:fetch_failed:{result.get('error') or result.get('status_code')}")
            summaries[granularity] = {"ok": False, "granularity": granularity, "reason": str(result.get("error") or result.get("status_code") or "fetch_failed")}
            continue
        df = candles_to_dataframe(result)
        summary = summarize_market_dataframe(df, instrument, granularity, hint_side)
        summaries[granularity] = summary
        comparisons[granularity] = compare_alert_to_market_context(feature_row, summary, hint_side)

    ok_summaries = [summary for summary in summaries.values() if summary.get("ok")]
    aligned_count = sum(1 for summary in ok_summaries if summary.get("hint_side_aligned"))
    h1 = summaries.get("H1", {})
    h4 = summaries.get("H4", {})
    higher_tf_conflict = False
    side = 1 if hint_side == "BUY" else -1 if hint_side == "SELL" else 0
    if side:
        for label, summary in [("H1", h1), ("H4", h4)]:
            if not summary.get("ok"):
                continue
            ema50_sign = _sign_for_direction(safe_float(summary.get("ema50_dist"), 0.0), 0.0)
            if ema50_sign == -side:
                higher_tf_conflict = True
                errors.append(f"{label}_ema50_countertrend")

    return {
        "enabled": True,
        "ok": bool(ok_summaries) and (not MARKET_CONTEXT_REQUIRED or not errors),
        "instrument": instrument,
        "pair": pair6,
        "hint_side": hint_side,
        "granularities": MARKET_CONTEXT_GRANULARITIES,
        "summaries": summaries,
        "comparisons": comparisons,
        "aligned_timeframes": aligned_count,
        "available_timeframes": len(ok_summaries),
        "higher_timeframe_conflict": higher_tf_conflict,
        "errors": errors,
    }



# ============================================================
# SIDE-AWARE AI REVIEWER HELPERS
# ============================================================
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

    This is intentionally independent of the alert side. It fixes the old reviewer behavior
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
        return ai_side_norm(candle_ctx.get("candle_bias") or context.get("latest_m15_candle_bias"))
    return ai_side_norm(context.get("latest_m15_candle_bias"))


def ai_side_aware_rule_review(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic side-aware AI review.

    This runs before/around the LLM reviewer and is used as a safe fallback.
    Key fix:
    - bearish trend supports SELL
    - bullish trend supports BUY
    """
    side = ai_side_norm(context.get("side"))
    prob = safe_float(context.get("model_approval_probability"), 0.0)
    market_context = context.get("external_market_context") or {}
    summaries = market_context.get("summaries") or {}
    risk_context = context.get("risk_context") or {}
    model_features = context.get("model_features") or {}

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

    # 2. Side-aware MTF trend alignment
    trends: Dict[str, str] = {}
    aligned_count = 0
    conflict_count = 0
    for tf in ("M15", "H1", "H4"):
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

    # 3. Latest candle/pattern side-awareness
    pattern_bias = ai_pattern_bias_from_context(context)
    candle_ctx = context.get("latest_candlestick_context") or {}
    pattern_name = str(candle_ctx.get("pattern") or context.get("latest_m15_candle_pattern") or "unknown")
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
    if risk_context.get("live_quote_guard_passed") is False:
        risk += 15
        reasons.append("live_quote_guard_failed")
    if market_context.get("higher_timeframe_conflict") and conflict_count > aligned_count:
        risk += 10
        reasons.append("higher_timeframe_conflict")

    # 5. Spread/ATR
    spread_pips = safe_float(risk_context.get("spread_pips"), safe_float(model_features.get("spread_pips"), 0.0))
    atr_pips = safe_float(model_features.get("atr_pips"), 0.0)
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

    # 6. Historical pattern stats only when sample is usable
    hist = context.get("historical_pattern_performance") or {}
    pair_stats = hist.get("pair_pattern_stats") or {}
    min_closed = safe_int(hist.get("min_closed_for_confidence"), PATTERN_STATS_MIN_CLOSED)
    closed_samples = safe_int(pair_stats.get("closed_samples"), 0)
    win_rate = safe_float(pair_stats.get("win_rate"), -1.0)
    if closed_samples >= min_closed and win_rate >= 0:
        if win_rate < 0.45:
            risk += 10
            reasons.append(f"historical_pattern_win_rate_low:{win_rate:.3f}")
        elif win_rate >= 0.55:
            risk -= 5
            supports.append(f"historical_pattern_win_rate_ok:{win_rate:.3f}")

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

    explanation = (
        f"Side-aware review for {side}: risk={risk}, probability={prob * 100:.1f}%. "
        f"Trends M15={trends.get('M15')}, H1={trends.get('H1')}, H4={trends.get('H4')}. "
        f"Supports={supports[:6]}. Conflicts={conflicts[:6]}. Reasons={reasons[:6]}."
    )

    return {
        "enabled": True,
        "provider": "side_aware_rules",
        "model": "deterministic",
        "ai_verdict": verdict,
        "decision": decision,
        "risk_score": risk,
        "reason": f"{reason_code}. {explanation}",
        "side": side,
        "model_probability": prob,
        "m15_trend": trends.get("M15"),
        "h1_trend": trends.get("H1"),
        "h4_trend": trends.get("H4"),
        "aligned_timeframes": aligned_count,
        "conflicting_timeframes": conflict_count,
        "supports": supports,
        "conflicts": conflicts,
        "risk_reasons": reasons,
        "spread_atr": spread_atr,
    }


def ai_merge_llm_and_rule_reviews(rule_review: Dict[str, Any], llm_review: Dict[str, Any]) -> Dict[str, Any]:
    """Keep LLM context, but never let wording bugs reverse side logic.

    The LLM may still reject for genuinely low probability, high spread, or conflict.
    But if the deterministic side-aware reviewer approves and the LLM rejects only because
    it misreads bearish-as-bad-for-sell or bullish-as-bad-for-buy, this correction prevents
    that specific false block.
    """
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

    # If LLM approves, keep approval but include rule telemetry.
    if llm_verdict == "APPROVE" and llm_risk < AI_REVIEW_HARD_BLOCK_SCORE:
        return merged

    # If deterministic side-aware reviewer approves and LLM is only in the conditional range,
    # allow. This is the fix for SELL + bearish trend being mistakenly treated as conflict.
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


def build_ai_review_context(
    pair6: str,
    instrument: str,
    hint_side: str,
    model_type: str,
    decision_prob: float,
    gate: float,
    feature_row: Dict[str, Any],
    market_context: Dict[str, Any],
    risk_context: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "pair": pair6,
        "instrument": instrument,
        "side": hint_side,
        "model_type": model_type,
        "model_approval_probability": float(decision_prob),
        "required_conf_gate": float(gate),
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
        "model_features": {
            "ret1": safe_float(feature_row.get("ret1"), 0.0),
            "ret3": safe_float(feature_row.get("ret3"), 0.0),
            "ret5": safe_float(feature_row.get("ret5"), 0.0),
            "ema20_dist": safe_float(feature_row.get("ema20_dist"), 0.0),
            "ema50_dist": safe_float(feature_row.get("ema50_dist"), 0.0),
            "ema200_dist": safe_float(feature_row.get("ema200_dist"), 0.0),
            "rsi14": safe_float(feature_row.get("rsi14"), 50.0),
            "macd_hist": safe_float(feature_row.get("macd_hist"), 0.0),
            "atr_pips": safe_float(feature_row.get("atr_pips"), 0.0),
            "spread_pips": safe_float(feature_row.get("spread_pips"), 0.0),
            "range_pips": safe_float(feature_row.get("range_pips"), 0.0),
            "body_pips": safe_float(feature_row.get("body_pips"), 0.0),
            "body_range_ratio": safe_float(feature_row.get("body_range_ratio"), 0.0),
        },
        "latest_candlestick_context": ((market_context.get("summaries") or {}).get("M15") or {}).get("candle_pattern", {}),
        "latest_m15_candle_bias": ((market_context.get("summaries") or {}).get("M15") or {}).get("last_candle_bias"),
        "latest_m15_candle_pattern": ((market_context.get("summaries") or {}).get("M15") or {}).get("last_candle_pattern"),
        "historical_pattern_performance": get_pattern_performance_summary(pair6, hint_side, market_context),
        "risk_context": risk_context,
        "external_market_context": market_context,
    }


def _parse_json_object_from_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def review_signal_with_ai(context: Dict[str, Any]) -> Dict[str, Any]:
    """AI compares alert features with fresh market context, then approves or rejects.

    v16 change:
    - Runs a deterministic side-aware review first.
    - Bearish M15/H1/H4 supports SELL.
    - Bullish M15/H1/H4 supports BUY.
    - Uses conditional approval instead of one hard 60 risk-score block.
    - If API key is missing and fallback is enabled, it uses side-aware rules instead of blocking all trades.
    """
    if not AI_REVIEW_ENABLED:
        return {"enabled": False, "ai_verdict": "SKIPPED", "risk_score": 0, "reason": "AI review disabled"}

    rule_review = ai_side_aware_rule_review(context)

    # Optional deterministic mode for testing:
    # AI_REVIEW_PROVIDER=rules
    if AI_REVIEW_PROVIDER in {"rules", "rule", "deterministic", "none"}:
        return rule_review

    system_prompt = (
        "You are a conservative forex trade risk reviewer. You do not place trades. "
        "Return JSON only with ai_verdict APPROVE or REJECT, risk_score integer 0-100, and reason. "
        "CRITICAL SIDE-AWARE RULES: bullish trend supports BUY; bearish trend supports SELL. "
        "Do not reject a SELL because EMA/momentum/trend is bearish. That supports SELL. "
        "Do not reject a BUY because EMA/momentum/trend is bullish. That supports BUY. "
        "Reject when the alert side conflicts with the trend: SELL conflicts with bullish; BUY conflicts with bearish. "
        "Use the provided side_aware_rule_review as a safety reference. "
        "Approve only when model probability, trend, EMA/momentum, spread, latest candle behavior, and usable historical stats mostly agree. "
        "Use historical_pattern_performance only when sample size is usable; treat low-sample stats as weak context. "
        "Reject genuine conflicts: alert direction disagrees with M15/H1/H4 trend, latest candle pattern or bias conflicts with the trade, "
        "high spread vs ATR, price near exhaustion, live reversal risk, weak momentum, or noisy/doji indecision."
    )

    context_with_rules = dict(context)
    context_with_rules["side_aware_rule_review"] = rule_review
    user_payload = json.dumps(context_with_rules, default=str, separators=(",", ":"))

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

        # Tiered validation instead of "risk > 60 blocks everything".
        prob = safe_float(context.get("model_approval_probability"), 0.0)
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

def live_quote_guard_reprice_sltp(
    side: str,
    instrument: str,
    atr14: float,
    sl_atr: float,
    tp_atr: float,
    existing_sl_price: Optional[str],
    existing_tp_price: Optional[str],
) -> Tuple[bool, str, Dict[str, Any]]:
    if not LIVE_PRICE_GUARD_ENABLED:
        return True, "live_price_guard_disabled", {"live_quote_guard_enabled": False}
    quote = fetch_live_oanda_quote(instrument)
    if not quote.get("ok"):
        if LIVE_PRICE_GUARD_REQUIRED:
            return False, f"Live quote guard blocked: {quote.get('error')}", {"live_quote_guard_enabled": True, "live_quote_guard_passed": False, "live_quote_error": quote.get("error")}
        return True, f"Live quote guard unavailable but optional: {quote.get('error')}", {"live_quote_guard_enabled": True, "live_quote_guard_passed": True, "live_quote_error": quote.get("error")}
    bid = safe_float(quote.get("bid"), 0.0)
    ask = safe_float(quote.get("ask"), 0.0)
    pip = instrument_pip_size(instrument)
    buffer = LIVE_PRICE_BUFFER_PIPS * pip
    entry_reference = ask if side == "BUY" else bid
    if LIVE_PRICE_REPRICE_SLTP:
        sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(
            side,
            entry_reference,
            atr14,
            instrument,
            sl_atr,
            tp_atr,
        )
    else:
        sl_price = existing_sl_price
        tp_price = existing_tp_price
        sl_pips = None
        tp_pips = None
    sl_value = safe_float(sl_price, np.nan)
    tp_value = safe_float(tp_price, np.nan)
    invalid_reason = None
    if side == "BUY":
        if not np.isfinite(tp_value) or tp_value <= ask + buffer:
            invalid_reason = f"live_quote_invalid_buy_tp:{tp_value}<=ask_plus_buffer:{ask + buffer}"
        elif not np.isfinite(sl_value) or sl_value >= bid - buffer:
            invalid_reason = f"live_quote_invalid_buy_sl:{sl_value}>=bid_minus_buffer:{bid - buffer}"
    else:
        if not np.isfinite(tp_value) or tp_value >= bid - buffer:
            invalid_reason = f"live_quote_invalid_sell_tp:{tp_value}>=bid_minus_buffer:{bid - buffer}"
        elif not np.isfinite(sl_value) or sl_value <= ask + buffer:
            invalid_reason = f"live_quote_invalid_sell_sl:{sl_value}<=ask_plus_buffer:{ask + buffer}"
    metrics = {
        "live_quote_guard_enabled": True,
        "live_quote_guard_passed": invalid_reason is None,
        "live_bid": bid,
        "live_ask": ask,
        "live_quote_time": quote.get("time"),
        "live_quote_age_seconds": quote.get("quote_age_seconds"),
        "live_price_buffer_pips": LIVE_PRICE_BUFFER_PIPS,
        "live_price_reprice_sltp": LIVE_PRICE_REPRICE_SLTP,
        "live_reprice_reference_price": entry_reference,
        "live_reprice_final_sl_price": sl_price,
        "live_reprice_final_tp_price": tp_price,
        "live_reprice_final_sl_pips": sl_pips,
        "live_reprice_final_tp_pips": tp_pips,
    }
    if invalid_reason:
        return False, f"Live quote guard blocked: {invalid_reason}", metrics
    return True, "live_quote_sltp_valid", metrics


def submit_oanda_order(instrument: str, units_signed: int, sl_price: str, tp_price: str, client_id: str) -> Dict[str, Any]:
    payload = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units_signed)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientExtensions": {"id": client_id, "tag": "fx_m15_signal_approval", "comment": APP_VERSION},
            "tradeClientExtensions": {"id": client_id, "tag": "fx_m15_signal_approval", "comment": APP_VERSION},
            "stopLossOnFill": {"price": sl_price, "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": tp_price, "timeInForce": "GTC"},
        }
    }
    return oanda_request("POST", f"/v3/accounts/{OANDA_ACCOUNT_ID}/orders", payload, timeout=30)


def close_oanda_trade_by_specifier(trade_specifier: str) -> Dict[str, Any]:
    if not trade_specifier:
        return {"ok": False, "error": "Missing trade_specifier"}
    return oanda_request("PUT", f"/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_specifier}/close", {"units": "ALL"})


def close_oanda_position_side(instrument: str, side: str) -> Dict[str, Any]:
    payload = {"longUnits": "NONE", "shortUnits": "NONE"}
    if side == "BUY":
        payload["longUnits"] = "ALL"
    elif side == "SELL":
        payload["shortUnits"] = "ALL"
    else:
        return {"ok": False, "error": f"Unsupported side: {side}"}
    return oanda_request("PUT", f"/v3/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}/close", payload)



def trade_specifier_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    broker_trade_id = str(meta.get("broker_trade_id") or "").strip()
    client_trade_id = str(meta.get("client_trade_id") or "").strip()
    if broker_trade_id:
        return broker_trade_id
    if client_trade_id:
        return f"@{client_trade_id}"
    return None


def get_oanda_trade_details(trade_specifier: str) -> Dict[str, Any]:
    """Fetch the full OANDA trade representation for a trade ID or @clientTradeID."""
    if not trade_specifier:
        return {"ok": False, "error": "Missing trade_specifier"}
    return oanda_request(
        "GET",
        f"/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_specifier}",
        timeout=20,
    )


def get_oanda_transaction_details(transaction_id: str) -> Dict[str, Any]:
    """Fetch one OANDA account transaction by ID."""
    if not transaction_id:
        return {"ok": False, "error": "Missing transaction_id"}
    return oanda_request(
        "GET",
        f"/v3/accounts/{OANDA_ACCOUNT_ID}/transactions/{transaction_id}",
        timeout=20,
    )


def classify_closed_trade_from_transactions(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a broker-detected closed trade using OANDA closing transaction details.

    OANDA's closed Trade object exposes closingTransactionIDs. The referenced
    transaction is commonly an ORDER_FILL whose `reason` indicates whether the
    trade was closed by a Take Profit, Stop Loss, Trailing Stop Loss, or an
    explicit market trade close.
    """
    closing_ids = [str(x) for x in (trade.get("closingTransactionIDs") or []) if str(x).strip()]
    result: Dict[str, Any] = {
        "status": "CLOSED",
        "close_classification": "CLOSED_UNKNOWN",
        "close_reason_code": "UNKNOWN",
        "close_reason_label": "Closed - reason unavailable",
        "closing_transaction_ids": closing_ids,
        "closing_transaction_id_used": None,
        "closing_transaction_type": None,
        "closing_transaction_reason": None,
        "closing_transaction": None,
        "classification_source": "none",
        "classification_errors": [],
    }

    if not CLOSED_TRADE_CLASSIFICATION_ENABLED:
        result.update(
            {
                "close_classification": "CLOSED_UNCLASSIFIED",
                "close_reason_code": "CLASSIFICATION_DISABLED",
                "close_reason_label": "Closed - classification disabled",
                "classification_source": "disabled",
            }
        )
        return result

    if not closing_ids:
        result["classification_errors"].append("no_closing_transaction_ids")
        return result

    # Inspect most recent closing transaction first; that usually represents
    # the transaction that fully closed the trade.
    max_to_check = max(1, CLOSED_TRADE_CLASSIFICATION_MAX_TRANSACTIONS)
    ids_to_check = list(reversed(closing_ids))[:max_to_check]

    # Map OANDA OrderFillReason values to our trade-event statuses.
    reason_map = {
        "TAKE_PROFIT_ORDER": ("TAKE_PROFIT", "TAKE_PROFIT", "Take Profit hit"),
        "STOP_LOSS_ORDER": ("STOPPED", "STOP_LOSS", "Stop Loss hit"),
        "GUARANTEED_STOP_LOSS_ORDER": ("STOPPED", "GUARANTEED_STOP_LOSS", "Guaranteed Stop Loss hit"),
        "TRAILING_STOP_LOSS_ORDER": ("STOPPED", "TRAILING_STOP_LOSS", "Trailing Stop Loss hit"),
        "MARKET_ORDER_TRADE_CLOSE": ("MANUAL", "MANUAL_TRADE_CLOSE", "Trade manually/explicitly closed"),
        "MARKET_ORDER_POSITION_CLOSEOUT": ("MANUAL", "MANUAL_POSITION_CLOSE", "Position manually/explicitly closed"),
        # These are preserved as CLOSED rather than mislabeled.
        "MARKET_ORDER_MARGIN_CLOSEOUT": ("CLOSED", "MARGIN_CLOSEOUT", "Closed by margin closeout"),
        "MARKET_ORDER_DELAYED_TRADE_CLOSE": ("CLOSED", "DELAYED_TRADE_CLOSE", "Delayed trade close"),
    }

    for transaction_id in ids_to_check:
        tx_result = get_oanda_transaction_details(transaction_id)
        if not tx_result.get("ok"):
            result["classification_errors"].append(
                f"transaction_lookup_failed:{transaction_id}:{tx_result.get('error')}"
            )
            continue

        tx_data = tx_result.get("data") or {}
        transaction = tx_data.get("transaction") or {}
        tx_type = str(transaction.get("type") or "").upper()
        tx_reason = str(transaction.get("reason") or "").upper()

        result.update(
            {
                "closing_transaction_id_used": transaction_id,
                "closing_transaction_type": tx_type or None,
                "closing_transaction_reason": tx_reason or None,
                "closing_transaction": transaction,
                "classification_source": "oanda_closing_transaction",
            }
        )

        if tx_type == "ORDER_FILL" and tx_reason in reason_map:
            status, code, label = reason_map[tx_reason]
            result.update(
                {
                    "status": status,
                    "close_classification": status,
                    "close_reason_code": code,
                    "close_reason_label": label,
                }
            )
            return result

        # Sometimes a manually-created fill may not use one of the explicit
        # MARKET_ORDER_* trade-close reasons above. Keep it generic, but retain
        # the transaction details in logs for later inspection.
        if tx_type == "ORDER_FILL":
            result.update(
                {
                    "close_classification": "CLOSED_ORDER_FILL_UNKNOWN",
                    "close_reason_code": tx_reason or "ORDER_FILL_UNKNOWN",
                    "close_reason_label": f"Closed by order fill ({tx_reason or 'unknown reason'})",
                }
            )

    return result


def _trade_close_status_from_oanda(trade: Dict[str, Any]) -> str:
    """Compatibility helper: return the classified status when possible."""
    return classify_closed_trade_from_transactions(trade).get("status", "CLOSED")


def register_closed_trade_from_oanda(
    tracking_key: str,
    meta: Dict[str, Any],
    trade_specifier: str,
    trade: Dict[str, Any],
    broker_response: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a CLOSED trade event once OANDA reports the trade state as CLOSED."""
    classification = classify_closed_trade_from_transactions(trade)
    status = str(classification.get("status") or "CLOSED")
    close_time = str(trade.get("closeTime") or utc_ts())
    realized_pl = safe_float(trade.get("realizedPL"), 0.0)
    average_close_price = safe_float(trade.get("averageClosePrice"), 0.0)
    closing_transaction_ids = classification.get("closing_transaction_ids") or trade.get("closingTransactionIDs") or []
    close_reason_code = str(classification.get("close_reason_code") or "UNKNOWN")
    close_reason_label = str(classification.get("close_reason_label") or "Closed - reason unavailable")
    closing_tx_id_used = classification.get("closing_transaction_id_used")
    closing_tx_reason = classification.get("closing_transaction_reason")
    reason = (
        f"OANDA trade sync registered {status}; specifier={trade_specifier}; "
        f"close_reason_code={close_reason_code}; close_reason_label={close_reason_label}; "
        f"close_time={close_time}; average_close_price={average_close_price}; "
        f"closing_transaction_id_used={closing_tx_id_used or 'none'}; "
        f"closing_transaction_reason={closing_tx_reason or 'none'}; "
        f"closing_transaction_ids={','.join(map(str, closing_transaction_ids)) if closing_transaction_ids else 'none'}"
    )

    row = {
        "instrument": meta.get("instrument"),
        "side": meta.get("side"),
        "units_signed": meta.get("units_signed"),
        "entry_price": meta.get("entry_price"),
        "sl_price": meta.get("sl_price"),
        "tp_price": meta.get("tp_price"),
        "status": status,
        "pnl": realized_pl,
        "order_id": meta.get("order_id"),
        "reason": reason,
        "pair_score": meta.get("pair_score"),
        "ts": close_time,
        "tracking_key": str(tracking_key),
        "broker_trade_id": meta.get("broker_trade_id") or trade.get("id"),
        "broker_order_id": meta.get("broker_order_id"),
        "client_trade_id": meta.get("client_trade_id"),
    }
    write_trade_row(row)

    write_trade_management_event(
        {
            "ts": utc_ts(),
            "tracking_key": str(tracking_key),
            "instrument": meta.get("instrument"),
            "side": meta.get("side"),
            "action": "CLOSED_TRADE_SYNC",
            "trade_specifier": trade_specifier,
            "entry_price": meta.get("entry_price"),
            "live_bid": None,
            "live_ask": None,
            "favorable_pips": meta.get("last_favorable_pips"),
            "initial_risk_pips": initial_risk_pips_from_meta(meta),
            "current_r": meta.get("last_current_r"),
            "previous_sl_price": meta.get("sl_price"),
            "requested_sl_price": None,
            "updated_sl_price": meta.get("sl_price"),
            "success": True,
            "reason": reason,
            "broker_response": {
                "trade_sync_response": broker_response or trade,
                "close_classification": classification,
            },
        }
    )

    note_trade_closed(tracking_key)
    _open_trade_meta.pop(str(tracking_key), None)
    return {
        "ok": True,
        "action": "CLOSED_TRADE_SYNC",
        "tracking_key": str(tracking_key),
        "trade_specifier": trade_specifier,
        "status": status,
        "pnl": realized_pl,
        "close_time": close_time,
        "average_close_price": average_close_price,
        "closing_transaction_ids": closing_transaction_ids,
        "close_reason_code": close_reason_code,
        "close_reason_label": close_reason_label,
        "closing_transaction_id_used": closing_tx_id_used,
        "closing_transaction_reason": closing_tx_reason,
        "close_classification": classification,
    }


def sync_single_tracked_trade_close(tracking_key: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Check one tracked open trade in OANDA and register it if it is now CLOSED."""
    if not CLOSED_TRADE_SYNC_ENABLED:
        return {"ok": True, "action": "SKIP", "reason": "closed_trade_sync_disabled"}
    trade_specifier = trade_specifier_from_meta(meta)
    if CLOSED_TRADE_SYNC_REQUIRE_TRADE_SPECIFIER and not trade_specifier:
        return {"ok": False, "action": "SKIP", "reason": "missing_trade_specifier_for_closed_trade_sync"}
    if not trade_specifier:
        return {"ok": False, "action": "SKIP", "reason": "closed_trade_sync_requires_trade_specifier"}

    result = get_oanda_trade_details(trade_specifier)
    if not result.get("ok"):
        return {
            "ok": False,
            "action": "SYNC_ERROR",
            "reason": f"oanda_trade_details_failed:{result.get('error')}",
            "broker_response": result,
        }

    data = result.get("data") or {}
    trade = data.get("trade") or {}
    state = str(trade.get("state") or "").upper()
    if state == "CLOSED":
        return register_closed_trade_from_oanda(
            tracking_key=tracking_key,
            meta=meta,
            trade_specifier=trade_specifier,
            trade=trade,
            broker_response=result,
        )
    return {
        "ok": True,
        "action": "OPEN_STILL_ACTIVE",
        "reason": f"oanda_trade_state:{state or 'UNKNOWN'}",
        "trade_specifier": trade_specifier,
    }


def modify_trade_stop_loss(trade_specifier: str, instrument: str, stop_loss_price: float | str) -> Dict[str, Any]:
    if not trade_specifier:
        return {"ok": False, "error": "Missing trade_specifier"}
    formatted = format_oanda_price(float(stop_loss_price), instrument)
    body = {
        "stopLoss": {
            "timeInForce": "GTC",
            "price": formatted,
        }
    }
    return oanda_request("PUT", f"/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_specifier}/orders", body, timeout=30)


def favorable_move_pips(side: str, entry_price: float, live_bid: float, live_ask: float, instrument: str) -> float:
    pip = instrument_pip_size(instrument)
    if pip <= 0:
        return 0.0
    if side == "BUY":
        return (live_bid - entry_price) / pip
    if side == "SELL":
        return (entry_price - live_ask) / pip
    return 0.0


def initial_risk_pips_from_meta(meta: Dict[str, Any]) -> float:
    instrument = str(meta.get("instrument") or "")
    entry = safe_float(meta.get("entry_price"), 0.0)
    sl = safe_float(meta.get("original_sl_price"), safe_float(meta.get("sl_price"), 0.0))
    pip = instrument_pip_size(instrument) if instrument else 0.0
    if entry <= 0 or sl <= 0 or pip <= 0:
        return 0.0
    return abs(entry - sl) / pip


def stop_is_improvement(side: str, previous_sl: float, requested_sl: float, minimum_improvement_price: float) -> bool:
    if previous_sl <= 0:
        return True
    if side == "BUY":
        return requested_sl > previous_sl + minimum_improvement_price
    if side == "SELL":
        return requested_sl < previous_sl - minimum_improvement_price
    return False


def requested_stop_is_valid_against_live_price(side: str, requested_sl: float, live_bid: float, live_ask: float, instrument: str) -> Tuple[bool, str]:
    pip = instrument_pip_size(instrument)
    buffer_price = STOP_UPDATE_LIVE_BUFFER_PIPS * pip
    if side == "BUY":
        if requested_sl >= live_bid - buffer_price:
            return False, f"buy_stop_not_below_live_bid:{requested_sl}>={live_bid - buffer_price}"
    elif side == "SELL":
        if requested_sl <= live_ask + buffer_price:
            return False, f"sell_stop_not_above_live_ask:{requested_sl}<={live_ask + buffer_price}"
    else:
        return False, "invalid_side"
    return True, "stop_price_valid"



def _recursive_find_numeric(obj: Any, keys: set[str]) -> Optional[float]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in keys:
                found = safe_float(value, np.nan)
                if np.isfinite(found):
                    return float(found)
        for value in obj.values():
            found = _recursive_find_numeric(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _recursive_find_numeric(item, keys)
            if found is not None:
                return found
    return None


def extract_close_fill_details(close_result: Dict[str, Any]) -> Dict[str, Any]:
    data = close_result.get("data") if isinstance(close_result, dict) else {}
    if not isinstance(data, dict):
        data = {}
    fill = data.get("orderFillTransaction") or data.get("transaction") or {}
    if not isinstance(fill, dict):
        fill = {}
    pnl = _recursive_find_numeric(data, {"pl", "realizedPL"})
    price = safe_float(fill.get("price"), 0.0)
    tx_id = fill.get("id")
    reason = fill.get("reason")
    return {
        "pnl": pnl,
        "close_price": price,
        "transaction_id": tx_id,
        "transaction_reason": reason,
        "fill_transaction": fill,
    }



def close_trade_for_adverse_exit(
    tracking_key: str,
    meta: Dict[str, Any],
    trade_specifier: str,
    quote: Dict[str, Any],
    favorable_pips_value: float,
    peak_favorable_pips: float,
    adverse_pips: float,
    initial_risk_pips: float,
    current_r: float,
    peak_r: float,
    minutes_open: float,
    reason: str,
) -> Dict[str, Any]:
    """Close a weak trade that starts losing and stays negative after the grace period."""
    instrument = str(meta.get("instrument") or "")
    side = normalize_side(meta.get("side"))
    broker_result = close_oanda_trade_by_specifier(trade_specifier)
    success = bool(broker_result.get("ok"))
    fill_details = extract_close_fill_details(broker_result) if success else {}
    close_price = safe_float(fill_details.get("close_price"), 0.0)
    pnl = fill_details.get("pnl")

    event = {
        "ts": utc_ts(),
        "tracking_key": str(tracking_key),
        "instrument": instrument,
        "side": side,
        "action": "ADVERSE_EXIT_EARLY_LOSS",
        "trade_specifier": trade_specifier,
        "entry_price": safe_float(meta.get("entry_price"), 0.0),
        "live_bid": safe_float(quote.get("bid"), 0.0),
        "live_ask": safe_float(quote.get("ask"), 0.0),
        "favorable_pips": favorable_pips_value,
        "adverse_pips": adverse_pips,
        "peak_favorable_pips": peak_favorable_pips,
        "initial_risk_pips": initial_risk_pips,
        "current_r": current_r,
        "peak_r": peak_r,
        "minutes_open": minutes_open,
        "previous_sl_price": safe_float(meta.get("sl_price"), 0.0),
        "requested_sl_price": None,
        "updated_sl_price": safe_float(meta.get("sl_price"), 0.0),
        "success": success,
        "reason": reason if success else f"{reason}; broker_error={broker_result.get('error') or broker_result.get('status_code')}",
        "broker_response": {
            "close_result": broker_result,
            "close_fill_details": fill_details,
        },
    }
    write_trade_management_event(event)

    if success:
        row = {
            "instrument": instrument,
            "side": side,
            "units_signed": meta.get("units_signed"),
            "entry_price": meta.get("entry_price"),
            "sl_price": meta.get("sl_price"),
            "tp_price": meta.get("tp_price"),
            "status": "MANUAL",
            "pnl": pnl,
            "order_id": meta.get("order_id"),
            "reason": (
                f"ADVERSE_EXIT_EARLY_LOSS; {reason}; close_price={close_price or 'unknown'}; "
                f"adverse_pips={adverse_pips:.3f}; current_favorable_pips={favorable_pips_value:.3f}; "
                f"peak_favorable_pips={peak_favorable_pips:.3f}; current_r={current_r:.3f}; "
                f"peak_r={peak_r:.3f}; minutes_open={minutes_open:.2f}"
            ),
            "pair_score": meta.get("pair_score"),
            "ts": utc_ts(),
            "tracking_key": str(tracking_key),
            "broker_trade_id": meta.get("broker_trade_id"),
            "broker_order_id": meta.get("broker_order_id"),
            "client_trade_id": meta.get("client_trade_id"),
        }
        write_trade_row(row)
        note_trade_closed(tracking_key)
        _open_trade_meta.pop(str(tracking_key), None)

    return {"ok": success, "action": "ADVERSE_EXIT_EARLY_LOSS", "event": event, "broker_result": broker_result, "close_fill_details": fill_details}


def close_trade_for_reversal_profit_lock(
    tracking_key: str,
    meta: Dict[str, Any],
    trade_specifier: str,
    quote: Dict[str, Any],
    favorable_pips_value: float,
    peak_favorable_pips: float,
    giveback_pips: float,
    initial_risk_pips: float,
    current_r: float,
    peak_r: float,
    giveback_r: float,
    reason: str,
) -> Dict[str, Any]:
    """Close a still-profitable trade when the favorable move starts reversing."""
    instrument = str(meta.get("instrument") or "")
    side = normalize_side(meta.get("side"))
    broker_result = close_oanda_trade_by_specifier(trade_specifier)
    success = bool(broker_result.get("ok"))
    fill_details = extract_close_fill_details(broker_result) if success else {}
    close_price = safe_float(fill_details.get("close_price"), 0.0)
    pnl = fill_details.get("pnl")

    event = {
        "ts": utc_ts(),
        "tracking_key": str(tracking_key),
        "instrument": instrument,
        "side": side,
        "action": "REVERSAL_PROFIT_EXIT",
        "trade_specifier": trade_specifier,
        "entry_price": safe_float(meta.get("entry_price"), 0.0),
        "live_bid": safe_float(quote.get("bid"), 0.0),
        "live_ask": safe_float(quote.get("ask"), 0.0),
        "favorable_pips": favorable_pips_value,
        "initial_risk_pips": initial_risk_pips,
        "current_r": current_r,
        "previous_sl_price": safe_float(meta.get("sl_price"), 0.0),
        "requested_sl_price": None,
        "updated_sl_price": safe_float(meta.get("sl_price"), 0.0),
        "success": success,
        "reason": reason if success else f"{reason}; broker_error={broker_result.get('error') or broker_result.get('status_code')}",
        "broker_response": {
            "close_result": broker_result,
            "close_fill_details": fill_details,
            "peak_favorable_pips": peak_favorable_pips,
            "giveback_pips": giveback_pips,
            "peak_r": peak_r,
            "giveback_r": giveback_r,
        },
    }
    write_trade_management_event(event)

    if success:
        row = {
            "instrument": instrument,
            "side": side,
            "units_signed": meta.get("units_signed"),
            "entry_price": meta.get("entry_price"),
            "sl_price": meta.get("sl_price"),
            "tp_price": meta.get("tp_price"),
            "status": "MANUAL",
            "pnl": pnl,
            "order_id": meta.get("order_id"),
            "reason": (
                f"REVERSAL_PROFIT_EXIT; {reason}; close_price={close_price or 'unknown'}; "
                f"peak_favorable_pips={peak_favorable_pips:.3f}; current_favorable_pips={favorable_pips_value:.3f}; "
                f"giveback_pips={giveback_pips:.3f}; peak_r={peak_r:.3f}; current_r={current_r:.3f}; "
                f"giveback_r={giveback_r:.3f}"
            ),
            "pair_score": meta.get("pair_score"),
            "ts": utc_ts(),
            "tracking_key": str(tracking_key),
            "broker_trade_id": meta.get("broker_trade_id"),
            "broker_order_id": meta.get("broker_order_id"),
            "client_trade_id": meta.get("client_trade_id"),
        }
        write_trade_row(row)
        note_trade_closed(tracking_key)
        _open_trade_meta.pop(str(tracking_key), None)

    return {"ok": success, "action": "REVERSAL_PROFIT_EXIT", "event": event, "broker_result": broker_result, "close_fill_details": fill_details}


def update_stop_for_open_trade(
    tracking_key: str,
    meta: Dict[str, Any],
    action: str,
    trade_specifier: str,
    requested_sl: float,
    quote: Dict[str, Any],
    favorable_pips_value: float,
    initial_risk_pips: float,
    current_r: float,
    reason: str,
) -> Dict[str, Any]:
    instrument = str(meta.get("instrument") or "")
    previous_sl = safe_float(meta.get("sl_price"), 0.0)
    broker_result = modify_trade_stop_loss(trade_specifier, instrument, requested_sl)
    success = bool(broker_result.get("ok"))
    formatted_requested = safe_float(format_oanda_price(requested_sl, instrument), requested_sl)
    updated_sl = formatted_requested if success else previous_sl
    event = {
        "ts": utc_ts(),
        "tracking_key": tracking_key,
        "instrument": instrument,
        "side": meta.get("side"),
        "action": action,
        "trade_specifier": trade_specifier,
        "entry_price": safe_float(meta.get("entry_price"), 0.0),
        "live_bid": safe_float(quote.get("bid"), 0.0),
        "live_ask": safe_float(quote.get("ask"), 0.0),
        "favorable_pips": favorable_pips_value,
        "initial_risk_pips": initial_risk_pips,
        "current_r": current_r,
        "previous_sl_price": previous_sl,
        "requested_sl_price": formatted_requested,
        "updated_sl_price": updated_sl,
        "success": success,
        "reason": reason if success else f"{reason}; broker_error={broker_result.get('error') or broker_result.get('status_code')}",
        "broker_response": broker_result,
    }
    write_trade_management_event(event)
    if success:
        meta["sl_price"] = updated_sl
        meta["last_stop_update_ts"] = event["ts"]
        if action == "BREAKEVEN":
            meta["breakeven_done"] = True
        if action == "TRAILING_STOP":
            meta["trailing_active"] = True
    return {"ok": success, "event": event, "broker_result": broker_result}


def manage_single_open_trade(tracking_key: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    if not broker_ready():
        return {"ok": False, "action": "SKIP", "reason": "broker_not_ready"}
    instrument = str(meta.get("instrument") or "").upper()
    side = normalize_side(meta.get("side"))
    entry_price = safe_float(meta.get("entry_price"), 0.0)
    if not instrument or side not in {"BUY", "SELL"} or entry_price <= 0:
        return {"ok": False, "action": "SKIP", "reason": "invalid_open_trade_meta"}

    trade_specifier = trade_specifier_from_meta(meta)
    if PROFIT_PROTECTION_REQUIRE_TRADE_SPECIFIER and not trade_specifier:
        return {"ok": False, "action": "SKIP", "reason": "missing_trade_specifier"}
    if not trade_specifier:
        return {"ok": False, "action": "SKIP", "reason": "missing_trade_specifier_optional_but_required_for_stop_update"}

    quote = fetch_live_oanda_quote(instrument)
    if not quote.get("ok"):
        return {"ok": False, "action": "SKIP", "reason": f"live_quote_unavailable:{quote.get('error')}"}

    live_bid = safe_float(quote.get("bid"), 0.0)
    live_ask = safe_float(quote.get("ask"), 0.0)
    favorable_pips_value = favorable_move_pips(side, entry_price, live_bid, live_ask, instrument)
    initial_risk_pips = initial_risk_pips_from_meta(meta)
    if initial_risk_pips <= 0:
        return {"ok": False, "action": "SKIP", "reason": "initial_risk_pips_missing_or_zero"}

    current_r = favorable_pips_value / initial_risk_pips
    meta["last_favorable_pips"] = favorable_pips_value
    meta["last_current_r"] = current_r

    previous_peak = safe_float(meta.get("peak_favorable_pips"), favorable_pips_value)
    peak_favorable_pips = max(previous_peak, favorable_pips_value)
    meta["peak_favorable_pips"] = peak_favorable_pips
    peak_r = peak_favorable_pips / initial_risk_pips if initial_risk_pips > 0 else 0.0
    giveback_pips = max(0.0, peak_favorable_pips - favorable_pips_value)
    giveback_r = giveback_pips / initial_risk_pips if initial_risk_pips > 0 else 0.0
    meta["peak_current_r"] = peak_r
    meta["last_giveback_pips"] = giveback_pips
    meta["last_giveback_r"] = giveback_r

    # 0A) Early adverse-start exit. This closes a trade that starts losing and stays weak
    # after a short grace period, instead of waiting for the full stop loss.
    if ADVERSE_EXIT_ENABLED:
        opened_at = meta.get("opened_at_dt")
        minutes_open = 0.0
        if opened_at is not None:
            try:
                minutes_open = max(0.0, (now_utc() - opened_at).total_seconds() / 60.0)
            except Exception:
                minutes_open = 0.0
        adverse_pips = max(0.0, -favorable_pips_value)
        min_loss_needed_pips = max(ADVERSE_EXIT_MIN_LOSS_PIPS, ADVERSE_EXIT_MIN_LOSS_R * initial_risk_pips)
        max_allowed_peak_pips = max(ADVERSE_EXIT_MAX_PEAK_PROFIT_PIPS, ADVERSE_EXIT_MAX_PEAK_PROFIT_R * initial_risk_pips)
        waited_long_enough = minutes_open >= ADVERSE_EXIT_AFTER_MINUTES
        loss_large_enough = adverse_pips >= min_loss_needed_pips
        no_meaningful_recovery = (peak_favorable_pips <= max_allowed_peak_pips) if ADVERSE_EXIT_REQUIRE_NO_RECOVERY else True
        if waited_long_enough and loss_large_enough and no_meaningful_recovery:
            return close_trade_for_adverse_exit(
                tracking_key=tracking_key,
                meta=meta,
                trade_specifier=trade_specifier,
                quote=quote,
                favorable_pips_value=favorable_pips_value,
                peak_favorable_pips=peak_favorable_pips,
                adverse_pips=adverse_pips,
                initial_risk_pips=initial_risk_pips,
                current_r=current_r,
                peak_r=peak_r,
                minutes_open=minutes_open,
                reason=(
                    f"adverse_exit_triggered:minutes_open={minutes_open:.2f}>={ADVERSE_EXIT_AFTER_MINUTES};"
                    f"loss={adverse_pips:.3f}pips>={min_loss_needed_pips:.3f}pips;"
                    f"peak_profit={peak_favorable_pips:.3f}pips<={max_allowed_peak_pips:.3f}pips;"
                    f"current_r={current_r:.3f}"
                ),
            )

    # 0B) Early reversal profit lock. This closes at market while the trade is still green
    # when it has reached a small profit and has started giving back enough of that profit.
    if REVERSAL_EXIT_ENABLED:
        min_profit_needed_pips = max(REVERSAL_EXIT_MIN_PROFIT_PIPS, REVERSAL_EXIT_MIN_PROFIT_R * initial_risk_pips)
        giveback_needed_pips = max(REVERSAL_EXIT_GIVEBACK_PIPS, REVERSAL_EXIT_GIVEBACK_R * initial_risk_pips)
        reversal_profit_reached = peak_favorable_pips >= min_profit_needed_pips
        reversal_started = giveback_pips >= giveback_needed_pips
        still_profitable = favorable_pips_value >= REVERSAL_EXIT_MIN_CURRENT_PROFIT_PIPS
        if reversal_profit_reached and reversal_started and still_profitable:
            return close_trade_for_reversal_profit_lock(
                tracking_key=tracking_key,
                meta=meta,
                trade_specifier=trade_specifier,
                quote=quote,
                favorable_pips_value=favorable_pips_value,
                peak_favorable_pips=peak_favorable_pips,
                giveback_pips=giveback_pips,
                initial_risk_pips=initial_risk_pips,
                current_r=current_r,
                peak_r=peak_r,
                giveback_r=giveback_r,
                reason=(
                    f"reversal_profit_lock_triggered:peak={peak_favorable_pips:.3f}pips>={min_profit_needed_pips:.3f}pips;"
                    f"giveback={giveback_pips:.3f}pips>={giveback_needed_pips:.3f}pips;"
                    f"current_profit={favorable_pips_value:.3f}pips>={REVERSAL_EXIT_MIN_CURRENT_PROFIT_PIPS:.3f}pips"
                ),
            )

    pip = instrument_pip_size(instrument)
    previous_sl = safe_float(meta.get("sl_price"), 0.0)
    min_improvement_price = TRAILING_MIN_IMPROVEMENT_PIPS * pip

    # 1) Move stop to breakeven + a small buffer once the trade reaches the configured R threshold.
    if current_r >= BREAKEVEN_TRIGGER_R and not bool(meta.get("breakeven_done", False)):
        if side == "BUY":
            requested_sl = entry_price + BREAKEVEN_BUFFER_PIPS * pip
        else:
            requested_sl = entry_price - BREAKEVEN_BUFFER_PIPS * pip
        valid, valid_reason = requested_stop_is_valid_against_live_price(side, requested_sl, live_bid, live_ask, instrument)
        if valid and stop_is_improvement(side, previous_sl, requested_sl, min_improvement_price):
            return update_stop_for_open_trade(
                tracking_key=tracking_key,
                meta=meta,
                action="BREAKEVEN",
                trade_specifier=trade_specifier,
                requested_sl=requested_sl,
                quote=quote,
                favorable_pips_value=favorable_pips_value,
                initial_risk_pips=initial_risk_pips,
                current_r=current_r,
                reason=f"breakeven_triggered:{current_r:.3f}R>={BREAKEVEN_TRIGGER_R:.3f}R",
            )
        return {"ok": False, "action": "SKIP", "reason": f"breakeven_not_valid_or_not_improvement:{valid_reason}"}

    # 2) Trail the stop once the trade reaches the configured R threshold.
    if current_r >= TRAILING_TRIGGER_R:
        trailing_distance_pips = max(TRAILING_DISTANCE_R * initial_risk_pips, TRAILING_MIN_IMPROVEMENT_PIPS)
        if side == "BUY":
            requested_sl = live_bid - trailing_distance_pips * pip
            floor_sl = entry_price + BREAKEVEN_BUFFER_PIPS * pip
            requested_sl = max(requested_sl, floor_sl)
        else:
            requested_sl = live_ask + trailing_distance_pips * pip
            ceiling_sl = entry_price - BREAKEVEN_BUFFER_PIPS * pip
            requested_sl = min(requested_sl, ceiling_sl)
        valid, valid_reason = requested_stop_is_valid_against_live_price(side, requested_sl, live_bid, live_ask, instrument)
        if valid and stop_is_improvement(side, previous_sl, requested_sl, min_improvement_price):
            return update_stop_for_open_trade(
                tracking_key=tracking_key,
                meta=meta,
                action="TRAILING_STOP",
                trade_specifier=trade_specifier,
                requested_sl=requested_sl,
                quote=quote,
                favorable_pips_value=favorable_pips_value,
                initial_risk_pips=initial_risk_pips,
                current_r=current_r,
                reason=f"trailing_triggered:{current_r:.3f}R>={TRAILING_TRIGGER_R:.3f}R;distance={trailing_distance_pips:.3f}pips",
            )
        return {"ok": False, "action": "SKIP", "reason": f"trailing_not_valid_or_not_improvement:{valid_reason}"}

    return {"ok": True, "action": "NO_ACTION", "reason": f"profit_protection_not_triggered:{current_r:.3f}R"}


def profit_protection_worker() -> None:
    while True:
        try:
            if not PROFIT_PROTECTION_ENABLED:
                time.sleep(OPEN_TRADE_MANAGER_CHECK_SECONDS)
                continue
            # Critical: do not hit OANDA unless the server is currently tracking open trades.
            if not _open_trade_meta:
                time.sleep(OPEN_TRADE_MANAGER_CHECK_SECONDS)
                continue
            if not broker_ready():
                time.sleep(OPEN_TRADE_MANAGER_CHECK_SECONDS)
                continue
            for tracking_key, meta in list(_open_trade_meta.items()):
                try:
                    sync_result = sync_single_tracked_trade_close(str(tracking_key), meta)
                    if sync_result.get("action") == "CLOSED_TRADE_SYNC":
                        continue
                    manage_single_open_trade(str(tracking_key), meta)
                except Exception as inner_exc:
                    write_trade_management_event(
                        {
                            "ts": utc_ts(),
                            "tracking_key": str(tracking_key),
                            "instrument": meta.get("instrument"),
                            "side": meta.get("side"),
                            "action": "ERROR",
                            "trade_specifier": trade_specifier_from_meta(meta),
                            "entry_price": meta.get("entry_price"),
                            "live_bid": None,
                            "live_ask": None,
                            "favorable_pips": None,
                            "initial_risk_pips": initial_risk_pips_from_meta(meta),
                            "current_r": None,
                            "previous_sl_price": meta.get("sl_price"),
                            "requested_sl_price": None,
                            "updated_sl_price": meta.get("sl_price"),
                            "success": False,
                            "reason": f"profit_protection_inner_exception:{repr(inner_exc)}",
                            "broker_response": None,
                        }
                    )
        except Exception as exc:
            print(f"PROFIT_PROTECTION_WORKER_EXCEPTION: {exc}")
        time.sleep(OPEN_TRADE_MANAGER_CHECK_SECONDS)


def closed_trade_sync_worker() -> None:
    """Background reconciliation for TP/SL/manual broker closures.

    It only calls OANDA while the server already has open trades tracked locally.
    This makes closed-trade registration independent of whether profit protection
    is enabled.
    """
    while True:
        try:
            if not CLOSED_TRADE_SYNC_ENABLED:
                time.sleep(CLOSED_TRADE_SYNC_CHECK_SECONDS)
                continue
            if not _open_trade_meta:
                time.sleep(CLOSED_TRADE_SYNC_CHECK_SECONDS)
                continue
            if not broker_ready():
                time.sleep(CLOSED_TRADE_SYNC_CHECK_SECONDS)
                continue
            for tracking_key, meta in list(_open_trade_meta.items()):
                try:
                    sync_single_tracked_trade_close(str(tracking_key), meta)
                except Exception as inner_exc:
                    write_trade_management_event(
                        {
                            "ts": utc_ts(),
                            "tracking_key": str(tracking_key),
                            "instrument": meta.get("instrument"),
                            "side": meta.get("side"),
                            "action": "CLOSED_TRADE_SYNC_ERROR",
                            "trade_specifier": trade_specifier_from_meta(meta),
                            "entry_price": meta.get("entry_price"),
                            "live_bid": None,
                            "live_ask": None,
                            "favorable_pips": meta.get("last_favorable_pips"),
                            "initial_risk_pips": initial_risk_pips_from_meta(meta),
                            "current_r": meta.get("last_current_r"),
                            "previous_sl_price": meta.get("sl_price"),
                            "requested_sl_price": None,
                            "updated_sl_price": meta.get("sl_price"),
                            "success": False,
                            "reason": f"closed_trade_sync_inner_exception:{repr(inner_exc)}",
                            "broker_response": None,
                        }
                    )
        except Exception as exc:
            print(f"CLOSED_TRADE_SYNC_WORKER_EXCEPTION: {exc}")
        time.sleep(CLOSED_TRADE_SYNC_CHECK_SECONDS)


def auto_close_worker() -> None:
    while True:
        try:
            if not AUTO_CLOSE_ENABLED:
                time.sleep(AUTO_CLOSE_CHECK_SECONDS)
                continue
            if not _open_trade_meta:
                time.sleep(AUTO_CLOSE_CHECK_SECONDS)
                continue
            if not broker_ready():
                time.sleep(AUTO_CLOSE_CHECK_SECONDS)
                continue
            now = now_utc()
            for tracking_key, meta in list(_open_trade_meta.items()):
                opened_at = meta.get("opened_at_dt")
                if opened_at is None:
                    continue
                if (now - opened_at).total_seconds() / 60.0 < MAX_HOLD_MINUTES:
                    continue
                spec = meta.get("broker_trade_id") or (f"@{meta.get('client_trade_id')}" if meta.get("client_trade_id") else None)
                if spec:
                    result = close_oanda_trade_by_specifier(spec)
                elif AUTO_CLOSE_ALLOW_POSITION_FALLBACK:
                    result = close_oanda_position_side(meta["instrument"], meta["side"])
                else:
                    result = {"ok": False, "error": "no_trade_specifier_and_position_fallback_disabled"}
                if result.get("ok"):
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
        except Exception as exc:
            print(f"AUTO_CLOSE_WORKER_EXCEPTION: {exc}")
        time.sleep(AUTO_CLOSE_CHECK_SECONDS)

# ============================================================
# MODEL LOADING / SCORING
# ============================================================
def extract_summary_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    winner = metadata.get("winner_summary") or {}
    return {
        "selected_model_type": metadata.get("selected_model_type", "unknown"),
        "mean_auc": safe_float(winner.get("mean_auc"), 0.0),
        "mean_brier": safe_float(winner.get("mean_brier"), 0.0),
        "mean_accuracy": safe_float(winner.get("mean_accuracy"), 0.0),
        "mean_precision_at_gate": safe_float(winner.get("mean_precision_at_gate"), 0.0),
        "mean_coverage_at_gate": safe_float(winner.get("mean_coverage_at_gate"), 0.0),
        "total_signals_at_gate": safe_int(winner.get("total_signals_at_gate"), 0),
        "horizon_bars": safe_int(metadata.get("horizon_bars"), 8),
        "horizon_minutes": safe_int(metadata.get("horizon_minutes"), 120),
        "target_pips": safe_float(metadata.get("target_pips"), 8.0),
        "rows": safe_int(metadata.get("rows"), 0),
        "signal_success_rate": safe_float(metadata.get("signal_success_rate"), 0.0),
    }


def pair_passes_static_training_filter(pair6: str, summary: Dict[str, Any]) -> Tuple[bool, str]:
    if not STRICT_MODEL_FILTER_ENABLED:
        return True, "strict_filter_disabled"
    reasons = []
    if safe_float(summary.get("mean_auc"), 0.0) < LIVE_MIN_AUC:
        reasons.append("auc_too_low")
    if safe_float(summary.get("mean_precision_at_gate"), 0.0) < LIVE_MIN_PRECISION:
        reasons.append("precision_too_low")
    if safe_int(summary.get("total_signals_at_gate"), 0) < LIVE_MIN_TRADES_AT_GATE:
        reasons.append("signals_too_low")
    if pair6 not in PRIMARY_LIVE_PAIRS:
        reasons.append("pair_not_in_primary_live_pairs")
    return (False, "; ".join(reasons)) if reasons else (True, "static_filter_passed")


def load_signal_approval_bundle_file(path: Path) -> Optional[Dict[str, Any]]:
    name = path.name
    if not name.endswith("_M15_signal_approval_bundle.joblib"):
        return None
    instrument = name.replace("_M15_signal_approval_bundle.joblib", "").upper()
    pair6 = instrument.replace("_", "")
    if pair6 not in PAIR_MAP:
        print(f"WARNING: ignoring unknown pair bundle: {path}")
        return None
    try:
        raw = joblib.load(path)
    except Exception as exc:
        print(f"WARNING: could not load signal approval bundle {path}: {exc}")
        return None
    architecture = str(raw.get("architecture") or raw.get("metadata", {}).get("architecture") or "")
    if architecture != "signal_conditioned_hint_approval":
        print(f"WARNING: skipping non-signal-conditioned bundle: {path}")
        return None
    model = raw.get("model")
    metadata = raw.get("metadata", {})
    selected_model_type = str(raw.get("selected_model_type") or metadata.get("selected_model_type") or "unknown")
    if model is None:
        print(f"WARNING: missing model object in {path}")
        return None
    candidate_models = raw.get("candidate_models") or {selected_model_type: model}
    candidate_metrics = raw.get("candidate_metrics") or metadata.get("candidate_metrics") or {}
    if selected_model_type not in candidate_models:
        candidate_models[selected_model_type] = model
    summary = extract_summary_from_metadata(metadata)
    static_ok, static_reason = pair_passes_static_training_filter(pair6, summary)
    feature_order = metadata.get("feature_cols") or FEATURE_COLS
    feature_order = [feature for feature in feature_order if feature in FEATURE_COLS]
    if not feature_order:
        feature_order = FEATURE_COLS
    gate = safe_float(metadata.get("selection_conf_gate"), APPROVAL_GATE)
    margin_gate = safe_float(metadata.get("selection_margin_gate"), APPROVAL_MARGIN_GATE)
    override = read_active_override_from_db(pair6)
    active_override = None
    active_previous = None
    active_reason = None
    active_updated_at = None
    if override:
        override_model = str(override.get("active_model") or "").strip()
        if override_model in candidate_models:
            active_override = override_model
            active_previous = override.get("previous_model")
            active_reason = override.get("reason")
            active_updated_at = override.get("updated_at")
            model = candidate_models[override_model]
            selected_model_type = override_model
    active_metric = candidate_metrics.get(selected_model_type, {})
    active_auc = safe_float(active_metric.get("auc"), summary["mean_auc"])
    active_precision = safe_float(active_metric.get("precision_at_gate"), summary["mean_precision_at_gate"])
    active_trades = safe_int(active_metric.get("trades_at_gate"), summary["total_signals_at_gate"])
    return {
        "pair6": pair6,
        "instrument": pair_to_instrument(pair6),
        "model": model,
        "model_type": selected_model_type,
        "saved_best_model_type": str(raw.get("selected_model_type") or metadata.get("selected_model_type") or "unknown"),
        "active_model_override": active_override,
        "active_override_previous_model": active_previous,
        "active_override_reason": active_reason,
        "active_override_updated_at": active_updated_at,
        "candidate_models": candidate_models,
        "candidate_metrics": candidate_metrics,
        "feature_order": feature_order,
        "metadata": metadata,
        "summary": summary,
        "avg_auc": active_auc,
        "precision_at_gate": active_precision,
        "trades_at_gate": active_trades,
        "static_filter_passed": static_ok,
        "static_filter_reason": static_reason,
        "gate": gate,
        "margin_gate": margin_gate,
        "labeling": {"sl_atr": DEFAULT_SL_ATR, "tp_atr": DEFAULT_TP_ATR},
        "model_version": f"{pair6}:M15:signal_approval:{selected_model_type}",
        "_bundle_path": str(path),
    }


def load_bundles(models_dir: str) -> Dict[str, Dict[str, Any]]:
    bundles: Dict[str, Dict[str, Any]] = {}
    root = Path(models_dir)
    if not root.exists():
        print(f"WARNING: MODELS_DIR does not exist: {models_dir}")
        return bundles
    for path in sorted(root.glob("*_M15_signal_approval_bundle.joblib")):
        bundle = load_signal_approval_bundle_file(path)
        if bundle:
            bundles[bundle["pair6"]] = bundle
    print(f"Loaded {len(bundles)} M15 signal-approval bundles from {models_dir}")
    return bundles


def predict_approval_probability(model: Any, X: pd.DataFrame) -> float:
    proba = model.predict_proba(X)[0]
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


def approval_margin(probability: float) -> float:
    return max(0.0, probability - 0.5)


def primary_metric_is_good_for_order(bundle: Dict[str, Any]) -> Tuple[bool, str]:
    auc = safe_float(bundle.get("avg_auc"), 0.0)
    precision = safe_float(bundle.get("precision_at_gate"), 0.0)
    trades = safe_int(bundle.get("trades_at_gate"), 0)
    reasons = []
    if auc < PRIMARY_MIN_AUC_FOR_ORDER:
        reasons.append(f"primary_auc_too_low:{auc:.4f}<{PRIMARY_MIN_AUC_FOR_ORDER:.4f}")
    if precision < PRIMARY_MIN_PRECISION_FOR_ORDER:
        reasons.append(f"primary_precision_too_low:{precision:.4f}<{PRIMARY_MIN_PRECISION_FOR_ORDER:.4f}")
    if trades < PRIMARY_MIN_TRADES_AT_GATE_FOR_ORDER:
        reasons.append(f"primary_trades_too_low:{trades}<{PRIMARY_MIN_TRADES_AT_GATE_FOR_ORDER}")
    return (False, "; ".join(reasons)) if reasons else (True, "primary_training_quality_passed")


def candidate_metric_good(metric: Dict[str, Any], use_fallback_thresholds: bool) -> Tuple[bool, str]:
    auc_min = FALLBACK_MIN_AUC if use_fallback_thresholds else SWITCH_MIN_AUC
    precision_min = FALLBACK_MIN_PRECISION if use_fallback_thresholds else SWITCH_MIN_PRECISION
    trades_min = FALLBACK_MIN_TRADES_AT_GATE if use_fallback_thresholds else SWITCH_MIN_TRADES_AT_GATE
    auc = safe_float(metric.get("auc"), 0.0)
    precision = safe_float(metric.get("precision_at_gate"), 0.0)
    trades = safe_int(metric.get("trades_at_gate"), 0)
    tradable = bool(metric.get("tradable", False))
    reasons = []
    if not tradable:
        reasons.append("candidate_not_tradable")
    if auc < auc_min:
        reasons.append(f"auc_too_low:{auc:.4f}<{auc_min:.4f}")
    if precision < precision_min:
        reasons.append(f"precision_too_low:{precision:.4f}<{precision_min:.4f}")
    if trades < trades_min:
        reasons.append(f"trades_too_low:{trades}<{trades_min}")
    return (False, "; ".join(reasons)) if reasons else (True, "candidate_quality_passed")


def evaluate_candidate_models_for_fallback(
    bundle: Dict[str, Any],
    X: pd.DataFrame,
    primary_model_type: str,
    primary_conf_gate: float,
    primary_margin_gate: float,
) -> Dict[str, Any]:
    if not FALLBACK_MODE_ENABLED:
        return {"fallback_allowed": False, "fallback_used": False, "fallback_reason": "fallback_mode_disabled", "candidate_votes": {}}
    candidate_models = bundle.get("candidate_models") or {}
    candidate_metrics = bundle.get("candidate_metrics") or {}
    fallback_conf_gate = primary_conf_gate + FALLBACK_CONF_EDGE
    fallback_margin_gate = primary_margin_gate + FALLBACK_MARGIN_EDGE
    candidate_votes: Dict[str, Dict[str, Any]] = {}
    best_candidate = None
    for model_name, model in candidate_models.items():
        if model_name == primary_model_type:
            continue
        metric = candidate_metrics.get(model_name, {})
        quality_ok, quality_reason = candidate_metric_good(metric, use_fallback_thresholds=True)
        try:
            approval_probability = predict_approval_probability(model, X)
        except Exception as exc:
            candidate_votes[model_name] = {"ok": False, "reason": f"prediction_error:{repr(exc)}"}
            continue
        approval_probability = max(0.0, min(1.0, float(approval_probability)))
        margin = approval_margin(approval_probability)
        gate_ok = approval_probability >= fallback_conf_gate and margin >= fallback_margin_gate
        ok = bool(quality_ok and gate_ok)
        vote = {
            "ok": ok,
            "approval_probability": approval_probability,
            "margin": margin,
            "quality_ok": quality_ok,
            "quality_reason": quality_reason,
            "gate_ok": gate_ok,
            "fallback_conf_gate": fallback_conf_gate,
            "fallback_margin_gate": fallback_margin_gate,
            "auc": safe_float(metric.get("auc"), 0.0),
            "precision_at_gate": safe_float(metric.get("precision_at_gate"), 0.0),
            "trades_at_gate": safe_int(metric.get("trades_at_gate"), 0),
        }
        candidate_votes[model_name] = vote
        if ok:
            record = {"model_name": model_name, **vote, "metric": metric}
            if best_candidate is None:
                best_candidate = record
            elif record["approval_probability"] > best_candidate["approval_probability"]:
                best_candidate = record
            elif math.isclose(record["approval_probability"], best_candidate["approval_probability"]) and record["precision_at_gate"] > best_candidate["precision_at_gate"]:
                best_candidate = record
    if best_candidate is None:
        return {
            "fallback_allowed": False,
            "fallback_used": False,
            "fallback_reason": "no_candidate_passed_signal_approval_fallback_rules",
            "candidate_votes": candidate_votes,
        }
    return {
        "fallback_allowed": True,
        "fallback_used": True,
        "fallback_reason": "signal_approval_fallback_candidate_passed",
        "fallback_model": best_candidate["model_name"],
        "fallback_approval_probability": best_candidate["approval_probability"],
        "fallback_margin": best_candidate["margin"],
        "candidate_votes": candidate_votes,
    }


def log_model_signal_events(
    pair6: str,
    instrument: str,
    signal_id: str,
    hint_side: str,
    primary_model_type: str,
    primary_probability: float,
    primary_margin: float,
    primary_would_order: bool,
    final_order_allowed: bool,
    decision_source: str,
    conf_gate: float,
    margin_gate: float,
    fallback_result: Dict[str, Any],
    reason: str,
) -> None:
    ts = utc_ts()
    insert_model_signal_event(
        {
            "ts": ts,
            "signal_id": signal_id,
            "pair": pair6,
            "instrument": instrument,
            "role": "primary",
            "model_name": primary_model_type,
            "hint_side": hint_side,
            "approval_probability": primary_probability,
            "margin": primary_margin,
            "model_would_order": primary_would_order,
            "actual_order_sent": bool(final_order_allowed and decision_source == "primary"),
            "decision_source": decision_source,
            "conf_gate": conf_gate,
            "margin_gate": margin_gate,
            "reason": reason,
        }
    )
    for model_name, vote in (fallback_result.get("candidate_votes") or {}).items():
        insert_model_signal_event(
            {
                "ts": ts,
                "signal_id": signal_id,
                "pair": pair6,
                "instrument": instrument,
                "role": "candidate",
                "model_name": model_name,
                "hint_side": hint_side,
                "approval_probability": safe_float(vote.get("approval_probability"), 0.0),
                "margin": safe_float(vote.get("margin"), 0.0),
                "model_would_order": bool(vote.get("ok", False)),
                "actual_order_sent": bool(final_order_allowed and decision_source == "fallback" and fallback_result.get("fallback_model") == model_name),
                "decision_source": decision_source,
                "conf_gate": safe_float(vote.get("fallback_conf_gate"), conf_gate),
                "margin_gate": safe_float(vote.get("fallback_margin_gate"), margin_gate),
                "reason": str(vote.get("quality_reason") or fallback_result.get("fallback_reason") or reason),
            }
        )


def maybe_auto_switch_model(pair6: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    if not AUTO_MODEL_SWITCH_ENABLED:
        return {"switched": False, "reason": "auto_switch_disabled"}
    active_model = str(bundle.get("model_type") or "")
    if not active_model:
        return {"switched": False, "reason": "no_active_model"}
    existing = read_active_override_from_db(pair6)
    if existing and existing.get("updated_at"):
        try:
            updated_at = pd.to_datetime(existing["updated_at"], utc=True).to_pydatetime()
            age_min = (now_utc() - updated_at).total_seconds() / 60.0
            if age_min < SWITCH_COOLDOWN_MINUTES:
                return {"switched": False, "reason": f"switch_cooldown_active:{age_min:.1f}m<{SWITCH_COOLDOWN_MINUTES}m"}
        except Exception:
            pass
    conn = db_conn()
    primary_rows = conn.execute(
        """
        SELECT * FROM model_signal_events
        WHERE pair = ? AND role = 'primary' AND model_name = ?
        ORDER BY id DESC LIMIT ?
        """,
        (pair6, active_model, SWITCH_LOOKBACK_EVENTS),
    ).fetchall()
    primary_alerts = len(primary_rows)
    if primary_alerts < SWITCH_MIN_ALERTS:
        conn.close()
        return {"switched": False, "reason": f"not_enough_primary_alerts:{primary_alerts}<{SWITCH_MIN_ALERTS}"}
    primary_would = sum(int(row["model_would_order"] or 0) for row in primary_rows)
    primary_avg_prob = float(np.mean([safe_float(row["approval_probability"], 0.0) for row in primary_rows])) if primary_rows else 0.0
    if primary_would > SWITCH_MAX_PRIMARY_WOULD_ORDERS:
        conn.close()
        return {"switched": False, "reason": f"primary_not_inactive:primary_would={primary_would}>{SWITCH_MAX_PRIMARY_WOULD_ORDERS}"}
    candidate_rows = conn.execute(
        """
        SELECT * FROM model_signal_events
        WHERE pair = ? AND role = 'candidate'
        ORDER BY id DESC LIMIT ?
        """,
        (pair6, SWITCH_LOOKBACK_EVENTS * max(2, len(bundle.get("candidate_models") or {}))),
    ).fetchall()
    conn.close()
    by_model: Dict[str, List[Any]] = {}
    for row in candidate_rows:
        model_name = str(row["model_name"])
        if model_name != active_model:
            by_model.setdefault(model_name, []).append(row)
    candidate_metrics = bundle.get("candidate_metrics") or {}
    best_candidate = None
    for model_name, rows in by_model.items():
        rows = rows[:SWITCH_LOOKBACK_EVENTS]
        would_count = sum(int(row["model_would_order"] or 0) for row in rows)
        avg_prob = float(np.mean([safe_float(row["approval_probability"], 0.0) for row in rows])) if rows else 0.0
        quality_ok, quality_reason = candidate_metric_good(candidate_metrics.get(model_name, {}), use_fallback_thresholds=False)
        if not quality_ok:
            continue
        if would_count < SWITCH_MIN_CANDIDATE_WOULD_ORDERS:
            continue
        if avg_prob < primary_avg_prob + SWITCH_MIN_CONF_EDGE:
            continue
        record = {"model_name": model_name, "would_count": would_count, "avg_prob": avg_prob, "quality_reason": quality_reason}
        if best_candidate is None or (record["would_count"], record["avg_prob"]) > (best_candidate["would_count"], best_candidate["avg_prob"]):
            best_candidate = record
    if best_candidate is None:
        return {
            "switched": False,
            "reason": "no_candidate_met_switch_rules",
            "primary_alerts": primary_alerts,
            "primary_would": primary_would,
            "primary_avg_probability": primary_avg_prob,
        }
    reason = (
        f"auto_switch_primary_inactive | primary={active_model}, primary_alerts={primary_alerts}, "
        f"primary_would={primary_would}, primary_avg_probability={primary_avg_prob:.4f}, "
        f"new_model={best_candidate['model_name']}, candidate_would={best_candidate['would_count']}, "
        f"candidate_avg_probability={best_candidate['avg_prob']:.4f}"
    )
    write_active_model_override(pair6, best_candidate["model_name"], active_model, reason, bundle)
    return {"switched": True, "reason": reason, "new_model": best_candidate["model_name"], "previous_model": active_model}

# ============================================================
# PAYLOAD MODELS
# ============================================================
class TVPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = "fx"
    symbol: str
    tf: Optional[str] = "15"
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
    ret1: Optional[float] = None
    ret3: Optional[float] = None
    ret5: Optional[float] = None
    ema20_dist: Optional[float] = None
    ema50_dist: Optional[float] = None
    ema200_dist: Optional[float] = None
    rsi14: Optional[float] = None
    atr14_pct: Optional[float] = None
    bb_width: Optional[float] = None
    macd_hist: Optional[float] = None
    vol_z: Optional[float] = None
    hour_utc: Optional[int] = None
    dayofweek: Optional[int] = None
    setup_pullback: Optional[float] = None
    setup_ema_cross: Optional[float] = None


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
    model_config = ConfigDict(extra="allow")
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
    signal_id: Optional[str] = None
    tracking_key: Optional[str] = None

# ============================================================
# RESPONSE / APP
# ============================================================
def response_base(
    payload: TVPayload,
    pair6: str,
    instrument: str,
    bundle: Optional[Dict[str, Any]],
    hint_side: str,
    equity_used: float,
    approval_probability: float = 0.0,
    margin: float = 0.0,
) -> Dict[str, Any]:
    model_version = bundle.get("model_version") if bundle else ""
    avg_auc = bundle.get("avg_auc") if bundle else 0.0
    precision_at_gate = bundle.get("precision_at_gate") if bundle else None
    clean_symbol = pair6 or normalize_pair(payload.symbol) or str(payload.symbol).upper().replace("_", "")
    return {
        "ts": utc_ts(),
        "pair": pair6,
        "instrument": instrument,
        "symbol": clean_symbol,
        "raw_symbol": payload.symbol,
        "tf": payload.tf or "15",
        "granularity": "M15",
        "architecture": "signal_conditioned_hint_approval",
        "hint_side": hint_side,
        "model_version": model_version,
        "avg_auc": avg_auc,
        "precision_at_gate": precision_at_gate,
        "equity_used": equity_used,
        "approval_probability": float(approval_probability),
        "confidence": float(approval_probability),
        "margin": float(margin),
        "spread_pips": float(getattr(payload, "spread_pips", 0.0) or 0.0),
        "spread_atr": float(getattr(payload, "spread_atr", 0.0) or 0.0),
    }


app = FastAPI(title="FX M15 Signal-Conditioned Approval Server", version=APP_VERSION)


@app.on_event("startup")
def startup() -> None:
    global BUNDLES, NEWS_EVENTS
    init_db()
    restore_open_trades_from_db()
    seed_history_from_csv(DATA_DIR)
    BUNDLES = load_bundles(MODELS_DIR)
    NEWS_EVENTS = load_news_events()
    print(f"Loaded {len(NEWS_EVENTS)} news blackout events")
    if AUTO_CLOSE_ENABLED:
        threading.Thread(target=auto_close_worker, daemon=True).start()
    if PROFIT_PROTECTION_ENABLED:
        threading.Thread(target=profit_protection_worker, daemon=True).start()
    if CLOSED_TRADE_SYNC_ENABLED:
        threading.Thread(target=closed_trade_sync_worker, daemon=True).start()


@app.post("/predict")
def predict(payload_obj: TVPayload):
    pair6 = normalize_pair(payload_obj.symbol)
    hint_side = normalize_side(getattr(payload_obj, "hint_side", "") or "")
    equity_used = get_equity_used(payload_obj)
    if pair6 is None or pair6 not in PAIR_MAP:
        out = {
            "decision": "NONE",
            "why": "Symbol not allowed",
            "would_order": False,
            "order_allowed": False,
            **response_base(payload_obj, "", "", None, hint_side, equity_used),
        }
        write_audit_row(out)
        return out
    instrument = pair_to_instrument(pair6)
    if REQUIRE_HINT_SIDE and hint_side not in {"BUY", "SELL"}:
        out = {
            "decision": "NONE",
            "why": "Missing or invalid hint_side. Signal-conditioned M15 model requires BUY or SELL.",
            "would_order": False,
            "order_allowed": False,
            **response_base(payload_obj, pair6, instrument, None, hint_side, equity_used),
        }
        write_audit_row(out)
        return out
    payload = payload_obj.model_dump()
    pip = instrument_pip_size(instrument)
    if payload.get("spread_pips") in (None, ""):
        spread_c = safe_float(payload.get("spread_c"), np.nan)
        bid_c = safe_float(payload.get("bid_c"), np.nan)
        ask_c = safe_float(payload.get("ask_c"), np.nan)
        if (not np.isfinite(spread_c) or spread_c <= 0) and np.isfinite(bid_c) and np.isfinite(ask_c) and ask_c >= bid_c:
            spread_c = ask_c - bid_c
            payload["spread_c"] = spread_c
        payload["spread_pips"] = spread_c / pip if np.isfinite(spread_c) and spread_c >= 0 else 0.0
        payload_obj.spread_pips = payload["spread_pips"]
    if payload.get("spread_atr") in (None, ""):
        atr14 = safe_float(payload.get("atr14"), 0.0)
        spread_c = safe_float(payload.get("spread_c"), 0.0)
        payload["spread_atr"] = spread_c / atr14 if atr14 > 0 else 0.0
        payload_obj.spread_atr = payload["spread_atr"]
    bundle = BUNDLES.get(pair6)
    if not bundle:
        out = {
            "decision": "NONE",
            "why": "M15 signal-approval model not loaded for symbol",
            "would_order": False,
            "order_allowed": False,
            **response_base(payload_obj, pair6, instrument, None, hint_side, equity_used),
        }
        write_audit_row(out)
        return out
    static_ok = bool(bundle.get("static_filter_passed", False))
    static_reason = str(bundle.get("static_filter_reason", ""))
    sanity_reason = payload_sanity_checks(payload, instrument)
    # Payload sanity remains a hard block.
    # Static training filter is now informational by default so that:
    #   - weak selected primary models do not block the whole pair early,
    #   - other saved candidate models can be scored,
    #   - fallback can rescue a valid signal,
    #   - auto-switch can collect evidence and move away from weak primaries.
    # Set STATIC_FILTER_EARLY_BLOCK_ENABLED=true only if you intentionally want the old behavior.
    if sanity_reason or (STATIC_FILTER_EARLY_BLOCK_ENABLED and not static_ok):
        why = sanity_reason if sanity_reason else f"Static training filter blocked early: {static_reason}"
        out = {
            "decision": "NONE",
            "why": why,
            "would_order": False,
            "order_allowed": False,
            "model_type": bundle.get("model_type"),
            "trades_at_gate": bundle.get("trades_at_gate"),
            "static_filter_passed": static_ok,
            "static_filter_reason": static_reason,
            "static_filter_early_block_enabled": STATIC_FILTER_EARLY_BLOCK_ENABLED,
            **response_base(payload_obj, pair6, instrument, bundle, hint_side, equity_used),
        }
        write_audit_row(out)
        insert_prediction_event(
            {
                "ts": out["ts"],
                "signal_id": None,
                "pair": pair6,
                "instrument": instrument,
                "model_type": bundle.get("model_type"),
                "hint_side": hint_side,
                "decision": "NONE",
                "approval_probability": 0.0,
                "margin": 0.0,
                "would_order": False,
                "order_allowed": False,
                "reason": why,
            }
        )
        return out
    try:
        feature_row = build_runtime_feature_row(payload, pair6, instrument, hint_side, bundle["feature_order"])
        X = pd.DataFrame([{feature: feature_row.get(feature, 0.0) for feature in bundle["feature_order"]}], columns=bundle["feature_order"])
        primary_model_type = str(bundle.get("model_type"))
        primary_prob = max(0.0, min(1.0, float(predict_approval_probability(bundle["model"], X))))
        primary_margin = approval_margin(primary_prob)
        # Always use live Railway env gates, not the saved model-bundle training gate.
        # This keeps /predict aligned with /health approval_gate and approval_margin_gate.
        gate = APPROVAL_GATE
        margin_gate = APPROVAL_MARGIN_GATE
        fingerprint = make_signal_fingerprint(instrument, hint_side, payload_obj.t, float(payload_obj.mid_c), payload_obj.tf)
        noise_ok, noise_reason, noise_metrics = runtime_noise_filter(payload, feature_row, instrument)
        news_ok, news_reason, news_metrics = runtime_news_filter(pair6, payload)
        staleness_ok, staleness_reason, staleness_metrics = signal_staleness_guard(payload)
        entry_guard_ok, entry_guard_reason, entry_guard_metrics = entry_reversal_guard(payload, feature_row, instrument, hint_side)
        direction_ok, direction_reason, direction_metrics = direction_consensus_guard(payload, feature_row, instrument, hint_side)
        base = response_base(payload_obj, pair6, instrument, bundle, hint_side, equity_used, primary_prob, primary_margin)
        base.update({
            "model_feature_source_requested": MODEL_FEATURE_SOURCE,
            "model_feature_source_used": feature_row.get("_model_feature_source_used", "unknown"),
            "model_feature_source_ok": bool(feature_row.get("_model_feature_source_ok", False)),
            "model_feature_source_reason": feature_row.get("_model_feature_source_reason", ""),
            "model_feature_granularity": feature_row.get("_model_feature_granularity", MODEL_FEATURE_OANDA_GRANULARITY),
            "model_feature_time": feature_row.get("_model_feature_time", ""),
            "model_feature_last_close": feature_row.get("_model_feature_last_close", 0.0),
            "model_feature_candles": feature_row.get("_model_feature_candles", 0),
        })
        if not noise_ok or not news_ok or not staleness_ok or not entry_guard_ok or not direction_ok:
            if not noise_ok:
                why = noise_reason
                source = "noise_filter_block"
            elif not news_ok:
                why = news_reason
                source = "news_filter_block"
            elif not staleness_ok:
                why = staleness_reason
                source = "signal_staleness_block"
            elif not entry_guard_ok:
                why = entry_guard_reason
                source = "entry_reversal_guard_block"
            else:
                why = direction_reason
                source = "direction_confirmation_block"
            out = {
                "decision": "NONE",
                "why": why,
                "would_order": False,
                "order_allowed": False,
                "model_type": bundle.get("model_type"),
                "selected_model_type": bundle.get("model_type"),
                "trades_at_gate": bundle.get("trades_at_gate"),
                "static_filter_passed": static_ok,
                "static_filter_reason": static_reason,
                "static_filter_early_block_enabled": STATIC_FILTER_EARLY_BLOCK_ENABLED,
                "conf_gate": gate,
                "margin_gate": margin_gate,
                "decision_source": source,
                "signal_id": fingerprint,
                "primary_model_type": primary_model_type,
                "primary_approval_probability": primary_prob,
                "primary_margin": primary_margin,
                "fallback_used": False,
                "fallback_allowed": False,
                "fallback_reason": f"{source}_before_fallback",
                "candidate_votes": {},
                "noise_filter_passed": noise_ok,
                "noise_filter_reason": noise_reason,
                "news_filter_passed": news_ok,
                "news_filter_reason": news_reason,
                "signal_staleness_guard_passed": staleness_ok,
                "signal_staleness_guard_reason": staleness_reason,
                "entry_reversal_guard_passed": entry_guard_ok,
                "entry_reversal_guard_reason": entry_guard_reason,
                "direction_confirmation_passed": direction_ok,
                "direction_confirmation_reason": direction_reason,
                **noise_metrics,
                **news_metrics,
                **staleness_metrics,
                **entry_guard_metrics,
                **direction_metrics,
                **base,
            }
            write_audit_row(out)
            insert_prediction_event(
                {
                    "ts": out["ts"],
                    "signal_id": fingerprint,
                    "pair": pair6,
                    "instrument": instrument,
                    "model_type": bundle.get("model_type"),
                    "hint_side": hint_side,
                    "decision": "NONE",
                    "approval_probability": primary_prob,
                    "margin": primary_margin,
                    "would_order": False,
                    "order_allowed": False,
                    "reason": why,
                }
            )
            return out
        primary_quality_ok, primary_quality_reason = primary_metric_is_good_for_order(bundle)
        primary_gate_ok = primary_prob >= gate and primary_margin >= margin_gate
        primary_would_order = bool(primary_quality_ok and primary_gate_ok)
        primary_block_reasons = []
        if not primary_quality_ok:
            primary_block_reasons.append(primary_quality_reason)
        if not primary_gate_ok:
            primary_block_reasons.append(f"primary_below_approval_gate:approval={primary_prob:.2f}/{gate:.2f},margin={primary_margin:.2f}/{margin_gate:.2f}")
        fallback_result = evaluate_candidate_models_for_fallback(bundle, X, primary_model_type, gate, margin_gate)
        would_order = primary_would_order
        decision_source = "primary"
        model_type = primary_model_type
        decision_prob = primary_prob
        decision_margin = primary_margin
        avg_auc = safe_float(bundle.get("avg_auc"), 0.0)
        precision_at_gate = safe_float(bundle.get("precision_at_gate"), 0.0)
        trades_at_gate = safe_int(bundle.get("trades_at_gate"), 0)
        if primary_would_order:
            fallback_result["fallback_allowed"] = False
            fallback_result["fallback_used"] = False
            fallback_result["fallback_reason"] = "primary_passed_fallback_not_needed"
        elif fallback_result.get("fallback_allowed"):
            decision_source = "fallback"
            model_type = str(fallback_result.get("fallback_model"))
            decision_prob = safe_float(fallback_result.get("fallback_approval_probability"), 0.0)
            decision_margin = safe_float(fallback_result.get("fallback_margin"), 0.0)
            would_order = True
            metric = (bundle.get("candidate_metrics") or {}).get(model_type, {})
            avg_auc = safe_float(metric.get("auc"), avg_auc)
            precision_at_gate = safe_float(metric.get("precision_at_gate"), precision_at_gate)
            trades_at_gate = safe_int(metric.get("trades_at_gate"), trades_at_gate)
            base = response_base(payload_obj, pair6, instrument, {**bundle, "model_version": f"{pair6}:M15:signal_approval:{model_type}:fallback", "avg_auc": avg_auc, "precision_at_gate": precision_at_gate}, hint_side, equity_used, decision_prob, decision_margin)
        if would_order and is_duplicate_signal(pair6, fingerprint):
            why = f"Duplicate signal blocked for {instrument}"
            out = {
                "decision": "NONE",
                "why": why,
                "would_order": False,
                "order_allowed": False,
                "model_type": model_type,
                "selected_model_type": model_type,
                "trades_at_gate": trades_at_gate,
                "static_filter_passed": static_ok,
                "static_filter_reason": static_reason,
                "static_filter_early_block_enabled": STATIC_FILTER_EARLY_BLOCK_ENABLED,
                "conf_gate": gate,
                "margin_gate": margin_gate,
                "decision_source": "duplicate_block",
                "signal_id": fingerprint,
                "primary_model_type": primary_model_type,
                "primary_approval_probability": primary_prob,
                "primary_margin": primary_margin,
                "primary_training_quality_ok": primary_quality_ok,
                "primary_training_quality_reason": primary_quality_reason,
                "primary_gate_ok": primary_gate_ok,
                "primary_would_order": primary_would_order,
                "fallback_used": bool(fallback_result.get("fallback_used", False)),
                "fallback_allowed": bool(fallback_result.get("fallback_allowed", False)),
                "fallback_reason": fallback_result.get("fallback_reason"),
                "fallback_model": fallback_result.get("fallback_model"),
                "candidate_votes": fallback_result.get("candidate_votes", {}),
                "noise_filter_passed": True,
                "noise_filter_reason": noise_reason,
                "news_filter_passed": True,
                "news_filter_reason": news_reason,
                **noise_metrics,
                **news_metrics,
                **base,
            }
            write_audit_row(out)
            log_model_signal_events(pair6, instrument, fingerprint, hint_side, primary_model_type, primary_prob, primary_margin, primary_would_order, False, "duplicate_block", gate, margin_gate, fallback_result, why)
            insert_prediction_event(
                {
                    "ts": out["ts"],
                    "signal_id": fingerprint,
                    "pair": pair6,
                    "instrument": instrument,
                    "model_type": model_type,
                    "hint_side": hint_side,
                    "decision": "NONE",
                    "approval_probability": decision_prob,
                    "margin": decision_margin,
                    "would_order": False,
                    "order_allowed": False,
                    "reason": why,
                }
            )
            return out
        block_reason = None
        if would_order and trades_today_total() >= MAX_TRADES_PER_DAY_TOTAL:
            would_order = False
            block_reason = f"Daily lock: total max trades reached ({MAX_TRADES_PER_DAY_TOTAL})"
        elif would_order and trades_today(pair6) >= MAX_TRADES_PER_DAY_PER_PAIR:
            would_order = False
            block_reason = f"Daily lock: max trades for {instrument} reached"
        elif would_order and not can_open_trade():
            would_order = False
            block_reason = f"One-trade lock active: open={current_open_trade_count()}, pending={current_pending_trade_count()}, max={MAX_OPEN_TRADES}"
        if not would_order and block_reason is None:
            block_reason = "; ".join(primary_block_reasons) or str(fallback_result.get("fallback_reason") or "primary_and_fallback_blocked")
        log_model_signal_events(pair6, instrument, fingerprint, hint_side, primary_model_type, primary_prob, primary_margin, primary_would_order, bool(would_order), decision_source, gate, margin_gate, fallback_result, block_reason or "")
        auto_switch_result = maybe_auto_switch_model(pair6, bundle)
        units_abs = None
        units_signed = None
        sl_pips = None
        tp_pips = None
        sl_price = None
        tp_price = None
        order_result = None
        live_guard_metrics: Dict[str, Any] = {}
        live_guard_reason = None
        market_context: Dict[str, Any] = {"enabled": False, "ok": True, "reason": "not_built_yet"}
        ai_review: Dict[str, Any] = {"enabled": False, "ai_verdict": "SKIPPED", "risk_score": 0, "reason": "not_called"}
        decision = "NONE"
        why = block_reason or f"Signal rejected | approval={decision_prob:.2f}/{gate:.2f}, margin={decision_margin:.2f}/{margin_gate:.2f}, model={model_type}, precision_at_gate={precision_at_gate:.2f}, trades_at_gate={trades_at_gate}"
        if would_order:
            runtime_atr = safe_float(payload.get("atr14"), 0.0)
            if runtime_atr <= 0:
                hist = update_bar_history(pair6, payload)
                feat_df = add_signal_approval_runtime_features(hist, instrument, hint_side, payload)
                runtime_atr = safe_float(feat_df.iloc[-1].get("atr14"), instrument_pip_size(instrument) * 8)
            sl_pips, tp_pips, sl_price, tp_price = compute_sl_tp_prices(hint_side, float(payload_obj.mid_c), float(runtime_atr), instrument, bundle["labeling"]["sl_atr"], bundle["labeling"]["tp_atr"])
            guard_ok, live_guard_reason, live_guard_metrics = live_quote_guard_reprice_sltp(hint_side, instrument, float(runtime_atr), bundle["labeling"]["sl_atr"], bundle["labeling"]["tp_atr"], sl_price, tp_price)
            if not guard_ok:
                would_order = False
                block_reason = live_guard_reason
                why = live_guard_reason
                decision = "NONE"
                units_abs = None
                units_signed = None
                sl_pips = None
                tp_pips = None
                sl_price = None
                tp_price = None
            else:
                sl_price = live_guard_metrics.get("live_reprice_final_sl_price") or sl_price
                tp_price = live_guard_metrics.get("live_reprice_final_tp_price") or tp_price
                sl_pips = live_guard_metrics.get("live_reprice_final_sl_pips") or sl_pips
                tp_pips = live_guard_metrics.get("live_reprice_final_tp_pips") or tp_pips
                # Build fresh OANDA market context and compare it with the alert features before final approval.
                market_context = build_external_market_context(pair6, instrument, hint_side, feature_row)
                if MARKET_CONTEXT_REQUIRED and not bool(market_context.get("ok", False)):
                    would_order = False
                    block_reason = "Market context required but unavailable/conflicting: " + "; ".join(market_context.get("errors") or [market_context.get("reason", "unknown")])
                    why = block_reason
                    decision = "NONE"
                    units_abs = None
                    units_signed = None
                    sl_pips = None
                    tp_pips = None
                    sl_price = None
                    tp_price = None
                else:
                    ai_context = build_ai_review_context(
                        pair6=pair6,
                        instrument=instrument,
                        hint_side=hint_side,
                        model_type=model_type,
                        decision_prob=decision_prob,
                        gate=gate,
                        feature_row=feature_row,
                        market_context=market_context,
                        risk_context={
                            "direction_confirmation_passed": direction_ok,
                            "entry_reversal_guard_passed": entry_guard_ok,
                            "noise_filter_passed": noise_ok,
                            "news_filter_passed": news_ok,
                            "live_guard_passed": guard_ok,
                            "live_bid": live_guard_metrics.get("live_bid"),
                            "live_ask": live_guard_metrics.get("live_ask"),
                            "live_reprice_reference_price": live_guard_metrics.get("live_reprice_reference_price"),
                            "spread_pips": feature_row.get("spread_pips"),
                            "spread_atr": getattr(payload_obj, "spread_atr", 0.0),
                            "active_trade_count": current_active_trade_count(),
                        },
                    )
                    ai_review = review_signal_with_ai(ai_context)
                    if AI_REVIEW_ENABLED and ai_review.get("ai_verdict") != "APPROVE":
                        would_order = False
                        block_reason = f"AI reviewer rejected: {ai_review.get('reason')}"
                        why = block_reason
                        decision = "NONE"
                        units_abs = None
                        units_signed = None
                        sl_pips = None
                        tp_pips = None
                        sl_price = None
                        tp_price = None
                    else:
                        units_abs = compute_units_dynamic(instrument, sl_pips, avg_auc, precision_at_gate, equity_used, payload_obj.force_units_abs)
                        units_signed = units_abs if hint_side == "BUY" else -units_abs
                        decision = hint_side
                        ai_note = f", ai_review={ai_review.get('ai_verdict')}" if AI_REVIEW_ENABLED else ""
                        why = f"OK: {hint_side} signal approved | model={model_type}, approval={decision_prob:.2f}, margin={decision_margin:.2f}, precision_at_gate={precision_at_gate:.2f}, trades_at_gate={trades_at_gate}, equity_used={equity_used:.2f}{ai_note}"
                        if LIVE_TRADING:
                            client_id = f"m15_{pair6}_{payload_obj.t}_{hint_side.lower()}"
                            order_result = submit_oanda_order(instrument, units_signed, sl_price, tp_price, client_id)
        pending_trade_id = None
        if would_order:
            pending_trade_id = reserve_pending_trade(
                signal_id=fingerprint,
                pair6=pair6,
                instrument=instrument,
                side=hint_side,
                model_type=model_type,
                decision_source=decision_source,
            )

        out = {
            "decision": decision,
            "why": why,
            "would_order": bool(would_order),
            "order_allowed": bool(would_order),
            "max_open_trades": MAX_OPEN_TRADES,
            "open_trade_count": current_open_trade_count(),
            "pending_trade_count": current_pending_trade_count(),
            "active_trade_count": current_active_trade_count(),
            "pending_trade_lock_enabled": PENDING_TRADE_LOCK_ENABLED,
            "pending_trade_id": pending_trade_id,
            "pending_trade_timeout_seconds": PENDING_TRADE_TIMEOUT_SECONDS,
            "units": units_abs,
            "units_signed": units_signed,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "model_type": model_type,
            "selected_model_type": model_type,
            "active_model_type": primary_model_type,
            "decision_model_type": model_type,
            "trades_at_gate": trades_at_gate,
            "static_filter_passed": static_ok,
            "static_filter_reason": static_reason,
            "static_filter_early_block_enabled": STATIC_FILTER_EARLY_BLOCK_ENABLED,
            "conf_gate": gate,
            "margin_gate": margin_gate,
            "decision_source": decision_source,
            "signal_id": fingerprint,
            "primary_model_type": primary_model_type,
            "primary_approval_probability": primary_prob,
            "primary_margin": primary_margin,
            "primary_training_quality_ok": primary_quality_ok,
            "primary_training_quality_reason": primary_quality_reason,
            "primary_gate_ok": primary_gate_ok,
            "primary_would_order": primary_would_order,
            "fallback_used": bool(fallback_result.get("fallback_used", False)),
            "fallback_allowed": bool(fallback_result.get("fallback_allowed", False)),
            "fallback_reason": fallback_result.get("fallback_reason"),
            "fallback_model": fallback_result.get("fallback_model"),
            "candidate_votes": fallback_result.get("candidate_votes", {}),
            "auto_switch_result": auto_switch_result,
            "noise_filter_passed": True,
            "noise_filter_reason": noise_reason,
            "news_filter_passed": True,
            "news_filter_reason": news_reason,
            "signal_staleness_guard_passed": staleness_ok,
            "signal_staleness_guard_reason": staleness_reason,
            "entry_reversal_guard_passed": entry_guard_ok,
            "entry_reversal_guard_reason": entry_guard_reason,
            "direction_confirmation_passed": direction_ok,
            "direction_confirmation_reason": direction_reason,
            "live_quote_guard_reason": live_guard_reason,
            "market_context_enabled": MARKET_CONTEXT_ENABLED,
            "market_context_required": MARKET_CONTEXT_REQUIRED,
            "external_market_context": market_context,
            "ai_review_enabled": AI_REVIEW_ENABLED,
            "ai_review": ai_review,
            **noise_metrics,
            **news_metrics,
            **staleness_metrics,
            **entry_guard_metrics,
            **direction_metrics,
            **live_guard_metrics,
            "live_trading": LIVE_TRADING,
            "order_submitted": bool(order_result and order_result.get("ok")),
            "order_result": order_result,
            **base,
        }
        if would_order:
            remember_signal(pair6, fingerprint)
            inc_trade(pair6)
        write_audit_row(out)
        insert_or_update_pattern_history_from_prediction(out, feature_row, market_context, ai_review)
        insert_prediction_event(
            {
                "ts": out["ts"],
                "signal_id": fingerprint,
                "pair": pair6,
                "instrument": instrument,
                "model_type": model_type,
                "hint_side": hint_side,
                "decision": decision,
                "approval_probability": decision_prob,
                "margin": decision_margin,
                "would_order": bool(would_order),
                "order_allowed": bool(would_order),
                "reason": why,
            }
        )
        return out
    except Exception as exc:
        out = {
            "decision": "NONE",
            "why": f"Prediction error: {repr(exc)}",
            "would_order": False,
            "order_allowed": False,
            "model_type": bundle.get("model_type"),
            "trades_at_gate": bundle.get("trades_at_gate"),
            "static_filter_passed": static_ok,
            "static_filter_reason": static_reason,
            "static_filter_early_block_enabled": STATIC_FILTER_EARLY_BLOCK_ENABLED,
            **response_base(payload_obj, pair6, instrument, bundle, hint_side, equity_used),
        }
        write_audit_row(out)
        return out


@app.post("/score")
def score(payload_obj: TVPayload):
    return predict(payload_obj)


@app.post("/order")
def order(payload_obj: TVPayload):
    return predict(payload_obj)


@app.post("/trade_event")
def trade_event(event: TradeEvent):
    row = event.model_dump()
    if not row.get("ts"):
        row["ts"] = utc_ts()
    event.instrument = str(event.instrument).upper()
    event.side = normalize_side(event.side)
    if not row.get("symbol"):
        row["symbol"] = instrument_to_symbol(event.instrument)

    tracking_key = make_tracking_key(
        event.order_id,
        event.broker_trade_id,
        event.client_trade_id,
        event.instrument,
        event.side,
        row["ts"],
    )

    write_trade_row(
        {
            "instrument": event.instrument,
            "side": event.side,
            "units_signed": event.units_signed,
            "entry_price": event.entry_price,
            "sl_price": event.sl_price,
            "tp_price": event.tp_price,
            "status": event.status,
            "pnl": event.pnl,
            "order_id": event.order_id,
            "reason": event.reason,
            "pair_score": event.pair_score,
            "ts": row["ts"],
            "tracking_key": str(tracking_key),
            "broker_trade_id": event.broker_trade_id,
            "broker_order_id": event.broker_order_id,
            "client_trade_id": event.client_trade_id,
        }
    )

    update_pattern_history_from_trade_event(
        {
            **row,
            "instrument": event.instrument,
            "side": event.side,
            "status": event.status,
            "pnl": event.pnl,
            "order_id": event.order_id,
            "reason": event.reason,
            "broker_trade_id": event.broker_trade_id,
            "broker_order_id": event.broker_order_id,
            "client_trade_id": event.client_trade_id,
        },
        str(tracking_key),
    )

    if event.status == "OPEN":
        note_trade_opened(tracking_key)
        opened_at_dt = dt.datetime.now(dt.timezone.utc)
        try:
            opened_at_dt = pd.to_datetime(row["ts"], utc=True).to_pydatetime()
        except Exception:
            pass

        instrument = event.instrument
        pip = instrument_pip_size(instrument)
        initial_risk_pips = abs(float(event.entry_price) - float(event.sl_price)) / pip if pip > 0 else 0.0

        _open_trade_meta[str(tracking_key)] = {
            "tracking_key": str(tracking_key),
            "instrument": instrument,
            "symbol": row["symbol"],
            "side": event.side,
            "units_signed": event.units_signed,
            "entry_price": event.entry_price,
            "sl_price": event.sl_price,
            "original_sl_price": event.sl_price,
            "tp_price": event.tp_price,
            "pair_score": event.pair_score,
            "opened_at_dt": opened_at_dt,
            "order_id": event.order_id,
            "broker_trade_id": event.broker_trade_id,
            "broker_order_id": event.broker_order_id,
            "client_trade_id": event.client_trade_id,
            "initial_risk_pips": initial_risk_pips,
            "breakeven_done": False,
            "trailing_active": False,
            "last_stop_update_ts": None,
            "last_favorable_pips": None,
            "last_current_r": None,
            "peak_favorable_pips": 0.0,
            "peak_current_r": 0.0,
            "last_giveback_pips": 0.0,
            "last_giveback_r": 0.0,
            "ts": row["ts"],
        }

    if event.status in ("CLOSED", "STOPPED", "TAKE_PROFIT", "MANUAL"):
        for key in [tracking_key, event.broker_trade_id, event.client_trade_id, event.order_id]:
            if key:
                note_trade_closed(key)
                _open_trade_meta.pop(str(key), None)

    return {
        "ok": True,
        "open_trades": current_open_trade_count(),
        "pending_trades": current_pending_trade_count(),
        "active_trades": current_active_trade_count(),
        "status": event.status,
        "order_id": event.order_id,
        "tracking_key": tracking_key,
        "broker_trade_id": event.broker_trade_id,
        "profit_protection_tracking_ready": bool(
            event.status == "OPEN"
            and (event.broker_trade_id or event.client_trade_id)
        ),
    }


@app.get("/")
def root():
    return {
        "ok": True,
        "app": "FX M15 Signal-Conditioned Approval Server",
        "version": APP_VERSION,
        "pairs_loaded": len(BUNDLES),
        "live_trading": LIVE_TRADING,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": utc_ts(),
        "version": APP_VERSION,
        "architecture": "signal_conditioned_hint_approval",
        "pairs_loaded": len(BUNDLES),
        "pairs": sorted([pair_to_instrument(pair) for pair in BUNDLES.keys()]),
        "pair_details": {
            pair: {
                "instrument": bundle.get("instrument"),
                "model_type": bundle.get("model_type"),
                "saved_best_model_type": bundle.get("saved_best_model_type"),
                "candidate_models_loaded": sorted(list((bundle.get("candidate_models") or {}).keys())),
                "auc": bundle.get("avg_auc"),
                "precision_at_gate": bundle.get("precision_at_gate"),
                "trades_at_gate": bundle.get("trades_at_gate"),
                "static_filter_passed": bundle.get("static_filter_passed"),
                "static_filter_reason": bundle.get("static_filter_reason"),
                "bundle_path": bundle.get("_bundle_path"),
            }
            for pair, bundle in BUNDLES.items()
        },
        "approval_gate": APPROVAL_GATE,
        "approval_margin_gate": APPROVAL_MARGIN_GATE,
        "strict_model_filter_enabled": STRICT_MODEL_FILTER_ENABLED,
        "static_filter_early_block_enabled": STATIC_FILTER_EARLY_BLOCK_ENABLED,
        "primary_min_precision_for_order": PRIMARY_MIN_PRECISION_FOR_ORDER,
        "fallback_mode_enabled": FALLBACK_MODE_ENABLED,
        "auto_model_switch_enabled": AUTO_MODEL_SWITCH_ENABLED,
        "news_filter_enabled": NEWS_FILTER_ENABLED,
        "news_events_loaded": len(NEWS_EVENTS),
        "noise_filter_enabled": NOISE_FILTER_ENABLED,
        "live_price_guard_enabled": LIVE_PRICE_GUARD_ENABLED,
        "live_price_guard_required": LIVE_PRICE_GUARD_REQUIRED,
        "live_price_reprice_sltp": LIVE_PRICE_REPRICE_SLTP,
        "live_price_buffer_pips": LIVE_PRICE_BUFFER_PIPS,
        "live_price_max_age_seconds": LIVE_PRICE_MAX_AGE_SECONDS,
        "signal_staleness_guard_enabled": SIGNAL_STALENESS_GUARD_ENABLED,
        "signal_max_age_seconds": SIGNAL_MAX_AGE_SECONDS,
        "entry_reversal_guard_enabled": ENTRY_REVERSAL_GUARD_ENABLED,
        "entry_reversal_guard_required": ENTRY_REVERSAL_GUARD_REQUIRED,
        "entry_reversal_max_adverse_pips": ENTRY_REVERSAL_MAX_ADVERSE_PIPS,
        "entry_reversal_max_spread_pips": ENTRY_REVERSAL_MAX_SPREAD_PIPS,
        "direction_confirmation_enabled": DIRECTION_CONFIRMATION_ENABLED,
        "direction_confirmation_required": DIRECTION_CONFIRMATION_REQUIRED,
        "direction_confirm_min_score": DIRECTION_CONFIRM_MIN_SCORE,
        "direction_confirm_require_ema20_side": DIRECTION_CONFIRM_REQUIRE_EMA20_SIDE,
        "direction_confirm_require_ret3_align": DIRECTION_CONFIRM_REQUIRE_RET3_ALIGN,
        "direction_confirm_reject_countertrend_ema50": DIRECTION_CONFIRM_REJECT_COUNTERTREND_EMA50,
        "oanda_token_present": bool(OANDA_TOKEN),
        "oanda_account_id_present": bool(OANDA_ACCOUNT_ID),
        "live_trading": LIVE_TRADING,
        "auto_close_enabled": AUTO_CLOSE_ENABLED,
        "auto_close_only_checks_oanda_when_open_trade_exists": True,
        "profit_protection_enabled": PROFIT_PROTECTION_ENABLED,
        "open_trade_manager_check_seconds": OPEN_TRADE_MANAGER_CHECK_SECONDS,
        "profit_protection_only_checks_oanda_when_open_trade_exists": True,
        "breakeven_trigger_r": BREAKEVEN_TRIGGER_R,
        "breakeven_buffer_pips": BREAKEVEN_BUFFER_PIPS,
        "trailing_trigger_r": TRAILING_TRIGGER_R,
        "trailing_distance_r": TRAILING_DISTANCE_R,
        "trailing_min_improvement_pips": TRAILING_MIN_IMPROVEMENT_PIPS,
        "stop_update_live_buffer_pips": STOP_UPDATE_LIVE_BUFFER_PIPS,
        "profit_protection_require_trade_specifier": PROFIT_PROTECTION_REQUIRE_TRADE_SPECIFIER,
        "reversal_exit_enabled": REVERSAL_EXIT_ENABLED,
        "reversal_exit_min_profit_r": REVERSAL_EXIT_MIN_PROFIT_R,
        "reversal_exit_min_profit_pips": REVERSAL_EXIT_MIN_PROFIT_PIPS,
        "reversal_exit_giveback_r": REVERSAL_EXIT_GIVEBACK_R,
        "reversal_exit_giveback_pips": REVERSAL_EXIT_GIVEBACK_PIPS,
        "reversal_exit_min_current_profit_pips": REVERSAL_EXIT_MIN_CURRENT_PROFIT_PIPS,
        "adverse_exit_enabled": ADVERSE_EXIT_ENABLED,
        "adverse_exit_after_minutes": ADVERSE_EXIT_AFTER_MINUTES,
        "adverse_exit_min_loss_r": ADVERSE_EXIT_MIN_LOSS_R,
        "adverse_exit_min_loss_pips": ADVERSE_EXIT_MIN_LOSS_PIPS,
        "adverse_exit_require_no_recovery": ADVERSE_EXIT_REQUIRE_NO_RECOVERY,
        "adverse_exit_max_peak_profit_r": ADVERSE_EXIT_MAX_PEAK_PROFIT_R,
        "adverse_exit_max_peak_profit_pips": ADVERSE_EXIT_MAX_PEAK_PROFIT_PIPS,
        "closed_trade_sync_enabled": CLOSED_TRADE_SYNC_ENABLED,
        "closed_trade_sync_check_seconds": CLOSED_TRADE_SYNC_CHECK_SECONDS,
        "closed_trade_sync_only_checks_oanda_when_open_trade_exists": True,
        "closed_trade_sync_require_trade_specifier": CLOSED_TRADE_SYNC_REQUIRE_TRADE_SPECIFIER,
        "closed_trade_classification_enabled": CLOSED_TRADE_CLASSIFICATION_ENABLED,
        "closed_trade_classification_max_transactions": CLOSED_TRADE_CLASSIFICATION_MAX_TRANSACTIONS,
        "market_context_enabled": MARKET_CONTEXT_ENABLED,
        "market_context_required": MARKET_CONTEXT_REQUIRED,
        "market_context_granularities": MARKET_CONTEXT_GRANULARITIES,
        "market_context_candle_count": MARKET_CONTEXT_CANDLE_COUNT,
        "candle_pattern_context_enabled": CANDLE_PATTERN_CONTEXT_ENABLED,
        "pattern_history_enabled": PATTERN_HISTORY_ENABLED,
        "pattern_stats_for_ai_enabled": PATTERN_STATS_FOR_AI_ENABLED,
        "pattern_stats_lookback": PATTERN_STATS_LOOKBACK,
        "pattern_stats_min_closed": PATTERN_STATS_MIN_CLOSED,
        "model_feature_source": MODEL_FEATURE_SOURCE,
        "model_feature_oanda_granularity": MODEL_FEATURE_OANDA_GRANULARITY,
        "model_feature_oanda_candle_count": MODEL_FEATURE_OANDA_CANDLE_COUNT,
        "model_feature_oanda_min_candles": MODEL_FEATURE_OANDA_MIN_CANDLES,
        "model_feature_fallback_to_alert": MODEL_FEATURE_FALLBACK_TO_ALERT,
        "ai_review_enabled": AI_REVIEW_ENABLED,
        "ai_review_provider": AI_REVIEW_PROVIDER,
        "ai_review_model": AI_REVIEW_MODEL,
        "ai_review_max_risk_score": AI_REVIEW_MAX_RISK_SCORE,
        "ai_review_conditional_risk_score": AI_REVIEW_CONDITIONAL_RISK_SCORE,
        "ai_review_hard_block_score": AI_REVIEW_HARD_BLOCK_SCORE,
        "ai_review_min_model_prob": AI_REVIEW_MIN_MODEL_PROB,
        "ai_review_strong_model_prob": AI_REVIEW_STRONG_MODEL_PROB,
        "ai_review_max_spread_atr": AI_REVIEW_MAX_SPREAD_ATR,
        "ai_review_fallback_to_rules": AI_REVIEW_FALLBACK_TO_RULES,
        "max_open_trades": MAX_OPEN_TRADES,
        "current_open_trades": current_open_trade_count(),
        "current_pending_trades": current_pending_trade_count(),
        "current_active_trades": current_active_trade_count(),
        "pending_trade_lock_enabled": PENDING_TRADE_LOCK_ENABLED,
        "pending_trade_timeout_seconds": PENDING_TRADE_TIMEOUT_SECONDS,
    }


@app.get("/pattern_stats")
def pattern_stats(pair: Optional[str] = None, pattern: Optional[str] = None, side: Optional[str] = None, limit: int = 500):
    pair6 = normalize_pair(pair or "") if pair else None
    side_norm = normalize_side(side or "") if side else None
    clauses = ["trade_outcome IN ('win','loss','breakeven')"]
    params: List[Any] = []
    if pair6:
        clauses.append("pair=?")
        params.append(pair6)
    if pattern:
        clauses.append("candle_pattern=?")
        params.append(pattern)
    if side_norm:
        clauses.append("hint_side=?")
        params.append(side_norm)
    sql = f"""
        SELECT pair, hint_side, candle_pattern, candle_bias,
               COUNT(*) AS closed_samples,
               SUM(CASE WHEN trade_outcome='win' THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN trade_outcome='loss' THEN 1 ELSE 0 END) AS losses,
               SUM(CASE WHEN trade_outcome='breakeven' THEN 1 ELSE 0 END) AS breakeven,
               AVG(pnl) AS avg_pnl,
               SUM(CASE WHEN ai_verdict='APPROVE' THEN 1 ELSE 0 END) AS ai_approved,
               SUM(CASE WHEN ai_verdict='REJECT' THEN 1 ELSE 0 END) AS ai_rejected
        FROM signal_pattern_history
        WHERE {' AND '.join(clauses)}
        GROUP BY pair, hint_side, candle_pattern, candle_bias
        ORDER BY closed_samples DESC, wins DESC
        LIMIT ?
    """
    params.append(max(1, min(int(limit), 2000)))
    conn = db_conn()
    try:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
    for r in rows:
        total = int(r.get("closed_samples") or 0)
        wins = int(r.get("wins") or 0)
        losses = int(r.get("losses") or 0)
        r["win_rate"] = round(wins / total, 4) if total else None
        r["loss_rate"] = round(losses / total, 4) if total else None
        r["confidence"] = "usable" if total >= PATTERN_STATS_MIN_CLOSED else "low_sample"
    return {
        "ok": True,
        "enabled": PATTERN_HISTORY_ENABLED,
        "min_closed_for_confidence": PATTERN_STATS_MIN_CLOSED,
        "count": len(rows),
        "rows": rows,
    }


@app.get("/pattern_history")
def pattern_history(pair: Optional[str] = None, limit: int = 100):
    pair6 = normalize_pair(pair or "") if pair else None
    clauses = []
    params: List[Any] = []
    if pair6:
        clauses.append("pair=?")
        params.append(pair6)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    conn = db_conn()
    try:
        rows = [dict(r) for r in conn.execute(
            f"""
            SELECT ts, signal_id, pair, instrument, hint_side, model_type,
                   model_probability, conf_gate, decision, decision_source,
                   would_order, order_allowed, ai_verdict, ai_risk_score, ai_reason,
                   candle_pattern, candle_bias, pattern_confidence,
                   h1_hint_aligned, h4_hint_aligned, htf_conflict,
                   trade_status, trade_outcome, pnl, close_reason
            FROM signal_pattern_history
            {where}
            ORDER BY ts DESC LIMIT ?
            """,
            params,
        ).fetchall()]
    finally:
        conn.close()
    return {"ok": True, "enabled": PATTERN_HISTORY_ENABLED, "count": len(rows), "rows": rows}


@app.get("/export/pattern_history.csv")
def export_pattern_history_csv():
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM signal_pattern_history ORDER BY ts DESC", conn)
    finally:
        conn.close()
    path = os.path.join(LOG_DIR, "signal_pattern_history_export.csv")
    df.to_csv(path, index=False)
    return FileResponse(path, media_type="text/csv", filename="signal_pattern_history_export.csv")


@app.get("/model-info/{pair}")
def model_info(pair: str):
    pair6 = normalize_pair(pair)
    if not pair6 or pair6 not in BUNDLES:
        return {"ok": False, "pair": pair, "error": "pair_not_loaded"}
    bundle = BUNDLES[pair6]
    return {
        "ok": True,
        "pair": pair6,
        "instrument": bundle.get("instrument"),
        "selected_model_type": bundle.get("model_type"),
        "summary": bundle.get("summary"),
        "metadata": bundle.get("metadata"),
        "candidate_models_loaded": sorted(list((bundle.get("candidate_models") or {}).keys())),
        "candidate_metrics": bundle.get("candidate_metrics"),
    }


@app.post("/reload-models")
def reload_models():
    global BUNDLES, NEWS_EVENTS
    BUNDLES = load_bundles(MODELS_DIR)
    NEWS_EVENTS = load_news_events()
    return {"ok": True, "pairs_loaded": len(BUNDLES), "pairs": sorted(BUNDLES.keys()), "ts": utc_ts()}


@app.get("/news_events")
def news_events():
    return {"ok": True, "events_loaded": len(NEWS_EVENTS), "events": NEWS_EVENTS}


@app.post("/reload-news")
def reload_news():
    global NEWS_EVENTS
    NEWS_EVENTS = load_news_events()
    return {"ok": True, "events_loaded": len(NEWS_EVENTS), "events": NEWS_EVENTS, "ts": utc_ts()}


@app.post("/news_event")
def add_news_event(event: NewsEventPayload):
    global NEWS_EVENTS
    raw = event.model_dump(exclude_none=True)
    normalized = normalize_news_event(raw)
    if not normalized:
        return {"ok": False, "error": "invalid_news_event", "received": raw}
    NEWS_EVENTS.append(normalized)
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
    return {"ok": True, "event": normalized, "events_loaded": len(NEWS_EVENTS), "news_events_file": NEWS_EVENTS_FILE}


@app.get("/stats")
def stats():
    df = read_audit_df()
    if df.empty:
        return {"ok": True, "rows": 0, "would_order_count": 0, "decision_counts": {}, "pair_counts": {}, "last_ts": None}
    ts_series = pd.to_datetime(df["ts"], errors="coerce") if "ts" in df.columns else pd.Series(dtype="datetime64[ns]")
    last_ts = ts_series.dropna().max().isoformat() if not ts_series.dropna().empty else None
    return {
        "ok": True,
        "rows": int(len(df)),
        "would_order_count": int(safe_bool_series(df, "would_order").sum()),
        "decision_counts": df["decision"].value_counts(dropna=False).to_dict() if "decision" in df.columns else {},
        "pair_counts": df["instrument"].value_counts(dropna=False).to_dict() if "instrument" in df.columns else {},
        "last_ts": last_ts,
    }


@app.get("/model_performance")
def model_performance():
    conn = db_conn()
    rows = conn.execute(
        """
        SELECT pair, instrument, role, model_name,
               COUNT(*) AS signals_seen,
               SUM(model_would_order) AS would_order_count,
               SUM(actual_order_sent) AS actual_order_count,
               AVG(approval_probability) AS avg_approval_probability,
               AVG(margin) AS avg_margin,
               MAX(ts) AS last_ts
        FROM model_signal_events
        GROUP BY pair, instrument, role, model_name
        ORDER BY pair, role, model_name
        """
    ).fetchall()
    overrides = conn.execute("SELECT * FROM model_active_overrides").fetchall()
    conn.close()
    output: Dict[str, Any] = {}
    for row in rows:
        pair = row["pair"]
        output.setdefault(pair, {"models": {}, "active_override": None})
        output[pair]["models"][row["model_name"]] = {
            "role": row["role"],
            "instrument": row["instrument"],
            "signals_seen": safe_int(row["signals_seen"], 0),
            "would_order_count": safe_int(row["would_order_count"], 0),
            "actual_order_count": safe_int(row["actual_order_count"], 0),
            "avg_approval_probability": safe_float(row["avg_approval_probability"], 0.0),
            "avg_margin": safe_float(row["avg_margin"], 0.0),
            "last_ts": row["last_ts"],
        }
    for row in overrides:
        pair = row["pair"]
        output.setdefault(pair, {"models": {}, "active_override": None})
        output[pair]["active_override"] = dict(row)
    return {"ok": True, "ts": utc_ts(), "pairs": output}


@app.get("/open_trades")
def open_trades():
    return {
        "ok": True,
        "ts": utc_ts(),
        "open_trade_count": current_open_trade_count(),
        "pending_trade_count": current_pending_trade_count(),
        "active_trade_count": current_active_trade_count(),
        "max_open_trades": MAX_OPEN_TRADES,
        "pending_trade_lock_enabled": PENDING_TRADE_LOCK_ENABLED,
        "pending_trade_timeout_seconds": PENDING_TRADE_TIMEOUT_SECONDS,
        "pending_trades": _pending_trade_meta,
        "open_trades": {
            tracking_key: {
                "instrument": meta.get("instrument"),
                "symbol": meta.get("symbol"),
                "side": meta.get("side"),
                "units_signed": meta.get("units_signed"),
                "entry_price": meta.get("entry_price"),
                "current_sl_price": meta.get("sl_price"),
                "original_sl_price": meta.get("original_sl_price"),
                "tp_price": meta.get("tp_price"),
                "broker_trade_id": meta.get("broker_trade_id"),
                "client_trade_id": meta.get("client_trade_id"),
                "initial_risk_pips": meta.get("initial_risk_pips"),
                "breakeven_done": bool(meta.get("breakeven_done", False)),
                "trailing_active": bool(meta.get("trailing_active", False)),
                "last_stop_update_ts": meta.get("last_stop_update_ts"),
                "last_favorable_pips": meta.get("last_favorable_pips"),
                "last_current_r": meta.get("last_current_r"),
                "restored_from_db": bool(meta.get("restored_from_db", False)),
                "opened_at": meta.get("opened_at_dt").isoformat() if meta.get("opened_at_dt") else None,
            }
            for tracking_key, meta in _open_trade_meta.items()
        },
    }


@app.post("/clear_pending_trade_lock")
def clear_pending_trade_lock():
    """Manual emergency release for a pending lock if OANDA rejected an order and Make did not register an OPEN trade.

    Normally you should not need this because pending locks auto-expire after PENDING_TRADE_TIMEOUT_SECONDS.
    """
    before = current_pending_trade_count()
    clear_pending_trades("manual_api_clear")
    return {
        "ok": True,
        "ts": utc_ts(),
        "cleared_pending_trades": before,
        "open_trade_count": current_open_trade_count(),
        "pending_trade_count": current_pending_trade_count(),
        "active_trade_count": current_active_trade_count(),
    }


@app.get("/trade_management_stats")
def trade_management_stats():
    conn = db_conn()
    rows = conn.execute(
        """
        SELECT action,
               COUNT(*) AS event_count,
               SUM(success) AS success_count,
               MAX(ts) AS last_ts
        FROM trade_management_events
        GROUP BY action
        ORDER BY event_count DESC
        """
    ).fetchall()
    latest = conn.execute(
        """
        SELECT *
        FROM trade_management_events
        ORDER BY id DESC
        LIMIT 50
        """
    ).fetchall()
    conn.close()
    return {
        "ok": True,
        "ts": utc_ts(),
        "summary": [dict(row) for row in rows],
        "latest_events": [dict(row) for row in latest],
    }


@app.post("/manage_open_trades_now")
def manage_open_trades_now():
    results = {}
    if not _open_trade_meta:
        return {"ok": True, "ts": utc_ts(), "managed": 0, "reason": "no_open_trades_tracked", "results": results}
    if not broker_ready():
        return {"ok": False, "ts": utc_ts(), "managed": 0, "reason": "broker_not_ready", "results": results}
    for tracking_key, meta in list(_open_trade_meta.items()):
        sync_result = sync_single_tracked_trade_close(str(tracking_key), meta)
        if sync_result.get("action") == "CLOSED_TRADE_SYNC":
            results[str(tracking_key)] = sync_result
            continue
        results[str(tracking_key)] = manage_single_open_trade(str(tracking_key), meta)
    return {"ok": True, "ts": utc_ts(), "managed": len(results), "results": results}


@app.post("/sync_closed_trades_now")
def sync_closed_trades_now():
    results = {}
    if not _open_trade_meta:
        return {"ok": True, "ts": utc_ts(), "checked": 0, "closed_registered": 0, "reason": "no_open_trades_tracked", "results": results}
    if not broker_ready():
        return {"ok": False, "ts": utc_ts(), "checked": 0, "closed_registered": 0, "reason": "broker_not_ready", "results": results}
    closed_registered = 0
    for tracking_key, meta in list(_open_trade_meta.items()):
        result = sync_single_tracked_trade_close(str(tracking_key), meta)
        results[str(tracking_key)] = result
        if result.get("action") == "CLOSED_TRADE_SYNC":
            closed_registered += 1
    return {
        "ok": True,
        "ts": utc_ts(),
        "checked": len(results),
        "closed_registered": closed_registered,
        "remaining_open_trades": current_open_trade_count(),
        "results": results,
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
    pnl = pd.to_numeric(closed.get("pnl"), errors="coerce").fillna(0.0) if not closed.empty else pd.Series(dtype=float)
    return {
        "ok": True,
        "trades": int(len(df)),
        "closed_trades": int(len(closed)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).sum() / len(closed)) if len(closed) else None,
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "avg_pnl": float(pnl.mean()) if len(pnl) else None,
        "open_trades": current_open_trade_count(),
    }


@app.get("/export/closed_trades.xlsx")
def export_closed_trades_xlsx():
    df = read_closed_trades_df().copy()
    output_path = os.path.join(LOG_DIR, "closed_trades_m15_signal_approval_export.xlsx")
    if df.empty:
        df = pd.DataFrame(columns=["ts", "instrument", "side", "units_signed", "entry_price", "sl_price", "tp_price", "status", "pnl", "order_id", "reason", "pair_score"])
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="closed_trades")
    return FileResponse(output_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="closed_trades_m15_signal_approval_export.xlsx")


@app.get("/dashboard")
def dashboard():
    audit_df = read_audit_df()
    trades_df = read_trades_df()
    total_rows = len(audit_df)
    would_count = int(safe_bool_series(audit_df, "would_order").sum()) if not audit_df.empty else 0
    latest = audit_df.tail(50).to_html(index=False, escape=False) if not audit_df.empty else "<p>No audit data yet.</p>"
    trades = trades_df.tail(50).to_html(index=False, escape=False) if not trades_df.empty else "<p>No trade data yet.</p>"
    html = f"""
    <html>
    <head><title>FX M15 Signal Approval Dashboard</title><meta http-equiv=\"refresh\" content=\"15\"></head>
    <body style=\"font-family:Arial;padding:24px;\">
        <h1>FX M15 Signal-Conditioned Approval Dashboard</h1>
        <p>Total predictions: <b>{total_rows}</b> | Would order: <b>{would_count}</b> | Open trades: <b>{current_open_trade_count()}</b></p>
        <p><a href=\"/health\">Health</a> | <a href=\"/model_performance\">Model Performance</a> | <a href=\"/pnl_stats\">PnL</a></p>
        <h2>Latest Predictions</h2>{latest}
        <h2>Latest Trade Events</h2>{trades}
    </body>
    </html>
    """
    return HTMLResponse(content=html)
