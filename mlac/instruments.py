"""Instrument metadata + price/units helpers. Pure Python, no external deps.

Extracted verbatim from fx_api_sniper_CLperpair.py. The env-configurable constants are
represented here as module defaults (their production defaults); the live service still
reads them from the environment.
"""
from __future__ import annotations
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

PAIR_MAP = {
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

# Production defaults (env-overridable in the live service).
UNITS_JPY = 1000
UNITS_NON_JPY = 2000
MIN_UNITS_JPY = 100
MIN_UNITS_NON_JPY = 100
MAX_UNITS_JPY = 3000
MAX_UNITS_NON_JPY = 5000
MIN_ATR_JPY = 0.005
MIN_ATR_NON_JPY = 0.00005


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


def round_down_to_pip(price: float, pip: float) -> float:
    return math.floor(price / pip) * pip


def round_up_to_pip(price: float, pip: float) -> float:
    return math.ceil(price / pip) * pip
