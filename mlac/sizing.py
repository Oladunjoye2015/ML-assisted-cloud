"""Position sizing + SL/TP price computation. Extracted verbatim from the monolith.

Pure functions: given inputs, deterministic outputs. `compute_units_dynamic` takes its
risk config explicitly (defaults match production) so it can be tested without the env.
"""
from __future__ import annotations
from typing import Optional, Tuple

from . import instruments as I

# Production risk defaults (env-overridable in the live service).
USE_EQUITY_SIZING = True
RISK_PCT = 0.0015


def compute_units_dynamic(
    instrument: str,
    sl_pips: float,
    avg_auc: float,
    pair_score: float,
    equity_used: float,
    force_units_abs: Optional[int] = None,
    use_equity_sizing: bool = USE_EQUITY_SIZING,
    risk_pct: float = RISK_PCT,
) -> int:
    if force_units_abs is not None:
        return max(1, abs(int(force_units_abs)))
    if sl_pips is None or sl_pips <= 0:
        return 0

    base = I.base_units_for_instrument(instrument)
    if use_equity_sizing:
        risk_cap = equity_used * risk_pct
        risk_per_1000 = float(sl_pips) * I.pip_value_per_1000(instrument)
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

    return min(I.max_units_for_instrument(instrument), max(I.min_units_for_instrument(instrument), base))


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

    pip = I.instrument_pip_size(instrument)
    atrv = max(float(atr14), pip)
    sl_dist = max(sl_atr * atrv, min_dist_pips * pip)
    tp_dist = max(tp_atr * atrv, min_dist_pips * pip)

    if side == "BUY":
        sl_price = I.round_down_to_pip(mid_c - sl_dist, pip)
        tp_price = I.round_up_to_pip(mid_c + tp_dist, pip)
        if sl_price >= mid_c:
            sl_price = I.round_down_to_pip(mid_c - (min_dist_pips * pip), pip)
        if tp_price <= mid_c:
            tp_price = I.round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
    else:
        sl_price = I.round_up_to_pip(mid_c + sl_dist, pip)
        tp_price = I.round_down_to_pip(mid_c - tp_dist, pip)
        if sl_price <= mid_c:
            sl_price = I.round_up_to_pip(mid_c + (min_dist_pips * pip), pip)
        if tp_price >= mid_c:
            tp_price = I.round_down_to_pip(mid_c - (min_dist_pips * pip), pip)

    sl_str = I.format_oanda_price(sl_price, instrument)
    tp_str = I.format_oanda_price(tp_price, instrument)
    mid_str = I.format_oanda_price(mid_c, instrument)

    sl_price_f = float(sl_str)
    tp_price_f = float(tp_str)
    mid_c_f = float(mid_str)

    sl_pips = abs(mid_c_f - sl_price_f) / pip
    tp_pips = abs(tp_price_f - mid_c_f) / pip
    return float(sl_pips), float(tp_pips), sl_str, tp_str
