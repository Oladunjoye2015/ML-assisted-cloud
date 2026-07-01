import math
from mlac import sizing as S


def test_sl_tp_buy_basic():
    sl_pips, tp_pips, sl_str, tp_str = S.compute_sl_tp_prices(
        "BUY", 1.08240, 0.00110, "EUR_USD", sl_atr=0.9, tp_atr=1.4
    )
    assert sl_str == "1.08140" and tp_str == "1.08400"
    assert math.isclose(sl_pips, 10.0, abs_tol=1e-6)
    assert math.isclose(tp_pips, 16.0, abs_tol=1e-6)


def test_sl_tp_sell_is_mirror():
    sl_pips, tp_pips, sl_str, tp_str = S.compute_sl_tp_prices(
        "SELL", 1.08240, 0.00110, "EUR_USD", sl_atr=0.9, tp_atr=1.4
    )
    assert float(sl_str) > 1.08240   # SL above entry for a short
    assert float(tp_str) < 1.08240   # TP below entry for a short
    assert sl_pips > 0 and tp_pips > 0


def test_sl_tp_enforces_min_distance():
    # Tiny ATR -> min_dist_pips floor (5 pips) kicks in.
    sl_pips, tp_pips, _, _ = S.compute_sl_tp_prices(
        "BUY", 1.10000, 1e-9, "EUR_USD", sl_atr=0.9, tp_atr=1.4, min_dist_pips=5.0
    )
    assert sl_pips >= 5.0 - 1e-6 and tp_pips >= 5.0 - 1e-6   # float tolerance on pip math


def test_sl_tp_invalid_side_returns_none():
    assert S.compute_sl_tp_prices("NONE", 1.1, 0.001, "EUR_USD", 1.0, 1.3) == (None, None, None, None)


def test_units_force_override_wins():
    assert S.compute_units_dynamic("EUR_USD", 10, 0.56, 0.7, 200, force_units_abs=-777) == 777


def test_units_zero_when_no_stop():
    assert S.compute_units_dynamic("EUR_USD", 0, 0.56, 0.7, 200) == 0
    assert S.compute_units_dynamic("EUR_USD", -1, 0.56, 0.7, 200) == 0


def test_units_respect_bounds():
    # Very tight stop with equity sizing would blow past the max -> clamp to max.
    u = S.compute_units_dynamic("EUR_USD", 0.1, 0.60, 0.9, 100000, use_equity_sizing=True)
    assert u <= 5000
    # Never below the per-instrument minimum.
    u2 = S.compute_units_dynamic("EUR_USD", 5000, 0.50, 0.40, 1.0, use_equity_sizing=True)
    assert u2 >= 100
