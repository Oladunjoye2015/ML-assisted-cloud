from mlac import instruments as I


def test_normalize_pair_accepts_known_and_rejects_unknown():
    assert I.normalize_pair("eur/usd") == "EURUSD"
    assert I.normalize_pair("EUR_GBP") == "EURGBP"
    assert I.normalize_pair("usdjpy") == "USDJPY"
    assert I.normalize_pair("BTCUSD") is None
    assert I.normalize_pair("") is None


def test_jpy_detection_and_precision():
    assert I.instrument_is_jpy("USD_JPY") is True
    assert I.instrument_is_jpy("EUR_GBP") is False
    assert I.instrument_precision("USD_JPY") == 3
    assert I.instrument_precision("EUR_GBP") == 5
    assert I.instrument_pip_size("USD_JPY") == 0.01
    assert I.instrument_pip_size("EUR_GBP") == 0.0001


def test_units_bounds():
    assert I.base_units_for_instrument("USD_JPY") == 1000
    assert I.base_units_for_instrument("EUR_GBP") == 2000
    assert I.max_units_for_instrument("EUR_GBP") == 5000
    assert I.min_units_for_instrument("USD_JPY") == 100


def test_price_formatting_rounds_half_up_to_precision():
    assert I.format_oanda_price(1.081405, "EUR_USD") == "1.08141"
    assert I.format_oanda_price(157.3846, "USD_JPY") == "157.385"


def test_pip_rounding_helpers():
    import math
    assert math.isclose(I.round_down_to_pip(1.081419, 0.0001), 1.0814, abs_tol=1e-9)
    assert math.isclose(I.round_up_to_pip(1.081401, 0.0001), 1.0815, abs_tol=1e-9)
