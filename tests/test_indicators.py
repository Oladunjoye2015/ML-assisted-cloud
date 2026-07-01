import numpy as np
import pandas as pd
from mlac import indicators as ind


def test_rsi_high_on_uptrend_with_pullbacks():
    # Mostly gains with small periodic pullbacks (pure monotonic gives NaN by design,
    # since there are no losses to divide by — this mirrors the trainer's rsi()).
    deltas = np.array([1.0] * 60)
    deltas[::7] = -0.3
    close = pd.Series(100 + np.cumsum(deltas))
    rsi = ind.rsi(close, 14).iloc[-1]
    assert rsi > 70


def test_rsi_low_on_downtrend_with_pullbacks():
    deltas = np.array([-1.0] * 60)
    deltas[::7] = 0.3
    close = pd.Series(200 + np.cumsum(deltas))
    rsi = ind.rsi(close, 14).iloc[-1]
    assert rsi < 30


def test_atr_is_positive_and_tracks_range():
    n = 60
    df = pd.DataFrame({
        "mid_h": np.full(n, 101.0),
        "mid_l": np.full(n, 99.0),
        "mid_c": np.full(n, 100.0),
    })
    atr = ind.atr(df, 14).iloc[-1]
    assert 1.5 < atr < 2.5   # true range ~2.0


def test_ema_between_min_and_max():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    e = ind.ema(s, 3).iloc[-1]
    assert 1.0 < e < 5.0


def test_adx_finite_for_trending_series():
    n = 80
    up = np.linspace(100, 120, n)
    df = pd.DataFrame({"mid_h": up + 0.5, "mid_l": up - 0.5, "mid_c": up})
    adx = ind.adx(df, 14).iloc[-1]
    assert np.isfinite(adx) and adx > 0
