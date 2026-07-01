"""ATR triple-barrier direction labels. Extracted verbatim from 03_train_h1_auto_models.py.

y = 1  -> long TP was hit before short TP (up move resolved first)
y = 0  -> short TP was hit first
NaN    -> neutral/ambiguous within the horizon (dropped by the trainer)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HORIZON_BARS = 8
TP_ATR = 1.3
SL_ATR = 1.0


def build_atr_direction_labels(
    df: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
    tp_atr: float = TP_ATR,
    sl_atr: float = SL_ATR,
) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    y = np.full(n, np.nan)
    label_reason = np.full(n, "neutral", dtype=object)

    highs = df["mid_h"].values
    lows = df["mid_l"].values
    closes = df["mid_c"].values
    atrs = df["atr14"].values

    for i in range(n - horizon_bars - 1):
        entry = closes[i]
        a = atrs[i]
        if not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
            continue

        long_tp = entry + tp_atr * a
        long_sl = entry - sl_atr * a
        short_tp = entry - tp_atr * a
        short_sl = entry + sl_atr * a

        for j in range(1, horizon_bars + 1):
            h = highs[i + j]
            l = lows[i + j]

            long_tp_hit = h >= long_tp
            long_sl_hit = l <= long_sl
            short_tp_hit = l <= short_tp
            short_sl_hit = h >= short_sl

            if long_tp_hit and long_sl_hit:
                break
            if short_tp_hit and short_sl_hit:
                break
            if long_tp_hit and not short_tp_hit:
                y[i] = 1
                label_reason[i] = "long_tp_first"
                break
            if short_tp_hit and not long_tp_hit:
                y[i] = 0
                label_reason[i] = "short_tp_first"
                break
            if long_sl_hit and short_sl_hit:
                break

    df["y"] = y
    df["label_reason"] = label_reason
    return df


def build_horizon_resolved_labels(
    df: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
    tp_atr: float = TP_ATR,
    sl_atr: float = SL_ATR,
) -> pd.DataFrame:
    """More realistic label: same ATR barriers, but instead of DROPPING bars where no
    barrier is hit within the horizon (which biases training vs. live — audit C3), resolve
    them by the sign of the net move at the horizon end. Every scored bar gets a label, so
    the training distribution matches what the live server actually scores.

    Empirically (see tools/eval_labels.py) this was the least-negative of the tested label
    variants and removes the selection bias, though on the current feature set it does NOT
    by itself reach a positive out-of-sample edge.
    """
    df = df.copy()
    n = len(df)
    y = np.full(n, np.nan)
    reason = np.full(n, "neutral", dtype=object)
    highs, lows, closes, atrs = (
        df["mid_h"].values, df["mid_l"].values, df["mid_c"].values, df["atr14"].values
    )
    for i in range(n - horizon_bars - 1):
        entry, a = closes[i], atrs[i]
        if not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
            continue
        long_tp, long_sl = entry + tp_atr * a, entry - sl_atr * a
        short_tp, short_sl = entry - tp_atr * a, entry + sl_atr * a
        decided = False
        for j in range(1, horizon_bars + 1):
            h, l = highs[i + j], lows[i + j]
            if h >= long_tp and l <= long_sl:
                y[i] = 0; reason[i] = "ambiguous_down"; decided = True; break
            if l <= short_tp and h >= short_sl:
                y[i] = 1; reason[i] = "ambiguous_up"; decided = True; break
            if h >= long_tp and not (l <= short_tp):
                y[i] = 1; reason[i] = "long_tp_first"; decided = True; break
            if l <= short_tp and not (h >= long_tp):
                y[i] = 0; reason[i] = "short_tp_first"; decided = True; break
        if not decided:
            y[i] = 1 if closes[i + horizon_bars] > entry else 0
            reason[i] = "horizon_end"
    df["y"] = y
    df["label_reason"] = reason
    return df


def purged_split_index(n: int, valid_fraction: float, horizon_bars: int = HORIZON_BARS, embargo: int = 0):
    """AUDIT FIX (C2): chronological split with the last `horizon_bars` (+embargo) training
    rows purged, because their forward-looking labels overlap the validation window.
    Returns (train_end, valid_start) index boundaries."""
    split_idx = int(n * (1.0 - valid_fraction))
    train_end = max(0, split_idx - horizon_bars - embargo)
    return train_end, split_idx
