import numpy as np
import pandas as pd
from mlac.labeling import (
    build_atr_direction_labels,
    build_horizon_resolved_labels,
    purged_split_index,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["mid_h", "mid_l", "mid_c", "atr14"])


def test_long_tp_first_labels_1():
    # entry close 100, atr 1, tp_atr=1.3 -> long TP at 101.3, short TP at 98.7.
    # Bar 1 spikes up to 102 (hits long TP) without hitting short TP.
    # NB: labeler only scores rows i < n - horizon_bars - 1, so pad with filler rows.
    df = _frame([
        [100, 100, 100, 1.0],       # i=0 entry
        [102.0, 101.0, 101.5, 1.0],  # bar 1: long TP hit
        [100, 100, 100, 1.0],
        [100, 100, 100, 1.0],
        [100, 100, 100, 1.0],
    ])
    out = build_atr_direction_labels(df, horizon_bars=2, tp_atr=1.3, sl_atr=1.0)
    assert out["y"].iloc[0] == 1
    assert out["label_reason"].iloc[0] == "long_tp_first"


def test_short_tp_first_labels_0():
    df = _frame([
        [100, 100, 100, 1.0],
        [100.5, 98.0, 98.5, 1.0],   # bar 1: dips to 98 -> short TP (98.7) hit first
        [100, 100, 100, 1.0],
        [100, 100, 100, 1.0],
        [100, 100, 100, 1.0],
    ])
    out = build_atr_direction_labels(df, horizon_bars=2, tp_atr=1.3, sl_atr=1.0)
    assert out["y"].iloc[0] == 0
    assert out["label_reason"].iloc[0] == "short_tp_first"


def test_neutral_stays_nan():
    # Price barely moves -> no barrier hit within horizon -> NaN.
    df = _frame([[100, 100, 100, 1.0]] + [[100.1, 99.9, 100.0, 1.0]] * 5)
    out = build_atr_direction_labels(df, horizon_bars=2, tp_atr=1.3, sl_atr=1.0)
    assert np.isnan(out["y"].iloc[0])


def test_horizon_resolved_labels_never_neutral():
    # A flat, no-barrier series: baseline drops it (NaN); horizon-resolved labels it by
    # the net move at the end. Here it drifts slightly UP over the horizon -> y=1.
    rows = [[100, 100, 100, 1.0]]
    rows += [[100.05 + 0.01 * k, 99.95 + 0.01 * k, 100.0 + 0.01 * k, 1.0] for k in range(1, 6)]
    df = _frame(rows)
    base = build_atr_direction_labels(df, horizon_bars=3, tp_atr=1.3, sl_atr=1.0)
    res = build_horizon_resolved_labels(df, horizon_bars=3, tp_atr=1.3, sl_atr=1.0)
    assert np.isnan(base["y"].iloc[0])          # baseline: neutral -> dropped
    assert res["y"].iloc[0] == 1                 # resolved: net up -> 1
    assert res["label_reason"].iloc[0] == "horizon_end"


def test_purged_split_removes_horizon_overlap():
    # 1000 rows, 20% valid -> split at 800; purge 8 -> train ends at 792.
    train_end, valid_start = purged_split_index(1000, 0.20, horizon_bars=8, embargo=0)
    assert valid_start == 800
    assert train_end == 792
    # embargo widens the gap
    train_end2, _ = purged_split_index(1000, 0.20, horizon_bars=8, embargo=5)
    assert train_end2 == 787
