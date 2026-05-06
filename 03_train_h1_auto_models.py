#!/usr/bin/env python3
"""
train_h1_auto_models.py

1H Forex auto-model trainer with recommended safe TCN policy.

Models:
1. LogisticRegression
2. ExtraTrees
3. LightGBM
4. CatBoost
5. Neural TCN challenger

Recommended approach:
- Tabular models compete normally.
- Neural TCN trains as challenger.
- TCN only becomes live model if it clearly beats the best tabular model.

Input folder default:
    data/oanda_h1_ba_live/

Expected CSV columns:
    time
    bid_o, bid_h, bid_l, bid_c
    ask_o, ask_h, ask_l, ask_c

Or:
    time
    mid_o, mid_h, mid_l, mid_c

Output:
    models/<PAIR>/best_model.pkl
    models/<PAIR>/best_model_type.json
    models/<PAIR>/thresholds.json
    models/<PAIR>/metrics.json
    models/<PAIR>/feature_columns.json
    models/<PAIR>/candidate_models/*.pkl
    models/<PAIR>/candidate_metrics.json

If TCN wins:
    models/<PAIR>/best_tcn.pt
    models/<PAIR>/tcn_imputer.pkl
    models/<PAIR>/tcn_scaler.pkl
    models/<PAIR>/tcn_config.json
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# Settings
# ============================================================

DATA_DIR = Path(os.getenv("DATA_DIR", "data/oanda_h1_ba_live"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

HORIZON_BARS = int(os.getenv("HORIZON_BARS", "8"))
SL_ATR = float(os.getenv("SL_ATR", "1.0"))
TP_ATR = float(os.getenv("TP_ATR", "1.3"))

VALID_FRACTION = float(os.getenv("VALID_FRACTION", "0.20"))
MIN_ROWS_AFTER_FEATURES = int(os.getenv("MIN_ROWS_AFTER_FEATURES", "1500"))
MIN_VALID_ROWS = int(os.getenv("MIN_VALID_ROWS", "250"))

MIN_GATE_TRADES = int(os.getenv("MIN_GATE_TRADES", "30"))
MIN_AUC_TO_TRADE = float(os.getenv("MIN_AUC_TO_TRADE", "0.52"))
MIN_PRECISION_TO_TRADE = float(os.getenv("MIN_PRECISION_TO_TRADE", "0.55"))
MIN_PAIR_SCORE_TO_TRADE = float(os.getenv("MIN_PAIR_SCORE_TO_TRADE", "0.30"))

GATE_GRID = [
    float(x)
    for x in os.getenv(
        "GATE_GRID",
        "0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70",
    ).split(",")
]

MARGIN_GRID = [
    float(x)
    for x in os.getenv(
        "MARGIN_GRID",
        "0.02,0.04,0.06,0.08,0.10,0.12",
    ).split(",")
]

# TCN challenger policy
TRAIN_TCN = os.getenv("TRAIN_TCN", "true").lower() == "true"
TCN_WINDOW = int(os.getenv("TCN_WINDOW", "48"))
TCN_EPOCHS = int(os.getenv("TCN_EPOCHS", "12"))
TCN_BATCH_SIZE = int(os.getenv("TCN_BATCH_SIZE", "256"))
TCN_LR = float(os.getenv("TCN_LR", "0.001"))

TCN_MIN_PRECISION_EDGE = float(os.getenv("TCN_MIN_PRECISION_EDGE", "0.03"))
TCN_MIN_AUC = float(os.getenv("TCN_MIN_AUC", "0.53"))
TCN_MIN_TRADES = int(os.getenv("TCN_MIN_TRADES", "30"))
TCN_MIN_PAIR_SCORE = float(os.getenv("TCN_MIN_PAIR_SCORE", "0.35"))


# ============================================================
# Optional imports
# ============================================================

HAS_LIGHTGBM = False
HAS_CATBOOST = False
HAS_TORCH = False

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def pip_size_from_pair(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, p))
    except Exception:
        return 0.5


def safe_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, p))
    except Exception:
        return 1.0


def read_pair_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    has_mid = all(c in df.columns for c in ["mid_o", "mid_h", "mid_l", "mid_c"])

    has_bid_ask = all(
        c in df.columns
        for c in [
            "bid_o",
            "bid_h",
            "bid_l",
            "bid_c",
            "ask_o",
            "ask_h",
            "ask_l",
            "ask_c",
        ]
    )

    if not has_mid and not has_bid_ask:
        raise ValueError(
            f"{path.name} must contain either mid_o/mid_h/mid_l/mid_c "
            f"or bid/ask OHLC columns."
        )

    if not has_mid:
        df["mid_o"] = (df["bid_o"] + df["ask_o"]) / 2.0
        df["mid_h"] = (df["bid_h"] + df["ask_h"]) / 2.0
        df["mid_l"] = (df["bid_l"] + df["ask_l"]) / 2.0
        df["mid_c"] = (df["bid_c"] + df["ask_c"]) / 2.0

    if "spread_c" not in df.columns and has_bid_ask:
        df["spread_c"] = df["ask_c"] - df["bid_c"]

    if "volume" not in df.columns:
        df["volume"] = 0

    return df


# ============================================================
# Indicators / features
# ============================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["mid_h"]
    low = df["mid_l"]
    close = df["mid_c"]
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


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["mid_h"]
    low = df["mid_l"]
    close = df["mid_c"]

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

    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr_val.replace(0, np.nan)
    )

    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr_val.replace(0, np.nan)
    )

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100

    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(20)


def add_features(df: pd.DataFrame, pair: str) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()

    ps = pip_size_from_pair(pair)
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

    df["ema20"] = ema(c, 20)
    df["ema50"] = ema(c, 50)
    df["ema100"] = ema(c, 100)
    df["ema200"] = ema(c, 200)

    df["dist_ema20_pips"] = (c - df["ema20"]) / ps
    df["dist_ema50_pips"] = (c - df["ema50"]) / ps
    df["dist_ema100_pips"] = (c - df["ema100"]) / ps
    df["dist_ema200_pips"] = (c - df["ema200"]) / ps

    df["ema20_slope"] = df["ema20"].diff(3) / ps
    df["ema50_slope"] = df["ema50"].diff(6) / ps
    df["ema200_slope"] = df["ema200"].diff(12) / ps

    df["rsi14"] = rsi(c, 14)
    df["rsi7"] = rsi(c, 7)

    df["atr14"] = atr(df, 14)
    df["atr14_pips"] = df["atr14"] / ps

    df["adx14"] = adx(df, 14)

    ema12 = ema(c, 12)
    ema26 = ema(c, 26)

    df["macd"] = ema12 - ema26
    df["macd_signal"] = ema(df["macd"], 9)
    df["macdh"] = df["macd"] - df["macd_signal"]
    df["macdh_pips"] = df["macdh"] / ps

    roll20 = c.rolling(20)
    df["bb_mid"] = roll20.mean()
    df["bb_std"] = roll20.std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    df["bb_width_pips"] = (df["bb_upper"] - df["bb_lower"]) / ps
    df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(
        0, np.nan
    )

    if "spread_c" in df.columns:
        df["spread_pips"] = df["spread_c"] / ps
    else:
        df["spread_pips"] = 0.0

    df["spread_atr"] = df["spread_pips"] / df["atr14_pips"].replace(0, np.nan)

    if "time" in df.columns:
        dt = pd.to_datetime(df["time"], utc=True, errors="coerce")

        df["hour_utc"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month

        df["hour_sin"] = np.sin(2 * np.pi * df["hour_utc"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_utc"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    else:
        df["hour_utc"] = 0
        df["day_of_week"] = 0
        df["month"] = 0
        df["hour_sin"] = 0
        df["hour_cos"] = 0
        df["dow_sin"] = 0
        df["dow_cos"] = 0

    df["trend_up"] = (df["ema20"] > df["ema50"]).astype(int)
    df["trend_down"] = (df["ema20"] < df["ema50"]).astype(int)
    df["price_above_ema200"] = (c > df["ema200"]).astype(int)

    feature_cols = [
        "ret1",
        "ret2",
        "ret3",
        "ret6",
        "ret12",
        "ret24",
        "range_pips",
        "body_pips",
        "upper_wick_pips",
        "lower_wick_pips",
        "dist_ema20_pips",
        "dist_ema50_pips",
        "dist_ema100_pips",
        "dist_ema200_pips",
        "ema20_slope",
        "ema50_slope",
        "ema200_slope",
        "rsi14",
        "rsi7",
        "atr14_pips",
        "adx14",
        "macdh_pips",
        "bb_width_pips",
        "bb_pos",
        "spread_pips",
        "spread_atr",
        "hour_utc",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "trend_up",
        "trend_down",
        "price_above_ema200",
        "volume",
    ]

    return df, feature_cols


# ============================================================
# ATR barrier labels
# ============================================================

def build_atr_direction_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    y = 1 means long TP was hit first.
    y = 0 means short TP was hit first.
    neutral / ambiguous rows remain NaN and are dropped.
    """

    df = df.copy()

    y = np.full(len(df), np.nan)
    label_reason = np.full(len(df), "neutral", dtype=object)

    highs = df["mid_h"].values
    lows = df["mid_l"].values
    closes = df["mid_c"].values
    atrs = df["atr14"].values

    n = len(df)

    for i in range(n - HORIZON_BARS - 1):
        entry = closes[i]
        a = atrs[i]

        if not np.isfinite(entry) or not np.isfinite(a) or a <= 0:
            continue

        long_tp = entry + TP_ATR * a
        long_sl = entry - SL_ATR * a

        short_tp = entry - TP_ATR * a
        short_sl = entry + SL_ATR * a

        for j in range(1, HORIZON_BARS + 1):
            h = highs[i + j]
            l = lows[i + j]

            long_tp_hit = h >= long_tp
            long_sl_hit = l <= long_sl

            short_tp_hit = l <= short_tp
            short_sl_hit = h >= short_sl

            # Skip ambiguous same-bar events.
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


# ============================================================
# Metrics / selection
# ============================================================

@dataclass
class ModelMetrics:
    model_name: str
    auc: float
    brier: float
    accuracy: float
    best_gate: float
    best_margin_gate: float
    precision_at_gate: float
    trades_at_gate: int
    long_rate: float
    pair_score: float
    tradable: bool


def evaluate_probabilities(
    model_name: str,
    y_true: np.ndarray,
    p_long: np.ndarray,
) -> ModelMetrics:
    y_true = np.asarray(y_true).astype(int)

    p_long = np.asarray(p_long).astype(float)
    p_long = np.clip(p_long, 1e-6, 1 - 1e-6)

    pred = (p_long >= 0.5).astype(int)

    auc = safe_auc(y_true, p_long)
    brier = safe_brier(y_true, p_long)
    acc = float(accuracy_score(y_true, pred))

    confidence = np.maximum(p_long, 1 - p_long)
    margin = np.abs(p_long - 0.5) * 2.0
    correct = (pred == y_true).astype(int)

    best_precision = 0.0
    best_gate = 0.56
    best_margin_gate = 0.06
    best_trades = 0

    for gate in GATE_GRID:
        for mgate in MARGIN_GRID:
            mask = (confidence >= gate) & (margin >= mgate)
            trades = int(mask.sum())

            if trades < MIN_GATE_TRADES:
                continue

            precision = float(correct[mask].mean())

            if precision > best_precision or (
                math.isclose(precision, best_precision) and trades > best_trades
            ):
                best_precision = precision
                best_gate = gate
                best_margin_gate = mgate
                best_trades = trades

    if best_trades == 0:
        fallback_gate = 0.56
        fallback_mgate = 0.06

        mask = (confidence >= fallback_gate) & (margin >= fallback_mgate)
        best_trades = int(mask.sum())
        best_precision = float(correct[mask].mean()) if best_trades > 0 else 0.0
        best_gate = fallback_gate
        best_margin_gate = fallback_mgate

    trade_sample_score = min(best_trades / max(MIN_GATE_TRADES, 1), 2.0) / 2.0
    auc_score = max(0.0, (auc - 0.50) / 0.10)
    precision_component = max(0.0, (best_precision - 0.50) / 0.15)

    pair_score = (
        0.55 * precision_component
        + 0.30 * auc_score
        + 0.15 * trade_sample_score
    )

    pair_score = float(max(0.0, min(pair_score, 1.0)))

    tradable = (
        auc >= MIN_AUC_TO_TRADE
        and best_precision >= MIN_PRECISION_TO_TRADE
        and pair_score >= MIN_PAIR_SCORE_TO_TRADE
        and best_trades >= MIN_GATE_TRADES
    )

    return ModelMetrics(
        model_name=model_name,
        auc=float(auc),
        brier=float(brier),
        accuracy=float(acc),
        best_gate=float(best_gate),
        best_margin_gate=float(best_margin_gate),
        precision_at_gate=float(best_precision),
        trades_at_gate=int(best_trades),
        long_rate=float(np.mean(p_long >= 0.5)),
        pair_score=float(pair_score),
        tradable=bool(tradable),
    )


def rank_key(m: ModelMetrics):
    return (
        m.precision_at_gate,
        m.pair_score,
        m.auc,
        -m.brier,
        m.trades_at_gate,
    )


def choose_best_with_safe_tcn_policy(
    metrics: List[ModelMetrics],
) -> Tuple[ModelMetrics, Optional[ModelMetrics], Optional[ModelMetrics], bool, Optional[str]]:
    """
    Returns:
        best_live_model
        best_tabular_model
        tcn_metric
        tcn_live_allowed
        tcn_block_reason
    """

    if not metrics:
        raise ValueError("No model metrics provided.")

    tabular_metrics = [m for m in metrics if m.model_name != "neural_tcn"]
    tcn_metrics = [m for m in metrics if m.model_name == "neural_tcn"]

    best_tabular = (
        sorted(tabular_metrics, key=rank_key, reverse=True)[0]
        if tabular_metrics
        else None
    )

    tcn_metric = tcn_metrics[0] if tcn_metrics else None

    if best_tabular is None and tcn_metric is not None:
        return tcn_metric, None, tcn_metric, True, None

    if best_tabular is not None and tcn_metric is None:
        return best_tabular, best_tabular, None, False, "tcn_not_trained_or_failed"

    assert best_tabular is not None

    precision_edge = tcn_metric.precision_at_gate - best_tabular.precision_at_gate

    tcn_live_allowed = (
        precision_edge >= TCN_MIN_PRECISION_EDGE
        and tcn_metric.auc >= TCN_MIN_AUC
        and tcn_metric.trades_at_gate >= TCN_MIN_TRADES
        and tcn_metric.pair_score >= TCN_MIN_PAIR_SCORE
    )

    if tcn_live_allowed:
        return tcn_metric, best_tabular, tcn_metric, True, None

    reasons = []

    if precision_edge < TCN_MIN_PRECISION_EDGE:
        reasons.append(
            f"precision_edge_too_small: {precision_edge:.4f} < {TCN_MIN_PRECISION_EDGE:.4f}"
        )

    if tcn_metric.auc < TCN_MIN_AUC:
        reasons.append(f"auc_too_low: {tcn_metric.auc:.4f} < {TCN_MIN_AUC:.4f}")

    if tcn_metric.trades_at_gate < TCN_MIN_TRADES:
        reasons.append(
            f"not_enough_tcn_trades: {tcn_metric.trades_at_gate} < {TCN_MIN_TRADES}"
        )

    if tcn_metric.pair_score < TCN_MIN_PAIR_SCORE:
        reasons.append(
            f"pair_score_too_low: {tcn_metric.pair_score:.4f} < {TCN_MIN_PAIR_SCORE:.4f}"
        )

    return best_tabular, best_tabular, tcn_metric, False, "; ".join(reasons)


# ============================================================
# Classical models
# ============================================================

def build_classical_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["logistic_regression"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models["extra_trees"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    if HAS_LIGHTGBM:
        models["lightgbm"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=700,
                        learning_rate=0.025,
                        num_leaves=31,
                        max_depth=-1,
                        min_child_samples=60,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.5,
                        objective="binary",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    else:
        print("WARNING: LightGBM not installed. Skipping LightGBM.")

    if HAS_CATBOOST:
        models["catboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=700,
                        depth=6,
                        learning_rate=0.025,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    else:
        print("WARNING: CatBoost not installed. Skipping CatBoost.")

    return models


def train_classical_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> Tuple[Dict[str, Any], List[ModelMetrics]]:
    models = build_classical_models()

    fitted: Dict[str, Any] = {}
    metrics: List[ModelMetrics] = []

    for name, model in models.items():
        print(f"    Training {name}...")

        try:
            model.fit(X_train, y_train)
            p = model.predict_proba(X_valid)[:, 1]

            m = evaluate_probabilities(name, y_valid, p)

            fitted[name] = model
            metrics.append(m)

            print(
                f"      AUC={m.auc:.4f} | "
                f"Precision@Gate={m.precision_at_gate:.4f} | "
                f"Trades={m.trades_at_gate} | "
                f"PairScore={m.pair_score:.3f} | "
                f"Tradable={m.tradable}"
            )

        except Exception as e:
            print(f"      SKIPPED {name}: {e}")

    return fitted, metrics


# ============================================================
# Neural TCN
# ============================================================

if HAS_TORCH:

    class SimpleTCN(nn.Module):
        def __init__(self, n_features: int, channels: int = 64, dropout: float = 0.20):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv1d(n_features, channels, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=8, dilation=4),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(channels, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x shape: batch, window, features
            x = x.transpose(1, 2)
            z = self.net(x)
            return self.head(z).squeeze(-1)


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for i in range(window - 1, len(X)):
        xs.append(X[i - window + 1 : i + 1])
        ys.append(y[i])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_tcn_model(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_valid_df: pd.DataFrame,
    y_valid: np.ndarray,
) -> Tuple[Optional[Any], Optional[ModelMetrics], Optional[Dict[str, Any]]]:
    if not TRAIN_TCN:
        print("    Skipping neural_tcn: TRAIN_TCN=false.")
        return None, None, None

    if not HAS_TORCH:
        print("    Skipping neural_tcn: torch not installed.")
        return None, None, None

    if len(X_train_df) < TCN_WINDOW * 5 or len(X_valid_df) < TCN_WINDOW * 2:
        print("    Skipping neural_tcn: not enough rows for sequence training.")
        return None, None, None

    print("    Training neural_tcn...")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_df)
    X_valid_imp = imputer.transform(X_valid_df)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)

    Xtr_seq, ytr_seq = make_sequences(X_train_scaled, y_train, TCN_WINDOW)
    Xva_seq, yva_seq = make_sequences(X_valid_scaled, y_valid, TCN_WINDOW)

    # Speed control for large datasets
    MAX_TCN_TRAIN_SEQUENCES = int(os.getenv("MAX_TCN_TRAIN_SEQUENCES", "12000"))

    if len(Xtr_seq) > MAX_TCN_TRAIN_SEQUENCES:
        Xtr_seq = Xtr_seq[-MAX_TCN_TRAIN_SEQUENCES:]
        ytr_seq = ytr_seq[-MAX_TCN_TRAIN_SEQUENCES:]
        print(f"    TCN training capped to last {MAX_TCN_TRAIN_SEQUENCES} sequences.")

    if len(Xtr_seq) < 200 or len(Xva_seq) < 50:
        print("    Skipping neural_tcn: too few sequences.")
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.tensor(Xtr_seq, dtype=torch.float32),
        torch.tensor(ytr_seq, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TCN_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    model = SimpleTCN(n_features=Xtr_seq.shape[-1]).to(device)

    pos = float(np.sum(ytr_seq == 1))
    neg = float(np.sum(ytr_seq == 0))
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TCN_LR, weight_decay=1e-4)

    best_auc = -1.0
    best_state = None

    Xva_tensor = torch.tensor(Xva_seq, dtype=torch.float32).to(device)

    for epoch in range(1, TCN_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)

        model.eval()

        with torch.no_grad():
            logits = model(Xva_tensor)
            p = torch.sigmoid(logits).detach().cpu().numpy()

        epoch_auc = safe_auc(yva_seq.astype(int), p)

        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"      TCN epoch {epoch}/{TCN_EPOCHS} | "
            f"loss={total_loss / max(len(train_ds), 1):.5f} | "
            f"valid_auc={epoch_auc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        logits = model(Xva_tensor)
        p = torch.sigmoid(logits).detach().cpu().numpy()

    m = evaluate_probabilities("neural_tcn", yva_seq.astype(int), p)

    print(
        f"      AUC={m.auc:.4f} | "
        f"Precision@Gate={m.precision_at_gate:.4f} | "
        f"Trades={m.trades_at_gate} | "
        f"PairScore={m.pair_score:.3f} | "
        f"Tradable={m.tradable}"
    )

    bundle = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "window": TCN_WINDOW,
        "n_features": Xtr_seq.shape[-1],
        "device": str(device),
    }

    return model, m, bundle


# ============================================================
# Per-pair trainer
# ============================================================

def train_pair(csv_path: Path) -> Dict[str, Any]:
    pair = csv_path.stem.upper()

    print("\n==============================")
    print(f"Training pair: {pair}")
    print(f"CSV: {csv_path}")
    print("==============================")

    pair_dir = MODELS_DIR / pair
    ensure_dir(pair_dir)

    raw = read_pair_csv(csv_path)
    df, feature_cols = add_features(raw, pair)
    df = build_atr_direction_labels(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    needed = feature_cols + ["y"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    if len(df) < MIN_ROWS_AFTER_FEATURES:
        result = {
            "pair": pair,
            "ok": False,
            "reason": f"not_enough_rows_after_features: {len(df)}",
        }
        save_json(pair_dir / "metrics.json", result)
        print(f"  SKIPPED: {result['reason']}")
        return result

    X = df[feature_cols].copy()
    y = df["y"].astype(int).values

    if len(np.unique(y)) < 2:
        result = {
            "pair": pair,
            "ok": False,
            "reason": "only_one_label_class_after_filtering",
        }
        save_json(pair_dir / "metrics.json", result)
        print(f"  SKIPPED: {result['reason']}")
        return result

    split_idx = int(len(df) * (1.0 - VALID_FRACTION))

    X_train = X.iloc[:split_idx].copy()
    y_train = y[:split_idx]

    X_valid = X.iloc[split_idx:].copy()
    y_valid = y[split_idx:]

    if len(X_valid) < MIN_VALID_ROWS:
        result = {
            "pair": pair,
            "ok": False,
            "reason": f"not_enough_valid_rows: {len(X_valid)}",
        }
        save_json(pair_dir / "metrics.json", result)
        print(f"  SKIPPED: {result['reason']}")
        return result

    print(f"  Total rows: {len(df)}")
    print(f"  Train rows: {len(X_train)}")
    print(f"  Valid rows: {len(X_valid)}")
    print(f"  Long-label rate: {float(np.mean(y)):.3f}")
    print(f"  Feature count: {len(feature_cols)}")

    fitted_models, metric_list = train_classical_models(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    tcn_model, tcn_metrics, tcn_bundle = train_tcn_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    if tcn_metrics is not None:
        metric_list.append(tcn_metrics)

    if not metric_list:
        result = {
            "pair": pair,
            "ok": False,
            "reason": "no_models_successfully_trained",
        }
        save_json(pair_dir / "metrics.json", result)
        print(f"  SKIPPED: {result['reason']}")
        return result

    (
        best,
        best_tabular,
        tcn_metric,
        tcn_live_allowed,
        tcn_block_reason,
    ) = choose_best_with_safe_tcn_policy(metric_list)

    print("\n  Model leaderboard:")
    for m in sorted(metric_list, key=rank_key, reverse=True):
        print(
            f"    {m.model_name:20s} | "
            f"Precision={m.precision_at_gate:.4f} | "
            f"AUC={m.auc:.4f} | "
            f"Brier={m.brier:.4f} | "
            f"Trades={m.trades_at_gate:4d} | "
            f"Score={m.pair_score:.3f} | "
            f"Tradable={m.tradable}"
        )

    print("\n  Winner selected by safe policy:")
    print(f"    Best live model: {best.model_name}")
    print(f"    Tradable: {best.tradable}")
    print(f"    Pair score: {best.pair_score:.3f}")
    print(f"    Gate: {best.best_gate:.2f}")
    print(f"    Margin gate: {best.best_margin_gate:.2f}")
    print(f"    TCN live allowed: {tcn_live_allowed}")
    print(f"    TCN block reason: {tcn_block_reason}")

    save_json(pair_dir / "feature_columns.json", feature_cols)

    thresholds = {
        "pair": pair,
        "model_name": best.model_name,
        "live_model_type": best.model_name,
        "gate": best.best_gate,
        "margin_gate": best.best_margin_gate,
        "min_pair_score_to_trade": MIN_PAIR_SCORE_TO_TRADE,
        "pair_score": best.pair_score,
        "tradable": best.tradable,
        "horizon_bars": HORIZON_BARS,
        "sl_atr": SL_ATR,
        "tp_atr": TP_ATR,
        "tcn_policy": {
            "mode": "challenger_until_clear_winner",
            "tcn_min_precision_edge": TCN_MIN_PRECISION_EDGE,
            "tcn_min_auc": TCN_MIN_AUC,
            "tcn_min_trades": TCN_MIN_TRADES,
            "tcn_min_pair_score": TCN_MIN_PAIR_SCORE,
            "tcn_live_allowed": tcn_live_allowed,
            "tcn_block_reason": tcn_block_reason,
            "best_tabular_model": best_tabular.model_name if best_tabular else None,
            "best_tabular_precision_at_gate": (
                best_tabular.precision_at_gate if best_tabular else None
            ),
            "best_tabular_auc": best_tabular.auc if best_tabular else None,
            "best_tabular_pair_score": best_tabular.pair_score if best_tabular else None,
            "tcn_precision_at_gate": tcn_metric.precision_at_gate if tcn_metric else None,
            "tcn_auc": tcn_metric.auc if tcn_metric else None,
            "tcn_pair_score": tcn_metric.pair_score if tcn_metric else None,
            "tcn_trades_at_gate": tcn_metric.trades_at_gate if tcn_metric else None,
        },
    }

    save_json(pair_dir / "thresholds.json", thresholds)
    save_json(pair_dir / "best_model_type.json", {"model_type": best.model_name})

    all_metrics = {
        "pair": pair,
        "ok": True,
        "rows_total": int(len(df)),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "label_long_rate": float(np.mean(y)),
        "best_model": best.model_name,
        "best_live_model": best.model_name,
        "best": asdict(best),
        "best_tabular_model": best_tabular.model_name if best_tabular else None,
        "best_tabular": asdict(best_tabular) if best_tabular else None,
        "tcn_metric": asdict(tcn_metric) if tcn_metric else None,
        "tcn_live_allowed": tcn_live_allowed,
        "tcn_block_reason": tcn_block_reason,
        "all_models": [asdict(m) for m in metric_list],
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "tradable": best.tradable,
    }

    save_json(pair_dir / "metrics.json", all_metrics)


    # ============================================================
    # Save all candidate tabular models for server fallback / shadow logic
    # ============================================================
    candidate_dir = pair_dir / "candidate_models"
    ensure_dir(candidate_dir)

    candidate_metrics = {m.model_name: asdict(m) for m in metric_list}
    save_json(pair_dir / "candidate_metrics.json", candidate_metrics)

    for model_name, fitted_model in fitted_models.items():
        try:
            joblib.dump(fitted_model, candidate_dir / f"{model_name}.pkl")
            print(f"  Saved candidate model: {model_name}")
        except Exception as e:
            print(f"  WARNING: could not save candidate model {model_name}: {e}")

    # Save the selected live model.
    if best.model_name == "neural_tcn":
        if tcn_bundle is None:
            raise RuntimeError("TCN selected but TCN bundle is missing.")

        torch.save(
            {
                "state_dict": tcn_bundle["model"].state_dict(),
                "n_features": tcn_bundle["n_features"],
                "window": tcn_bundle["window"],
            },
            pair_dir / "best_tcn.pt",
        )

        joblib.dump(tcn_bundle["imputer"], pair_dir / "tcn_imputer.pkl")
        joblib.dump(tcn_bundle["scaler"], pair_dir / "tcn_scaler.pkl")

        save_json(
            pair_dir / "tcn_config.json",
            {
                "window": tcn_bundle["window"],
                "n_features": tcn_bundle["n_features"],
                "model_class": "SimpleTCN",
            },
        )
    else:
        best_model = fitted_models[best.model_name]
        joblib.dump(best_model, pair_dir / "best_model.pkl")

    # Also save TCN as challenger if it trained, even if it did not win.
    if tcn_bundle is not None and tcn_metric is not None:
        try:
            torch.save(
                {
                    "state_dict": tcn_bundle["model"].state_dict(),
                    "n_features": tcn_bundle["n_features"],
                    "window": tcn_bundle["window"],
                },
                pair_dir / "challenger_tcn.pt",
            )

            joblib.dump(tcn_bundle["imputer"], pair_dir / "challenger_tcn_imputer.pkl")
            joblib.dump(tcn_bundle["scaler"], pair_dir / "challenger_tcn_scaler.pkl")

            save_json(
                pair_dir / "challenger_tcn_config.json",
                {
                    "window": tcn_bundle["window"],
                    "n_features": tcn_bundle["n_features"],
                    "model_class": "SimpleTCN",
                    "live_allowed": tcn_live_allowed,
                    "block_reason": tcn_block_reason,
                },
            )
        except Exception as e:
            print(f"  WARNING: could not save challenger TCN: {e}")

    return all_metrics


# ============================================================
# Main
# ============================================================

def find_csv_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {data_dir}\n"
            f"Create it and put your 1H CSV files inside."
        )

    files = sorted([p for p in data_dir.glob("*.csv") if not p.name.startswith(".")])

    return files


def main() -> None:
    print("SCRIPT STARTED")
    print("========================================")
    print("1H Forex Auto Model Trainer")
    print("========================================")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"HORIZON_BARS: {HORIZON_BARS}")
    print(f"SL_ATR: {SL_ATR}")
    print(f"TP_ATR: {TP_ATR}")
    print(f"TRAIN_TCN: {TRAIN_TCN}")
    print(f"TCN_WINDOW: {TCN_WINDOW}")
    print(f"TCN_EPOCHS: {TCN_EPOCHS}")
    print(f"TCN_MIN_PRECISION_EDGE: {TCN_MIN_PRECISION_EDGE}")
    print(f"TCN_MIN_AUC: {TCN_MIN_AUC}")
    print(f"TCN_MIN_TRADES: {TCN_MIN_TRADES}")
    print(f"TCN_MIN_PAIR_SCORE: {TCN_MIN_PAIR_SCORE}")
    print(f"LightGBM available: {HAS_LIGHTGBM}")
    print(f"CatBoost available: {HAS_CATBOOST}")
    print(f"Torch available: {HAS_TORCH}")
    print("========================================")

    ensure_dir(MODELS_DIR)

    csv_files = find_csv_files(DATA_DIR)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}.\n"
            f"Expected files like EURUSD.csv, GBPUSD.csv, USDJPY.csv."
        )

    print(f"Found {len(csv_files)} CSV files.")

    summary = []

    for csv_path in csv_files:
        try:
            result = train_pair(csv_path)
            summary.append(result)
        except Exception as e:
            pair = csv_path.stem.upper()
            print(f"\nERROR training {pair}: {e}")

            pair_dir = MODELS_DIR / pair
            ensure_dir(pair_dir)

            result = {
                "pair": pair,
                "ok": False,
                "reason": str(e),
            }

            save_json(pair_dir / "metrics.json", result)
            summary.append(result)

    save_json(MODELS_DIR / "training_summary.json", summary)

    print("\n========================================")
    print("Training complete")
    print("========================================")

    for item in summary:
        pair = item.get("pair", "UNKNOWN")
        ok = item.get("ok", False)

        if not ok:
            print(f"{pair:8s} | FAILED/BLOCKED | {item.get('reason')}")
            continue

        best_model = item.get("best_live_model")
        tradable = item.get("tradable")
        best = item.get("best", {})

        score = best.get("pair_score", 0.0)
        precision = best.get("precision_at_gate", 0.0)
        auc = best.get("auc", 0.0)
        trades = best.get("trades_at_gate", 0)

        tcn_live_allowed = item.get("tcn_live_allowed")
        tcn_block_reason = item.get("tcn_block_reason")

        print(
            f"{pair:8s} | "
            f"{best_model:18s} | "
            f"Tradable={tradable} | "
            f"Score={score:.3f} | "
            f"Precision={precision:.3f} | "
            f"AUC={auc:.3f} | "
            f"Trades={trades} | "
            f"TCN_live={tcn_live_allowed} | "
            f"TCN_reason={tcn_block_reason}"
        )


if __name__ == "__main__":
    main()
