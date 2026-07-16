#!/usr/bin/env python3
"""Research fixed live gates per pair with purged walk-forward predictions.

This is intentionally a research tool, not a registry writer. It answers:
"If we predeclare a fixed confidence/margin gate, does this pair/model show
positive out-of-sample expectancy with enough trades?"

The live registry should only be loosened after reviewing this output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRAIN_FILE = REPO / "03_train_h1_auto_models.py"
OUT_DIR = REPO / "tools" / "eval"


def load_trainmod():
    spec = importlib.util.spec_from_file_location("trainmod", TRAIN_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainmod"] = mod
    spec.loader.exec_module(mod)
    return mod


def auc_score(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, n, replace=True).mean() for _ in range(1000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def oos_predictions(pair: str, model_name: str, T, args):
    path = Path(args.data_dir) / f"{pair}.csv"
    raw = T.read_pair_csv(path)
    df, fcols = T.add_features(raw, pair)
    df = T.build_atr_direction_labels(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fcols + ["y"]).reset_index(drop=True)
    if len(df) < T.MIN_ROWS_AFTER_FEATURES:
        raise ValueError(f"not enough feature rows: {len(df)}")

    X = df[fcols]
    y = df["y"].astype(int).values
    spread_atr = df["spread_atr"].values.astype(float) if "spread_atr" in df else np.zeros(len(df))
    n = len(df)
    test_start = int(n * (1 - args.test_fraction))
    test_len = (n - test_start) // args.folds
    if test_len < 80:
        raise ValueError(f"too few test rows per fold: {test_len}")

    preds, ys, costs = [], [], []
    horizon = T.HORIZON_BARS
    for fold in range(args.folds):
        start = test_start + fold * test_len
        end = start + test_len if fold < args.folds - 1 else n
        train_end = start - horizon - args.embargo
        if train_end < 300:
            continue
        if len(np.unique(y[:train_end])) < 2:
            continue

        model = T.build_classical_models()[model_name]
        model.fit(X.iloc[:train_end], y[:train_end])
        preds.append(model.predict_proba(X.iloc[start:end])[:, 1])
        ys.append(y[start:end])
        costs.append(spread_atr[start:end])

    if not preds:
        raise ValueError("no valid folds")
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "lightgbm"])
    parser.add_argument("--gates", nargs="+", type=float, default=[0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.00, 0.02, 0.04])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.40)
    parser.add_argument("--embargo", type=int, default=0)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--data-dir", default=str(REPO / "oanda_h1_ba_live"))
    parser.add_argument("--out", default=str(OUT_DIR / "pair_gate_research.json"))
    args = parser.parse_args()

    T = load_trainmod()
    rows = []
    for pair in [p.upper() for p in args.pairs]:
        for model_name in args.models:
            try:
                p, y, spread_atr = oos_predictions(pair, model_name, T, args)
            except Exception as exc:
                rows.append({"pair": pair, "model": model_name, "ok": False, "reason": str(exc)})
                continue

            conf = np.maximum(p, 1 - p)
            margin = np.abs(p - 0.5) * 2
            correct = ((p >= 0.5).astype(int) == y)
            base = {
                "pair": pair,
                "model": model_name,
                "ok": True,
                "oos_rows": int(len(y)),
                "oos_auc": auc_score(y, p),
            }
            for gate in args.gates:
                for margin_gate in args.margins:
                    mask = (conf >= gate) & (margin >= margin_gate)
                    trades = int(mask.sum())
                    if trades < 10:
                        continue
                    pnl = np.where(correct[mask], T.TP_ATR, -T.SL_ATR) - spread_atr[mask]
                    ci_lo, ci_hi = bootstrap_ci(pnl, seed=42)
                    rows.append({
                        **base,
                        "gate": gate,
                        "margin_gate": margin_gate,
                        "n_trades": trades,
                        "precision": float(correct[mask].mean()),
                        "expectancy_R": float(pnl.mean()),
                        "exp_ci_lo": ci_lo,
                        "exp_ci_hi": ci_hi,
                        "clears_live_bar": bool(trades >= args.min_trades and ci_lo > 0),
                    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.write_text(json.dumps(rows, indent=2))

    eligible = [r for r in rows if r.get("clears_live_bar")]
    eligible.sort(key=lambda r: (r["exp_ci_lo"], r["n_trades"]), reverse=True)
    print(f"Wrote {out.relative_to(REPO)}")
    print(f"{'PAIR':8} {'MODEL':18} {'AUC':>7} {'GATE':>5} {'MARG':>5} {'N':>5} {'PREC':>6} {'EXP_R':>7} {'CI_LO':>7}")
    for r in eligible[:20]:
        print(
            f"{r['pair']:8} {r['model']:18} {r['oos_auc']:7.3f} {r['gate']:5.2f} "
            f"{r['margin_gate']:5.2f} {r['n_trades']:5d} {r['precision']:6.3f} "
            f"{r['expectancy_R']:7.3f} {r['exp_ci_lo']:7.3f}"
        )


if __name__ == "__main__":
    main()
