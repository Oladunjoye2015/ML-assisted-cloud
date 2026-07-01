#!/usr/bin/env python3
"""
Purged, embargoed walk-forward evaluator — the honest replacement for the single
in-sample holdout used by 03_train_h1_auto_models.py.

Run in your venv (needs scikit-learn etc.):

    python tools/walk_forward_eval.py                 # all pairs in DATA_DIR
    python tools/walk_forward_eval.py --pairs EURUSD USDJPY --folds 6

What it fixes versus the current trainer
----------------------------------------
* C2 leakage: PURGES the last HORIZON_BARS training rows each fold (their ATR-barrier
  labels look into the test window) and applies an EMBARGO after the split.
* C1 overfit gate: selects the conf/margin gate on the TRAIN portion of each fold and
  applies it to the untouched TEST portion — no in-sample peeking.
* Adds a cost-aware P&L in ATR-R units (win=+TP_ATR, loss=-SL_ATR, minus spread) with a
  bootstrap 95% CI, so "edge" means "expectancy CI lower bound > 0", not "high AUC".
* C3 selection bias: reports the % of neutral bars dropped, which the live server scores
  anyway.

It reuses the REAL feature/label/model code from 03_train_h1_auto_models.py via importlib
so the evaluation reflects the deployed pipeline, including tree/boosting models.

Outputs: tools/eval/walk_forward_results.json and a printed scorecard.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "tools" / "eval"
TRAIN_FILE = REPO / "03_train_h1_auto_models.py"


def load_trainmod():
    spec = importlib.util.spec_from_file_location("trainmod", TRAIN_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainmod"] = mod
    spec.loader.exec_module(mod)
    return mod


def auc_score(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def choose_gate(conf, margin, correct, gate_grid, margin_grid, min_trades):
    best = (-1.0, 0.56, 0.06, 0)
    for g in gate_grid:
        for m in margin_grid:
            mask = (conf >= g) & (margin >= m)
            t = int(mask.sum())
            if t < min_trades:
                continue
            prec = float(correct[mask].mean())
            if prec > best[0] or (math.isclose(prec, best[0]) and t > best[3]):
                best = (prec, g, m, t)
    return best[1], best[2]


def evaluate_pair(path, T, args):
    pair = Path(path).stem.upper()
    raw = T.read_pair_csv(Path(path))
    raw_rows = len(raw)
    df, fcols = T.add_features(raw, pair)
    df = T.build_atr_direction_labels(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=fcols + ["y"]).reset_index(drop=True)
    if len(df) < T.MIN_ROWS_AFTER_FEATURES:
        return {"pair": pair, "ok": False, "reason": f"rows={len(df)}"}

    X = df[fcols]; y = df["y"].astype(int).values
    spread_atr = df["spread_atr"].values.astype(float) if "spread_atr" in df else np.zeros(len(df))
    n = len(df)
    test_start = int(n * (1 - args.test_fraction))
    test_len = (n - test_start) // args.folds
    if test_len < 80:
        return {"pair": pair, "ok": False, "reason": "too_few_test_rows"}

    rng = np.random.default_rng(42)
    oos_p, oos_y, pnl, wins = [], [], [], []
    H = T.HORIZON_BARS
    for k in range(args.folds):
        ts = test_start + k * test_len
        te = ts + test_len if k < args.folds - 1 else n
        tr_end = ts - H - args.embargo            # PURGE + EMBARGO
        if tr_end < 300:
            continue
        Xtr, ytr = X.iloc[:tr_end], y[:tr_end]
        if len(np.unique(ytr)) < 2:
            continue
        Xte, yte = X.iloc[ts:te], y[ts:te]
        sate = spread_atr[ts:te]

        models = T.build_classical_models()
        for name, model in models.items():
            if args.models and name not in args.models:
                continue
            try:
                model.fit(Xtr, ytr)
                ptr = model.predict_proba(Xtr)[:, 1]
                pte = model.predict_proba(Xte)[:, 1]
            except Exception as e:
                print(f"  [{pair}/{name}] fit failed: {e}")
                continue
            # only aggregate the model we're scoring (default: all, but keyed per model)
            _record(pair, name, ptr, ytr, pte, yte, sate, T, args, rng,
                    store.setdefault(name, {"p": [], "y": [], "pnl": [], "win": []}))
    return None  # results collected in `store`


def _record(pair, name, ptr, ytr, pte, yte, sate, T, args, rng, bucket):
    conf_tr = np.maximum(ptr, 1 - ptr); marg_tr = np.abs(ptr - 0.5) * 2
    ctr = ((ptr >= 0.5).astype(int) == ytr).astype(int)
    g, m = choose_gate(conf_tr, marg_tr, ctr, T.GATE_GRID, T.MARGIN_GRID, T.MIN_GATE_TRADES)
    conf_te = np.maximum(pte, 1 - pte); marg_te = np.abs(pte - 0.5) * 2
    corr = ((pte >= 0.5).astype(int) == yte).astype(int)
    mask = (conf_te >= g) & (marg_te >= m)
    bucket["p"].append(pte); bucket["y"].append(yte)
    for idx in np.where(mask)[0]:
        win = corr[idx] == 1
        cost = args.cost_mult * (sate[idx] if np.isfinite(sate[idx]) else 0.0)
        bucket["pnl"].append((T.TP_ATR if win else -T.SL_ATR) - cost)
        bucket["win"].append(int(win))


def summarise(pair, name, bucket, raw_rows, df_rows, rng):
    if not bucket["p"]:
        return {"pair": pair, "model": name, "ok": False, "reason": "no_folds"}
    P = np.concatenate(bucket["p"]); Y = np.concatenate(bucket["y"])
    pnl = np.array(bucket["pnl"]); win = np.array(bucket["win"]); ntr = len(pnl)
    res = {"pair": pair, "model": name, "ok": True, "oos_rows": int(len(Y)),
           "oos_auc": round(auc_score(Y, P), 4), "n_trades": ntr,
           "neutral_dropped_pct": round(100 * (1 - df_rows / max(1, raw_rows)), 1)}
    if ntr >= 10:
        exp = float(pnl.mean())
        boot = np.array([rng.choice(pnl, ntr, replace=True).mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res.update(precision=round(float(win.mean()), 4), expectancy_R=round(exp, 4),
                   exp_ci_lo=round(float(lo), 4), exp_ci_hi=round(float(hi), 4),
                   edge=bool(lo > 0))
    else:
        res.update(precision=None, expectancy_R=None, exp_ci_lo=None, exp_ci_hi=None, edge=False)
    return res


# global per-model accumulator (reset per pair)
store: dict = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="*", help="pair codes (default: all CSVs in DATA_DIR)")
    ap.add_argument("--models", nargs="*", help="model names (default: all classical)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=0, help="extra bars purged after split")
    ap.add_argument("--test-fraction", type=float, default=0.40)
    ap.add_argument("--cost-mult", type=float, default=1.0, help="round-trip cost = mult * spread(ATR)")
    args = ap.parse_args()

    T = load_trainmod()
    data_dir = Path(os.getenv("DATA_DIR", REPO / "oanda_h1_ba_live"))
    csvs = sorted(data_dir.glob("*.csv"))
    if args.pairs:
        want = {p.upper() for p in args.pairs}
        csvs = [c for c in csvs if c.stem.upper() in want]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    all_rows = []
    for path in csvs:
        store.clear()
        raw_rows = len(T.read_pair_csv(path))
        df, fcols = T.add_features(T.read_pair_csv(path), path.stem.upper())
        df = T.build_atr_direction_labels(df).replace([np.inf, -np.inf], np.nan).dropna(subset=fcols + ["y"])
        evaluate_pair(path, T, args)
        for name, bucket in store.items():
            row = summarise(path.stem.upper(), name, bucket, raw_rows, len(df), rng)
            all_rows.append(row); print(json.dumps(row))

    out = OUT_DIR / "walk_forward_results.json"
    out.write_text(json.dumps(all_rows, indent=2))

    # scorecard: best model per pair by expectancy CI lower bound
    print("\n================ SCORECARD (best model per pair) ================")
    print(f"{'PAIR':8} {'MODEL':18} {'OOS_AUC':>7} {'TRADES':>6} {'PREC':>6} {'EXP_R':>7} {'CI_LO':>7} {'EDGE':>5}")
    bypair: dict = {}
    for r in all_rows:
        if r.get("ok") and r.get("expectancy_R") is not None:
            bypair.setdefault(r["pair"], []).append(r)
    edges = 0
    for pair in sorted(bypair):
        best = max(bypair[pair], key=lambda r: (r["exp_ci_lo"] if r["exp_ci_lo"] is not None else -9))
        edges += 1 if best.get("edge") else 0
        print(f"{pair:8} {best['model']:18} {best['oos_auc']:7.3f} {best['n_trades']:6d} "
              f"{best['precision']:6.3f} {best['expectancy_R']:7.3f} {best['exp_ci_lo']:7.3f} "
              f"{'YES' if best.get('edge') else 'no':>5}")
    print("-" * 66)
    print(f"Pairs with statistically significant positive edge (CI_LO > 0): {edges}/{len(bypair)}")
    print(f"Full results: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
