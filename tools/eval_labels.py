#!/usr/bin/env python3
"""Label experiment: which target definition improves out-of-sample edge?

Trains an LR proxy (walk-forward, purged, out-of-sample gate) on each candidate LABEL,
but scores every candidate with the SAME realistic P&L engine so the comparison is fair:
    enter the predicted direction, exit at the FIRST of {TP=1.3 ATR, SL=1.0 ATR, horizon
    end}, realised return in ATR-R units, minus one spread (ATR units).
Reports mean OOS expectancy and #pairs whose expectancy 95% CI lower bound > 0.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_preview as EP   # reuse features/LR/standardise/gate helpers

TP_ATR, SL_ATR = 1.3, 1.0
N_FOLDS, TEST_FRACTION = 5, 0.40
MIN_GATE_TRADES = 30
RNG = np.random.default_rng(7)
DATA_DIR = EP.DATA_DIR

# ---------------- candidate label builders -> returns y in {0,1,nan} ----------
def label_baseline(C, H, L, A, horizon=8):
    """Symmetric ATR triple-barrier, first-to-hit; neutral -> nan (current trainer)."""
    n = len(C); y = np.full(n, np.nan)
    for i in range(n - horizon - 1):
        e, a = C[i], A[i]
        if not (np.isfinite(e) and np.isfinite(a) and a > 0): continue
        ltp, lsl, stp, ssl = e + 1.3*a, e - 1.0*a, e - 1.3*a, e + 1.0*a
        for j in range(1, horizon + 1):
            h, l = H[i+j], L[i+j]
            if h >= ltp and l <= lsl: break
            if l <= stp and h >= ssl: break
            if h >= ltp and not (l <= stp): y[i] = 1; break
            if l <= stp and not (h >= ltp): y[i] = 0; break
    return y

def label_horizon_end(C, H, L, A, horizon=8):
    """Barrier first; if neither hit within horizon, resolve by sign of net move at H.
    Every bar gets a label -> removes the neutral-drop selection bias (audit C3)."""
    n = len(C); y = np.full(n, np.nan)
    for i in range(n - horizon - 1):
        e, a = C[i], A[i]
        if not (np.isfinite(e) and np.isfinite(a) and a > 0): continue
        ltp, lsl, stp, ssl = e + 1.3*a, e - 1.0*a, e - 1.3*a, e + 1.0*a
        decided = False
        for j in range(1, horizon + 1):
            h, l = H[i+j], L[i+j]
            if h >= ltp and l <= lsl: y[i] = 0; decided = True; break   # ambiguous -> treat as loss-side
            if l <= stp and h >= ssl: y[i] = 1; decided = True; break
            if h >= ltp and not (l <= stp): y[i] = 1; decided = True; break
            if l <= stp and not (h >= ltp): y[i] = 0; decided = True; break
        if not decided:
            y[i] = 1 if C[i+horizon] > e else 0
    return y

def label_forward_return(C, H, L, A, horizon=8):
    """Pure direction of the net move H bars ahead (no barriers). Simple + realistic."""
    n = len(C); y = np.full(n, np.nan)
    for i in range(n - horizon - 1):
        e = C[i]
        if not np.isfinite(e): continue
        y[i] = 1 if C[i+horizon] > e else 0
    return y

def label_baseline_h16(C, H, L, A, horizon=16):
    return label_baseline(C, H, L, A, horizon=16)

def label_horizon_end_h16(C, H, L, A, horizon=16):
    return label_horizon_end(C, H, L, A, horizon=16)

VARIANTS = {
    "baseline_h8":     (label_baseline, 8),
    "horizon_end_h8":  (label_horizon_end, 8),
    "fwd_return_h8":   (label_forward_return, 8),
    "baseline_h16":    (label_baseline_h16, 16),
    "horizon_end_h16": (label_horizon_end_h16, 16),
}

# ---------------- realistic P&L for a taken trade (same for all variants) ------
def realised_R(side, i, C, H, L, A, horizon, spread_atr):
    """Enter predicted direction at bar i; exit at first of TP/SL/horizon. Return R net."""
    e, a = C[i], A[i]
    if not (np.isfinite(e) and np.isfinite(a) and a > 0):
        return None
    cost = spread_atr[i] if np.isfinite(spread_atr[i]) else 0.0
    if side == "BUY":
        tp, sl = e + TP_ATR*a, e - SL_ATR*a
        for j in range(1, horizon + 1):
            if i+j >= len(C): break
            if H[i+j] >= tp: return TP_ATR - cost
            if L[i+j] <= sl: return -SL_ATR - cost
        end = i + min(horizon, len(C)-1-i)
        return (C[end] - e)/a - cost
    else:
        tp, sl = e - TP_ATR*a, e + SL_ATR*a
        for j in range(1, horizon + 1):
            if i+j >= len(C): break
            if L[i+j] <= tp: return TP_ATR - cost
            if H[i+j] >= sl: return -SL_ATR - cost
        end = i + min(horizon, len(C)-1-i)
        return (e - C[end])/a - cost

def eval_variant(paths, builder, horizon):
    per_pair = {}
    for path in paths:
        pair = os.path.basename(path).split(".")[0].upper()
        df = EP.read_pair_csv(path); df, fcols = EP.add_features(df, pair)
        df = df.replace([np.inf, -np.inf], np.nan)
        C = df["mid_c"].values.astype(float); Hh = df["mid_h"].values.astype(float)
        Ll = df["mid_l"].values.astype(float); A = df["atr14"].values.astype(float)
        sa = df["spread_atr"].values.astype(float)
        y = builder(C, Hh, Ll, A, horizon)
        df["_y"] = y
        keep = df[fcols + ["_y"]].dropna()
        if len(keep) < 1500: continue
        idx = keep.index.values
        X = keep[fcols].values.astype(float); Y = keep["_y"].astype(int).values
        n = len(idx); ts0 = int(n*(1-TEST_FRACTION)); tl = (n-ts0)//N_FOLDS
        if tl < 80: continue
        pnl = []
        for k in range(N_FOLDS):
            s = ts0 + k*tl; e = s+tl if k < N_FOLDS-1 else n
            tr_end = s - horizon
            if tr_end < 300: continue
            Xtr, Ytr = X[:tr_end], Y[:tr_end]
            if len(np.unique(Ytr)) < 2: continue
            med, mu, sd = EP.standardize_fit(Xtr)
            w, b = EP.fit_logreg(EP.standardize_apply(Xtr, med, mu, sd), Ytr)
            ptr = EP.predict_proba(EP.standardize_apply(Xtr, med, mu, sd), w, b)
            ctr = ((ptr >= .5).astype(int) == Ytr).astype(int)
            g, m = EP.choose_gate(np.maximum(ptr, 1-ptr), np.abs(ptr-.5)*2, ctr)
            pte = EP.predict_proba(EP.standardize_apply(X[s:e], med, mu, sd), w, b)
            conf = np.maximum(pte, 1-pte); marg = np.abs(pte-.5)*2
            mask = (conf >= g) & (marg >= m)
            for local in np.where(mask)[0]:
                gi = idx[s+local]                      # map back to full-frame row
                side = "BUY" if pte[local] >= .5 else "SELL"
                r = realised_R(side, gi, C, Hh, Ll, A, horizon, sa)
                if r is not None: pnl.append(r)
        if len(pnl) >= 10:
            pnl = np.array(pnl)
            boot = np.array([RNG.choice(pnl, len(pnl), replace=True).mean() for _ in range(800)])
            per_pair[pair] = (float(pnl.mean()), float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5)), len(pnl))
    return per_pair

def main():
    import json
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    which = sys.argv[1] if len(sys.argv) > 1 else None
    names = [which] if which in VARIANTS else list(VARIANTS)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_labels_results.json")
    results = {}
    if os.path.exists(out_path):
        try: results = json.load(open(out_path))
        except Exception: results = {}
    for name in names:
        builder, H = VARIANTS[name]
        pp = eval_variant(paths, builder, H)
        exps = [v[0] for v in pp.values()]
        edges = [p for p, v in pp.items() if v[1] > 0]
        mean_exp = float(np.mean(exps)) if exps else float("nan")
        best = max(pp.items(), key=lambda kv: kv[1][1]) if pp else ("-", (0,0,0,0))
        results[name] = {"mean_exp_R": mean_exp, "edge_pairs": len(edges), "total": len(pp),
                         "best_pair": best[0], "best_exp": best[1][0], "best_ci_lo": best[1][1],
                         "best_n": best[1][3], "per_pair": {k: v for k, v in pp.items()}}
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"{name:16} mean_exp_R={mean_exp:+.3f} | edge(CI_lo>0)={len(edges)}/{len(pp)} "
              f"| best={best[0]} exp={best[1][0]:+.3f} CI_lo={best[1][1]:+.3f} n={best[1][3]}")

if __name__ == "__main__":
    main()
