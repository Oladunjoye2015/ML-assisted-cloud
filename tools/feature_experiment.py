#!/usr/bin/env python3
"""Feature experiment: does a richer feature set create edge the base set can't?

Same honest protocol as the label experiment (walk-forward, purged split, out-of-sample
gate, realistic P&L, horizon-end label). Compares:
  base      = the current trainer features
  extended  = base + higher-timeframe trend state (shifted H4/D EMAs, causal),
              volatility regime, Donchian/range position, momentum streaks.

Run one set per call (keeps under the sandbox time limit):
    python feature_experiment.py base
    python feature_experiment.py extended
Results accumulate in feature_experiment_results.json.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_preview as EP
import eval_labels as EL

HORIZON = 8
N_FOLDS, TEST_FRACTION, MIN_GATE_TRADES = 5, 0.40, 30
RNG = np.random.default_rng(11)
DATA_DIR = EP.DATA_DIR


def add_extended(df: pd.DataFrame, base_fcols):
    """Add causal higher-timeframe + regime + range features. Returns extra col names."""
    extra = []
    c = df["mid_c"]
    atr = df["atr14"].replace(0, np.nan)

    # --- Higher-timeframe trend state (shifted so only COMPLETED HTF bars are visible) ---
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        s = pd.Series(c.values, index=t)
        for tf, span, tag in [("4h", 50, "h4"), ("1D", 30, "d1")]:
            htf = s.resample(tf).last()
            htf_ema = htf.ewm(span=span, adjust=False).mean().shift(1)   # shift => causal
            ema_on_h1 = htf_ema.reindex(t, method="ffill").values
            slope = htf_ema.diff().reindex(t, method="ffill").values
            df[f"dist_{tag}ema_atr"] = (c.values - ema_on_h1) / atr.values
            df[f"{tag}_trend_up"] = (c.values > ema_on_h1).astype(float)
            df[f"{tag}_ema_slope_sign"] = np.sign(slope)
            extra += [f"dist_{tag}ema_atr", f"{tag}_trend_up", f"{tag}_ema_slope_sign"]

    # --- Volatility regime ---
    atr_mean = atr.rolling(100).mean()
    atr_std = atr.rolling(100).std().replace(0, np.nan)
    df["atr_z"] = ((atr - atr_mean) / atr_std)
    df["atr_ratio_fast_slow"] = atr / atr.rolling(96).mean()
    extra += ["atr_z", "atr_ratio_fast_slow"]

    # --- Range / breakout position (Donchian) ---
    for w in (20, 55):
        hi = df["mid_h"].rolling(w).max()
        lo = df["mid_l"].rolling(w).min()
        rng = (hi - lo).replace(0, np.nan)
        df[f"donch_pos_{w}"] = (c - lo) / rng
        df[f"dist_hi{w}_atr"] = (hi - c) / atr
        df[f"dist_lo{w}_atr"] = (c - lo) / atr
        extra += [f"donch_pos_{w}", f"dist_hi{w}_atr", f"dist_lo{w}_atr"]

    # --- Momentum streak (consecutive same-sign 1-bar returns) ---
    sign = np.sign(c.diff().fillna(0.0).values)
    streak = np.zeros(len(sign))
    for i in range(1, len(sign)):
        streak[i] = streak[i-1] + sign[i] if sign[i] == sign[i-1] and sign[i] != 0 else sign[i]
    df["ret_streak"] = streak
    df["mom_6_atr"] = (c - c.shift(6)) / atr
    df["mom_24_atr"] = (c - c.shift(24)) / atr
    extra += ["ret_streak", "mom_6_atr", "mom_24_atr"]

    return list(base_fcols) + extra


def run_set(paths, mode):
    per_pair = {}
    for path in paths:
        pair = os.path.basename(path).split(".")[0].upper()
        df = EP.read_pair_csv(path)
        df, base_fcols = EP.add_features(df, pair)
        fcols = list(base_fcols) if mode == "base" else add_extended(df, base_fcols)
        df = df.replace([np.inf, -np.inf], np.nan)
        C = df["mid_c"].values.astype(float); H = df["mid_h"].values.astype(float)
        L = df["mid_l"].values.astype(float); A = df["atr14"].values.astype(float)
        sa = df["spread_atr"].values.astype(float)
        df["_y"] = EL.label_horizon_end(C, H, L, A, HORIZON)
        keep = df[fcols + ["_y"]].dropna()
        if len(keep) < 1500:
            continue
        idx = keep.index.values
        X = keep[fcols].values.astype(float); Y = keep["_y"].astype(int).values
        n = len(idx); ts0 = int(n*(1-TEST_FRACTION)); tl = (n-ts0)//N_FOLDS
        if tl < 80:
            continue
        pnl = []
        for k in range(N_FOLDS):
            s = ts0 + k*tl; e = s+tl if k < N_FOLDS-1 else n
            tr_end = s - HORIZON
            if tr_end < 300:
                continue
            Xtr, Ytr = X[:tr_end], Y[:tr_end]
            if len(np.unique(Ytr)) < 2:
                continue
            med, mu, sd = EP.standardize_fit(Xtr)
            w, b = EP.fit_logreg(EP.standardize_apply(Xtr, med, mu, sd), Ytr)
            ptr = EP.predict_proba(EP.standardize_apply(Xtr, med, mu, sd), w, b)
            ctr = ((ptr >= .5).astype(int) == Ytr).astype(int)
            g, m = EP.choose_gate(np.maximum(ptr, 1-ptr), np.abs(ptr-.5)*2, ctr)
            pte = EP.predict_proba(EP.standardize_apply(X[s:e], med, mu, sd), w, b)
            mask = (np.maximum(pte, 1-pte) >= g) & (np.abs(pte-.5)*2 >= m)
            for local in np.where(mask)[0]:
                gi = idx[s+local]
                side = "BUY" if pte[local] >= .5 else "SELL"
                r = EL.realised_R(side, gi, C, H, L, A, HORIZON, sa)
                if r is not None:
                    pnl.append(r)
        if len(pnl) >= 10:
            pnl = np.array(pnl)
            boot = np.array([RNG.choice(pnl, len(pnl), replace=True).mean() for _ in range(800)])
            per_pair[pair] = (float(pnl.mean()), float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5)), len(pnl))
    return per_pair


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    pp = run_set(paths, mode)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_experiment_results.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    exps = [v[0] for v in pp.values()]
    edges = [p for p, v in pp.items() if v[1] > 0]
    res[mode] = {"mean_exp_R": float(np.mean(exps)) if exps else None,
                 "edge_pairs": len(edges), "total": len(pp),
                 "per_pair": {k: v for k, v in pp.items()},
                 "n_features": None}
    json.dump(res, open(out, "w"), indent=2)
    best = max(pp.items(), key=lambda kv: kv[1][1]) if pp else ("-", (0, 0, 0, 0))
    print(f"{mode:9} mean_exp_R={res[mode]['mean_exp_R']:+.3f} | edge(CI_lo>0)={len(edges)}/{len(pp)} "
          f"| best={best[0]} exp={best[1][0]:+.3f} CI_lo={best[1][1]:+.3f} n={best[1][3]}")


if __name__ == "__main__":
    main()
