#!/usr/bin/env python3
"""What conf gate is 'optimal'? Sweep the confidence threshold over all out-of-sample
predictions (walk-forward, horizon-end label, base features) and report expectancy +
trade count at each gate, so the choice is data-backed rather than guessed."""
from __future__ import annotations
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_preview as EP
import eval_labels as EL

HORIZON, N_FOLDS, TEST_FRACTION = 8, 5, 0.40
RNG = np.random.default_rng(3)
confs_all, R_all = [], []

for path in sorted(glob.glob(os.path.join(EP.DATA_DIR, "*.csv"))):
    pair = os.path.basename(path).split(".")[0].upper()
    df = EP.read_pair_csv(path); df, fcols = EP.add_features(df, pair)
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
        pte = EP.predict_proba(EP.standardize_apply(X[s:e], med, mu, sd), w, b)
        conf = np.maximum(pte, 1-pte)
        for local in range(len(pte)):
            gi = idx[s+local]
            side = "BUY" if pte[local] >= .5 else "SELL"
            r = EL.realised_R(side, gi, C, H, L, A, HORIZON, sa)
            if r is not None:
                confs_all.append(float(conf[local])); R_all.append(float(r))

conf = np.array(confs_all); R = np.array(R_all)
print(f"total OOS predictions: {len(R)}\n")
print(f"{'conf_gate':>9} {'trades':>7} {'%taken':>7} {'mean_R':>8} {'CI_lo':>8} {'CI_hi':>8}")
for g in [0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.75]:
    m = conf >= g
    nt = int(m.sum())
    if nt < 10:
        print(f"{g:>9.2f} {nt:>7} {100*nt/len(R):>6.1f}% {'--':>8}")
        continue
    r = R[m]
    boot = np.array([RNG.choice(r, nt, replace=True).mean() for _ in range(1000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{g:>9.2f} {nt:>7} {100*nt/len(R):>6.1f}% {r.mean():>+8.3f} {lo:>+8.3f} {hi:>+8.3f}")
