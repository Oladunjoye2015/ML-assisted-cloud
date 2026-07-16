from pathlib import Path
import json

MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Walk-forward tradability bar (Phase 1).
# A pair is only allowed to trade live if it has cleared an HONEST, out-of-sample
# evaluation: expectancy 95% CI lower bound > 0 after costs, on a usable number of
# trades. That evidence comes from tools/walk_forward_eval.py (or the numpy preview),
# NOT from the in-sample training metrics. If no evaluation exists for a pair, it is
# NOT tradable — deployment must never outrun the evidence.
# ---------------------------------------------------------------------------
MIN_OOS_TRADES_TO_TRADE = 100     # target sample size for a trustworthy verdict
WF_RESULTS_CANDIDATES = [
    Path("tools/eval/walk_forward_results.json"),   # full model-set evaluator
    Path("tools/eval/eval_preview_results.json"),   # numpy LR preview (fallback)
]


def load_walkforward_edge():
    """Return {PAIR: {edge, expectancy_R, exp_ci_lo, n_trades, oos_auc, model, source}}
    keyed by the best (highest CI lower bound) result per pair, or {} if none found."""
    best = {}
    for path in WF_RESULTS_CANDIDATES:
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        for r in rows:
            if not r.get("ok") or r.get("exp_ci_lo") is None:
                continue
            pair = str(r.get("pair", "")).upper()
            ci_lo = r.get("exp_ci_lo")
            n = int(r.get("n_trades") or 0)
            cleared = bool(ci_lo is not None and ci_lo > 0 and n >= MIN_OOS_TRADES_TO_TRADE)
            cur = best.get(pair)
            cand = {
                "edge": cleared,
                "expectancy_R": r.get("expectancy_R"),
                "exp_ci_lo": ci_lo,
                "exp_ci_hi": r.get("exp_ci_hi"),
                "n_trades": n,
                "oos_auc": r.get("oos_auc"),
                "model": r.get("model", "logistic_regression"),
                "source": path.name,
            }
            if cur is None or (ci_lo is not None and ci_lo > (cur["exp_ci_lo"] or -9)):
                best[pair] = cand
        if best:
            break  # prefer the first (most authoritative) file that has data
    return best


WALKFORWARD_EDGE = load_walkforward_edge()
if WALKFORWARD_EDGE:
    print(f"Loaded walk-forward evidence for {len(WALKFORWARD_EDGE)} pair(s).")
else:
    print("WARNING: no walk-forward results found — every pair will be marked NOT tradable.")

pairs = {}

for pair_dir in sorted(MODELS_DIR.iterdir()):
    if not pair_dir.is_dir():
        continue

    pair = pair_dir.name.upper()
    best_model_path = pair_dir / "best_model.pkl"
    best_type_path = pair_dir / "best_model_type.json"
    thresholds_path = pair_dir / "thresholds.json"
    metrics_path = pair_dir / "metrics.json"

    if not best_model_path.exists():
        continue

    best_model = "sklearn_pipeline"
    if best_type_path.exists():
        try:
            data = json.loads(best_type_path.read_text())
            best_model = str(data.get("model_type") or data.get("best_model") or best_model).lower()
        except Exception:
            pass

    thresholds = {}
    if thresholds_path.exists():
        try:
            thresholds = json.loads(thresholds_path.read_text())
        except Exception:
            pass
    gate_override = {}
    gate_override_path = pair_dir / "live_gate_override.json"
    if gate_override_path.exists():
        try:
            gate_override = json.loads(gate_override_path.read_text())
        except Exception:
            gate_override = {}

    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            pass

    # AUDIT FIX (C5): the measured per-pair metrics live under metrics["best"], not at
    # the top level. The old code did metrics.get("avg_auc") or metrics.get("auc") -> both
    # None -> hardcoded 0.56 for EVERY pair, which then fed live position sizing. Read the
    # real values (nested "best" first), and only fall back to a placeholder if genuinely
    # absent. A weak pair should look weak here, not be flattened to 0.56.
    best = metrics.get("best") or {}

    def pick(*candidates, default):
        for c in candidates:
            if c is not None:
                return c
        return default

    real_auc = pick(metrics.get("avg_auc"), best.get("auc"), metrics.get("auc"), default=None)
    real_pair_score = pick(best.get("pair_score"), metrics.get("pair_score"), default=None)
    real_gate = pick(best.get("best_gate"), thresholds.get("conf_gate"),
                     thresholds.get("conf"), thresholds.get("threshold"), default=0.54)
    real_margin = pick(best.get("best_margin_gate"), thresholds.get("margin_gate"),
                       thresholds.get("margin"), default=0.04)
    feature_columns = (
        metrics.get("features")
        or metrics.get("feature_order")
        or metrics.get("feature_cols")
        or metrics.get("feature_columns")
        or []
    )
    if not feature_columns:
        feature_columns_path = pair_dir / "feature_columns.json"
        if feature_columns_path.exists():
            try:
                feature_columns = json.loads(feature_columns_path.read_text())
            except Exception:
                feature_columns = []

    # Tradability is decided by the walk-forward bar, NOT the in-sample training flag.
    wf = WALKFORWARD_EDGE.get(pair)
    if wf is None:
        tradable = False
        tradable_reason = "no_walkforward_evidence"
    elif wf["edge"]:
        tradable = True
        tradable_reason = f"cleared_walkforward (CI_lo={wf['exp_ci_lo']}, n={wf['n_trades']})"
        real_gate = pick(gate_override.get("conf_gate"), gate_override.get("gate"), real_gate, default=real_gate)
        real_margin = pick(gate_override.get("margin_gate"), gate_override.get("margin"), real_margin, default=real_margin)
    else:
        tradable = False
        tradable_reason = (
            f"failed_walkforward (CI_lo={wf['exp_ci_lo']}, n={wf['n_trades']}, "
            f"need CI_lo>0 & n>={MIN_OOS_TRADES_TO_TRADE})"
        )

    pairs[pair] = {
        "pair": pair,
        "best_model": best_model,
        "model_path": f"models/{pair}/best_model.pkl",
        "features": feature_columns,
        # Keep a placeholder ONLY when the value is truly missing, and record that fact
        # so downstream logic (and humans) know the number is not measured.
        "avg_auc": real_auc if real_auc is not None else 0.50,
        "avg_auc_is_measured": real_auc is not None,
        "pair_score": real_pair_score if real_pair_score is not None else 0.50,
        # Honest, out-of-sample verdict (overrides the optimistic in-sample flag).
        "tradable": tradable,
        "tradable_reason": tradable_reason,
        "tradable_in_sample_flag": bool(best.get("tradable", False)),
        "walkforward": wf,
        "default_gate": real_gate,
        "default_margin": real_margin,
        "gate_override": gate_override or None,
        "sl_atr": 1.0,
        "tp_atr": 1.3,
        "summary": metrics,
    }

registry = {
    "architecture": "h1_auto_registry",
    "created_by": "create_h1_registry.py",
    "pairs": pairs,
}

out = MODELS_DIR / "registry.json"
out.write_text(json.dumps(registry, indent=2))
print(f"Created {out} with {len(pairs)} pairs")
print(sorted(pairs))
