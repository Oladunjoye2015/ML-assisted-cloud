from pathlib import Path
import json

MODELS_DIR = Path("models")

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
            thresholds = {}

    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}

    features = (
        metrics.get("features")
        or metrics.get("feature_order")
        or metrics.get("feature_cols")
        or thresholds.get("features")
        or thresholds.get("feature_order")
        or []
    )

    if not features:
        # H1 server default feature order. This should match your H1 trainer/server.
        features = [
            "spread_pips","spread_atr","ema20","ema50","ema200","rsi14","macdh",
            "adx14","plus_di14","minus_di14","atr14","atr_pct","bbw",
            "ret1","ret3","ret6","ret12","ret24",
            "d20","d50","d200","s20","s50","s200",
            "dist_high_12","dist_low_12","dist_high_24","dist_low_24",
            "range_pips","body_pips","upper_wick_pips","lower_wick_pips",
            "hour","dow","month","session",
            "ema50_h4","adx14_h4","ema20_d1","rsi14_d1",
            "trend_regime","vol_regime"
        ]

    default_gate = (
        thresholds.get("conf")
        or thresholds.get("conf_gate")
        or thresholds.get("threshold")
        or thresholds.get("approval_gate")
        or 0.54
    )
    default_margin = (
        thresholds.get("margin")
        or thresholds.get("margin_gate")
        or 0.04
    )

    avg_auc = metrics.get("avg_auc") or metrics.get("auc") or metrics.get("best_auc") or 0.56
    pair_score = metrics.get("pair_score") or 0.50

    pairs[pair] = {
        "pair": pair,
        "best_model": best_model,
        "model_path": str(best_model_path),
        "features": features,
        "avg_auc": avg_auc,
        "pair_score": pair_score,
        "default_gate": default_gate,
        "default_margin": default_margin,
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
print("Pairs:", ", ".join(sorted(pairs)))
