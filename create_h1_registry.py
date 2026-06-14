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
            pass

    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            pass

    pairs[pair] = {
        "pair": pair,
        "best_model": best_model,
        "model_path": f"models/{pair}/best_model.pkl",
        "features": metrics.get("features") or metrics.get("feature_order") or metrics.get("feature_cols") or [],
        "avg_auc": metrics.get("avg_auc") or metrics.get("auc") or 0.56,
        "pair_score": metrics.get("pair_score") or 0.50,
        "default_gate": thresholds.get("conf_gate") or thresholds.get("conf") or thresholds.get("threshold") or 0.54,
        "default_margin": thresholds.get("margin_gate") or thresholds.get("margin") or 0.04,
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
