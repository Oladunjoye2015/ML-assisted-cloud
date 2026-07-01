#!/usr/bin/env python3
"""Model store management (Phase 5).

The runtime only loads models/<PAIR>/best_model.pkl + metadata; the per-pair
candidate_models/ dirs (~430 MB total) are never served. This tool reports what's on
disk, verifies each pair has the artifacts the service needs, and can prune the unused
candidates from disk (they are already untracked from git).

    python tools/model_store.py report
    python tools/model_store.py verify
    python tools/model_store.py prune-candidates --yes
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", REPO / "models"))

REQUIRED = ["best_model.pkl", "feature_columns.json", "thresholds.json", "metrics.json"]


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def _pairs():
    return sorted(p for p in MODELS_DIR.iterdir() if p.is_dir())


def _registry_tradable():
    reg = MODELS_DIR / "registry.json"
    if not reg.exists():
        return {}
    try:
        pairs = json.loads(reg.read_text()).get("pairs", {})
        return {k: v.get("tradable") for k, v in pairs.items()}
    except Exception:
        return {}


def cmd_report(_args):
    tradable = _registry_tradable()
    total_cand = 0
    print(f"{'PAIR':8} {'best.pkl':>9} {'candidates':>10} {'cand.size':>10} {'tradable':>9}")
    for d in _pairs():
        best = d / "best_model.pkl"
        cand = d / "candidate_models"
        cand_n = len(list(cand.glob("*.pkl"))) if cand.exists() else 0
        cand_sz = _dir_size(cand) if cand.exists() else 0
        total_cand += cand_sz
        best_sz = _human(best.stat().st_size) if best.exists() else "MISSING"
        print(f"{d.name:8} {best_sz:>9} {cand_n:>10} {_human(cand_sz):>10} {str(tradable.get(d.name)):>9}")
    print(f"\nUnused candidate_models on disk: {_human(total_cand)} (safe to prune)")


def cmd_verify(_args):
    ok = True
    for d in _pairs():
        missing = [f for f in REQUIRED if not (d / f).exists()]
        if missing:
            ok = False
            print(f"  {d.name}: MISSING {', '.join(missing)}")
    print("All pairs have required artifacts." if ok else "Some pairs are missing artifacts (see above).")
    return 0 if ok else 1


def cmd_prune(args):
    targets = [d / "candidate_models" for d in _pairs() if (d / "candidate_models").exists()]
    total = sum(_dir_size(t) for t in targets)
    print(f"Will delete {len(targets)} candidate_models dir(s), freeing {_human(total)}.")
    if not args.yes:
        print("Dry run. Re-run with --yes to actually delete.")
        return 0
    for t in targets:
        shutil.rmtree(t)
    print(f"Deleted {len(targets)} dir(s).")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report").set_defaults(func=cmd_report)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    p = sub.add_parser("prune-candidates"); p.add_argument("--yes", action="store_true"); p.set_defaults(func=cmd_prune)
    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
