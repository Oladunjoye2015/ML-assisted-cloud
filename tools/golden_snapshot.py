#!/usr/bin/env python3
"""
Golden-output snapshot harness for the /predict endpoint.

Purpose (Phase 0 safety net): capture the service's current decisions on a fixed set
of payloads BEFORE refactoring, then prove that a refactor did not change behaviour by
re-running with --check.

    python tools/golden_snapshot.py --write    # record current behaviour as golden
    # ...refactor...
    python tools/golden_snapshot.py --check     # exit 1 if ANY output changed

Design notes
------------
* Determinism: this harness forces an *offline* configuration (no live OANDA / LLM
  calls) via environment variables set BEFORE the app is imported. That makes outputs
  reproducible on any machine without credentials or a live market. It exercises the
  model → gates → sizing → response-assembly core plus the offline guards — which is
  exactly where refactor regressions are most likely and most costly.
* Guards that require the network (external market context, technical review, entry
  reversal, live-price staleness, AI review) are disabled here so the snapshot is
  stable. Re-enable selectively once those paths are themselves made testable.
* Volatile fields (wall-clock timestamps) are normalised out before comparison.

Run from the repo root so the app can find models/ and its data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tools" / "golden"
PAYLOADS = REPO / "tools" / "golden_payloads.json"

# --- Force a deterministic, offline config BEFORE importing the app -----------
# setdefault so an intentional override from the caller's shell still wins.
_OFFLINE_ENV = {
    "MODEL_FEATURE_SOURCE": "alert",          # use payload features; no candle fetch
    "MARKET_CONTEXT_ENABLED": "false",        # no multi-timeframe network fetch
    "TECHNICAL_REVIEW_ENABLED": "false",      # depends on market context
    "TECHNICAL_REVIEW_REQUIRED": "false",
    "ENTRY_REVERSAL_GUARD_ENABLED": "false",  # calls live quote
    "SIGNAL_STALENESS_GUARD_ENABLED": "false",# wall-clock dependent
    "AI_REVIEW_ENABLED": "false",             # network / non-deterministic
    "AUTO_CLOSE_ENABLED": "false",            # don't spawn broker activity
    # Neutralise credentials so nothing accidentally hits a real account.
    "OANDA_TOKEN": "",
    "OANDA_ACCOUNT_ID": "",
}
for k, v in _OFFLINE_ENV.items():
    os.environ.setdefault(k, v)

# Keys whose values are wall-clock / environment dependent -> blank them out.
_VOLATILE_KEYS = {"ts", "time", "timestamp", "server_time", "generated_at"}


def _normalise(obj):
    """Recursively blank volatile fields so snapshots are time-independent."""
    if isinstance(obj, dict):
        return {
            k: ("<VOLATILE>" if k in _VOLATILE_KEYS else _normalise(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 8)  # kill float formatting jitter across libs
    return obj


def _load_client():
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        sys.exit(f"fastapi.testclient unavailable ({e}); pip install -r requirements.txt")
    # Import the app AFTER env is set.
    sys.path.insert(0, str(REPO))
    import fx_api_sniper_CLperpair as appmod
    return TestClient(appmod.app)


def _run_all(client) -> dict:
    payloads = json.loads(PAYLOADS.read_text())
    results = {}
    for case in payloads:
        name = case["name"]
        body = case["payload"]
        resp = client.post("/predict", json=body)
        results[name] = {
            "status_code": resp.status_code,
            "response": _normalise(resp.json()),
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="record current output as golden")
    g.add_argument("--check", action="store_true", help="compare current output to golden")
    args = ap.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    client = _load_client()
    current = _run_all(client)

    golden_path = GOLDEN_DIR / "predict_golden.json"

    if args.write:
        golden_path.write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"Wrote {len(current)} golden case(s) -> {golden_path.relative_to(REPO)}")
        return 0

    # --check
    if not golden_path.exists():
        sys.exit(f"No golden file at {golden_path}. Run with --write first.")
    golden = json.loads(golden_path.read_text())

    changed = []
    for name in sorted(set(golden) | set(current)):
        if golden.get(name) != current.get(name):
            changed.append(name)

    if not changed:
        print(f"OK — all {len(current)} case(s) match golden. Behaviour unchanged.")
        return 0

    print(f"MISMATCH in {len(changed)} case(s): {', '.join(changed)}\n")
    for name in changed:
        g = json.dumps(golden.get(name), indent=2, sort_keys=True).splitlines()
        c = json.dumps(current.get(name), indent=2, sort_keys=True).splitlines()
        import difflib
        diff = difflib.unified_diff(g, c, fromfile=f"golden/{name}", tofile=f"current/{name}", lineterm="")
        print("\n".join(diff))
        print("-" * 60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
