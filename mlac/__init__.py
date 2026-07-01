"""mlac — pure, dependency-light core of the ML-assisted-cloud FX system.

These modules hold logic extracted verbatim from the monolithic service/trainer so it
can be unit-tested WITHOUT importing fastapi / sklearn / joblib. They are the intended
single source of truth for the target architecture (see AUDIT_AND_REFACTOR_PLAN.md §5).

Migration status: the live entrypoint (fx_api_sniper_CLperpair.py) has NOT yet been
rewired to import from here — that step must be verified with tools/golden_snapshot.py
before deploying, since the entrypoint auto-deploys to live trading. Until then these are
the tested reference implementations.
"""
__all__ = ["instruments", "indicators", "labeling", "sizing", "gating", "reservation"]
