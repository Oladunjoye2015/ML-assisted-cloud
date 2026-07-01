"""Pure decision predicates extracted from the /predict flow. No I/O, no state."""
from __future__ import annotations


def side_from_p_up(p_up: float) -> str:
    """Model direction from P(up)."""
    return "BUY" if p_up >= 0.5 else "SELL"


def side_probability(p_up: float, side: str) -> float:
    return p_up if side == "BUY" else (1.0 - p_up)


def margin_from_p_up(p_up: float) -> float:
    """Distance from the coin-flip line, scaled to [0,1]."""
    return float(abs(p_up - 0.5) * 2.0)


def passes_sniper_gate(conf: float, margin: float, conf_gate: float, margin_gate: float) -> bool:
    return (conf >= conf_gate) and (margin >= margin_gate)


def hint_disagrees(side: str, hint_side: str) -> bool:
    return hint_side in ("BUY", "SELL") and side != hint_side


def blocked_by_hint_disagreement(
    side: str, hint_side: str, conf: float, disagree_conf_gate: float
) -> bool:
    """The model disagrees with an incoming hint and isn't confident enough to override it."""
    return hint_disagrees(side, hint_side) and conf < disagree_conf_gate
