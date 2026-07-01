import math
from mlac import gating as G


def test_side_and_margin():
    assert G.side_from_p_up(0.62) == "BUY"
    assert G.side_from_p_up(0.5) == "BUY"     # boundary -> BUY (>= 0.5)
    assert G.side_from_p_up(0.49) == "SELL"
    assert G.margin_from_p_up(0.5) == 0.0
    assert math.isclose(G.margin_from_p_up(0.75), 0.5)
    assert math.isclose(G.side_probability(0.7, "SELL"), 0.3)


def test_sniper_gate():
    assert G.passes_sniper_gate(0.60, 0.10, 0.54, 0.04) is True
    assert G.passes_sniper_gate(0.53, 0.10, 0.54, 0.04) is False   # conf too low
    assert G.passes_sniper_gate(0.60, 0.02, 0.54, 0.04) is False   # margin too low


def test_hint_disagreement_block():
    # Model says BUY, hint says SELL, conf below the disagreement gate -> blocked.
    assert G.blocked_by_hint_disagreement("BUY", "SELL", conf=0.60, disagree_conf_gate=0.64) is True
    # Same disagreement but confident enough -> allowed.
    assert G.blocked_by_hint_disagreement("BUY", "SELL", conf=0.70, disagree_conf_gate=0.64) is False
    # Agreement -> never blocked.
    assert G.blocked_by_hint_disagreement("BUY", "BUY", conf=0.10, disagree_conf_gate=0.64) is False
    # No hint -> never blocked.
    assert G.blocked_by_hint_disagreement("BUY", "", conf=0.10, disagree_conf_gate=0.64) is False
