"""Domain-independent, deterministic Action scoring. No model calls."""
from typing import Any, Dict, Mapping, Optional, Sequence

SCORING_VERSION = "action-gated-gold-soft-2to1-v1.1"
TIERS = ("must_have", "preferred", "optional")
STATUSES = {"satisfied", "violated", "unknown"}


def score_action_constraints(*, tiers: Mapping[str, str], agent: Mapping[str, str],
                             gold: Optional[Mapping[str, str]], world_feasible: Optional[bool],
                             out_of_scope: Sequence[str] = (), action_valid: bool = True) -> Dict[str, Any]:
    fields = {f: t for f, t in tiers.items() if f not in set(out_of_scope)}
    bad = [f for f, t in fields.items() if t not in TIERS]
    if bad:
        raise ValueError("Active constraints need exactly one priority: " + ", ".join(sorted(bad)))
    for who, statuses in (("agent", agent), ("gold", gold)):
        if statuses is None:
            continue
        if any(f not in statuses or statuses[f] not in STATUSES for f in fields):
            raise ValueError(who + ": missing or invalid constraint judgments")
    if world_feasible is not None and type(world_feasible) is not bool:
        raise ValueError("World Feasibility must be a human boolean")
    if world_feasible is None:
        world_feasible = True
    total = {t: sum(v == t for v in fields.values()) for t in TIERS}
    counts = {who: {t: sum(fields[f] == t and (statuses is None or statuses[f] == "satisfied")
                          for f in fields) for t in TIERS}
              for who, statuses in (("agent", agent), ("gold", gold))}
    a, g = counts["agent"], counts["gold"]
    # Missing Gold is perfect, so either feasibility branch requires all Must.
    must_gate = a["must_have"] >= (g["must_have"] if world_feasible is False else total["must_have"])
    sa, sg = (2 * c["preferred"] + c["optional"] for c in (a, g))
    soft = min(1.0, sa / sg) if sg else 1.0
    gate = must_gate and action_valid
    all_met = all(agent[f] == "satisfied" for f in fields)
    return {"scoring_version": SCORING_VERSION, "world_feasible": world_feasible,
            "gold_assumed_perfect": gold is None, "totals": total, "counts": counts,
            "must_gate": must_gate, "action_valid": action_valid, "gate_pass": gate,
            "S_a": sa, "S_g": sg, "soft_score": soft, "action_score": soft if gate else 0.0,
            "all_constraints_satisfied": all_met,
            "action_success": gate and (all_met if gold is None else soft == 1.0),
            "failed_musts": sorted(f for f in fields if fields[f] == "must_have" and agent[f] != "satisfied")}
