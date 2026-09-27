"""Shared baseline identities and human annotation normalization."""
import copy
import hashlib
import json

BASELINE_VERSION = "frozen-baseline-v3.1"
TIER_NAMES = {"high": "must_have", "medium": "preferred", "low": "optional",
              "must_have": "must_have", "preferred": "preferred", "optional": "optional"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def human_world_feasibility(turn):
    """Read explicit human labels; absent/null labels default to feasible."""
    found = []
    for obj in (turn, turn.get("gold_action") or {}, turn.get("annotation_review") or {}):
        if "world_feasibility" not in obj:
            continue
        value = obj["world_feasibility"]
        value = value.get("feasible") if isinstance(value, dict) else value
        if value is None:
            continue
        if type(value) is not bool:
            raise ValueError("world_feasibility.feasible must be a boolean")
        found.append(value)
    if len(set(found)) > 1:
        raise ValueError("Conflicting human World Feasibility annotations")
    return found[0] if found else True


def normalize_gold(gold):
    """Flatten entity constraints, preserve missing/conflicting priorities for audit."""
    result = copy.deepcopy(gold)
    constraints = {str(k): v for k, v in (gold.get("constraints") or {}).items() if v is not None}
    memberships = {}
    def add_priority(priority, prefix=""):
        for tier, names in (priority or {}).items():
            if tier not in TIER_NAMES:
                continue
            for name in names:
                name = str(name)
                if name.startswith("constraints."):
                    name = name[len("constraints."):]
                memberships.setdefault(prefix + name, set()).add(TIER_NAMES[tier])
    add_priority(gold.get("priority"))
    for eid, entity in (gold.get("entities") or {}).items():
        if not isinstance(entity, dict):
            continue
        prefix = "entities." + str(eid) + ".constraints."
        for field, value in (entity.get("constraints") or {}).items():
            if value is not None:
                constraints[prefix + field] = {"reference": entity.get("reference", eid), "value": value}
        add_priority(entity.get("priority"), prefix)
    result["constraints"] = constraints
    result["priority"] = {t: [] for t in ("must_have", "preferred", "optional")}
    for field, tiers in memberships.items():
        for tier in tiers:
            result["priority"][tier].append(field)
    return result


def constraint_tiers(gold):
    memberships = {}
    for name, fields in (gold.get("priority") or {}).items():
        if name not in TIER_NAMES:
            continue
        for field in fields:
            memberships.setdefault(str(field), set()).add(TIER_NAMES[name])
    return {str(f): next(iter(memberships[f])) if len(memberships.get(f, set())) == 1 else "unlabeled"
            for f, v in (gold.get("constraints") or {}).items() if v is not None}


def validate_scoring_priorities(gold, out_of_scope=()):
    bad = [f for f, tier in constraint_tiers(gold).items() if tier == "unlabeled" and f not in out_of_scope]
    if bad:
        raise ValueError("Missing/conflicting in-scope priorities: " + ", ".join(sorted(bad)))
