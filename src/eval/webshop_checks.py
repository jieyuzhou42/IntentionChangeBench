"""Catalog identity and selected-variant checks; never infer unchosen options."""
import math


def selection(turn, gold=False):
    action = (turn.get("gold_action") if gold else turn.get("action") or turn.get("agent_action")) or {}
    payload = action.get("action_payload") or action
    if gold:
        return {"asin": payload.get("selected_asin") or payload.get("asin"),
                "options": payload.get("selected_options") or {}, "rationale": action.get("rationale", "")}
    feedback = turn.get("env_feedback") or {}
    trace = turn.get("rollout_trace") or []
    last = trace[-1] if trace else {}
    evidence = turn.get("action_evidence") or {}
    options = payload.get("selected_options")
    if options is None:
        options = feedback.get("selected_options", last.get("selected_options", evidence.get("selected_options", {})))
    return {"asin": feedback.get("selected_asin") or evidence.get("selected_asin") or payload.get("selected_asin") or payload.get("asin"),
            "options": options or {}, "rationale": action.get("rationale") or (last.get("action") or {}).get("rationale", "")}


def validate_selection(selected, catalog):
    asin = str(selected.get("asin") or "").upper()
    if not asin:
        return {"valid": False, "reason": "no_final_selection", "product": None}
    if asin not in catalog:
        raise ValueError("Selected ASIN missing from catalog snapshot (data error): " + asin)
    product = catalog[asin]
    options = selected.get("options") or {}
    if not isinstance(options, dict):
        return {"valid": False, "reason": "invalid_options", "product": product}
    available = product.get("options") or {}
    for name, value in options.items():
        if name not in available or value not in available[name]:
            return {"valid": False, "reason": "unsupported_selected_option:" + str(name), "product": product}
    return {"valid": True, "reason": None, "product": product}


def numeric_price(product):
    value = (product or {}).get("price")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None
