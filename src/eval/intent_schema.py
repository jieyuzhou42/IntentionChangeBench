"""Atomic intention output for evaluation; independent of the action schema."""
import copy
import json
from typing import Any, Dict

INTENT_SCHEMA = {
    "intent": [{"field": "semantic_field", "value": "active value", "priority": "must_have"}]
}
INTENT_SCHEMA_JSON = json.dumps(INTENT_SCHEMA, indent=2)
INTENT_RULES = """Return cumulative active requirements as an intent array.
Each item has exactly field, value, and priority. Do not emit relation or scope.
- field: a concise semantic field name; use the supplied vocabulary where applicable.
  Preserve necessary context in the field or value, e.g. budget_max,
  day_2_required_activity, day_3_excluded_activity, or entity_2_mobility.
- value: the actual typed JSON value, preserving dates, travelers, limits and
  inclusion/exclusion meaning. Use numbers for numeric limits where the field
  already identifies the limit; structured values may carry necessary context.
- priority: must_have (required), preferred (tradeable preference), or optional
  (may be omitted). Infer from user requirements and trade-offs, not recency.
Preserve a tier until the user changes it; remove superseded requirements.
Split distinct requirements into separate items. Never merge items only because
they share a field name. Keep stable traveler references across turns.
Do not invent requirements, values or flight IDs absent from the available context.
Do not emit separate constraints, entities, ranked_fields or priority objects.
""".strip()


def normalize_intent_prediction(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("intent"), list):
        raise ValueError("Intention prediction must contain an intent array")
    if any(key in raw for key in ("constraints", "entities", "priority", "entity_priority", "ranked_fields")):
        raise ValueError("Use atomic intent items instead of legacy intention fields")
    fields = {"field", "value", "priority"}
    result = []
    seen = set()
    for item in raw["intent"]:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("Each intent item requires exactly field, value, priority")
        row = copy.deepcopy(item)
        for key in fields - {"value"}:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("intent.%s must be a nonempty string" % key)
            row[key] = row[key].strip()
        if row["priority"] not in {"must_have", "preferred", "optional"}:
            raise ValueError("Invalid intent priority: " + row["priority"])
        if row["value"] is None:
            raise ValueError("Active intent value cannot be null")
        identity = json.dumps({k:v for k,v in row.items() if k != "priority"}, sort_keys=True, allow_nan=False)
        if identity in seen:
            raise ValueError("Duplicate intent item (possibly conflicting priorities)")
        seen.add(identity)
        result.append(row)
    return {"intent": result}


def environment_intention(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Internal legacy view, retaining the complete intent array.

    Only unique fields already understood by legacy environments are projected;
    contextual and repeated requirements remain exclusively in the intent list.
    """
    if "intent" not in prediction:
        return prediction
    result = copy.deepcopy(prediction)
    constraints = {}
    priorities = {"high": [], "medium": [], "low": []}
    aliases = {"budget_max": "budget"}
    supported = {"budget", "days", "people_number", "party_size", "org", "origin",
                 "dest", "destination", "date", "start_date", "end_date",
                 "visiting_city_number", "required_cities", "room_type", "cuisine",
                 "house_rule", "transportation", "accommodation_rating", "restaurant_rating"}
    atoms = prediction["intent"]
    for item in atoms:
        field = aliases.get(item["field"], item["field"])
        same_field = [x for x in atoms if aliases.get(x["field"], x["field"]) == field]
        if len(same_field) != 1 or field not in supported:
            continue
        constraints[field] = copy.deepcopy(item["value"])
        priorities[{"must_have":"high", "preferred":"medium", "optional":"low"}[item["priority"]]].append(field)
    result.update(constraints=constraints, priority=priorities)
    return result
