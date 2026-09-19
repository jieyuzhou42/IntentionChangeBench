"""Tiered agent priorities, compatible with the annotation tier names."""
from typing import Any, Dict, List

PRIORITY_ALIASES = {"must_have": "high", "preferred": "medium", "optional": "low"}
PRIORITY_RULES = """Assign each active constraint exactly once to a priority tier:
- must_have: required conditions the plan must satisfy.
- preferred: desired conditions the user allows trading off.
- optional: nice-to-have conditions that may be omitted.
Infer tiers from the user's requirements and explicit trade-offs across the conversation.
Preserve a requirement's tier unless the user changes it; recency alone never determines its tier.
Remove superseded requirements. Order within a tier has no meaning.
Use constraint names (or constraints.<name>) and entities.<id>.constraints.<name> for person-specific requirements.
""".strip()


def normalize_priority(priority: Any, constraints: Dict[str, Any], entities: Any = None) -> Dict[str, List[str]]:
    if not isinstance(priority, dict) or "ranked_fields" in priority:
        raise ValueError("priority must contain must_have, preferred, optional arrays; a ranking cannot determine tiers")
    canonical = set(PRIORITY_ALIASES)
    legacy = set(PRIORITY_ALIASES.values())
    if set(priority) not in (canonical, legacy):
        raise ValueError("priority must contain exactly must_have/preferred/optional (or high/medium/low)")
    roots = {"constraints." + str(k) for k, v in constraints.items() if v is not None}
    if isinstance(entities, dict):
        for entity_id, entity in entities.items():
            if isinstance(entity, dict) and isinstance(entity.get("constraints"), dict):
                roots.update("entities.%s.constraints.%s" % (entity_id, k) for k,v in entity["constraints"].items() if v is not None)
    document = {"constraints": constraints, "entities": entities or {}}
    result = {}
    seen = set()
    for tier, alias in PRIORITY_ALIASES.items():
        values = priority[tier if tier in priority else alias]
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("priority.%s must be an array of field names" % tier)
        result[tier] = []
        for field in values:
            field = field.strip()
            path = field if field.startswith(('constraints.', 'entities.')) else 'constraints.' + field
            # Match literal field names first (they can contain punctuation).
            valid = path in roots
            if not valid:
                value = document
                for part in path.split('.'):
                    if not isinstance(value, dict) or part not in value:
                        value = None
                        break
                    value = value[part]
                valid = value is not None and any(path.startswith(root + '.') for root in roots)
            if not valid:
                raise ValueError("Unknown or inactive priority field: " + field)
            if any(path == old or path.startswith(old + '.') or old.startswith(path + '.') for old in seen):
                raise ValueError("Duplicate or overlapping priority field: " + field)
            seen.add(path)
            result[tier].append(field)
    missing = [root for root in roots if not any(path == root or path.startswith(root + '.') for path in seen)]
    if missing:
        raise ValueError("Active fields missing from priority: " + ', '.join(sorted(missing)))
    return result
