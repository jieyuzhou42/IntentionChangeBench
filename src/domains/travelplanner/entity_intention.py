from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Optional


ENTITY_PATH_PREFIX = "entities."


def normalize_entity_id(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized or None


def default_entities(people_number: Any) -> Dict[str, Dict[str, Any]]:
    try:
        count = max(1, int(float(people_number)))
    except (TypeError, ValueError):
        count = 1

    entities: Dict[str, Dict[str, Any]] = {}
    for index in range(1, count + 1):
        entity_id = f"entity_{index}"
        entities[entity_id] = {
            "reference": "the user" if index == 1 else "another traveler",
            "constraints": {},
        }
    return entities


def ensure_entity_state(intention: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized TravelPlanner intention without mutating the input."""
    normalized = copy.deepcopy(intention or {})
    constraints = normalized.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
        normalized["constraints"] = constraints

    raw_entities = normalized.get("entities")
    entities: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_entities, dict):
        for raw_id, raw_entity in raw_entities.items():
            entity_id = normalize_entity_id(raw_id)
            if not entity_id or not isinstance(raw_entity, dict):
                continue
            entity = copy.deepcopy(raw_entity)
            entity_constraints = entity.get("constraints")
            entity["constraints"] = (
                entity_constraints if isinstance(entity_constraints, dict) else {}
            )
            entity.setdefault(
                "reference",
                str(entity.get("display_name") or raw_id).strip() or entity_id,
            )
            entities[entity_id] = entity

    if not entities:
        entities = default_entities(
            constraints.get("people_number", constraints.get("party_size", 1))
        )
    normalized["entities"] = entities
    normalized["entity_priority"] = normalize_entity_priority(
        normalized.get("entity_priority"), entities
    )
    return normalized


def next_entity_id(entities: Dict[str, Dict[str, Any]]) -> str:
    index = 1
    while f"entity_{index}" in entities:
        index += 1
    return f"entity_{index}"


def normalize_entity_priority(
    raw_priority: Any,
    entities: Dict[str, Dict[str, Any]],
) -> List[str]:
    known_paths = {
        entity_constraint_path(entity_id, field)
        for entity_id, entity in entities.items()
        for field, value in (entity.get("constraints") or {}).items()
        if value is not None
    }
    if not isinstance(raw_priority, list):
        return []
    priority: List[str] = []
    for raw_path in raw_priority:
        path = str(raw_path or "").strip()
        if path in known_paths and path not in priority:
            priority.append(path)
    return priority


def entity_constraint_path(entity_id: str, field: str) -> str:
    return f"{ENTITY_PATH_PREFIX}{entity_id}.constraints.{field}"


def parse_entity_constraint_path(path: Any) -> Optional[tuple[str, str]]:
    parts = str(path or "").split(".")
    if len(parts) != 4 or parts[0] != "entities" or parts[2] != "constraints":
        return None
    entity_id = normalize_entity_id(parts[1])
    field = normalize_entity_id(parts[3])
    if not entity_id or not field:
        return None
    return entity_id, field


def iter_entity_constraints(
    intention: Dict[str, Any],
) -> Iterable[tuple[str, Dict[str, Any], str, Any]]:
    entities = intention.get("entities") if isinstance(intention, dict) else None
    if not isinstance(entities, dict):
        return
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        constraints = entity.get("constraints")
        if not isinstance(constraints, dict):
            continue
        for field, value in constraints.items():
            if value is not None:
                yield str(entity_id), entity, str(field), value
