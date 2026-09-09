"""Re-extract TravelPlanner constraint state from user utterances with an LLM."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.llm_clients import create_llm_client_from_env
from prompt_logging import log_prompt
from scripts.classify_constraint_priorities import classify_instance


ENTITY_PREFIX = "entities."
ENTITY_MARKER = ".constraints."
REMOVAL_OPS = {"remove", "delete", "drop"}
TRAVEL_CONTEXT_FIELDS = {"org", "dest", "start_date", "end_date", "visiting_city_number"}


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _clean_field(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_.")


def _clean_entity_id(value: Any) -> str:
    match = re.fullmatch(r"entity_([1-9][0-9]*)", _clean_field(value))
    return f"entity_{int(match.group(1))}" if match else ""


def _normalize_path(value: Any) -> str:
    text = str(value or "").strip()
    entity_match = re.fullmatch(
        r"entities\.([^.]+)\.constraints\.(.+)", text, flags=re.IGNORECASE
    )
    if entity_match:
        entity_id = _clean_entity_id(entity_match.group(1))
        field = _clean_field(entity_match.group(2))
        return f"entities.{entity_id}.constraints.{field}" if entity_id and field else ""
    return _clean_field(text)


def _non_null_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, Any] = {}
    for raw_field, raw_value in value.items():
        field = _clean_field(raw_field)
        if field and raw_value is not None:
            result[field] = copy.deepcopy(raw_value)
    return result


def _normalize_entities(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    entities: Dict[str, Dict[str, Any]] = {}
    for raw_id, raw_entity in value.items():
        entity_id = _clean_entity_id(raw_id)
        if not entity_id or not isinstance(raw_entity, dict):
            continue
        reference = re.sub(r"\s+", " ", str(raw_entity.get("reference") or entity_id)).strip()
        entities[entity_id] = {
            "reference": reference or entity_id,
            "constraints": _non_null_mapping(raw_entity.get("constraints")),
        }
    return entities


def _state_parts(value: Any) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    value = value if isinstance(value, dict) else {}
    return _non_null_mapping(value.get("constraints")), _normalize_entities(value.get("entities"))


def _flatten(
    constraints: Mapping[str, Any], entities: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    result = {str(field): copy.deepcopy(value) for field, value in constraints.items()}
    for entity_id, entity in entities.items():
        for field, value in (entity.get("constraints") or {}).items():
            result[f"entities.{entity_id}.constraints.{field}"] = copy.deepcopy(value)
    return result


def _set_path(
    constraints: MutableMapping[str, Any],
    entities: MutableMapping[str, Dict[str, Any]],
    path: str,
    value: Any,
    output_entities: Mapping[str, Mapping[str, Any]],
) -> None:
    if path.startswith(ENTITY_PREFIX) and ENTITY_MARKER in path:
        entity_id, field = path[len(ENTITY_PREFIX) :].split(ENTITY_MARKER, 1)
        if entity_id not in entities:
            source = output_entities.get(entity_id) or {}
            entities[entity_id] = {
                "reference": str(source.get("reference") or entity_id),
                "constraints": {},
            }
        entities[entity_id].setdefault("constraints", {})[field] = copy.deepcopy(value)
    else:
        constraints[path] = copy.deepcopy(value)


def _remove_path(
    constraints: MutableMapping[str, Any],
    entities: MutableMapping[str, Dict[str, Any]],
    path: str,
) -> None:
    if path.startswith(ENTITY_PREFIX) and ENTITY_MARKER in path:
        entity_id, field = path[len(ENTITY_PREFIX) :].split(ENTITY_MARKER, 1)
        if entity_id in entities:
            entities[entity_id].setdefault("constraints", {}).pop(field, None)
        return
    if path.startswith(ENTITY_PREFIX) and ENTITY_MARKER not in path:
        entities.pop(path[len(ENTITY_PREFIX) :], None)
        return
    constraints.pop(path, None)


def _existing_delta_fields(turn: Mapping[str, Any]) -> List[str]:
    delta = turn.get("gold_delta") or {}
    if not isinstance(delta, dict):
        return []
    return [path for field in delta if field != "priority" for path in [_normalize_path(field)] if path]


def _build_prompt(
    instance_id: str,
    turn_index: int,
    utterance: str,
    previous_state: Optional[Dict[str, Any]],
    existing_hint: Dict[str, Any],
    existing_delta: Dict[str, Any],
) -> str:
    context = {
        "instance_id": instance_id,
        "turn_index": turn_index,
        "previous_confirmed_state": previous_state,
        "current_user_utterance": utterance,
        "existing_generated_state_hint": existing_hint,
        "existing_generated_delta_hint": existing_delta,
    }
    return f"""
You are a meticulous TravelPlanner constraint-state annotator.
Infer the user's complete active constraints after the current utterance.
Return exactly one JSON object and no prose.

Scope rules:
- `constraints` contains requirements shared by the whole travel party.
- `entities` contains person-specific requirements under stable opaque IDs such as entity_1.
- For a one-person trip, first-person requirements are shared constraints, not entity_1 constraints.
- Preserve a previous constraint unless this utterance explicitly changes, relaxes, or removes it.
- A changed value replaces the old value. An explicit rejection/removal must be listed in removed_fields.
- Extract only user requirements. Do not infer constraints from tool results, candidate availability, or an assistant plan.
- Keep exact dates, cities, counts, budgets, named places, cuisines, transport choices, lodging rules, and schedule limits.
- `mentioned_fields` must list every active constraint explicitly stated, repeated, changed, or emphasized in this utterance, in mention order.
- Origin, destination, start date, end date, and visiting-city count are itinerary context, not constraints. Do not include org, dest, start_date, end_date, or visiting_city_number in constraints or mentioned_fields.
- days, people_number, and budget are constraints only when grounded in the user's words.
- Person-specific fields use `entities.<entity_id>.constraints.<field>` paths.
- Shared fields use plain snake_case names such as budget, transportation, or schedule.
- `removed_fields` uses the same paths and contains only explicit removals.
- The existing generated state and delta are hints, not authoritative; correct them when the utterance disagrees.
- Do not output priority. Priority is computed separately.

Required schema:
{{
  "constraints": {{"shared_field": "active value"}},
  "entities": {{
    "entity_1": {{"reference": "natural reference", "constraints": {{"field": "value"}}}}
  }},
  "mentioned_fields": ["field", "entities.entity_2.constraints.field"],
  "removed_fields": ["field"],
  "rationales": {{"field": "short utterance-grounded reason"}}
}}

INPUT_JSON:
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _call_with_retries(client: Any, prompt: str, retries: int) -> Dict[str, Any]:
    error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            result = client.generate_json(prompt)
            if not isinstance(result, dict):
                raise ValueError("LLM response must be a JSON object")
            return result
        except BaseException as exc:  # Preserve the final provider error for diagnostics.
            error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"LLM constraint extraction failed after {retries + 1} attempts") from error


def _apply_extraction(
    previous_state: Optional[Dict[str, Any]],
    existing_hint: Dict[str, Any],
    existing_delta: Dict[str, Any],
    extracted: Dict[str, Any],
    *,
    turn_index: int,
) -> Tuple[Dict[str, Any], List[str], List[str], Dict[str, str]]:
    previous_constraints, previous_entities = _state_parts(previous_state)
    hint_constraints, hint_entities = _state_parts(existing_hint)
    output_constraints, output_entities = _state_parts(extracted)

    constraints = copy.deepcopy(previous_constraints)
    entities = copy.deepcopy(previous_entities)
    output_flat = _flatten(output_constraints, output_entities)
    hint_flat = _flatten(hint_constraints, hint_entities)

    mentioned = [
        path
        for value in extracted.get("mentioned_fields") or []
        for path in [_normalize_path(value)]
        if path
    ]
    mentioned = list(dict.fromkeys(mentioned))
    removed = [
        path
        for value in extracted.get("removed_fields") or []
        for path in [_normalize_path(value)]
        if path
    ]
    removed = list(dict.fromkeys(removed))

    if turn_index == 0:
        # The initial structured record is only a value hint. A field becomes
        # a constraint only when the LLM grounds it in the actual utterance.
        # This prevents hidden org/dest/date metadata from leaking into gold.
        constraints = {}
        entities = {
            entity_id: {"reference": entity["reference"], "constraints": {}}
            for entity_id, entity in hint_entities.items()
        }
        for path in mentioned:
            if path in output_flat:
                _set_path(constraints, entities, path, output_flat[path], output_entities)
            elif path in hint_flat:
                _set_path(constraints, entities, path, hint_flat[path], hint_entities)
        for path in removed:
            _remove_path(constraints, entities, path)
    else:
        supported = list(dict.fromkeys(mentioned + removed + _existing_delta_fields({"gold_delta": existing_delta})))
        for path in supported:
            if path in removed:
                _remove_path(constraints, entities, path)
            elif path in output_flat:
                _set_path(constraints, entities, path, output_flat[path], output_entities)
            elif path in hint_flat:
                _set_path(constraints, entities, path, hint_flat[path], hint_entities)

    for entity_id, output_entity in output_entities.items():
        if entity_id in entities and output_entity.get("reference"):
            entities[entity_id]["reference"] = str(output_entity["reference"])

    for field in TRAVEL_CONTEXT_FIELDS:
        constraints.pop(field, None)
    active = set(_flatten(constraints, entities))
    mentioned = [path for path in mentioned if path in active]
    rationales = {
        path: re.sub(r"\s+", " ", str(reason)).strip()
        for raw_path, reason in (extracted.get("rationales") or {}).items()
        for path in [_normalize_path(raw_path)]
        if path and reason is not None
    } if isinstance(extracted.get("rationales"), dict) else {}
    return constraints, entities, mentioned, removed, rationales


def _make_delta(
    previous_state: Optional[Dict[str, Any]],
    current_state: Dict[str, Any],
    old_delta: Any,
    rationales: Mapping[str, str],
) -> Dict[str, Any]:
    if previous_state is None:
        return {}
    previous = _flatten(*_state_parts(previous_state))
    current = _flatten(*_state_parts(current_state))
    old_delta = old_delta if isinstance(old_delta, dict) else {}
    delta: Dict[str, Any] = {}
    for path in list(previous) + [path for path in current if path not in previous]:
        old_value = previous.get(path)
        new_value = current.get(path)
        if path in previous and path in current and old_value == new_value:
            continue
        old_change = old_delta.get(path) if isinstance(old_delta.get(path), dict) else {}
        if path not in current:
            op = "remove"
        elif path not in previous:
            op = "add"
        else:
            op = str(old_change.get("op") or "override")
            if op in REMOVAL_OPS:
                op = "override"
        delta[path] = {
            "op": op,
            "old": copy.deepcopy(old_value),
            "new": copy.deepcopy(new_value),
            "rationale": rationales.get(path) or old_change.get("rationale") or "LLM extraction from user utterance",
        }
    if isinstance(old_delta.get("priority"), dict):
        delta["priority"] = copy.deepcopy(old_delta["priority"])
    return delta


def extract_instance(
    instance: Dict[str, Any],
    client: Any,
    *,
    retries: int,
    model_name: str,
) -> Dict[str, Any]:
    result = copy.deepcopy(instance)
    previous_state: Optional[Dict[str, Any]] = None
    for turn_index, turn in enumerate(result.get("turns") or []):
        existing_hint = copy.deepcopy(turn.get("gold_current_intention") or {})
        existing_delta = copy.deepcopy(turn.get("gold_delta") or {})
        prompt = _build_prompt(
            str(result.get("instance_id") or ""),
            turn_index,
            str(turn.get("user_utterance") or ""),
            previous_state,
            existing_hint,
            existing_delta,
        )
        log_prompt(
            "travelplanner_constraint_extraction",
            prompt,
            metadata={"instance_id": result.get("instance_id"), "turn_index": turn_index},
        )
        extracted = _call_with_retries(client, prompt, retries)
        constraints, entities, mentioned, removed, rationales = _apply_extraction(
            previous_state,
            existing_hint,
            existing_delta,
            extracted,
            turn_index=turn_index,
        )
        current_state = copy.deepcopy(existing_hint)
        current_state["constraints"] = constraints
        current_state["entities"] = entities
        current_state["entity_priority"] = [
            path for path in current_state.get("entity_priority") or [] if path in _flatten(constraints, entities)
        ]
        current_state["domain"] = "travelplanner"
        turn["gold_delta"] = _make_delta(previous_state, current_state, existing_delta, rationales)
        turn["gold_current_intention"] = current_state
        turn["constraint_extraction"] = {
            "method": "llm",
            "model": model_name,
            "mentioned_fields": mentioned,
            "removed_fields": removed,
        }
        previous_state = copy.deepcopy(current_state)

    classify_instance(result)
    return result


def load_instances(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Expected a JSON list of trajectory objects")
    return value


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.parallelism < 1:
        raise ValueError("--parallelism must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env.llm")
    client = create_llm_client_from_env(timeout=args.timeout)
    model_name = str(getattr(client, "model", getattr(client, "deployment", "unknown")))
    instances = load_instances(args.input)
    completed: List[Optional[Dict[str, Any]]] = [None] * len(instances)
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = {
            executor.submit(
                extract_instance,
                instance,
                client,
                retries=args.retries,
                model_name=model_name,
            ): index
            for index, instance in enumerate(instances)
        }
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            print(f"completed={index + 1}/{len(instances)} instance_id={instances[index].get('instance_id')}", flush=True)

    save_json_atomic(args.output, completed)
    print(f"instances={len(completed)}")
    print(f"turns={sum(len(item.get('turns') or []) for item in completed if item)}")
    print(f"model={model_name}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
