from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from models import EnvFeedback, ShiftOp
from simulation.simulation.base_user_simulator import (
    REALIZATION_CONTEXT_MARKER,
    SHIFT_CONTEXT_MARKER,
    HumanSimulator,
    _clean_string,
    _safe_json_dumps,
)

from .entity_intention import (
    ensure_entity_state,
    entity_constraint_path,
    next_entity_id,
    normalize_entity_id,
    normalize_entity_priority,
    parse_entity_constraint_path,
)


ENTITY_CHANGE_CATEGORY = "entity"


@dataclass
class TravelPlannerEntityShift(ShiftOp):
    entity_id: Optional[str] = None
    entity_reference: Optional[str] = None
    replacement_entity_id: Optional[str] = None


class TravelPlannerUserSimulator(HumanSimulator):
    """TravelPlanner-only user simulator with fail-fast LLM behavior."""

    def _build_shift_prompt(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
        current_gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
        distribution_guidance: Optional[Dict[str, Any]] = None,
    ) -> str:
        normalized_current = ensure_entity_state(current_intention)
        normalized_history = []
        for turn in intention_history or []:
            normalized_turn = copy.deepcopy(turn)
            if isinstance(normalized_turn.get("gold_intention"), dict):
                normalized_turn["gold_intention"] = ensure_entity_state(
                    normalized_turn["gold_intention"]
                )
            normalized_history.append(normalized_turn)
        context = {
            "intention_timeline": self._serialize_intention_timeline(
                normalized_current,
                normalized_history,
                current_gold_delta=current_gold_delta,
            ),
            "latest_env_feedback": self._serialize_env_feedback(env_feedback),
            "entity_id_guidance": {
                "existing": list(normalized_current.get("entities") or {}),
                "next_for_add": next_entity_id(normalized_current.get("entities") or {}),
            },
        }
        if distribution_guidance:
            context["distribution_guidance"] = copy.deepcopy(distribution_guidance)
        instructions = """
Pretend you are a real user working with a travel planning assistant.
Return a single JSON object only. You MUST make one or more meaningful changes.

The intention has two scopes:
- constraints: requirements shared by the whole travel party.
- entities: individual travelers keyed by opaque stable IDs such as entity_1. Each entity has a natural-language reference and constraints.

Allowed conditions:
- user_preference
- real_world_feasibility
- agent_misunderstanding

Allowed categories:
- add
- relax
- override
- reprioritize
- entity

Rules:
- Return every change in the top-level changes array, including when there is only one change.
- Multiple changes are allowed in the same turn when they form one coherent user decision. Shared-party changes and person-specific entity changes may appear together.
- Changes are applied in array order, so a traveler may be added before a later change assigns that traveler a separate constraint.
- Do not bundle unrelated changes merely to increase the number of changes.
- Actively consider whether the next change belongs to one traveler rather than the whole party; when that is plausible, prefer category="entity" while keeping the person's reference unconstrained.
- Use category="entity" for every change whose target is an individual traveler, their identity/reference, or their separate part of the plan.
- For category="entity", use op to describe the ordinary operation: add, relax, override, reprioritize, or scope_correction.
- entity_id is an opaque internal ID. Copy an existing ID exactly, or use entity_id_guidance.next_for_add when a new traveler joins.
- Never put a relationship, name, or user wording into entity_id; IDs must remain in opaque entity_N form.
- reference is free-form text describing how the simulated user naturally refers to that person in this turn. Infer it from the evolving situation instead of selecting from a fixed vocabulary or defaulting to any relationship.
- Keep the same entity_id when its reference changes or becomes more specific across turns.
- With field=null: op=add adds a traveler; op=relax removes one; op=override plus replacement_entity_id replaces one.
- With a non-null field, op modifies only that entity's constraint. A separate activity/meal/transport plan is just an entity constraint such as activity, lunch, transportation, or schedule; it is not a separate change category.
- Rejoining a shared plan is op=relax with the separate-plan field and value=null.
- Adding/removing/replacing a traveler updates people_number deterministically.
- A person's constraints may differ from group constraints or another person's constraints.
- Use fields grounded in travel planning, such as cuisine, activity, accessibility, mobility, room_type, house_rule, transportation, budget, and schedule.
- Ground environment-driven changes in search_results, submitted_plan, or constraint_debug.
- Do not repeatedly toggle between the same two values.
- Preserve a coherent trajectory above every diversity objective: the next change must follow naturally from the current intention, earlier changes, and concrete environment evidence.
- Treat distribution_guidance only as a weak tie-breaker between changes that are already equally plausible. Ignore it when its suggested direction would invent a motive, contradict the trajectory, repeat or toggle a prior change, or fit the evidence less well.
- When compound_update_preferred_when_natural is true, prefer a multi-change update only if all changes express one natural decision. Never target a fixed number of changes.
- Never mention dataset distributions, counters, balancing, or this guidance in the user-facing rationale or utterance.

Required JSON schema:
{
  "intention_changed": true,
  "condition": "user_preference | real_world_feasibility | agent_misunderstanding",
  "changes": [
    {
      "category": "add | relax | override | reprioritize | entity",
      "op": "add | relax | override | reprioritize | scope_correction",
      "entity_id": "opaque existing/next entity_N id or null",
      "replacement_entity_id": "opaque new entity_N id for replacement or null",
      "reference": "free-form natural reference for this person or null",
      "field": "constraint field name or null",
      "old_value": "previous value or null",
      "value": "new value or null",
      "priority_update": ["entities.entity_2.constraints.mobility"] or null,
      "rationale": "short explanation"
    }
  ],
  "rationale": "short explanation",
  "utterance_plan": {
    "style": "explicit | partial | elliptical",
    "directness": "direct | indirect",
    "mention_old_value": true
  }
}

Entity example:
{
  "intention_changed": true,
  "condition": "user_preference",
  "changes": [{
    "category": "entity",
    "op": "add",
    "entity_id": "entity_2",
    "replacement_entity_id": null,
    "reference": "someone traveling with me",
    "field": "cuisine",
    "old_value": null,
    "value": ["Chinese"],
    "priority_update": null,
    "rationale": "Another traveler wants different meals."
  }],
  "rationale": "Another traveler wants different meals.",
  "utterance_plan": {"style": "partial", "directness": "direct", "mention_old_value": false}
}
""".strip()
        return f"{instructions}\n\n{SHIFT_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def _parse_shift_output(
        self,
        llm_output: Optional[Dict[str, Any]],
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> ShiftOp:
        raw_changes = (llm_output or {}).get("changes")
        if isinstance(raw_changes, list):
            return self._parse_travelplanner_multi_shift_output(
                llm_output or {},
                current_intention,
                env_feedback=env_feedback,
            )

        category = _clean_string((llm_output or {}).get("category")).lower().replace("-", "_")
        if category == "scope_correction":
            legacy_payload = copy.deepcopy(llm_output or {})
            if normalize_entity_id(legacy_payload.get("entity_id")):
                legacy_payload["category"] = ENTITY_CHANGE_CATEGORY
                legacy_payload.setdefault("op", "scope_correction")
                llm_output = legacy_payload
                category = ENTITY_CHANGE_CATEGORY
            else:
                legacy_payload["category"] = "override"
                legacy_payload["op"] = "override"
                return super()._parse_shift_output(
                    legacy_payload,
                    current_intention,
                    env_feedback,
                )
        if category != ENTITY_CHANGE_CATEGORY:
            return super()._parse_shift_output(llm_output, current_intention, env_feedback)

        payload = llm_output or {}
        if payload.get("intention_changed") is False:
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale=_clean_string(payload.get("rationale")) or "no_change")
        condition = self._normalize_shift_condition(payload.get("condition"))
        if condition == "none":
            condition = "user_preference"
        op = self._normalize_change_category(payload.get("op"))
        if op == "none":
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_entity_op")
        entity_id = normalize_entity_id(payload.get("entity_id"))
        if not entity_id:
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale="invalid_entity_id",
            )

        normalized = ensure_entity_state(current_intention)
        entities = normalized["entities"]
        expected_new_id = next_entity_id(entities)
        reference = _clean_string(payload.get("reference")) or None
        rationale = _clean_string(payload.get("rationale")) or "entity_level_change"
        utterance_plan = self._normalize_utterance_plan(payload.get("utterance_plan"))
        field = normalize_entity_id(payload.get("field"))
        value = copy.deepcopy(payload.get("value"))
        replacement_id = normalize_entity_id(payload.get("replacement_entity_id"))

        if field:
            if entity_id not in entities and not (op == "add" and entity_id == expected_new_id):
                return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="unknown_entity")
            path = entity_constraint_path(entity_id, field)
            old_value = (
                (entities[entity_id].get("constraints") or {}).get(field)
                if entity_id in entities
                else None
            )
            if op == "add" and old_value is not None:
                op = "override"
            if op in {"add", "override", "scope_correction"} and value is None:
                return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="missing_entity_value")
            priority_update = payload.get("priority_update") if op == "reprioritize" else None
            if op == "reprioritize" and (not isinstance(priority_update, list) or not priority_update):
                priority_update = [path]
            return TravelPlannerEntityShift(
                op=op,
                intention_changed=True,
                condition=condition,
                change_category=ENTITY_CHANGE_CATEGORY,
                field=path,
                old_value=copy.deepcopy(old_value),
                value=value,
                rationale=rationale,
                priority_update=copy.deepcopy(priority_update),
                utterance_plan=utterance_plan,
                entity_id=entity_id,
                entity_reference=reference,
            )

        if op == "add":
            if entity_id in entities or entity_id != expected_new_id:
                return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_new_entity_id")
            entity_value = value if isinstance(value, dict) else {}
            return TravelPlannerEntityShift(op=op, intention_changed=True, condition=condition, change_category=ENTITY_CHANGE_CATEGORY, field=f"entities.{entity_id}", old_value=None, value=copy.deepcopy(entity_value), rationale=rationale, utterance_plan=utterance_plan, entity_id=entity_id, entity_reference=reference)

        if entity_id not in entities:
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="unknown_entity")
        if op == "relax":
            if len(entities) <= 1:
                return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="cannot_remove_last_entity")
            return TravelPlannerEntityShift(op=op, intention_changed=True, condition=condition, change_category=ENTITY_CHANGE_CATEGORY, field=f"entities.{entity_id}", old_value=copy.deepcopy(entities[entity_id]), value=None, rationale=rationale, utterance_plan=utterance_plan, entity_id=entity_id, entity_reference=reference)
        if op == "override":
            if not replacement_id or replacement_id in entities or replacement_id != expected_new_id:
                return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_entity_replacement")
            replacement_value = value if isinstance(value, dict) else {}
            return TravelPlannerEntityShift(op=op, intention_changed=True, condition=condition, change_category=ENTITY_CHANGE_CATEGORY, field=f"entities.{entity_id}", old_value=copy.deepcopy(entities[entity_id]), value=copy.deepcopy(replacement_value), rationale=rationale, utterance_plan=utterance_plan, entity_id=entity_id, entity_reference=reference, replacement_entity_id=replacement_id)
        if op == "scope_correction" and reference:
            return TravelPlannerEntityShift(op=op, intention_changed=True, condition=condition, change_category=ENTITY_CHANGE_CATEGORY, field=f"entities.{entity_id}.reference", old_value=entities[entity_id].get("reference"), value=reference, rationale=rationale, utterance_plan=utterance_plan, entity_id=entity_id, entity_reference=reference)
        return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="entity_field_required")

    def _parse_travelplanner_multi_shift_output(
        self,
        llm_output: Dict[str, Any],
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> ShiftOp:
        """Parse ordered shared-party and entity changes from one user turn."""
        if llm_output.get("intention_changed") is False:
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale=_clean_string(llm_output.get("rationale")) or "no_change",
            )

        working_intention = ensure_entity_state(current_intention)
        parsed_changes: List[ShiftOp] = []
        common_condition = llm_output.get("condition")
        common_plan = llm_output.get("utterance_plan")
        for raw_change in llm_output.get("changes") or []:
            if not isinstance(raw_change, dict):
                continue
            child_payload = copy.deepcopy(raw_change)
            child_payload["intention_changed"] = True
            child_payload.setdefault("condition", common_condition)
            child_payload.setdefault("utterance_plan", common_plan)
            child = self._parse_shift_output(
                child_payload,
                working_intention,
                env_feedback=env_feedback,
            )
            if child.op == "none":
                continue
            updated_intention, child_delta = self.apply_shift(working_intention, child)
            if not child_delta:
                continue
            parsed_changes.append(child)
            working_intention = updated_intention

        if not parsed_changes:
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale="invalid_llm_output",
            )
        if len(parsed_changes) == 1:
            return parsed_changes[0]

        first = parsed_changes[0]
        normalized_condition = self._normalize_shift_condition(common_condition)
        if normalized_condition == "none":
            normalized_condition = str(first.condition or "user_preference")
        return ShiftOp(
            op="multiple",
            intention_changed=True,
            condition=normalized_condition,
            change_category="multiple",
            rationale=_clean_string(llm_output.get("rationale")) or first.rationale,
            utterance_plan=self._normalize_utterance_plan(common_plan) or first.utterance_plan,
            changes=parsed_changes,
        )

    def apply_shift(
        self,
        current_intention: Dict[str, Any],
        shift: ShiftOp,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if shift.changes:
            new_state = ensure_entity_state(current_intention)
            combined_delta: Dict[str, Dict[str, Any]] = {}
            for change in shift.changes:
                new_state, child_delta = self.apply_shift(new_state, change)
                combined_delta.update(child_delta)
            return new_state, combined_delta

        if shift.change_category != ENTITY_CHANGE_CATEGORY:
            return super().apply_shift(current_intention, shift)

        new_state = ensure_entity_state(current_intention)
        entities = new_state["entities"]
        delta: Dict[str, Dict[str, Any]] = {}
        field = str(shift.field or "")
        entity_id = normalize_entity_id(getattr(shift, "entity_id", None))
        reference = _clean_string(getattr(shift, "entity_reference", None)) or None
        membership_changed = False

        if field.startswith("entities.") and ".constraints." not in field:
            if not entity_id:
                return new_state, delta
            if field.endswith(".reference"):
                old_reference = entities[entity_id].get("reference")
                entities[entity_id]["reference"] = reference or str(shift.value)
                delta[field] = {"op": shift.op, "category": ENTITY_CHANGE_CATEGORY, "old": old_reference, "new": entities[entity_id]["reference"], "rationale": shift.rationale}
            elif shift.op == "add":
                entity_value = copy.deepcopy(shift.value) if isinstance(shift.value, dict) else {}
                constraints = entity_value.get("constraints")
                entity_value["constraints"] = constraints if isinstance(constraints, dict) else {}
                entity_value["reference"] = reference or str(entity_value.get("reference") or "another traveler")
                entities[entity_id] = entity_value
                membership_changed = True
            elif shift.op == "relax":
                entities.pop(entity_id, None)
                membership_changed = True
            elif shift.op == "override":
                replacement = copy.deepcopy(shift.value) if isinstance(shift.value, dict) else {}
                replacement_id = normalize_entity_id(getattr(shift, "replacement_entity_id", None))
                if replacement_id:
                    entities.pop(entity_id, None)
                    constraints = replacement.get("constraints")
                    replacement["constraints"] = constraints if isinstance(constraints, dict) else {}
                    replacement["reference"] = reference or str(replacement.get("reference") or "another traveler")
                    entities[replacement_id] = replacement
            if not field.endswith(".reference"):
                delta[field] = {"op": shift.op, "category": ENTITY_CHANGE_CATEGORY, "old": copy.deepcopy(shift.old_value), "new": copy.deepcopy(shift.value), "rationale": shift.rationale}
        else:
            parsed = parse_entity_constraint_path(field)
            if not parsed:
                return new_state, delta
            entity_id, constraint_field = parsed
            if entity_id not in entities:
                entities[entity_id] = {
                    "reference": reference or "another traveler",
                    "constraints": {},
                }
                membership_changed = True
            elif reference:
                entities[entity_id]["reference"] = reference
            entity_constraints = entities[entity_id].setdefault("constraints", {})
            if shift.op == "reprioritize":
                old_priority = list(new_state.get("entity_priority") or [])
                requested = list(shift.priority_update or [field])
                remaining = [path for path in old_priority if path not in requested]
                new_state["entity_priority"] = normalize_entity_priority(requested + remaining, entities)
                delta["entity_priority"] = {"op": "reprioritize", "category": ENTITY_CHANGE_CATEGORY, "old": old_priority, "new": list(new_state["entity_priority"]), "rationale": shift.rationale}
            else:
                old_value = entity_constraints.get(constraint_field)
                if shift.op == "relax" and shift.value is None:
                    entity_constraints.pop(constraint_field, None)
                else:
                    entity_constraints[constraint_field] = copy.deepcopy(shift.value)
                delta[field] = {"op": shift.op, "category": ENTITY_CHANGE_CATEGORY, "old": copy.deepcopy(old_value), "new": copy.deepcopy(shift.value), "rationale": shift.rationale}

        if membership_changed:
            constraints = new_state.setdefault("constraints", {})
            old_count = constraints.get("people_number", constraints.get("party_size"))
            constraints["people_number"] = len(entities)
            if old_count != len(entities):
                delta["people_number"] = {"op": "scope_correction", "category": ENTITY_CHANGE_CATEGORY, "old": old_count, "new": len(entities), "rationale": "synchronized with travel-party entities"}

        new_state["entity_priority"] = normalize_entity_priority(new_state.get("entity_priority"), entities)
        return new_state, delta

    def _build_realization_prompt(
        self,
        shift: ShiftOp,
        current_intention: Dict[str, Any],
        style: str,
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
        current_gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        if shift.changes:
            normalized_current = ensure_entity_state(current_intention)
            context = {
                "requested_style": style,
                "intention_timeline": self._serialize_intention_timeline(
                    normalized_current,
                    intention_history,
                    current_gold_delta=current_gold_delta,
                ),
                "shift": asdict(shift),
                "latest_env_feedback": self._serialize_env_feedback(env_feedback),
            }
            instructions = """
Write the user's next utterance as one concise, natural sentence or tightly connected pair of sentences.
Express every entry in shift.changes, preserving their array order and combining them as one coherent decision.
Do not invent changes that are absent from shift.changes.
For entity changes, use entity_reference to identify the traveler naturally and preserve ownership of that person's constraint.
Never expose opaque IDs such as entity_1 in the utterance.
Shared-party constraints must remain shared; person-specific constraints must remain assigned only to that person.
Follow requested_style: explicit is direct, partial is natural but incomplete, and elliptical is short/fragment-like.
Return plain text only, with no quotes and no JSON.
""".strip()
            return f"{instructions}\n\n{REALIZATION_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

        if shift.change_category != ENTITY_CHANGE_CATEGORY:
            return super()._build_realization_prompt(
                shift,
                current_intention,
                style,
                env_feedback=env_feedback,
                intention_history=intention_history,
                current_gold_delta=current_gold_delta,
            )
        normalized_current = ensure_entity_state(current_intention)
        context = {
            "requested_style": style,
            "intention_timeline": self._serialize_intention_timeline(
                normalized_current,
                intention_history,
                current_gold_delta=current_gold_delta,
            ),
            "shift": asdict(shift),
            "latest_env_feedback": self._serialize_env_feedback(env_feedback),
        }
        instructions = """
Write the user's next utterance as one concise, natural sentence.
The shift has category=entity. Use entity_reference to identify the affected traveler naturally.
Never expose opaque IDs such as entity_1 in the utterance.
Preserve ownership of the preference; do not turn a person's constraint into a whole-group constraint.
- When field names the whole entity, infer joins/leaves/replacement from op.
- When field is a constraint path, say whose preference or need was added, relaxed, corrected, overridden, or reprioritized.
- A person-specific activity, meal, transportation, or schedule constraint should sound like a natural separate-plan request.
- Relaxing such a field to null should sound like rejoining or dropping that separate requirement.
Ground the sentence only in the shift object. Do not invent another change.
Follow requested_style: explicit is direct, partial is natural but incomplete, and elliptical is short/fragment-like.
Return plain text only, with no quotes and no JSON.
""".strip()
        return f"{instructions}\n\n{REALIZATION_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def _infer_domain(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> str:
        return "travelplanner"


__all__ = ["TravelPlannerUserSimulator"]
