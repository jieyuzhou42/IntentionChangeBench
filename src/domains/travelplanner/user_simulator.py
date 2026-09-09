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

from .date_utils import inclusive_date_count
from .entity_intention import (
    ensure_entity_state,
    entity_constraint_path,
    next_entity_id,
    normalize_entity_id,
    normalize_entity_priority,
    parse_entity_constraint_path,
)


ENTITY_CHANGE_CATEGORY = "entity"
TRAVELPLANNER_SHIFT_RETRY_LIMIT = 8
FORBIDDEN_CORRECTION_TERMS = (
    "validator",
    "participant assignment",
    "participant_assignment",
    "exact listed wording",
    "exact wording",
    "verifiable address",
    "search-result provenance",
    "field name",
    "scope correction",
)


@dataclass
class TravelPlannerEntityShift(ShiftOp):
    entity_id: Optional[str] = None
    entity_reference: Optional[str] = None
    replacement_entity_id: Optional[str] = None


class TravelPlannerUserSimulator(HumanSimulator):
    """TravelPlanner-only user simulator with fail-fast LLM behavior."""

    def decide_shift(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
        current_gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
        candidate_samples: int = 1,
        max_candidate_samples: Optional[int] = None,
        prefer_multi: bool = False,
        rng: Optional[Any] = None,
        distribution_controller: Optional[Any] = None,
    ) -> ShiftOp:
        shift = super().decide_shift(
            current_intention,
            env_feedback=env_feedback,
            intention_history=intention_history,
            current_gold_delta=current_gold_delta,
            candidate_samples=candidate_samples,
            max_candidate_samples=max(
                TRAVELPLANNER_SHIFT_RETRY_LIMIT,
                int(max_candidate_samples or 1),
            ),
            prefer_multi=prefer_multi,
            rng=rng,
            distribution_controller=distribution_controller,
        )
        if shift.op == "none":
            raise ValueError(
                "TravelPlanner user simulator could not produce a genuine intention change "
                f"after {TRAVELPLANNER_SHIFT_RETRY_LIMIT} attempts."
            )
        return shift

    def _postprocess_shift_candidate(
        self,
        llm_output: Dict[str, Any],
        shift: ShiftOp,
        *,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback],
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> ShiftOp:
        changes = shift.changes or [shift]
        if shift.condition == "agent_misunderstanding" or any(
            change.condition == "agent_misunderstanding"
            or change.op == "scope_correction"
            for change in changes
        ):
            return self._rejected_shift("correction_is_not_intention_change")

        candidate_text = _safe_json_dumps(llm_output).lower()
        if any(term in candidate_text for term in FORBIDDEN_CORRECTION_TERMS):
            return self._rejected_shift("validator_or_execution_correction")

        tradeoff = self._normalize_tradeoff_metadata(llm_output.get("tradeoff"))
        shift.tradeoff = tradeoff
        if tradeoff["is_tradeoff"]:
            rejection_reason = self._tradeoff_rejection_reason(shift, tradeoff)
            if rejection_reason:
                return self._rejected_shift(rejection_reason)
            tradeoff["validated"] = True
            tradeoff["actual_targets"] = sorted(self._shift_targets(shift))
            shift.sampling_metadata["tradeoff"] = copy.deepcopy(tradeoff)
        elif self._tradeoff_required_now(intention_history):
            return self._rejected_shift("required_substantive_tradeoff_missing")
        else:
            active_conflict = self._active_conflict_requiring_tradeoff(
                shift,
                current_intention,
                intention_history,
            )
            if active_conflict:
                return self._rejected_shift(
                    f"active_constraint_conflict_requires_tradeoff:{active_conflict}"
                )

        targets = self._shift_targets(shift)
        recent_targets = [
            set(str(target) for target in turn.get("shift_targets") or [] if target)
            for turn in (intention_history or [])[-2:]
            if isinstance(turn, dict) and turn.get("shift_targets")
        ]
        if (
            targets
            and len(recent_targets) == 2
            and targets == recent_targets[-1] == recent_targets[-2]
        ):
            return self._rejected_shift("same_target_three_turns_in_a_row")
        prior_target_counts = {
            target: sum(
                target
                in {
                    str(item)
                    for item in turn.get("shift_targets") or []
                    if item
                }
                for turn in intention_history or []
                if isinstance(turn, dict)
            )
            for target in targets
        }
        if any(count >= 2 for count in prior_target_counts.values()):
            return self._rejected_shift("target_already_changed_twice")

        latest_delta = {}
        if intention_history and isinstance(intention_history[-1], dict):
            latest_delta = intention_history[-1].get("gold_delta") or {}
        if any(
            change.field in latest_delta
            and str((latest_delta.get(change.field) or {}).get("op")) == change.op
            for change in changes
            if change.field
        ):
            return self._rejected_shift("same_operation_on_same_target_as_previous_turn")

        recent_categories = [
            str(turn.get("change_category") or "")
            for turn in (intention_history or [])[-3:]
            if isinstance(turn, dict) and turn.get("change_category")
        ]
        candidate_categories = {
            str(change.change_category or change.op)
            for change in changes
        }
        if (
            len(recent_categories) == 3
            and all(category == "add" for category in recent_categories)
            and candidate_categories == {"add"}
        ):
            return self._rejected_shift("too_many_consecutive_add_turns")
        return shift

    @staticmethod
    def _normalize_tradeoff_metadata(raw_tradeoff: Any) -> Dict[str, Any]:
        payload = raw_tradeoff if isinstance(raw_tradeoff, dict) else {}

        def clean_list(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            cleaned: List[str] = []
            for item in value:
                text = _clean_string(item)
                if text and text not in cleaned:
                    cleaned.append(text)
            return cleaned

        raw_flag = payload.get("is_tradeoff", False)
        is_tradeoff = (
            raw_flag
            if isinstance(raw_flag, bool)
            else _clean_string(raw_flag).lower() in {"true", "yes", "1"}
        )
        return {
            "is_tradeoff": bool(is_tradeoff),
            "conflicting_constraints": clean_list(
                payload.get("conflicting_constraints")
            ),
            "chosen_priority": _clean_string(payload.get("chosen_priority")),
            "concession": _clean_string(payload.get("concession")),
            "plan_impacts": clean_list(payload.get("plan_impacts")),
            "evidence": _clean_string(payload.get("evidence")),
            "validated": False,
        }

    def _tradeoff_rejection_reason(
        self,
        shift: ShiftOp,
        tradeoff: Dict[str, Any],
    ) -> Optional[str]:
        if len(tradeoff["conflicting_constraints"]) < 2:
            return "tradeoff_needs_two_conflicting_constraints"
        if len(tradeoff["plan_impacts"]) < 2:
            return "tradeoff_needs_multiple_plan_impacts"
        if not tradeoff["chosen_priority"]:
            return "tradeoff_missing_chosen_priority"
        if not tradeoff["concession"]:
            return "tradeoff_missing_concession"
        if not tradeoff["evidence"]:
            return "tradeoff_missing_plan_evidence"

        changes = shift.changes or [shift]
        targets = self._shift_targets(shift)
        if len(changes) < 2 or len(targets) < 2:
            return "tradeoff_needs_multiple_actual_targets"

        reprioritize_changes = [
            change for change in changes if change.op == "reprioritize"
        ]
        if not reprioritize_changes:
            return "tradeoff_needs_explicit_reprioritization"
        non_priority_targets = {
            str(change.field)
            for change in changes
            if change.op != "reprioritize" and change.field
        }
        priority_targets = {
            str(
                change.field
                or (
                    change.priority_update[0]
                    if change.priority_update
                    else ""
                )
            )
            for change in reprioritize_changes
        }
        if not non_priority_targets or not any(
            priority_target
            and non_priority_target != priority_target
            for priority_target in priority_targets
            for non_priority_target in non_priority_targets
        ):
            return "tradeoff_needs_priority_and_concession_on_distinct_targets"
        return None

    @staticmethod
    def _history_has_validated_tradeoff(
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> bool:
        for turn in intention_history or []:
            if not isinstance(turn, dict):
                continue
            candidates = [turn.get("tradeoff")]
            shift_condition = turn.get("shift_condition")
            if isinstance(shift_condition, dict):
                details = shift_condition.get("details")
                if isinstance(details, dict):
                    candidates.append(details.get("tradeoff"))
            for candidate in candidates:
                if (
                    isinstance(candidate, dict)
                    and candidate.get("is_tradeoff") is True
                    and candidate.get("validated") is True
                ):
                    return True
        return False

    @staticmethod
    def _history_has_complex_tradeoff(
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> bool:
        for turn in intention_history or []:
            if not isinstance(turn, dict):
                continue
            candidates = [turn.get("tradeoff")]
            shift_condition = turn.get("shift_condition")
            if isinstance(shift_condition, dict):
                details = shift_condition.get("details")
                if isinstance(details, dict):
                    candidates.append(details.get("tradeoff"))
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if (
                    candidate.get("is_tradeoff") is True
                    and candidate.get("validated") is True
                    and len(candidate.get("plan_impacts") or []) >= 3
                    and len(candidate.get("actual_targets") or []) >= 3
                ):
                    return True
        return False

    @staticmethod
    def _history_has_participant_change(
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> bool:
        for turn in intention_history or []:
            if not isinstance(turn, dict):
                continue
            targets = {
                str(target)
                for target in turn.get("shift_targets") or []
                if target
            }
            if any(target.startswith("entities.") for target in targets):
                return True
            delta = turn.get("gold_delta") or {}
            if isinstance(delta, dict) and any(
                str(field).startswith("entities.") for field in delta
            ):
                return True
        return False

    def _tradeoff_required_now(
        self,
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> bool:
        # run_instance supplies a turn-0 history entry. Requiring the first
        # accepted shift to be substantive gives every successful trajectory a
        # hard trade-off guarantee while keeping direct parser/unit callers
        # backward compatible when they do not provide trajectory history.
        has_trajectory_start = any(
            isinstance(turn, dict) and turn.get("turn_id") == 0
            for turn in intention_history or []
        )
        return has_trajectory_start and not self._history_has_validated_tradeoff(
            intention_history
        )

    def _active_conflict_requiring_tradeoff(
        self,
        shift: ShiftOp,
        current_intention: Dict[str, Any],
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> Optional[str]:
        # Apply this stricter coupling check only inside a real trajectory. It
        # prevents a later single-field preference from silently breaking an
        # already active budget, schedule, flight, lodging, or pace constraint.
        if not any(
            isinstance(turn, dict) and turn.get("turn_id") == 0
            for turn in intention_history or []
        ):
            return None

        targets = self._shift_targets(shift)
        constraints = (current_intention or {}).get("constraints") or {}
        active = set(str(field) for field in constraints)

        accommodation_targets = {
            "accommodation",
            "accommodation_rating",
            "room_type",
            "house_rule",
            "accommodation_nights",
        }
        flight_targets = {"outbound_transportation", "return_transportation"}
        schedule_targets = {
            "schedule",
            "activity",
            "daily_attraction_limit",
            "max_daily_attractions",
            "first_day_activity_limit",
            "final_day_schedule",
        }
        date_targets = {"days", "start_date", "end_date"}
        destination_targets = {
            "dest",
            "required_cities",
            "visiting_city_number",
        }
        origin_targets = {"org"}
        location_couplings = (
            accommodation_targets
            | flight_targets
            | schedule_targets
            | date_targets
            | {
                "budget",
                "cuisine",
                "dining_style",
                "local_transportation",
            }
        )

        if targets & accommodation_targets and "budget" in active:
            return "accommodation_vs_budget"
        if targets & flight_targets and active & (schedule_targets | date_targets):
            return "flight_time_vs_schedule"
        if targets & schedule_targets and active & flight_targets:
            return "schedule_vs_transportation"
        if targets & date_targets and active & (
            accommodation_targets | flight_targets | schedule_targets | {"budget"}
        ):
            return "trip_length_vs_lodging_transport_budget"
        if targets & destination_targets and active & location_couplings:
            return "destination_scope_vs_budget_transport_schedule"
        if targets & origin_targets and active & (
            flight_targets
            | date_targets
            | {"budget", "local_transportation", "dest"}
        ):
            return "origin_vs_airfare_time_and_ground_transfer"
        if "restaurant_rating" in targets and active & {
            "budget",
            "cuisine",
            "dining_style",
        }:
            return "restaurant_rating_vs_price_or_variety"
        if any(target.startswith("entities.") for target in targets) and len(
            (current_intention or {}).get("entities") or {}
        ) > 1:
            return "participant_preference_vs_shared_itinerary"
        return None

    def _should_resample_shift_candidate(
        self,
        shift: ShiftOp,
        *,
        sampled_count: int,
        sample_limit: int,
    ) -> bool:
        return shift.op == "none" and sampled_count < sample_limit

    @staticmethod
    def _rejected_shift(rationale: str) -> ShiftOp:
        return ShiftOp(
            op="none",
            intention_changed=False,
            condition="none",
            change_category="none",
            rationale=rationale,
        )

    @staticmethod
    def _shift_targets(shift: ShiftOp) -> set[str]:
        targets: set[str] = set()
        for change in shift.changes or [shift]:
            target = change.field
            if change.op == "reprioritize" and not target and change.priority_update:
                target = str(change.priority_update[0])
            if target:
                targets.add(str(target))
        return targets

    def _parse_shared_shift_output(
        self,
        payload: Dict[str, Any],
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback],
    ) -> ShiftOp:
        shift = super()._parse_shift_output(payload, current_intention, env_feedback)
        constraints = (current_intention or {}).get("constraints") or {}
        if (
            shift.op == "relax"
            and shift.field in {"days", "start_date", "end_date"}
            and shift.value is None
        ):
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale="persistent_date_constraint",
            )
        if (
            shift.field == "days"
            and shift.op != "none"
            and constraints.get("start_date")
            and constraints.get("end_date")
            and not payload.get("_compound_child")
        ):
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale="duration_change_requires_date_update",
            )
        if (
            shift.field in {"days", "start_date", "end_date"}
            and shift.op != "none"
            and not payload.get("_compound_child")
        ):
            updated_intention, delta = self.apply_shift(current_intention, shift)
            if delta and not self._date_constraints_are_consistent(updated_intention):
                return ShiftOp(
                    op="none",
                    intention_changed=False,
                    condition="none",
                    change_category="none",
                    rationale="date_change_requires_consistent_range_and_duration",
                )
        if shift.op != "none":
            self._normalize_shared_travel_semantics(shift, current_intention)
        return shift

    @staticmethod
    def _date_constraints_are_consistent(intention: Dict[str, Any]) -> bool:
        constraints = (intention or {}).get("constraints") or {}
        start = constraints.get("start_date")
        end = constraints.get("end_date")
        days = constraints.get("days")
        if start is None or end is None:
            return True
        count = inclusive_date_count(start, end)
        if count is None:
            return False
        if days is None:
            return True
        try:
            return count == int(days)
        except (TypeError, ValueError):
            return False

    def _normalize_shared_travel_semantics(
        self,
        shift: ShiftOp,
        current_intention: Dict[str, Any],
    ) -> None:
        """Keep free-form preferences out of strict canonical validator fields."""
        field = str(shift.field or "")
        value_text = _clean_string(shift.value).lower()

        if field in {"room_type", "room type"}:
            if "entire" in value_text or "whole apartment" in value_text:
                shift.value = "Entire home/apt"
            elif "private" in value_text:
                shift.value = "Private room"
            elif "shared" in value_text:
                shift.value = "Shared room"
            shift.field = "room_type"
            return

        remapped_field: Optional[str] = None
        if field == "cuisine" and any(
            cue in value_text
            for cue in (
                "sit-down",
                "sit down",
                "local restaurant",
                "fast-food",
                "fast food",
                "service style",
            )
        ):
            remapped_field = "dining_style"
        elif field == "transportation":
            if "flight" in value_text:
                remapped_field = (
                    "return_transportation"
                    if any(cue in value_text for cue in ("return", "back to", "flight home"))
                    else "outbound_transportation"
                )
            elif any(
                cue in value_text
                for cue in (
                    "public transit",
                    "getting around",
                    "local transportation",
                    "within ",
                )
            ):
                remapped_field = "local_transportation"
        elif field == "accommodation" and any(
            cue in value_text for cue in ("rated", "rating", "review")
        ):
            remapped_field = "accommodation_rating"

        if not remapped_field:
            return
        constraints = (current_intention or {}).get("constraints") or {}
        old_value = constraints.get(remapped_field)
        shift.field = remapped_field
        shift.old_value = copy.deepcopy(old_value)
        if shift.op != "reprioritize":
            if old_value is None:
                shift.op = "add"
                shift.change_category = "add"
            elif shift.op == "add":
                shift.op = "override"
                shift.change_category = "override"

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
        tradeoff_already_satisfied = self._history_has_validated_tradeoff(
            normalized_history
        )
        tradeoff_required_now = self._tradeoff_required_now(normalized_history)
        complex_tradeoff_already_satisfied = self._history_has_complex_tradeoff(
            normalized_history
        )
        party_size = len(normalized_current.get("entities") or {})
        participant_change_already_satisfied = self._history_has_participant_change(
            normalized_history
        )
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
            "party_size": party_size,
            "recent_shift_conditions": [
                {
                    "turn_id": turn.get("turn_id"),
                    "condition": turn.get("shift_condition"),
                    "category": turn.get("change_category"),
                    "rationale": turn.get("shift_rationale"),
                    "tradeoff": copy.deepcopy(turn.get("tradeoff")),
                }
                for turn in normalized_history[-4:]
                if turn.get("shift_condition")
            ],
            "trajectory_tradeoff_requirement": {
                "already_satisfied": tradeoff_already_satisfied,
                "required_now": tradeoff_required_now,
                "rule": (
                    "Every trajectory must contain at least one validated "
                    "multi-constraint trade-off."
                ),
            },
            "trajectory_complexity_requirement": {
                "already_satisfied": complex_tradeoff_already_satisfied,
                "required_now": (
                    tradeoff_required_now
                    and not complex_tradeoff_already_satisfied
                ),
                "rule": (
                    "Every trajectory should contain a high-order trade-off "
                    "that changes at least three coupled trip components."
                ),
            },
            "trajectory_participant_requirement": {
                "applies": party_size >= 2,
                "already_satisfied": participant_change_already_satisfied,
                "required_now": (
                    party_size >= 2
                    and not participant_change_already_satisfied
                ),
                "rule": (
                    "A multi-traveler trajectory must surface at least one "
                    "traveler-specific need that conflicts with the shared itinerary."
                ),
            },
        }
        if distribution_guidance:
            safe_guidance = copy.deepcopy(distribution_guidance)
            safe_guidance["preferred_conditions_when_natural"] = [
                item
                for item in safe_guidance.get(
                    "preferred_conditions_when_natural", []
                )
                if item != "agent_misunderstanding"
            ]
            context["distribution_guidance"] = safe_guidance
        instructions = """
Pretend you are a real user working with a travel planning assistant.
Return a single JSON object only. You MUST make one or more meaningful changes.

The intention has two scopes:
- constraints: requirements shared by the whole travel party.
- entities: individual travelers keyed by opaque stable IDs such as entity_1. Each entity has a natural-language reference and constraints.

Allowed conditions:
- user_preference
- real_world_feasibility

Allowed categories:
- add
- relax
- override
- reprioritize
- entity

Rules:
- Return every change in the top-level changes array, including when there is only one change.
- Every turn must change what trip the user wants after seeing the latest itinerary. A request to repair, restate, clarify, enforce, or reformat an unchanged requirement is not an intention change and is forbidden.
- The latest itinerary is the trigger, not the object to debug: notice a concrete option or tradeoff in it, then make a new preference or feasibility decision about the future itinerary.
- Multiple changes are allowed in the same turn when they form one coherent user decision. Shared-party changes and person-specific entity changes may appear together.
- Changes are applied in array order, so a traveler may be added before a later change assigns that traveler a separate constraint.
- Do not bundle unrelated changes merely to increase the number of changes.
- Use category="entity" only when two or more travelers have genuinely different needs, or when a traveler joins, leaves, or is replaced.
- In a one-person trip, I/me/my refers to the whole itinerary. Represent room type, dining, activities, flights, and schedule as top-level shared constraints, never as entities.entity_1 constraints or scope corrections.
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
- A one-person trip may become a multi-person trip when someone naturally joins after seeing the itinerary. Add the traveler first, then add their individual need in the same changes array when appropriate; do not invent a companion merely to manipulate people_number.
- Use fields grounded in travel planning, including org, dest, required_cities, visiting_city_number, start_date, end_date, days, cuisine, activity, accessibility, mobility, room_type, house_rule, transportation, budget, and schedule.
- Use cuisine only for actual cuisines such as Italian or Chinese. Use dining_style for preferences such as local, sit-down, or avoiding fast food.
- Use room_type only with a canonical value: Entire home/apt, Private room, or Shared room.
- Use outbound_transportation or return_transportation for a named or timed flight. Use local_transportation for getting around within the destination.
- Use accommodation_rating and restaurant_rating for review thresholds instead of putting rating prose into accommodation, dining, or cuisine.
- Ground environment-driven changes in actual travel availability from search_results or submitted_plan. constraint_debug is diagnostic metadata, not a user motive.
- Never correct field names, scopes, participant assignments, exact wording, search-result provenance, addresses, or itinerary formatting.
- Never restate, make mandatory, re-scope, or rename the same requirement in later turns. Each turn must be a real change in what trip the user wants.
- Do not target the same travel dimension for a third consecutive turn. After at most one follow-up tradeoff on a dimension, react to a different part of the latest itinerary.
- Across the whole trajectory, change any one travel dimension at most twice. One follow-up relaxation or override is enough; do not return to that dimension a third time.
- Never apply the same operation to the same travel dimension in two consecutive turns. Combine that decision into one turn instead.
- A requirement remains in the intention after the itinerary satisfies it. Relax or remove it only when the user genuinely no longer wants it, never because the submitted plan currently complies.
- Treat the dataset org/dest fields as the authoritative current baseline, not as immutable forever. Never infer a city correction from an accommodation name or suspicious listing wording. Change org or dest only when the user intentionally chooses a different origin, airport, city, or region because of concrete itinerary/search evidence, and use an option supported by the available travel evidence.
- Dates remain active unless the user explicitly overrides them. Never relax days/start_date/end_date to null. A date shift must keep start_date, end_date, and days mutually consistent; update every affected date field in the same changes array. A duration change must update end_date (and any other affected date field) in that array.
- Destination and trip-scope changes are valid intention changes: use dest to replace the destination city/region, required_cities to choose specific cities, and visiting_city_number to add or remove cities. Keep these fields mutually consistent and replan the whole route.
- Origin changes are valid intention changes: use org when the user chooses a different departure city or airport, then reconsider airfare, departure/arrival time, and ground transfer.
- Time changes are valid intention changes: the user may move start_date/end_date earlier or later, or shorten/extend days, because of airfare, lodging availability, opening days, or traveler schedules.
- When an origin, destination, date, duration, or city-scope change makes an active named flight, stay, attraction, meal, or schedule obsolete, include an override or relax change for every obsolete field in the same changes array. Never carry an old route-specific or date-specific constraint forward merely because it was previously active.
- Increasing a maximum travel budget is category/op=relax; decreasing it is override.
- Reprioritize must actually move the named top priority to the first position. Never emit the current priority unchanged.
- After proposing any new or modified requirement, check it against the active budget, dates, flight times, local travel time, lodging, meals, activities, and each traveler's needs.
- A genuine trade-off is one decision in which at least two constraints conflict, the user explicitly chooses which matters more, accepts a concrete concession, and causes replanning across at least two components of the itinerary.
- Read trajectory_tradeoff_requirement.required_now. When it is true, this turn MUST be a genuine trade-off and MUST include: (1) at least two distinct changed fields, (2) at least one non-reprioritize change, (3) a reprioritize change that moves the chosen priority to the front, and (4) the complete tradeoff object below. A single-field tweak, several unrelated additions, or metadata without matching changes is invalid.
- Read trajectory_complexity_requirement.required_now. When it is true, prefer one high-order conflict spanning at least three actual targets and three plan impacts, such as traveler schedule + flight + budget + activities, or lodging + meals + transportation + budget. The resulting request should force a substantial rewrite across days or components.
- Read trajectory_participant_requirement. When required_now is true, this turn MUST include at least one entity change for a specific traveler and show how that person's need conflicts with the shared itinerary. Keep the group together where feasible, but if a hard time window, fixed place, mobility need, or incompatible preference makes that impossible, explicitly choose between splitting temporarily and changing the shared flight, budget, lodging, meal, or activity plan.
- Strong natural trade-off patterns include: lodging quality or room type versus total budget; arrival time versus first-day sights and meals; return time versus last-day sightseeing; attraction count versus pace and travel time; restaurant rating versus price or cuisine variety; a specific activity versus opening hours or distance; different travelers' preferences versus a shared itinerary; changing the destination versus budget, direct-flight availability, attraction preferences, and travel time; changing the departure city or airport versus airfare, flight time, and ground transfer; moving the trip dates earlier or later versus airfare, lodging availability, opening days, and traveler schedules; shortening or extending the trip versus hotel nights, return travel, budget, and pace; and adding or removing cities versus intercity transportation, sightseeing time, budget, and itinerary intensity.
- Build difficult but realistic chains of consequences. Examples include: the party wants attractions A, B, C, D, and E but there is not enough time, so the user either accepts a quick-service meal to preserve one more attraction or protects the favorite attractions and drops the rest; better lodging consumes money needed for meals or a better-timed flight; a cheaper flight removes a meal or sightseeing window; or a better flight requires a higher budget and a cheaper hotel.
- For multiple travelers, start from a preference to stay together. A later itinerary may reveal that one traveler must be at a specific place during a fixed time window. If that overlaps an attraction, meal, intercity transfer, or flight, decide naturally whether that person temporarily separates or the whole party changes flights. If the replacement flight costs more, also change the budget or concede lodging, dining, or activities. Encode the person's fixed commitment as an entity constraint and all shared consequences as shared constraints.
- Prefer trade-offs that force a substantial rewrite of multiple days or plan components. Do not label a cosmetic wording preference or tiny local adjustment as a trade-off.
- If trajectory_tradeoff_requirement.already_satisfied is true, later turns may be a natural single or compound change, but they must still check for conflicts before merely accumulating another requirement.
- A later change MUST still use tradeoff.is_tradeoff=true when it modifies a strongly coupled pair that is already active: accommodation quality/room type versus budget; flights versus dates, activities, or meals; destination versus budget, flights, lodging, activities, meals, or travel time; origin versus airfare, flight time, dates, or ground transfer; trip dates/duration versus hotel nights, flights, opening days, traveler schedules, or budget; city count/required cities versus intercity transportation, sightseeing time, pace, or budget; attraction density versus flight/local-travel time; restaurant rating versus budget or cuisine variety; or one traveler's separate need versus the shared itinerary. Include every resulting concession and reprioritization in changes.
- Never claim a higher-rated or different room fits the current budget without checking the full submitted-plan cost plus the replacement lodging price. If the evidence does not prove it fits, either relax the budget, relax another cost-driving requirement, or do not add that lodging requirement.
- Do not produce a fourth consecutive turn containing only Add changes.
- Do not repeatedly toggle between the same two values.
- Preserve a coherent trajectory above every diversity objective: the next change must follow naturally from the current intention, earlier changes, and concrete environment evidence.
- Treat distribution_guidance only as a weak tie-breaker between changes that are already equally plausible. Ignore it when its suggested direction would invent a motive, contradict the trajectory, repeat or toggle a prior change, or fit the evidence less well.
- Choose the number of changes freely from the situation. A turn may contain one change or any natural combination of changes; there is no target ratio, quota, minimum compound count, or maximum change count.
- Include multiple changes when one user decision naturally affects several shared or person-specific requirements. Do not split a naturally compound decision merely to keep the turn simple, and do not bundle unrelated changes merely to make it compound.
- Never mention dataset distributions, counters, balancing, or this guidance in the user-facing rationale or utterance.

Required JSON schema:
{
  "intention_changed": true,
  "condition": "user_preference | real_world_feasibility",
  "tradeoff": {
    "is_tradeoff": true,
    "conflicting_constraints": ["at least two user-facing constraints"],
    "chosen_priority": "the requirement the user chooses to protect",
    "concession": "the requirement or plan component the user accepts changing",
    "plan_impacts": ["at least two of budget, origin, destination, city_scope, outbound_transportation, return_transportation, intercity_transportation, accommodation, activities, meals, local_transportation, trip_dates, trip_duration, participant_schedule"],
    "evidence": "a concrete fact from the latest submitted_plan or search_results"
  },
  "changes": [
    {
      "category": "add | relax | override | reprioritize | entity",
      "op": "add | relax | override | reprioritize",
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
  "tradeoff": {
    "is_tradeoff": false,
    "conflicting_constraints": [],
    "chosen_priority": "",
    "concession": "",
    "plan_impacts": [],
    "evidence": ""
  },
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

Trade-off shape example (adapt values to the actual itinerary; do not copy facts):
{
  "intention_changed": true,
  "condition": "real_world_feasibility",
  "tradeoff": {
    "is_tradeoff": true,
    "conflicting_constraints": ["accommodation quality", "total budget"],
    "chosen_priority": "accommodation quality",
    "concession": "raise the total budget and trim a lower-priority activity",
    "plan_impacts": ["budget", "accommodation", "activities"],
    "evidence": "The submitted plan uses a lower-rated room while the preferred lodging costs more."
  },
  "changes": [
    {"category": "add", "op": "add", "field": "accommodation_rating", "old_value": null, "value": 4.0, "priority_update": null, "rationale": "Choose better-reviewed lodging."},
    {"category": "relax", "op": "relax", "field": "budget", "old_value": 900, "value": 1150, "priority_update": null, "rationale": "Allow enough room for better lodging."},
    {"category": "reprioritize", "op": "reprioritize", "field": "accommodation_rating", "old_value": null, "value": null, "priority_update": ["accommodation_rating", "budget", "activity"], "rationale": "Lodging quality now matters more than minimizing cost."}
  ],
  "rationale": "The user chooses better lodging over the original cost and sightseeing density.",
  "utterance_plan": {"style": "explicit", "directness": "direct", "mention_old_value": true}
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
            return self._parse_shared_shift_output(
                llm_output or {}, current_intention, env_feedback
            )

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

        # In a solo trip, first-person wording describes the itinerary, not a
        # participant-specific sub-plan. Convert accidental entity_1 fields to
        # ordinary shared constraints. A new entity_N with op=add still means a
        # real traveler is joining and remains an entity operation.
        if field and len(entities) == 1 and entity_id in entities:
            if op == "scope_correction":
                return ShiftOp(
                    op="none",
                    intention_changed=False,
                    condition="none",
                    change_category="none",
                    rationale="solo_trip_scope_correction",
                )
            shared_payload = copy.deepcopy(payload)
            shared_payload["category"] = op
            shared_payload["field"] = field
            shared_payload.pop("entity_id", None)
            shared_payload.pop("replacement_entity_id", None)
            shared_payload.pop("reference", None)
            return self._parse_shared_shift_output(
                shared_payload,
                current_intention,
                env_feedback,
            )

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
            if op in {"override", "scope_correction"} and old_value == value:
                return ShiftOp(
                    op="none",
                    intention_changed=False,
                    condition="none",
                    change_category="none",
                    rationale="no_op_entity_change",
                )
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
            child_payload["_compound_child"] = True
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

        duration_changed = any(change.field == "days" for change in parsed_changes)
        fixed_dates = bool(
            working_intention.get("constraints", {}).get("start_date")
            and working_intention.get("constraints", {}).get("end_date")
        )
        date_changed = any(
            change.field in {"start_date", "end_date"} for change in parsed_changes
        )
        if duration_changed and fixed_dates and not date_changed:
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale="duration_change_requires_date_update",
            )
        if any(
            change.field in {"days", "start_date", "end_date"}
            for change in parsed_changes
        ) and not self._date_constraints_are_consistent(working_intention):
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale="date_change_requires_consistent_range_and_duration",
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
            new_state, delta = super().apply_shift(current_intention, shift)
            constraints = new_state.get("constraints", {})
            for field, change in list(delta.items()):
                if (
                    field != "priority"
                    and isinstance(change, dict)
                    and change.get("op") == "relax"
                    and change.get("new") is None
                ):
                    constraints.pop(field, None)
                    new_state["priority"] = [
                        item for item in new_state.get("priority", []) if item != field
                    ]
            return ensure_entity_state(new_state), delta

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
When shift.tradeoff.is_tradeoff is true, naturally express the conflict, which preference now wins, and the concrete concession. Make the downstream replanning consequence understandable without reciting metadata.
Sound like a traveler reacting to the itinerary, not an evaluator debugging it. Never mention validators, fields, scopes, participant assignments, exact wording, search-result provenance, or other system mechanics.
For reprioritize changes, state only the natural comparison that motivated the change (for example, lodging matters more than activities). Never recite the complete priority_update list, internal field names, or phrases such as "old priority order".
For entity changes, use entity_reference to identify the traveler naturally and preserve ownership of that person's constraint.
Never expose opaque IDs such as entity_1 in the utterance.
Shared-party constraints must remain shared; person-specific constraints must remain assigned only to that person.
Follow requested_style: explicit is direct, partial is natural but incomplete, and elliptical is short/fragment-like.
Return plain text only, with no quotes and no JSON.
""".strip()
            return f"{instructions}\n\n{REALIZATION_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

        if shift.change_category != ENTITY_CHANGE_CATEGORY:
            prompt = super()._build_realization_prompt(
                shift,
                current_intention,
                style,
                env_feedback=env_feedback,
                intention_history=intention_history,
                current_gold_delta=current_gold_delta,
            )
            instructions, marker_and_context = prompt.split(
                REALIZATION_CONTEXT_MARKER, 1
            )
            instructions = (
                instructions.rstrip()
                + "\nSound like a traveler reacting to the itinerary, not an evaluator debugging it. "
                "Never mention validators, fields, scopes, participant assignments, exact wording, "
                "search-result provenance, or other system mechanics.\n"
                + "\nFor reprioritize changes, express only which user-facing preference matters more. "
                "Never recite priority_update, internal field names, or the full ordering.\n"
            )
            return f"{instructions}\n{REALIZATION_CONTEXT_MARKER}{marker_and_context}"
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
Sound like a traveler reacting to the itinerary, not an evaluator debugging it. Never mention validators, fields, scopes, participant assignments, exact wording, search-result provenance, or other system mechanics.
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
