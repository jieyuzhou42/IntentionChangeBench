from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from base_tasks import get_all_base_tasks
from models import BaseTask, DialogueInstance, ShiftCondition, ShiftOp, TurnRecord


STYLES = ["explicit", "partial", "elliptical"]
CONDITION_TYPES = ["real_world_feasibility", "user_preference"]
CONDITION_REASON_SPACE = {
    "scheduling": {
        "real_world_feasibility": ["time_conflict", "location_constraint", "availability_issue"],
        "user_preference": ["convenience_preference_change", "quality_preference_change", "new_optional_preference"],
    },
    "retrieval_ranking": {
        "real_world_feasibility": ["availability_issue", "budget_issue", "location_constraint"],
        "user_preference": ["cost_preference_change", "quality_preference_change", "convenience_preference_change"],
    },
    "transaction": {
        "real_world_feasibility": ["availability_issue", "system_requirement", "time_conflict"],
        "user_preference": ["cost_preference_change", "convenience_preference_change", "entity_preference_change"],
    },
}

FIELD_POOLS = {
    "scheduling": {
        "real_world_feasibility": {
            "add": ["avoid_days"],
            "relax": ["time_preference", "location"],
            "override": ["day_preference", "time_preference", "location"],
            "reprioritize": ["priority"],
        },
        "user_preference": {
            "add": ["location"],
            "relax": ["time_preference", "day_preference"],
            "override": ["day_preference", "time_preference", "location"],
            "reprioritize": ["priority"],
        },
    },
    "retrieval_ranking": {
        "real_world_feasibility": {
            "add": ["budget_max", "distance_max", "rent_max"],
            "relax": ["budget_max", "distance_max", "price_level_max", "weight_lb_max"],
            "override": ["area", "location", "arrival_time", "mode", "venue", "destination", "cuisine"],
            "reprioritize": ["priority"],
        },
        "user_preference": {
            "add": ["pet_friendly", "remote", "international_ok", "mode"],
            "relax": ["budget_max", "area", "location", "distance_max"],
            "override": ["area", "location", "cuisine", "topic", "field", "mode", "time"],
            "reprioritize": ["priority"],
        },
    },
    "transaction": {
        "real_world_feasibility": {
            "add": ["insurance", "delivery", "shipping"],
            "relax": ["seat", "class", "delivery_mode", "shipping"],
            "override": ["date", "check_in", "pickup_date", "seat", "room_type", "delivery", "payment_method"],
            "reprioritize": ["priority"],
        },
        "user_preference": {
            "add": ["insurance", "delivery", "color"],
            "relax": ["seat", "class", "section", "delivery_mode"],
            "override": ["seat", "payment_method", "room_type", "class", "delivery", "color", "car_type", "section"],
            "reprioritize": ["priority"],
        },
    },
}

DEFAULT_OP_DISTS = {
    "real_world_feasibility": [("override", 0.4), ("relax", 0.3), ("add", 0.2), ("reprioritize", 0.1)],
    "user_preference": [("reprioritize", 0.35), ("override", 0.3), ("add", 0.25), ("relax", 0.1)],
}

STRING_ALTERNATIVES = {
    "Mon": "Tue", "Tue": "Thu", "Wed": "Fri", "Thu": "Mon", "Fri": "Sat", "Sat": "Sun", "Sun": "Fri",
    "morning": "afternoon", "afternoon": "evening", "evening": "morning", "midday": "evening",
    "Zoom": "Office", "Office": "Zoom", "Virtual": "HQ", "Library": "Cafe", "Phone": "Zoom",
    "downtown": "midtown", "Downtown": "Midtown", "midtown": "downtown", "Midtown": "Downtown",
    "Boston": "Chicago", "Chicago": "Boston", "New York": "Remote", "US": "Europe",
    "window": "aisle", "aisle": "window", "Visa": "Amex", "Mastercard": "Visa",
    "standard": "express", "email": "sms", "delivery": "pickup", "economy": "business",
    "black": "silver", "card": "cash", "sedan": "SUV", "mobile": "print",
    "in-person": "online", "Italian": "Japanese", "NLP": "ML systems", "CHI": "UIST",
    "student": "general", "king": "queen", "haircut": "color", "basic": "full", "middle": "front",
    "San Francisco": "Seattle", "ATL": "JFK", "SFO": "LAX", "CS": "Design", "HCI": "AI",
}

RELAX_MAP = {
    "downtown": "any", "midtown": "any", "Downtown": "any", "Midtown": "any",
    "morning": "any_time", "afternoon": "any_time", "evening": "any_time", "midday": "any_time",
    "window": "either", "aisle": "either", "economy": "any_class", "business": "any_class",
    "delivery": "pickup_or_delivery", "pickup": "pickup_or_delivery", "standard": "any_shipping",
    "express": "any_shipping", "email": "any_delivery", "sms": "any_delivery",
    "Office": "any_location", "Zoom": "any_location", "Virtual": "any_location",
}

DATE_FIELDS = {"date", "check_in", "check_out", "pickup_date"}
FIELD_TO_WORLD_KEYS = {
    "location": ["locations"],
    "seat": ["seat_options"],
    "payment_method": ["payment_methods"],
    "room_type": ["room_types"],
    "color": ["colors"],
    "shipping": ["shipping"],
    "delivery": ["delivery", "delivery_modes"],
    "delivery_mode": ["delivery_modes"],
    "ticket_type": ["ticket_types"],
    "class": ["classes"],
    "service": ["service_types"],
    "car_type": ["car_types"],
    "insurance": ["insurance_options"],
    "section": ["sections"],
    "contact_method": ["contact_methods"],
    "resume": ["document_options"],
}


def weighted_choice(rng: random.Random, pairs: List[Tuple[str, float]]) -> str:
    total = sum(weight for _, weight in pairs)
    target = rng.random() * total
    acc = 0.0
    for item, weight in pairs:
        acc += weight
        if target <= acc:
            return item
    return pairs[-1][0]


def infer_action_implication(op: str, field: str) -> str:
    if op == "reprioritize":
        return "rerank_only"
    if field in DATE_FIELDS | {"day_preference", "time_preference", "arrival_time"}:
        return "requery_required"
    if field in {"seat", "payment_method", "room_type", "class", "delivery", "shipping", "insurance", "section", "car_type"}:
        return "rebook_required"
    return "replan_required"


class IntentionShiftGenerator:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def sample_condition(self, task_type: str) -> ShiftCondition:
        cond_type = weighted_choice(
            self.rng,
            [("real_world_feasibility", 0.5), ("user_preference", 0.5)],
        )
        reason = self.rng.choice(CONDITION_REASON_SPACE[task_type][cond_type])
        explicitly_mentioned = self.rng.random() < (0.65 if cond_type == "real_world_feasibility" else 0.45)
        return ShiftCondition(type=cond_type, reason=reason, explicitly_mentioned=explicitly_mentioned)

    def sample_operation(self, condition: ShiftCondition) -> str:
        return weighted_choice(self.rng, DEFAULT_OP_DISTS[condition.type])

    def sample_shift_program(self, task: BaseTask, num_turns: int = 3) -> List[ShiftOp]:
        state = copy.deepcopy(task.initial_intention)
        shifts: List[ShiftOp] = []
        for turn in range(2, num_turns + 2):
            condition = self.sample_condition(task.task_type)
            op = self.sample_operation(condition)
            field, value = self.sample_field_and_value(task, state, op, condition)
            style = self.rng.choice(STYLES)
            shift = ShiftOp(turn=turn, op=op, field=field, value=value, style=style, condition=condition)
            shifts.append(shift)
            state = self.apply_shift(state, shift)
        return shifts

    def sample_field_and_value(
        self,
        task: BaseTask,
        state: Dict[str, Any],
        op: str,
        condition: ShiftCondition,
    ) -> Tuple[str, Any]:
        if op == "reprioritize":
            current_priority = copy.deepcopy(state.get("priority", []))
            if len(current_priority) > 1:
                self.rng.shuffle(current_priority)
            return "priority", current_priority

        pool = FIELD_POOLS.get(task.task_type, {}).get(condition.type, {}).get(op, [])
        valid_pool = [field for field in pool if field in state or op == "add"]
        candidate_fields = valid_pool or [k for k in state.keys() if k not in {"goal", "priority"}]
        field = self.rng.choice(candidate_fields)
        current_value = state.get(field)
        new_value = self._sample_new_value(field, current_value, task.world_state, op, condition)
        return field, new_value

    def _sample_new_value(
        self,
        field: str,
        current: Any,
        world_state: Dict[str, Any],
        op: str,
        condition: ShiftCondition,
    ) -> Any:
        if field == "avoid_days":
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            existing = current or []
            remaining = [d for d in days if d not in existing]
            chosen = self.rng.choice(remaining) if remaining else self.rng.choice(days)
            return existing + [chosen]

        if field in DATE_FIELDS and isinstance(current, str):
            month, day = current.split("-")[-2:]
            new_day = max(1, min(28, int(day) + self.rng.choice([-3, -2, 2, 3])))
            prefix = "-".join(current.split("-")[:-1])
            return f"{prefix}-{new_day:02d}"

        for key in FIELD_TO_WORLD_KEYS.get(field, []):
            options = world_state.get(key)
            if isinstance(options, list) and options:
                alts = [v for v in options if v != current]
                if alts:
                    return self.rng.choice(alts)

        if isinstance(current, bool):
            return not current if op != "relax" else current

        if isinstance(current, (int, float)):
            if op == "relax":
                return round(current * 1.2, 2)
            if condition.type == "real_world_feasibility":
                return round(current * 0.9, 2) if current > 10 else current + 1
            return round(current * 1.1, 2)

        if isinstance(current, list):
            if op == "relax" and current:
                return current[:-1] or current
            new_list = current[:]
            self.rng.shuffle(new_list)
            return new_list

        if isinstance(current, str):
            if op == "relax":
                return RELAX_MAP.get(current, f"more_flexible_{current}")
            return STRING_ALTERNATIVES.get(current, f"alternative_{current}")

        if current is None:
            add_defaults = {
                "pet_friendly": True,
                "remote": True,
                "international_ok": True,
                "insurance": "full",
                "delivery": "email",
                "shipping": "express",
                "color": "silver",
                "mode": "online",
                "location": "Zoom",
            }
            return add_defaults.get(field, "new_requirement")

        return current

    def apply_shift(self, state: Dict[str, Any], shift: ShiftOp) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        if shift.op in {"add", "relax", "override"}:
            new_state[shift.field] = shift.value
        elif shift.op == "reprioritize":
            new_state["priority"] = shift.value
        return new_state

    def realize_initial_utterance(self, task: BaseTask) -> str:
        i = task.initial_intention
        goal = i.get("goal", "complete the task").replace("_", " ")
        parts = [f"I need help to {goal}."]
        for key, value in i.items():
            if key in {"goal", "priority"}:
                continue
            parts.append(f"{key.replace('_', ' ')}: {value}.")
        return " ".join(parts)

    def realize_shift_utterance(self, shift: ShiftOp) -> str:
        field_text = shift.field.replace("_", " ")
        reason_text = self._reason_phrase(shift.condition)
        value_text = ", ".join(map(str, shift.value)) if isinstance(shift.value, list) else str(shift.value)

        if shift.op == "add":
            explicit_core = f"Please also add this requirement: {field_text} should be {value_text}."
            partial_core = f"Also, make {field_text} {value_text}."
            elliptical_core = f"And {value_text} too."
        elif shift.op == "relax":
            explicit_core = f"We can relax the {field_text} constraint. It can be {value_text}."
            partial_core = f"Actually, {field_text} can be more flexible — {value_text} is fine."
            elliptical_core = f"That part can be more flexible."
        elif shift.op == "override":
            explicit_core = f"Please change the {field_text} to {value_text}."
            partial_core = f"Actually, {value_text} instead for {field_text}."
            elliptical_core = f"{value_text} instead."
        else:
            explicit_core = f"Please update the priority order to: {value_text}."
            first_priority = shift.value[0] if isinstance(shift.value, list) and shift.value else "that"
            partial_core = f"I care more about {first_priority} now."
            elliptical_core = "Priorities changed — optimize differently."

        style_to_core = {
            "explicit": explicit_core,
            "partial": partial_core,
            "elliptical": elliptical_core,
        }
        core = style_to_core[shift.style]
        if shift.condition.explicitly_mentioned:
            return f"{reason_text} {core}"
        return core

    def _reason_phrase(self, condition: ShiftCondition) -> str:
        reason_templates = {
            "time_conflict": "I just realized there is a time conflict.",
            "location_constraint": "That setup is less feasible from a logistics standpoint.",
            "availability_issue": "It looks like the original option may not be available.",
            "budget_issue": "The original plan is pushing the budget too much.",
            "system_requirement": "The system seems to require a different setup.",
            "cost_preference_change": "I am leaning more toward saving money now.",
            "quality_preference_change": "I care more about quality now.",
            "convenience_preference_change": "Convenience matters more to me now.",
            "entity_preference_change": "I would rather go with a different option now.",
            "new_optional_preference": "I have one more preference to add.",
        }
        return reason_templates.get(condition.reason, "I want to adjust the plan.")

    def build_dialogue_instance(self, task: BaseTask, num_shift_turns: int = 3) -> DialogueInstance:
        shifts = self.sample_shift_program(task, num_turns=num_shift_turns)
        state = copy.deepcopy(task.initial_intention)
        turns: List[TurnRecord] = [
            TurnRecord(
                turn_id=1,
                user_utterance=self.realize_initial_utterance(task),
                shift_condition={"type": "initial_request", "reason": "initial_goal_specification", "explicitly_mentioned": True},
                gold_delta={"add": {k: v for k, v in task.initial_intention.items() if k != "priority"}},
                gold_current_intention=copy.deepcopy(state),
                linguistic_style="explicit",
                action_implication="initial_query",
            )
        ]
        for shift in shifts:
            state = self.apply_shift(state, shift)
            turns.append(
                TurnRecord(
                    turn_id=shift.turn,
                    user_utterance=self.realize_shift_utterance(shift),
                    shift_condition=shift.condition.to_dict(),
                    gold_delta={shift.op: {shift.field: shift.value}},
                    gold_current_intention=copy.deepcopy(state),
                    linguistic_style=shift.style,
                    action_implication=infer_action_implication(shift.op, shift.field),
                )
            )
        return DialogueInstance(
            instance_id=task.instance_id,
            task_type=task.task_type,
            subtype=task.subtype,
            world_state=task.world_state,
            turns=turns,
        )

    def generate_dataset(self, num_shift_turns: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        base_tasks = get_all_base_tasks()
        return {
            task_type: [self.build_dialogue_instance(task, num_shift_turns).to_dict() for task in tasks]
            for task_type, tasks in base_tasks.items()
        }

    def export_dataset(self, output_path: str | Path, num_shift_turns: int = 3) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = self.generate_dataset(num_shift_turns=num_shift_turns)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
