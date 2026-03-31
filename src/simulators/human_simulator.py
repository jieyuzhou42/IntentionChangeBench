from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from models import ShiftCondition, ShiftOp


class HumanSimulator:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def decide_shift(
        self,
        trigger: ShiftCondition,
        user_state: Dict[str, Any],
    ) -> ShiftOp:
        constraints = user_state.get("constraints", {})
        priority = user_state.get("priority", list(constraints.keys()))
        details = trigger.details or {}
        violated = details.get("violated_constraints", [])

        if trigger.type == "real_world_feasibility":
            if trigger.reason == "no_matching_results":
                field = self._pick_target_field(violated, priority, constraints)
                if field is None:
                    field = self._pick_relaxable_field(priority, constraints)
                if field is None:
                    return ShiftOp(op="none", rationale="no relaxable field")
                return self._make_relax(field, constraints)

            if trigger.reason in {"partial_constraint_failure", "constraint_mismatch"}:
                field = self._pick_target_field(violated, priority, constraints)
                if field is None:
                    field = self._pick_relaxable_field(priority, constraints)
                if field is None:
                    return ShiftOp(op="none", rationale="no candidate field")
                return self._make_relax(field, constraints)

            return ShiftOp(op="none", rationale="feasibility issue but no shift selected")

        if trigger.type == "user_preference":
            pref_mismatches = details.get("preference_mismatches", [])
            field = self._pick_target_field(pref_mismatches, priority, constraints)
            if field is None:
                return ShiftOp(op="none", rationale="no preference field selected")
            actual_value = details.get("result", {}).get(field)
            return ShiftOp(
                op="override",
                field=field,
                old_value=constraints.get(field),
                value=actual_value,
                rationale="user changes preference after inspecting result",
            )

        return ShiftOp(op="none", rationale="no trigger")

    def apply_shift(self, user_state: Dict[str, Any], shift: ShiftOp) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        new_state = copy.deepcopy(user_state)
        delta: Dict[str, Dict[str, Any]] = {}

        if shift.op == "none" or not shift.field:
            return new_state, delta

        old_value = new_state["constraints"].get(shift.field)
        new_state["constraints"][shift.field] = shift.value
        delta[shift.field] = {
            "op": shift.op,
            "old": old_value,
            "new": shift.value,
            "rationale": shift.rationale,
        }
        return new_state, delta

    def realize_shift(self, shift: ShiftOp, user_state: Dict[str, Any], style: str) -> str:
        if shift.op == "none":
            return "That looks fine, please continue."

        field = shift.field
        value = shift.value
        old_value = shift.old_value

        if style == "explicit":
            if shift.op == "relax":
                return f"That constraint may be too strict. You can relax the {field} requirement to {value}."
            if shift.op == "override":
                return f"Actually, let's change the {field} from {old_value} to {value}."

        if style == "partial":
            if shift.op == "relax":
                return f"Okay, the {field} can be a bit more flexible."
            if shift.op == "override":
                return f"Let's go with {value} instead."

        if style == "elliptical":
            if shift.op == "relax":
                return "That part can be more flexible."
            if shift.op == "override":
                return "Actually, switch it."

        return f"Please update the {field} requirement."

    def _pick_target_field(
        self,
        violated: List[str],
        priority: List[str],
        constraints: Dict[str, Any],
    ) -> Optional[str]:
        valid = []
        for f in violated:
            # cannot relax a field that is already None / unconstrained
            if f in constraints and constraints.get(f) is not None:
                valid.append(f)

        ordered = [f for f in reversed(priority) if f in valid]
        return ordered[0] if ordered else (valid[0] if valid else None)

    def _pick_relaxable_field(self, priority: List[str], constraints: Dict[str, Any]) -> Optional[str]:
        relax_order = [
            f for f in reversed(priority)
            if f in constraints and constraints.get(f) is not None
        ]
        if relax_order:
            return relax_order[0]

        for field, value in constraints.items():
            if value is not None:
                return field
        return None

    def _make_relax(self, field: str, constraints: Dict[str, Any]) -> ShiftOp:
        old = constraints.get(field)

        if field == "budget_max" and isinstance(old, (int, float)):
            return ShiftOp(
                op="relax",
                field=field,
                old_value=old,
                value=round(old * 1.25, 2),
                rationale="increase budget to admit feasible candidates",
            )

        if field == "color":
            return ShiftOp(
                op="relax",
                field=field,
                old_value=old,
                value=None,
                rationale="drop strict color requirement",
            )

        if field == "brand":
            return ShiftOp(
                op="relax",
                field=field,
                old_value=old,
                value=None,
                rationale="drop strict brand requirement",
            )

        return ShiftOp(
            op="relax",
            field=field,
            old_value=old,
            value=None,
            rationale="relax lower-priority constraint",
        )
