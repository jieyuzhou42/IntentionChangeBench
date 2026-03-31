from __future__ import annotations

from typing import Dict, Optional

from models import EnvFeedback, ShiftCondition, TriggerEvidence


PREFERENCE_FIELDS = {"brand", "color"}


def detect_trigger(env_feedback: EnvFeedback, user_state: Dict[str, any]) -> tuple[Optional[ShiftCondition], TriggerEvidence]:
    if not env_feedback.feasible:
        return (
            ShiftCondition(
                type="real_world_feasibility",
                reason=env_feedback.reason or "infeasible",
                source="environment",
                details={
                    "violated_constraints": env_feedback.violated_constraints,
                    "result": env_feedback.result,
                },
            ),
            TriggerEvidence(
                trigger_type="infeasible",
                source="environment",
                details={
                    "reason": env_feedback.reason,
                    "violated_constraints": env_feedback.violated_constraints,
                },
            ),
        )

    if env_feedback.status == "partial" and env_feedback.violated_constraints:
        return (
            ShiftCondition(
                type="real_world_feasibility",
                reason="partial_constraint_failure",
                source="environment",
                details={
                    "violated_constraints": env_feedback.violated_constraints,
                    "satisfied_constraints": env_feedback.satisfied_constraints,
                    "result": env_feedback.result,
                },
            ),
            TriggerEvidence(
                trigger_type="partial_mismatch",
                source="environment",
                details={
                    "violated_constraints": env_feedback.violated_constraints,
                    "satisfied_constraints": env_feedback.satisfied_constraints,
                },
            ),
        )

    result = env_feedback.result or {}
    constraints = user_state.get("constraints", {})

    pref_mismatches = []
    for field in PREFERENCE_FIELDS:
        desired = constraints.get(field)
        if desired is None:
            continue
        actual = result.get(field)
        if actual is not None and str(actual).lower() != str(desired).lower():
            pref_mismatches.append(field)

    if pref_mismatches and env_feedback.feasible:
        return (
            ShiftCondition(
                type="user_preference",
                reason="result_not_preferred",
                source="result_inspection",
                details={
                    "preference_mismatches": pref_mismatches,
                    "result": result,
                },
            ),
            TriggerEvidence(
                trigger_type="preference_mismatch",
                source="result_inspection",
                details={"preference_mismatches": pref_mismatches},
            ),
        )

    return None, TriggerEvidence(trigger_type="none", source="none", details={})