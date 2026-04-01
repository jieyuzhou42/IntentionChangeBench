from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from models import EnvFeedback, ShiftCondition, TriggerEvidence


NON_PREFERENCE_FIELDS = {"budget_max", "budget_min", "price_max", "price_min", "brand"}


def _norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null"}:
        return None
    return text


def _tokens(value: Any) -> List[str]:
    text = _norm(value)
    if not text:
        return []
    parts = re.split(r"[^a-z0-9]+", text)
    return [p for p in parts if p]


def _normalize_text_blob(env_feedback: EnvFeedback) -> str:
    observation = getattr(env_feedback, "observation", None) or {}
    raw_text = observation.get("raw_text", "") or ""
    clickables = observation.get("clickables", []) or []
    visible_items = observation.get("visible_items", []) or []
    result = env_feedback.result or {}

    parts: List[str] = [raw_text]
    parts.extend(str(item) for item in clickables)
    for item in visible_items:
        if isinstance(item, dict):
            parts.append(str(item.get("title", "")))
            parts.append(str(item.get("asin", "")))
    for key, value in result.items():
        parts.append(f"{key}: {value}")
    return " ".join(parts).lower()


def _page_type(env_feedback: EnvFeedback) -> str:
    observation = getattr(env_feedback, "observation", None) or {}
    return str(observation.get("page_type", "") or "").lower()


def _requested_fields(user_state: Dict[str, Any]) -> Dict[str, Any]:
    constraints = user_state.get("constraints", {}) or {}
    return {k: v for k, v in constraints.items() if _norm(v) is not None}


def _constraint_textually_supported(field: str, desired: Any, text_blob: str) -> bool:
    desired_norm = _norm(desired)
    if not desired_norm:
        return False
    if desired_norm in text_blob:
        return True

    pieces = _tokens(desired)
    if not pieces:
        return False

    if field in {"color", "size", "category"} and len(pieces) == 1:
        return pieces[0] in text_blob

    if len(pieces) >= 2:
        return all(piece in text_blob for piece in pieces)

    return False


def _field_mismatch(field: str, desired: Any, result: Dict[str, Any], text_blob: str) -> bool:
    desired_norm = _norm(desired)
    if not desired_norm:
        return False

    actual_norm = _norm(result.get(field))
    if actual_norm is not None:
        return actual_norm != desired_norm

    return not _constraint_textually_supported(field, desired, text_blob)


def _available_but_not_selected(fields: Dict[str, Any], result: Dict[str, Any], text_blob: str) -> List[str]:
    missing: List[str] = []
    for field, desired in fields.items():
        if field in NON_PREFERENCE_FIELDS:
            continue
        desired_norm = _norm(desired)
        if not desired_norm:
            continue
        if _constraint_textually_supported(field, desired, text_blob):
            actual_norm = _norm(result.get(field))
            if actual_norm is None or actual_norm != desired_norm:
                missing.append(field)
    return missing


def detect_trigger(env_feedback: EnvFeedback, user_state: Dict[str, Any]) -> tuple[Optional[ShiftCondition], TriggerEvidence]:
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
    constraints = _requested_fields(user_state)
    text_blob = _normalize_text_blob(env_feedback)
    page_type = _page_type(env_feedback)

    mismatches = []
    for field, desired in constraints.items():
        if field in NON_PREFERENCE_FIELDS:
            continue
        if _field_mismatch(field, desired, result, text_blob):
            mismatches.append(field)

    if page_type == "item":
        available_not_selected = _available_but_not_selected(constraints, result, text_blob)
        if available_not_selected:
            return (
                ShiftCondition(
                    type="real_world_feasibility",
                    reason="available_option_not_selected",
                    source="observation",
                    details={
                        "available_but_not_selected": available_not_selected,
                        "result": result,
                    },
                ),
                TriggerEvidence(
                    trigger_type="available_option_not_selected",
                    source="observation",
                    details={"available_but_not_selected": available_not_selected},
                ),
            )

    if mismatches:
        return (
            ShiftCondition(
                type="user_preference",
                reason="result_not_preferred",
                source="result_inspection",
                details={
                    "preference_mismatches": mismatches,
                    "page_type": page_type,
                    "result": result,
                },
            ),
            TriggerEvidence(
                trigger_type="preference_mismatch",
                source="result_inspection",
                details={"preference_mismatches": mismatches, "page_type": page_type},
            ),
        )

    return None, TriggerEvidence(trigger_type="none", source="none", details={})
