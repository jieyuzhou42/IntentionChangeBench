from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol, Tuple

from models import ShiftCondition, ShiftOp


ALLOWED_SHIFT_OPS = {
    "none",
    "relax",
    "override",
    "reprioritize",
    "scope_correction",
}

OBSERVATION_REPAIR_REASONS = {"available_option_not_selected"}
ALLOWED_STYLES = {"explicit", "partial", "elliptical"}
ALLOWED_DIRECTNESS = {"direct", "indirect"}
SHIFT_CONTEXT_MARKER = "SHIFT_CONTEXT_JSON:"
REALIZATION_CONTEXT_MARKER = "REALIZATION_CONTEXT_JSON:"


class LLMClientProtocol(Protocol):
    """Minimal interface for an injected LLM client."""

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        ...

    def generate_text(self, prompt: str) -> str:
        ...


def _safe_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _parse_json_like(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_prompt_payload(prompt: str, marker: str) -> Dict[str, Any]:
    marker_index = prompt.rfind(marker)
    if marker_index < 0:
        return {}

    raw_payload = prompt[marker_index + len(marker):].strip()
    parsed = _parse_json_like(raw_payload)
    return parsed or {}


def _clean_string(value: Any) -> str:
    return str(value).strip()


def _normalize_none_like(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "no preference"
    if isinstance(value, float):
        rendered = f"{value:.2f}".rstrip("0").rstrip(".")
        return rendered
    return str(value)


class MockLLMClient:
    """
    Deterministic local client for tests and offline runs.

    It reads the machine-readable prompt payload and returns structured
    decisions / utterances without external dependencies.
    """

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        context = _extract_prompt_payload(prompt, SHIFT_CONTEXT_MARKER)
        return self._decide_shift_from_context(context)

    def generate_text(self, prompt: str) -> str:
        context = _extract_prompt_payload(prompt, REALIZATION_CONTEXT_MARKER)
        return self._realize_from_context(context)

    def _decide_shift_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trigger = context.get("trigger", {})
        trigger_type = trigger.get("type")
        details = trigger.get("details", {}) or {}
        user_state = context.get("user_state", {}) or {}
        constraints = user_state.get("constraints", {}) or {}
        priority = user_state.get("priority", list(constraints.keys())) or list(constraints.keys())
        result = context.get("result", {}) or details.get("result", {}) or {}
        violated = context.get("violated_constraints", []) or details.get("violated_constraints", []) or []
        pref_mismatches = (
            context.get("preference_mismatches", [])
            or details.get("preference_mismatches", [])
            or []
        )

        if trigger_type == "real_world_feasibility":
            reason = trigger.get("reason")
            available_not_selected = details.get("available_but_not_selected", []) or []
            if reason in OBSERVATION_REPAIR_REASONS and available_not_selected:
                field = available_not_selected[0]
                desired_value = constraints.get(field)
                return {
                    "op": "scope_correction",
                    "field": field,
                    "old_value": constraints.get(field),
                    "value": desired_value,
                    "priority_update": priority,
                    "rationale": "clarify the original requested option rather than changing preferences",
                    "utterance_plan": {
                        "style": "explicit",
                        "directness": "direct",
                        "mention_old_value": False,
                    },
                }

            field = self._pick_feasibility_field(violated, priority, constraints)
            if field is None:
                return {
                    "op": "none",
                    "field": None,
                    "old_value": None,
                    "value": None,
                    "priority_update": None,
                    "rationale": "no relaxable field",
                    "utterance_plan": {
                        "style": "partial",
                        "directness": "indirect",
                        "mention_old_value": False,
                    },
                }

            old_value = constraints.get(field)
            value = self._default_relax_value(field, old_value)
            return {
                "op": "relax",
                "field": field,
                "old_value": old_value,
                "value": value,
                "priority_update": priority,
                "rationale": "relax a lower-priority constraint to restore feasibility",
                "utterance_plan": {
                    "style": "partial",
                    "directness": "indirect",
                    "mention_old_value": False,
                },
            }

        if trigger_type == "user_preference":
            field = self._pick_low_priority_field(pref_mismatches, priority, constraints)
            if field is None:
                return {
                    "op": "none",
                    "field": None,
                    "old_value": None,
                    "value": None,
                    "priority_update": None,
                    "rationale": "no preference field selected",
                    "utterance_plan": {
                        "style": "partial",
                        "directness": "direct",
                        "mention_old_value": False,
                    },
                }

            actual_value = result.get(field)
            if actual_value is None:
                return {
                    "op": "none",
                    "field": None,
                    "old_value": None,
                    "value": None,
                    "priority_update": None,
                    "rationale": "result missing preference value",
                    "utterance_plan": {
                        "style": "partial",
                        "directness": "direct",
                        "mention_old_value": False,
                    },
                }

            return {
                "op": "override",
                "field": field,
                "old_value": constraints.get(field),
                "value": actual_value,
                "priority_update": priority,
                "rationale": "user changes preference after inspecting the result",
                "utterance_plan": {
                    "style": "partial",
                    "directness": "direct",
                    "mention_old_value": False,
                },
            }

        return {
            "op": "none",
            "field": None,
            "old_value": None,
            "value": None,
            "priority_update": None,
            "rationale": "no trigger",
            "utterance_plan": {
                "style": "partial",
                "directness": "direct",
                "mention_old_value": False,
            },
        }

    def _realize_from_context(self, context: Dict[str, Any]) -> str:
        shift = context.get("shift", {}) or {}
        plan = shift.get("utterance_plan", {}) or {}
        style = context.get("requested_style") or plan.get("style") or "explicit"
        style = style if style in ALLOWED_STYLES else "explicit"
        directness = plan.get("directness", "direct")
        mention_old_value = bool(plan.get("mention_old_value"))
        op = shift.get("op", "none")
        field = shift.get("field")
        value = shift.get("value")
        old_value = shift.get("old_value")
        priority_update = shift.get("priority_update") or []
        field_text = str(field).replace("_", " ") if field else "that"
        value_text = _format_value(value)
        old_text = _format_value(old_value)

        if op == "none":
            return "Let's keep the current constraints for now."

        if op == "relax":
            if style == "explicit":
                if mention_old_value and old_value is not None:
                    return (
                        f"We can ease the {field_text} constraint a bit, from "
                        f"{old_text} to {value_text}."
                    )
                return f"We can be a bit more flexible on {field_text}; set it to {value_text}."
            if style == "partial":
                if directness == "indirect":
                    return f"I'm okay being a little looser on {field_text}."
                return f"Let's relax {field_text} a bit."
            return f"{field_text} can be more flexible."

        if op == "override":
            if style == "explicit":
                if mention_old_value and old_value is not None:
                    return f"Actually, change the {field_text} from {old_text} to {value_text}."
                return f"Actually, let's make the {field_text} {value_text}."
            if style == "partial":
                return f"Let's go with {value_text} instead."
            return f"{value_text} instead."

        if op == "reprioritize":
            top_field = priority_update[0] if priority_update else field
            top_text = str(top_field).replace("_", " ") if top_field else "that"
            if style == "explicit":
                return f"Let's keep the same constraints, but prioritize {top_text} first."
            if style == "partial":
                return f"Let's focus more on {top_text}."
            return f"{top_text} first."

        if op == "scope_correction":
            if style == "explicit":
                return f"I still want {field_text} {value_text}."
            if style == "partial":
                return f"Still need {field_text} {value_text}."
            return f"{field_text} {value_text}."

        return "Please update that requirement."

    def _pick_low_priority_field(
        self,
        candidate_fields: List[str],
        priority: List[str],
        constraints: Dict[str, Any],
    ) -> Optional[str]:
        valid = []
        for field in candidate_fields:
            if field in constraints and constraints.get(field) is not None:
                valid.append(field)

        ordered = [field for field in reversed(priority) if field in valid]
        return ordered[0] if ordered else (valid[0] if valid else None)

    def _pick_feasibility_field(
        self,
        violated: List[str],
        priority: List[str],
        constraints: Dict[str, Any],
    ) -> Optional[str]:
        top_priority = priority[0] if priority else None
        non_top_violated = [
            field
            for field in reversed(priority)
            if field in violated and constraints.get(field) is not None and field != top_priority
        ]
        if non_top_violated:
            return non_top_violated[0]

        relaxable_non_top = [
            field
            for field in reversed(priority)
            if field in constraints and constraints.get(field) is not None and field != top_priority
        ]
        if relaxable_non_top:
            return relaxable_non_top[0]

        violated_any = [
            field
            for field in reversed(priority)
            if field in violated and constraints.get(field) is not None
        ]
        if violated_any:
            return violated_any[0]

        relaxable_any = [
            field
            for field in reversed(priority)
            if field in constraints and constraints.get(field) is not None
        ]
        return relaxable_any[0] if relaxable_any else None

    def _default_relax_value(self, field: str, old_value: Any) -> Any:
        if old_value is None:
            return None
        if field == "budget_max" and isinstance(old_value, (int, float)):
            return round(float(old_value) * 1.25, 2)
        if field.endswith("_max") and isinstance(old_value, (int, float)):
            return round(float(old_value) * 1.25, 2)
        if field.endswith("_min") and isinstance(old_value, (int, float)):
            return round(float(old_value) * 0.8, 2)
        if field in {"brand", "color"}:
            return None
        return None


class LLMHumanSimulator:
    """
    LLM-backed human simulator with deterministic state application.

    `decide_shift` delegates structured reasoning to an injected client that
    returns JSON. `apply_shift` remains deterministic so the benchmark state
    transitions stay stable and easy to inspect.
    """

    def __init__(self, llm_client: Optional[LLMClientProtocol] = None, seed: int = 7):
        self.rng = random.Random(seed)
        self.llm_client = llm_client or MockLLMClient(seed=seed)

    def _build_shift_prompt(
        self,
        trigger: ShiftCondition,
        user_state: Dict[str, Any],
    ) -> str:
        constraints = self._constraints_from_state(user_state)
        priority = self._priority_from_state(user_state, constraints)
        details = trigger.details or {}

        context = {
            "trigger": {
                "type": trigger.type,
                "reason": trigger.reason,
                "source": trigger.source,
                "details": details,
            },
            "user_state": {
                "request": user_state.get("request"),
                "constraints": constraints,
                "priority": priority,
            },
            "violated_constraints": details.get("violated_constraints", []),
            "satisfied_constraints": details.get("satisfied_constraints", []),
            "preference_mismatches": details.get("preference_mismatches", []),
            "result": details.get("result", {}),
        }

        instructions = """
You are simulating a human user who may revise their intent after receiving environment feedback.
Return a single JSON object only.

Allowed ops:
- none
- relax
- override
- reprioritize
- scope_correction

Rules:
- Prefer repairing the agent or clarifying the original request before changing preferences.
- If the desired option is visibly available on the page but not selected, use scope_correction on the original field/value.
- Only relax lower-priority constraints for real_world_feasibility when the environment suggests the original request may be infeasible.
- Keep high-priority constraints stable unless the evidence strongly supports changing them.
- For user_preference, allow natural override behavior when the inspected result genuinely changes the user's mind.
- Keep the decision grounded in trigger evidence, result fields, observation details, and current priority ordering.
- If no change is appropriate, return op="none".

Required JSON schema:
{
  "op": "none | relax | override | reprioritize | scope_correction",
  "field": "constraint field name or null",
  "old_value": "previous value or null",
  "value": "new value or null",
  "priority_update": ["ordered", "priority", "fields"] or null,
  "rationale": "short explanation",
  "utterance_plan": {
    "style": "explicit | partial | elliptical",
    "directness": "direct | indirect",
    "mention_old_value": true
  }
}
""".strip()

        return f"{instructions}\n\n{SHIFT_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def _call_llm_for_shift(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            raw_output = self.llm_client.generate_json(prompt)
        except Exception:
            return None
        return _parse_json_like(raw_output)

    def _parse_shift_output(
        self,
        llm_output: Optional[Dict[str, Any]],
        trigger: ShiftCondition,
        user_state: Dict[str, Any],
    ) -> ShiftOp:
        if not llm_output:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        constraints = self._constraints_from_state(user_state)
        priority = self._priority_from_state(user_state, constraints)
        result = (trigger.details or {}).get("result", {}) or {}

        op = _clean_string(llm_output.get("op", "none")).lower()
        if op not in ALLOWED_SHIFT_OPS:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        field = self._match_field_name(llm_output.get("field"), constraints, priority)
        rationale = _clean_string(llm_output.get("rationale", "")) or "llm_decision"
        priority_update = self._normalize_priority_update(
            llm_output.get("priority_update"),
            constraints,
            priority,
        )
        utterance_plan = self._normalize_utterance_plan(llm_output.get("utterance_plan"))

        if op == "none":
            return ShiftOp(
                op="none",
                rationale=rationale,
                priority_update=priority_update,
                utterance_plan=utterance_plan,
            )

        if op == "reprioritize":
            if priority_update is None:
                if field is None:
                    return ShiftOp(op="none", rationale="invalid_llm_output")
                priority_update = self._move_field_to_front(field, priority)
            return ShiftOp(
                op="reprioritize",
                field=field,
                old_value=priority,
                value=priority_update,
                rationale=rationale,
                priority_update=priority_update,
                utterance_plan=utterance_plan,
            )

        if field is None or field not in constraints:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        old_value = llm_output.get("old_value", constraints.get(field))
        old_value = self._coerce_value(field, old_value, constraints.get(field))
        if old_value is None:
            old_value = constraints.get(field)

        raw_value = llm_output.get("value")
        value = self._coerce_value(field, raw_value, old_value)
        if value is None and op == "relax":
            value = self._default_relax_value(field, old_value)
        if value is None and op in {"override", "scope_correction"}:
            value = result.get(field)
            value = self._coerce_value(field, value, old_value)

        if op in {"override", "scope_correction"} and value is None:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        if op == "relax" and old_value is not None and not self._looks_like_relaxation(field, old_value, value):
            value = self._default_relax_value(field, old_value)

        return ShiftOp(
            op=op,
            field=field,
            old_value=old_value,
            value=value,
            rationale=rationale,
            priority_update=priority_update,
            utterance_plan=utterance_plan,
        )

    def decide_shift(
        self,
        trigger: ShiftCondition,
        user_state: Dict[str, Any],
    ) -> ShiftOp:
        prompt = self._build_shift_prompt(trigger, user_state)
        llm_output = self._call_llm_for_shift(prompt)
        return self._parse_shift_output(llm_output, trigger, user_state)

    def apply_shift(
        self,
        user_state: Dict[str, Any],
        shift: ShiftOp,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        new_state = copy.deepcopy(user_state)
        new_state.setdefault("constraints", {})
        new_state["priority"] = self._priority_from_state(new_state, new_state["constraints"])
        delta: Dict[str, Dict[str, Any]] = {}

        if shift.op == "none":
            return new_state, delta

        if shift.op in {"relax", "override", "scope_correction"} and shift.field:
            old_value = new_state["constraints"].get(shift.field)
            new_state["constraints"][shift.field] = shift.value
            delta[shift.field] = {
                "op": shift.op,
                "old": old_value,
                "new": shift.value,
                "rationale": shift.rationale,
            }

        if shift.priority_update:
            normalized_priority = self._normalize_priority_update(
                shift.priority_update,
                new_state["constraints"],
                new_state.get("priority", []),
            )
            if normalized_priority and normalized_priority != new_state.get("priority", []):
                old_priority = list(new_state.get("priority", []))
                new_state["priority"] = normalized_priority
                delta["priority"] = {
                    "op": "reprioritize",
                    "old": old_priority,
                    "new": normalized_priority,
                    "rationale": shift.rationale,
                }

        return new_state, delta

    def _build_realization_prompt(
        self,
        shift: ShiftOp,
        user_state: Dict[str, Any],
        style: str,
    ) -> str:
        requested_style = style if style in ALLOWED_STYLES else "explicit"
        context = {
            "requested_style": requested_style,
            "user_state": {
                "request": user_state.get("request"),
                "constraints": self._constraints_from_state(user_state),
                "priority": self._priority_from_state(
                    user_state,
                    self._constraints_from_state(user_state),
                ),
            },
            "shift": asdict(shift),
        }

        instructions = """
Write the user's next utterance as a single short sentence.
Ground the utterance strictly in the structured shift decision.
Do not invent new constraints or changes that are not present in the shift object.

Style guide:
- explicit: directly state the change
- partial: hint at the change naturally
- elliptical: short and fragment-like

Return plain text only, with no quotes and no JSON.
""".strip()

        return f"{instructions}\n\n{REALIZATION_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def _call_llm_for_realization(self, prompt: str) -> Optional[str]:
        try:
            raw_output = self.llm_client.generate_text(prompt)
        except Exception:
            return None

        if not isinstance(raw_output, str):
            return None

        cleaned = raw_output.strip().strip('"').strip()
        return cleaned or None

    def realize_shift(self, shift: ShiftOp, user_state: Dict[str, Any], style: str) -> str:
        prompt = self._build_realization_prompt(shift, user_state, style)
        utterance = self._call_llm_for_realization(prompt)
        if utterance:
            return utterance
        return self._fallback_realization(shift, style)

    def _fallback_realization(self, shift: ShiftOp, style: str) -> str:
        effective_style = style if style in ALLOWED_STYLES else "explicit"
        plan = shift.utterance_plan or {}
        mention_old_value = bool(plan.get("mention_old_value"))
        field_text = shift.field.replace("_", " ") if shift.field else "that"
        value_text = _format_value(shift.value)
        old_text = _format_value(shift.old_value)

        if shift.op == "none":
            return "Let's keep the current constraints for now."

        if shift.op == "relax":
            if effective_style == "explicit":
                if mention_old_value and shift.old_value is not None:
                    return f"We can relax {field_text} from {old_text} to {value_text}."
                return f"We can relax {field_text} a bit."
            if effective_style == "partial":
                return f"{field_text} can be a little more flexible."
            return "That part can be more flexible."

        if shift.op == "override":
            if effective_style == "explicit":
                if mention_old_value and shift.old_value is not None:
                    return f"Actually, change {field_text} from {old_text} to {value_text}."
                return f"Actually, let's make it {value_text}."
            if effective_style == "partial":
                return f"Let's go with {value_text} instead."
            return f"{value_text} instead."

        if shift.op == "reprioritize":
            target = shift.priority_update[0] if shift.priority_update else shift.field
            target_text = str(target).replace("_", " ") if target else "that"
            if effective_style == "explicit":
                return f"Let's prioritize {target_text} first."
            if effective_style == "partial":
                return f"Focus more on {target_text}."
            return f"{target_text} first."

        if shift.op == "scope_correction":
            if effective_style == "explicit":
                return f"I still want {field_text} {value_text}."
            if effective_style == "partial":
                return f"Still need {field_text} {value_text}."
            return f"{field_text} {value_text}."

        return "Please update that requirement."

    def _constraints_from_state(self, user_state: Dict[str, Any]) -> Dict[str, Any]:
        constraints = user_state.get("constraints", {}) or {}
        return constraints if isinstance(constraints, dict) else {}

    def _priority_from_state(
        self,
        user_state: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> List[str]:
        raw_priority = user_state.get("priority", [])
        priority = raw_priority if isinstance(raw_priority, list) else []
        ordered: List[str] = []
        seen = set()

        for field in priority:
            matched = self._match_field_name(field, constraints, priority)
            if matched and matched not in seen:
                ordered.append(matched)
                seen.add(matched)

        for field in constraints.keys():
            if field not in seen:
                ordered.append(field)
                seen.add(field)

        return ordered

    def _match_field_name(
        self,
        raw_field: Any,
        constraints: Dict[str, Any],
        priority: List[str],
    ) -> Optional[str]:
        if raw_field is None:
            return None

        candidate = _clean_string(raw_field)
        if not candidate:
            return None

        normalized = candidate.replace(" ", "_").lower()
        known_fields = list(constraints.keys()) + [field for field in priority if field not in constraints]
        for field in known_fields:
            if field.lower() == normalized:
                return field
        return None

    def _normalize_priority_update(
        self,
        raw_priority: Any,
        constraints: Dict[str, Any],
        current_priority: List[str],
    ) -> Optional[List[str]]:
        if not isinstance(raw_priority, list) or not raw_priority:
            return None

        normalized: List[str] = []
        seen = set()
        known_fields = current_priority + [field for field in constraints.keys() if field not in current_priority]

        for raw_field in raw_priority:
            matched = self._match_field_name(raw_field, constraints, known_fields)
            if matched and matched not in seen:
                normalized.append(matched)
                seen.add(matched)

        for field in known_fields:
            if field not in seen:
                normalized.append(field)
                seen.add(field)

        return normalized or None

    def _normalize_utterance_plan(self, raw_plan: Any) -> Dict[str, Any]:
        if not isinstance(raw_plan, dict):
            return {
                "style": "partial",
                "directness": "direct",
                "mention_old_value": False,
            }

        style = _clean_string(raw_plan.get("style", "partial")).lower()
        directness = _clean_string(raw_plan.get("directness", "direct")).lower()

        return {
            "style": style if style in ALLOWED_STYLES else "partial",
            "directness": directness if directness in ALLOWED_DIRECTNESS else "direct",
            "mention_old_value": bool(raw_plan.get("mention_old_value")),
        }

    def _move_field_to_front(self, field: str, priority: List[str]) -> List[str]:
        new_priority = [field]
        for existing_field in priority:
            if existing_field != field:
                new_priority.append(existing_field)
        return new_priority

    def _coerce_value(self, field: str, value: Any, old_value: Any) -> Any:
        value = _normalize_none_like(value)
        if value is None:
            return None

        if isinstance(old_value, bool):
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
            return bool(value)

        if isinstance(old_value, int) and not isinstance(old_value, bool):
            try:
                parsed = float(value)
                return int(parsed) if parsed.is_integer() else parsed
            except (TypeError, ValueError):
                return value

        if isinstance(old_value, float):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        if field.endswith("_max") or field.endswith("_min"):
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        if isinstance(value, str):
            return value.strip()
        return value

    def _default_relax_value(self, field: str, old_value: Any) -> Any:
        if old_value is None:
            return None
        if field == "budget_max" and isinstance(old_value, (int, float)):
            return round(float(old_value) * 1.25, 2)
        if field.endswith("_max") and isinstance(old_value, (int, float)):
            return round(float(old_value) * 1.25, 2)
        if field.endswith("_min") and isinstance(old_value, (int, float)):
            return round(float(old_value) * 0.8, 2)
        if field in {"brand", "color"}:
            return None
        return None

    def _looks_like_relaxation(self, field: str, old_value: Any, new_value: Any) -> bool:
        if new_value is None:
            return old_value is not None

        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if field.endswith("_min"):
                return float(new_value) <= float(old_value)
            return float(new_value) >= float(old_value)

        return new_value != old_value


class HumanSimulator(LLMHumanSimulator):
    """Backward-compatible simulator name used by the existing pipeline."""


def build_example_usage() -> Dict[str, Any]:
    """
    Small end-to-end example that exercises the three-stage flow with the
    built-in mock client.
    """

    simulator = LLMHumanSimulator(seed=7)
    trigger = ShiftCondition(
        type="real_world_feasibility",
        reason="no_matching_results",
        source="environment",
        details={
            "violated_constraints": ["budget_max", "color"],
            "satisfied_constraints": ["category"],
            "result": {
                "category": "office chair",
                "color": "blue",
                "budget_max": 49.99,
                "brand": "Acme",
            },
        },
    )
    user_state = {
        "request": "Find me a black office chair under 40 dollars.",
        "constraints": {
            "category": "office chair",
            "color": "black",
            "budget_max": 40.0,
            "brand": None,
        },
        "priority": ["category", "budget_max", "color", "brand"],
    }

    shift = simulator.decide_shift(trigger, user_state)
    new_state, delta = simulator.apply_shift(user_state, shift)
    user_utterance = simulator.realize_shift(shift, user_state, style="partial")

    return {
        "trigger_input": asdict(trigger),
        "user_state_input": copy.deepcopy(user_state),
        "shift_output": asdict(shift),
        "updated_state": new_state,
        "delta": delta,
        "realized_user_utterance": user_utterance,
    }


__all__ = [
    "HumanSimulator",
    "LLMHumanSimulator",
    "LLMClientProtocol",
    "MockLLMClient",
    "build_example_usage",
]
