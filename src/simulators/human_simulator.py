from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol, Tuple

from models import AgentAction, EnvFeedback, ShiftOp


ALLOWED_SHIFT_OPS = {
    "none",
    "add",
    "relax",
    "override",
    "reprioritize",
    "scope_correction",
}

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


class HumanSimulator:
    """
    LLM-backed human simulator with deterministic state application.

    `decide_shift` delegates structured reasoning to an injected client that
    returns JSON. `apply_shift` remains deterministic so the benchmark state
    transitions stay stable and easy to inspect.
    """

    def __init__(self, llm_client: LLMClientProtocol):
        self.llm_client = llm_client

    def _serialize_agent_action(self, agent_action: Optional[AgentAction]) -> Optional[Dict[str, Any]]:
        if agent_action is None:
            return None
        return {
            "action_type": agent_action.action_type,
            "action_payload": dict(agent_action.action_payload or {}),
        }

    def _serialize_env_feedback(self, env_feedback: Optional[EnvFeedback]) -> Optional[Dict[str, Any]]:
        if env_feedback is None:
            return None

        observation = env_feedback.observation or {}
        raw_text = str(observation.get("raw_text", "") or "").strip()
        clickables = observation.get("clickables", []) or []
        visible_items = observation.get("visible_items", []) or []
        item_context = observation.get("item_context")
        serialized_item_context = None
        if isinstance(item_context, dict):
            serialized_item_context = {
                "asin": item_context.get("asin"),
                "title": item_context.get("title"),
                "price": item_context.get("price"),
                "pricing": list(item_context.get("pricing") or [])[:2],
                "category": item_context.get("category"),
                "product_category": item_context.get("product_category"),
                "query": item_context.get("query"),
                "description": str(item_context.get("description", "") or "").strip()[:3000],
                "bullet_points": list(item_context.get("bullet_points") or [])[:8],
                "rating": item_context.get("rating"),
                "attributes": list(item_context.get("attributes") or [])[:12],
                "main_image": item_context.get("main_image"),
                "options": item_context.get("options") or {},
                "selected_options": item_context.get("selected_options") or {},
                "reviews": list(item_context.get("reviews") or [])[:3],
                "instruction_text": item_context.get("instruction_text"),
                "instruction_attributes": item_context.get("instruction_attributes"),
                "brand": item_context.get("brand"),
                "color": item_context.get("color"),
            }

        return {
            "status": env_feedback.status,
            "feasible": env_feedback.feasible,
            "reason": env_feedback.reason,
            "result": env_feedback.result or {},
            "satisfied_constraints": list(env_feedback.satisfied_constraints or []),
            "violated_constraints": list(env_feedback.violated_constraints or []),
            "observation": {
                "page_type": observation.get("page_type"),
                "instruction": observation.get("instruction"),
                "executed_action": observation.get("executed_action"),
                "reward": observation.get("reward"),
                "selected_item": observation.get("selected_item"),
                "selected_asin": observation.get("selected_asin"),
                "selected_options": observation.get("selected_options"),
                "item_context": serialized_item_context,
                "clickables": clickables[:40],
                "visible_items": visible_items[:10],
                "raw_text": raw_text[:4000],
            },
        }

    def _serialize_history(
        self,
        history: Optional[List[Dict[str, Any]]],
        max_items: int = 4,
    ) -> List[Dict[str, Any]]:
        if not isinstance(history, list) or not history:
            return []

        serialized: List[Dict[str, Any]] = []
        for turn in history[-max_items:]:
            if not isinstance(turn, dict):
                continue

            role = _clean_string(turn.get("role", "unknown")) or "unknown"
            content = turn.get("content")
            if isinstance(content, dict):
                normalized_content = copy.deepcopy(content)
            else:
                normalized_content = _clean_string(content)

            serialized.append(
                {
                    "role": role,
                    "content": normalized_content,
                }
            )

        return serialized

    def _build_shift_prompt(
        self,
        user_state: Dict[str, Any],
        agent_action: Optional[AgentAction] = None,
        env_feedback: Optional[EnvFeedback] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        constraints = self._constraints_from_state(user_state)
        priority = self._priority_from_state(user_state, constraints)
        context = {
            "user_state": {
                "request": user_state.get("request"),
                "constraints": constraints,
                "priority": priority,
            },
            "latest_agent_action": self._serialize_agent_action(agent_action),
            "latest_env_feedback": self._serialize_env_feedback(env_feedback),
            "recent_history": self._serialize_history(history),
        }

        instructions = """
You are simulating a human user reacting to the latest WebShop page and agent action.
Return a single JSON object only.

Allowed ops:
- none
- add
- relax
- override
- reprioritize
- scope_correction

Rules:
- First decide whether the user would keep the current intention or revise it after seeing this page.
- Prefer repairing the agent or clarifying the original request before changing preferences.
- If the desired option is visibly available on the page but not selected, prefer scope_correction on the original field/value.
- Only relax lower-priority constraints when the page suggests the original request may be hard to satisfy.
- Keep high-priority constraints stable unless the evidence strongly supports changing them.
- Allow natural override behavior when the inspected result genuinely changes the user's mind.
- Treat adapter-provided status / feasible / reason as hints, not ground truth. Use the page text, visible items, selected item, and action context as the main evidence.
- If no change is appropriate, return op="none".

Required JSON schema:
{
  "op": "none | add | relax | override | reprioritize | scope_correction",
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
        user_state: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> ShiftOp:
        if not llm_output:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        constraints = self._constraints_from_state(user_state)
        priority = self._priority_from_state(user_state, constraints)
        result = (env_feedback.result if env_feedback is not None else None) or {}

        op = _clean_string(llm_output.get("op", "none")).lower()
        if op not in ALLOWED_SHIFT_OPS:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        field = self._match_field_name(llm_output.get("field"), constraints, priority)
        if op == "add" and field is None:
            field = self._normalize_new_field_name(llm_output.get("field"))
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

        if field is None:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        if op != "add" and field not in constraints:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        current_value = constraints.get(field)
        old_value = self._coerce_value(field, llm_output.get("old_value"), current_value)
        if old_value is None:
            old_value = current_value

        raw_value = llm_output.get("value")
        value = self._coerce_value(field, raw_value, old_value)
        if value is None and op == "relax":
            value = self._default_relax_value(field, old_value)
        if value is None and op in {"add", "override", "scope_correction"}:
            value = result.get(field)
            value = self._coerce_value(field, value, old_value)

        if op == "add" and old_value is not None:
            return ShiftOp(op="none", rationale="invalid_llm_output")

        if op in {"add", "override", "scope_correction"} and value is None:
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
        user_state: Dict[str, Any],
        agent_action: Optional[AgentAction] = None,
        env_feedback: Optional[EnvFeedback] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> ShiftOp:
        prompt = self._build_shift_prompt(
            user_state,
            agent_action=agent_action,
            env_feedback=env_feedback,
            history=history,
        )
        llm_output = self._call_llm_for_shift(prompt)
        return self._parse_shift_output(llm_output, user_state, env_feedback=env_feedback)

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

        if shift.op in {"add", "relax", "override", "scope_correction"} and shift.field:
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

        new_state["priority"] = self._priority_from_state(new_state, new_state["constraints"])
        return new_state, delta

    def _build_realization_prompt(
        self,
        shift: ShiftOp,
        user_state: Dict[str, Any],
        style: str,
        agent_action: Optional[AgentAction] = None,
        env_feedback: Optional[EnvFeedback] = None,
        history: Optional[List[Dict[str, Any]]] = None,
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
            "latest_agent_action": self._serialize_agent_action(agent_action),
            "latest_env_feedback": self._serialize_env_feedback(env_feedback),
            "recent_history": self._serialize_history(history),
        }

        instructions = """
Write the user's next utterance as a single short sentence.
Ground the utterance strictly in the structured shift decision.
Do not invent new constraints or changes that are not present in the shift object.
Make the utterance responsive to the latest agent action and the current page feedback when that context is relevant.

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

    def realize_shift(
        self,
        shift: ShiftOp,
        user_state: Dict[str, Any],
        style: str,
        agent_action: Optional[AgentAction] = None,
        env_feedback: Optional[EnvFeedback] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        prompt = self._build_realization_prompt(
            shift,
            user_state,
            style,
            agent_action=agent_action,
            env_feedback=env_feedback,
            history=history,
        )
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

        if shift.op == "add":
            if effective_style == "explicit":
                return f"Please also add {field_text} {value_text}."
            if effective_style == "partial":
                return f"Also make it {value_text}."
            return f"{field_text} {value_text} too."

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

    def _normalize_new_field_name(self, raw_field: Any) -> Optional[str]:
        candidate = _clean_string(raw_field)
        if not candidate:
            return None
        normalized = candidate.replace(" ", "_").lower()
        return normalized or None

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


def build_example_usage(llm_client: LLMClientProtocol) -> Dict[str, Any]:
    """
    Small end-to-end example that exercises the three-stage flow with an
    injected LLM client.
    """

    simulator = HumanSimulator(llm_client=llm_client)
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
    env_feedback = EnvFeedback(
        status="observed",
        feasible=True,
        reason="no_matching_results",
        observation={
            "page_type": "results",
            "raw_text": "No results matched the request for a black office chair under $40.",
            "visible_items": [],
            "clickables": ["back to search"],
        },
        result={
            "category": "office chair",
            "color": "blue",
            "price": 49.99,
            "brand": "Acme",
        },
        satisfied_constraints=["category"],
        violated_constraints=["budget_max", "color"],
    )

    shift = simulator.decide_shift(user_state, env_feedback=env_feedback)
    new_state, delta = simulator.apply_shift(user_state, shift)
    user_utterance = simulator.realize_shift(
        shift,
        user_state,
        style="partial",
        env_feedback=env_feedback,
    )

    return {
        "user_state_input": copy.deepcopy(user_state),
        "env_feedback_input": asdict(env_feedback),
        "shift_output": asdict(shift),
        "updated_state": new_state,
        "delta": delta,
        "realized_user_utterance": user_utterance,
    }


__all__ = [
    "HumanSimulator",
    "LLMClientProtocol",
    "build_example_usage",
]
