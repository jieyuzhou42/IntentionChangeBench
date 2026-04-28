from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol, Tuple

from models import AgentAction, EnvFeedback, ShiftOp
from prompt_logging import log_prompt


ALLOWED_SHIFT_OPS = {
    "none",
    "add",
    "relax",
    "override",
    "reprioritize",
    "scope_correction",
}
ALLOWED_SHIFT_CONDITIONS = {
    "none",
    "user_preference",
    "real_world_feasibility",
    "agent_misunderstanding",
}
ALLOWED_CHANGE_CATEGORIES = ALLOWED_SHIFT_OPS

ALLOWED_STYLES = {"explicit", "partial", "elliptical"}
ALLOWED_DIRECTNESS = {"direct", "indirect"}
SHIFT_CONTEXT_MARKER = "SHIFT_CONTEXT_JSON:"
SEARCH_QUERY_CONTEXT_MARKER = "SEARCH_QUERY_CONTEXT_JSON:"
REALIZATION_CONTEXT_MARKER = "REALIZATION_CONTEXT_JSON:"
FORCED_SHIFT_RETRY_PROBABILITY = 0.5


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

    def _serialize_env_feedback(self, env_feedback: Optional[EnvFeedback]) -> Optional[Dict[str, Any]]:
        if env_feedback is None:
            return None

        observation = env_feedback.observation or {}
        return {
            "feedback_type": "candidate_items",
            "status": env_feedback.status,
            "page_type": observation.get("page_type"),
            "candidate_items": copy.deepcopy(list(observation.get("candidate_items") or [])[:10]),
            "selected_candidate": copy.deepcopy(observation.get("selected_candidate")),
            "rerank_info": copy.deepcopy(observation.get("rerank_info")),
        }

    def _serialize_intention_timeline(
        self,
        current_intention: Dict[str, Any],
        intention_history: Optional[List[Dict[str, Any]]],
        max_items: int = 4,
    ) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        if isinstance(intention_history, list):
            for turn in intention_history[-max_items:]:
                if not isinstance(turn, dict):
                    continue

                gold_intention = copy.deepcopy(turn.get("gold_intention") or {})
                serialized.append(
                    {
                        "turn_id": turn.get("turn_id"),
                        "is_current": False,
                        "constraints": copy.deepcopy(gold_intention.get("constraints") or {}),
                        "gold_search_query": gold_intention.get("gold_search_query"),
                    }
                )

        serialized.append(
            {
                "turn_id": "current",
                "is_current": True,
                "constraints": copy.deepcopy(current_intention.get("constraints") or {}),
                "gold_search_query": current_intention.get("gold_search_query"),
            }
        )

        return serialized

    def _build_shift_prompt(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        context = {
            "intention_timeline": self._serialize_intention_timeline(current_intention, intention_history),
            "latest_env_feedback": self._serialize_env_feedback(env_feedback),
        }

        instructions = """
Pretend you are a real user shopping on amazon.com.
Return a single JSON object only.

Allowed conditions:
- user_preference
- real_world_feasibility

Allowed change categories:
- add
- relax
- override
- reprioritize
- scope_correction

Task:
- Feel free to change your primary goal or constraints entirely based on your whims or what you see on the page. Your initial goal is just a starting point, not a contract.
- You are currently dissatisfied. You MUST either add a new constraint, relax a constrain, override an existing one, reprioritize some constrains over others, or shift your entire focus.

Rules:
- Treat searching status as a low-level environment signal, not ground truth. Use the candidate items and their constraint matches as the main evidence.
- Use condition="user_preference" when the user changes or adds preferences because of what they just saw.
- Use condition="real_world_feasibility" when exact constraints seem unavailable or hard to satisfy.
- Do not introduce correction, termination, or no_change_continue as top-level reaction classes.
- Do not mention rating/review/star/customer-score constraints. Those signals are unavailable to the executor and are out of scope for this simulator run
- Do not repeatedly toggle between two values across turns.


Required JSON schema:
{
  "intention_changed": true,
  "condition": "user_preference | real_world_feasibility",
  "category": "add | relax | override | reprioritize | scope_correction",
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

Examples:
{
  "intention_changed": true,
  "condition": "user_preference",
  "category": "override",
  "field": "color",
  "old_value": "green stripe",
  "value": "navy",
  "priority_update": null,
  "rationale": "After seeing the pattern, the user now prefers a different color.",
  "utterance_plan": {"style": "partial", "directness": "direct", "mention_old_value": false}
}
{
  "intention_changed": true,
  "condition": "real_world_feasibility",
  "category": "relax",
  "field": "color",
  "old_value": "green stripe",
  "value": null,
  "priority_update": null,
  "rationale": "The exact option does not seem available, so something close is acceptable.",
  "utterance_plan": {"style": "partial", "directness": "direct", "mention_old_value": true}
}
""".strip()

        return f"{instructions}\n\n{SHIFT_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def _preview_intention_after_shift(
        self,
        current_intention: Dict[str, Any],
        shift: ShiftOp,
    ) -> Dict[str, Any]:
        preview = copy.deepcopy(current_intention)
        preview.setdefault("constraints", {})
        constraints = preview["constraints"] if isinstance(preview["constraints"], dict) else {}
        preview["constraints"] = constraints

        if shift.op in {"add", "relax", "override", "scope_correction"} and shift.field:
            constraints[shift.field] = shift.value
        if shift.priority_update:
            preview["priority"] = self._normalize_priority_update(
                shift.priority_update,
                constraints,
                preview.get("priority", []),
            ) or preview.get("priority", [])
        preview.pop("gold_search_query", None)
        preview["priority"] = self._priority_from_state(preview, constraints)
        return preview

    def _build_gold_search_query_prompt(
        self,
        updated_gold_intention: Dict[str, Any],
    ) -> str:
        context = {
            "updated_gold_intention": copy.deepcopy(updated_gold_intention),
        }

        instructions = """
Generate the gold WebShop search query for current shopping intention.
Return a single JSON object only.

Task:
- Produce gold_search_query for retrieving a broad but relevant candidate pool.
- The search query should primarily identify the correct product type/category.
- Use concise keyword phrases, not a full sentence.
- Include only the most search-critical positive descriptors when they are necessary to identify the product family, such as gender, product type.
- Do not include too many fine-grained constraints in the search query. Fine-grained constraints will be evaluated later by a separate candidate filtering/reranking module.
- Do not include budget/price limits unless those words are naturally part of the product name; budget is evaluated after search.
- Do not include rating/review/star/customer-score constraints.
- Do not include washing/care constraints such as "machine wash" unless the user is explicitly searching for a care-related product.
- Do not include exact size constraints such as "large", "x-large", or "4x-large" unless size is central to the product type, such as "plus size".
- Avoid negative phrasing such as "not v neck". Prefer positive descriptors that imply the same intent, such as "crew neck".

Required JSON schema:
{
  "gold_search_query": "concise WebShop search query for broad candidate retrieval"
}

Examples:
{"gold_search_query": "women's jumpsuits rompers overalls"}
{"gold_search_query": "men's henley shirt"}
{"gold_search_query": "men's loafers slip ons"}
{"gold_search_query": "casual modest crew neck dress"}
{"gold_search_query": "navy office chair"}
""".strip()

        return f"{instructions}\n\n{SEARCH_QUERY_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def _call_llm_for_gold_search_query(self, prompt: str) -> Optional[str]:
        log_prompt("simulator.gold_search_query", prompt)
        try:
            raw_output = self.llm_client.generate_json(prompt)
        except Exception:
            return None

        parsed = _parse_json_like(raw_output)
        if not parsed:
            return None
        return self._normalize_gold_search_query(parsed.get("gold_search_query"))

    def generate_gold_search_query_for_intention(
        self,
        gold_intention: Dict[str, Any],
    ) -> Optional[str]:
        prompt = self._build_gold_search_query_prompt(gold_intention)
        query = self._call_llm_for_gold_search_query(prompt)
        if query:
            return query
        return self._query_from_intention(gold_intention)

    def _generate_gold_search_query(
        self,
        current_intention: Dict[str, Any],
        shift: ShiftOp,
    ) -> Optional[str]:
        if shift.op == "none":
            return self._query_from_intention(current_intention)

        return self.generate_gold_search_query_for_intention(
            self._preview_intention_after_shift(current_intention, shift)
        )

    def _call_llm_for_shift(self, prompt: str) -> Optional[Dict[str, Any]]:
        log_prompt("simulator.shift", prompt)
        try:
            raw_output = self.llm_client.generate_json(prompt)
        except Exception:
            return None
        return _parse_json_like(raw_output)

    def _parse_shift_output(
        self,
        llm_output: Optional[Dict[str, Any]],
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> ShiftOp:
        if not llm_output:
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_llm_output")

        constraints = self._constraints_from_state(current_intention)
        priority = self._priority_from_state(current_intention, constraints)
        condition = self._normalize_shift_condition(llm_output.get("condition"))
        change_category = self._normalize_change_category(
            llm_output.get("category", llm_output.get("change_category"))
        )
        op = self._normalize_change_category(llm_output.get("op"))
        if op == "none" and change_category != "none":
            op = change_category
        if change_category == "none" and op != "none":
            change_category = op

        raw_intention_changed = llm_output.get("intention_changed")
        if isinstance(raw_intention_changed, bool):
            intention_changed = raw_intention_changed
        else:
            intention_changed = change_category != "none" or op != "none"

        raw_field = llm_output.get("field")
        field = self._match_field_name(raw_field, constraints, priority)
        normalized_new_field = None
        if field is None:
            normalized_new_field = self._normalize_new_field_name(raw_field)
            if normalized_new_field and normalized_new_field not in constraints and normalized_new_field not in priority:
                field = normalized_new_field
                if change_category not in {"add", "reprioritize"}:
                    change_category = "add"
                if op not in {"add", "reprioritize"}:
                    op = "add"
        rationale = _clean_string(llm_output.get("rationale", "")) or "llm_decision"
        priority_update = self._normalize_priority_update(
            llm_output.get("priority_update"),
            constraints,
            priority,
        )
        utterance_plan = self._normalize_utterance_plan(llm_output.get("utterance_plan"))
        gold_search_query = self._normalize_gold_search_query(llm_output.get("gold_search_query"))

        if not intention_changed:
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale=rationale,
                priority_update=priority_update,
                utterance_plan=utterance_plan,
                gold_search_query=gold_search_query or self._query_from_intention(current_intention),
            )

        if change_category not in ALLOWED_CHANGE_CATEGORIES or change_category == "none":
            inferred_category = self._infer_change_category(
                llm_output=llm_output,
                field=field,
                priority_update=priority_update,
            )
            change_category = inferred_category
        if change_category not in ALLOWED_CHANGE_CATEGORIES or change_category == "none":
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_llm_output")
        if op not in ALLOWED_SHIFT_OPS or op == "none":
            op = change_category
        if condition not in ALLOWED_SHIFT_CONDITIONS or condition == "none":
            condition = self._infer_shift_condition(
                op=op,
                llm_output=llm_output,
                env_feedback=env_feedback,
                field=field,
            )
        if condition not in ALLOWED_SHIFT_CONDITIONS or condition == "none":
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_llm_output")

        if op == "reprioritize":
            if priority_update is None:
                if field is None:
                    return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_llm_output")
                priority_update = self._move_field_to_front(field, priority)
            return ShiftOp(
                op="reprioritize",
                intention_changed=True,
                condition=condition,
                change_category=change_category,
                field=field,
                old_value=priority,
                value=priority_update,
                rationale=rationale,
                priority_update=priority_update,
                utterance_plan=utterance_plan,
                gold_search_query=gold_search_query or self._query_from_intention(current_intention),
            )

        if field is None:
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_llm_output")

        is_new_field = field not in constraints
        if is_new_field and op != "reprioritize":
            op = "add"
            change_category = "add"

        current_value = constraints.get(field)
        old_value = None if is_new_field else self._coerce_value(field, llm_output.get("old_value"), current_value)
        if old_value is None and not is_new_field:
            old_value = current_value

        raw_value = llm_output.get("value")
        value = self._coerce_value(field, raw_value, old_value)
        if value is None and op == "relax":
            value = self._default_relax_value(field, old_value)
        if op == "add" and old_value is not None:
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_llm_output")

        if op in {"add", "override", "scope_correction"} and value is None:
            return ShiftOp(op="none", intention_changed=False, condition="none", change_category="none", rationale="invalid_llm_output")

        if op == "relax" and old_value is not None and not self._looks_like_relaxation(field, old_value, value):
            value = self._default_relax_value(field, old_value)

        return ShiftOp(
            op=op,
            intention_changed=True,
            condition=condition,
            change_category=change_category,
            field=field,
            old_value=old_value,
            value=value,
            rationale=rationale,
            priority_update=priority_update,
            utterance_plan=utterance_plan,
            gold_search_query=gold_search_query,
        )

    def decide_shift(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ShiftOp:
        prompt = self._build_shift_prompt(
            current_intention,
            env_feedback=env_feedback,
            intention_history=intention_history,
        )
        llm_output = self._call_llm_for_shift(prompt)
        if (
            llm_output
            and not bool(llm_output.get("intention_changed", True))
            and random.random() < FORCED_SHIFT_RETRY_PROBABILITY
        ):
            prompt = (
                f"{prompt}\n\n"
                "CRITICAL: You are too satisfied. Find a reason to change your mind or goal NOW."
            )
            llm_output = self._call_llm_for_shift(prompt)
        shift = self._parse_shift_output(llm_output, current_intention, env_feedback=env_feedback)
        shift.gold_search_query = self._generate_gold_search_query(
            current_intention,
            shift,
        )
        return shift

    def apply_shift(
        self,
        current_intention: Dict[str, Any],
        shift: ShiftOp,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        new_state = copy.deepcopy(current_intention)
        new_state.setdefault("constraints", {})
        new_state["priority"] = self._priority_from_state(new_state, new_state["constraints"])
        delta: Dict[str, Dict[str, Any]] = {}

        if shift.op == "none":
            new_state["gold_search_query"] = (
                self._normalize_gold_search_query(shift.gold_search_query)
                or self._query_from_intention(new_state)
            )
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

        new_state["gold_search_query"] = (
            self._normalize_gold_search_query(shift.gold_search_query)
            or self._query_from_intention(new_state)
        )
        new_state["priority"] = self._priority_from_state(new_state, new_state["constraints"])
        return new_state, delta

    def _build_realization_prompt(
        self,
        shift: ShiftOp,
        current_intention: Dict[str, Any],
        style: str,
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        requested_style = style if style in ALLOWED_STYLES else "explicit"
        context = {
            "requested_style": requested_style,
            "intention_timeline": self._serialize_intention_timeline(current_intention, intention_history),
            "shift": asdict(shift),
            "latest_env_feedback": self._serialize_env_feedback(env_feedback),
        }

        instructions = """
Write the user's next utterance as a single short sentence.
Ground the utterance strictly in the structured shift decision.
Do not invent new constraints or changes that are not present in the shift object.
Make the utterance responsive to the current page feedback when that context is relevant.
The utterance should sound consistent with the chosen change_category:
- add: add one more preference naturally
- relax: soften an exact requirement
- override: replace the old preference with a new one
- reprioritize: shift which constraint matters more
- scope_correction: clarify or refine the intended value
Keep the language concise and realistic.
Do not always make it overly explicit.

Style guide:
- explicit: directly state the change
- partial: hint at the change naturally
- elliptical: short and fragment-like

Return plain text only, with no quotes and no JSON.
""".strip()

        return f"{instructions}\n\n{REALIZATION_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def _call_llm_for_realization(self, prompt: str) -> Optional[str]:
        log_prompt("simulator.realization", prompt)
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
        current_intention: Dict[str, Any],
        style: str,
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        prompt = self._build_realization_prompt(
            shift,
            current_intention,
            style,
            env_feedback=env_feedback,
            intention_history=intention_history,
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
            return "Let's keep looking."

        if shift.op == "relax":
            if effective_style == "explicit":
                if mention_old_value and shift.old_value is not None:
                    return f"It doesn't have to be exactly {old_text}; {value_text} is fine."
                return f"It doesn't have to be exactly {field_text}."
            if effective_style == "partial":
                return f"{field_text} can be a bit flexible."
            return "Close is fine."

        if shift.op == "add":
            if effective_style == "explicit":
                return f"Also, I'd prefer {field_text} {value_text}."
            if effective_style == "partial":
                return f"Also, {value_text} would be nice."
            return f"{value_text} too."

        if shift.op == "override":
            if effective_style == "explicit":
                if mention_old_value and shift.old_value is not None:
                    return f"Actually, change {field_text} from {old_text} to {value_text}."
                return f"Actually, make it {value_text} instead."
            if effective_style == "partial":
                return f"Let's go with {value_text} instead."
            return f"{value_text} instead."

        if shift.op == "reprioritize":
            target = shift.priority_update[0] if shift.priority_update else shift.field
            target_text = str(target).replace("_", " ") if target else "that"
            if effective_style == "explicit":
                return f"{target_text.capitalize()} matters more now."
            if effective_style == "partial":
                return f"Focus more on {target_text}."
            return f"{target_text} first."

        if shift.op == "scope_correction":
            if effective_style == "explicit":
                if mention_old_value and shift.old_value is not None:
                    return f"I mean {value_text}, not just {old_text}."
                return f"I mean {field_text} {value_text}."
            if effective_style == "partial":
                return f"More specifically, {value_text}."
            return f"{field_text} {value_text}."

        return "Please update that requirement."

    def _constraints_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        constraints = state.get("constraints", {}) or {}
        return constraints if isinstance(constraints, dict) else {}

    def _normalize_gold_search_query(self, value: Any) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        return text

    def _query_from_intention(self, state: Dict[str, Any]) -> Optional[str]:
        query = self._normalize_gold_search_query(state.get("gold_search_query"))
        if query:
            return query

        constraints = self._constraints_from_state(state)
        parts: List[str] = []
        category = constraints.get("category")
        if category:
            parts.append(str(category))
        for field in ("color", "brand", "size"):
            value = constraints.get(field)
            if value is not None:
                parts.append(str(value))
        if parts:
            return re.sub(r"\s+", " ", " ".join(parts)).strip()
        return self._normalize_gold_search_query(state.get("request"))

    def _priority_from_state(
        self,
        state: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> List[str]:
        raw_priority = state.get("priority", [])
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

    def _normalize_shift_condition(self, raw_condition: Any) -> str:
        candidate = _clean_string(raw_condition).lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "": "none",
            "null": "none",
            "preference": "user_preference",
            "preference_change": "user_preference",
            "userpreference": "user_preference",
            "feasibility": "real_world_feasibility",
            "real_world": "real_world_feasibility",
            "realworldfeasibility": "real_world_feasibility",
            "agent_error": "agent_misunderstanding",
            "misunderstanding": "agent_misunderstanding",
        }
        normalized = aliases.get(candidate, candidate)
        return normalized if normalized in ALLOWED_SHIFT_CONDITIONS else "none"

    def _normalize_change_category(self, raw_category: Any) -> str:
        candidate = _clean_string(raw_category).lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "": "none",
            "null": "none",
            "scope(entity)_correction": "scope_correction",
            "scope_entity_correction": "scope_correction",
        }
        normalized = aliases.get(candidate, candidate)
        return normalized if normalized in ALLOWED_CHANGE_CATEGORIES else "none"

    def _infer_change_category(
        self,
        llm_output: Dict[str, Any],
        field: Optional[str],
        priority_update: Optional[List[str]],
    ) -> str:
        if priority_update:
            return "reprioritize"
        raw_value = _normalize_none_like(llm_output.get("value"))
        raw_old_value = _normalize_none_like(llm_output.get("old_value"))
        rationale = _clean_string(llm_output.get("rationale", "")).lower()
        if "refin" in rationale or "clarif" in rationale or "specific" in rationale:
            return "scope_correction"
        if raw_old_value is not None and raw_value is not None and raw_old_value != raw_value:
            return "override"
        if raw_old_value is not None and raw_value is None:
            return "relax"
        if field and raw_old_value is None and raw_value is not None:
            return "add"
        return "none"

    def _infer_shift_condition(
        self,
        op: str,
        llm_output: Dict[str, Any],
        env_feedback: Optional[EnvFeedback],
        field: Optional[str],
    ) -> str:
        rationale = _clean_string(llm_output.get("rationale", "")).lower()
        preference_cues = (
            "suddenly",
            "don't want",
            "do not want",
            "want something else",
            "rather",
            "instead",
            "bored",
            "tired of",
            "change my mind",
            "changed my mind",
            "don't like",
            "do not like",
        )
        if op == "scope_correction":
            return "agent_misunderstanding"
        if any(cue in rationale for cue in preference_cues):
            return "user_preference"
        if "unavailable" in rationale or "hard to satisfy" in rationale or "not available" in rationale:
            return "real_world_feasibility"
        if "misunder" in rationale or "clarif" in rationale or "too coarse" in rationale:
            return "agent_misunderstanding"
        return "user_preference"

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
    current_intention = {
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

    shift = simulator.decide_shift(current_intention, env_feedback=env_feedback)
    new_state, delta = simulator.apply_shift(current_intention, shift)
    user_utterance = simulator.realize_shift(
        shift,
        current_intention,
        style="partial",
        env_feedback=env_feedback,
    )

    return {
        "current_intention_input": copy.deepcopy(current_intention),
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
