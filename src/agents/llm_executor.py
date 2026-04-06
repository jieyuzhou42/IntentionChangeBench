from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Protocol

from agents.execution_agent import ExecutionAgent
from models import AgentAction


ALLOWED_ACTION_TYPES = {
    "search",
    "click",
    "buy",
    "back_to_search",
    "next_page",
    "prev_page",
    "refine",
}


class LLMClientProtocol(Protocol):
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        ...


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null"} else text


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


class LLMWebShopExecutor(ExecutionAgent):
    """
    LLM-driven WebShop executor with deterministic output validation.

    The LLM proposes the next action; invalid proposals fall back to a small
    deterministic emergency policy so rollout remains stable.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
    ):
        self.llm_client = llm_client

    def act(
        self,
        history: List[Dict[str, Any]],
        current_intention: Dict[str, Any],
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        prompt = self._build_prompt(history, current_intention, env_observation)
        llm_output = self._call_llm(prompt)
        action = self._parse_action_output(llm_output, env_observation)
        if action is not None:
            return action
        return self._emergency_fallback_action(current_intention, env_observation)

    def _build_prompt(
        self,
        history: List[Dict[str, Any]],
        current_intention: Dict[str, Any],
        env_observation: Dict[str, Any],
    ) -> str:
        context = {
            "current_intention": {
                "request": current_intention.get("request"),
                "constraints": copy.deepcopy(current_intention.get("constraints", {}) or {}),
                "priority": list(current_intention.get("priority", []) or []),
            },
            "page": self._serialize_observation(env_observation),
            "recent_history": self._serialize_history(history),
        }

        instructions = """
You are controlling a WebShop execution agent.
Decide the single best next action for the current page.
Return one JSON object only.

Allowed action_type values:
- search
- click
- buy
- back_to_search
- next_page
- prev_page
- refine

Rules:
- Use the current intention, page contents, item context, and recent history.
- If you need to select an option on an item page before buying, use action_type="click" with the exact option value from clickables.
- If a navigation button like "back to search", "next >", or "< prev" is the right move, prefer the dedicated action types when possible.
- Only use action_type="buy" when the current item looks like a strong match or after selecting necessary options.
- When using action_type="click", the target must exactly match one available clickable string.
- When using action_type="search" or "refine", include a non-empty query.
- If the current page provides too little evidence to buy, inspect or navigate instead.

Required JSON schema:
{
  "action_type": "search | click | buy | back_to_search | next_page | prev_page | refine",
  "action_payload": {
    "query": "string when needed",
    "target": "clickable string when needed"
  },
  "rationale": "short explanation"
}
""".strip()
        return f"{instructions}\n\nEXECUTOR_CONTEXT_JSON:\n{_safe_json_dumps(context)}"

    def _serialize_observation(self, env_observation: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = str(env_observation.get("raw_text", "") or "").strip()
        clickables = env_observation.get("clickables", []) or []
        visible_items = env_observation.get("visible_items", []) or []
        item_context = env_observation.get("item_context")
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
                "options": item_context.get("options") or {},
                "selected_options": item_context.get("selected_options") or {},
                "reviews": list(item_context.get("reviews") or [])[:3],
                "brand": item_context.get("brand"),
                "color": item_context.get("color"),
            }

        return {
            "page_type": env_observation.get("page_type"),
            "instruction": env_observation.get("instruction"),
            "selected_item": env_observation.get("selected_item"),
            "selected_asin": env_observation.get("selected_asin"),
            "selected_options": env_observation.get("selected_options"),
            "clickables": clickables[:40],
            "visible_items": visible_items[:10],
            "item_context": serialized_item_context,
            "raw_text": raw_text[:4000],
        }

    def _serialize_history(self, history: List[Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
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
            serialized.append({"role": role, "content": normalized_content})
        return serialized

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            raw_output = self.llm_client.generate_json(prompt)
        except Exception:
            return None
        return _parse_json_like(raw_output)

    def _parse_action_output(
        self,
        llm_output: Optional[Dict[str, Any]],
        env_observation: Dict[str, Any],
    ) -> Optional[AgentAction]:
        if not llm_output:
            return None

        action_type = self._normalize_action_type(llm_output.get("action_type"))
        if action_type not in ALLOWED_ACTION_TYPES:
            return None

        payload = llm_output.get("action_payload")
        if not isinstance(payload, dict):
            payload = {}

        clickables = [str(c) for c in env_observation.get("clickables", []) or []]
        clickable_map = {c.lower(): c for c in clickables}

        if action_type in {"search", "refine"}:
            query = _clean_string(payload.get("query"))
            return AgentAction(action_type, {"query": query}) if query else None

        if action_type == "click":
            target = _clean_string(payload.get("target"))
            if not target:
                return None
            canonical_target = clickable_map.get(target.lower())
            if canonical_target is None:
                return None
            return AgentAction("click", {"target": canonical_target})

        if action_type == "buy":
            return AgentAction("buy", {})

        if action_type == "back_to_search":
            return AgentAction("back_to_search", {})

        if action_type == "next_page":
            return AgentAction("next_page", {})

        if action_type == "prev_page":
            return AgentAction("prev_page", {})

        return None

    def _normalize_action_type(self, raw_action_type: Any) -> str:
        action_type = _clean_string(raw_action_type).lower()
        aliases = {
            "search_results": "search",
            "inspect": "click",
            "open": "click",
            "select": "click",
            "purchase": "buy",
            "back": "back_to_search",
            "previous_page": "prev_page",
        }
        return aliases.get(action_type, action_type)

    def _emergency_fallback_action(
        self,
        current_intention: Dict[str, Any],
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        page_type = str(env_observation.get("page_type", "search") or "search").lower()
        constraints = current_intention.get("constraints", {}) or {}

        if page_type in {"search", "unknown"}:
            query_parts = []
            for field in ("color", "brand", "category"):
                value = constraints.get(field)
                if value is not None:
                    query_parts.append(str(value))
            query = " ".join(query_parts).strip() or str(current_intention.get("request", "product") or "product")
            return AgentAction("search", {"query": query})

        if page_type == "results":
            visible_items = env_observation.get("visible_items", []) or []
            if visible_items:
                first_item = visible_items[0]
                target = (
                    first_item.get("click_target")
                    or first_item.get("asin")
                    or first_item.get("title")
                )
                target = _clean_string(target)
                if target:
                    return AgentAction("click", {"target": target})
            return AgentAction("refine", {"query": str(current_intention.get("request", "product") or "product")})

        if page_type == "item":
            option_target = self._choose_option_target(current_intention, env_observation)
            if option_target:
                return AgentAction("click", {"target": option_target})
            clickables = {str(c).lower(): str(c) for c in env_observation.get("clickables", []) or []}
            for key in ("buy now", "buy"):
                if key in clickables:
                    return AgentAction("buy", {})
            if "back to search" in clickables:
                return AgentAction("back_to_search", {})
            return AgentAction("buy", {})

        return AgentAction("search", {"query": str(current_intention.get("request", "product") or "product")})

    def _choose_option_target(
        self,
        current_intention: Dict[str, Any],
        env_observation: Dict[str, Any],
    ) -> str | None:
        item_context = env_observation.get("item_context") or {}
        if not isinstance(item_context, dict):
            return None

        options = item_context.get("options") or {}
        selected_options = item_context.get("selected_options") or {}
        clickables = {str(c).lower(): str(c) for c in env_observation.get("clickables", []) or []}
        constraints = current_intention.get("constraints", {}) or {}
        request_text = str(current_intention.get("request", "") or "").lower()

        if not isinstance(options, dict) or not clickables:
            return None

        for option_name, option_values in options.items():
            if not isinstance(option_values, list) or not option_values:
                continue

            desired_candidates: List[str] = []
            constraint_value = constraints.get(option_name)
            if constraint_value is not None:
                desired_candidates.append(str(constraint_value).lower())

            if option_name == "color" and constraints.get("color") is not None:
                desired_candidates.append(str(constraints["color"]).lower())
            if option_name == "brand" and constraints.get("brand") is not None:
                desired_candidates.append(str(constraints["brand"]).lower())

            matched = self._match_option_value(option_values, desired_candidates, request_text)
            if matched is None:
                continue

            selected_value = str(selected_options.get(option_name, "") or "").lower()
            if selected_value == matched:
                continue

            target = clickables.get(matched)
            if target:
                return target

        return None

    def _match_option_value(
        self,
        option_values: List[Any],
        desired_candidates: List[str],
        request_text: str,
    ) -> str | None:
        normalized_values = [str(v).strip().lower() for v in option_values if str(v).strip()]
        if not normalized_values:
            return None

        for candidate in desired_candidates:
            candidate = candidate.strip().lower()
            if not candidate:
                continue
            for value in normalized_values:
                if value == candidate or value in candidate or candidate in value:
                    return value

        for value in normalized_values:
            if value in request_text:
                return value

        return None


__all__ = ["LLMWebShopExecutor"]
