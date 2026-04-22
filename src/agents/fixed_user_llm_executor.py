from __future__ import annotations

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


class FixedUserLLMWebShopExecutor(ExecutionAgent):
    """
    WebShop executor that only conditions on trajectory user utterances.

    This variant deliberately ignores assistant actions and other structured
    history so the LLM only sees the recorded user utterance sequence plus the
    current page state.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
    ):
        self.llm_client = llm_client

    def act(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        prompt = self._build_prompt(history, user_utterance, env_observation)
        llm_output = self._call_llm(prompt)
        action = self._parse_action_output(llm_output, env_observation)
        if action is not None:
            return action
        return self._emergency_fallback_action(user_utterance, env_observation)

    def _build_prompt(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> str:
        context = {
            "current_user_utterance": _clean_string(user_utterance),
            "trajectory_user_utterances": self._serialize_user_utterances(
                history,
                current_user_utterance=user_utterance,
            ),
            "page": self._serialize_observation(env_observation),
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
- Use the page contents, item context, WebShop instruction text, and the trajectory user utterances only.
- Do not rely on assistant action history, rollout traces, or other context-history summaries.
- Infer the active user requirements from the latest user utterance while using earlier user utterances only as user-side trajectory context.
- If the latest user utterance changes or narrows the request, decide whether to search/refine based on that utterance and the returned items.
- When composing a search/refine query, avoid negative phrasing such as "not sexy" or "not v neck". Prefer positive descriptors that imply the same intent, such as "casual modest" or "crew neck", or leave hard-to-express negatives out of the query and filter them during item inspection.
- Use action_type="buy" when the selected item page is the final rollout result. This is a virtual finalization action, not a WebShop click.
- Do not use action_type="click" with target "Buy Now" or "Buy".
- If you need to select an option on an item page, use action_type="click" with the exact option value from clickables.
- If a navigation button like "back to search", "next >", or "< prev" is the right move, prefer the dedicated action types when possible.
- When using action_type="click", the target must exactly match one available clickable string.
- When using action_type="search" or "refine", include a non-empty query.
- If the current page provides too little evidence to select, inspect or navigate instead.

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

    def _serialize_user_utterances(
        self,
        history: List[Dict[str, Any]],
        *,
        current_user_utterance: str,
        max_items: int = 8,
    ) -> List[str]:
        utterances: List[str] = []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            if _clean_string(turn.get("role")).lower() != "user":
                continue
            content = turn.get("content")
            if isinstance(content, dict):
                continue
            utterance = _clean_string(content)
            if utterance:
                utterances.append(utterance)

        current_clean = _clean_string(current_user_utterance)
        if current_clean and (not utterances or utterances[-1] != current_clean):
            utterances.append(current_clean)
        return utterances[-max_items:]

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
            if canonical_target.lower() in {"buy now", "buy"}:
                return None
            return AgentAction("click", {"target": canonical_target})

        if action_type == "buy":
            page_type = str(env_observation.get("page_type", "") or "").lower()
            if page_type != "item":
                return None
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
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        page_type = str(env_observation.get("page_type", "search") or "search").lower()
        request_text = _clean_string(user_utterance)

        if page_type in {"search", "unknown"}:
            return AgentAction("search", {"query": request_text})

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
            return AgentAction("refine", {"query": request_text})

        if page_type == "item":
            option_target = self._choose_option_target(user_utterance, env_observation)
            if option_target:
                return AgentAction("click", {"target": option_target})
            return AgentAction("buy", {})

        return AgentAction("search", {"query": request_text})

    def _choose_option_target(
        self,
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> str | None:
        item_context = env_observation.get("item_context") or {}
        if not isinstance(item_context, dict):
            return None

        options = item_context.get("options") or {}
        selected_options = item_context.get("selected_options") or {}
        clickables = {str(c).lower(): str(c) for c in env_observation.get("clickables", []) or []}

        if not isinstance(options, dict) or not clickables:
            return None

        request_text = _clean_string(user_utterance).lower()
        for option_name, option_values in options.items():
            if not isinstance(option_values, list) or not option_values:
                continue

            matched = self._match_option_value(option_values, request_text)
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
        request_text: str,
    ) -> str | None:
        normalized_values = [str(v).strip().lower() for v in option_values if str(v).strip()]
        if not normalized_values:
            return None

        for value in normalized_values:
            if value in request_text:
                return value

        return None


__all__ = ["FixedUserLLMWebShopExecutor"]
