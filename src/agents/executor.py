from __future__ import annotations

import re
from typing import Any, Dict, List, Protocol

from agents.execution_agent import ExecutionAgent
from models import AgentAction, EnvFeedback


class WebShopSearchEnvProtocol(Protocol):
    def search_candidates(
        self,
        query: str,
        user_state: Dict[str, Any],
        *,
        search_limit: int = 50,
        return_limit: int = 10,
    ) -> EnvFeedback:
        ...


def _clean_query(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


class WebShopExecutor(ExecutionAgent):
    """
    Deterministic WebShop search executor.

    This executor intentionally does not call an LLM, inspect WebShop pages, or
    choose interactive actions. It only consumes the human simulator's
    gold_search_query, runs WebShop search, and returns candidate items.
    """

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    def act(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        query = _clean_query(env_observation.get("gold_search_query"))
        if not query:
            query = _clean_query(user_utterance)
        return AgentAction("search", {"query": query})

    def search(
        self,
        env: WebShopSearchEnvProtocol,
        current_intention: Dict[str, Any],
        user_utterance: str = "",
    ) -> tuple[AgentAction, EnvFeedback]:
        query = self._gold_search_query(current_intention, user_utterance)
        action = AgentAction("search", {"query": query})
        return action, env.search_candidates(
            query,
            current_intention,
            search_limit=50,
            return_limit=10,
        )

    def _gold_search_query(
        self,
        current_intention: Dict[str, Any],
        user_utterance: str = "",
    ) -> str:
        query = _clean_query(current_intention.get("gold_search_query"))
        if query:
            return query

        query = _clean_query(current_intention.get("search_query"))
        if query:
            return query

        query = _clean_query(current_intention.get("request"))
        if query:
            return query

        return _clean_query(user_utterance)


__all__ = ["WebShopExecutor"]
