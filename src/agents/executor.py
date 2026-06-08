from __future__ import annotations

import re
import copy
import json
from typing import Any, Dict, List, Optional, Protocol

from agents.execution_agent import ExecutionAgent
from agents.reranker import RerankerConfig, rerank_candidates_with_llm
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

    def __init__(
        self,
        llm_client: Any = None,
        reranker_config: Optional[RerankerConfig] = None,
    ):
        self.llm_client = llm_client
        self.reranker_config = reranker_config or RerankerConfig()

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
        gold_delta: Optional[Dict[str, Any]] = None,
    ) -> tuple[AgentAction, EnvFeedback]:
        query = self._gold_search_query(current_intention, user_utterance)
        action = AgentAction("search", {"query": query})
        config = self.reranker_config
        raw_return_limit = config.rerank_top_n if config.enable_reranking else config.rerank_return_k
        env_feedback = env.search_candidates(
            query,
            current_intention,
            search_limit=max(50, raw_return_limit),
            return_limit=raw_return_limit,
        )
        if not config.enable_reranking:
            self._annotate_reranking_disabled(env_feedback, config)
            return action, env_feedback

        self._rerank_feedback_candidates(
            env,
            env_feedback,
            current_intention,
            gold_delta or {},
            config,
        )
        return action, env_feedback

    def _annotate_reranking_disabled(
        self,
        env_feedback: EnvFeedback,
        config: RerankerConfig,
    ) -> None:
        observation = env_feedback.observation or {}
        candidate_items = list(observation.get("candidate_items") or [])[: config.rerank_return_k]
        observation["candidate_items"] = candidate_items
        observation["rerank_info"] = {
            "enabled": False,
            "gold_search_query": observation.get("gold_search_query"),
            "raw_candidate_count": len(candidate_items),
            "returned_candidate_count": len(candidate_items),
            "succeeded": False,
            "fallback_used": False,
        }
        env_feedback.observation = observation

    def _rerank_feedback_candidates(
        self,
        env: WebShopSearchEnvProtocol,
        env_feedback: EnvFeedback,
        current_intention: Dict[str, Any],
        gold_delta: Dict[str, Any],
        config: RerankerConfig,
    ) -> None:
        observation = env_feedback.observation or {}
        candidate_items = list(observation.get("candidate_items") or [])
        if not candidate_items:
            observation["rerank_info"] = {
                "enabled": True,
                "gold_search_query": observation.get("gold_search_query"),
                "raw_candidate_count": 0,
                "returned_candidate_count": 0,
                "succeeded": True,
                "fallback_used": False,
            }
            env_feedback.observation = observation
            return

        reranked_items, rerank_info = rerank_candidates_with_llm(
            llm_client=self.llm_client,
            current_intention=current_intention,
            gold_delta=gold_delta,
            candidates=candidate_items[: config.rerank_top_n],
            top_k=config.rerank_return_k,
            model=config.reranker_model,
            debug=config.reranker_debug,
        )
        rerank_info["gold_search_query"] = observation.get("gold_search_query")
        observation["candidate_items"] = reranked_items
        observation["rerank_info"] = rerank_info
        if rerank_info.get("fallback_used"):
            observation["rerank_failed"] = True
            observation["rerank_error"] = rerank_info.get("rerank_error")
            observation["rerank_fallback"] = rerank_info.get("rerank_fallback")
        env_feedback.observation = observation
        self._refresh_constraint_summary(env, env_feedback, current_intention)

    def _refresh_constraint_summary(
        self,
        env: WebShopSearchEnvProtocol,
        env_feedback: EnvFeedback,
        current_intention: Dict[str, Any],
    ) -> None:
        observation = env_feedback.observation or {}
        candidate_items = list(observation.get("candidate_items") or [])
        if not candidate_items:
            return

        check_constraints = getattr(env, "_check_constraints", None)
        if not callable(check_constraints):
            return
        try:
            satisfied, violated, constraint_debug = check_constraints(
                candidate_items[0],
                current_intention,
                include_debug=True,
            )
        except Exception:
            return
        env_feedback.satisfied_constraints = satisfied
        env_feedback.violated_constraints = violated
        observation["constraint_debug"] = constraint_debug
        env_feedback.observation = observation

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


class TravelPlannerExecutor(ExecutionAgent):
    """
    LLM-backed executor that turns the current user intention into a structured
    TravelPlanner itinerary and submits it to the environment adapter.
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def act(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        plan = self._generate_plan(history, user_utterance, env_observation)
        return AgentAction("submit_plan", {"plan": plan})

    def execute(
        self,
        env,
        current_intention: Dict[str, Any],
        user_utterance: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
        gold_delta: Optional[Dict[str, Any]] = None,
    ) -> tuple[AgentAction, EnvFeedback]:
        env_observation = env.get_observation()
        prompt_history = list(history or [])
        if user_utterance:
            prompt_history.append({"role": "user", "content": user_utterance})
        plan = self._generate_plan(
            prompt_history,
            user_utterance,
            {
                **env_observation,
                "current_intention": copy.deepcopy(current_intention),
                "gold_delta": copy.deepcopy(gold_delta or {}),
            },
        )
        action = AgentAction("submit_plan", {"plan": plan})
        feedback = env.step(action, current_intention)
        return action, feedback

    def _generate_plan(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.llm_client is None or not hasattr(self.llm_client, "generate_json"):
            return self._fallback_plan(env_observation, user_utterance)

        prompt = self._build_plan_prompt(history, user_utterance, env_observation)
        try:
            raw_plan = self.llm_client.generate_json(prompt)
        except Exception:
            return self._fallback_plan(env_observation, user_utterance)
        if isinstance(raw_plan, dict):
            if isinstance(raw_plan.get("itinerary"), list):
                return raw_plan
            if isinstance(raw_plan.get("days"), list):
                return {"itinerary": raw_plan["days"], **{k: v for k, v in raw_plan.items() if k != "days"}}
        if isinstance(raw_plan, list):
            return {"itinerary": raw_plan}
        return self._fallback_plan(env_observation, user_utterance)

    def _build_plan_prompt(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> str:
        context = {
            "latest_user_utterance": user_utterance,
            "dialogue_history": history[-6:],
            "current_intention": env_observation.get("current_intention"),
            "gold_delta": env_observation.get("gold_delta"),
            "query_data": env_observation.get("query_data"),
            "reference_information": env_observation.get("reference_information"),
        }
        return f"""
You are a TravelPlanner execution agent. Create one concrete itinerary that follows the current user instruction and constraints.
Return a single JSON object only.

Required schema:
{{
  "itinerary": [
    {{
      "day": 1,
      "current_city": "City name or route context",
      "transportation": "transportation choice, include from/to and flight number or cost if available",
      "breakfast": "restaurant name, city, cuisine/cost if available",
      "lunch": "restaurant name, city, cuisine/cost if available",
      "dinner": "restaurant name, city, cuisine/cost if available",
      "attraction": "attraction name(s), city",
      "accommodation": "accommodation name, city, room type, house rules, price if available"
    }}
  ],
  "rationale": "brief reason the plan satisfies the constraints"
}}

Rules:
- Use only options present in reference_information when it is provided.
- Respect updated current_intention and gold_delta over older dialogue.
- Include prices/costs in the item strings whenever they are available.
- Do not output markdown or explanatory text outside JSON.

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()

    def _fallback_plan(self, env_observation: Dict[str, Any], user_utterance: str) -> Dict[str, Any]:
        query_data = env_observation.get("query_data") or {}
        days = query_data.get("days") or 1
        try:
            num_days = max(1, int(days))
        except (TypeError, ValueError):
            num_days = 1
        destination = query_data.get("dest") or query_data.get("destination") or "destination city"
        origin = query_data.get("org") or query_data.get("origin") or "origin city"
        itinerary = []
        for day in range(1, num_days + 1):
            itinerary.append(
                {
                    "day": day,
                    "current_city": str(destination),
                    "transportation": (
                        f"Self-driving from {origin} to {destination}"
                        if day == 1
                        else "-"
                    ),
                    "breakfast": "-",
                    "lunch": "-",
                    "dinner": "-",
                    "attraction": str(destination),
                    "accommodation": str(destination),
                }
            )
        return {
            "itinerary": itinerary,
            "rationale": f"Fallback plan generated from user request: {user_utterance}",
        }


__all__ = ["WebShopExecutor", "TravelPlannerExecutor"]
