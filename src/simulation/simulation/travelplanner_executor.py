from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from common.execution_agent import ExecutionAgent
from models import AgentAction, EnvFeedback


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
        reference = env_observation.get("reference_information") or {}
        days = query_data.get("days") or 1
        try:
            num_days = max(1, int(days))
        except (TypeError, ValueError):
            num_days = 1
        destination = query_data.get("dest") or query_data.get("destination") or "destination city"
        origin = query_data.get("org") or query_data.get("origin") or "origin city"
        restaurant = self._first_reference_item(reference, "restaurants", f"Restaurants in {destination}")
        attraction = self._first_reference_item(reference, "attractions", f"Attractions in {destination}")
        accommodation = self._first_reference_item(reference, "accommodations", f"Accommodations in {destination}")
        restaurant_text = self._restaurant_text(restaurant) if restaurant else "-"
        attraction_text = self._attraction_text(attraction) if attraction else str(destination)
        accommodation_text = self._accommodation_text(accommodation) if accommodation else str(destination)
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
                    "breakfast": restaurant_text,
                    "lunch": restaurant_text,
                    "dinner": restaurant_text,
                    "attraction": attraction_text,
                    "accommodation": accommodation_text,
                }
            )
        return {
            "itinerary": itinerary,
            "rationale": f"Fallback plan generated from user request: {user_utterance}",
        }

    def _first_reference_item(self, reference: Dict[str, Any], *keys: str) -> Optional[Dict[str, Any]]:
        if not isinstance(reference, dict):
            return None
        for key in keys:
            values = reference.get(key)
            if isinstance(values, list) and values:
                item = values[0]
                return item if isinstance(item, dict) else None
        lowered = {str(key).lower(): value for key, value in reference.items()}
        for key in keys:
            values = lowered.get(key.lower())
            if isinstance(values, list) and values:
                item = values[0]
                return item if isinstance(item, dict) else None
        return None

    def _restaurant_text(self, item: Dict[str, Any]) -> str:
        name = item.get("name") or item.get("Name") or "restaurant"
        city = item.get("city") or item.get("City")
        cuisine = item.get("cuisine") or item.get("Cuisines")
        cost = item.get("average_cost") or item.get("Average Cost") or item.get("cost")
        parts = [str(name)]
        if city:
            parts.append(str(city))
        if cuisine:
            parts.append(str(cuisine))
        if cost is not None:
            parts.append(f"cost: {cost}")
        return ", ".join(parts)

    def _attraction_text(self, item: Dict[str, Any]) -> str:
        name = item.get("name") or item.get("Name") or "attraction"
        city = item.get("city") or item.get("City")
        return f"{name}, {city}" if city else str(name)

    def _accommodation_text(self, item: Dict[str, Any]) -> str:
        name = item.get("name") or item.get("NAME") or item.get("Name") or "accommodation"
        city = item.get("city") or item.get("City")
        room_type = item.get("room_type") or item.get("room type")
        price = item.get("price") or item.get("Price")
        parts = [str(name)]
        if city:
            parts.append(str(city))
        if room_type:
            parts.append(str(room_type))
        if price is not None:
            parts.append(f"price: {price}")
        return ", ".join(parts)


__all__ = ["TravelPlannerExecutor"]
