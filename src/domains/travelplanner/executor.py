from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.execution_agent import ExecutionAgent
from models import AgentAction


ORIGINAL_ACTION_TYPES = (
    "FlightSearch",
    "AttractionSearch",
    "AccommodationSearch",
    "RestaurantSearch",
    "CitySearch",
    "GoogleDistanceMatrix",
    "NotebookWrite",
    "Planner",
)

CITY_TOOL_ACTIONS = (
    "AttractionSearch",
    "AccommodationSearch",
    "RestaurantSearch",
)


class TravelPlannerExecutor(ExecutionAgent):
    """ReAct-style executor using the original TravelPlanner action vocabulary."""

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def act(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        if self.llm_client is None or not hasattr(self.llm_client, "generate_json"):
            raise RuntimeError("TravelPlannerExecutor requires an LLM client with generate_json().")

        prompt = self._build_action_prompt(history, user_utterance, env_observation)
        raw_action = self.llm_client.generate_json(prompt)

        action = self._action_from_llm(raw_action)
        if action is None:
            raise ValueError(f"TravelPlanner executor returned an invalid action: {raw_action!r}")
        if env_observation.get("pending_notebook") and action.action_type != "NotebookWrite":
            raise ValueError(
                "TravelPlanner executor must use NotebookWrite immediately after a tool result; "
                f"received {action.action_type}."
            )
        if action.action_type == "Planner":
            if not self._minimum_research_complete(env_observation):
                missing = self._missing_required_actions(env_observation)
                raise ValueError(
                    "TravelPlanner executor invoked Planner before required research was complete; "
                    f"missing actions: {missing!r}."
                )
            payload = dict(action.action_payload or {})
            if not self._payload_has_plan(payload):
                payload["plan"] = self._generate_plan(history, user_utterance, env_observation)
            payload.setdefault("query", user_utterance or env_observation.get("instruction", ""))
            return self._with_original_argument(AgentAction(
                "Planner",
                payload,
                rationale=action.rationale,
                predicted_current_intention=action.predicted_current_intention,
            ))
        return self._with_original_argument(action)

    def _with_original_argument(self, action: AgentAction) -> AgentAction:
        payload = dict(action.action_payload or {})
        if payload.get("argument") is None:
            if action.action_type == "FlightSearch":
                argument = ", ".join(str(payload.get(key) or "") for key in ("origin", "destination", "date"))
            elif action.action_type == "GoogleDistanceMatrix":
                argument = ", ".join(str(payload.get(key) or "") for key in ("origin", "destination", "mode"))
            elif action.action_type == "CitySearch":
                argument = payload.get("state") or payload.get("region") or payload.get("query") or ""
            elif action.action_type in CITY_TOOL_ACTIONS:
                argument = payload.get("city") or payload.get("query") or ""
            elif action.action_type == "NotebookWrite":
                argument = payload.get("description") or payload.get("short_description") or ""
            else:
                argument = payload.get("query") or ""
            payload["argument"] = str(argument).strip()
        return AgentAction(
            action.action_type,
            payload,
            rationale=action.rationale,
            predicted_current_intention=copy.deepcopy(action.predicted_current_intention),
        )

    def _build_action_prompt(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        observation: Dict[str, Any],
    ) -> str:
        notebook = observation.get("notebook") or []
        compact_notebook = [
            {
                "index": entry.get("index"),
                "short_description": entry.get("short_description"),
                "source_action": entry.get("source_action"),
            }
            for entry in notebook
            if isinstance(entry, dict)
        ]
        context = {
            "latest_user_utterance": user_utterance,
            "dialogue_history": self._compact_history(history[-6:]),
            "current_intention": observation.get("current_intention"),
            "query_data": observation.get("query_data"),
            "available_cities": observation.get("available_cities"),
            "last_tool_name": observation.get("tool_name"),
            "last_tool_argument": observation.get("tool_argument"),
            "last_tool_result": observation.get("tool_result"),
            "pending_notebook": observation.get("pending_notebook", False),
            "notebook_index": compact_notebook,
            "completed_actions": observation.get("completed_actions") or [],
            "required_action_checklist": self._required_action_checklist(observation),
            "missing_required_actions": self._missing_required_actions(observation),
        }
        return f"""
You are the original TravelPlanner ReAct execution agent. Select exactly one next action.
Return one JSON object only, with keys action_type, action_payload, and rationale.

The only valid action_type values and payloads are:
- FlightSearch: {{"origin": "city", "destination": "city", "date": "YYYY-MM-DD"}}
- AttractionSearch: {{"city": "city"}}
- AccommodationSearch: {{"city": "city"}}
- RestaurantSearch: {{"city": "city"}}
- CitySearch: {{"state": "state name"}}
- GoogleDistanceMatrix: {{"origin": "city", "destination": "city", "mode": "self-driving or taxi"}}
- NotebookWrite: {{"description": "short description of the immediately preceding tool result"}}
- Planner: {{"query": "the current travel request"}}

Rules:
- Reproduce the original TravelPlanner workflow: search, write useful results to Notebook, then call Planner.
- If pending_notebook is true, use NotebookWrite before another search.
- Search attractions, accommodations, and restaurants for every selected destination city.
- Search transportation for every required route leg.
- Use CitySearch when the destination is a state/region and multiple cities must be chosen.
- Invoke Planner only after the required information has been collected.
- missing_required_actions is authoritative: complete every listed action before Planner.
- Respect the latest current intention over older dialogue.
- Never invent tool results and never return markdown.

Context:
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}
""".strip()

    def _action_from_llm(self, raw: Any) -> Optional[AgentAction]:
        if not isinstance(raw, dict):
            return None
        action_type = self._normalize_action_type(raw.get("action_type") or raw.get("action"))
        if action_type not in ORIGINAL_ACTION_TYPES:
            return None
        payload = raw.get("action_payload") or raw.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {"argument": str(payload)}
        if action_type == "Planner" and "plan" not in payload:
            if isinstance(raw.get("plan"), (dict, list)):
                payload["plan"] = copy.deepcopy(raw["plan"])
            elif isinstance(raw.get("itinerary"), list):
                payload["plan"] = {"itinerary": copy.deepcopy(raw["itinerary"])}
        return AgentAction(
            action_type,
            payload,
            rationale=str(raw.get("rationale") or "").strip() or None,
            predicted_current_intention=(
                copy.deepcopy(raw.get("predicted_current_intention"))
                if isinstance(raw.get("predicted_current_intention"), dict)
                else None
            ),
        )

    def _compact_history(self, history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if isinstance(content, dict):
                content = {
                    key: copy.deepcopy(content.get(key))
                    for key in (
                        "action_type",
                        "action_payload",
                        "page_type",
                        "tool_name",
                        "tool_argument",
                        "notebook_size",
                        "internal_step",
                    )
                    if content.get(key) is not None
                }
                action_payload = content.get("action_payload")
                if isinstance(action_payload, dict) and "plan" in action_payload:
                    content["action_payload"] = {
                        key: value for key, value in action_payload.items() if key != "plan"
                    }
            compact.append({"role": entry.get("role"), "content": content})
        return compact

    def _normalize_action_type(self, value: Any) -> str:
        text = str(value or "").strip()
        aliases = {action.lower(): action for action in ORIGINAL_ACTION_TYPES}
        aliases.update({
            "flight_search": "FlightSearch",
            "attraction_search": "AttractionSearch",
            "accommodation_search": "AccommodationSearch",
            "restaurant_search": "RestaurantSearch",
            "city_search": "CitySearch",
            "distance_search": "GoogleDistanceMatrix",
            "google_distance_matrix": "GoogleDistanceMatrix",
            "notebook_write": "NotebookWrite",
            "submit_plan": "Planner",
            "plan": "Planner",
        })
        return aliases.get(text.lower(), text)

    def _minimum_research_complete(self, observation: Dict[str, Any]) -> bool:
        return not self._missing_required_actions(observation)

    def _required_action_checklist(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_data = self._effective_query_data(observation)
        cities = self._destination_cities(observation, query_data)
        if not cities:
            return [{"action_type": "destination_city_unresolved", "action_payload": {}}]
        checklist = [
            {"action_type": action_type, "action_payload": {"city": city}, "argument": city}
            for city in cities
            for action_type in CITY_TOOL_ACTIONS
        ]
        checklist.extend(
            {"action_type": action_type, "action_payload": payload, "argument": argument}
            for action_type, payload, argument in self._transport_actions(query_data, cities)
        )
        return checklist

    def _missing_required_actions(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        completed = [str(item) for item in observation.get("completed_actions") or []]
        return [
            item
            for item in self._required_action_checklist(observation)
            if not self._completed(completed, item["action_type"], item.get("argument", ""))
        ]

    def _completed(self, completed: Sequence[str], action_type: str, argument: str) -> bool:
        prefix = f"{action_type}:".lower()
        target = str(argument or "").strip().lower()
        return any(item.lower().startswith(prefix) and target in item.lower() for item in completed)

    def _effective_query_data(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        query_data = copy.deepcopy(observation.get("query_data") or {})
        intention = observation.get("current_intention") or {}
        constraints = intention.get("constraints") if isinstance(intention, dict) else {}
        if not isinstance(constraints, dict):
            return query_data
        aliases = {
            "budget": "budget",
            "budget_max": "budget",
            "days": "days",
            "people_number": "people_number",
            "party_size": "people_number",
            "org": "org",
            "dest": "dest",
            "date": "date",
            "visiting_city_number": "visiting_city_number",
        }
        for source, target in aliases.items():
            if constraints.get(source) is not None:
                query_data[target] = copy.deepcopy(constraints[source])
        local = query_data.setdefault("local_constraint", {})
        for source, target in {
            "cuisine": "cuisine",
            "room_type": "room type",
            "room type": "room type",
            "house_rule": "house rule",
            "house rule": "house rule",
            "transportation": "transportation",
        }.items():
            if constraints.get(source) is not None:
                local[target] = copy.deepcopy(constraints[source])
        return query_data

    def _destination_cities(self, observation: Dict[str, Any], query_data: Dict[str, Any]) -> List[str]:
        available = [str(city).strip() for city in observation.get("available_cities") or [] if str(city).strip()]
        constraints = (observation.get("current_intention") or {}).get("constraints") or {}
        required = constraints.get("required_cities") if isinstance(constraints, dict) else None
        if isinstance(required, list) and required:
            cities = [str(city).strip() for city in required if str(city).strip()]
        else:
            destination = str(query_data.get("dest") or query_data.get("destination") or "").strip()
            exact = [city for city in available if city.lower() == destination.lower()]
            searched = []
            for completed in observation.get("completed_actions") or []:
                text = str(completed)
                if not any(text.startswith(f"{action_type}:") for action_type in CITY_TOOL_ACTIONS):
                    continue
                city = text.split(":", 1)[1].strip()
                if city and city not in searched:
                    searched.append(city)
            cities = exact or [*searched, *[city for city in available if city not in searched]]
            if not cities and destination:
                cities = [destination]
        count = max(1, self._safe_int(query_data.get("visiting_city_number"), default=len(cities) or 1))
        return cities[:count]

    def _transport_actions(
        self,
        query_data: Dict[str, Any],
        cities: List[str],
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        origin = str(query_data.get("org") or query_data.get("origin") or "").strip()
        if not origin or not cities:
            return []
        route = [origin, *cities, origin]
        dates = query_data.get("date") or []
        if not isinstance(dates, list):
            dates = [dates] if dates else []
        transportation = str((query_data.get("local_constraint") or {}).get("transportation") or "").lower()
        use_distance = "self" in transportation or "no flight" in transportation or not dates
        mode = "taxi" if "no self" in transportation or "taxi" in transportation else "self-driving"
        actions: List[Tuple[str, Dict[str, Any], str]] = []
        route_dates: List[str] = []
        if dates:
            last_index = len(dates) - 1
            route_dates.append(str(dates[0]))
            for city_index in range(1, len(cities)):
                transition_index = min(last_index, max(0, round(city_index * len(dates) / len(cities))))
                route_dates.append(str(dates[transition_index]))
            route_dates.append(str(dates[-1]))
        for index, (start, end) in enumerate(zip(route, route[1:])):
            if use_distance:
                payload = {"origin": start, "destination": end, "mode": mode}
                argument = f"{start}, {end}, {mode}"
                actions.append(("GoogleDistanceMatrix", payload, argument))
            else:
                date = route_dates[min(index, len(route_dates) - 1)]
                payload = {"origin": start, "destination": end, "date": date}
                argument = f"{start}, {end}, {date}"
                actions.append(("FlightSearch", payload, argument))
        return actions

    def _generate_plan(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.llm_client is None or not hasattr(self.llm_client, "generate_json"):
            raise RuntimeError("TravelPlanner Planner requires an LLM client with generate_json().")
        prompt = self._build_plan_prompt(history, user_utterance, observation)
        raw_plan = self.llm_client.generate_json(prompt)
        plan = self._normalize_plan(raw_plan)
        if plan is None:
            raise ValueError(f"TravelPlanner Planner returned an invalid itinerary: {raw_plan!r}")
        return plan

    def _build_plan_prompt(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        observation: Dict[str, Any],
    ) -> str:
        context = {
            "latest_user_utterance": user_utterance,
            "dialogue_history": self._compact_history(history[-6:]),
            "current_intention": observation.get("current_intention"),
            "query_data": self._effective_query_data(observation),
            "notebook": observation.get("notebook") or [],
        }
        return f"""
You are the original TravelPlanner Planner tool. Use only information collected in Notebook and create one concrete itinerary.
Return one JSON object only with an itinerary list. Every day must contain day, current_city, transportation,
breakfast, lunch, dinner, attraction, and accommodation. Use '-' when a field is unnecessary. Include available
flight numbers, prices, room types, and house rules. Respect the latest current intention.

Entity-level planning:
- current_intention.constraints applies to the whole party.
- current_intention.entities contains opaque entity_N IDs, free-form natural references, and person-specific constraints.
- If any entity has a non-empty constraints object, include participant_assignments in every relevant itinerary day.
- participant_assignments must be an object keyed by the exact stable entity_id. Each value may contain
  transportation, breakfast, lunch, dinner, attraction, accommodation, or another explicitly requested field.
- Shared itinerary fields remain the group default. A participant assignment records only that person's differing
  or explicitly constrained choice; use the same grounded Notebook item when people share an option.
- Do not silently collapse conflicting person-specific constraints into one group preference.
- Never surface an opaque entity_N ID to the user; use the entity's natural reference in prose.

Selection rules:
- Prefer an exact constraint match from Notebook.
- If no exact match exists, select the semantically closest real result from the same search category.
- Never invent a restaurant, accommodation, attraction, route, price, amenity, cuisine, or transportation option.
- A message such as "no valid information" is not a candidate. If a category has no real result at all, use '-'.
- When using a closest match or '-', include a top-level closest_match_substitutions list. Each entry must contain
  constraint, requested_value, selected_value, evidence_category, and reason.

Context:
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}
""".strip()

    def _normalize_plan(self, raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, list):
            return {"itinerary": copy.deepcopy(raw)}
        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("itinerary"), list):
            return copy.deepcopy(raw)
        if isinstance(raw.get("days"), list):
            return {"itinerary": copy.deepcopy(raw["days"]), **{key: value for key, value in raw.items() if key != "days"}}
        if isinstance(raw.get("plan"), list):
            return {"itinerary": copy.deepcopy(raw["plan"])}
        if isinstance(raw.get("plan"), dict):
            return self._normalize_plan(raw["plan"])
        return None

    def _payload_has_plan(self, payload: Dict[str, Any]) -> bool:
        return self._normalize_plan(payload.get("plan")) is not None or isinstance(payload.get("itinerary"), list)

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            return default


__all__ = ["ORIGINAL_ACTION_TYPES", "TravelPlannerExecutor"]
