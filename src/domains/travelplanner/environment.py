from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from envs.base_env import BaseEnv
from models import AgentAction, EnvFeedback


PLAN_FIELDS = (
    "current_city",
    "transportation",
    "breakfast",
    "lunch",
    "dinner",
    "attraction",
    "accommodation",
)

TRAVELPLANNER_ACTIONS = (
    "FlightSearch",
    "AttractionSearch",
    "AccommodationSearch",
    "RestaurantSearch",
    "CitySearch",
    "GoogleDistanceMatrix",
    "NotebookWrite",
    "Planner",
)

SEARCH_ACTIONS = {
    "FlightSearch",
    "AttractionSearch",
    "AccommodationSearch",
    "RestaurantSearch",
    "CitySearch",
    "GoogleDistanceMatrix",
}


class TravelPlannerEnvAdapter(BaseEnv):
    """
    Adapter for TravelPlanner-style planning tasks.

    The adapter works with the reference-information JSON bundled in this repo,
    so it can run even when the full TravelPlanner CSV database has not been
    downloaded. If the executor submits a structured itinerary, the adapter
    checks the main TravelPlanner hard constraints heuristically and returns
    compact feedback for the user simulator.
    """

    def __init__(self):
        self.task = None
        self.query_data: Dict[str, Any] = {}
        self.reference_information: Any = None
        self.last_observation: Dict[str, Any] = {}
        self.last_feedback: Optional[EnvFeedback] = None
        self.active_intention: Dict[str, Any] = {}
        self.notebook: List[Dict[str, Any]] = []
        self.last_tool_result: Any = None
        self.last_tool_action: Optional[Dict[str, Any]] = None
        self.completed_actions: List[str] = []
        self.done = False

    def reset(self, task=None) -> Dict[str, Any]:
        self.task = task
        world_state = copy.deepcopy(getattr(task, "world_state", {}) or {})
        initial_intention = copy.deepcopy(getattr(task, "initial_intention", {}) or {})
        self.query_data = copy.deepcopy(world_state.get("travelplanner_query_data") or {})
        if not self.query_data:
            self.query_data = self._query_data_from_intention(initial_intention)

        self.reference_information = world_state.get("reference_information")
        return self.prepare_turn(initial_intention, self.get_instruction_text(), {})

    def prepare_turn(
        self,
        current_intention: Dict[str, Any],
        user_utterance: str = "",
        gold_delta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Reset the original TravelPlanner notebook/tool state for one user turn."""
        self.active_intention = copy.deepcopy(current_intention or {})
        self.notebook = []
        self.last_tool_result = None
        self.last_tool_action = None
        self.completed_actions = []
        self.done = False
        self.last_feedback = None
        self.last_observation = {
            "domain": "travelplanner",
            "page_type": "planning",
            "feedback_type": "travel_planning",
            "instruction": self.get_instruction_text(),
            "latest_user_utterance": str(user_utterance or ""),
            "query_data": copy.deepcopy(self.query_data),
            "current_intention": copy.deepcopy(self.active_intention),
            "gold_delta": copy.deepcopy(gold_delta or {}),
            "available_actions": list(TRAVELPLANNER_ACTIONS),
            "available_cities": self._available_cities(),
            "notebook": [],
            "completed_actions": [],
        }
        return self.last_observation

    def get_observation(self) -> Dict[str, Any]:
        return self.last_observation

    def get_instruction_text(self) -> str:
        query = self.query_data.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
        intention = getattr(self.task, "initial_intention", {}) if self.task is not None else {}
        request = intention.get("request") if isinstance(intention, dict) else None
        return str(request or "").strip()

    def summarize_current_state(self, user_state: Dict[str, Any]) -> EnvFeedback:
        if self.last_feedback is not None:
            return self.last_feedback
        return EnvFeedback(
            status="observed",
            feasible=True,
            reason="awaiting_plan",
            observation=copy.deepcopy(self.last_observation),
            result={},
            satisfied_constraints=[],
            violated_constraints=[],
        )

    def step(self, agent_action: AgentAction, user_state: Dict[str, Any]) -> EnvFeedback:
        action_type = self._normalize_action_type(agent_action.action_type)
        if action_type in SEARCH_ACTIONS:
            return self._execute_search_action(action_type, agent_action.action_payload or {})
        if action_type == "NotebookWrite":
            return self._execute_notebook_write(agent_action.action_payload or {})
        if action_type not in {"Planner", "submit_plan", "plan", "finish"}:
            feedback = EnvFeedback(
                status="error",
                feasible=False,
                reason=f"unsupported_travelplanner_action: {agent_action.action_type}",
                observation=copy.deepcopy(self.last_observation),
                result={},
                satisfied_constraints=[],
                violated_constraints=self._active_constraint_fields(user_state),
            )
            self.last_feedback = feedback
            return feedback

        plan_payload = agent_action.action_payload or {}
        plan = self._normalize_plan_payload(plan_payload)
        result = self._evaluate_plan(plan, user_state)
        search_results = self._structured_search_feedback()
        observation = {
            "domain": "travelplanner",
            "feedback_type": "travel_search_results",
            "page_type": "search_results",
            "instruction": self.get_instruction_text(),
            "query_data": copy.deepcopy(self.query_data),
            "current_intention": copy.deepcopy(self.active_intention or user_state),
            "available_actions": list(TRAVELPLANNER_ACTIONS),
            "notebook": copy.deepcopy(self.notebook),
            "completed_actions": list(self.completed_actions),
            # TravelPlanner exposes structured search pages to the simulated
            # user. Generated itineraries are never represented as candidates.
            "search_results": copy.deepcopy(search_results),
            "submitted_plan": copy.deepcopy(plan),
            "constraint_debug": copy.deepcopy(result.get("constraint_debug", {})),
        }
        self.last_observation = observation
        self.done = not result.get("violated_constraints")
        feedback = EnvFeedback(
            status="observed",
            feasible=True,
            reason=None if self.done else "constraint_mismatch",
            observation=observation,
            result=result,
            satisfied_constraints=list(result.get("satisfied_constraints") or []),
            violated_constraints=list(result.get("violated_constraints") or []),
        )
        self.last_feedback = feedback
        return feedback

    def _normalize_action_type(self, action_type: Any) -> str:
        text = str(action_type or "").strip()
        aliases = {action.lower(): action for action in TRAVELPLANNER_ACTIONS}
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
        })
        return aliases.get(text.lower(), text)

    def _execute_search_action(self, action_type: str, payload: Dict[str, Any]) -> EnvFeedback:
        argument = self._action_argument(action_type, payload)
        try:
            result = self._search_reference(action_type, payload)
            status = "observed"
            feasible = True
            reason = None
        except (TypeError, ValueError) as exc:
            result = {"error": str(exc)}
            status = "error"
            feasible = False
            reason = str(exc)

        action_key = f"{action_type}:{argument}"
        if action_key not in self.completed_actions:
            self.completed_actions.append(action_key)
        self.last_tool_result = copy.deepcopy(result)
        self.last_tool_action = {
            "action_type": action_type,
            "action_payload": copy.deepcopy(payload),
            "argument": argument,
        }
        observation = {
            "domain": "travelplanner",
            "feedback_type": "travel_tool_result",
            "page_type": "tool_result",
            "instruction": self.get_instruction_text(),
            "latest_user_utterance": self.last_observation.get("latest_user_utterance", ""),
            "query_data": copy.deepcopy(self.query_data),
            "current_intention": copy.deepcopy(self.active_intention),
            "available_actions": list(TRAVELPLANNER_ACTIONS),
            "available_cities": self._available_cities(),
            "tool_name": action_type,
            "tool_argument": argument,
            "tool_result": copy.deepcopy(result),
            "pending_notebook": True,
            "notebook": copy.deepcopy(self.notebook),
            "completed_actions": list(self.completed_actions),
        }
        self.last_observation = observation
        feedback = EnvFeedback(
            status=status,
            feasible=feasible,
            reason=reason,
            observation=observation,
            result={"tool_name": action_type, "argument": argument, "content": copy.deepcopy(result)},
            satisfied_constraints=[],
            violated_constraints=[],
        )
        self.last_feedback = feedback
        return feedback

    def _execute_notebook_write(self, payload: Dict[str, Any]) -> EnvFeedback:
        if self.last_tool_action is None:
            feedback = EnvFeedback(
                status="error",
                feasible=False,
                reason="NotebookWrite requires a preceding TravelPlanner search action.",
                observation=copy.deepcopy(self.last_observation),
                result={},
                satisfied_constraints=[],
                violated_constraints=[],
            )
            self.last_feedback = feedback
            return feedback

        description = str(
            payload.get("description")
            or payload.get("short_description")
            or payload.get("argument")
            or f"{self.last_tool_action['action_type']}[{self.last_tool_action['argument']}]"
        ).strip()
        entry = {
            "index": len(self.notebook),
            "short_description": description,
            "source_action": copy.deepcopy(self.last_tool_action),
            "content": copy.deepcopy(self.last_tool_result),
        }
        self.notebook.append(entry)
        action_key = f"NotebookWrite:{self.last_tool_action['action_type']}:{self.last_tool_action['argument']}"
        if action_key not in self.completed_actions:
            self.completed_actions.append(action_key)
        observation = {
            "domain": "travelplanner",
            "feedback_type": "travel_notebook",
            "page_type": "notebook",
            "instruction": self.get_instruction_text(),
            "latest_user_utterance": self.last_observation.get("latest_user_utterance", ""),
            "query_data": copy.deepcopy(self.query_data),
            "current_intention": copy.deepcopy(self.active_intention),
            "available_actions": list(TRAVELPLANNER_ACTIONS),
            "available_cities": self._available_cities(),
            "pending_notebook": False,
            "notebook": copy.deepcopy(self.notebook),
            "notebook_size": len(self.notebook),
            "completed_actions": list(self.completed_actions),
        }
        self.last_observation = observation
        feedback = EnvFeedback(
            status="observed",
            feasible=True,
            reason=None,
            observation=observation,
            result={"notebook_index": entry["index"], "description": description},
            satisfied_constraints=[],
            violated_constraints=[],
        )
        self.last_feedback = feedback
        return feedback

    def _action_argument(self, action_type: str, payload: Dict[str, Any]) -> str:
        if payload.get("argument") is not None:
            return str(payload["argument"]).strip()
        if action_type == "CitySearch":
            return str(payload.get("state") or payload.get("region") or payload.get("query") or "").strip()
        if action_type in {"AttractionSearch", "AccommodationSearch", "RestaurantSearch"}:
            return str(payload.get("city") or payload.get("query") or "").strip()
        if action_type == "FlightSearch":
            return ", ".join(str(payload.get(key) or "").strip() for key in ("origin", "destination", "date"))
        if action_type == "GoogleDistanceMatrix":
            return ", ".join(str(payload.get(key) or "").strip() for key in ("origin", "destination", "mode"))
        return str(payload.get("query") or "").strip()

    def _payload_parts(self, action_type: str, payload: Dict[str, Any]) -> List[str]:
        argument = str(payload.get("argument") or "")
        parts = [part.strip() for part in argument.split(",")]
        if action_type == "FlightSearch":
            return [
                str(payload.get("origin") or (parts[0] if len(parts) > 0 else "")).strip(),
                str(payload.get("destination") or (parts[1] if len(parts) > 1 else "")).strip(),
                str(payload.get("date") or (parts[2] if len(parts) > 2 else "")).strip(),
            ]
        if action_type == "GoogleDistanceMatrix":
            return [
                str(payload.get("origin") or (parts[0] if len(parts) > 0 else "")).strip(),
                str(payload.get("destination") or (parts[1] if len(parts) > 1 else "")).strip(),
                str(payload.get("mode") or (parts[2] if len(parts) > 2 else "driving")).strip(),
            ]
        return []

    def _search_reference(self, action_type: str, payload: Dict[str, Any]) -> Any:
        if action_type == "CitySearch":
            state = self._action_argument(action_type, payload)
            cities = self._available_cities()
            return {"state": state, "cities": cities}

        if action_type in {"AttractionSearch", "AccommodationSearch", "RestaurantSearch"}:
            city = self._action_argument(action_type, payload)
            if not city:
                raise ValueError(f"{action_type} requires a city.")
            available_cities = self._available_cities()
            if available_cities and city.lower() not in {item.lower() for item in available_cities}:
                raise ValueError(f"Invalid city for this task: {city}")
            prefix = {
                "AttractionSearch": "attraction",
                "AccommodationSearch": "accommodation",
                "RestaurantSearch": "restaurant",
            }[action_type]
            matches = self._matching_reference_entries(prefix, city)
            return self._collapse_reference_matches(matches, f"No {prefix} information for {city}.")

        if action_type == "FlightSearch":
            origin, destination, date = self._payload_parts(action_type, payload)
            if not origin or not destination or not date:
                raise ValueError("FlightSearch requires origin, destination, and date (YYYY-MM-DD).")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                raise ValueError(f"Invalid flight date format: {date}; expected YYYY-MM-DD.")
            matches = self._matching_reference_entries("flight", f"{origin} to {destination}", date)
            return self._collapse_reference_matches(
                matches,
                f"There is no flight from {origin} to {destination} on {date}.",
            )

        if action_type == "GoogleDistanceMatrix":
            origin, destination, mode = self._payload_parts(action_type, payload)
            if not origin or not destination:
                raise ValueError("GoogleDistanceMatrix requires origin, destination, and mode.")
            mode_text = mode.lower()
            if mode_text not in {"driving", "self-driving", "taxi"}:
                raise ValueError("GoogleDistanceMatrix mode must be self-driving/driving or taxi.")
            prefix = "taxi" if "taxi" in mode_text else "self-driving"
            matches = self._matching_reference_entries(prefix, f"{origin} to {destination}")
            return self._collapse_reference_matches(
                matches,
                f"{mode or 'driving'}, from {origin} to {destination}, no valid information.",
            )
        raise ValueError(f"Unsupported TravelPlanner search action: {action_type}")

    def _reference_entries(self) -> List[Tuple[str, Any]]:
        reference = self.reference_information
        entries: List[Tuple[str, Any]] = []
        if isinstance(reference, list):
            for unit in reference:
                if isinstance(unit, dict) and ("Description" in unit or "description" in unit):
                    entries.append((str(unit.get("Description") or unit.get("description")), copy.deepcopy(unit.get("Content") if "Content" in unit else unit.get("content"))))
            return entries
        if not isinstance(reference, dict):
            return entries
        generic_names = {
            "restaurants": "Restaurants",
            "attractions": "Attractions",
            "accommodations": "Accommodations",
            "flights": "Flights",
        }
        for key, value in reference.items():
            description = str(key)
            lowered = description.lower()
            if lowered in generic_names:
                city = self._city_from_items(value)
                description = f"{generic_names[lowered]} in {city}" if city else generic_names[lowered]
            entries.append((description, copy.deepcopy(value)))
        return entries

    def _matching_reference_entries(self, *needles: str) -> List[Tuple[str, Any]]:
        normalized = [str(needle or "").strip().lower() for needle in needles if str(needle or "").strip()]
        matches = []
        for description, content in self._reference_entries():
            haystack = description.lower()
            if all(needle in haystack for needle in normalized):
                matches.append((description, content))
        return matches

    def _collapse_reference_matches(self, matches: List[Tuple[str, Any]], empty_message: str) -> Any:
        if not matches:
            return empty_message
        if len(matches) == 1:
            return {"description": matches[0][0], "content": matches[0][1]}
        return [{"description": description, "content": content} for description, content in matches]

    def _city_from_items(self, value: Any) -> Optional[str]:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    city = item.get("city") or item.get("City")
                    if city:
                        return str(city)
        return None

    def _available_cities(self) -> List[str]:
        cities = set()
        for description, content in self._reference_entries():
            match = re.search(r"(?:Attractions|Restaurants|Accommodations) in (.+)$", description, flags=re.IGNORECASE)
            if match:
                cities.add(match.group(1).strip())
            city = self._city_from_items(content)
            if city:
                cities.add(city)
        return sorted(cities)

    def _query_data_from_intention(self, intention: Dict[str, Any]) -> Dict[str, Any]:
        constraints = intention.get("constraints", {}) if isinstance(intention, dict) else {}
        query_data = {
            "query": intention.get("request", "") if isinstance(intention, dict) else "",
            "budget": constraints.get("budget") or constraints.get("budget_max"),
            "days": constraints.get("days"),
            "people_number": constraints.get("people_number") or constraints.get("party_size"),
            "local_constraint": {
                "cuisine": constraints.get("cuisine"),
                "room type": constraints.get("room_type") or constraints.get("room type"),
                "house rule": constraints.get("house_rule") or constraints.get("house rule"),
                "transportation": constraints.get("transportation"),
            },
        }
        for key in ("org", "dest", "date", "visiting_city_number"):
            if constraints.get(key) is not None:
                query_data[key] = constraints[key]
        return query_data

    def _normalize_plan_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload.get("plan"), dict):
            return copy.deepcopy(payload["plan"])
        if isinstance(payload.get("itinerary"), list):
            return {"itinerary": copy.deepcopy(payload["itinerary"])}
        if isinstance(payload.get("days"), list):
            return {"itinerary": copy.deepcopy(payload["days"])}

        plan_text = payload.get("plan_text") or payload.get("answer") or payload.get("text")
        parsed = self._parse_json_like(plan_text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"itinerary": parsed}
        return {"plan_text": str(plan_text or ""), "itinerary": []}

    def _parse_json_like(self, value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        for candidate in (text, self._extract_json_block(text)):
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _extract_json_block(self, text: str) -> Optional[str]:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
        if not start_positions:
            return None
        start = min(start_positions)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            return None
        return text[start : end + 1]

    def _itinerary_days(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_days = plan.get("itinerary") or plan.get("days") or plan.get("plan") or []
        if isinstance(raw_days, dict):
            raw_days = list(raw_days.values())
        if not isinstance(raw_days, list):
            return []
        days = []
        for unit in raw_days:
            if not isinstance(unit, dict):
                continue
            day = {field: unit.get(field) or unit.get(field.replace("_", " ")) for field in PLAN_FIELDS}
            for key, value in unit.items():
                day.setdefault(key, value)
            days.append(day)
        return days

    def _evaluate_plan(self, plan: Dict[str, Any], user_state: Dict[str, Any]) -> Dict[str, Any]:
        days = self._itinerary_days(plan)
        query_data = self._merged_query_data(user_state)
        checks: Dict[str, Tuple[Optional[bool], Optional[str]]] = {
            "valid_days": self._check_days(query_data, days),
            "valid_cost": self._check_cost(query_data, days),
            "valid_cuisine": self._check_cuisine(query_data, days),
            "valid_room_rule": self._check_room_rule(query_data, days),
            "valid_transportation": self._check_transportation(query_data, days),
            "valid_room_type": self._check_room_type(query_data, days),
        }
        satisfied: List[str] = []
        violated: List[str] = []
        debug: Dict[str, Any] = {}
        for field, (passed, reason) in checks.items():
            debug[field] = {"passed": passed, "reason": reason}
            if passed is True:
                satisfied.append(field)
            elif passed is False:
                violated.append(field)
        estimated_cost = self._estimate_total_cost(days, query_data)
        if "valid_cost" in debug:
            debug["valid_cost"]["estimated_cost"] = estimated_cost

        return {
            "plan": copy.deepcopy(plan),
            "itinerary": copy.deepcopy(days),
            "estimated_cost": estimated_cost,
            "satisfied_constraints": satisfied,
            "violated_constraints": violated,
            "constraint_debug": debug,
        }

    def _merged_query_data(self, user_state: Dict[str, Any]) -> Dict[str, Any]:
        query_data = copy.deepcopy(self.query_data)
        constraints = user_state.get("constraints", {}) if isinstance(user_state, dict) else {}
        if constraints.get("budget") is not None:
            query_data["budget"] = constraints["budget"]
        if constraints.get("budget_max") is not None:
            query_data["budget"] = constraints["budget_max"]
        if constraints.get("days") is not None:
            query_data["days"] = constraints["days"]
        local = query_data.setdefault("local_constraint", {})
        aliases = {
            "cuisine": "cuisine",
            "room_type": "room type",
            "room type": "room type",
            "house_rule": "house rule",
            "house rule": "house rule",
            "transportation": "transportation",
        }
        for source, target in aliases.items():
            if constraints.get(source) is not None:
                local[target] = constraints[source]
        return query_data

    def _check_days(self, query_data: Dict[str, Any], days: List[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[str]]:
        expected = query_data.get("days")
        if expected is None:
            return (None, None)
        try:
            expected_int = int(expected)
        except (TypeError, ValueError):
            return (None, None)
        if len(days) == expected_int:
            return (True, None)
        return (False, f"Expected {expected_int} days but got {len(days)}.")

    def _check_cost(self, query_data: Dict[str, Any], days: List[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[str]]:
        budget = query_data.get("budget")
        if budget is None:
            return (None, None)
        try:
            budget_value = float(budget)
        except (TypeError, ValueError):
            return (None, None)
        cost = self._estimate_total_cost(days, query_data)
        passed = cost <= budget_value
        reason = None if passed else f"Estimated cost {cost:.2f} exceeds budget {budget_value:.2f}."
        return (passed, reason)

    def _check_cuisine(self, query_data: Dict[str, Any], days: List[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[str]]:
        desired = (query_data.get("local_constraint") or {}).get("cuisine")
        desired_values = self._as_list(desired)
        if not desired_values:
            return (None, None)
        plan_text = self._plan_text(days).lower()
        missing = [value for value in desired_values if str(value).lower() not in plan_text]
        if not missing:
            return (True, None)
        return (False, f"Missing cuisine constraint(s): {', '.join(map(str, missing))}.")

    def _check_room_rule(self, query_data: Dict[str, Any], days: List[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[str]]:
        desired = (query_data.get("local_constraint") or {}).get("house rule")
        if not desired:
            return (None, None)
        text = self._plan_text(days).lower()
        desired_text = str(desired).lower()
        if desired_text in text or f"allow {desired_text}" in text or f"{desired_text} allowed" in text:
            return (True, None)
        if f"no {desired_text}" in text:
            return (False, f"Accommodation appears to disallow {desired}.")
        return (None, None)

    def _check_transportation(self, query_data: Dict[str, Any], days: List[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[str]]:
        desired = (query_data.get("local_constraint") or {}).get("transportation")
        if not desired:
            return (None, None)
        text = self._plan_text(days).lower()
        desired_text = str(desired).lower()
        if desired_text.startswith("no "):
            forbidden = desired_text[3:].strip()
            if forbidden and forbidden in text:
                return (False, f"Transportation should not use {forbidden}.")
            return (True, None)
        if desired_text in text:
            return (True, None)
        return (False, f"Transportation should include {desired}.")

    def _check_room_type(self, query_data: Dict[str, Any], days: List[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[str]]:
        desired = (query_data.get("local_constraint") or {}).get("room type")
        if not desired:
            return (None, None)
        text = self._plan_text(days).lower()
        desired_text = str(desired).lower()
        aliases = {
            "entire room": ["entire room", "entire home", "entire home/apt"],
            "private room": ["private room"],
            "shared room": ["shared room"],
            "not shared room": ["private room", "entire room", "entire home", "entire home/apt"],
        }
        accepted = aliases.get(desired_text, [desired_text])
        if any(alias in text for alias in accepted):
            return (True, None)
        return (False, f"Room type should be {desired}.")

    def _estimate_total_cost(self, days: List[Dict[str, Any]], query_data: Dict[str, Any]) -> float:
        people = self._safe_int(query_data.get("people_number"), default=1)
        total = 0.0
        for day in days:
            for field in ("transportation", "breakfast", "lunch", "dinner", "accommodation"):
                total += self._extract_cost(day.get(field)) * max(people, 1)
        return total

    def _extract_cost(self, value: Any) -> float:
        if isinstance(value, dict):
            for key in ("cost", "price", "Average Cost", "Price"):
                if value.get(key) is not None:
                    return self._safe_float(value.get(key), default=0.0)
            value = " ".join(str(v) for v in value.values())
        text = str(value or "")
        patterns = [
            r"(?:cost|price|average cost)\s*[:=]?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
            r"\$\s*([0-9]+(?:\.[0-9]+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._safe_float(match.group(1), default=0.0)
        return 0.0

    def _structured_search_feedback(
        self,
        limit_per_page: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        category_by_action = {
            "CitySearch": "cities",
            "AttractionSearch": "attractions",
            "AccommodationSearch": "accommodations",
            "RestaurantSearch": "restaurants",
            "FlightSearch": "transportation",
            "GoogleDistanceMatrix": "transportation",
        }
        search_results: Dict[str, List[Dict[str, Any]]] = {
            "attractions": [],
            "accommodations": [],
            "restaurants": [],
            "transportation": [],
            "cities": [],
        }
        for entry in self.notebook:
            if not isinstance(entry, dict):
                continue
            source = entry.get("source_action") or {}
            action_type = str(source.get("action_type") or "")
            category = category_by_action.get(action_type)
            if category is None:
                continue
            query = str(source.get("argument") or "")
            sampled = self._sample_search_results(entry.get("content"), limit_per_page)
            real_results = [item for item in sampled if not self._is_no_result(item)]
            no_result_messages = [str(item) for item in sampled if self._is_no_result(item)]
            items: List[Dict[str, Any]] = []
            for result_index, item in enumerate(real_results, start=1):
                result = copy.deepcopy(item) if isinstance(item, dict) else {"value": copy.deepcopy(item)}
                result["result_index"] = result_index
                items.append(result)
            pages = search_results[category]
            pages.append(
                {
                    "page_index": len(pages) + 1,
                    "source_action": action_type,
                    "query": query,
                    "status": "results" if items else "no_results",
                    "sampled_result_count": len(items),
                    "message": no_result_messages[0] if no_result_messages else None,
                    "items": items,
                }
            )
        return search_results

    def _is_no_result(self, value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("error"):
                return True
            if "cities" in value and not value.get("cities"):
                return True
            return False
        text = str(value or "").strip().lower()
        return not text or (text.startswith("no ") and " information" in text) or any(
            marker in text
            for marker in (
                "no valid information",
                "no information",
                "no results",
                "not found",
            )
        )

    def _sample_search_results(self, value: Any, limit: int) -> List[Any]:
        if isinstance(value, dict) and "content" in value:
            return self._sample_search_results(value.get("content"), limit)
        if isinstance(value, list):
            sampled: List[Any] = []
            for item in value:
                if isinstance(item, dict) and "content" in item:
                    sampled.extend(self._sample_search_results(item.get("content"), limit - len(sampled)))
                else:
                    sampled.append(copy.deepcopy(item))
                if len(sampled) >= limit:
                    break
            return sampled[:limit]
        if value is None:
            return []
        return [copy.deepcopy(value)]

    def _active_constraint_fields(self, user_state: Dict[str, Any]) -> List[str]:
        constraints = user_state.get("constraints", {}) if isinstance(user_state, dict) else {}
        return [field for field, value in constraints.items() if value is not None]

    def _plan_text(self, days: List[Dict[str, Any]]) -> str:
        return json.dumps(days, ensure_ascii=False, sort_keys=True)

    def _as_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if item]
        return [value]

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(float(str(value).replace(",", "")))
        except (TypeError, ValueError):
            return default


def load_travelplanner_ref_info(set_type: str) -> List[Dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "TravelPlanner" / "database" / f"{set_type}_ref_info.jsonl"
    if not path.is_file():
        return []
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
