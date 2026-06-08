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
        self.done = False

    def reset(self, task=None) -> Dict[str, Any]:
        self.task = task
        world_state = copy.deepcopy(getattr(task, "world_state", {}) or {})
        initial_intention = copy.deepcopy(getattr(task, "initial_intention", {}) or {})
        self.query_data = copy.deepcopy(world_state.get("travelplanner_query_data") or {})
        if not self.query_data:
            self.query_data = self._query_data_from_intention(initial_intention)

        self.reference_information = world_state.get("reference_information")
        self.done = False
        self.last_feedback = None
        self.last_observation = {
            "domain": "travelplanner",
            "page_type": "planning",
            "instruction": self.get_instruction_text(),
            "query_data": copy.deepcopy(self.query_data),
            "reference_information": copy.deepcopy(self.reference_information),
            "candidate_items": [],
            "selected_candidate": None,
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
        if agent_action.action_type not in {"submit_plan", "plan", "finish"}:
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
        candidate = self._candidate_from_result(result)
        observation = {
            "domain": "travelplanner",
            "feedback_type": "travel_plan",
            "page_type": "plan",
            "instruction": self.get_instruction_text(),
            "query_data": copy.deepcopy(self.query_data),
            "candidate_items": [candidate],
            "selected_candidate": candidate,
            "submitted_plan": copy.deepcopy(plan),
            "constraint_debug": copy.deepcopy(result.get("constraint_debug", {})),
            "reference_information": copy.deepcopy(self.reference_information),
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

        return {
            "plan": copy.deepcopy(plan),
            "itinerary": copy.deepcopy(days),
            "estimated_cost": debug.get("valid_cost", {}).get("estimated_cost"),
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

    def _candidate_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        itinerary = result.get("itinerary") or []
        return {
            "rank": 1,
            "title": "Submitted travel plan",
            "estimated_cost": result.get("estimated_cost"),
            "num_days": len(itinerary),
            "satisfied_constraints": list(result.get("satisfied_constraints") or []),
            "violated_constraints": list(result.get("violated_constraints") or []),
            "constraint_debug": copy.deepcopy(result.get("constraint_debug", {})),
            "itinerary_preview": copy.deepcopy(itinerary[:3]),
        }

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
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "TravelPlanner" / "database" / f"{set_type}_ref_info.jsonl"
    if not path.is_file():
        return []
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
