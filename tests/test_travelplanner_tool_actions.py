from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domains.travelplanner.environment import TravelPlannerEnvAdapter
from domains.travelplanner.executor import TravelPlannerExecutor
from models import AgentAction, BaseTask, EnvFeedback
from simulation.simulation import run_simulation
from simulation.simulation.run_simulation import execute_turn


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_json(self, _prompt):
        if not self.responses:
            raise AssertionError("Unexpected extra LLM call")
        return self.responses.pop(0)


def _tool_then_notebook(action_type, payload):
    return [
        {"action_type": action_type, "action_payload": payload},
        {"action_type": "NotebookWrite", "action_payload": {"description": action_type}},
    ]


def _plan_response(days, cities):
    return {
        "itinerary": [
            {
                "day": day,
                "current_city": cities[min(day - 1, len(cities) - 1)],
                "transportation": "-",
                "breakfast": "-",
                "lunch": "-",
                "dinner": "-",
                "attraction": "-",
                "accommodation": "-",
            }
            for day in range(1, days + 1)
        ]
    }


def _smoke_task() -> BaseTask:
    path = Path(__file__).resolve().parents[1] / "data" / "simulation" / "_travelplanner_smoke_task.json"
    raw = json.loads(path.read_text(encoding="utf-8"))[0]
    query_data = raw["travelplanner_query_data"]
    local = query_data["local_constraint"]
    return BaseTask(
        instance_id=raw["instance_id"],
        task_type=raw["task_type"],
        subtype=raw["subtype"],
        world_state={
            "domain": "travelplanner",
            "travelplanner_query_data": query_data,
            "reference_information": raw["reference_information"],
        },
        initial_intention={
            "constraints": {
                "org": query_data["org"],
                "dest": query_data["dest"],
                "days": query_data["days"],
                "people_number": query_data["people_number"],
                "budget": query_data["budget"],
                "visiting_city_number": query_data["visiting_city_number"],
                "cuisine": local["cuisine"],
                "room_type": local["room type"],
                "transportation": local["transportation"],
            },
            "priority": ["dest", "days", "budget"],
            "domain": "travelplanner",
        },
    )


def test_llm_executor_uses_original_travelplanner_actions_before_planner():
    task = _smoke_task()
    env = TravelPlannerEnvAdapter()
    observation = env.reset(task)
    responses = []
    responses += _tool_then_notebook("AttractionSearch", {"city": "Boston"})
    responses += _tool_then_notebook("AccommodationSearch", {"city": "Boston"})
    responses += _tool_then_notebook("RestaurantSearch", {"city": "Boston"})
    responses += _tool_then_notebook(
        "GoogleDistanceMatrix", {"origin": "New York", "destination": "Boston", "mode": "self-driving"}
    )
    responses += _tool_then_notebook(
        "GoogleDistanceMatrix", {"origin": "Boston", "destination": "New York", "mode": "self-driving"}
    )
    responses += [
        {"action_type": "Planner", "action_payload": {}},
        {
            "itinerary": [
                {
                    "day": 1,
                    "current_city": "Boston",
                    "transportation": "Self-driving from New York to Boston",
                    "breakfast": "North End Italian Kitchen, Boston, Italian, cost: 35",
                    "lunch": "North End Italian Kitchen, Boston, Italian, cost: 35",
                    "dinner": "North End Italian Kitchen, Boston, Italian, cost: 35",
                    "attraction": "Boston Common, Boston",
                    "accommodation": "Boston Private Stay, Boston, private room, price: 180",
                }
            ]
        },
    ]
    executor = TravelPlannerExecutor(llm_client=ScriptedLLM(responses))
    utterance = task.world_state["travelplanner_query_data"]["query"]

    rollout = execute_turn(
        env=env,
        execution_agent=executor,
        history=[{"role": "user", "content": utterance}],
        user_utterance=utterance,
        current_intention=task.initial_intention,
        env_observation=observation,
        max_internal_steps=30,
    )

    action_types = [step["action"]["action_type"] for step in rollout.rollout_trace]
    assert action_types[-1] == "Planner"
    assert "submit_plan" not in action_types
    assert "AttractionSearch" in action_types
    assert "AccommodationSearch" in action_types
    assert "RestaurantSearch" in action_types
    assert "GoogleDistanceMatrix" in action_types
    assert action_types.count("NotebookWrite") == 5
    assert rollout.num_search_actions == 5
    assert rollout.stop_reason == "env_done"
    observation = rollout.final_env_feedback.observation
    assert observation["feedback_type"] == "travel_search_results"
    assert observation["page_type"] == "search_results"
    assert "candidate_items" not in observation
    results = observation["search_results"]
    assert results["attractions"][0]["items"][0]["name"] == "Boston Common"
    assert results["accommodations"][0]["items"][0]["name"] == "Boston Private Stay"
    assert results["restaurants"][0]["items"][0]["name"] == "North End Italian Kitchen"
    assert [page["status"] for page in results["transportation"]] == ["no_results", "no_results"]


def test_executor_without_llm_fails_instead_of_falling_back():
    task = _smoke_task()
    env = TravelPlannerEnvAdapter()
    observation = env.reset(task)
    with pytest.raises(RuntimeError, match="requires an LLM client"):
        TravelPlannerExecutor(llm_client=None).act([], "plan a trip", observation)


def test_planner_prompt_requires_grounded_closest_match_selection():
    task = _smoke_task()
    env = TravelPlannerEnvAdapter()
    observation = env.reset(task)
    prompt = TravelPlannerExecutor(llm_client=object())._build_plan_prompt([], "plan a trip", observation)

    assert "semantically closest real result" in prompt
    assert "Never invent" in prompt
    assert "closest_match_substitutions" in prompt
    assert "no valid information" in prompt


def test_search_feedback_flattens_samples_and_keeps_empty_page_metadata():
    env = TravelPlannerEnvAdapter()
    env.notebook = [
        {
            "source_action": {"action_type": "RestaurantSearch", "argument": "Boston"},
            "content": {
                "description": "Restaurants in Boston",
                "content": [{"name": f"Restaurant {index}"} for index in range(1, 13)],
            },
        },
        {
            "source_action": {
                "action_type": "GoogleDistanceMatrix",
                "argument": "New York, Boston, self-driving",
            },
            "content": "self-driving, from New York to Boston, no valid information.",
        },
    ]

    results = env._structured_search_feedback()

    restaurant_items = results["restaurants"][0]["items"]
    assert [item["name"] for item in restaurant_items] == [f"Restaurant {index}" for index in range(1, 11)]
    assert [item["result_index"] for item in restaurant_items] == list(range(1, 11))
    assert results["transportation"][0]["status"] == "no_results"
    assert results["transportation"][0]["sampled_result_count"] == 0


def test_travelplanner_case_retries_three_times_then_succeeds(monkeypatch):
    attempts = 0
    expected = object()

    def flaky_simulation(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise RuntimeError("temporary API failure")
        return expected

    monkeypatch.setattr(run_simulation, "simulate_dialogue_instance", flaky_simulation)
    result = run_simulation._simulate_task_with_retries(
        domain="travelplanner",
        task=_smoke_task(),
        env=object(),
        execution_agent=object(),
        human_simulator=object(),
        max_turns=1,
        max_internal_steps=1,
        seed=7,
    )

    assert result is expected
    assert attempts == 4


def test_infeasible_planner_submission_ends_turn_for_user_inspection():
    class PlannerOnlyExecutor:
        def act(self, *_args):
            return AgentAction("Planner", {"plan": {"itinerary": []}})

    class InfeasiblePlanEnv:
        done = False

        def step(self, _action, _intention):
            return EnvFeedback(
                status="observed",
                feasible=True,
                reason="constraint_mismatch",
                observation={"domain": "travelplanner", "page_type": "search_results", "search_results": {}},
                result={},
                satisfied_constraints=[],
                violated_constraints=["valid_cuisine"],
            )

        def get_observation(self):
            return {"domain": "travelplanner", "page_type": "plan"}

    rollout = execute_turn(
        env=InfeasiblePlanEnv(),
        execution_agent=PlannerOnlyExecutor(),
        history=[],
        user_utterance="plan a trip",
        current_intention={"domain": "travelplanner", "constraints": {"cuisine": ["Seafood"]}},
        env_observation={"domain": "travelplanner"},
        max_internal_steps=30,
    )

    assert rollout.num_internal_steps == 1
    assert rollout.stop_reason == "planner_submitted"
    assert rollout.final_env_feedback.violated_constraints == ["valid_cuisine"]


def test_environment_rejects_planner_shortcut_only_after_supporting_original_actions():
    task = _smoke_task()
    env = TravelPlannerEnvAdapter()
    observation = env.reset(task)

    assert "FlightSearch" in observation["available_actions"]
    assert "NotebookWrite" in observation["available_actions"]
    assert "Planner" in observation["available_actions"]
    assert "reference_information" not in observation


def test_two_city_task_uses_city_and_flight_tools_with_original_action_names():
    query_data = {
        "query": "Visit two cities in Minnesota for five days.",
        "org": "Atlanta",
        "dest": "Minnesota",
        "days": 5,
        "date": ["2022-03-03", "2022-03-04", "2022-03-05", "2022-03-06", "2022-03-07"],
        "people_number": 1,
        "visiting_city_number": 2,
        "budget": 10000,
        "local_constraint": {},
    }
    reference = {}
    for city in ("Duluth", "Minneapolis"):
        reference[f"Attractions in {city}"] = [{"Name": f"{city} Park", "City": city}]
        reference[f"Restaurants in {city}"] = [{"Name": f"{city} Cafe", "City": city, "Average Cost": 10}]
        reference[f"Accommodations in {city}"] = [{"NAME": f"{city} Stay", "city": city, "price": 50}]
    reference.update(
        {
            "Flight from Atlanta to Duluth on 2022-03-03": [{"Flight Number": "F1", "Price": 100}],
            "Flight from Duluth to Minneapolis on 2022-03-05": [{"Flight Number": "F2", "Price": 50}],
            "Flight from Minneapolis to Atlanta on 2022-03-07": [{"Flight Number": "F3", "Price": 100}],
        }
    )
    intention = {
        "constraints": {
            "org": "Atlanta",
            "dest": "Minnesota",
            "days": 5,
            "date": query_data["date"],
            "people_number": 1,
            "visiting_city_number": 2,
            "budget": 10000,
        },
        "priority": ["dest", "days"],
        "domain": "travelplanner",
    }
    task = BaseTask(
        instance_id="two_city",
        task_type="planning",
        subtype="travel",
        world_state={
            "domain": "travelplanner",
            "travelplanner_query_data": query_data,
            "reference_information": reference,
        },
        initial_intention=intention,
    )
    env = TravelPlannerEnvAdapter()
    observation = env.reset(task)
    responses = []
    responses += _tool_then_notebook("CitySearch", {"state": "Minnesota"})
    for city in ("Duluth", "Minneapolis"):
        responses += _tool_then_notebook("AttractionSearch", {"city": city})
        responses += _tool_then_notebook("AccommodationSearch", {"city": city})
        responses += _tool_then_notebook("RestaurantSearch", {"city": city})
    responses += _tool_then_notebook(
        "FlightSearch", {"origin": "Atlanta", "destination": "Duluth", "date": "2022-03-03"}
    )
    responses += _tool_then_notebook(
        "FlightSearch", {"origin": "Duluth", "destination": "Minneapolis", "date": "2022-03-05"}
    )
    responses += _tool_then_notebook(
        "FlightSearch", {"origin": "Minneapolis", "destination": "Atlanta", "date": "2022-03-07"}
    )
    responses += [
        {"action_type": "Planner", "action_payload": {}},
        _plan_response(5, ["Duluth", "Duluth", "Minneapolis", "Minneapolis", "Minneapolis"]),
    ]
    rollout = execute_turn(
        env=env,
        execution_agent=TravelPlannerExecutor(llm_client=ScriptedLLM(responses)),
        history=[{"role": "user", "content": query_data["query"]}],
        user_utterance=query_data["query"],
        current_intention=intention,
        env_observation=observation,
        max_internal_steps=30,
    )

    actions = [step["action"] for step in rollout.rollout_trace]
    action_types = [action["action_type"] for action in actions]
    assert action_types[0] == "CitySearch"
    assert action_types.count("AttractionSearch") == 2
    assert action_types.count("AccommodationSearch") == 2
    assert action_types.count("RestaurantSearch") == 2
    assert action_types.count("FlightSearch") == 3
    assert action_types.count("NotebookWrite") == 10
    assert action_types[-1] == "Planner"
    assert rollout.num_internal_steps == 21
    assert rollout.stop_reason == "env_done"
    flight_dates = [
        action["action_payload"]["date"]
        for action in actions
        if action["action_type"] == "FlightSearch"
    ]
    assert flight_dates == ["2022-03-03", "2022-03-05", "2022-03-07"]
