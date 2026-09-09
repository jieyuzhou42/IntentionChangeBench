from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.single_agent_real_pilot import (
    SingleAgentTravelPlannerExecutor,
    public_travel_observation,
    travel_task_from_annotated_instance,
)


class _UnusedClient:
    pass


def test_travel_prompt_has_one_agent_output_and_no_hidden_state() -> None:
    executor = SingleAgentTravelPlannerExecutor(llm_client=_UnusedClient())
    prompt = executor._build_single_agent_prompt(
        [
            {"role": "user", "content": "Plan a trip to Boston."},
            {"role": "user", "content": "Keep the budget under $1,000."},
        ],
        {
            "latest_user_utterance": "Keep the budget under $1,000.",
            "available_actions": ["FlightSearch", "Planner"],
            "query_data": {"dest": "secret"},
            "available_cities": ["secret"],
            "gold_delta": {"budget": 1000},
            "tool_result": {"flights": []},
        },
    )

    assert "predicted_current_intention" in prompt
    assert "Plan a trip to Boston." in prompt
    assert '"query_data"' not in prompt
    assert '"available_cities"' not in prompt
    assert '"gold_delta"' not in prompt


def test_public_travel_observation_filters_hidden_fields() -> None:
    public = public_travel_observation(
        {
            "page_type": "tool_result",
            "tool_result": {"items": [1]},
            "query_data": {"dest": "Boston"},
            "available_cities": ["Boston"],
            "gold_delta": {"budget": 1000},
            "reference_information": {"secret": True},
        }
    )
    assert public == {
        "page_type": "tool_result",
        "tool_result": {"items": [1]},
    }


def test_travel_task_keeps_only_private_reference_database() -> None:
    task = travel_task_from_annotated_instance(
        {
            "instance_id": "t1",
            "world_state": {
                "travelplanner_query_data": {"dest": "Boston"},
                "reference_information": {"flights": [1]},
                "simulation_only": "secret",
            },
        }
    )
    assert task.world_state == {
        "domain": "travelplanner",
        "reference_information": {"flights": [1]},
    }
