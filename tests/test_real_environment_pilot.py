from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.real_environment_pilot import (
    build_blind_intention_prompt,
    normalize_intention_prediction,
    travel_task_from_instance,
)


def test_blind_intention_prompt_contains_no_gold_payload() -> None:
    prompt = build_blind_intention_prompt(
        domain="travelplanner",
        user_utterances=["Plan a trip.", "Raise the budget to $2,000."],
    )
    assert "gold_current_intention" not in prompt
    assert "Raise the budget to $2,000." in prompt


def test_normalize_intention_prediction_uses_ranked_priority() -> None:
    normalized = normalize_intention_prediction(
        {
            "constraints": {"budget": 2000, "dest": "Boston"},
            "priority": {"ranked_fields": ["dest", "budget"]},
            "explanation": "Boston matters most.",
        },
        domain="travelplanner",
    )
    assert normalized["priority"] == ["dest", "budget"]
    assert normalized["constraints"]["budget"] == 2000


def test_travel_task_does_not_copy_gold_intention() -> None:
    task = travel_task_from_instance(
        {
            "instance_id": "t1",
            "world_state": {
                "domain": "travelplanner",
                "travelplanner_query_data": {"query": "Plan a trip.", "dest": "Boston"},
            },
            "turns": [
                {
                    "gold_current_intention": {
                        "constraints": {"secret_gold_field": "must not leak"}
                    }
                }
            ],
        }
    )
    assert task.initial_intention["constraints"] == {}
    assert "secret_gold_field" not in str(task.world_state)
