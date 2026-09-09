from __future__ import annotations

import copy

from scripts.extract_travelplanner_constraints import extract_instance


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_json(self, _prompt):
        return copy.deepcopy(self.responses.pop(0))


def test_llm_extraction_updates_state_and_drives_temporal_priority():
    initial = {
        "constraints": {
            "dest": "Boston",
            "start_date": "2026-09-10",
            "budget": 1000,
            # Structured metadata that the user never states must not leak in.
            "visiting_city_number": 1,
        },
        "entities": {
            "entity_1": {"reference": "the user", "constraints": {}},
            "entity_2": {
                "reference": "my partner",
                "constraints": {"cuisine": "any"},
            },
        },
        "entity_priority": [],
        "priority": [],
    }
    instance = {
        "instance_id": "travel_test",
        "turns": [
            {
                "user_utterance": "We are going to Boston September 10 with a $1,000 budget.",
                "gold_current_intention": copy.deepcopy(initial),
                "gold_delta": {},
            },
            {
                "user_utterance": (
                    "September 10 is still essential. Drop the budget cap, and my partner "
                    "needs vegetarian food."
                ),
                "gold_current_intention": copy.deepcopy(initial),
                "gold_delta": {},
            },
        ],
    }
    client = FakeClient(
        [
            {
                "constraints": initial["constraints"],
                "entities": initial["entities"],
                "mentioned_fields": ["dest", "start_date", "budget"],
                "removed_fields": [],
                "rationales": {},
            },
            {
                "constraints": {
                    "dest": "Boston",
                    "start_date": "2026-09-10",
                    "hotel_style": "luxury",
                },
                "entities": {
                    "entity_1": {"reference": "the user", "constraints": {}},
                    "entity_2": {
                        "reference": "my partner",
                        "constraints": {"cuisine": "vegetarian"},
                    },
                },
                "mentioned_fields": [
                    "start_date",
                    "entities.entity_2.constraints.cuisine",
                ],
                "removed_fields": ["budget"],
                "rationales": {
                    "budget": "The user dropped the cap.",
                    "entities.entity_2.constraints.cuisine": "The partner needs vegetarian food.",
                },
            },
        ]
    )

    result = extract_instance(instance, client, retries=0, model_name="fake-model")
    second = result["turns"][1]
    gold = second["gold_current_intention"]

    assert "budget" not in gold["constraints"]
    assert "hotel_style" not in gold["constraints"]
    assert set(result["turns"][0]["gold_current_intention"]["constraints"]) == {"budget"}
    assert "dest" not in gold["constraints"]
    assert "start_date" not in gold["constraints"]
    assert gold["entities"]["entity_2"]["constraints"]["cuisine"] == "vegetarian"
    assert second["gold_delta"]["budget"]["op"] == "remove"
    assert second["constraint_extraction"] == {
        "method": "llm",
        "model": "fake-model",
        "mentioned_fields": ["entities.entity_2.constraints.cuisine"],
        "removed_fields": ["budget"],
    }
    assert gold["priority"]["high"] == ["entities.entity_2.constraints.cuisine"]
    assert gold["priority"]["medium"] == []
