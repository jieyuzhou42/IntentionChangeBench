from __future__ import annotations

from scripts.classify_constraint_priorities import classify_instance


def test_explicit_reprioritize_stays_high_in_classified_state():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "constraints": {"budget": 1000, "days": 3},
                    "priority": ["days", "budget"],
                },
                "gold_delta": {},
            },
            {
                "gold_current_intention": {
                    "constraints": {"budget": 1000, "days": 3},
                    "priority": ["budget", "days"],
                },
                "gold_delta": {
                    "priority": {
                        "op": "reprioritize",
                        "old": ["days", "budget"],
                        "new": ["budget", "days"],
                    }
                },
                "shift_condition": {
                    "details": {
                        "change_category": "reprioritize",
                        "priority_update": ["budget", "days"],
                    }
                },
            },
        ]
    }

    classify_instance(instance)

    assert instance["turns"][0]["gold_current_intention"]["priority"] == {
        "high": ["budget", "days"],
        "medium": [],
        "low": [],
    }
    priority = instance["turns"][1]["gold_current_intention"]["priority"]
    delta = instance["turns"][1]["gold_delta"]["priority"]
    assert priority["high"] == ["budget"]
    assert priority["medium"] == ["days"]
    assert delta["new"]["high"] == ["budget"]


def test_removed_constraint_is_not_retained_in_low_priority():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "constraints": {"days": 3, "room_type": "Entire home/apt"},
                    "priority": ["days", "room_type"],
                },
                "gold_delta": {},
            },
            {
                "gold_current_intention": {
                    "constraints": {"days": 3},
                    "priority": ["days"],
                },
                "gold_delta": {
                    "room_type": {
                        "op": "relax",
                        "old": "Entire home/apt",
                        "new": None,
                    }
                },
            },
        ]
    }

    classify_instance(instance)

    priority = instance["turns"][1]["gold_current_intention"]["priority"]
    assert "room_type" not in priority["high"] + priority["medium"] + priority["low"]


def test_only_current_and_immediately_previous_focus_stay_elevated():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "constraints": {"category": "chair", "budget": 100, "color": "black"},
                    "priority": [],
                },
                "gold_delta": {},
            },
            {
                "gold_current_intention": {
                    "constraints": {"category": "chair", "budget": 120, "color": "black"},
                    "priority": [],
                },
                "gold_delta": {"budget": {"op": "relax", "old": 100, "new": 120}},
            },
            {
                "gold_current_intention": {
                    "constraints": {"category": "chair", "budget": 120, "color": "navy"},
                    "priority": [],
                },
                "gold_delta": {"color": {"op": "override", "old": "black", "new": "navy"}},
            },
            {
                "gold_current_intention": {
                    "constraints": {
                        "category": "office chair",
                        "budget": 120,
                        "color": "navy",
                    },
                    "priority": [],
                },
                "gold_delta": {
                    "category": {"op": "override", "old": "chair", "new": "office chair"}
                },
            },
        ]
    }

    classify_instance(instance)

    priorities = [turn["gold_current_intention"]["priority"] for turn in instance["turns"]]
    assert priorities[0] == {
        "high": ["category", "budget", "color"],
        "medium": [],
        "low": [],
    }
    assert priorities[1] == {
        "high": ["budget"],
        "medium": ["category", "color"],
        "low": [],
    }
    assert priorities[2] == {
        "high": ["color"],
        "medium": ["budget"],
        "low": ["category"],
    }
    assert priorities[3] == {
        "high": ["category"],
        "medium": ["color"],
        "low": ["budget"],
    }


def test_explicit_removal_repairs_stale_constraint_state():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "constraints": {"category": "chair", "color": "black"},
                    "priority": [],
                },
                "gold_delta": {},
            },
            {
                "gold_current_intention": {
                    # Simulate a malformed trajectory that forgot to remove it.
                    "constraints": {"category": "chair", "color": "black"},
                    "priority": [],
                },
                "gold_delta": {"color": {"op": "remove", "old": "black", "new": None}},
            },
        ]
    }

    classify_instance(instance)

    gold = instance["turns"][1]["gold_current_intention"]
    assert "color" not in gold["constraints"]
    assert "color" not in gold["priority"]["high"]
    assert "color" not in gold["priority"]["medium"]
    assert "color" not in gold["priority"]["low"]


def test_priority_only_does_not_mutate_stale_constraint_state():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "constraints": {"category": "chair", "color": "black"},
                    "priority": [],
                },
                "gold_delta": {},
            },
            {
                "gold_current_intention": {
                    "constraints": {"category": "chair", "color": "black"},
                    "priority": [],
                },
                "gold_delta": {"color": {"op": "remove", "old": "black", "new": None}},
            },
        ]
    }

    classify_instance(instance, repair_removed_constraints=False)

    gold = instance["turns"][1]["gold_current_intention"]
    assert gold["constraints"]["color"] == "black"
    assert gold["priority"]["medium"] == ["category"]
    assert gold["priority"]["low"] == ["color"]


def test_authoritative_mentions_ignore_stale_delta_focus():
    instance = {
        "turns": [
            {
                "turn_id": 0,
                "gold_current_intention": {
                    "constraints": {"dest": "Boston", "budget": 1000, "schedule": "flexible"},
                    "priority": [],
                },
                "gold_delta": {},
            },
            {
                "turn_id": 1,
                "gold_current_intention": {
                    "constraints": {"dest": "Boston", "budget": 1200, "schedule": "fixed"},
                    "priority": [],
                },
                "gold_delta": {
                    "budget": {"op": "relax", "old": 1000, "new": 1200},
                    "schedule": {"op": "override", "old": "flexible", "new": "fixed"},
                },
                "constraint_extraction": {
                    "method": "human",
                    "mentioned_fields": ["budget"],
                },
            },
        ]
    }

    classify_instance(instance)

    priority = instance["turns"][1]["gold_current_intention"]["priority"]
    assert priority["high"] == ["budget"]
    assert priority["medium"] == ["dest", "schedule"]


def test_travelplanner_context_fields_can_be_removed_from_constraints():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "domain": "travelplanner",
                    "constraints": {
                        "org": "Tampa",
                        "dest": "Cleveland",
                        "start_date": "2022-03-02",
                        "end_date": "2022-03-04",
                        "visiting_city_number": 1,
                        "days": 3,
                        "people_number": 1,
                        "budget": 1800,
                    },
                    "priority": [],
                },
                "gold_delta": {},
            }
        ]
    }

    classify_instance(instance, drop_travelplanner_context_fields=True)

    gold = instance["turns"][0]["gold_current_intention"]
    assert gold["constraints"] == {"days": 3, "people_number": 1, "budget": 1800}
    assert gold["priority"] == {
        "high": ["days", "people_number", "budget"],
        "medium": [],
        "low": [],
    }
