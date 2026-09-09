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

    priority = instance["turns"][1]["gold_current_intention"]["priority"]
    delta = instance["turns"][1]["gold_delta"]["priority"]
    assert priority["high"] == ["budget"]
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


def test_priority_downgrades_exactly_by_last_mentioned_turn():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "constraints": {"days": 3, "budget": 1000},
                    "priority": ["days", "budget"],
                },
                "gold_delta": {},
            },
            {
                "gold_current_intention": {
                    "constraints": {
                        "days": 3,
                        "budget": 1000,
                        "room_type": "Entire home/apt",
                    },
                    "priority": ["room_type", "days", "budget"],
                },
                "gold_delta": {
                    "room_type": {
                        "op": "add",
                        "old": None,
                        "new": "Entire home/apt",
                    }
                },
            },
            {
                "gold_current_intention": {
                    "constraints": {
                        "days": 3,
                        "budget": 1000,
                        "room_type": "Entire home/apt",
                        "cuisine": ["Thai"],
                    },
                    "priority": ["cuisine", "room_type", "days", "budget"],
                },
                "gold_delta": {
                    "cuisine": {
                        "op": "add",
                        "old": None,
                        "new": ["Thai"],
                    }
                },
            },
        ]
    }

    classify_instance(instance)

    priorities = [
        turn["gold_current_intention"]["priority"]
        for turn in instance["turns"]
    ]
    assert priorities[0] == {
        "high": ["days", "budget"],
        "medium": [],
        "low": [],
    }
    assert priorities[1] == {
        "high": ["room_type"],
        "medium": ["days", "budget"],
        "low": [],
    }
    assert priorities[2] == {
        "high": ["cuisine"],
        "medium": ["room_type"],
        "low": ["days", "budget"],
    }


def test_override_keeps_only_the_new_active_constraint_value():
    instance = {
        "turns": [
            {
                "gold_current_intention": {
                    "constraints": {"budget": 1000},
                    "priority": ["budget"],
                },
                "gold_delta": {},
            },
            {
                "gold_current_intention": {
                    "constraints": {"budget": 1300},
                    "priority": ["budget"],
                },
                "gold_delta": {
                    "budget": {"op": "override", "old": 1000, "new": 1300}
                },
            },
        ]
    }

    classify_instance(instance)

    gold = instance["turns"][1]["gold_current_intention"]
    assert gold["constraints"] == {"budget": 1300}
    assert gold["priority"]["high"] == ["budget"]
