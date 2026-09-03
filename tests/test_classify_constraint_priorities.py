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
