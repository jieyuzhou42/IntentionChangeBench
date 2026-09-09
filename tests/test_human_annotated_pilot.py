from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.human_annotated_pilot import (
    aggregate_scored_rows,
    priority_weights,
    score_judged_turn,
    select_webshop_pilot,
)


def test_select_webshop_pilot_takes_two_from_each_sorted_shard(tmp_path: Path) -> None:
    import json

    paths = []
    for shard in range(1, 6):
        path = tmp_path / f"shard_{shard:03d}_human_annotated.json"
        path.write_text(
            json.dumps(
                [
                    {"instance_id": f"{shard}-a", "turns": []},
                    {"instance_id": f"{shard}-b", "turns": []},
                    {"instance_id": f"{shard}-c", "turns": []},
                ]
            ),
            encoding="utf-8",
        )
        paths.append(path)

    selected = select_webshop_pilot(reversed(paths), per_shard=2)

    assert [item["instance_id"] for item in selected] == [
        "1-a",
        "1-b",
        "2-a",
        "2-b",
        "3-a",
        "3-b",
        "4-a",
        "4-b",
        "5-a",
        "5-b",
    ]


def test_priority_weights_support_level_dict_and_ranked_list() -> None:
    assert priority_weights(
        {
            "constraints": {"category": "desk", "budget": 100, "color": "black"},
            "priority": {"high": ["category"], "medium": ["budget"], "low": ["color"]},
        }
    ) == {"category": 3.0, "budget": 2.0, "color": 1.0}
    assert priority_weights(
        {
            "constraints": {"dest": "A", "dates": "B", "budget": 100},
            "priority": ["dest", "dates", "budget"],
        }
    ) == {"dest": 3.0, "dates": 2.0, "budget": 1.0}


def test_score_judged_turn_applies_priority_weights() -> None:
    scores = score_judged_turn(
        gold_intention={
            "constraints": {"category": "desk", "budget": 100},
            "priority": {"high": ["category"], "low": ["budget"]},
        },
        judgment={
            "constraint_judgments": [
                {
                    "gold_field": "category",
                    "recognized": True,
                    "value_match": True,
                    "action_status": "satisfied",
                },
                {
                    "gold_field": "budget",
                    "recognized": True,
                    "value_match": False,
                    "action_status": "violated",
                },
            ],
            "priority_order_score": 1.0,
        },
    )

    assert scores["intention_understanding"]["weighted_constraint_value_accuracy"] == 0.75
    assert scores["intention_understanding"]["combined_score"] == 0.875
    assert scores["action_compliance"]["weighted_constraint_score"] == 0.75
    assert scores["action_compliance"]["hard_priority_violation"] is False


def test_aggregate_scored_rows_summarizes_domains() -> None:
    base_scores = {
        "intention_understanding": {
            "combined_score": 0.5,
            "weighted_constraint_value_accuracy": 0.4,
            "priority_order_score": 0.6,
        },
        "action_compliance": {
            "weighted_constraint_score": 0.7,
            "hard_priority_violation": False,
        },
    }
    aggregate = aggregate_scored_rows(
        [
            {
                "domain": "webshop",
                "instance_id": "w1",
                "turn_id": 0,
                "scores": base_scores,
            },
            {
                "domain": "travelplanner",
                "instance_id": "t1",
                "turn_id": 0,
                "scores": base_scores,
            },
        ]
    )

    assert aggregate["overall"]["instances"] == 2
    assert aggregate["overall"]["turns"] == 2
    assert aggregate["by_domain"]["webshop"]["action_compliance_score"] == 0.7


def test_appropriate_no_match_gets_full_action_credit() -> None:
    scores = score_judged_turn(
        gold_intention={
            "constraints": {"category": "desk", "size": "48 x 24"},
            "priority": {"high": ["category", "size"]},
        },
        judgment={
            "constraint_judgments": [
                {
                    "gold_field": "category",
                    "recognized": True,
                    "value_match": True,
                    "action_status": "violated",
                },
                {
                    "gold_field": "size",
                    "recognized": True,
                    "value_match": True,
                    "action_status": "violated",
                },
            ],
            "priority_order_score": 1.0,
            "no_match_appropriate": True,
        },
    )

    assert scores["action_compliance"]["weighted_constraint_score"] == 1.0
    assert scores["action_compliance"]["no_match_appropriate"] is True
