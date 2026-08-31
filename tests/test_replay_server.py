from __future__ import annotations

import json

from src.replay_server import create_app, prepare_state


def _travelplanner_instance():
    return {
        "instance_id": "travelplanner_test_001",
        "world_state": {"domain": "travelplanner"},
        "turns": [
            {
                "turn_id": 0,
                "user_utterance": "Plan a trip to Boston.",
                "gold_current_intention": {
                    "constraints": {"dest": "Boston", "budget": 500},
                    "priority": ["dest", "budget"],
                },
                "agent_action": {
                    "action_type": "Planner",
                    "action_payload": {
                        "plan": {
                            "itinerary": [
                                {
                                    "day": 1,
                                    "current_city": "Boston",
                                    "transportation": "Flight F1",
                                    "breakfast": "Cafe",
                                    "lunch": "Bistro",
                                    "dinner": "Restaurant",
                                    "attraction": "Boston Common",
                                    "accommodation": "Hotel",
                                }
                            ]
                        }
                    },
                },
                "env_feedback": {
                    "search_results": {
                        "attractions": [
                            {
                                "source_action": "AttractionSearch",
                                "query": "Boston",
                                "status": "results",
                                "items": [{"name": "Boston Common", "city": "Boston"}],
                            }
                        ],
                        "accommodations": [],
                        "restaurants": [],
                        "transportation": [],
                        "cities": [],
                    }
                },
                "rollout_trace": [
                    {
                        "step_index": 1,
                        "action": {"action_type": "AttractionSearch"},
                        "tool_result": {"content": [{"name": "Boston Common"}]},
                        "notebook_size": 0,
                    }
                ],
            }
        ],
    }


def test_prepare_state_detects_travelplanner_and_preserves_search_results():
    state = prepare_state([_travelplanner_instance()], image_map={})

    assert state["domain"] == "travelplanner"
    feedback = state["instances"][0]["turns"][0]["env_feedback"]
    assert feedback["search_results"]["attractions"][0]["items"][0]["name"] == "Boston Common"
    assert "candidate_items" not in feedback


def test_prepare_state_keeps_webshop_candidate_image_behavior():
    instances = [
        {
            "instance_id": "webshop_test_001",
            "world_state": {"domain": "webshop"},
            "turns": [
                {
                    "env_feedback": {"candidate_items": [{"asin": "A1", "title": "Chair"}]},
                    "gold_current_intention": {"constraints": {}, "priority": []},
                }
            ],
        }
    ]

    state = prepare_state(instances, image_map={"A1": "https://example.test/a1.jpg"})

    assert state["domain"] == "webshop"
    item = state["instances"][0]["turns"][0]["env_feedback"]["candidate_items"][0]
    assert item["image_url"] == "https://example.test/a1.jpg"


def test_travelplanner_replay_page_renders_and_saves_annotations(tmp_path):
    instances = [_travelplanner_instance()]
    dataset_path = tmp_path / "travelplanner.json"
    dataset_path.write_text(json.dumps(instances), encoding="utf-8")
    state = prepare_state(instances, image_map={})
    app = create_app(state, instances, dataset_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Submitted Itinerary" in page.data
    assert b"Travel Search Evidence" in page.data
    assert b"Tool Action Trace" in page.data

    response = client.post(
        "/api/update_turn",
        json={
            "instance_index": 0,
            "turn_index": 0,
            "user_utterance": "Plan a cheaper Boston trip.",
            "constraints": {"dest": "Boston", "budget": 400},
            "priority": {"high": ["budget"], "medium": ["dest"], "low": []},
        },
    )
    assert response.status_code == 200
    saved = json.loads(dataset_path.read_text(encoding="utf-8"))
    turn = saved[0]["turns"][0]
    assert turn["user_utterance"] == "Plan a cheaper Boston trip."
    assert turn["gold_current_intention"]["constraints"]["budget"] == 400
    assert turn["gold_current_intention"]["priority"]["high"] == ["budget"]
