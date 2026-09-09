from __future__ import annotations

import json

import pytest

from annotation.replay_server import (
    ANNOTATION_DATA_DIR,
    HTML,
    annotation_input_path,
    attach_webshop_gold_items,
    candidate_item_from_catalog,
    create_app,
    collect_webshop_goal_asins,
    default_annotation_path,
    expanded_candidate_items,
    infer_shard_context,
    enrich_webshop_constraints_from_metadata,
    prepare_state,
    set_initial_constraints_must_have,
)


def test_candidate_map_index_cannot_be_interpreted_as_original_gold_flag():
    assert ".map(renderItem)" not in HTML
    assert "renderItem(originalGoldItem, { originalGoldReference: true })" in HTML


def test_constraint_inputs_are_not_draggable_priority_sources():
    assert '<div class="constraint-row" draggable="true"' not in HTML
    assert 'target.closest(".priority-chip")' in HTML
    assert "const key = draggedKey;" in HTML


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
                    "entities": {
                        "entity_1": {"reference": "me", "constraints": {}},
                        "entity_2": {
                            "reference": "my friend",
                            "constraints": {"cuisine": {"avoid": ["Seafood"]}},
                        },
                    },
                },
                "gold_delta": {
                    "budget": {"op": "override", "old": 400, "new": 500},
                    "entities.entity_2.constraints.cuisine": {
                        "op": "add",
                        "category": "entity",
                        "old": None,
                        "new": {"avoid": ["Seafood"]},
                    },
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
                    "env_feedback": {
                        "candidate_items": [
                            {
                                "asin": "A1",
                                "title": "Chair",
                                "options": {"color": ["red", "blue"]},
                            }
                        ]
                    },
                    "gold_current_intention": {"constraints": {}, "priority": []},
                }
            ],
        }
    ]

    state = prepare_state(instances, image_map={"A1": "https://example.test/a1.jpg"})

    assert state["domain"] == "webshop"
    item = state["instances"][0]["turns"][0]["env_feedback"]["candidate_items"][0]
    assert item["image_url"] == "https://example.test/a1.jpg"


def test_webshop_original_goal_item_is_collected_and_attached_with_exact_options():
    instances = [
        {
            "instance_id": "webshop_goal_00000",
            "world_state": {
                "domain": "webshop",
                "webshop_selection_metadata": {
                    "asin": "GOLD1",
                    "name": "Gold Pillow",
                    "query": "decorative pillows",
                    "options": {"size": '28" x 28"'},
                },
            },
            "turns": [{"gold_current_intention": {"constraints": {}, "priority": []}}],
        }
    ]
    state = prepare_state(instances, image_map={})

    assert collect_webshop_goal_asins(instances) == {"GOLD1"}
    assert attach_webshop_gold_items(
        state,
        {"GOLD1": {"asin": "GOLD1", "title": "Catalog Gold Pillow", "options": {"size": ["18 x 18", "28 x 28"]}}},
    ) == 1
    gold_item = state["instances"][0]["webshop_gold_item"]
    assert gold_item["asin"] == "GOLD1"
    assert gold_item["title"] == "Catalog Gold Pillow"
    assert gold_item["gold_selected_options"] == {"size": '28" x 28"'}


def test_expanded_candidates_include_reranked_items_and_next_five_raw_results():
    current = [{"asin": f"A{i}", "title": f"Reranked {i}", "rerank_rank": i} for i in range(1, 11)]
    raw = [{"asin": f"A{i}", "original_rank": i} for i in range(1, 31)]
    catalog = {
        f"A{i}": {"asin": f"A{i}", "title": f"Catalog {i}", "options": {"color": ["blue"]}}
        for i in range(1, 31)
    }
    turn = {"env_feedback": {"candidate_items": current, "rerank_info": {"raw_top_candidates": raw}}}

    expanded = expanded_candidate_items(turn, catalog)

    assert len(expanded) == 15
    assert [item["asin"] for item in expanded[:10]] == [f"A{i}" for i in range(1, 11)]
    assert [item["asin"] for item in expanded[10:]] == [f"A{i}" for i in range(11, 16)]
    assert expanded[0]["title"] == "Reranked 1"
    assert expanded[10]["title"] == "Catalog 11"
    assert expanded[10]["original_rank"] == 11


def test_catalog_product_conversion_keeps_title_image_and_options():
    item = candidate_item_from_catalog(
        {
            "asin": "A1",
            "name": "Blue Chair",
            "pricing": "$29.99",
            "images": ["https://example.test/a1.jpg"],
            "customization_options": {
                "color": [{"value": "blue"}, {"value": "red"}],
                "size": ["small", "large"],
            },
            "small_description": ["Comfortable"],
            "product_information": {"Material": "Cotton", "Dimensions": "20 x 20 in"},
            "average_rating": "4.7 out of 5 stars",
            "availability_status": "In stock",
            "seller_name": "Chair Store",
            "total_reviews": "123",
        }
    )

    assert item["title"] == "Blue Chair"
    assert item["price"] == 29.99
    assert item["image_url"] == "https://example.test/a1.jpg"
    assert item["options"] == {"color": ["blue", "red"], "size": ["small", "large"]}
    assert item["product_information"]["Material"] == "Cotton"
    assert item["average_rating"] == "4.7 out of 5 stars"
    assert item["availability_status"] == "In stock"
    assert item["seller_name"] == "Chair Store"
    assert item["total_reviews"] == "123"


def test_replay_restores_metadata_constraints_and_preserves_human_equivalents():
    instances = [
        {
            "instance_id": "webshop_goal_00000",
            "world_state": {
                "domain": "webshop",
                "webshop_selection_metadata": {
                    "query": "decorative pillows",
                    "price_upper": 30.0,
                    "attributes": ["double sided", "machine washable", "printing technology"],
                    "options": {"size": '28" x 28"'},
                },
            },
            "turns": [
                {
                    "turn_id": 0,
                    "gold_current_intention": {
                        "constraints": {
                            "category": "decorative pillows",
                            "budget_max": 30,
                            "constraint_4": "machine washable",
                        },
                        "priority": {"high": ["category"], "medium": [], "low": ["budget_max"]},
                    },
                    "gold_delta": {},
                },
                {
                    "turn_id": 1,
                    "gold_current_intention": {
                        "constraints": {"category": "decorative pillows", "budget_max": 30},
                        "priority": {"high": ["category"], "medium": [], "low": ["budget_max"]},
                    },
                    "gold_delta": {},
                },
            ],
        }
    ]

    added = enrich_webshop_constraints_from_metadata(instances)

    assert added == 7
    first = instances[0]["turns"][0]["gold_current_intention"]
    second = instances[0]["turns"][1]["gold_current_intention"]
    assert "machine_washable" not in first["constraints"]
    assert first["constraints"]["double_sided"] is True
    assert first["constraints"]["printing_technology"] is True
    assert first["constraints"]["size"] == '28" x 28"'
    assert second["constraints"]["machine_washable"] is True
    assert "printing_technology" in second["priority"]["low"]


def test_webshop_page_shows_options_and_saves_confirmed_gold_product(tmp_path):
    instances = [
        {
            "instance_id": "webshop_test_001",
            "world_state": {"domain": "webshop"},
            "turns": [
                {
                    "turn_id": 0,
                    "user_utterance": "Find a blue chair.",
                    "gold_current_intention": {"constraints": {}, "priority": []},
                    "env_feedback": {
                        "candidate_items": [
                            {
                                "asin": "A1",
                                "title": "Chair",
                                "options": {"color": ["red", "blue"], "size": ["small", "large"]},
                            }
                        ]
                    },
                }
            ],
        }
    ]
    annotation_path = tmp_path / "webshop_annotated.json"
    state = prepare_state(instances, image_map={"A1": "https://example.test/a1.jpg"})
    client = create_app(state, instances, annotation_path).test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Gold Action Confirmation" in page.data
    assert b"Save Trajectory" in page.data
    assert b"inherit forward to later turns" in page.data
    assert b"propagateConstraintChanges" in page.data
    assert b"Select as gold product" in page.data
    assert b"option-summary" in page.data

    response = client.post(
        "/api/update_turn",
        json={
            "instance_index": 0,
            "turn_index": 0,
            "user_utterance": "Find a blue chair.",
            "constraints": {},
            "priority": {},
            "gold_action": {
                "action_type": "Buy",
                "confirmed": True,
                "action_payload": {"selected_asin": "A1", "selected_options": {"color": "blue"}},
            },
        },
    )

    assert response.status_code == 200
    saved_action = json.loads(annotation_path.read_text(encoding="utf-8"))[0]["turns"][0]["gold_action"]
    assert saved_action["confirmed"] is True
    assert saved_action["action_payload"]["selected_asin"] == "A1"
    assert saved_action["action_payload"]["selected_options"] == {"color": "blue"}


def test_travelplanner_replay_page_renders_and_saves_annotations(tmp_path):
    instances = [_travelplanner_instance()]
    source_path = tmp_path / "travelplanner.json"
    annotation_path = tmp_path / "annotations" / "travelplanner_annotated.json"
    source_path.write_text(json.dumps(instances), encoding="utf-8")
    original_source = source_path.read_bytes()
    state = prepare_state(
        instances,
        image_map={},
        source_path=source_path,
        annotation_path=annotation_path,
    )
    app = create_app(state, instances, annotation_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Agent Submitted Itinerary (Reference)" not in page.data
    assert b"day-card" in page.data
    assert b"Travel Search Evidence" in page.data
    assert b"Tool Action Trace" not in page.data
    assert b"Proposed itinerary cost breakdown" in page.data
    assert b"renderTravelCostSummary" in page.data
    assert b"maximum occupancy" in page.data
    assert b"Entity Intentions &amp; Gold Changes" in page.data
    assert b"renderEntityIntentions" in page.data
    assert b"entities.length <= 1" in page.data

    response = client.post(
        "/api/update_turn",
        json={
            "instance_index": 0,
            "turn_index": 0,
            "user_utterance": "Plan a cheaper Boston trip.",
            "constraints": {"dest": "Boston", "budget": 400},
            "priority": {"high": ["budget"], "medium": ["dest"], "low": []},
            "gold_action": {
                "action_type": "Planner",
                "confirmed": True,
                "action_payload": {
                    "plan": {
                        "itinerary": [
                            {
                                "day": "Day 1",
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
        },
    )
    assert response.status_code == 200
    saved = json.loads(annotation_path.read_text(encoding="utf-8"))
    turn = saved[0]["turns"][0]
    assert turn["user_utterance"] == "Plan a cheaper Boston trip."
    assert turn["gold_current_intention"]["constraints"]["budget"] == 400
    assert turn["gold_current_intention"]["priority"]["high"] == ["budget"]
    assert turn["gold_current_intention"]["entities"]["entity_2"]["reference"] == "my friend"
    assert turn["gold_action"]["confirmed"] is True
    assert turn["gold_action"]["action_payload"]["plan"]["itinerary"][0]["accommodation"] == "Hotel"
    assert source_path.read_bytes() == original_source


def test_turns_can_be_added_deleted_and_renumbered(tmp_path):
    instances = [_travelplanner_instance()]
    annotation_path = tmp_path / "travelplanner_annotated.json"
    state = prepare_state(instances, image_map={})
    client = create_app(state, instances, annotation_path).test_client()

    added = client.post("/api/turns", json={"instance_index": 0, "after_turn_index": 0})
    assert added.status_code == 200
    assert added.get_json()["turn_index"] == 1
    saved = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert [turn["turn_id"] for turn in saved[0]["turns"]] == [0, 1]
    assert saved[0]["turns"][1]["user_utterance"] == ""
    assert saved[0]["turns"][1]["gold_current_intention"]["constraints"]["dest"] == "Boston"
    assert saved[0]["turns"][1]["env_feedback"] == {}

    deleted = client.delete("/api/turns/0/0")
    assert deleted.status_code == 200
    saved = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert len(saved[0]["turns"]) == 1
    assert saved[0]["turns"][0]["turn_id"] == 0

    rejected = client.delete("/api/turns/0/0")
    assert rejected.status_code == 400
    assert b"at least one turn" in rejected.data


def test_confirmed_gold_action_requires_a_concrete_selection(tmp_path):
    instances = [_travelplanner_instance()]
    annotation_path = tmp_path / "travelplanner_annotated.json"
    state = prepare_state(instances, image_map={})
    client = create_app(state, instances, annotation_path).test_client()

    response = client.post(
        "/api/update_turn",
        json={
            "instance_index": 0,
            "turn_index": 0,
            "user_utterance": "Plan a trip.",
            "constraints": {},
            "priority": {},
            "gold_action": {
                "action_type": "Planner",
                "confirmed": True,
                "action_payload": {"plan": {"itinerary": []}},
            },
        },
    )

    assert response.status_code == 400
    assert b"at least one day" in response.data


def test_trajectory_save_persists_all_turns_without_replay_only_fields(tmp_path):
    instances = [
        {
            "instance_id": "webshop_test_trajectory",
            "world_state": {"domain": "webshop"},
            "turns": [
                {
                    "turn_id": 0,
                    "user_utterance": "Original",
                    "gold_current_intention": {"constraints": {}, "priority": []},
                    "env_feedback": {"candidate_items": [{"asin": "A1", "options": {}}]},
                }
            ],
        }
    ]
    annotation_path = tmp_path / "trajectory.json"
    state = prepare_state(instances, image_map={"A1": "https://example.test/a1.jpg"})
    client = create_app(state, instances, annotation_path).test_client()
    submitted_turns = state["instances"][0]["turns"] + [
        {
            "turn_id": 1,
            "user_utterance": "Added in trajectory draft",
            "gold_current_intention": {
                "constraints": {"color": "blue"},
                "priority": {"high": ["color"], "medium": [], "low": []},
            },
            "gold_delta": {},
            "gold_action": {},
            "agent_action": {},
            "env_feedback": {},
            "rollout_trace": [],
            "rationales": ["display only"],
        }
    ]
    submitted_turns[0]["user_utterance"] = "Edited before trajectory save"

    response = client.post(
        "/api/update_trajectory",
        json={"instance_index": 0, "turns": submitted_turns},
    )

    assert response.status_code == 200
    assert response.get_json()["turn_count"] == 2
    saved_turns = json.loads(annotation_path.read_text(encoding="utf-8"))[0]["turns"]
    assert [turn["turn_id"] for turn in saved_turns] == [0, 1]
    assert saved_turns[0]["user_utterance"] == "Edited before trajectory save"
    assert "image_url" not in saved_turns[0]["env_feedback"]["candidate_items"][0]
    assert "rationales" not in saved_turns[0]
    assert "rationales" not in saved_turns[1]


def test_delete_trajectory_persists_whole_instance_removal_and_keeps_one(tmp_path):
    instances = [
        {
            "instance_id": "delete_me",
            "world_state": {"domain": "webshop"},
            "turns": [{"turn_id": 0, "gold_current_intention": {"constraints": {}, "priority": []}}],
        },
        {
            "instance_id": "keep_me",
            "world_state": {"domain": "webshop"},
            "turns": [{"turn_id": 0, "gold_current_intention": {"constraints": {}, "priority": []}}],
        },
    ]
    annotation_path = tmp_path / "trajectory.json"
    state = prepare_state(instances, image_map={})
    client = create_app(state, instances, annotation_path).test_client()

    response = client.delete("/api/trajectories/0")

    assert response.status_code == 200
    assert response.get_json()["deleted_instance_id"] == "delete_me"
    assert [item["instance_id"] for item in instances] == ["keep_me"]
    assert [item["instance_id"] for item in state["instances"]] == ["keep_me"]
    saved = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert [item["instance_id"] for item in saved] == ["keep_me"]

    response = client.delete("/api/trajectories/0")
    assert response.status_code == 400
    assert b"keep at least one trajectory" in response.data


def test_annotation_input_resumes_output_and_rejects_source_overwrite(tmp_path):
    source_path = tmp_path / "rollout.json"
    annotation_path = tmp_path / "rollout_annotated.json"
    source_path.write_text("[]", encoding="utf-8")

    assert annotation_input_path(source_path, annotation_path) == source_path
    annotation_path.write_text("[]", encoding="utf-8")
    assert annotation_input_path(source_path, annotation_path) == annotation_path
    with pytest.raises(ValueError, match="must differ"):
        annotation_input_path(source_path, source_path)


def test_default_annotation_path_uses_annotation_data_directory(tmp_path):
    source_path = tmp_path / "rollout.json"

    assert default_annotation_path(source_path) == ANNOTATION_DATA_DIR / "rollout_annotated.json"


def test_infer_shard_context_reads_number_and_manifest(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "manifest.json").write_text('{"shard_count": 35}', encoding="utf-8")

    assert infer_shard_context(shard_dir / "shard_002.json") == (2, 35)
    assert infer_shard_context(shard_dir / "dataset.json") == (None, None)


def test_initial_turn_constraints_are_all_must_have_without_changing_later_turns():
    instances = [
        {
            "turns": [
                {
                    "gold_current_intention": {
                        "constraints": {"category": "chair", "budget": 50, "color": "blue"},
                        "priority": {"high": ["category"], "medium": ["color"], "low": ["budget"]},
                    }
                },
                {
                    "gold_current_intention": {
                        "constraints": {"category": "chair", "budget": 50, "color": "blue"},
                        "priority": {"high": ["category"], "medium": ["color"], "low": ["budget"]},
                    }
                },
            ]
        }
    ]

    assert set_initial_constraints_must_have(instances) == 1
    assert instances[0]["turns"][0]["gold_current_intention"]["priority"] == {
        "high": ["category", "budget", "color"],
        "medium": [],
        "low": [],
    }
    assert instances[0]["turns"][1]["gold_current_intention"]["priority"]["low"] == ["budget"]


def test_switch_shard_endpoint_launches_delayed_background_switch(tmp_path, monkeypatch):
    instances = [
        {
            "instance_id": "id_0",
            "world_state": {"domain": "webshop"},
            "turns": [{"turn_id": 0, "gold_current_intention": {"constraints": {}, "priority": []}}],
        }
    ]
    state = prepare_state(instances, image_map={})
    state.update({"shard_index": 2, "shard_count": 35})
    launched = {}

    def fake_popen(arguments, **kwargs):
        launched["arguments"] = arguments
        launched["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("annotation.replay_server.subprocess.Popen", fake_popen)
    client = create_app(state, instances, tmp_path / "annotation.json").test_client()

    response = client.post("/api/switch_shard", json={"shard": 3})

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "shard": 3}
    assert "start_webshop_shard.ps1" in " ".join(launched["arguments"])
    assert launched["arguments"][-5:] == ["-Shard", "3", "-DelayMilliseconds", "1500", "-NoBrowser"]

    response = client.post("/api/switch_shard", json={"shard": 36})
    assert response.status_code == 400
