from __future__ import annotations

import json
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domains.travelplanner.environment import TravelPlannerEnvAdapter
from domains.travelplanner.executor import TravelPlannerExecutor
from domains.travelplanner.user_simulator import TravelPlannerUserSimulator
from domains.webshop.user_simulator import WebShopUserSimulator
from simulation.simulation.base_user_simulator import (
    SHIFT_CONTEXT_MARKER,
    ShiftDistributionController,
)
from simulation.simulation.run_simulation import (
    _distribution_controller_from_baseline,
    _travelplanner_initial_intention,
)


class NoopLLM:
    def generate_json(self, _prompt):
        return {}

    def generate_text(self, _prompt):
        return ""


class OneEntityShiftLLM:
    def generate_json(self, _prompt):
        return {
            "intention_changed": True,
            "condition": "user_preference",
            "category": "entity",
            "op": "add",
            "entity_id": "entity_2",
            "reference": "the other traveler",
            "field": "cuisine",
            "old_value": None,
            "value": ["Chinese"],
            "rationale": "The other traveler wants a different cuisine.",
        }

    def generate_text(self, _prompt):
        return "The other traveler would prefer Chinese food."


def _entity_intention():
    return {
        "domain": "travelplanner",
        "constraints": {"people_number": 2, "days": 1, "budget": 1000},
        "priority": ["days", "budget"],
        "entities": {
            "entity_1": {
                "reference": "me",
                "constraints": {"cuisine": ["Italian"]},
            },
            "entity_2": {
                "reference": "my mom",
                "constraints": {"cuisine": ["Chinese"]},
            },
        },
        "entity_priority": ["entities.entity_2.constraints.cuisine"],
    }


def test_travelplanner_initial_intention_creates_stable_party_entities():
    intention = _travelplanner_initial_intention(
        {
            "query": "Plan a trip for two people.",
            "people_number": 2,
            "days": 3,
            "budget": 1200,
            "local_constraint": {},
        }
    )

    assert list(intention["entities"]) == ["entity_1", "entity_2"]
    assert intention["entities"]["entity_1"]["reference"] == "the user"
    assert intention["entities"]["entity_2"]["constraints"] == {}
    assert intention["entity_priority"] == []


def test_travelplanner_initial_intention_recovers_people_number_from_test_query():
    intention = _travelplanner_initial_intention(
        {
            "query": "Please make a five-day itinerary for three travelers.",
            "days": 5,
            "budget": 3000,
            "local_constraint": {},
        }
    )

    assert intention["constraints"]["people_number"] == 3
    assert list(intention["entities"]) == ["entity_1", "entity_2", "entity_3"]

    intention = _travelplanner_initial_intention(
        {
            "query": "We require a travel itinerary for two leaving from Boston.",
            "days": 3,
            "local_constraint": {},
        }
    )
    assert intention["constraints"]["people_number"] == 2


def test_entity_attribute_shift_updates_only_the_target_traveler():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = _entity_intention()
    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "user_preference",
            "category": "entity",
            "op": "override",
            "entity_id": "entity_2",
            "reference": "my mom",
            "field": "cuisine",
            "old_value": ["Chinese"],
            "value": ["Vegetarian"],
            "rationale": "My mom changed her mind.",
        },
        current,
    )

    updated, delta = simulator.apply_shift(current, shift)

    assert shift.field == "entities.entity_2.constraints.cuisine"
    assert updated["entities"]["entity_2"]["constraints"]["cuisine"] == ["Vegetarian"]
    assert updated["entities"]["entity_1"]["constraints"]["cuisine"] == ["Italian"]
    assert updated["constraints"]["people_number"] == 2
    assert delta[shift.field]["category"] == "entity"
    assert delta[shift.field]["op"] == "override"


def test_entity_add_and_remove_synchronize_people_number():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {"people_number": 1},
        "priority": [],
        "entities": {
            "entity_1": {"reference": "me", "constraints": {}}
        },
    }
    add_shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "category": "entity",
            "op": "add",
            "entity_id": "entity_2",
            "reference": "my mom",
            "value": {"constraints": {"mobility": "low walking"}},
            "rationale": "My mom is joining.",
        },
        current,
    )
    with_mom, add_delta = simulator.apply_shift(current, add_shift)

    assert with_mom["constraints"]["people_number"] == 2
    assert with_mom["entities"]["entity_2"]["reference"] == "my mom"
    assert with_mom["entities"]["entity_2"]["constraints"]["mobility"] == "low walking"
    assert add_delta["people_number"]["new"] == 2

    remove_shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "category": "entity",
            "op": "relax",
            "entity_id": "entity_2",
            "reference": "the other traveler",
            "rationale": "My mom can no longer come.",
        },
        with_mom,
    )
    without_mom, remove_delta = simulator.apply_shift(with_mom, remove_shift)

    assert list(without_mom["entities"]) == ["entity_1"]
    assert without_mom["constraints"]["people_number"] == 1
    assert remove_delta["people_number"]["new"] == 1

    env = TravelPlannerEnvAdapter()
    env.query_data = {"people_number": 7, "local_constraint": {}}
    assert env._merged_query_data(without_mom)["people_number"] == 1


def test_entity_constraints_are_evaluated_against_the_correct_assignment():
    env = TravelPlannerEnvAdapter()
    intention = _entity_intention()
    correct_plan = {
        "itinerary": [
            {
                "day": 1,
                "current_city": "Boston",
                "transportation": "-",
                "breakfast": "-",
                "lunch": "-",
                "dinner": "-",
                "attraction": "-",
                "accommodation": "-",
                "participant_assignments": {
                    "entity_1": {"lunch": "Italian lunch at Roma Cafe"},
                    "entity_2": {"lunch": "Chinese lunch at Jade Garden"},
                },
            }
        ]
    }

    result = env._evaluate_plan(correct_plan, intention)

    assert "entities.entity_1.constraints.cuisine" in result["satisfied_constraints"]
    assert "entities.entity_2.constraints.cuisine" in result["satisfied_constraints"]

    incorrect_plan = json.loads(json.dumps(correct_plan))
    incorrect_plan["itinerary"][0]["participant_assignments"]["entity_2"] = {
        "lunch": "Italian lunch at Roma Cafe"
    }
    result = env._evaluate_plan(incorrect_plan, intention)

    assert "entities.entity_2.constraints.cuisine" in result["violated_constraints"]
    assert "entities.entity_1.constraints.cuisine" in result["satisfied_constraints"]


def test_travelplanner_planner_prompt_requires_participant_assignments():
    prompt = TravelPlannerExecutor(llm_client=object())._build_plan_prompt(
        [],
        "My mom wants Chinese food and I want Italian.",
        {
            "current_intention": _entity_intention(),
            "query_data": {},
            "notebook": [],
        },
    )

    assert "participant_assignments" in prompt
    assert "exact stable entity_id" in prompt
    assert "Do not silently collapse" in prompt


def test_travelplanner_shift_prompt_uses_one_entity_category_and_opaque_ids():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    prompt = simulator._build_shift_prompt(_entity_intention())
    instructions, raw_context = prompt.split(SHIFT_CONTEXT_MARKER, 1)
    context = json.loads(raw_context)

    assert 'category="entity"' in instructions
    assert '"category": "add | relax | override | reprioritize | entity"' in instructions
    assert '"changes": [' in instructions
    assert "Shared-party changes and person-specific entity changes may appear together" in instructions
    assert "entity_attribute_override" not in instructions
    assert "entity_plan_diverge" not in instructions
    assert "entities.mom" not in prompt
    assert context["entity_id_guidance"] == {
        "existing": ["entity_1", "entity_2"],
        "next_for_add": "entity_3",
    }


def test_travelplanner_decide_shift_accepts_shared_prompt_guidance_interface():
    simulator = TravelPlannerUserSimulator(OneEntityShiftLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {"people_number": 1},
        "priority": [],
        "entities": {
            "entity_1": {"reference": "the user", "constraints": {}}
        },
    }

    shift = simulator.decide_shift(current)

    assert shift.change_category == "entity"
    assert shift.field == "entities.entity_2.constraints.cuisine"


def test_travelplanner_mixed_multi_shift_applies_shared_and_entity_changes():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = _entity_intention()
    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "user_preference",
            "changes": [
                {
                    "category": "override",
                    "op": "override",
                    "field": "budget",
                    "old_value": 1000,
                    "value": 1400,
                    "rationale": "The group wants more room in the budget.",
                },
                {
                    "category": "entity",
                    "op": "add",
                    "entity_id": "entity_3",
                    "reference": "my sister",
                    "field": "cuisine",
                    "old_value": None,
                    "value": ["Japanese"],
                    "rationale": "My sister is joining and wants Japanese food.",
                },
                {
                    "category": "reprioritize",
                    "op": "reprioritize",
                    "field": "budget",
                    "priority_update": ["days", "budget"],
                    "rationale": "This repeats the existing priority and is a no-op.",
                },
            ],
            "rationale": "My sister is joining, so the trip needs a larger budget and her meals.",
            "utterance_plan": {"style": "explicit", "directness": "direct"},
        },
        current,
    )

    assert shift.op == "multiple"
    assert [change.change_category for change in shift.changes] == [
        "override",
        "entity",
    ]
    updated, delta = simulator.apply_shift(current, shift)
    assert updated["constraints"]["budget"] == 1400
    assert updated["constraints"]["people_number"] == 3
    assert updated["entities"]["entity_3"]["reference"] == "my sister"
    assert updated["entities"]["entity_3"]["constraints"]["cuisine"] == [
        "Japanese"
    ]
    assert set(delta) == {
        "budget",
        "entities.entity_3.constraints.cuisine",
        "people_number",
    }

    realization_prompt = simulator._build_realization_prompt(
        shift, current, "explicit"
    )
    assert "Express every entry in shift.changes" in realization_prompt
    assert "Never expose opaque IDs" in realization_prompt


def test_travelplanner_shift_prompt_uses_deficits_only_as_soft_guidance():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    controller = ShiftDistributionController(
        category_counts={
            "add": 10,
            "relax": 10,
            "override": 10,
            "reprioritize": 10,
            "entity": 0,
        },
        condition_counts={
            "user_preference": 10,
            "real_world_feasibility": 10,
            "agent_misunderstanding": 0,
        },
        categories=["add", "relax", "override", "reprioritize", "entity"],
        conditions=[
            "user_preference",
            "real_world_feasibility",
            "agent_misunderstanding",
        ],
        control_mode="prompt",
    )
    guidance = controller.prompt_guidance()
    prompt = simulator._build_shift_prompt(
        _entity_intention(), distribution_guidance=guidance
    )
    instructions, raw_context = prompt.split(SHIFT_CONTEXT_MARKER, 1)
    context = json.loads(raw_context)

    assert guidance["preferred_change_categories_when_natural"] == ["entity"]
    assert guidance["preferred_conditions_when_natural"] == [
        "agent_misunderstanding"
    ]
    assert context["distribution_guidance"] == guidance
    assert "coherent trajectory above every diversity objective" in instructions
    assert "weak tie-breaker" in instructions


def test_travelplanner_baseline_controller_tracks_entity_and_three_conditions(tmp_path):
    baseline = [
        {
            "turns": [
                {
                    "shift_condition": {
                        "type": "user_preference",
                        "details": {"change_category": "entity", "changes": []},
                    }
                },
                {
                    "shift_condition": {
                        "type": "real_world_feasibility",
                        "details": {"change_category": "relax", "changes": []},
                    }
                },
            ]
        }
    ]
    baseline_path = tmp_path / "travel_baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    controller = _distribution_controller_from_baseline(
        str(baseline_path),
        balance_strength=6.0,
        control_mode="prompt",
        domain="travelplanner",
    )

    assert controller.category_counts == {
        "add": 0,
        "relax": 1,
        "override": 0,
        "reprioritize": 0,
        "entity": 1,
    }
    assert controller.condition_counts == {
        "user_preference": 1,
        "real_world_feasibility": 1,
        "agent_misunderstanding": 0,
    }


def test_webshop_shift_prompt_does_not_receive_entity_level_schema():
    prompt = WebShopUserSimulator(NoopLLM())._build_shift_prompt(
        {"constraints": {"category": "running shoes"}, "priority": ["category"]}
    )
    instructions, context = prompt.split(SHIFT_CONTEXT_MARKER, 1)

    assert "entity_add" not in instructions
    assert "participant_assignments" not in instructions
    assert "entities" not in json.loads(context)["intention_timeline"][-1]["gold_intention"]
