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


class SequenceShiftLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_json(self, _prompt):
        self.calls += 1
        if not self.responses:
            raise AssertionError("Unexpected extra LLM call")
        return self.responses.pop(0)

    def generate_text(self, _prompt):
        return "Please update the trip."


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


def test_solo_first_person_entity_constraint_is_promoted_to_shared_scope():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {"people_number": 1},
        "priority": [],
        "entities": {"entity_1": {"reference": "me", "constraints": {}}},
    }
    shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "category": "entity",
            "op": "add",
            "entity_id": "entity_1",
            "reference": "me",
            "field": "room_type",
            "value": "Entire home/apt",
            "rationale": "I want an entire apartment.",
        },
        current,
    )

    updated, delta = simulator.apply_shift(current, shift)

    assert shift.change_category == "add"
    assert shift.field == "room_type"
    assert updated["constraints"]["room_type"] == "Entire home/apt"
    assert updated["entities"]["entity_1"]["constraints"] == {}
    assert set(delta) == {"room_type"}


def test_budget_increase_is_normalized_to_relax_and_null_relax_is_deleted():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {"people_number": 1, "budget": 1000, "room_type": "Entire home/apt"},
        "priority": ["budget", "room_type"],
    }
    budget_shift = simulator._parse_shift_output(
        {
            "condition": "real_world_feasibility",
            "category": "override",
            "field": "budget",
            "value": 1400,
            "rationale": "More budget makes the trip feasible.",
        },
        current,
    )
    assert budget_shift.op == "relax"

    room_shift = simulator._parse_shift_output(
        {
            "condition": "real_world_feasibility",
            "category": "relax",
            "field": "room_type",
            "value": None,
            "rationale": "A private room is acceptable.",
        },
        current,
    )
    updated, _ = simulator.apply_shift(current, room_shift)
    assert "room_type" not in updated["constraints"]
    assert "room_type" not in updated["priority"]


def test_shared_travel_values_are_normalized_to_semantic_fields():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {"people_number": 1},
        "priority": [],
    }

    room_shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "category": "add",
            "field": "room_type",
            "value": "Entire home/apt rather than a private room",
            "rationale": "I want more privacy.",
        },
        current,
    )
    dining_shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "category": "add",
            "field": "cuisine",
            "value": "Favor local sit-down restaurants over fast food",
            "rationale": "I want a different dining style.",
        },
        current,
    )
    flight_shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "category": "add",
            "field": "transportation",
            "value": "Take the afternoon flight from Boston to Chicago",
            "rationale": "The afternoon flight suits me better.",
        },
        current,
    )

    assert room_shift.field == "room_type"
    assert room_shift.value == "Entire home/apt"
    assert dining_shift.field == "dining_style"
    assert flight_shift.field == "outbound_transportation"


def test_duration_change_with_fixed_dates_requires_compound_date_update():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {
            "people_number": 1,
            "days": 3,
            "start_date": "2022-03-12",
            "end_date": "2022-03-14",
        },
        "priority": ["start_date", "end_date", "days"],
    }
    shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "changes": [
                {"category": "override", "field": "days", "value": 4},
            ],
            "rationale": "Stay one more day.",
        },
        current,
    )
    assert shift.op == "none"
    assert shift.rationale == "duration_change_requires_date_update"


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
        "relax",
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
            "real_world_feasibility": 0,
        },
        categories=["add", "relax", "override", "reprioritize", "entity"],
        conditions=[
            "user_preference",
            "real_world_feasibility",
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
        "real_world_feasibility"
    ]
    assert context["distribution_guidance"] == guidance
    assert "coherent trajectory above every diversity objective" in instructions
    assert "weak tie-breaker" in instructions


def test_travelplanner_baseline_controller_tracks_entity_and_two_conditions(tmp_path):
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
    }


def test_travelplanner_retries_corrections_until_it_gets_a_real_change():
    llm = SequenceShiftLLM(
        [
            {
                "intention_changed": True,
                "condition": "agent_misunderstanding",
                "changes": [
                    {
                        "category": "override",
                        "op": "override",
                        "field": "activity",
                        "value": "Use the exact listed wording",
                        "rationale": "The validator rejected the wording.",
                    }
                ],
            },
            {
                "intention_changed": True,
                "condition": "user_preference",
                "changes": [
                    {
                        "category": "add",
                        "op": "add",
                        "field": "cuisine",
                        "value": ["Thai"],
                        "rationale": "The latest itinerary made Thai food sound appealing.",
                    }
                ],
            },
        ]
    )
    simulator = TravelPlannerUserSimulator(llm)

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {"people_number": 1, "days": 3},
            "priority": ["days"],
        }
    )

    assert llm.calls == 2
    assert shift.condition == "user_preference"
    assert shift.field == "cuisine"


def test_travelplanner_rejects_a_third_consecutive_change_to_the_same_target():
    llm = SequenceShiftLLM(
        [
            {
                "intention_changed": True,
                "condition": "user_preference",
                "changes": [
                    {
                        "category": "override",
                        "op": "override",
                        "field": "activity",
                        "value": "Museum C",
                        "rationale": "Try one more museum.",
                    }
                ],
            },
            {
                "intention_changed": True,
                "condition": "user_preference",
                "changes": [
                    {
                        "category": "add",
                        "op": "add",
                        "field": "cuisine",
                        "value": ["Mexican"],
                        "rationale": "The dinner options prompted a new preference.",
                    }
                ],
            },
        ]
    )
    simulator = TravelPlannerUserSimulator(llm)
    history = [
        {"turn_id": 1, "shift_targets": ["activity"]},
        {"turn_id": 2, "shift_targets": ["activity"]},
    ]

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {
                "people_number": 1,
                "days": 3,
                "activity": "Museum B",
            },
            "priority": ["days", "activity"],
        },
        intention_history=history,
    )

    assert llm.calls == 2
    assert shift.field == "cuisine"


def test_travelplanner_rejects_a_fourth_consecutive_add_only_turn():
    llm = SequenceShiftLLM(
        [
            {
                "intention_changed": True,
                "condition": "user_preference",
                "changes": [
                    {
                        "category": "add",
                        "op": "add",
                        "field": "restaurant_rating",
                        "value": 4,
                        "rationale": "Add one more preference.",
                    }
                ],
            },
            {
                "intention_changed": True,
                "condition": "user_preference",
                "changes": [
                    {
                        "category": "reprioritize",
                        "op": "reprioritize",
                        "field": "budget",
                        "priority_update": ["budget", "activity", "room_type"],
                        "rationale": "Cost now matters more than the other preferences.",
                    }
                ],
            },
        ]
    )
    simulator = TravelPlannerUserSimulator(llm)
    history = [
        {"turn_id": 1, "change_category": "add", "shift_targets": ["activity"]},
        {"turn_id": 2, "change_category": "add", "shift_targets": ["room_type"]},
        {"turn_id": 3, "change_category": "add", "shift_targets": ["dining"]},
    ]

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {
                "people_number": 1,
                "budget": 1000,
                "activity": "parks",
                "room_type": "Entire home/apt",
            },
            "priority": ["activity", "room_type", "budget"],
        },
        intention_history=history,
    )

    assert llm.calls == 2
    assert shift.op == "reprioritize"


def test_travelplanner_rejects_a_third_change_to_one_dimension_anywhere():
    llm = SequenceShiftLLM(
        [
            {
                "intention_changed": True,
                "condition": "real_world_feasibility",
                "changes": [
                    {
                        "category": "relax",
                        "op": "relax",
                        "field": "budget",
                        "value": 1800,
                        "rationale": "Raise the budget again.",
                    }
                ],
            },
            {
                "intention_changed": True,
                "condition": "user_preference",
                "changes": [
                    {
                        "category": "add",
                        "op": "add",
                        "field": "activity",
                        "value": "parks",
                        "rationale": "The latest itinerary made parks appealing.",
                    }
                ],
            },
        ]
    )
    simulator = TravelPlannerUserSimulator(llm)
    history = [
        {"turn_id": 1, "change_category": "override", "shift_targets": ["budget"]},
        {"turn_id": 2, "change_category": "add", "shift_targets": ["room_type"]},
        {"turn_id": 3, "change_category": "relax", "shift_targets": ["budget"]},
    ]

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {
                "people_number": 1,
                "budget": 1600,
                "room_type": "Entire home/apt",
            },
            "priority": ["budget", "room_type"],
        },
        intention_history=history,
    )

    assert llm.calls == 2
    assert shift.field == "activity"


def test_travelplanner_rejects_the_same_operation_on_last_turns_target():
    llm = SequenceShiftLLM(
        [
            {
                "intention_changed": True,
                "condition": "real_world_feasibility",
                "changes": [
                    {
                        "category": "relax",
                        "op": "relax",
                        "field": "budget",
                        "value": 1800,
                        "rationale": "Raise the budget again.",
                    }
                ],
            },
            {
                "intention_changed": True,
                "condition": "user_preference",
                "changes": [
                    {
                        "category": "add",
                        "op": "add",
                        "field": "restaurant_rating",
                        "value": 3.5,
                        "rationale": "The low-rated meals prompted a new preference.",
                    }
                ],
            },
        ]
    )
    simulator = TravelPlannerUserSimulator(llm)
    history = [
        {
            "turn_id": 1,
            "change_category": "relax",
            "shift_targets": ["budget"],
            "gold_delta": {"budget": {"op": "relax", "old": 800, "new": 950}},
        }
    ]

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {"people_number": 1, "budget": 950},
            "priority": ["budget"],
        },
        intention_history=history,
    )

    assert llm.calls == 2
    assert shift.field == "restaurant_rating"


def test_webshop_shift_prompt_does_not_receive_entity_level_schema():
    prompt = WebShopUserSimulator(NoopLLM())._build_shift_prompt(
        {"constraints": {"category": "running shoes"}, "priority": ["category"]}
    )
    instructions, context = prompt.split(SHIFT_CONTEXT_MARKER, 1)

    assert "entity_add" not in instructions
    assert "participant_assignments" not in instructions
    assert "entities" not in json.loads(context)["intention_timeline"][-1]["gold_intention"]


def _valid_budget_lodging_tradeoff():
    return {
        "intention_changed": True,
        "condition": "real_world_feasibility",
        "tradeoff": {
            "is_tradeoff": True,
            "conflicting_constraints": [
                "accommodation quality",
                "total budget",
            ],
            "chosen_priority": "accommodation quality",
            "concession": "raise the budget",
            "plan_impacts": ["accommodation", "budget"],
            "evidence": "The submitted lodging is cheaper but below the preferred rating.",
        },
        "changes": [
            {
                "category": "add",
                "op": "add",
                "field": "accommodation_rating",
                "old_value": None,
                "value": 4.0,
                "rationale": "Choose better-reviewed lodging.",
            },
            {
                "category": "relax",
                "op": "relax",
                "field": "budget",
                "old_value": 900,
                "value": 1150,
                "rationale": "Make the lodging choice feasible.",
            },
            {
                "category": "reprioritize",
                "op": "reprioritize",
                "field": "accommodation_rating",
                "priority_update": [
                    "accommodation_rating",
                    "days",
                    "budget",
                ],
                "rationale": "Lodging quality now matters more than cost.",
            },
        ],
        "rationale": "Prefer better lodging even if the trip costs more.",
    }


def test_travelplanner_first_trajectory_shift_requires_a_valid_tradeoff():
    llm = SequenceShiftLLM(
        [
            {
                "intention_changed": True,
                "condition": "user_preference",
                "tradeoff": {"is_tradeoff": False},
                "changes": [
                    {
                        "category": "add",
                        "op": "add",
                        "field": "cuisine",
                        "value": ["Thai"],
                        "rationale": "Thai food sounds appealing.",
                    }
                ],
            },
            _valid_budget_lodging_tradeoff(),
        ]
    )
    simulator = TravelPlannerUserSimulator(llm)
    history = [
        {
            "turn_id": 0,
            "gold_delta": {},
            "shift_targets": [],
            "gold_intention": {},
        }
    ]

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {"people_number": 1, "days": 3, "budget": 900},
            "priority": ["days", "budget"],
        },
        intention_history=history,
    )

    assert llm.calls == 2
    assert shift.op == "multiple"
    assert shift.tradeoff["validated"] is True
    assert shift.tradeoff["actual_targets"] == [
        "accommodation_rating",
        "budget",
    ]
    assert [change.op for change in shift.changes] == [
        "add",
        "relax",
        "reprioritize",
    ]


def test_travelplanner_rejects_tradeoff_metadata_without_matching_priority_change():
    fake_tradeoff = _valid_budget_lodging_tradeoff()
    fake_tradeoff["changes"] = fake_tradeoff["changes"][:2]
    llm = SequenceShiftLLM([fake_tradeoff, _valid_budget_lodging_tradeoff()])
    simulator = TravelPlannerUserSimulator(llm)

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {"people_number": 1, "days": 3, "budget": 900},
            "priority": ["days", "budget"],
        },
        intention_history=[{"turn_id": 0, "gold_delta": {}, "shift_targets": []}],
    )

    assert llm.calls == 2
    assert shift.tradeoff["validated"] is True
    assert any(change.op == "reprioritize" for change in shift.changes)


def test_travelplanner_prior_validated_tradeoff_allows_later_natural_single_change():
    llm = SequenceShiftLLM(
        [
            {
                "intention_changed": True,
                "condition": "user_preference",
                "tradeoff": {"is_tradeoff": False},
                "changes": [
                    {
                        "category": "add",
                        "op": "add",
                        "field": "cuisine",
                        "value": ["Thai"],
                        "rationale": "The latest menu made Thai food appealing.",
                    }
                ],
            }
        ]
    )
    simulator = TravelPlannerUserSimulator(llm)
    history = [
        {"turn_id": 0, "gold_delta": {}, "shift_targets": []},
        {
            "turn_id": 1,
            "gold_delta": {"budget": {"op": "relax"}},
            "shift_targets": ["budget", "accommodation_rating"],
            "tradeoff": {
                "is_tradeoff": True,
                "validated": True,
            },
        },
    ]

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {
                "people_number": 1,
                "days": 3,
                "budget": 1150,
                "accommodation_rating": 4.0,
            },
            "priority": ["accommodation_rating", "days", "budget"],
        },
        intention_history=history,
    )

    assert llm.calls == 1
    assert shift.field == "cuisine"
    assert shift.tradeoff["is_tradeoff"] is False


def test_travelplanner_shift_prompt_marks_tradeoff_required_for_new_trajectory():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    prompt = simulator._build_shift_prompt(
        {
            "domain": "travelplanner",
            "constraints": {"people_number": 1, "days": 3, "budget": 900},
            "priority": ["days", "budget"],
        },
        intention_history=[
            {
                "turn_id": 0,
                "gold_delta": {},
                "shift_targets": [],
                "gold_intention": {},
            }
        ],
    )
    instructions, raw_context = prompt.split(SHIFT_CONTEXT_MARKER, 1)
    context = json.loads(raw_context)

    requirement = context["trajectory_tradeoff_requirement"]
    assert requirement["required_now"] is True
    assert requirement["already_satisfied"] is False
    assert "at least two distinct changed fields" in instructions
    assert "arrival time versus first-day sights and meals" in instructions


def test_travelplanner_later_lodging_change_rechecks_active_budget_tradeoff():
    single_lodging_change = {
        "intention_changed": True,
        "condition": "user_preference",
        "tradeoff": {"is_tradeoff": False},
        "changes": [
            {
                "category": "add",
                "op": "add",
                "field": "accommodation_rating",
                "value": 3.0,
                "rationale": "Use a better-rated apartment.",
            }
        ],
    }
    revised_tradeoff = _valid_budget_lodging_tradeoff()
    revised_tradeoff["changes"][0]["value"] = 3.0
    llm = SequenceShiftLLM([single_lodging_change, revised_tradeoff])
    simulator = TravelPlannerUserSimulator(llm)
    history = [
        {"turn_id": 0, "gold_delta": {}, "shift_targets": []},
        {
            "turn_id": 1,
            "gold_delta": {"schedule": {"op": "override"}},
            "shift_targets": ["schedule", "return_transportation"],
            "tradeoff": {"is_tradeoff": True, "validated": True},
        },
    ]

    shift = simulator.decide_shift(
        {
            "domain": "travelplanner",
            "constraints": {
                "people_number": 1,
                "days": 3,
                "budget": 1000,
                "room_type": "Entire home/apt",
                "schedule": "One sightseeing day",
                "return_transportation": "Early flight",
            },
            "priority": ["schedule", "budget", "room_type"],
        },
        intention_history=history,
    )

    assert llm.calls == 2
    assert shift.tradeoff["validated"] is True
    assert {change.field for change in shift.changes} >= {
        "accommodation_rating",
        "budget",
    }


def test_travelplanner_shift_prompt_explicitly_supports_location_and_time_changes():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    prompt = simulator._build_shift_prompt(
        {
            "domain": "travelplanner",
            "constraints": {
                "org": "Boston",
                "dest": "Chicago",
                "days": 3,
                "start_date": "2022-03-12",
                "end_date": "2022-03-14",
                "budget": 1200,
            },
            "priority": ["dest", "start_date", "end_date", "days", "budget"],
        }
    )
    instructions, _ = prompt.split(SHIFT_CONTEXT_MARKER, 1)

    assert "Destination and trip-scope changes are valid intention changes" in instructions
    assert "Origin changes are valid intention changes" in instructions
    assert "Time changes are valid intention changes" in instructions
    assert "changing the destination versus budget" in instructions
    assert "adding or removing cities versus intercity transportation" in instructions
    assert "every obsolete field in the same changes array" in instructions


def test_compound_date_shift_accepts_a_consistent_new_range():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {
            "people_number": 1,
            "days": 3,
            "start_date": "2022-03-12",
            "end_date": "2022-03-14",
        },
        "priority": ["start_date", "end_date", "days"],
    }
    shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "changes": [
                {
                    "category": "override",
                    "op": "override",
                    "field": "start_date",
                    "value": "2022-03-13",
                },
                {
                    "category": "override",
                    "op": "override",
                    "field": "end_date",
                    "value": "2022-03-15",
                },
            ],
            "rationale": "Move the whole trip one day later.",
        },
        current,
    )

    assert shift.op == "multiple"
    updated, _ = simulator.apply_shift(current, shift)
    assert updated["constraints"]["start_date"] == "2022-03-13"
    assert updated["constraints"]["end_date"] == "2022-03-15"
    assert updated["constraints"]["days"] == 3


def test_compound_date_shift_rejects_an_inconsistent_range():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {
            "people_number": 1,
            "days": 3,
            "start_date": "2022-03-12",
            "end_date": "2022-03-14",
        },
        "priority": ["start_date", "end_date", "days"],
    }
    shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "changes": [
                {
                    "category": "override",
                    "op": "override",
                    "field": "start_date",
                    "value": "2022-03-13",
                },
                {
                    "category": "override",
                    "op": "override",
                    "field": "end_date",
                    "value": "2022-03-16",
                },
            ],
            "rationale": "Move the trip later.",
        },
        current,
    )

    assert shift.op == "none"
    assert shift.rationale == "date_change_requires_consistent_range_and_duration"


def test_destination_change_is_a_strongly_coupled_tradeoff():
    simulator = TravelPlannerUserSimulator(NoopLLM())
    current = {
        "domain": "travelplanner",
        "constraints": {
            "dest": "Chicago",
            "days": 3,
            "budget": 1200,
            "outbound_transportation": "Morning flight",
            "activity": "Architecture tour",
        },
        "priority": ["dest", "days", "budget"],
    }
    shift = simulator._parse_shift_output(
        {
            "condition": "user_preference",
            "category": "override",
            "op": "override",
            "field": "dest",
            "value": "Milwaukee",
            "rationale": "Choose a closer city with a shorter journey.",
        },
        current,
    )

    conflict = simulator._active_conflict_requiring_tradeoff(
        shift,
        current,
        [{"turn_id": 0}],
    )
    assert conflict == "destination_scope_vs_budget_transport_schedule"
