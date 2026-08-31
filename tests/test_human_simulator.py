from __future__ import annotations

import json
import random
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import ShiftOp
from domains.webshop.user_simulator import WebShopUserSimulator
from simulation.simulation.base_user_simulator import (
    SEARCH_QUERY_CONTEXT_MARKER,
    SHIFT_CONTEXT_MARKER,
    ShiftDistributionController,
)


class RecordingLLMClient:
    def __init__(self):
        self.json_prompts = []

    def generate_json(self, prompt):
        self.json_prompts.append(prompt)
        return {"gold_search_query": "green running shoes"}

    def generate_text(self, prompt):
        return ""


class SequenceLLMClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate_json(self, _prompt):
        output = self.outputs[self.calls]
        self.calls += 1
        return output

    def generate_text(self, _prompt):
        return ""


def _search_query_context(prompt):
    _, payload = prompt.split(SEARCH_QUERY_CONTEXT_MARKER, 1)
    return json.loads(payload)


def _shift_context(prompt):
    _, payload = prompt.split(SHIFT_CONTEXT_MARKER, 1)
    return json.loads(payload)


def test_apply_shift_passes_updated_intention_and_delta_to_query_generator():
    llm_client = RecordingLLMClient()
    simulator = WebShopUserSimulator(llm_client=llm_client)
    current_intention = {
        "constraints": {
            "category": "running shoes",
            "color": "blue",
        },
        "priority": ["category", "color"],
        "gold_search_query": "blue running shoes",
    }
    shift = ShiftOp(
        op="override",
        field="color",
        value="green",
        rationale="user now wants green",
    )

    new_state, delta = simulator.apply_shift(current_intention, shift)

    assert new_state["gold_search_query"] == "green running shoes"
    assert delta == {
        "color": {
            "op": "override",
            "old": "blue",
            "new": "green",
            "rationale": "user now wants green",
        }
    }
    context = _search_query_context(llm_client.json_prompts[-1])
    assert context["updated_gold_intention"]["constraints"]["color"] == "green"
    assert context["gold_delta"] == delta


def test_initial_query_generation_omits_gold_delta():
    llm_client = RecordingLLMClient()
    simulator = WebShopUserSimulator(llm_client=llm_client)

    simulator.generate_gold_search_query_for_intention(
        {
            "constraints": {"category": "running shoes"},
            "priority": ["category"],
        }
    )

    context = _search_query_context(llm_client.json_prompts[-1])
    assert "gold_delta" not in context


def test_webshop_parses_and_applies_multiple_constraint_changes_in_one_turn():
    llm_client = RecordingLLMClient()
    simulator = WebShopUserSimulator(llm_client=llm_client)
    current_intention = {
        "constraints": {"category": "running shoes", "color": "blue"},
        "priority": ["category", "color"],
        "gold_search_query": "blue running shoes",
    }

    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "user_preference",
            "changes": [
                {
                    "category": "override",
                    "field": "color",
                    "old_value": "blue",
                    "value": "green",
                    "rationale": "Green is preferred now.",
                },
                {
                    "category": "add",
                    "field": "material",
                    "old_value": None,
                    "value": "mesh",
                    "rationale": "Mesh should be more breathable.",
                },
            ],
            "rationale": "The user changed color and material together.",
        },
        current_intention,
    )
    new_state, delta = simulator.apply_shift(current_intention, shift)

    assert shift.op == "multiple"
    assert [change.field for change in shift.changes] == ["color", "material"]
    assert new_state["constraints"]["color"] == "green"
    assert new_state["constraints"]["material"] == "mesh"
    assert set(delta) == {"color", "material"}
    assert delta["color"]["op"] == "override"
    assert delta["material"]["op"] == "add"


def test_webshop_shift_prompt_requests_a_changes_list():
    simulator = WebShopUserSimulator(llm_client=RecordingLLMClient())
    prompt = simulator._build_shift_prompt(
        {"constraints": {"category": "running shoes"}, "priority": ["category"]}
    )

    assert '"changes": [' in prompt
    assert "change multiple constraints in the same turn" in prompt


def _single_shift(field="color", value="green"):
    return {
        "intention_changed": True,
        "condition": "user_preference",
        "category": "override",
        "field": field,
        "old_value": "blue",
        "value": value,
        "rationale": "A natural single change.",
    }


def _multi_shift(change_count):
    return {
        "intention_changed": True,
        "condition": "user_preference",
        "changes": [
            {
                "category": "add",
                "field": f"preference_{index}",
                "old_value": None,
                "value": f"value_{index}",
                "rationale": f"Natural change {index}.",
            }
            for index in range(change_count)
        ],
        "rationale": "Several related preferences changed together.",
    }


def test_multi_preferred_sampling_keeps_sampling_until_a_multi_candidate_appears():
    llm_client = SequenceLLMClient(
        [_single_shift(), _single_shift("brand", "other"), _multi_shift(3)]
    )
    simulator = WebShopUserSimulator(llm_client=llm_client)

    shift = simulator.decide_shift(
        {"constraints": {"color": "blue"}, "priority": ["color"]},
        candidate_samples=2,
        max_candidate_samples=5,
        prefer_multi=True,
        rng=random.Random(7),
    )

    assert len(shift.changes) == 3
    assert llm_client.calls == 3
    assert shift.sampling_metadata["selection_mode"] == "multi"
    assert shift.sampling_metadata["candidate_samples"] == 3


def test_normal_sampling_prefers_single_candidate_from_the_same_candidate_pool():
    llm_client = SequenceLLMClient([_multi_shift(2), _single_shift(), _multi_shift(4)])
    simulator = WebShopUserSimulator(llm_client=llm_client)

    shift = simulator.decide_shift(
        {"constraints": {"color": "blue"}, "priority": ["color"]},
        candidate_samples=3,
        prefer_multi=False,
        rng=random.Random(7),
    )

    assert shift.field == "color"
    assert not shift.changes
    assert shift.sampling_metadata["selection_mode"] == "single"


def test_multi_sampling_does_not_impose_a_maximum_change_count():
    llm_client = SequenceLLMClient([_multi_shift(5)])
    simulator = WebShopUserSimulator(llm_client=llm_client)

    shift = simulator.decide_shift(
        {"constraints": {}, "priority": []},
        candidate_samples=1,
        max_candidate_samples=1,
        prefer_multi=True,
        rng=random.Random(7),
    )

    assert len(shift.changes) == 5
    assert shift.sampling_metadata["selected_change_count"] == 5


def test_distribution_controller_uses_v1_counts_to_favor_missing_directions():
    controller = ShiftDistributionController(
        category_counts={
            "add": 1270,
            "relax": 450,
            "override": 193,
            "reprioritize": 68,
            "scope_correction": 59,
        },
        condition_counts={
            "user_preference": 1471,
            "real_world_feasibility": 569,
        },
        balance_strength=6.0,
    )
    overrepresented = ShiftOp(
        op="add",
        intention_changed=True,
        condition="user_preference",
        change_category="add",
        field="material",
        value="leather",
    )
    underrepresented = ShiftOp(
        op="scope_correction",
        intention_changed=True,
        condition="real_world_feasibility",
        change_category="scope_correction",
        field="size",
        value="wide",
    )

    selected, metadata = controller.select(
        [overrepresented, underrepresented],
        random.Random(7),
    )

    assert selected is underrepresented
    assert metadata["selected_categories"] == ["scope_correction"]
    assert metadata["selected_condition"] == "real_world_feasibility"


def test_shift_prompt_passes_full_gold_intention_timeline():
    simulator = WebShopUserSimulator(llm_client=RecordingLLMClient())
    current_intention = {
        "constraints": {"category": "running shoes", "color": "green"},
        "priority": ["category", "color"],
        "gold_search_query": "green running shoes",
    }
    intention_history = [
        {
            "turn_id": 7,
            "gold_intention": {
                "constraints": {"category": "running shoes", "color": "blue"},
                "priority": ["color", "category"],
                "gold_search_query": "blue running shoes",
            },
            "gold_delta": {
                "color": {
                    "op": "override",
                    "old": "red",
                    "new": "blue",
                    "rationale": "user wanted blue",
                }
            },
        }
    ]
    current_gold_delta = {
        "color": {
            "op": "override",
            "old": "blue",
            "new": "green",
            "rationale": "user wanted green",
        }
    }

    context = _shift_context(
        simulator._build_shift_prompt(
            current_intention,
            intention_history=intention_history,
            current_gold_delta=current_gold_delta,
        )
    )

    timeline = context["intention_timeline"]
    assert timeline[0]["gold_intention"] == intention_history[0]["gold_intention"]
    assert timeline[0]["gold_delta"] == intention_history[0]["gold_delta"]
    assert timeline[1]["is_current"] is True
    assert timeline[1]["gold_intention"] == current_intention
    assert timeline[1]["gold_delta"] == current_gold_delta
    assert timeline[1]["gold_intention"]["priority"] == ["category", "color"]
