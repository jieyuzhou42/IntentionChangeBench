from __future__ import annotations

import json
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import ShiftOp
from simulators.human_simulator import SEARCH_QUERY_CONTEXT_MARKER, SHIFT_CONTEXT_MARKER, HumanSimulator


class RecordingLLMClient:
    def __init__(self):
        self.json_prompts = []

    def generate_json(self, prompt):
        self.json_prompts.append(prompt)
        return {"gold_search_query": "green running shoes"}

    def generate_text(self, prompt):
        return ""


def _search_query_context(prompt):
    _, payload = prompt.split(SEARCH_QUERY_CONTEXT_MARKER, 1)
    return json.loads(payload)


def _shift_context(prompt):
    _, payload = prompt.split(SHIFT_CONTEXT_MARKER, 1)
    return json.loads(payload)


def test_apply_shift_passes_updated_intention_and_delta_to_query_generator():
    llm_client = RecordingLLMClient()
    simulator = HumanSimulator(llm_client=llm_client)
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
    simulator = HumanSimulator(llm_client=llm_client)

    simulator.generate_gold_search_query_for_intention(
        {
            "constraints": {"category": "running shoes"},
            "priority": ["category"],
        }
    )

    context = _search_query_context(llm_client.json_prompts[-1])
    assert "gold_delta" not in context


def test_shift_prompt_passes_full_gold_intention_timeline():
    simulator = HumanSimulator(llm_client=RecordingLLMClient())
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
