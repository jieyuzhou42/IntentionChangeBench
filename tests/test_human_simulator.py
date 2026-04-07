from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import EnvFeedback, ShiftOp
from simulators.human_simulator import HumanSimulator


class DummyLLMClient:
    def generate_json(self, prompt: str):
        raise AssertionError("LLM calls are not expected in this unit test.")

    def generate_text(self, prompt: str) -> str:
        raise AssertionError("LLM calls are not expected in this unit test.")


def _make_user_state() -> dict:
    return {
        "request": "Find a green stripe shirt in large.",
        "constraints": {
            "category": "shirt",
            "color": "green stripe",
            "color_exact": "green stripe",
            "size": "large",
            "size_exact": "large",
            "brand": None,
            "brand_exact": None,
        },
        "priority": ["category", "color", "size", "brand"],
    }


def test_parse_shift_output_supports_add_constraint():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = _make_user_state()
    env_feedback = EnvFeedback(
        status="observed",
        feasible=True,
        reason=None,
        observation={"page_type": "item"},
        result={"brand": "Field & Pine"},
        satisfied_constraints=["category", "color", "size"],
        violated_constraints=[],
    )

    shift = simulator._parse_shift_output(
        {
            "op": "add",
            "field": "brand",
            "value": "Field & Pine",
            "rationale": "user now wants a specific brand",
        },
        user_state,
        env_feedback=env_feedback,
    )

    assert shift.op == "add"
    assert shift.field == "brand"
    assert shift.old_value is None
    assert shift.value == "Field & Pine"


def test_apply_shift_adds_new_constraint_to_state():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = _make_user_state()
    shift = ShiftOp(
        op="add",
        field="brand",
        old_value=None,
        value="Field & Pine",
        rationale="user adds a brand requirement",
    )

    new_state, delta = simulator.apply_shift(user_state, shift)

    assert new_state["constraints"]["brand"] == "Field & Pine"
    assert "brand" in new_state["priority"]
    assert delta["brand"] == {
        "op": "add",
        "old": None,
        "new": "Field & Pine",
        "rationale": "user adds a brand requirement",
    }


def test_fallback_realization_handles_add_constraint():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    utterance = simulator._fallback_realization(
        ShiftOp(op="add", field="brand", value="Field & Pine", rationale=""),
        style="explicit",
    )

    assert utterance == "Please also add brand Field & Pine."
