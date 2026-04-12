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


class SequencedLLMClient:
    def __init__(self, json_outputs):
        self.json_outputs = list(json_outputs)
        self.prompts = []

    def generate_json(self, prompt: str):
        self.prompts.append(prompt)
        if not self.json_outputs:
            raise AssertionError("No more queued JSON outputs.")
        return self.json_outputs.pop(0)

    def generate_text(self, prompt: str) -> str:
        raise AssertionError("Text calls are not expected in this unit test.")


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


def test_parse_shift_output_supports_user_preference_override_when_result_is_acceptable():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = _make_user_state()
    env_feedback = EnvFeedback(
        status="observed",
        feasible=True,
        reason="candidate_looks_good",
        observation={"page_type": "item"},
        result={"category": "shirt", "color": "green stripe", "size": "large"},
        satisfied_constraints=["category", "color", "size"],
        violated_constraints=[],
    )

    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "user_preference",
            "change_category": "override",
            "op": "override",
            "field": "color",
            "old_value": "green stripe",
            "value": "navy",
            "rationale": "After seeing the stripe pattern, the user now prefers a more solid color.",
        },
        user_state,
        env_feedback=env_feedback,
    )

    assert shift.intention_changed is True
    assert shift.condition == "user_preference"
    assert shift.change_category == "override"
    assert shift.op == "override"
    assert shift.field == "color"
    assert shift.old_value == "green stripe"
    assert shift.value == "navy"


def test_parse_shift_output_supports_user_preference_reprioritize_from_shown_candidate():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = {
        "request": "Find a green shirt under 40 dollars.",
        "constraints": {
            "category": "shirt",
            "color": "green",
            "budget_max": 40.0,
        },
        "priority": ["color", "budget_max", "category"],
    }
    env_feedback = EnvFeedback(
        status="observed",
        feasible=True,
        reason="candidate_is_close",
        observation={"page_type": "item"},
        result={"category": "shirt", "color": "green", "price": 34.99},
        satisfied_constraints=["category", "color", "budget_max"],
        violated_constraints=[],
    )

    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "user_preference",
            "change_category": "reprioritize",
            "op": "reprioritize",
            "field": "budget_max",
            "priority_update": ["budget_max", "color", "category"],
            "rationale": "Seeing this candidate makes the user care more about price than the exact color.",
        },
        user_state,
        env_feedback=env_feedback,
    )

    assert shift.intention_changed is True
    assert shift.condition == "user_preference"
    assert shift.change_category == "reprioritize"
    assert shift.op == "reprioritize"
    assert shift.priority_update == ["budget_max", "color", "category"]
    assert shift.value == ["budget_max", "color", "category"]


def test_parse_shift_output_supports_real_world_feasibility_relax_when_exact_option_is_unavailable():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = _make_user_state()
    env_feedback = EnvFeedback(
        status="observed",
        feasible=False,
        reason="no_exact_match_visible",
        observation={"page_type": "results"},
        result={"category": "shirt"},
        satisfied_constraints=["category", "size"],
        violated_constraints=["color_exact"],
    )

    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "real_world_feasibility",
            "change_category": "relax",
            "op": "relax",
            "field": "color_exact",
            "old_value": "green stripe",
            "value": None,
            "rationale": "The exact pattern does not seem available, so a close color match is enough.",
        },
        user_state,
        env_feedback=env_feedback,
    )

    assert shift.intention_changed is True
    assert shift.condition == "real_world_feasibility"
    assert shift.change_category == "relax"
    assert shift.op == "relax"
    assert shift.field == "color_exact"
    assert shift.old_value == "green stripe"
    assert shift.value is None


def test_scope_entity_correction_preserves_refinement_pattern():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = {
        "request": "Find men's loafers.",
        "constraints": {
            "category": "shoes",
            "category_exact": "men's loafers",
            "color": "black",
        },
        "priority": ["category_exact", "color", "category"],
    }
    env_feedback = EnvFeedback(
        status="observed",
        feasible=True,
        reason="agent_chose_broader_scope",
        observation={"page_type": "item"},
        result={"category": "shoes", "product_category": "men's loafers & slip-ons"},
        satisfied_constraints=["category"],
        violated_constraints=["category_exact"],
    )

    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "agent_misunderstanding",
            "change_category": "scope(entity) correction",
            "op": "scope(entity) correction",
            "field": "category_exact",
            "old_value": "men's loafers",
            "value": "men's loafers & slip-ons",
            "rationale": "The user is refining the intended entity, not changing away from loafers.",
        },
        user_state,
        env_feedback=env_feedback,
    )
    new_state, delta = simulator.apply_shift(user_state, shift)

    assert shift.condition == "agent_misunderstanding"
    assert shift.change_category == "scope_correction"
    assert shift.op == "scope_correction"
    assert new_state["constraints"]["category"] == "shoes"
    assert new_state["constraints"]["category_exact"] == "men's loafers & slip-ons"
    assert delta["category_exact"]["op"] == "scope_correction"


def test_parse_shift_output_can_emit_no_change():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = _make_user_state()

    shift = simulator._parse_shift_output(
        {
            "intention_changed": False,
            "condition": "none",
            "change_category": "none",
            "op": "none",
            "rationale": "The current interaction does not justify changing the request.",
        },
        user_state,
    )

    assert shift.op == "none"
    assert shift.intention_changed is False
    assert shift.condition == "none"
    assert shift.change_category == "none"


def test_parse_shift_output_turns_unknown_field_into_addition():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = _make_user_state()

    shift = simulator._parse_shift_output(
        {
            "intention_changed": True,
            "condition": "user_preference",
            "change_category": "override",
            "op": "override",
            "field": "material",
            "old_value": None,
            "value": "linen",
            "rationale": "I suddenly care about fabric more and want linen now.",
        },
        user_state,
    )

    assert shift.intention_changed is True
    assert shift.condition == "user_preference"
    assert shift.change_category == "add"
    assert shift.op == "add"
    assert shift.field == "material"
    assert shift.old_value is None
    assert shift.value == "linen"


def test_apply_shift_adds_new_constraint_to_state():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    user_state = _make_user_state()
    shift = ShiftOp(
        op="add",
        intention_changed=True,
        condition="user_preference",
        change_category="add",
        field="brand",
        old_value=None,
        value="Field & Pine",
        rationale="The user adds a brand preference after seeing the item.",
    )

    new_state, delta = simulator.apply_shift(user_state, shift)

    assert new_state["constraints"]["brand"] == "Field & Pine"
    assert "brand" in new_state["priority"]
    assert delta["brand"] == {
        "op": "add",
        "old": None,
        "new": "Field & Pine",
        "rationale": "The user adds a brand preference after seeing the item.",
    }


def test_fallback_realization_handles_add_constraint_concisely():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    utterance = simulator._fallback_realization(
        ShiftOp(
            op="add",
            intention_changed=True,
            condition="user_preference",
            change_category="add",
            field="brand",
            value="Field & Pine",
            rationale="",
        ),
        style="explicit",
    )

    assert utterance == "Also, I'd prefer brand Field & Pine."


def test_shift_prompt_does_not_introduce_invalid_top_level_reaction_classes():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    prompt = simulator._build_shift_prompt(_make_user_state())

    assert '"condition"' in prompt
    assert '"category"' in prompt
    assert "Feel free to change your primary goal or constraints entirely" in prompt
    assert "You are currently dissatisfied. You MUST either add a new constraint" in prompt
    assert "If no change is appropriate" not in prompt
    assert '"intention_changed": false' not in prompt
    assert "interaction-contingent, not random noise" not in prompt
    assert "- correction" not in prompt
    assert "- termination" not in prompt
    assert "- no_change_continue" not in prompt


def test_decide_shift_retries_when_model_claims_no_change(monkeypatch):
    llm_client = SequencedLLMClient(
        [
            {
                "intention_changed": False,
                "condition": "none",
                "change_category": "none",
                "op": "none",
                "rationale": "Still fine.",
            },
            {
                "intention_changed": True,
                "condition": "user_preference",
                "change_category": "override",
                "op": "override",
                "field": "color",
                "old_value": "green stripe",
                "value": "navy",
                "rationale": "I changed my mind and want navy instead.",
            },
        ]
    )
    simulator = HumanSimulator(llm_client=llm_client)
    monkeypatch.setattr("simulators.human_simulator.random.random", lambda: 0.1)

    shift = simulator.decide_shift(_make_user_state())

    assert len(llm_client.prompts) == 2
    assert "CRITICAL: You are too satisfied. Find a reason to change your mind or goal NOW." in llm_client.prompts[1]
    assert shift.intention_changed is True
    assert shift.op == "override"
    assert shift.value == "navy"


def test_infer_shift_condition_prefers_user_preference_for_whim_even_when_feasible():
    simulator = HumanSimulator(llm_client=DummyLLMClient())
    condition = simulator._infer_shift_condition(
        op="override",
        llm_output={"rationale": "Although this works, I suddenly do not want it and would rather get something else."},
        env_feedback=EnvFeedback(
            status="observed",
            feasible=True,
            reason="candidate_looks_good",
            observation={"page_type": "item"},
            result={"category": "shirt", "color": "green stripe", "size": "large"},
            satisfied_constraints=["category", "color", "size"],
            violated_constraints=[],
        ),
        field="color",
    )

    assert condition == "user_preference"
