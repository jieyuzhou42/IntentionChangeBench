from __future__ import annotations

import json
import random
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domains.webshop.candidate_diversity import select_diverse_candidates
from domains.webshop.executor import WebShopExecutor
from domains.webshop.user_simulator import WebShopUserSimulator
from models import EnvFeedback, ShiftOp
from simulation.simulation.base_user_simulator import SHIFT_CONTEXT_MARKER
from simulation.simulation.reranker import RerankerConfig


class RecordingLLM:
    def __init__(self, json_outputs=None, text_outputs=None):
        self.json_outputs = list(json_outputs or [])
        self.text_outputs = list(text_outputs or [])
        self.json_prompts = []
        self.text_prompts = []

    def generate_json(self, prompt):
        self.json_prompts.append(prompt)
        return self.json_outputs.pop(0) if self.json_outputs else {}

    def generate_text(self, prompt):
        self.text_prompts.append(prompt)
        return self.text_outputs.pop(0) if self.text_outputs else ""


def _candidate(asin, title, price, category, bullet):
    return {
        "asin": asin,
        "title": title,
        "price": price,
        "product_category": category,
        "query": category,
        "bullet_points": [bullet],
        "attributes": [],
        "options": {},
    }


def _desk_candidates():
    return [
        _candidate(
            "CONVERTER1",
            "Compact Standing Desk Converter Black 32 Inch",
            79.0,
            "standing desk converter",
            "Fits two monitors on an existing desk; manual lift",
        ),
        _candidate(
            "CONVERTER2",
            "Compact Standing Desk Converter White 32 Inch",
            82.0,
            "standing desk converter",
            "Fits two monitors on an existing desk; manual lift",
        ),
        _candidate(
            "FULLDESK01",
            "Electric Height Adjustable Complete Standing Desk",
            169.0,
            "electric standing desk",
            "Complete desk with motorized height adjustment and memory presets",
        ),
        _candidate(
            "MANUALDESK",
            "Crank Adjustable Full Standing Desk Heavy Duty",
            119.0,
            "manual standing desk",
            "Stable steel frame with manual crank adjustment",
        ),
        _candidate(
            "MOBILEDESK",
            "Small Mobile Sit Stand Laptop Cart",
            55.0,
            "mobile laptop cart",
            "Portable compact workstation for a laptop",
        ),
        _candidate(
            "FULLDESK02",
            "Electric Height Adjustable Complete Standing Desk Oak",
            175.0,
            "electric standing desk",
            "Complete desk with motorized height adjustment and memory presets",
        ),
    ]


def _feedback(candidates=None):
    return EnvFeedback(
        status="observed",
        feasible=True,
        observation={"candidate_items": list(candidates or _desk_candidates())},
    )


def _decision_output(chosen_asin="CONVERTER1", dimension="product form and price"):
    return {
        "intention_changed": True,
        "condition": "user_preference",
        "decision_point": {
            "stable_purchase_goal": "a height-adjustable workstation",
            "evidence_role": "trigger_only",
            "dimension": dimension,
            "options_compared": [
                {
                    "asin": "CONVERTER1",
                    "option": "desktop converter",
                    "tradeoff": "keeps the existing desk and costs less",
                },
                {
                    "asin": "FULLDESK01",
                    "option": "complete electric desk",
                    "tradeoff": "adds workspace but exceeds the old budget",
                },
            ],
            "chosen_option": "converter with dual-monitor space",
            "chosen_asin": chosen_asin,
            "purchase_reason": "reusing the current desk matters more than getting a full frame",
        },
        "changes": [
            {
                "category": "relax",
                "field": "category",
                "old_value": "standing desk",
                "value": "standing desk or desktop converter",
                "rationale": "the smaller form serves the same working use case",
            }
        ],
        "rationale": "A converter is more practical if it holds both monitors.",
    }


def test_candidate_selection_returns_five_and_exposes_non_variant_tradeoffs():
    selected, metadata = select_diverse_candidates(_desk_candidates(), limit=5)

    assert len(selected) == 5
    assert metadata["target_range"] == [3, 5]
    assert selected[0]["asin"] == "CONVERTER1"
    selected_asins = {item["asin"] for item in selected}
    assert {"FULLDESK01", "MANUALDESK", "MOBILEDESK"}.issubset(selected_asins)
    assert any(
        "product_form_or_category" in row["difference_axes"]
        for row in metadata["selections"]
    )


def test_shift_prompt_requires_cross_product_decision_point_and_carries_diversity():
    simulator = WebShopUserSimulator(RecordingLLM())
    feedback = _feedback(_desk_candidates()[:5])
    feedback.observation["candidate_diversity"] = {"target_range": [3, 5]}

    prompt = simulator._build_shift_prompt(
        {
            "constraints": {"category": "standing desk", "budget_max": 140},
            "priority": ["category", "budget_max"],
        },
        env_feedback=feedback,
    )
    instructions, raw_context = prompt.split(SHIFT_CONTEXT_MARKER, 1)
    context = json.loads(raw_context)

    assert '"decision_point"' in instructions
    assert "compare at least two distinct ASINs" in instructions
    assert "not a task of gradually revealing attributes" in instructions
    assert "trigger evidence, never the new target answer" in instructions
    assert "exactly one primary substantive change" in instructions
    assert context["latest_env_feedback"]["candidate_diversity"] == {
        "target_range": [3, 5]
    }


def test_real_turn_rejects_shift_without_two_real_evidence_asins():
    invalid_output = {
        "intention_changed": True,
        "condition": "user_preference",
        "changes": [
            {
                "category": "add",
                "field": "memory_controller",
                "value": True,
                "rationale": "copied from one title",
            }
        ],
        "rationale": "one product has it",
    }
    llm = RecordingLLM(
        json_outputs=[invalid_output, invalid_output, invalid_output]
    )
    simulator = WebShopUserSimulator(llm)

    shift = simulator.decide_shift(
        {"constraints": {"category": "standing desk"}, "priority": ["category"]},
        env_feedback=_feedback(),
        rng=random.Random(7),
    )

    assert shift.op == "none"
    assert shift.rationale.startswith("invalid_decision_point")
    assert len(llm.json_prompts) == 3


def test_variant_only_decision_point_is_rejected():
    output = _decision_output(dimension="color")
    llm = RecordingLLM(json_outputs=[output, output, output])
    simulator = WebShopUserSimulator(llm)

    shift = simulator.decide_shift(
        {"constraints": {"category": "standing desk"}, "priority": ["category"]},
        env_feedback=_feedback(),
        rng=random.Random(7),
    )

    assert shift.op == "none"
    assert "variant_level_decision_point" in shift.rationale


def test_valid_decision_evidence_is_saved_in_sampling_metadata():
    simulator = WebShopUserSimulator(RecordingLLM(json_outputs=[_decision_output()]))

    shift = simulator.decide_shift(
        {"constraints": {"category": "standing desk"}, "priority": ["category"]},
        env_feedback=_feedback(),
        rng=random.Random(7),
    )

    assert shift.op == "relax"
    assert shift.sampling_metadata["evidence_asins"] == ["CONVERTER1", "FULLDESK01"]
    assert shift.sampling_metadata["chosen_asin"] == "CONVERTER1"
    assert shift.sampling_metadata["decision_validation"]["valid"] is True
    assert shift.sampling_metadata["decision_point"]["evidence_role"] == "trigger_only"
    assert shift.sampling_metadata["tradeoff_validation"]["substantive_change_count"] == 1


def test_product_evidence_cannot_expand_into_multiple_target_attributes():
    output = _decision_output()
    output["changes"].extend(
        [
            {
                "category": "add",
                "field": "monitor_capacity",
                "old_value": None,
                "value": "two monitors",
                "rationale": "copied from the supporting converter",
            },
            {
                "category": "add",
                "field": "lift_method",
                "old_value": None,
                "value": "manual lift",
                "rationale": "another feature of the supporting converter",
            },
        ]
    )
    llm = RecordingLLM(json_outputs=[output, output, output])
    simulator = WebShopUserSimulator(llm)

    shift = simulator.decide_shift(
        {"constraints": {"category": "standing desk"}, "priority": ["category"]},
        env_feedback=_feedback(),
        rng=random.Random(7),
    )

    assert shift.op == "none"
    assert "multiple_substantive_changes" in shift.rationale
    assert len(llm.json_prompts) == 3


def test_identity_change_is_rejected_during_commitment_cooldown():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_tradeoff_scope(
        ShiftOp(
            op="override",
            field="product_form",
            value="complete electric desk",
        ),
        decision_point={
            "stable_purchase_goal": "a height-adjustable workstation",
            "evidence_role": "trigger_only",
        },
        intention_history=[
            {
                "turn_id": 1,
                "gold_delta": {
                    "category": {
                        "op": "relax",
                        "old": "standing desk",
                        "new": "standing desk or desktop converter",
                    }
                },
            }
        ],
    )

    assert validation["valid"] is False
    assert validation["reason"] == "identity_goal_changed_too_recently"


def test_tradeoff_scope_rejects_consecutive_attribute_additions():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_tradeoff_scope(
        ShiftOp(op="add", field="protein_min", value=15),
        decision_point={
            "stable_purchase_goal": "a complete recovery breakfast",
            "evidence_role": "trigger_only",
            "chosen_asin": "SECOND",
        },
        intention_history=[
            {"gold_delta": {"preparation": {"op": "add", "new": "ready to eat"}}}
        ],
    )

    assert validation["valid"] is False
    assert validation["reason"] == "consecutive_attribute_addition"


def test_tradeoff_scope_tolerates_string_shift_condition_history():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_tradeoff_scope(
        ShiftOp(op="override", field="budget_max", value=40),
        decision_point={
            "stable_purchase_goal": "organized cosmetic storage",
            "evidence_role": "trigger_only",
            "chosen_asin": "NEW-ASIN",
        },
        intention_history=[
            {
                "gold_delta": {"size": {"op": "override", "new": "two tiers"}},
                "shift_condition": "user_preference",
            }
        ],
    )

    assert validation["valid"] is True


def test_tradeoff_scope_rejects_reused_recent_product_evidence():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_tradeoff_scope(
        ShiftOp(op="override", field="budget_max", value=40),
        decision_point={
            "stable_purchase_goal": "organized cosmetic storage",
            "evidence_role": "trigger_only",
            "chosen_asin": "SAME-ASIN",
        },
        intention_history=[
            {
                "gold_delta": {"size": {"op": "override", "new": "two tiers"}},
                "shift_condition": {
                    "details": {
                        "candidate_sampling": {"chosen_asin": "SAME-ASIN"}
                    }
                },
            }
        ],
    )

    assert validation["valid"] is False
    assert validation["reason"] == "reused_recent_product_evidence"


def test_tradeoff_scope_allows_one_primary_change_plus_reprioritization():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_tradeoff_scope(
        ShiftOp(
            op="multiple",
            changes=[
                ShiftOp(op="relax", field="budget_max", value=180),
                ShiftOp(
                    op="reprioritize",
                    priority_update=["budget_max", "category"],
                ),
            ],
        ),
        decision_point={
            "stable_purchase_goal": "a height-adjustable workstation",
            "evidence_role": "trigger_only",
        },
        intention_history=[],
    )

    assert validation["valid"] is True
    assert validation["substantive_change_count"] == 1
    assert validation["priority_change_count"] == 1


def test_two_turn_sku_dominance_filters_another_change_from_same_product():
    simulator = WebShopUserSimulator(RecordingLLM())
    simulator._recent_decisions = [
        {"chosen_asin": "CONVERTER1"},
        {"chosen_asin": "CONVERTER1"},
    ]
    repeated = ShiftOp(
        op="add",
        field="color",
        value="black",
        sampling_metadata={"chosen_asin": "CONVERTER1", "decision_quality_score": 2.0},
    )
    alternative = ShiftOp(
        op="override",
        field="category",
        value="electric standing desk",
        sampling_metadata={"chosen_asin": "FULLDESK01", "decision_quality_score": 2.0},
    )

    pool, metadata = simulator._prepare_shift_selection_pool(
        [repeated, alternative],
        current_intention={},
        env_feedback=_feedback(),
        intention_history=None,
    )

    assert pool == [alternative]
    assert metadata["dominated_asin"] == "CONVERTER1"
    assert metadata["filtered_repeated_dominant_candidates"] == 1


def test_trajectory_guard_rejects_return_to_recent_value():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_trajectory_candidate(
        ShiftOp(op="override", field="category", value="standing desk"),
        current_intention={"constraints": {"category": "standing desk converter"}},
        intention_history=[
            {
                "turn_id": 1,
                "gold_delta": {
                    "category": {
                        "op": "override",
                        "old": "standing desk",
                        "new": "standing desk converter",
                    }
                },
            }
        ],
    )

    assert validation["valid"] is False
    assert validation["reason"] == "returns_to_recent_value"
    assert validation["field"] == "category"


def test_trajectory_guard_rejects_immediate_numeric_direction_reversal():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_trajectory_candidate(
        ShiftOp(op="override", field="budget_max", value=40),
        current_intention={"constraints": {"budget_max": 50}},
        intention_history=[
            {
                "turn_id": 2,
                "gold_delta": {
                    "budget_max": {"op": "override", "old": 30, "new": 50}
                },
            }
        ],
    )

    assert validation["valid"] is False
    assert validation["reason"] == "immediate_numeric_direction_reversal"


def test_trajectory_guard_rejects_third_consecutive_field_change():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_trajectory_candidate(
        ShiftOp(op="override", field="category", value="electric standing desk"),
        current_intention={"constraints": {"category": "manual standing desk"}},
        intention_history=[
            {
                "turn_id": 1,
                "gold_delta": {
                    "category": {
                        "op": "override",
                        "old": "standing desk",
                        "new": "standing desk converter",
                    }
                },
            },
            {
                "turn_id": 2,
                "gold_delta": {
                    "category": {
                        "op": "override",
                        "old": "standing desk converter",
                        "new": "manual standing desk",
                    }
                },
            },
        ],
    )

    assert validation["valid"] is False
    assert validation["reason"] == "third_consecutive_field_change"


def test_trajectory_guard_allows_a_new_decision_dimension():
    simulator = WebShopUserSimulator(RecordingLLM())
    validation = simulator._validate_trajectory_candidate(
        ShiftOp(op="add", field="monitor_capacity", value="two monitors"),
        current_intention={"constraints": {"category": "standing desk converter"}},
        intention_history=[
            {
                "turn_id": 1,
                "gold_delta": {
                    "category": {
                        "op": "override",
                        "old": "standing desk",
                        "new": "standing desk converter",
                    }
                },
            }
        ],
    )

    assert validation["valid"] is True
    assert validation["reason"] == "ok"


def test_invalid_trajectory_candidate_is_resampled():
    reversal = _decision_output(chosen_asin="FULLDESK01")
    reversal["changes"] = [
        {
            "category": "override",
            "field": "category",
            "old_value": "standing desk converter",
            "value": "standing desk",
            "rationale": "switch back to the earlier form",
        }
    ]
    valid = _decision_output()
    valid["changes"] = [
        {
            "category": "add",
            "field": "monitor_capacity",
            "old_value": None,
            "value": "two monitors",
            "rationale": "the converter still needs enough workspace",
        }
    ]
    llm = RecordingLLM(json_outputs=[reversal, valid])
    simulator = WebShopUserSimulator(llm)

    shift = simulator.decide_shift(
        {
            "constraints": {"category": "standing desk converter"},
            "priority": ["category"],
        },
        env_feedback=_feedback(),
        intention_history=[
            {
                "turn_id": 1,
                "gold_delta": {
                    "category": {
                        "op": "override",
                        "old": "standing desk",
                        "new": "standing desk converter",
                    }
                },
            }
        ],
        rng=random.Random(7),
    )

    assert shift.op == "add"
    assert shift.field == "monitor_capacity"
    assert shift.sampling_metadata["trajectory_validation"]["valid"] is True
    assert len(llm.json_prompts) == 2


def test_title_copy_is_retried_and_purchase_reason_is_used():
    copied = "I want the Electric Height Adjustable Complete Standing Desk because it looks good."
    paraphrase = "I'd pay more for a complete desk because the extra workspace matters more now."
    llm = RecordingLLM(text_outputs=[copied, paraphrase])
    simulator = WebShopUserSimulator(llm)
    shift = ShiftOp(
        op="relax",
        field="budget_max",
        old_value=140,
        value=180,
        sampling_metadata={"decision_point": _decision_output()["decision_point"]},
    )

    utterance = simulator.realize_shift(
        shift,
        {"constraints": {"budget_max": 140}},
        "explicit",
        env_feedback=_feedback(),
    )

    assert utterance == paraphrase
    assert len(llm.text_prompts) == 2


class _FakeSearchEnv:
    def search_candidates(self, query, user_state, *, search_limit, return_limit):
        return _feedback(_desk_candidates())


def test_executor_exposes_at_most_five_decision_candidates_without_reranking():
    executor = WebShopExecutor(
        reranker_config=RerankerConfig(enable_reranking=False, rerank_return_k=10)
    )

    _, feedback = executor.search(
        _FakeSearchEnv(),
        {"gold_search_query": "standing desk", "constraints": {}},
    )

    assert len(feedback.observation["candidate_items"]) == 5
    assert feedback.observation["candidate_diversity"]["target_range"] == [3, 5]
