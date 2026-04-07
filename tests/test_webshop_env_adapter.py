from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.webshop_env import WebShopEnvAdapter
from models import AgentAction


class DummyEnv:
    def __init__(self):
        self.server = SimpleNamespace(user_sessions={}, product_item_dict={})
        self.session = "test-session"

    def reset(self):
        return "", {}


class StepTracingEnv(DummyEnv):
    def __init__(self):
        super().__init__()
        self.server.user_sessions = {
            self.session: {
                "asin": "B000TEST01",
                "options": {"color": "red", "size": "medium"},
            }
        }
        self.server.product_item_dict = {
            "B000TEST01": {
                "Title": "Trail Shirt",
                "Description": "A red outdoor shirt.",
                "BulletPoints": ["Outdoor apparel"],
                "Brand": "Field & Pine",
                "pricing": [29.99],
                "category": "shirt",
                "product_category": "shirt",
                "options": {
                    "color": ["red", "green stripe"],
                    "size": ["medium", "large"],
                },
            }
        }

    def get_instruction_text(self):
        return "Find a shirt with color: green stripe"

    def get_available_actions(self):
        return {"clickables": ["green stripe", "large", "Buy Now"]}

    def step(self, action_text):
        if action_text == "click[green stripe]":
            self.server.user_sessions[self.session]["options"]["color"] = "green stripe"

        raw_obs = "Item Page[SEP]Trail Shirt[SEP]Price: $29.99[SEP]Buy Now"
        info = {"clickables": ["green stripe", "large", "Buy Now"]}
        return raw_obs, 0.25, False, info


def make_adapter():
    return WebShopEnvAdapter(DummyEnv())


def test_selected_color_beats_base_color_for_constraint_match():
    adapter = make_adapter()
    result = adapter._extract_result(
        {
            "selected_item": {"title": "Trail Shirt", "price": 29.99, "color": "red"},
            "selected_options": {"color": "green stripe"},
            "item_context": {
                "asin": "B000TEST01",
                "title": "Trail Shirt",
                "price": 29.99,
                "category": "shirt",
                "brand": "Field & Pine",
                "color": "red",
                "selected_options": {"color": "green stripe"},
            },
        }
    )

    satisfied, violated = adapter._check_constraints(
        result,
        {"constraints": {"color": "green stripe"}},
    )

    assert result["base_color"] == "red"
    assert result["selected_color"] == "green stripe"
    assert result["color"] == "green stripe"
    assert satisfied == ["color"]
    assert violated == []


def test_parse_instruction_preserves_letter_green_b_color():
    adapter = make_adapter()

    intention = adapter.parse_instruction_to_intention(
        "Instruction: Find a shirt with color: letter green b, price lower than 40"
    )

    assert intention["constraints"]["color"] == "letter green b"
    assert intention["constraints"]["color_exact"] == "letter green b"


def test_parse_instruction_preserves_mossy_oak_country_dna_color():
    adapter = make_adapter()

    intention = adapter.parse_instruction_to_intention(
        "Instruction: Need camo gear; color: mossy oak country dna; brand: Ridge Runner"
    )

    assert intention["constraints"]["color"] == "mossy oak country dna"
    assert intention["constraints"]["color_exact"] == "mossy oak country dna"


def test_parse_instruction_preserves_unlabeled_green_stripe_color_phrase():
    adapter = make_adapter()

    intention = adapter.parse_instruction_to_intention(
        "Instruction: Find a green stripe shirt price lower than 40"
    )

    assert intention["constraints"]["color"] == "green stripe"
    assert intention["constraints"]["color_exact"] == "green stripe"


def test_parse_instruction_preserves_unlabeled_hyphenated_color_phrase():
    adapter = make_adapter()

    intention = adapter.parse_instruction_to_intention(
        "Instruction: Need a b17-black office chair under 120"
    )

    assert intention["constraints"]["color"] == "b17-black"
    assert intention["constraints"]["color_exact"] == "b17-black"


def test_selected_size_satisfies_requested_size():
    adapter = make_adapter()
    result = adapter._extract_result(
        {
            "selected_item": {"title": "Trail Shirt", "price": 29.99},
            "selected_options": {"size": "Large"},
            "item_context": {
                "asin": "B000TEST01",
                "title": "Trail Shirt",
                "price": 29.99,
                "category": "shirt",
                "brand": "Field & Pine",
                "selected_options": {"size": "Large"},
            },
        }
    )

    satisfied, violated = adapter._check_constraints(
        result,
        {"constraints": {"size": "large"}},
    )

    assert result["selected_size"] == "Large"
    assert result["size"] == "Large"
    assert satisfied == ["size"]
    assert violated == []


def test_step_trace_exposes_selection_change_and_constraint_source():
    adapter = WebShopEnvAdapter(StepTracingEnv())

    feedback = adapter.step(
        AgentAction("click", {"target": "green stripe"}),
        {"constraints": {"color": "green stripe"}},
    )

    assert feedback.observation["executed_action"] == "click[green stripe]"
    assert feedback.observation["candidate_actions"] == ["click[green stripe]", "choose[green stripe]"]
    assert feedback.observation["pre_step_selected_options"] == {"color": "red", "size": "medium"}
    assert feedback.observation["post_step_selected_options"] == {
        "color": "green stripe",
        "size": "medium",
    }
    assert feedback.observation["selection_changed"] is True
    assert feedback.result["base_color"] == "red"
    assert feedback.result["selected_color"] == "green stripe"
    assert feedback.observation["constraint_debug"]["color"]["actual_source"] == "selected_options.color"
    assert feedback.satisfied_constraints == ["color"]
