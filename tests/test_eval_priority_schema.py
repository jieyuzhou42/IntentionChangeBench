import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eval.priority_schema import normalize_priority
from eval.human_annotated_pilot import normalize_agent_output, priority_weights
from eval.evaluators.constraint_importance_eval import evaluate_state_understanding
from eval.single_agent_real_pilot import SingleAgentTravelPlannerExecutor


@pytest.mark.parametrize('priority', [
    {'ranked_fields': ['budget']},
    {'must_have': ['budget'], 'preferred': ['constraints.budget'], 'optional': []},
    {'must_have': [], 'preferred': [], 'optional': []},
    {'must_have': ['nonexistent'], 'preferred': [], 'optional': []},
])
def test_invalid_tiers_do_not_silently_become_predictions(priority):
    with pytest.raises(ValueError):
        normalize_priority(priority, {'budget': 100})


def test_tiers_survive_generation_normalization_and_gold_scoring():
    pred = {"intent": [
        {"field": "budget_max", "value": 100, "priority": "must_have"},
        {"field": "room", "value": "private", "priority": "preferred"},
        {"field": "view", "value": "sea", "priority": "optional"}]}
    action = {"action_type": "plan", "itinerary": []}
    output = normalize_agent_output({"current_intention_understanding": pred, "action": action}, domain="travelplanner")
    assert output["current_intention_understanding"] == pred
    assert output["action"] == action
    gold = {"constraints": {"budget": 100, "room": "private", "view": "sea"},
            "priority": {"high": ["budget"], "medium": ["room"], "low": ["view"]}}
    assert evaluate_state_understanding(gold, pred)["priority_level_weighted_score"] == 1
    pred["intent"][0]["priority"] = "preferred"
    assert evaluate_state_understanding(gold, pred)["priority_level_weighted_score"] == .5


def test_single_agent_action_preserves_person_specific_field():
    prediction = {"intent": [{"field": "entity_2_mobility", "value": "limited",
                               "priority": "must_have"}]}
    class Client:
        def generate_json(self, prompt):
            assert '"intent"' in prompt
            assert 'Current-turn requirements are high' not in prompt
            return {"predicted_current_intention": prediction,
                    "action_type": "AccommodationSearch", "action_payload": {"city": "Boston"}}
    executor = SingleAgentTravelPlannerExecutor(llm_client=Client())
    action = executor.act([], "Plan a trip", {})
    assert action.predicted_current_intention == prediction
    assert action.action_type == "AccommodationSearch"
    assert action.action_payload["city"] == "Boston"


def test_legacy_tier_labels_map_without_inventing_a_tier_from_rank():
    assert normalize_priority({'high': ['budget'], 'medium': [], 'low': []}, {'budget': 100}) == {
        'must_have': ['budget'], 'preferred': [], 'optional': []}
