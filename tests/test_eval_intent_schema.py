import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eval.intent_schema import normalize_intent_prediction, environment_intention
from eval.evaluators.constraint_importance_eval import evaluate_state_understanding
from eval.fixed_user_llm_executor import FixedUserLLMWebShopExecutor
from eval.single_agent_real_pilot import run_single_agent_travel_instance


def atom(field='day_2_required_activity', value='Garden', priority='must_have'):
    return dict(field=field, value=value, priority=priority)


def test_repeated_fields_are_preserved_with_distinct_values():
    raw = {'intent': [atom(), atom(field='day_3_required_activity'), atom(value='Museum'), atom(field='day_2_excluded_activity', value='Park')]}
    normalized = normalize_intent_prediction(raw)
    assert normalized == raw
    assert len(normalized['intent']) == 4
    assert evaluate_state_understanding(raw, normalized)['combined_weighted_score'] == 1
    normalized['intent'].pop(1)
    assert evaluate_state_understanding(raw, normalized)['constraint_weighted_score'] == .75


@pytest.mark.parametrize('update', [{'field': 'day_1_required_activity'}, {'field': 'day_2_excluded_activity'}, {'value': 'Other'}])
def test_wrong_context_or_value_does_not_receive_value_credit(update):
    gold = {'intent': [atom()]}
    pred = copy.deepcopy(gold)
    pred['intent'][0].update(update)
    assert evaluate_state_understanding(gold, pred)['constraint_weighted_score'] == 0


@pytest.mark.parametrize('raw', [
    {'constraints': {}, 'priority': {}},
    {'intent': [dict(field='budget', value=2000)]},
    {'intent': [atom(priority='high')]},
    {'intent': [dict(atom(), relation='include')]},
    {'intent': [dict(atom(), scope='Day 2')]},
    {'intent': [atom(), atom(priority='optional')]},
])
def test_invalid_atom_schema_is_rejected(raw):
    with pytest.raises(ValueError):
        normalize_intent_prediction(raw)


def test_environment_view_preserves_contextual_and_repeated_items():
    raw = {'intent': [atom('budget_max', 2000, 'preferred'), atom(), atom(value='Museum'), atom('excluded_room_type', 'shared')]}
    view = environment_intention(raw)
    assert view['constraints'] == {'budget': 2000}
    assert view['intent'] == raw['intent']
    assert set(raw) == {'intent'}


def test_webshop_output_uses_same_intent_contract():
    raw = {'intent': [atom('budget_max', 100)]}
    executor = FixedUserLLMWebShopExecutor(None)
    assert executor._normalize_intention_understanding(raw) == raw
    prompt = executor._build_prompt([], 'Find a desk', {})
    assert '"intent"' in prompt and 'INTENT_SCHEMA_PLACEHOLDER' not in prompt


def test_saved_single_agent_result_contains_atomic_prediction_and_unchanged_action():
    prediction = {'intent': [atom('budget_max', 2000)]}
    plan = {'itinerary': [{'day': 1, 'current_city': 'Boston', 'accommodation': '-'}]}
    class Client:
        def generate_json(self, prompt):
            return {'predicted_current_intention': copy.deepcopy(prediction),
                    'action_type': 'Planner', 'action_payload': {'query': 'Plan a trip', 'plan': plan}}
    rows = run_single_agent_travel_instance(instance={
        'instance_id': 'schema_test', 'world_state': {'reference_information': {}},
        'turns': [{'turn_id': 0, 'user_utterance': 'Plan a trip with budget 2000',
                   'gold_current_intention': {'constraints': {'budget': 2000}, 'priority': {'high': ['budget']}}}]},
        client=Client(), max_internal_steps=1)
    assert rows[0]['agent_intention_prediction'] == prediction
    assert rows[0]['agent_action']['action_payload']['plan'] == plan
    assert rows[0]['rollout_trace'][0]['action']['predicted_current_intention'] == prediction
