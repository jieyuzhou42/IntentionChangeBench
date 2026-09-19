import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eval.human_annotated_pilot import normalize_agent_output, build_agent_prompt, build_judge_prompt
from eval.real_environment_pilot import EvaluationTravelPlannerExecutor
from eval.single_agent_real_pilot import SingleAgentTravelPlannerExecutor


def test_nonempty_candidates_require_selection_even_without_a_perfect_match():
    intention = {'intent': [{'field': 'budget_max', 'value': 50, 'priority': 'must_have'}]}
    raw = {'current_intention_understanding': intention,
           'action': {'action_type': 'no_match', 'selected_asin': ''}}
    with pytest.raises(ValueError, match='best available'):
        normalize_agent_output(raw, domain='webshop', valid_asins=['A'])
    raw['action'] = {'action_type': 'buy', 'selected_asin': 'A', 'rationale': 'Only candidate costs 60; exceeds the budget.'}
    output = normalize_agent_output(raw, domain='webshop', valid_asins=['A'])
    assert output['action']['selected_asin'] == 'A'
    assert output['current_intention_understanding'] == intention


def test_empty_catalog_does_not_force_fabricated_selection():
    raw = {'current_intention_understanding': {'intent': []},
           'action': {'action_type': 'no_match', 'selected_asin': ''}}
    assert normalize_agent_output(raw, domain='webshop', valid_asins=[])['action']['action_type'] == 'no_match'


def test_travel_entry_points_and_judge_share_compromise_policy():
    instance = {'turns': [{'user_utterance': 'Plan a trip', 'env_feedback': {}}]}
    prompts = [build_agent_prompt(domain='travelplanner', instance=instance, turn_index=0, field_vocabulary=[]),
               SingleAgentTravelPlannerExecutor(None)._build_single_agent_prompt([], {}),
               EvaluationTravelPlannerExecutor(None)._build_plan_prompt([], 'Plan a trip', {})]
    for prompt in prompts:
        assert 'best concrete available solution' in prompt
        assert 'minimum hotel nights and occupancy' in prompt
        assert 'Do not change the intention prediction' in prompt
    judge = build_judge_prompt(domain='travelplanner', instance_id='test', judged_turns=[])
    assert 'explanation of the trade-off does not earn compliance credit' in judge
