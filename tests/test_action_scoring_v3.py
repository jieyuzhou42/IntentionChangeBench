import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eval.action_scoring import score_action_constraints as score
from eval.baseline import human_world_feasibility, normalize_gold, constraint_tiers, validate_scoring_priorities
from eval import travelplanner_eval_v2 as T


class ActionScoringTests(unittest.TestCase):
    def setUp(self):
        self.tiers = {'m1':'must_have','m2':'must_have','p':'preferred','o':'optional'}
        self.perfect = {f:'satisfied' for f in self.tiers}

    def calc(self, agent=None, gold=None, feasible=True, **kwargs):
        return score(tiers=self.tiers, agent=agent or self.perfect, gold=gold,
                     world_feasible=feasible, **kwargs)

    def test_feasible_gate_does_not_relax_for_bad_gold(self):
        a={**self.perfect,'m1':'violated'}
        r=self.calc(a,a)
        self.assertEqual(r['action_score'],0)

    def test_not_feasible_counts_not_identity(self):
        a={**self.perfect,'m1':'violated'}
        g={**self.perfect,'m2':'violated'}
        self.assertEqual(self.calc(a,g,False)['action_score'],1)
        a['m2']='unknown'
        self.assertEqual(self.calc(a,g,False)['action_score'],0)

    def test_soft_two_to_one_and_cap(self):
        a={**self.perfect,'p':'violated'}
        self.assertAlmostEqual(self.calc(a,self.perfect)['action_score'],1/3)
        g={**self.perfect,'o':'violated'}
        self.assertEqual(self.calc(self.perfect,g)['action_score'],1)

    def test_missing_gold_partial_score_is_binary_failure(self):
        r=self.calc({**self.perfect,'o':'unknown'},None,None)
        self.assertAlmostEqual(r['action_score'],2/3)
        self.assertFalse(r['action_success'])
        self.assertTrue(r['gold_assumed_perfect'])

    def test_missing_gold_still_requires_all_must_when_not_feasible(self):
        self.assertFalse(self.calc({**self.perfect,'m1':'violated'},None,False)['must_gate'])

    def test_zero_gold_soft_and_hotel_gate(self):
        g={**self.perfect,'p':'violated','o':'violated'}
        self.assertEqual(self.calc(g,g)['action_score'],1)
        self.assertEqual(self.calc(g,g,action_valid=False)['action_score'],0)

    def test_out_of_scope_and_unknown(self):
        a={**self.perfect,'m1':'unknown','o':'unknown'}
        self.assertEqual(self.calc(a,self.perfect)['action_score'],0)
        r=self.calc(a,self.perfect,out_of_scope=['m1','o'])
        self.assertEqual(r['action_score'],1)
        self.assertEqual(r['totals']['optional'],0)

    def test_missing_human_label_defaults_feasible(self):
        a={**self.perfect,'m1':'violated'}
        result=self.calc(a,a,None)
        self.assertTrue(result['world_feasible'])
        self.assertFalse(result['must_gate'])
        self.assertTrue(self.calc(a,a,False)['must_gate'])
        self.assertTrue(human_world_feasibility({'gold_action':{'status':'not_feasible'}}))
        self.assertTrue(human_world_feasibility({'world_feasibility':None}))
        self.assertTrue(human_world_feasibility({'gold_action':{'world_feasibility':{'feasible':None}}}))
        self.assertFalse(human_world_feasibility({'gold_action':{'world_feasibility':{'feasible':False}}}))

    def test_human_feasibility_conflicts_and_invalid_values_still_fail(self):
        with self.assertRaises(ValueError):
            human_world_feasibility({'world_feasibility':True,'gold_action':{'world_feasibility':False}})
        with self.assertRaises(ValueError):
            human_world_feasibility({'world_feasibility':'false'})

    def test_travel_feasibility_defaults_true_without_inferring_from_gold(self):
        baseline={'has_gold_plan':True,'out_of_scope_fields':[],
                  'judge':{'gold_plan_judgments':[{'gold_field':'m','status':'violated'}]}}
        gold={'constraints':{'m':True},'priority':{'high':['m']}}
        result=T.feasibility(baseline,gold)
        self.assertTrue(result['world_feasible'])
        self.assertEqual(result['violated_musts'],['m'])
        baseline['world_feasible']=False
        self.assertFalse(T.feasibility(baseline,gold)['world_feasible'])

    def test_priority_conflict_and_entity_scope(self):
        gold=normalize_gold({'constraints':{'x':1},'priority':{'high':['x'],'low':['x']},
                             'entities':{'e':{'constraints':{'mobility':'limited'}}}})
        with self.assertRaises(ValueError): validate_scoring_priorities(gold)
        gold['priority']['optional']=[]
        validate_scoring_priorities(gold,['entities.e.constraints.mobility'])
        self.assertEqual(constraint_tiers(gold)['x'],'must_have')

    def test_budget_behavior_is_preserved(self):
        result=T.budget_verdict([{'lunch':'packed unpriced lunch'}],None,{},100,1)
        self.assertEqual(result['status'],'satisfied')
        self.assertTrue(result['unpriced'])


if __name__=='__main__': unittest.main()
