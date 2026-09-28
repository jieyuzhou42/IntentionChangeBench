import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_eval as runner
from eval import webshop_eval as W
from eval.baseline import fingerprint
from eval.webshop_checks import validate_selection
from eval.summarize import summarize


class FakeLLM:
    def generate_json(self,prompt):
        if 'Prepare a shared baseline' in prompt:
            return {'gold_atoms':[{'atom_id':'g'+f,'source_field':f,'value':v} for f,v in [('category','cape'),('color','pink')]],
                'constraint_criteria':[{'gold_field':f,'criteria':v,'quote':''} for f,v in [('category','cape'),('color','pink')]],
                'gold_plan_judgments':[], 'annotation_issues':[]}
        if 'Audit only this selected action' in prompt:
            return {'constraint_judgments':[{'gold_field':'category','action_status':'satisfied','evidence':'title: cape'},
                {'gold_field':'color','action_status':'violated','evidence':'no pink evidence'}], 'unmet_constraint_disclosures':[]}
        return {'pred_atoms':[]}


class WebShopPipelineTests(unittest.TestCase):
    def fixture(self,root):
        case={'instance_id':'webshop_fixture','turns':[{'turn_id':0,'user_utterance':'A pink cape',
            'gold_current_intention':{'constraints':{'category':'cape','color':'pink'},
                'priority':{'high':['category'],'medium':[],'low':['color']}},
            'gold_action':None,'world_feasibility':{'feasible':False}}]}
        trajectory={'trajectories':[{'instance_id':'webshop_fixture','turns':[{'turn_id':0,'user_utterance':'A pink cape',
            'env_feedback':{'selected_asin':'A'},'rollout_trace':[{'selected_options':{'size':'large'}}],
            'agent_intention_prediction':{'intent':[]}}]}]}
        for name,value in [('gold', [case]),('catalog',[{'asin':'A','title':'cape','price':12,'options':{'size':['large','small']}}]),('agent',trajectory)]:
            runner.write(root/(name+'.json'),value)
        return argparse.Namespace(stage='prepare',domain='webshop',gold=root/'gold.json',catalog=root/'catalog.json',
            trajectory=['test='+str(root/'agent.json')],instances=None,out=root/'out',
            judge_model='fake-local',dump_prompts=None,timeout=1,max_tokens=1000)

    def test_pipeline_prepare_run_summarize_and_stale_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            def init(obj,args): obj.args=args;obj.client=FakeLLM()
            with patch.object(runner.Judge,'__init__',init):
                runner.run_webshop(args)
                args.stage='run';runner.run_webshop(args)
                args.stage='summarize';runner.run_webshop(args)
            result=runner.read(args.out/'scored_rows.json')['rows'][0]
            self.assertTrue(result['action']['must_gate'])
            self.assertFalse(result['action']['action_success'])
            self.assertEqual(result['action']['action_score'],0)
            self.assertIsNotNone(result['intention'])
            metrics=runner.read(args.out/'metrics.json')['models']['test']
            self.assertEqual(metrics['action_scored_turns'],1)
            self.assertEqual(metrics['intention_scored_turns'],1)
            # Intention API/schema failures cannot remove completed Action results.
            (args.out/'intention/test/webshop_fixture__t0.json').unlink()
            runner.run_webshop(args)
            metrics=runner.read(args.out/'metrics.json')['models']['test']
            self.assertEqual(metrics['action_scored_turns'],1)
            self.assertEqual(metrics['intention_scored_turns'],0)
            catalog=runner.read(args.catalog);catalog[0]['price']=99;runner.write(args.catalog,catalog)
            with self.assertRaises(ValueError):runner.run_webshop(args)

    def test_options_and_missing_catalog(self):
        catalog={'A':{'asin':'A','options':{'size':['small','large']}}}
        self.assertFalse(validate_selection({'asin':'A','options':{'size':'XXL'}},catalog)['valid'])
        self.assertTrue(validate_selection({'asin':'A','options':{}},catalog)['valid'])
        with self.assertRaises(ValueError):validate_selection({'asin':'B'},catalog)
        self.assertFalse(validate_selection({'asin':None},catalog)['valid'])

    def test_missing_feasibility_with_actual_gold_defaults_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));case=runner.read(args.gold)[0]
            catalog=runner.catalog_records(runner.read(args.catalog))
            turn=case['turns'][0]
            del turn['world_feasibility']
            turn['gold_action']={'action_payload':{'selected_asin':'A'}}
            payload=W.make_input(case,0,catalog)
            self.assertTrue(payload['has_gold_action'])
            self.assertTrue(payload['world_feasible'])
            turn['world_feasibility']=False
            self.assertFalse(W.make_input(case,0,catalog)['world_feasible'])

    def test_current_gold_is_authoritative_over_delta_and_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));case=runner.read(args.gold)[0]
            catalog=runner.catalog_records(runner.read(args.catalog))
            turn=case['turns'][0]
            turn['gold_delta']={'kit_components':{'op':'add','new':'power adapter'},
                                'color':{'op':'remove','old':'pink'}}
            turn['agent_intention_prediction']={'constraints':{'kit_components':'power adapter'}}
            payload=W.make_input(case,0,catalog)
            prompt=W.baseline_prompt(payload)
            supplied=json.loads(prompt.split('\nINPUT:\n',1)[1])
            self.assertNotIn('gold_delta',supplied)
            self.assertEqual(supplied['gold']['constraints'],{'category':'cape','color':'pink'})
            self.assertIn('kit_components',payload['gold_delta'])
            raw=FakeLLM().generate_json(prompt)
            W.validate_baseline(raw,payload)
            extra=copy.deepcopy(raw)
            extra['constraint_criteria'].append({'gold_field':'kit_components','criteria':'adapter','quote':''})
            with self.assertRaises(ValueError):W.validate_baseline(extra,payload)
            missing=copy.deepcopy(raw)
            missing['constraint_criteria'].pop()
            with self.assertRaises(ValueError):W.validate_baseline(missing,payload)

    def test_pipeline_outputs_turn_priority_scores(self):
        class PriorityLLM(FakeLLM):
            def generate_json(self,prompt):
                if 'PAYLOAD:' not in prompt:
                    return super().generate_json(prompt)
                payload=json.loads(prompt.split('PAYLOAD:\n',1)[1])
                assert payload['predicted_items'][0]['priority']=='must_have'
                return {'pred_atoms':[
                    {'source_index':0,'gold_atom_id':'gcategory','value_match':True,'scope_match':True,
                     'change_vs_previous':'new','priority_change_vs_previous':'new'},
                    {'source_index':1,'gold_atom_id':'gcolor','value_match':False,'scope_match':True,
                     'change_vs_previous':'new','priority_change_vs_previous':'new'}]}
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            trajectory=runner.read(Path(tmp)/'agent.json')
            trajectory['trajectories'][0]['turns'][0]['agent_intention_prediction']={'intent':[
                {'field':'category','value':'cape','priority':'must_have'},
                {'field':'color','value':'blue','priority':'optional'}]}
            runner.write(Path(tmp)/'agent.json',trajectory)
            def init(obj,args):obj.args=args;obj.client=PriorityLLM()
            with patch.object(runner.Judge,'__init__',init):
                for stage in ('prepare','run','summarize'):
                    args.stage=stage;runner.run_webshop(args)
            metrics=runner.read(args.out/'metrics.json')
            self.assertEqual(metrics['intention_scoring_version'],runner.I.INTENTION_SCORING_VERSION)
            result=metrics['models']['test']['intention']
            self.assertEqual(result['conditional_priority_accuracy']['value'],1)
            self.assertEqual(result['priority_f1']['value'],.5)
            self.assertEqual(result['priority_recall']['value'],.5)
            self.assertNotIn('micro_precision',result)

    def test_baseline_is_model_independent_and_missing_evidence_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));case=runner.read(args.gold)[0];catalog=runner.catalog_records(runner.read(args.catalog))
            a=W.make_input(case,0,catalog)
            case['turns'][0]['agent_action']={'rationale':'I meet everything'}
            self.assertEqual(a,W.make_input(case,0,catalog))
            self.assertIn('without evidence',W.baseline_prompt(a))
            self.assertFalse(a['has_gold_action'])


if __name__=='__main__':unittest.main()
