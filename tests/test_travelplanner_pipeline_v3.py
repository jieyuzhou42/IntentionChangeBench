import argparse
import hashlib
import json
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_travelplanner_eval_v2 as runner


class FakeClient:
    def generate_json(self,prompt):
        if 'You prepare a frozen audit baseline' in prompt:
            return {'gold_atoms':[{'atom_id':'g1','source_field':'room_type','value':'Entire home/apt'}],
                'activity_requirements':[], 'out_of_scope':[],
                'constraint_criteria':[{'gold_field':'room_type','criteria':'An entire apartment','quote':''}],
                'gold_plan_judgments':[{'gold_field':'room_type','status':'satisfied','evidence':'DB type'}],
                'annotation_issues':[]}
        if 'final_plan' in prompt:
            return {'final_plan':[], 'applied_revisions':[], 'constraint_judgments':[
                {'gold_field':'room_type','action_status':'satisfied','evidence':'DB type'}],
                'unmet_constraint_disclosures':[]}
        return {'pred_atoms':[]}


class TravelPipelineTests(unittest.TestCase):
    def setUp(self):
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_prepare_run_summary_and_human_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);gold_dir=root/'gold';run=root/'run';out=root/'out'
            gold_dir.mkdir();(run/'output').mkdir(parents=True);out.mkdir()
            plan=[{'day':'2022-01-01','accommodation':'Big Loft'}, {'day':'2022-01-02','accommodation':'Big Loft'}]
            gt={'turn_id':0,'user_utterance':'An entire apartment',
                'gold_current_intention':{'constraints':{'room_type':'Entire home/apt'},
                    'priority':{'high':['room_type'],'medium':[],'low':[]}},
                'gold_action':{'world_feasibility':{'feasible':True}, 'action_payload':{'plan':{'itinerary':plan}}}}
            ref={'Accommodations in X':[{'NAME':'Big Loft','price':100,'minimum nights':2,'maximum occupancy':3,'room type':'Entire home/apt'}]}
            shard=[{'instance_id':'travel_fixture','world_state':{'reference_information':ref},'turns':[gt]}]
            raw=json.dumps(shard).encode();(gold_dir/'shard.json').write_bytes(raw)
            trajectory={'metadata':{'dataset_sha256':hashlib.sha256(raw).hexdigest()},'trajectories':[
                {'instance_id':'travel_fixture','turns':[{'turn_id':0,'user_utterance':gt['user_utterance'],
                    'action':{'itinerary':plan},'agent_intention_prediction':{'intent':[]}}]}]}
            (run/'output/model.json').write_text(json.dumps(trajectory))
            (run/'manifest.json').write_text(json.dumps({'outputs':[{'shard_file':'shard.json','output':'model.json','model_slug':'test','shard_slug':'s1'}]}))
            args=argparse.Namespace(out=out,instances=None,models=None,parallelism=1,tag='',compare_v1=None)
            data=runner.load_data(run,gold_dir)
            self.assertTrue(data['gold_turns'][('travel_fixture',0)]['world_feasible'])
            judge=object.__new__(runner.Judge)
            import threading
            judge.client=FakeClient();judge.model='fake-local';judge.out=out;judge.lock=threading.Lock()
            runner.prepare(args,data,judge);runner.run(args,data,judge);runner.summarize_cmd(args,data)
            result=json.loads((out/'scored_rows.json').read_text())['rows'][0]
            self.assertEqual(result['action']['action_score'],1)
            self.assertTrue(result['action']['hotel']['valid'])
            self.assertIsNotNone(result['intention'])
            # Cached runs validate the baseline and judge identities.
            runner.run(args,data,judge)
            (out/'intention/test__travel_fixture__t0.json').unlink()
            runner.summarize_cmd(args,data)
            result=json.loads((out/'scored_rows.json').read_text())['rows'][0]
            self.assertIsNotNone(result['action']);self.assertIsNone(result['intention'])


if __name__=='__main__':unittest.main()
