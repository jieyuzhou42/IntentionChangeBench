import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eval import travelplanner_eval_v2 as I
from eval.baseline import normalize_gold
from eval.summarize import summarize, tables


class IntentionPriorityTests(unittest.TestCase):
    def score(self, specs, fields=None, delta=None, first=False):
        fields = fields or {'budget': 'must_have', 'return': 'preferred'}
        gold = {'constraints': {f: 'requirement' for f in fields},
                'priority': {t: [f for f, tier in fields.items() if tier == t] for t in I.TIERS}}
        baseline = {'judge': {'gold_atoms': [
            {'atom_id': f, 'source_field': f, 'value': 'requirement'} for f in fields]}}
        items, atoms = [], []
        for n, spec in enumerate(specs):
            gid = spec.get('id')
            items.append({'field': gid or 'extra', 'value': 'prediction', 'priority': spec.get('priority', 'must_have')})
            atoms.append({'source_index': n, 'gold_atom_id': gid,
                          'value_match': spec.get('value', gid is not None),
                          'scope_match': spec.get('scope', gid is not None),
                          'change_vs_previous': spec.get('change', 'unchanged'),
                          'priority_change_vs_previous': spec.get('priority_change', 'unchanged')})
        return I.score_intention(gold=gold, gold_delta=delta or {}, baseline=baseline,
                                 judgment={'pred_atoms': atoms}, items=items, first_turn=first)

    def row(self, result, tid=0):
        return {'model': 'test', 'instance_id': 'fixture', 'turn_id': tid, 'shard': 'all',
                'action': None, 'intention': result, 'errors': {}}

    def test_wrong_budget_value_cannot_earn_priority_credit(self):
        r = self.score([{'id': 'budget', 'value': False}])
        self.assertEqual(r['priority_correct'], 0)
        self.assertEqual(r['priority_f1'], 0)
        self.assertIsNone(r['conditional_priority_accuracy'])

    def test_wrong_entity_scope_cannot_earn_content_or_priority_credit(self):
        r = self.score([{'id': 'budget', 'scope': False}])
        self.assertEqual(r['correct'], 0)
        self.assertEqual(r['priority_correct'], 0)
        self.assertIsNone(r['conditional_priority_accuracy'])

    def test_missing_extra_and_wrong_priority_have_distinct_denominators(self):
        r = self.score([{'id': 'budget'}, {'id': None},
                        {'id': 'return', 'priority': 'optional'}],
                       fields={'budget': 'must_have', 'return': 'preferred', 'room': 'optional'})
        self.assertEqual(r['correct'], 2)
        self.assertEqual(r['priority_correct'], 1)
        self.assertEqual(r['conditional_priority_accuracy'], .5)
        self.assertAlmostEqual(r['priority_precision'], 1/3)
        self.assertAlmostEqual(r['priority_recall'], 1/3)
        self.assertAlmostEqual(r['priority_f1'], 1/3)
        self.assertLessEqual(r['priority_recall'], r['recall'])
        self.assertLessEqual(r['priority_f1'], r['f1'])

    def test_undefined_conditional_empty_predictions_and_coverage(self):
        r = self.score([])
        self.assertIsNone(r['conditional_priority_accuracy'])
        self.assertEqual(r['priority_f1'], 0)
        m = summarize([self.row(r), self.row(None, 1)])['test']['intention']
        self.assertEqual(m['priority_f1'], {'value': 0, 'numerator': 0, 'denominator': 1, 'excluded': 1})
        self.assertEqual(m['conditional_priority_accuracy']['denominator'], 0)
        self.assertEqual(m['conditional_priority_accuracy']['excluded'], 2)

    def test_macro_is_not_pooled_and_f1_is_mean_of_turn_f1(self):
        a = self.score([{'id': 'a'}], fields={'a': 'must_have'})
        b = self.score([{'id': 'a', 'priority': 'optional'}, {'id': 'b', 'priority': 'optional'},
                        {'id': 'c', 'priority': 'optional'}],
                       fields={f: 'must_have' for f in ('a', 'b', 'c')})
        m = summarize([self.row(a), self.row(b, 1)])['test']['intention']
        self.assertEqual(m['conditional_priority_accuracy']['value'], .5)  # not 1/4
        self.assertEqual(m['priority_f1']['value'], .5)
        self.assertNotIn('micro_precision', m)
        self.assertNotIn('micro_recall', m)
        self.assertNotIn('priority_accuracy', m)
        c = self.score([{'id': 'budget'}])  # P=1, R=.5
        d = self.score([{'id': 'budget'}, {'id': None}], fields={'budget': 'must_have'})  # P=.5,R=1
        m = summarize([self.row(c), self.row(d, 1)])['test']['intention']
        self.assertAlmostEqual(m['priority_f1']['value'], 2/3)
        self.assertNotAlmostEqual(m['priority_f1']['value'], .75)

    def test_pure_reprioritize_is_scored_even_if_content_unchanged(self):
        delta = {'priority': {'op': 'reprioritize', 'old': {'high': ['budget'], 'medium': ['return']},
                             'new': {'high': ['return'], 'medium': ['budget']}}}
        fields = {'budget': 'preferred', 'return': 'must_have'}
        r = self.score([{'id': 'budget'}, {'id': 'return', 'priority': 'preferred'}], fields, delta)
        self.assertEqual(r['f1'], 1)
        self.assertIsNone(r['change'])
        self.assertEqual(r['priority_change']['gold'], 2)
        self.assertEqual(r['priority_change']['f1'], 0)
        corrected = self.score([{'id': 'budget', 'priority': 'preferred', 'priority_change': 'changed'},
                                {'id': 'return', 'priority_change': 'changed'}], fields, delta)
        self.assertEqual(corrected['priority_change']['f1'], 1)

    def test_override_relax_and_spurious_change_penalties(self):
        for op in ('override', 'relax', 'add', 'scope_correction'):
            with self.subTest(op=op):
                r = self.score([{'id': 'budget', 'change': 'changed'},
                                {'id': None, 'change': 'new'}], delta={'budget': {'op': op}})
                c = r['priority_change']
                self.assertEqual(c['precision'], .5)
                self.assertEqual(c['recall'], 1)
                self.assertAlmostEqual(c['f1'], 2/3)

    def test_wrong_priority_spurious_change_counts_as_false_positive(self):
        r = self.score([{'id': 'budget'}, {'id': 'return', 'priority': 'optional',
                                          'priority_change': 'changed'}],
                       delta={'budget': {'op': 'override'}})
        self.assertEqual(r['priority_change']['precision'], .5)
        self.assertEqual(r['priority_change']['recall'], 1)

    def test_missing_changed_constraint_is_false_negative(self):
        r = self.score([{'id': 'return', 'priority': 'preferred'}], delta={'budget': {'op': 'relax'}})
        self.assertEqual(r['priority_change']['recall'], 0)
        self.assertEqual(r['priority_change']['f1'], 0)

    def test_first_turn_no_changes_and_delta_only_field_excluded(self):
        r = self.score([{'id': 'budget'}], delta={'budget': {'op': 'add'}}, first=True)
        self.assertIsNone(r['priority_change'])
        self.assertIsNone(self.score([{'id': 'budget'}])['priority_change'])
        self.assertIsNone(self.score([{'id': 'budget'}], delta={'invented': {'op': 'add'}})['priority_change'])

    def test_entity_priority_map_and_current_gold_override_stale_delta(self):
        gold = normalize_gold({'constraints': {}, 'entities': {'kid': {
            'constraints': {'budget': 20}, 'priority': {'high': ['budget']}}}})
        delta = {'entities': {'kid': {'priority': {'op': 'reprioritize',
                  'old': {'low': ['budget']}, 'new': {'medium': ['budget', 'phantom']}}}}}
        self.assertEqual(I.changed_gold_fields(gold, delta, True), {'entities.kid.constraints.budget'})
        self.assertEqual(I.changed_gold_fields(gold, delta, False), set())

    def test_legacy_schema_and_duplicate_gold_matches_rejected(self):
        atom = {'source_index': 0, 'gold_atom_id': 'a', 'value_match': True,
                'scope_match': True, 'change_vs_previous': 'new', 'priority_change_vs_previous': 'new'}
        for key in ('scope_match', 'priority_change_vs_previous'):
            old = copy.deepcopy(atom); del old[key]
            with self.assertRaises(ValueError): I.validate_intention({'pred_atoms': [old]}, 1, {'a'}, False)
        bad = copy.deepcopy(atom); bad['value_match'] = 'false'
        with self.assertRaises(ValueError): I.validate_intention({'pred_atoms': [bad]}, 1, {'a'}, False)
        duplicate = {**atom, 'source_index': 1}
        with self.assertRaises(ValueError): I.validate_intention({'pred_atoms': [atom, duplicate]}, 2, {'a'}, False)
        with self.assertRaises(ValueError): summarize([self.row({'change': None})])

    def test_prompt_inputs_preserve_scope_and_priority(self):
        item = {'field': 'budget', 'value': 2000, 'priority': 'must_have', 'entity': 'kid', 'scope': 'Day 2'}
        self.assertEqual(I.prediction_for_judge([item], True), [{'index': 0, **item}])
        prompt = I.build_intention_prompt({}, I.load_rules())
        self.assertIn('scope_match', prompt)
        self.assertIn('priority_change_vs_previous', prompt)

    def test_report_has_explicit_names_and_no_micro(self):
        text = tables(summarize([self.row(self.score([{'id': 'budget'}]))]))
        for name in ('Conditional Priority Accuracy', 'Priority-aware F1', 'Delta Priority-aware F1'):
            self.assertIn(name, text)
        self.assertNotIn('micro', text.lower())


if __name__ == '__main__':
    unittest.main()
