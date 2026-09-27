"""Run the existing v3 evaluators on shard 1; retain data errors as unscored."""
import json
import argparse
import os
from pathlib import Path
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_eval as R
from common.llm_clients import OpenAIResponsesClient

OUT = ROOT / 'annotation/reports/webshop_shard1_api_gpt56luna_gold_authority_20260927'


def product_records(turn):
    """Prefer full saved product evidence over compact display projections."""
    feedback = turn.get('env_feedback') or {}
    values = list(feedback.get('candidate_items') or [])
    for key in ('selected_candidate', 'selected_item'):
        if isinstance(feedback.get(key), dict):
            values.append(feedback[key])
    records = {}
    for value in values:
        if not value.get('asin'):
            continue
        asin = str(value['asin']).upper()
        if asin not in records or len(json.dumps(value)) > len(json.dumps(records[asin])):
            records[asin] = value
    return records


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--turn', action='append', default=[], help='Limit to INSTANCE_ID:TURN_ID; retain full dialogue context')
    cli = parser.parse_args()
    OUT = cli.out
    for line in (ROOT / '.env.llm').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('\"').strip("'"))
    model = os.environ['OPENAI_MODEL']
    if 'luna' not in model.lower():
        raise ValueError('Expected configured Luna judge')
    with zipfile.ZipFile(ROOT / 'annotation/data/webshop_shard1_5_reviewed.zip') as z:
        gold = json.loads(z.read('shard1_5/shard_001_human_annotated.json'))
    ids = {c['instance_id'] for c in gold}
    catalog, trajectories, sources = {}, {}, []
    for path in sorted((ROOT / 'annotation/output/webshop_output').glob('*.json')):
        data = R.read(path)
        for case in data['trajectories']:
            if case['instance_id'] in ids:
                if case['instance_id'] in trajectories:
                    raise ValueError('Duplicate tested case')
                trajectories[case['instance_id']] = case
                sources.append(str(path.relative_to(ROOT)))
            for turn in case['turns']:
                for asin, record in product_records(turn).items():
                    if asin not in catalog or len(json.dumps(record)) > len(json.dumps(catalog[asin])):
                        catalog[asin] = record
    # Gold replay observations can contain reference products not seen by the agent.
    for case in gold:
        for turn in case['turns']:
            for asin, record in product_records(turn).items():
                if asin not in catalog:
                    catalog[asin] = record
    # Freeze only selected product records; retain all their original evidence fields.
    needed = set()
    for case in gold + list(trajectories.values()):
        for turn in case['turns']:
            for is_gold in (False, True):
                asin = R.selection(turn, gold=is_gold)['asin']
                if asin:
                    needed.add(str(asin).upper())
    catalog = {a: r for a, r in catalog.items() if a in needed}
    R.write(OUT / 'gold.json', gold)
    R.write(OUT / 'catalog.json', catalog)
    R.write(OUT / 'trajectory.json', {'trajectories': list(trajectories.values())})
    args = SimpleNamespace(judge_model=model, dump_prompts=None, timeout=180, max_tokens=16000)
    rows, jobs = [], []
    for case in gold:
        case['turns'].sort(key=lambda t: int(t['turn_id']))
        tested = {int(t['turn_id']): t for t in trajectories[case['instance_id']]['turns']}
        previous = None
        for index, gt in enumerate(case['turns']):
            tid = int(gt['turn_id'])
            turn = tested[tid]
            assert turn['user_utterance'] == gt['user_utterance']
            items = R.I.intent_items(turn.get('agent_intention_prediction'))
            row = dict(model='gpt-5.6-sol', domain='webshop', shard='all', instance_id=case['instance_id'],
                       turn_id=tid, action=None, intention=None, errors={})
            rows.append(row)
            try:
                payload = R.W.make_input(case, index, catalog)
                jobs.append((row, payload, turn, items, previous))
            except Exception as exc:
                row['errors']['data'] = str(exc)
            previous = [{'field': v.get('field'), 'value': v.get('value')} for v in items]
    if cli.turn:
        requested = set(cli.turn)
        def wanted(row):
            return row['instance_id'] + ':' + str(row['turn_id']) in requested
        rows = [r for r in rows if wanted(r)]
        jobs = [j for j in jobs if wanted(j[0])]
        if len(rows) != len(requested):
            raise ValueError('Unknown requested turn')
    R.write(OUT / 'preflight_errors.json', [r for r in rows if r['errors']])
    R.write(OUT / 'run_manifest.json', dict(judge_model=model, tested_model='gpt-5.6-sol',
        sources=sorted(set(sources)), eligible_turns=len(jobs), total_turns=len(rows),
        scoring_version=R.SCORING_VERSION, gold_source='annotation/data/webshop_shard1_5_reviewed.zip::shard1_5/shard_001_human_annotated.json'))
    print(f'Judge={model}; eligible={len(jobs)}/{len(rows)}', flush=True)

    def evaluate(job):
        row, payload, turn, items, previous = job
        judge = R.Judge(args)
        judge.client = OpenAIResponsesClient.from_env(timeout=180)
        stem = row['instance_id'] + '__t' + str(row['turn_id']) + '.json'
        invalid_attempts = []
        def validate_baseline(raw):
            try:
                return R.W.validate_baseline(raw, payload)
            except Exception as exc:
                invalid_attempts.append({'error': str(exc), 'response': raw})
                R.write(OUT / 'invalid_responses' / stem, invalid_attempts)
                raise
        try:
            baseline = judge.call(OUT / 'baseline' / stem, R.W.baseline_prompt(payload),
                validate_baseline,
                {'baseline_version': R.BASELINE_VERSION, 'input': payload, 'source_sha256': R.fingerprint(payload)})
        except Exception as exc:
            row['errors']['baseline'] = str(exc)
            return row
        meta = {'model': row['model'], 'instance_id': row['instance_id'], 'turn_id': row['turn_id'],
                'input_sha256': R.fingerprint(turn), 'baseline_sha256': R.fingerprint(baseline),
                'intention_rules_sha256': R.fingerprint(R.I.load_rules()['Intention rules'])}
        selected = R.selection(turn)
        first = len(payload['dialogue']) == 1
        ip = {'turn_id': row['turn_id'], 'dialogue_so_far': payload['dialogue'],
              'gold_atoms': [{'atom_id': a['atom_id'], 'field': a['source_field'], 'value': a['value']} for a in baseline['judge']['gold_atoms']],
              'predicted_items': [{'index': n, 'field': v.get('field'), 'value': v.get('value')} for n, v in enumerate(items)],
              'previous_turn_predicted_items': previous}
        for stage in ('action', 'intention'):
            try:
                if stage == 'action':
                    raw = judge.call(OUT / stage / row['model'] / stem, R.W.action_prompt(baseline, selected, catalog),
                        lambda raw: R.W.validate_action(raw, set(payload['gold']['constraints'])), meta)['judge']
                    row[stage] = R.W.score_action(baseline, raw, selected, catalog)
                else:
                    raw = judge.call(OUT / stage / row['model'] / stem, R.I.build_intention_prompt(ip, R.I.load_rules()),
                        lambda raw: R.I.validate_intention(raw, len(items), {a['atom_id'] for a in baseline['judge']['gold_atoms']}, first), meta)['judge']
                    row[stage] = R.I.score_intention(gold=payload['gold'], gold_delta=payload['gold_delta'],
                        baseline=baseline, judgment=raw, items=items, first_turn=first)
            except Exception as exc:
                row['errors'][stage] = str(exc)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(evaluate, j) for j in jobs]):
            row = future.result()
            print(row['instance_id'], row['turn_id'], 'errors=' + str(row['errors']), flush=True)
            R.write(OUT / 'scored_rows.json', {'scoring_version': R.SCORING_VERSION, 'rows': rows})
    metrics = R.summarize(rows)
    R.write(OUT / 'metrics.json', {'scoring_version': R.SCORING_VERSION, 'models': metrics})
    R.write(OUT / 'run_errors.json', [r for r in rows if r['errors']])
    (OUT / 'tables.md').write_text(R.tables(metrics), encoding='utf-8')
    print(R.tables(metrics), flush=True)


if __name__ == '__main__':
    main()
