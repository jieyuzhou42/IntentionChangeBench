"""Read-only annotation health audit. Standard library only; never calls models.

Accepts trajectory lists, {instances:[...]}, or a single trajectory; supports the
legacy gold_current_intention and intention-benchmark-v1 schema. A list priority
is reported, never silently converted to a new scoring convention.
"""
from __future__ import annotations
import argparse
import collections
import csv
import glob
import hashlib
import json
from pathlib import Path

LEVELS = ('high', 'medium', 'low')
PARAMETERS = {'days', 'org', 'dest', 'start_date', 'end_date', 'people_number', 'visiting_city_number'}
LEGACY_QUERY_FIELDS = PARAMETERS | {'budget'}


def instances_from(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get('instances'), list):
            payload = payload['instances']
        elif isinstance(payload.get('turns'), list):
            payload = [payload]
        else:
            return None
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(i, dict) and isinstance(i.get('turns'), list) for i in payload):
        return None
    return payload


def flatten_legacy(g):
    c = dict(g.get('constraints') or {}) if isinstance(g.get('constraints'), dict) else {}
    entities = g.get('entities')
    for eid, e in (entities if isinstance(entities, dict) else {}).items():
        if isinstance(e, dict) and isinstance(e.get('constraints'), dict):
            c.update({f'entities.{eid}.constraints.{k}': v for k, v in e['constraints'].items()})
    return c


def assess(instances, *, max_examples=50, require_two_levels=False, require_confirmed=False):
    counts = collections.Counter(instances=len(instances))
    issues = collections.Counter()
    examples = []
    def issue(code, severity, iid, tid, detail):
        issues[code] += 1
        counts[severity] += 1
        if len(examples) < max_examples:
            examples.append(dict(code=code, severity=severity, instance_id=iid, turn_id=tid, detail=detail))
    seen_ids = set()
    for inst in instances:
        iid = inst.get('instance_id', '<missing>')
        if iid in seen_ids:
            issue('duplicate_instance_id', 'errors', iid, None, 'Repeated ID in this file')
        seen_ids.add(iid)
        seed = (inst['turns'][0].get('gold_current_intention') or {}) if inst['turns'] else {}
        seed_c = seed.get('constraints') or seed.get('given_parameters') or {}
        seen_turns = set()
        for turn in inst['turns']:
            tid = turn.get('turn_id')
            if tid in seen_turns:
                issue('duplicate_turn_id', 'errors', iid, tid, 'Repeated turn ID')
            seen_turns.add(tid)
            counts['turns'] += 1
            g = turn.get('gold_current_intention')
            if not isinstance(g, dict):
                issue('missing_gold', 'errors', iid, tid, 'gold_current_intention is missing or not an object')
                continue
            domain = g.get('domain') or (inst.get('world_state') or {}).get('domain') or inst.get('domain')
            new_schema = 'preferences' in g or 'given_parameters' in g
            counts['new_schema_turns' if new_schema else 'legacy_turns'] += 1
            if new_schema:
                prefs, given = g.get('preferences'), g.get('given_parameters')
                if not isinstance(prefs, dict) or not isinstance(given, dict):
                    issue('invalid_layers', 'errors', iid, tid, 'given_parameters and preferences must both be dicts')
                    continue
                c = prefs
                overlap = set(given) & set(prefs)
                if overlap:
                    issue('given_in_preferences', 'errors', iid, tid, sorted(overlap))
                dynamic = set(g.get('dynamic_parameter_fields') or [])
                mixed = (set(prefs) & PARAMETERS) - dynamic if domain == 'travelplanner' else set()
                if mixed:
                    issue('unexplained_parameter_in_preferences', 'errors', iid, tid, sorted(mixed))
                if not dynamic <= set(prefs):
                    issue('dangling_dynamic_parameter', 'errors', iid, tid, sorted(dynamic - set(prefs)))
                if 'budget' in given:
                    issue('budget_hidden_from_scoring', 'errors', iid, tid, 'Budget is an active resource limit, not an unscored fact')
            else:
                c = flatten_legacy(g)
                if domain == 'travelplanner':
                    prefs = {k: v for k, v in c.items() if k not in PARAMETERS or seed_c.get(k) != v}
                else:
                    prefs = c
            if any(v is None for v in c.values()):
                issue('null_constraint', 'errors', iid, tid, 'Remove inactive constraints rather than leaving null')
            active = {k for k, v in c.items() if v is not None}
            counts['constraint_occurrences'] += len(active)
            counts['preference_occurrences'] += len(prefs)
            p = g.get('priority')
            tiered = {}
            if isinstance(p, list):
                counts['list_priority_turns'] += 1
                issue('priority_is_list', 'errors', iid, tid, 'Legacy rank list is incompatible with tier semantics; do not auto-convert')
            elif isinstance(p, dict):
                counts['dict_priority_turns'] += 1
                invalid_keys = set(p) - set(LEVELS)
                if invalid_keys:
                    issue('invalid_priority_keys', 'errors', iid, tid, sorted(invalid_keys))
                entries = []
                for level in LEVELS:
                    values = p.get(level)
                    if not isinstance(values, list):
                        issue('invalid_priority_level', 'errors', iid, tid, level)
                        values = []
                    for k in values:
                        if not isinstance(k, str):
                            issue('invalid_priority_reference', 'errors', iid, tid, repr(k))
                            continue
                        entries.append(k)
                        tiered[k] = level
                        counts[level] += 1
                        if level == 'high' and k in LEGACY_QUERY_FIELDS and domain == 'travelplanner':
                            counts['query_fields_in_high'] += 1
                if len(entries) != len(set(entries)):
                    issue('duplicate_priority_reference', 'errors', iid, tid, 'A field occurs more than once')
                if set(entries) != active:
                    issue('priority_field_mismatch', 'errors', iid, tid,
                          dict(missing=sorted(active-set(entries)), extra=sorted(set(entries)-active)))
                used = {tiered[k] for k in active if k in tiered}
                if active and used == {'high'}:
                    counts['all_high_turns'] += 1
                if active and len(used) == 1:
                    counts['single_tier_turns'] += 1
                pref_tiers = {tiered[k] for k in prefs if k in tiered}
                if len(prefs) >= 2:
                    counts['multi_preference_turns'] += 1
                    if len(pref_tiers) < 2:
                        counts['degenerate_preference_turns'] += 1
                        issue('priority_degenerate', 'errors' if require_two_levels else 'warnings', iid, tid,
                              'No tier contrast among >=2 preferences; review suitability for priority evaluation, not necessarily semantic error')
                else:
                    counts['tier_comparison_not_applicable'] += 1
            else:
                issue('missing_priority', 'errors', iid, tid, type(p).__name__)
            ga = turn.get('gold_action')
            if not isinstance(ga, dict):
                counts['gold_action_missing'] += 1
                issue('gold_action_missing', 'errors' if require_confirmed and tid != 0 else 'warnings', iid, tid, 'No adjudicated action')
            elif ga.get('confirmed') is not True:
                counts['gold_action_unconfirmed'] += 1
                issue('gold_action_unconfirmed', 'errors' if require_confirmed and tid != 0 else 'warnings', iid, tid, 'Confirmation is absent or false')
            else:
                counts['gold_action_confirmed'] += 1
                payload = ga.get('action_payload') or {}
                valid = isinstance(payload, dict)
                if domain == 'travelplanner':
                    itinerary = (payload.get('plan') or {}).get('itinerary') if valid and isinstance(payload.get('plan'),dict) else None
                    valid = isinstance(itinerary, list) and bool(itinerary)
                    if valid:
                        required = ['current_city','transportation','breakfast','lunch','dinner','attraction','accommodation']
                        valid = all(isinstance(day,dict) and all(str(day.get(k) or '').strip() for k in required) for day in itinerary)
                elif domain == 'webshop':
                    valid = valid and bool(str(payload.get('selected_asin') or '').strip())
                if not valid:
                    issue('confirmed_action_incomplete', 'errors', iid, tid, 'Confirmed flag without domain-required payload')
                else:
                    counts['gold_action_confirmed_structurally_valid'] += 1
    counts['issue_count'] = sum(issues.values())
    return dict(counts=dict(counts), issues=dict(issues), examples=examples)


def audit_file(path, **kwargs):
    raw = path.read_bytes()
    try:
        data = instances_from(json.loads(raw))
    except (ValueError, UnicodeDecodeError) as e:
        return dict(path=str(path), status='invalid_json', error=str(e), sha256=hashlib.sha256(raw).hexdigest())
    if data is None:
        return dict(path=str(path), status='not_annotation_dataset', sha256=hashlib.sha256(raw).hexdigest())
    return dict(path=str(path), status='audited', sha256=hashlib.sha256(raw).hexdigest(), **assess(data, **kwargs))


def discover(root):
    paths = []
    for base in [root/'data',root/'annotation'/'data']:
        for path in base.rglob('*.json'):
            if any('cache' in part.lower() for part in path.relative_to(base).parts):
                continue
            if 'intent_eval_runs' in path.parts or path.name.endswith('.revision.json'):
                continue
            paths.append(path)
    return sorted(set(paths))


def write_reports(records, out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    fields=['path','duplicate_of','instances','turns','high','medium','low','all_high_turns','list_priority_turns',
            'degenerate_preference_turns','gold_action_confirmed','gold_action_missing','gold_action_unconfirmed','errors','warnings']
    with (out/'audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            if r['status']=='audited':
                w.writerow({k:r.get(k,r['counts'].get(k,0)) for k in fields})
    lines=['# 全量标注体检（只读）','',
           '逐文件统计；相同 SHA-256 标记重复，不应跨行相加。high/medium/low 为原始标签次数；列表优先级不强行映射。',
           '偏好层退化：排除未变动的 TravelPlanner query 参数（预算保留）后，至少两个要求却只有一个层级。它是区分度提示，不自动证明标错。',
           'confirmed 仅检查标志与结构，不能证明行动正确。完整问题定位见 audit.json。','',
           '| 文件 | 条/轮 | H/M/L | 全 high 轮 | list 轮 | 偏好退化轮 | 已确认 action | 错误/提示 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        if r['status']!='audited':continue
        c=r['counts'];total=sum(c.get(k,0) for k in LEVELS)
        dist='/'.join(f'{100*c.get(k,0)/total:.1f}%' for k in LEVELS) if total else '—'
        label=r['path']+(' [duplicate]' if r.get('duplicate_of') else '')
        lines.append(f"| {label} | {c.get('instances',0)}/{c.get('turns',0)} | {dist} | {c.get('all_high_turns',0)} | {c.get('list_priority_turns',0)} | {c.get('degenerate_preference_turns',0)} | {c.get('gold_action_confirmed',0)} | {c.get('errors',0)}/{c.get('warnings',0)} |")
    (out/'audit.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',nargs='*',help='JSON files, directories, or quoted globs')
    parser.add_argument('--all',action='store_true',help='Discover annotation datasets under data/ and annotation/data/, excluding caches')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--require-two-levels',action='store_true',help='Treat tier degeneration as an error for a contrastive-priority subset')
    parser.add_argument('--require-confirmed',action='store_true',help='Require confirmed nonseed gold actions for release')
    parser.add_argument('--strict',action='store_true',help='Exit 1 on any reported errors, 2 on unreadable/invalid input')
    parser.add_argument('--max-examples',type=int,default=50)
    args=parser.parse_args()
    paths=discover(args.root) if args.all else []
    for spec in args.paths:
        matches=glob.glob(spec,recursive=True) or [spec]
        for match in matches:
            p=Path(match)
            paths.extend(sorted(p.rglob('*.json')) if p.is_dir() else [p])
    if not paths:parser.error('Provide paths or --all')
    records=[]; hashes={}
    for path in sorted(set(p.resolve() for p in paths)):
        if args.output_dir.resolve() in path.parents:continue
        try:r=audit_file(path,max_examples=args.max_examples,require_two_levels=args.require_two_levels,require_confirmed=args.require_confirmed)
        except (OSError,TypeError,AttributeError) as e:r=dict(path=str(path),status='unreadable_or_invalid_schema',error=str(e))
        if r['status']=='audited':
            if r['sha256'] in hashes:r['duplicate_of']=hashes[r['sha256']]
            else:hashes[r['sha256']]=str(path)
        records.append(r)
    write_reports(records,args.output_dir)
    print(json.dumps(dict(files=len(records),audited=sum(r['status']=='audited' for r in records),
                          unique_contents=len(hashes),output=str(args.output_dir)),ensure_ascii=False))
    if args.strict:
        if any(r['status'] in ['invalid_json','unreadable_or_invalid_schema'] for r in records):raise SystemExit(2)
        if not any(r['status']=='audited' for r in records):raise SystemExit(2)
        if any(r.get('counts',{}).get('errors',0) for r in records):raise SystemExit(1)


if __name__=='__main__':main()
