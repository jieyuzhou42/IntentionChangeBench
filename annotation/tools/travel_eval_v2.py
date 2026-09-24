#!/usr/bin/env python3
"""TravelPlanner v2 评测: agent 照旧跑，judge 换成 judge_v2 + 多票合并。

跟 travel_pilot_eval.py 的关系:

  travel_pilot_eval.py  旧口径，跟队友那套 pilot 流水线逐字一致，不要改
  这个脚本              v2 口径，附候选池、按轮判、多票、契约外违规计分

agent 这一侧完全复用 human_annotated_pilot.build_agent_prompt，所以两套跑出来的
agent 输出是同一份（命中同一个缓存），差异全部来自 judge 和计分。想对照的话把同一
份数据用两个脚本各跑一次即可。

judge 调用数 = 轮数 × 票数（旧版是 instance 数），成本会明显上去，先用 --dry-run 估。

用法:
    python annotation/tools/travel_eval_v2.py DATA.json \\
        --agent-model qwen/qwen3-235b-a22b-2507 \\
        --judge-model openai/gpt-5.6-sol --votes 3 --output out.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from eval.agent_v2 import agent_feasibility, build_agent_prompt_v2  # noqa: E402
from eval.human_annotated_pilot import (  # noqa: E402
    build_agent_prompt, constraint_field_vocabulary)
from eval.judge_v2 import build_judge_prompt_v2, merge_votes  # noqa: E402
from eval.pool_evidence import candidate_pool, pool_size_note  # noqa: E402
from eval.score_v2 import aggregate_v2, score_turn_v2  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'llm_intent_eval', Path(__file__).with_name('llm_intent_eval.py'))
lie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lie)

_tp = importlib.util.spec_from_file_location(
    'travel_pilot_eval', Path(__file__).with_name('travel_pilot_eval.py'))
tpe = importlib.util.module_from_spec(_tp)
_tp.loader.exec_module(tpe)

DOMAIN = 'travelplanner'


def touched_fields(instance: Dict[str, Any], turn_id: int) -> List[str]:
    """到这一轮为止用户真动过的字段。背景字段一旦被改过就不再算背景。"""
    out: List[str] = []
    for turn in instance.get('turns') or []:
        if int(turn.get('turn_id', -1)) > turn_id:
            break
        for field, change in (turn.get('gold_delta') or {}).items():
            if isinstance(change, dict) and change.get('op') != 'reprioritize':
                out.append(str(field))
    return out


def run_agent(instances, vocabulary, client, workers):
    """v2 的 agent 跑法: 和 v1 同一套解析，但 prompt 多一个 feasibility 通道。

    v1 的 prompt 一个字都不能动（团队拿它做对比，而且 OpenRouter 的缓存键是
    sha256(model + prompt)），所以这里复制 tpe.run_agent 的流程而不是改它。
    差别只有两处: 换成 build_agent_prompt_v2，以及把 feasibility 块带进每一行。
    """
    jobs = [(i, j) for i, inst in enumerate(instances)
            for j in range(len(inst.get('turns') or []))]
    results: List[Any] = [None] * len(jobs)

    def work(k: int) -> None:
        i, j = jobs[k]
        results[k] = client.complete(build_agent_prompt_v2(
            instance=instances[i], turn_index=j, field_vocabulary=vocabulary))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(len(jobs))))

    stats = {'no_json': 0, 'empty_constraints': 0, 'bad_action': 0, 'no_feasibility': 0}
    by_instance: Dict[int, List[Dict[str, Any]]] = {}
    for k, (i, j) in enumerate(jobs):
        raw = results[k]
        if raw is None:
            stats['no_json'] += 1
        understanding, has_constraints = tpe.lenient_understanding(raw)
        if not has_constraints:
            stats['empty_constraints'] += 1
        action, action_ok = tpe.valid_travel_action(raw)
        if not action_ok:
            stats['bad_action'] += 1
        feasibility = agent_feasibility(raw)
        if not feasibility['declared']:
            stats['no_feasibility'] += 1
        turn = (instances[i].get('turns') or [])[j]
        by_instance.setdefault(i, []).append({
            'domain': DOMAIN,
            'instance_id': str(instances[i].get('instance_id')),
            'turn_id': int(turn.get('turn_id', j)),
            'user_utterance': turn.get('user_utterance'),
            'gold_intention': copy.deepcopy(turn.get('gold_current_intention') or {}),
            'agent_intention_prediction': understanding,
            'action_evidence': {'action': action},
            'agent_feasibility': feasibility,
        })
    grouped = [(str(instances[i].get('instance_id')), by_instance[i])
               for i in sorted(by_instance)]
    return grouped, stats


def judge_turn(job: Dict[str, Any], client: Any, votes: int,
               ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    gold_fields = [str(f) for f, v in (job['gold_intention'].get('constraints') or {}).items()
                   if v is not None]
    raw_votes = []
    for index in range(votes):
        prompt = build_judge_prompt_v2(
            domain=DOMAIN,
            instance_id=job['instance_id'],
            turn_id=job['turn_id'],
            all_utterances=job['all_utterances'],
            gold_intention=job['gold_intention'],
            changed_this_turn=job['changed_this_turn'],
            agent_intention_prediction=job['agent_intention_prediction'],
            action=job['action'],
            candidate_pool=job['pool'],
            sample_index=index,
        )
        raw = client.complete(prompt)
        if isinstance(raw, dict):
            raw_votes.append(raw)
    if not raw_votes:
        return None, 'judge 没有返回合法 JSON'
    merged = merge_votes(raw_votes, gold_fields)
    seen = {str(r.get('gold_field')) for r in merged.get('constraint_judgments') or []}
    missing = sorted(set(gold_fields) - seen)
    if missing:
        return None, f'judge 漏判字段: {missing}'
    return merged, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--agent-model', required=True)
    parser.add_argument('--judge-model', default=None, help='默认同 --agent-model')
    parser.add_argument('--votes', type=int, default=3,
                        help='每轮判定的采样票数，多数合并。布尔可以投票，浮点不行 —— '
                             'v2 的 judge 不输出任何浮点，正是为了让投票有意义。')
    parser.add_argument('--only-instances', default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--judge-start-tokens', type=int, default=8000,
                        help='按轮判，单次输出比整条 instance 判短得多，起点可以调低。')
    parser.add_argument('--judge-max-tokens', type=int, default=48000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    lie.load_dotenv(ROOT / '.env.llm')
    judge_model = args.judge_model or args.agent_model
    instances = json.loads(args.dataset.read_text(encoding='utf-8'))
    if not isinstance(instances, list):
        raise SystemExit(f'{args.dataset} 不是 JSON 列表')
    vocabulary = constraint_field_vocabulary(instances)
    if args.only_instances:
        wanted = {v.strip() for v in args.only_instances.split(',') if v.strip()}
        instances = [i for i in instances if str(i.get('instance_id')) in wanted]
        missing = wanted - {str(i.get('instance_id')) for i in instances}
        if missing:
            raise SystemExit(f'找不到 instance: {sorted(missing)}')
    turns = sum(len(i.get('turns') or []) for i in instances)

    pools = {str(i['instance_id']): candidate_pool(i) for i in instances}
    if args.dry_run:
        first = instances[0]
        print(f'{len(instances)} 个 instance / {turns} 轮')
        print(f'judge 调用 = {turns} 轮 × {args.votes} 票 = {turns * args.votes} 次'
              f'（旧版是 {len(instances)} 次）')
        for iid, pool in list(pools.items())[:3]:
            print(f'  {iid} 候选池 {pool_size_note(pool)}')
        sample = build_judge_prompt_v2(
            domain=DOMAIN, instance_id=str(first['instance_id']), turn_id=1,
            all_utterances=[t.get('user_utterance') for t in first['turns'][:2]],
            gold_intention=first['turns'][1].get('gold_current_intention') or {},
            changed_this_turn=first['turns'][1].get('gold_delta') or {},
            agent_intention_prediction={}, action={},
            candidate_pool=pools[str(first['instance_id'])])
        print(f'\njudge prompt 长度 ~{len(sample)} 字符（~{len(sample)//3.5:.0f} token）')
        print(f'估算 judge 输入 {turns * args.votes * len(sample) / 3.5 / 1e6:.2f}M token')
        print('\n=== judge prompt 示例（截断 3000 字符）===')
        print(sample[:3000])
        return

    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise SystemExit('OPENROUTER_API_KEY 未设置（放在 .env.llm）')

    agent_client = lie.Client(api_key, args.agent_model)
    judge_client = lie.Client(api_key, judge_model)
    os.environ.setdefault('LLM_START_OUTPUT_TOKENS', str(args.judge_start_tokens))
    os.environ.setdefault('LLM_MAX_OUTPUT_TOKENS', str(args.judge_max_tokens))

    print(f'[agent] {turns} 轮，模型 {args.agent_model} ...', flush=True)
    grouped, stats = run_agent(instances, vocabulary, agent_client, args.workers)
    by_id = {str(i['instance_id']): i for i in instances}

    jobs: List[Dict[str, Any]] = []
    for instance_id, rows in grouped:
        instance = by_id[instance_id]
        utterances = [str(t.get('user_utterance') or '') for t in instance['turns']]
        for row in rows:
            turn = next(t for t in instance['turns']
                        if int(t.get('turn_id', -1)) == int(row['turn_id']))
            jobs.append({
                'instance_id': instance_id,
                'turn_id': int(row['turn_id']),
                'user_utterance': row['user_utterance'],
                'all_utterances': [u for u in utterances[:int(row['turn_id']) + 1] if u],
                'gold_intention': row['gold_intention'],
                'changed_this_turn': copy.deepcopy(turn.get('gold_delta') or {}),
                'agent_intention_prediction': row['agent_intention_prediction'],
                'action': (row['action_evidence'] or {}).get('action') or {},
                'pool': pools[instance_id],
                'touched': touched_fields(instance, int(row['turn_id'])),
                # t0 带进来的字段就是"没人提过就不该计分"的候选集合
                'baseline': sorted((instance['turns'][0].get('gold_current_intention') or {})
                                   .get('constraints') or {}),
                # 池子能做到的最好结果，second_best.py 枚举出来写进标注的
                'world_feasibility': copy.deepcopy(
                    (turn.get('gold_action') or {}).get('world_feasibility')),
                'agent_feasibility': row.get('agent_feasibility') or {},
            })

    print(f'[judge] {len(jobs)} 轮 × {args.votes} 票，模型 {judge_model} ...', flush=True)
    results: List[Any] = [None] * len(jobs)

    def work(k: int) -> None:
        results[k] = judge_turn(jobs[k], judge_client, args.votes)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, range(len(jobs))))

    rows: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []
    for job, (judgment, error) in zip(jobs, results):
        label = f"{job['instance_id']} t{job['turn_id']}"
        if error or judgment is None:
            failures.append((label, error or 'unknown'))
            continue
        try:
            scores = score_turn_v2(
                gold_intention=job['gold_intention'],
                changed_this_turn=job['changed_this_turn'],
                agent_intention_prediction=job['agent_intention_prediction'],
                judgment=judgment,
                touched_fields=job['touched'],
                baseline_fields=job['baseline'],
                world_feasibility=job['world_feasibility'],
                agent_feasibility=job['agent_feasibility'])
        except Exception as exc:  # noqa: BLE001
            failures.append((label, str(exc)))
            continue
        rows.append({
            'domain': DOMAIN,
            'instance_id': job['instance_id'],
            'turn_id': job['turn_id'],
            'user_utterance': job['user_utterance'],
            'gold_intention': job['gold_intention'],
            'changed_this_turn': job['changed_this_turn'],
            'agent_intention_prediction': job['agent_intention_prediction'],
            'action': job['action'],
            'agent_feasibility': job['agent_feasibility'],
            'judgment': judgment,
            'scores': scores,
        })

    if not rows:
        print('\n没有任何轮次打分成功。', file=sys.stderr)
        for label, error in failures[:10]:
            print(f'  {label}: {error}', file=sys.stderr)
        raise SystemExit(1)

    agg = aggregate_v2(rows)
    print(f'\n数据集: {args.dataset.name}')
    print(f'agent: {args.agent_model}   judge: {judge_model} × {args.votes} 票')
    print(f'{agg["instances"]} 个 instance / 计分 {agg["turns"]} 轮'
          f'（共 {agg["turns_seen"]} 轮，{agg["turns_skipped_unchanged_only"]} 轮全是未变约束已排除；'
          f'另排除未变约束判定 {agg["excluded_unchanged_judgments"]} 条）\n')
    print('—— 主指标 ——')
    print(f'{"变更捕捉(轮级)":<26}{agg["change_capture_turn"]:>8.3f}   n={agg["change_turns"]}')
    print(f'{"变更捕捉(字段级)":<26}{agg["change_capture_field"]:>8.3f}')
    print(f'{"    仅值变化(对照口径)":<26}{agg["change_capture_value_field"]:>8.3f}')
    for op, value in (agg.get('change_capture_by_op') or {}).items():
        print(f'{"    " + op:<26}{value:>8.3f}   n={(agg.get("change_turns_by_op") or {}).get(op, 0)} 轮')
    print('\n—— 池子下限（second_best 枚举） ——')
    print(f'{"达到下限的轮次占比":<26}{agg["sacrifice_optimality_rate"]:>8.3f}   n={agg["sacrifice_turns"]} 轮')
    print(f'{"平均多让了几条 must-have":<26}{agg["sacrifice_mean_regret"]:>8.3f}')
    if agg.get('sacrifice_judge_alarms'):
        print(f'{"  !! 低于下限(judge 算错)":<26}{agg["sacrifice_judge_alarms"]:>8d} 轮')
    print('\n—— 不可行上报 ——')
    print(f'{"声明正确率":<26}{agg["declaration_accuracy"]:>8.3f}')
    print(f'{"    池子真不可行时说出来":<26}{agg["declaration_recall"]:>8.3f}')
    print(f'{"    可行轮误报率":<26}{agg["declaration_false_alarm_rate"]:>8.3f}')
    print(f'{"    归因正确率":<26}{agg["blame_accuracy"]:>8.3f}   n={agg["blame_turns"]} 轮')
    print('\n—— 意图 ——')
    for name, key in [('Constraint 识别率', 'constraint_recall'),
                      ('Constraint 值准确率', 'constraint_value_accuracy'),
                      ('Constraint 精确率', 'constraint_precision'),
                      ('背景字段复述(诊断)', 'background_echo')]:
        print(f'{name:<26}{agg[key]:>8.3f}')
    print('\n—— 优先级 ——')
    print(f'{"优先级一致性":<26}{agg["priority_concordance"]:>8.3f}   n={agg["priority_turns"]} 轮')
    for kind, value in (agg.get('priority_by_kind') or {}).items():
        print(f'{"    " + kind:<26}{value:>8.3f}')
    print('\n—— 动作 ——')
    for name, key in [('动作合规', 'action_score'),
                      ('硬违规率', 'hard_violation_rate'),
                      ('动作合规(含未变约束)', 'action_with_implicit_all'),
                      ('硬违规率(含未变约束)', 'hard_violation_rate_all'),
                      ('自相矛盾轮占比', 'inconsistency_turns'),
                      ('隐含违规轮占比', 'implicit_violation_turns')]:
        print(f'{name:<26}{agg[key]:>8.3f}')

    print(f'\nagent: 未返回 JSON {stats["no_json"]} 轮；空 constraints '
          f'{stats["empty_constraints"]} 轮；action 不合法 {stats["bad_action"]} 轮')
    print(f'Token: agent 入 {agent_client.prompt_tokens} 出 {agent_client.completion_tokens}；'
          f'judge 入 {judge_client.prompt_tokens} 出 {judge_client.completion_tokens}')
    print(f'调用失败: agent {agent_client.failures}，judge {judge_client.failures}')
    if failures:
        print(f'\n{len(failures)}/{len(jobs)} 轮打分失败:')
        for label, error in failures[:15]:
            print(f'  {label}: {error}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'metadata': {
                'dataset': str(args.dataset),
                'agent_model': args.agent_model,
                'judge_model': judge_model,
                'votes': args.votes,
                'scorer': 'score_v2.score_turn_v2',
                'agent_stats': stats,
                'failed_turns': failures,
            },
            'rows': rows,
            'aggregate': agg,
        }, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print(f'\n逐轮结果已写入 {args.output}')


if __name__ == '__main__':
    main()
