"""给标注好的 TravelPlanner 数据跑 agent + LLM judge 打分。

跟 judge_intent_eval.py 的分工：

  judge_intent_eval   WebShop 专用，agent 预测从 llm_intent_cache 读现成的
  这个脚本            TravelPlanner 专用，agent 和 judge 都要现跑

agent prompt、judge prompt、打分器全部用 src/eval/human_annotated_pilot.py 的原件
（build_agent_prompt / build_judge_prompt / score_judged_turn / aggregate_scored_rows），
所以口径和队友那套 pilot 流水线一致。

两轮调用都走 llm_intent_eval.Client，因此共用 annotation/data/llm_intent_cache/ 的
(模型, prompt) 缓存：中断后重跑、或只改了部分标注重跑，都不重复计费。

用法:
    python annotation/tools/travel_pilot_eval.py \\
        annotation/data/travelplanner_hard_participant_pilot_10_20260906_annotated.json \\
        --agent-model qwen/qwen3-235b-a22b-2507 --dry-run
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
from eval.human_annotated_pilot import (  # noqa: E402
    aggregate_scored_rows, build_agent_prompt, build_judge_prompt,
    constraint_field_vocabulary, score_judged_turn)

_spec = importlib.util.spec_from_file_location(
    'llm_intent_eval', Path(__file__).with_name('llm_intent_eval.py'))
lie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lie)

DOMAIN = 'travelplanner'


def lenient_understanding(raw: Any) -> Tuple[Dict[str, Any], bool]:
    """取出 current_intention_understanding。

    normalize_agent_output 会因为 action 不合法整条抛错；这里把两层拆开，
    动作层不合法不影响意图层继续打分，只做记录。
    """
    if not isinstance(raw, dict):
        return {'constraints': {}, 'priority': {'ranked_fields': []}, 'explanation': ''}, False
    understanding = raw.get('current_intention_understanding')
    if not isinstance(understanding, dict):
        return {'constraints': {}, 'priority': {'ranked_fields': []}, 'explanation': ''}, False
    constraints = understanding.get('constraints')
    priority = understanding.get('priority')
    ranked = priority.get('ranked_fields') if isinstance(priority, dict) else []
    return {
        'constraints': copy.deepcopy(constraints if isinstance(constraints, dict) else {}),
        'priority': {'ranked_fields': [str(v) for v in ranked] if isinstance(ranked, list) else []},
        'explanation': str(understanding.get('explanation') or '').strip(),
    }, isinstance(constraints, dict) and bool(constraints)


def valid_travel_action(raw: Any) -> Tuple[Dict[str, Any], bool]:
    action = raw.get('action') if isinstance(raw, dict) else None
    if not isinstance(action, dict):
        return {'action_type': 'plan', 'itinerary': []}, False
    ok = (str(action.get('action_type') or '').strip().lower() == 'plan'
          and isinstance(action.get('itinerary'), list))
    return copy.deepcopy(action), ok


def run_agent(instances: List[Dict[str, Any]], vocabulary: List[str], client: Any,
              workers: int) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], Dict[str, int]]:
    jobs: List[Tuple[int, int]] = []
    for i, inst in enumerate(instances):
        for j in range(len(inst.get('turns') or [])):
            jobs.append((i, j))
    results: List[Any] = [None] * len(jobs)

    def work(k: int) -> None:
        i, j = jobs[k]
        prompt = build_agent_prompt(domain=DOMAIN, instance=instances[i],
                                    turn_index=j, field_vocabulary=vocabulary)
        results[k] = client.complete(prompt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(len(jobs))))

    stats = {'no_json': 0, 'empty_constraints': 0, 'bad_action': 0}
    by_instance: Dict[int, List[Dict[str, Any]]] = {}
    for k, (i, j) in enumerate(jobs):
        raw = results[k]
        if raw is None:
            stats['no_json'] += 1
        understanding, has_constraints = lenient_understanding(raw)
        if not has_constraints:
            stats['empty_constraints'] += 1
        action, action_ok = valid_travel_action(raw)
        if not action_ok:
            stats['bad_action'] += 1
        turn = (instances[i].get('turns') or [])[j]
        by_instance.setdefault(i, []).append({
            'domain': DOMAIN,
            'instance_id': str(instances[i].get('instance_id')),
            'turn_id': int(turn.get('turn_id', j)),
            'user_utterance': turn.get('user_utterance'),
            'gold_intention': copy.deepcopy(turn.get('gold_current_intention') or {}),
            'agent_intention_prediction': understanding,
            'action_evidence': {'action': action},
        })
    grouped = [(str(instances[i].get('instance_id')), by_instance[i])
               for i in sorted(by_instance)]
    return grouped, stats


ALLOWED_STATUS = {'satisfied', 'violated', 'unknown'}


def sanitize_judgment(judgment: Dict[str, Any], gold_fields: set,
                      counters: Dict[str, int]) -> Dict[str, Any]:
    """把 judge 输出里两类可修复的偏差拉回 score_judged_turn 能接受的形状。

    - 丢掉 gold 里没有的多余字段（打分器本来就只遍历 gold 字段，丢掉不影响任何分数）
    - 把 satisfied/violated/unknown 之外的 action_status 归到 unknown
      （unknown 和 violated 在动作分上同样是 0 分，区别只在 hard violation
        不会被触发，所以这是偏保守的一侧）
    """
    rows = judgment.get('constraint_judgments')
    if not isinstance(rows, list):
        return judgment
    kept = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get('gold_field') or '') not in gold_fields:
            counters['dropped_extra_field'] += 1
            continue
        status = str(row.get('action_status') or '').strip().lower()
        if status not in ALLOWED_STATUS:
            counters['status_to_unknown'] += 1
            row = dict(row)
            row['action_status'] = 'unknown'
        kept.append(row)
    out = dict(judgment)
    out['constraint_judgments'] = kept
    return out


def judge_instance(instance_id: str, rows: List[Dict[str, Any]], client: Any,
                   lenient: bool = False,
                   counters: Optional[Dict[str, int]] = None,
                   retries: int = 0,
                   ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """judge 偶尔会漏掉 gold 字段或漏掉整轮，这属于瞬时不合规。

    retries > 0 时对这类失败绕过缓存重试；prompt 一字不改，只是再问一次。
    """
    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        scored, error = _judge_once(instance_id, rows, client, lenient, counters,
                                    use_cache=(attempt == 0))
        if error is None:
            if attempt and counters is not None:
                counters['judge_retried'] = counters.get('judge_retried', 0) + 1
            return scored, None
        last_error = error
        # 只对"契约没满足"重试；模型压根没返回 JSON 那种重试也没用。
        if 'mismatch' not in error and '漏了' not in error:
            break
    return [], last_error


def _judge_once(instance_id: str, rows: List[Dict[str, Any]], client: Any,
                lenient: bool, counters: Optional[Dict[str, int]],
                use_cache: bool) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    payload = [{
        'turn_id': row['turn_id'],
        'user_utterance': row['user_utterance'],
        'gold_current_intention': row['gold_intention'],
        'agent_intention_prediction': row['agent_intention_prediction'],
        'action_evidence': row['action_evidence'],
    } for row in rows]
    raw = client.complete(build_judge_prompt(
        domain=DOMAIN, instance_id=instance_id, judged_turns=payload),
        use_cache=use_cache)
    if not isinstance(raw, dict) or not isinstance(raw.get('turns'), list):
        return [], 'judge 没有返回合法的 turns 列表'
    by_turn = {}
    for item in raw['turns']:
        if isinstance(item, dict) and item.get('turn_id') is not None:
            try:
                by_turn[int(item['turn_id'])] = item
            except (TypeError, ValueError):
                continue
    scored = []
    for row in rows:
        judgment = by_turn.get(row['turn_id'])
        if judgment is None:
            return [], f'judge 漏了 turn {row["turn_id"]}'
        if lenient:
            gold_fields = {str(field) for field, value
                           in (row['gold_intention'].get('constraints') or {}).items()
                           if value is not None}
            judgment = sanitize_judgment(judgment, gold_fields,
                                         counters if counters is not None else {})
        try:
            new_row = dict(row)
            new_row['scores'] = score_judged_turn(
                gold_intention=row['gold_intention'], judgment=judgment)
        except Exception as exc:  # noqa: BLE001
            return [], f'turn {row["turn_id"]}: {exc}'
        scored.append(new_row)
    return scored, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--agent-model', required=True)
    parser.add_argument('--judge-model', default=None, help='默认同 --agent-model')
    parser.add_argument('--output', type=Path, default=None,
                        help='把逐轮打分结果写成 JSON')
    parser.add_argument('--only-instances', default=None,
                        help='逗号分隔的 instance_id，只评这些。字段词表仍按整个文件计算，'
                             '所以 prompt 和全量跑时逐字相同，结果可直接比较。')
    parser.add_argument('--judge-start-tokens', type=int, default=16000,
                        help='judge 输出 token 起点。判定要覆盖整个 instance 的所有轮次，'
                             '约束多的 instance 在 8000 起点上会被截断，实测 16000 起可省掉大部分补跑。')
    parser.add_argument('--judge-max-tokens', type=int, default=96000)
    parser.add_argument('--judge-retries', type=int, default=1,
                        help='judge 漏字段/漏轮时绕过缓存重试的次数（prompt 不变）')
    parser.add_argument('--lenient-judge', action='store_true',
                        help='丢掉 judge 编出来的多余字段，并把非法 action_status 归到 unknown。'
                             '会逐条计数报告，默认关闭。')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    lie.load_dotenv(ROOT / '.env.llm')
    judge_model = args.judge_model or args.agent_model
    instances = json.loads(args.dataset.read_text(encoding='utf-8'))
    if not isinstance(instances, list):
        raise SystemExit(f'{args.dataset} 不是 JSON 列表')
    # 词表先按整份文件算，再做过滤，这样子集跑出来的 prompt 和全量跑逐字一致。
    vocabulary = constraint_field_vocabulary(instances)
    if args.only_instances:
        wanted = {value.strip() for value in args.only_instances.split(',') if value.strip()}
        instances = [inst for inst in instances
                     if str(inst.get('instance_id')) in wanted]
        missing = wanted - {str(inst.get('instance_id')) for inst in instances}
        if missing:
            raise SystemExit(f'找不到 instance: {sorted(missing)}')
        print(f'只评 {len(instances)} 个 instance；字段词表仍为全量 {len(vocabulary)} 个',
              file=sys.stderr)
    turns = sum(len(inst.get('turns') or []) for inst in instances)

    if args.dry_run:
        prompt = build_agent_prompt(domain=DOMAIN, instance=instances[0],
                                    turn_index=1, field_vocabulary=vocabulary)
        print('=== agent prompt 示例（截断到 2500 字符）===')
        print(prompt[:2500])
        avg = len(prompt) / 3.5
        agent_in = avg * turns
        judge_in = avg * 1.4 * len(instances)
        print(f'\n=== 估算 ===\nagent {args.agent_model}   judge {judge_model}')
        print(f'{len(instances)} 个 instance / {turns} 轮'
              f'（agent 每轮一次，judge 每 instance 一次）')
        print(f'canonical 字段词表 {len(vocabulary)} 个')
        print(f'按 $0.22/M 输入 + $0.88/M 输出估算：'
              f'agent ~${(agent_in * 0.22 + turns * 900 * 0.88) / 1e6:.2f}，'
              f'judge ~${(judge_in * 0.22 + len(instances) * 2500 * 0.88) / 1e6:.2f}')
        return

    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        print('没有读到 OPENROUTER_API_KEY。填进 .env.llm。', file=sys.stderr)
        raise SystemExit(1)
    workers = int(os.environ.get('OPENROUTER_CONCURRENCY', '6'))

    print(f'[agent] {turns} 轮，模型 {args.agent_model} ...', file=sys.stderr)
    agent_client = lie.Client(api_key, args.agent_model)
    grouped, stats = run_agent(instances, vocabulary, agent_client, workers)

    print(f'[judge] {len(grouped)} 个 instance，模型 {judge_model} ...', file=sys.stderr)
    os.environ['LLM_START_OUTPUT_TOKENS'] = str(args.judge_start_tokens)
    os.environ['LLM_MAX_OUTPUT_TOKENS'] = str(args.judge_max_tokens)
    os.environ['LLM_MAX_ATTEMPTS'] = '5'
    judge_client = lie.Client(api_key, judge_model)
    counters = {'dropped_extra_field': 0, 'status_to_unknown': 0, 'judge_retried': 0}
    results: List[Any] = [None] * len(grouped)

    def work(i: int) -> None:
        instance_id, rows = grouped[i]
        results[i] = judge_instance(instance_id, rows, judge_client,
                                    lenient=args.lenient_judge, counters=counters,
                                    retries=args.judge_retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(len(grouped))))

    rows: List[Dict[str, Any]] = []
    failures = []
    for (instance_id, _), (scored, error) in zip(grouped, results):
        if error:
            failures.append((instance_id, error))
        else:
            rows.extend(scored)

    if not rows:
        print('\n没有任何 instance 打分成功。', file=sys.stderr)
        for instance_id, error in failures:
            print(f'  {instance_id}: {error}', file=sys.stderr)
        raise SystemExit(1)

    aggregate = aggregate_scored_rows(rows)
    overall = aggregate['overall']
    print(f'\n数据集: {args.dataset.name}')
    print(f'agent: {args.agent_model}   judge: {judge_model}'
          f'   打分器: human_annotated_pilot.score_judged_turn')
    print()
    for name, key in [('意图理解', 'intention_understanding_score'),
                      ('Constraint准确率', 'constraint_value_accuracy'),
                      ('Priority顺序', 'priority_order_score'),
                      ('动作合规', 'action_compliance_score'),
                      ('Hard violation', 'hard_priority_violation_rate')]:
        print(f'{name:<22}{overall[key]:>10.3f}')
    print(f'{"instances":<22}{overall["instances"]:>10d}')
    print(f'{"轮数":<22}{overall["turns"]:>10d}')

    print(f'\nagent: 未返回 JSON {stats["no_json"]} 轮；预测出空 constraints '
          f'{stats["empty_constraints"]} 轮；action 不合法 {stats["bad_action"]} 轮')
    print(f'Token: agent 输入 {agent_client.prompt_tokens} 输出 {agent_client.completion_tokens}'
          f'；judge 输入 {judge_client.prompt_tokens} 输出 {judge_client.completion_tokens}')
    print(f'调用失败: agent {agent_client.failures}，judge {judge_client.failures}')
    if args.lenient_judge:
        print(f'宽松模式修正: 丢掉 judge 多余字段 {counters["dropped_extra_field"]} 条；'
              f'非法 action_status 归到 unknown {counters["status_to_unknown"]} 条')
    if counters.get('judge_retried'):
        print(f'judge 因漏字段/漏轮绕过缓存重试并成功的 instance: {counters["judge_retried"]} 个')
    if failures:
        print(f'\n{len(failures)}/{len(grouped)} 个 instance 打分失败，已排除：')
        for instance_id, error in failures:
            print(f'  {instance_id}: {error}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'metadata': {
                'dataset': str(args.dataset),
                'agent_model': args.agent_model,
                'judge_model': judge_model,
                'scorer': 'human_annotated_pilot.score_judged_turn',
                'agent_stats': stats,
                'lenient_judge': args.lenient_judge,
                'lenient_corrections': counters if args.lenient_judge else None,
                'failed_instances': failures,
            },
            'rows': rows,
            'aggregate': aggregate,
        }, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print(f'\n逐轮结果已写入 {args.output}')


if __name__ == '__main__':
    main()
