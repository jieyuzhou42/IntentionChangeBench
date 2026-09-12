"""用真模型做意图预测，再用仓里的 evaluator 打分。

跟 src/eval/run_benchmark.py 的关系：

  run_benchmark   utterance + 真实 WebShop 页面 -> LLM -> 动作 + intention
                  -> state_understanding_eval + action_selection_eval
  这个脚本        utterance + 当轮候选商品清单   -> LLM -> intention
                  -> state_understanding_eval

也就是只复现"意图理解"那一半（领导表里的 Constraint准确率 / Priority顺序 / 意图理解
三列），不跑 rollout，所以出不了动作合规和 Hard violation。换来的好处是不需要
Lucene 索引、pyserini、gym 和 py38 环境，一个 requests 就能跑。

打分用的是 src/eval/evaluators/constraint_importance_eval.py 里的
evaluate_state_understanding 本体，不是重写的，所以归一化规则和 3/2/1 加权
跟真 eval 完全一致。

prompt 尽量贴 fixed_user_llm_executor._build_prompt：同样给
trajectory_user_utterances（最近 8 条）+ current_user_utterance，
同样的 current_intention_understanding 输出 schema。区别是页面换成了
当轮候选商品的标题和价格（真页面拿不到），可以用 --candidates none 关掉。

响应会缓存到 annotation/data/llm_intent_cache/，同样的 (模型, prompt) 不会重复计费，
所以中断后重跑、或者改完标注只重跑受影响的分片都很便宜。

用法:
    # 先看 prompt 长什么样、估个价，不花钱
    python annotation/tools/llm_intent_eval.py --dry-run 'new=<glob>'

    # 小规模试跑，确认模型能稳定吐 JSON
    python annotation/tools/llm_intent_eval.py --limit 20 'new=<glob>'

    # 正式跑，改前改后对比
    python annotation/tools/llm_intent_eval.py \
        '改前=/tmp/old021/shard_021_human_annotated.json,/tmp/old022.json' \
        '改后=data/simulation/.../shard_021_human_annotated.json,...'
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'eval' / 'evaluators'))
from constraint_importance_eval import (  # noqa: E402
    evaluate_state_understanding, evaluate_action_selection)

sys.path.insert(0, str(ROOT / 'src'))
try:
    from domains.webshop.environment import WebShopEnvAdapter  # noqa: E402
    _CHECKER = WebShopEnvAdapter(None)
except Exception:  # noqa: BLE001
    _CHECKER = None


def action_scores(turn, gold, selected_asin):
    """用仓里真实的 _check_constraints 判定选中商品，再走 evaluate_action_selection。

    跟 run_benchmark 的区别只在于商品来自当轮候选池而不是 agent 自己搜出来的页面。
    判定逻辑本身是同一份代码：只认 budget_max / color / size / brand / category
    和 avoid_* 六类字段，其余一律算 unknown（0 分但仍占分母）。
    """
    if _CHECKER is None or not selected_asin:
        return None
    items = (turn.get('env_feedback') or {}).get('candidate_items') or []
    item = next((c for c in items if str(c.get('asin')) == str(selected_asin)), None)
    if item is None:
        return None
    satisfied, violated, _ = _CHECKER._check_constraints(item, gold, include_debug=True)
    payload = {
        'gold_current_intention': gold,
        'env_feedback': {
            'gold_eval_satisfied_constraints': satisfied,
            'gold_eval_violated_constraints': violated,
            'selected_asin': selected_asin,
        },
    }
    result = evaluate_action_selection(payload)
    highs = [pc for pc in result['per_constraint'] if pc['importance'] == 'high']
    hard = sum(1 for pc in highs if pc['status'] == 'violated')
    return {'weighted': result['weighted_score'],
            'hard_violated': hard, 'hard_total': len(highs)}

CACHE_DIR = ROOT / 'annotation' / 'data' / 'llm_intent_cache'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
LEVELS = ('high', 'medium', 'low')
MAX_UTTERANCES = 8

INSTRUCTIONS = """
You are the intention-tracking component of a WebShop agent.
The user is shopping across several turns and keeps changing what they want.
Read the whole conversation so far and state what the user wants RIGHT NOW.

Rules:
- The current intention is cumulative. Requirements from earlier turns still apply
  unless the user has relaxed or replaced them.
- Use short snake_case field names and short values. One field holds one idea.
- Put a field in "high" when it is a hard requirement the user would not buy without,
  "medium" when it is a real preference, "low" when the user has said it barely matters.
- Every field in "constraints" must appear in exactly one priority list.

- Also pick the single item on screen you would buy for this shopper right now.

Return one JSON object only, no prose and no code fences:
{
  "constraints": {"field_name": "value"},
  "priority": {"high": [], "medium": [], "low": []},
  "selected_asin": "asin of the item you would buy"
}
""".strip()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip())


def load_instances(spec: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pattern in spec.split(','):
        pattern = pattern.strip()
        if not pattern:
            continue
        for path in sorted(glob.glob(pattern)) or [pattern]:
            out += json.loads(Path(path).read_text(encoding='utf-8'))
    return out


def gold_intention(turn: Dict[str, Any]) -> Dict[str, Any]:
    gi = turn.get('gold_current_intention') or {}
    constraints = {k: v for k, v in (gi.get('constraints') or {}).items() if v is not None}
    priority = gi.get('priority') or {}
    return {
        'constraints': constraints,
        'priority': {lv: list(priority.get(lv) or []) for lv in LEVELS},
    }


def field_vocabulary(instances: List[Dict[str, Any]]) -> List[str]:
    """整个数据集里出现过的 gold 字段名并集。

    不是本条 instance 的字段名，所以不泄露"这个用户关心哪些维度"，
    只是把命名习惯统一掉。不给这个词表的话，模型会自己发明
    fragrance_type / price_lower_than 这类名字，而 gold 写的是
    category / budget_max，打分器按字段名精确匹配，直接 0 分。
    """
    names = set()
    for inst in instances:
        for turn in inst.get('turns') or []:
            gi = turn.get('gold_current_intention') or {}
            names.update((gi.get('constraints') or {}).keys())
    return sorted(names)


def build_prompt(turns: List[Dict[str, Any]], index: int, show_candidates: bool,
                 vocabulary: Optional[List[str]] = None,
                 seed_state: bool = False) -> str:
    history = [str(t.get('user_utterance') or '') for t in turns[: index + 1]]
    context: Dict[str, Any] = {
        'trajectory_user_utterances': history[-MAX_UTTERANCES:],
        'current_user_utterance': history[-1],
        'turn_index': index,
    }
    if show_candidates:
        items = (turns[index].get('env_feedback') or {}).get('candidate_items') or []
        context['items_on_screen'] = [
            {'asin': c.get('asin'), 'title': str(c.get('title') or '')[:110],
             'price': c.get('price')}
            for c in items[:10]
        ]
    extra = ''
    if seed_state and index > 0:
        # 模拟"为省 token 把上一轮状态直接塞进 prompt"的 harness 设计。
        # 注意这里塞的是上一轮的 gold，不是模型自己上一轮的输出，
        # 所以它同时也把 gold 的字段名和取值泄露给了模型。
        context['intention_state_before_this_turn'] = gold_intention(turns[index - 1])
        extra += ('\n- intention_state_before_this_turn is where the shopper stood after the'
                  ' previous turn. Start from it and apply only what the current utterance'
                  ' changes. Return the full updated state, not a diff.')
    if vocabulary:
        context['allowed_field_names'] = vocabulary
        extra += ('\n- Field names MUST be chosen from allowed_field_names. It is a shared'
                 ' vocabulary for this whole benchmark, so most of it is irrelevant to this'
                 ' shopper. Use only the ones that apply. Never invent a new field name.')
    return (f"{INSTRUCTIONS}{extra}\n\nCONTEXT_JSON:\n"
            f"{json.dumps(context, ensure_ascii=False, indent=1)}")


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r'^\s*```(?:json)?|```\s*$', '', str(text).strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def normalize_prediction(raw: Any) -> Dict[str, Any]:
    """把模型输出整理成 evaluator 认识的形状。字段名/值一律不改，交给打分器归一化。"""
    if not isinstance(raw, dict):
        return {'constraints': {}, 'priority': {lv: [] for lv in LEVELS}}
    inner = raw.get('current_intention_understanding')
    if isinstance(inner, dict):
        raw = inner
    constraints = raw.get('constraints')
    constraints = constraints if isinstance(constraints, dict) else {}
    priority = raw.get('priority')
    if isinstance(priority, dict):
        priority = {lv: list(priority.get(lv) or []) for lv in LEVELS}
    elif isinstance(priority, list):
        priority = priority
    else:
        priority = {lv: [] for lv in LEVELS}
    return {'constraints': constraints, 'priority': priority,
            'selected_asin': raw.get('selected_asin')}


class Client:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0
        # 有些模型（gpt-oss 系列）不允许关闭 reasoning，第一次 400 之后置位。
        self.reasoning_mandatory = False
        # OpenAI 的 json_object 模式要求 prompt 里出现 "json" 字样。与其改 prompt
        # （会影响跨模型可比性和缓存 key），不如退掉 response_format 靠 parse_json 兜底。
        self.json_mode_ok = True
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, prompt: str) -> Path:
        digest = hashlib.sha256(f'{self.model}\x00{prompt}'.encode()).hexdigest()[:32]
        return CACHE_DIR / f'{digest}.json'

    def complete(self, prompt: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        # use_cache=False 用于重试：judge 偶尔会漏掉 gold 字段，缓存住的话重试拿到的
        # 还是同一份残缺输出。新结果照常写回缓存。
        cache = self._cache_path(prompt)
        if use_cache and cache.exists():
            return json.loads(cache.read_text(encoding='utf-8')).get('parsed')

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/IntentionChangeBench',
            'X-Title': 'IntentionChangeBench offline intent eval',
        }
        # reasoning 必须显式关掉。qwen3.7-flash / gpt-5-nano 这类推理模型会把整个
        # max_tokens 预算烧在思考上，content 返回空字符串、finish_reason=length，
        # 看起来像"模型不会输出 JSON"，其实是被截断了。关掉之后同一个 prompt
        # 的成本从 $0.00039 降到 $0.00003。
        last_error = ''
        # judge prompt 要一次覆盖整个 instance 的所有轮次，输出比单轮意图预测长得多，
        # 所以起点和上限都可以按调用方调高。
        budget = int(os.environ.get('LLM_START_OUTPUT_TOKENS', '1500'))
        ceiling = int(os.environ.get('LLM_MAX_OUTPUT_TOKENS', '6000'))
        for attempt in range(int(os.environ.get('LLM_MAX_ATTEMPTS', '4'))):
            payload = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0,
                'max_tokens': budget,
            }
            if self.json_mode_ok:
                payload['response_format'] = {'type': 'json_object'}
            if not self.reasoning_mandatory:
                payload['reasoning'] = {'enabled': False}
            try:
                response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=90)
                if response.status_code in (429, 500, 502, 503, 529):
                    time.sleep(2 ** attempt)
                    last_error = f'HTTP {response.status_code}'
                    continue
                if response.status_code == 400 and self.json_mode_ok:
                    detail = ''
                    try:
                        error = response.json().get('error') or {}
                        detail = f"{error.get('message') or ''} {(error.get('metadata') or {}).get('raw') or ''}"
                    except Exception:  # noqa: BLE001
                        detail = response.text[:400]
                    if "contain the word 'json'" in detail:
                        self.json_mode_ok = False
                        last_error = 'json_object 模式要求 prompt 含 "json"，已改为不带 response_format 重试'
                        continue
                if response.status_code == 400 and not self.reasoning_mandatory:
                    # gpt-oss 这类模型强制开启 reasoning，关不掉。认下来重试一次，
                    # 代价是思考 token 也要计费，所以只对明确报这个错的模型生效。
                    detail = ''
                    try:
                        detail = str((response.json().get('error') or {}).get('message') or '')
                    except Exception:  # noqa: BLE001
                        detail = response.text[:200]
                    if 'reasoning' in detail.lower() and 'mandator' in detail.lower():
                        self.reasoning_mandatory = True
                        last_error = '该模型强制 reasoning，已改为保留 reasoning 重试'
                        continue
                response.raise_for_status()
                body = response.json()
                usage = body.get('usage') or {}
                with self.lock:
                    self.prompt_tokens += int(usage.get('prompt_tokens') or 0)
                    self.completion_tokens += int(usage.get('completion_tokens') or 0)
                choice = body['choices'][0]
                message = choice.get('message') or {}
                content = message.get('content') or ''
                parsed = parse_json(content)
                if parsed is None and not content.strip():
                    # 有些模型即使关了 reasoning 仍然把 JSON 放进 reasoning 字段
                    parsed = parse_json(message.get('reasoning')
                                        or message.get('reasoning_content') or '')
                if parsed is not None:
                    cache.write_text(json.dumps({'parsed': parsed}, ensure_ascii=False),
                                     encoding='utf-8')
                    return parsed
                if choice.get('finish_reason') == 'length':
                    # 确定性截断，重试同样的参数没有意义，加大预算再试一次
                    if budget < ceiling:
                        budget = min(budget * 2, ceiling)
                        last_error = 'finish_reason=length，已加大 max_tokens 重试'
                        continue
                    last_error = f'输出被截断且 max_tokens 已到 {budget}'
                else:
                    last_error = f'无法解析成 JSON: {content[:120]!r}'
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f'{type(exc).__name__}: {exc}'
                time.sleep(2 ** attempt)
        with self.lock:
            self.failures += 1
            if self.failures <= 5:
                print(f'  [warn] 放弃一次调用: {last_error}', file=sys.stderr)
            elif self.failures == 6:
                print('  [warn] 后续失败不再逐条打印', file=sys.stderr)
        return None


def collect_turns(instances: List[Dict[str, Any]], limit: Optional[int]) -> List[Tuple]:
    jobs = []
    for inst in instances:
        turns = inst.get('turns') or []
        for index, turn in enumerate(turns):
            gold = gold_intention(turn)
            if not gold['constraints']:
                continue
            jobs.append((inst.get('instance_id'), index, turns, gold))
    return jobs[:limit] if limit else jobs


def score(label: str, jobs: List[Tuple], client: Optional[Client],
          show_candidates: bool, vocabulary: Optional[List[str]] = None,
          seed_state: bool = False) -> Dict[str, Any]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(jobs)

    def work(i: int) -> None:
        _, index, turns, _ = jobs[i]
        prompt = build_prompt(turns, index, show_candidates, vocabulary, seed_state)
        results[i] = client.complete(prompt) if client else None

    if client:
        workers = int(os.environ.get('OPENROUTER_CONCURRENCY', '6'))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, range(len(jobs))))

    buckets = {'all': [], 't0': [], 'later': []}
    unparsed = 0
    no_pick = 0
    for i, (_, index, turns, gold) in enumerate(jobs):
        raw = results[i]
        if raw is None:
            unparsed += 1
        predicted = normalize_prediction(raw)
        s = evaluate_state_understanding(gold, predicted)
        act = action_scores(turns[index], gold, predicted.get('selected_asin'))
        if act is None:
            no_pick += 1
        row = (s['constraint_weighted_score'], s['priority_level_weighted_score'], act)
        buckets['all'].append(row)
        buckets['t0' if index == 0 else 'later'].append(row)

    def agg(rows):
        if not rows:
            return None
        c = sum(r[0] for r in rows) / len(rows)
        p = sum(r[1] for r in rows) / len(rows)
        acts = [r[2] for r in rows if r[2] is not None]
        action = sum(a['weighted'] for a in acts) / len(acts) if acts else float('nan')
        hv_n = sum(a['hard_violated'] for a in acts)
        hv_d = sum(a['hard_total'] for a in acts)
        return {'constraint': c, 'priority': p, 'combined': (c + p) / 2,
                'action': action, 'hard_violation': (hv_n / hv_d) if hv_d else float('nan'),
                'n': len(rows)}

    return {'label': label, 'unparsed': unparsed, 'no_pick': no_pick,
            **{k: agg(v) for k, v in buckets.items()}}


def carry_prev_reference(jobs: List[Tuple]) -> Dict[str, float]:
    rows = []
    for _, index, turns, gold in jobs:
        if index == 0:
            continue
        rows.append(evaluate_state_understanding(gold, gold_intention(turns[index - 1])))
    if not rows:
        return {}
    c = sum(r['constraint_weighted_score'] for r in rows) / len(rows)
    p = sum(r['priority_level_weighted_score'] for r in rows) / len(rows)
    return {'constraint': c, 'priority': p, 'combined': (c + p) / 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', help='<label>=<glob>[,<glob>...]')
    parser.add_argument('--model', default=None)
    parser.add_argument('--limit', type=int, default=None, help='只跑前 N 轮，用来小规模试跑')
    parser.add_argument('--candidates', choices=('titles', 'none'), default='titles',
                        help='是否把当轮候选商品的标题和价格放进 prompt')
    parser.add_argument('--seed-state', choices=('none', 'prev-gold'), default='none',
                        help="prev-gold: 把上一轮的 gold intention 放进 prompt 当起点，"
                             "复现'为省 token 直接传状态'的 harness 设计；"
                             "这会同时泄露 gold 的字段名和取值")
    parser.add_argument('--vocab', choices=('global', 'none'), default='global',
                        help="global: 把整份数据集的 gold 字段名并集给模型，统一命名习惯；"
                             "none: 让模型自己发明字段名，会因为命名对不上大量得 0 分")
    parser.add_argument('--dry-run', action='store_true', help='只打印 prompt 和成本估算，不调用')
    args = parser.parse_args()

    load_dotenv(ROOT / '.env.llm')
    model = args.model or os.environ.get('OPENROUTER_MODEL') or 'qwen/qwen3.7-flash'

    columns = []
    vocabulary: Optional[List[str]] = None
    all_instances: List[Dict[str, Any]] = []
    for spec in args.datasets:
        label, _, pattern = spec.partition('=')
        instances = load_instances(pattern or spec)
        all_instances += instances
        columns.append((label or spec, collect_turns(instances, args.limit)))
    if args.vocab == 'global':
        vocabulary = field_vocabulary(all_instances)

    if args.dry_run:
        label, jobs = columns[0]
        _, index, turns, _ = jobs[min(1, len(jobs) - 1)]
        prompt = build_prompt(turns, index, args.candidates == 'titles', vocabulary,
                              args.seed_state == 'prev-gold')
        print('=== prompt 示例 ===')
        print(prompt)
        total = sum(len(j) for _, j in columns)
        print(f'\n=== 估算 ===\n模型 {model}   字段词表 {args.vocab}'
              + (f'（{len(vocabulary)} 个字段名）' if vocabulary else ''))
        print(f'待打分轮数 {total}（{" + ".join(f"{l}:{len(j)}" for l, j in columns)}）')
        print(f'prompt 约 {len(prompt) // 4} token，输出上限 900 token')
        print('按 $0.05/M 输入 + $0.15/M 输出估算，成本约 '
              f'${total * (len(prompt) / 4 * 0.05 + 250 * 0.15) / 1e6:.4f}')
        print('\n确认无误后去掉 --dry-run。响应会缓存，重跑不重复计费。')
        return

    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        print('没有读到 OPENROUTER_API_KEY。把 key 填进 .env.llm 的 OPENROUTER_API_KEY= 后面。',
              file=sys.stderr)
        raise SystemExit(1)

    client = Client(api_key, model)
    reports = []
    for label, jobs in columns:
        print(f'[{label}] {len(jobs)} 轮，模型 {model} ...', file=sys.stderr)
        reports.append((score(label, jobs, client, args.candidates == 'titles', vocabulary,
                             args.seed_state == 'prev-gold'),
                        carry_prev_reference(jobs)))

    width = max(16, *(len(r['label']) + 2 for r, _ in reports))
    print(f'\n模型: {model}   候选商品: {args.candidates}   字段词表: {args.vocab}'
          f'   起点状态: {args.seed_state}'
          + (f'（{len(vocabulary)} 个）' if vocabulary else ''))
    print('注意: 这里只有意图理解两列，没有跑 rollout，所以没有动作合规和 Hard violation。\n')
    header = ''.join(f'{r["label"]:>{width}}' for r, _ in reports)
    print(f'{"":16}{"metric":16}{header}')
    for bucket, name in (('all', '全部轮次'), ('t0', '仅 t0'), ('later', '仅非首轮')):
        for metric, cn in (('combined', '意图理解'), ('constraint', 'Constraint准确率'),
                           ('priority', 'Priority顺序'), ('action', '动作合规'),
                           ('hard_violation', 'Hard violation')):
            cells = ''.join(
                f'{(r[bucket] or {}).get(metric, float("nan")):>{width}.3f}' for r, _ in reports)
            print(f'{name if metric == "combined" else "":16}{cn:16}{cells}')
        print(f'{"":16}{"轮数":16}'
              + ''.join(f'{(r[bucket] or {}).get("n", 0):>{width}}' for r, _ in reports))
    print(f'\n{"参考":16}{"carry_prev 意图理解":16}'
          + ''.join(f'{ref.get("combined", float("nan")):>{width}.3f}' for _, ref in reports))
    print('  （carry_prev = 把上一轮 gold 原样端出来，不需要模型。模型分低于它 = 数据白送的比模型强）')
    print(f'\nToken: 输入 {client.prompt_tokens} 输出 {client.completion_tokens}；'
          f'调用失败 {client.failures}；未解析出 JSON 的轮次见上表 unparsed')
    for r, _ in reports:
        if r.get('no_pick'):
            print(f'  [{r["label"]}] 有 {r["no_pick"]} 轮没选出候选池里的商品，未计入动作两列')
        if r['unparsed']:
            print(f'  [{r["label"]}] 有 {r["unparsed"]} 轮没拿到合法 JSON，已按 0 分计入')


if __name__ == '__main__':
    main()
