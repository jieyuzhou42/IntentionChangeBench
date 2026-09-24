#!/usr/bin/env python3
"""TravelPlanner evaluation v2 over saved agent trajectories.

    prepare    build the frozen per-turn baseline (one judge call per gold turn)
    run        per-turn action and intention judge calls for every model
    summarize  score in code, write tables and the calibration report

Rules: src/eval/rules/travelplanner_v2.md. Logic: src/eval/travelplanner_eval_v2.py.
Every judge call is cached as its own file under --out, so each stage resumes.
Judge failures are logged to <out>/errors.jsonl and never scored.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "WebShop"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.llm_clients import OpenRouterChatClient
from eval.human_annotated_pilot import _compact_travel_search_results, atomic_write_json
from eval import travelplanner_eval_v2 as V2
from eval.travelplanner_checks import hotel_validity
from run_single_agent_travel_eval_case import flatten_gold_for_entity_scoring
from mine_travelplanner_judge_cases import db_verdict

DEFAULT_GOLD_DIR = REPO_ROOT / "annotation" / "data" / "exports" / "travelplanner_v4"
MAX_ATTEMPTS = 4
DISPLAY = {
    "claude-sonnet-4.6": "Claude Sonnet 4.6", "claude-sonnet-5": "Claude Sonnet 5", "deepseek-r1": "DeepSeek R1",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-sol": "GPT-5.6 Sol", "grok-4.6": "Grok 4.6", "kimi-k2.5": "Kimi K2.5",
    "minimax-m2.5": "MiniMax M2.5", "nova-pro": "Nova Pro", "qwen3-235b": "Qwen3 235B", "qwen3-32b": "Qwen3 32B",
}


# --------------------------------------------------------------------------- data


def load_data(run_dir: Path, gold_dir: Path) -> Dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = V2.load_audit()
    gold_turns: Dict[tuple, Dict[str, Any]] = {}
    instances: Dict[str, Dict[str, Any]] = {}
    outputs = []
    loaded = {}
    for entry in manifest["outputs"]:
        path = gold_dir / entry["shard_file"]
        if entry["shard_file"] not in loaded:
            loaded[entry["shard_file"]] = (path.read_bytes(), json.loads(path.read_text(encoding="utf-8")))
        raw, shard = loaded[entry["shard_file"]]
        output = json.loads((run_dir / "output" / entry["output"]).read_text(encoding="utf-8"))
        if hashlib.sha256(raw).hexdigest() != output["metadata"]["dataset_sha256"]:
            raise ValueError(f"{entry['output']}: gold sha256 mismatch")
        for instance in shard:
            iid = instance["instance_id"]
            if iid in instances:
                continue
            instances[iid] = {"shard": entry["shard_slug"], "world": instance.get("world_state") or {}, "turns": instance["turns"]}
            for index, turn in enumerate(instance["turns"]):
                turn_id = int(turn.get("turn_id", index))
                gold = flatten_gold_for_entity_scoring(turn.get("gold_current_intention") or {})
                gold, applied = V2.apply_gold_audit(iid, turn_id, gold, audit)
                plan = (((turn.get("gold_action") or {}).get("action_payload") or {}).get("plan") or {}).get("itinerary")
                gold_turns[(iid, turn_id)] = {
                    "instance_id": iid, "turn_id": turn_id, "turn_index": index, "shard": entry["shard_slug"],
                    "gold": gold, "audit_applied": applied, "gold_delta": turn.get("gold_delta") or {},
                    "gold_plan": plan if isinstance(plan, list) and plan else None,
                    "pool": _compact_travel_search_results((turn.get("env_feedback") or {}).get("search_results")),
                    "linguistic_style": turn.get("linguistic_style"),
                }
        outputs.append((entry, output))
    return {"instances": instances, "gold_turns": gold_turns, "outputs": outputs}


def people_of(gold: Dict[str, Any], world: Dict[str, Any]) -> int:
    value = (gold.get("constraints") or {}).get("people_number")
    if not isinstance(value, int):
        value = (world.get("travelplanner_query_data") or {}).get("people_number")
    return int(value or 1)


# --------------------------------------------------------------------------- judge plumbing


class Judge:
    def __init__(self, out: Path, model: str, timeout: int, max_tokens: int):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.client = OpenRouterChatClient(api_key=key, model=model, timeout=timeout, max_tokens=max_tokens)
        self.model = model
        self.out = out
        self.lock = threading.Lock()

    def call(self, cache: Path, prompt: str, validate: Callable[[Dict[str, Any]], Dict[str, Any]], meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        last = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw = self.client.generate_json(prompt)
                result = validate(raw)
                record = {**meta, "judge_model": self.model, "rules_version": V2.RULES_VERSION,
                          "attempts": attempt, "prompt_chars": len(prompt), "judge": result}
                atomic_write_json(cache, record)
                return record
            except Exception as exc:  # noqa: BLE001 - every failure is retried, then logged
                last = exc
        with self.lock, (self.out / "errors.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**meta, "error": repr(last)}, ensure_ascii=False) + "\n")
        return None


class PromptDumper:
    """Stand-in for Judge that writes each prompt to ``directory`` instead of calling the API."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def call(self, cache: Path, prompt: str, validate, meta) -> None:
        (self.directory / f"{cache.parent.name}__{cache.stem}.txt").write_text(prompt, encoding="utf-8")
        return None


def run_parallel(jobs: List[Callable[[], Any]], parallelism: int, label: str) -> None:
    done = [0]
    lock = threading.Lock()

    def wrap(job):
        job()
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(jobs):
                print(f"[{label}] {done[0]}/{len(jobs)}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        list(pool.map(wrap, jobs))


# --------------------------------------------------------------------------- stages


def baseline_path(out: Path, iid: str, turn_id: int) -> Path:
    return out / "baseline" / f"{iid}__t{turn_id}.json"


def prepare(args, data, judge: Judge) -> None:
    rules = V2.load_rules()
    cases = V2.load_calibration()
    jobs = []
    for (iid, turn_id), g in sorted(data["gold_turns"].items()):
        if args.instances and iid not in args.instances:
            continue
        instance = data["instances"][iid]
        reference = instance["world"].get("reference_information")
        payload = {
            "turn_id": turn_id,
            "dialogue_so_far": V2.dialogue_so_far(instance["turns"], g["turn_index"]),
            "gold_constraints": [
                {"field": f, "value": v, "tier": V2.gold_tiers(g["gold"]).get(f, "entity")}
                for f, v in V2.gold_constraints(g["gold"]).items()
            ],
            "gold_delta_this_turn": g["gold_delta"],
            "gold_reference_plan": g["gold_plan"],
            "candidate_records": g["pool"],
        }
        prompt = V2.build_baseline_prompt(payload, rules, V2.fewshot_block("baseline", iid, cases))
        dialogue = payload["dialogue_so_far"]
        people = people_of(g["gold"], instance["world"])
        code = None
        if g["gold_plan"]:
            code = {
                "budget": V2.budget_verdict(g["gold_plan"], None, reference, V2.gold_constraints(g["gold"]).get("budget"), people),
                "hotel": hotel_validity(g["gold_plan"], reference, people),
            }

        def job(prompt=prompt, iid=iid, turn_id=turn_id, g=g, code=code, dialogue=dialogue):
            record = judge.call(
                baseline_path(args.out, iid, turn_id), prompt,
                lambda raw, g=g, d=dialogue: V2.validate_baseline(raw, g["gold"], bool(g["gold_plan"]), d),
                {"stage": "baseline", "instance_id": iid, "turn_id": turn_id,
                 "has_gold_plan": bool(g["gold_plan"]), "gold_plan_code": code, "audit_applied": g["audit_applied"]},
            )
            if record and "out_of_scope_fields" not in record:
                record["out_of_scope_fields"] = sorted(
                    {str(o["field"]) for o in record["judge"].get("out_of_scope") or []}
                )
                atomic_write_json(baseline_path(args.out, iid, turn_id), record)

        jobs.append(job)
    run_parallel(jobs, args.parallelism, "baseline")


def run(args, data, judge: Judge) -> None:
    rules = V2.load_rules()
    cases = V2.load_calibration()
    jobs = []
    for entry, output in data["outputs"]:
        model = entry["model_slug"]
        if args.models and model not in args.models:
            continue
        for trajectory in output["trajectories"]:
            iid = trajectory["instance_id"]
            if args.instances and iid not in args.instances:
                continue
            instance = data["instances"][iid]
            reference = instance["world"].get("reference_information")
            turns = sorted(trajectory["turns"], key=lambda t: int(t["turn_id"]))
            previous_items = None
            for turn in turns:
                turn_id = int(turn["turn_id"])
                g = data["gold_turns"][(iid, turn_id)]
                base_file = baseline_path(args.out, iid, turn_id)
                if not base_file.exists():
                    raise SystemExit(f"missing baseline {base_file}; run prepare first")
                baseline = json.loads(base_file.read_text(encoding="utf-8"))
                if baseline.get("rules_version") != V2.RULES_VERSION:
                    raise SystemExit(
                        f"{base_file} was built with {baseline.get('rules_version')}, not {V2.RULES_VERSION}; "
                        "run prepare into a new --out directory"
                    )
                criteria = {str(c["gold_field"]): c for c in baseline["judge"].get("constraint_criteria") or []}
                out_of_scope = set(baseline["out_of_scope_fields"])
                tiers = V2.gold_tiers(g["gold"])
                fields = [f for f in V2.gold_constraints(g["gold"]) if f not in out_of_scope and f != "budget"]
                dialogue = V2.dialogue_so_far(instance["turns"], g["turn_index"])
                action = turn.get("action") or {}
                # The action judge gets the frozen criteria instead of the dialogue, so every
                # model is audited against one interpretation of the user's requirements.
                action_payload = {
                    "turn_id": turn_id,
                    "fields_to_judge": [
                        {"field": f, "value": V2.gold_constraints(g["gold"])[f], "tier": tiers.get(f, "entity"),
                         "criteria": (criteria.get(f) or {}).get("criteria")}
                        for f in fields
                    ],
                    "activity_requirements": baseline["judge"].get("activity_requirements") or [],
                    "agent_action": {"itinerary": action.get("itinerary"), "rationale": action.get("rationale")},
                    "code_matches": V2.code_matches(action.get("itinerary"), reference),
                    "candidate_records": g["pool"],
                }
                action_prompt = V2.build_action_prompt(action_payload, rules, V2.fewshot_block("action", iid, cases))
                items = V2.intent_items(turn.get("agent_intention_prediction"))
                gold_atoms = baseline["judge"]["gold_atoms"]
                intention_payload = {
                    "turn_id": turn_id,
                    "dialogue_so_far": dialogue,
                    "gold_atoms": [{"atom_id": a["atom_id"], "field": a["source_field"], "value": a["value"]} for a in gold_atoms],
                    "predicted_items": [{"index": i, "field": it.get("field"), "value": it.get("value")} for i, it in enumerate(items)],
                    "previous_turn_predicted_items": None if previous_items is None else
                        [{"field": it.get("field"), "value": it.get("value")} for it in previous_items],
                }
                intention_prompt = V2.build_intention_prompt(intention_payload, rules)
                previous_items = items
                meta = {"model": model, "shard": entry["shard_slug"], "instance_id": iid, "turn_id": turn_id}
                tag = f"{model}__{iid}__t{turn_id}"

                def action_job(p=action_prompt, f=set(fields), m=meta, t=tag):
                    judge.call(args.out / f"action{args.tag}" / f"{t}.json", p,
                               lambda raw, f=f: V2.validate_action(raw, f), {**m, "stage": "action"})

                def intention_job(p=intention_prompt, n=len(items), ids={str(a["atom_id"]) for a in gold_atoms},
                                  first=(turn is turns[0]), m=meta, t=tag):
                    judge.call(args.out / f"intention{args.tag}" / f"{t}.json", p,
                               lambda raw, n=n, ids=ids, first=first: V2.validate_intention(raw, n, ids, first),
                               {**m, "stage": "intention"})

                jobs.append(action_job)
                jobs.append(intention_job)
    run_parallel(jobs, args.parallelism, "run")


# --------------------------------------------------------------------------- summarize


def pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def frac(m: Dict[str, Any]) -> str:
    return f"{pct(m['value'])} ({m['numerator']:.0f}/{m['denominator']})"


def summarize_cmd(args, data) -> None:
    rows = []
    for entry, output in data["outputs"]:
        model = entry["model_slug"]
        if args.models and model not in args.models:
            continue
        for trajectory in output["trajectories"]:
            iid = trajectory["instance_id"]
            if args.instances and iid not in args.instances:
                continue
            instance = data["instances"][iid]
            reference = instance["world"].get("reference_information")
            turns = sorted(trajectory["turns"], key=lambda t: int(t["turn_id"]))
            for turn in turns:
                turn_id = int(turn["turn_id"])
                g = data["gold_turns"][(iid, turn_id)]
                row = {"model": model, "shard": entry["shard_slug"], "instance_id": iid, "turn_id": turn_id,
                       "linguistic_style": g["linguistic_style"], "action": None, "intention": None,
                       "itinerary": (turn.get("action") or {}).get("itinerary")}
                base_file = baseline_path(args.out, iid, turn_id)
                a_file = args.out / f"action{args.tag}" / f"{model}__{iid}__t{turn_id}.json"
                i_file = args.out / f"intention{args.tag}" / f"{model}__{iid}__t{turn_id}.json"
                if base_file.exists() and a_file.exists() and i_file.exists():
                    baseline = json.loads(base_file.read_text(encoding="utf-8"))
                    a = json.loads(a_file.read_text(encoding="utf-8"))["judge"]
                    i = json.loads(i_file.read_text(encoding="utf-8"))["judge"]
                    row["action"] = V2.score_action(
                        gold=g["gold"], baseline=baseline, judgment=a,
                        itinerary=(turn.get("action") or {}).get("itinerary"), reference=reference,
                        gold_plan=g["gold_plan"], people=people_of(g["gold"], instance["world"]),
                    )
                    row["intention"] = V2.score_intention(
                        gold=g["gold"], gold_delta=g["gold_delta"], baseline=baseline, judgment=i,
                        items=V2.intent_items(turn.get("agent_intention_prediction")), first_turn=turn is turns[0],
                    )
                    row["action_judgment"] = a
                rows.append(row)

    shards = sorted({r["shard"] for r in rows})
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    metrics = {m: V2.summarize(r, shards) for m, r in sorted(by_model.items())}
    calibration = calibrate(rows, args.out)
    quality = {"v2": judge_quality(quality_records_v2(rows, data))}
    if args.compare_v1:
        quality["v1"] = judge_quality(quality_records_v1(args.compare_v1, data))
    atomic_write_json(args.out / f"scored_rows{args.tag}.json", {"rules_version": V2.RULES_VERSION, "rows": rows})
    atomic_write_json(args.out / f"metrics{args.tag}.json",
                      {"rules_version": V2.RULES_VERSION, "models": metrics, "calibration": calibration, "judge_quality": quality})
    text = tables(metrics, shards, calibration, quality)
    (args.out / f"tables{args.tag}.md").write_text(text, encoding="utf-8")
    print(text)


QUALITY_FIELDS = ("people_number", "room_type", "accommodation_rating", "restaurant_rating", "house_rule",
                  "accommodation_stay", "meal_cost", "cuisine", "activity", "schedule")


def judge_quality(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Unknown rate per field, and agreement with a database verdict where one exists.

    ``records``: {"field", "status", "gold_value", "itinerary", "reference"} per
    judged constraint. Budget is excluded (priced by code in v2).
    """
    stats: Dict[str, Dict[str, int]] = {}
    for rec in records:
        field = rec["field"]
        if field == "budget":
            continue
        key = field if field in QUALITY_FIELDS else "(all other fields)"
        for bucket in (key, "(all fields)"):
            s = stats.setdefault(bucket, {"n": 0, "unknown": 0, "db_checked": 0, "db_agree": 0})
            s["n"] += 1
            s["unknown"] += rec["status"] == "unknown"
            truth = db_verdict(field, rec["gold_value"], rec["itinerary"], rec["reference"])
            if truth:
                s["db_checked"] += 1
                s["db_agree"] += rec["status"] == truth
    return stats


def quality_records_v2(rows: List[Dict[str, Any]], data) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if not r.get("action"):
            continue
        g = data["gold_turns"][(r["instance_id"], r["turn_id"])]
        reference = data["instances"][r["instance_id"]]["world"].get("reference_information")
        constraints = V2.gold_constraints(g["gold"])
        for field, status in r["action"]["status"].items():
            out.append({"field": field, "status": status, "gold_value": constraints.get(field),
                        "itinerary": r["itinerary"], "reference": reference})
    return out


def quality_records_v1(path: Path, data) -> List[Dict[str, Any]]:
    out = []
    for r in json.loads(path.read_text(encoding="utf-8"))["rows"]:
        reference = data["instances"][r["instance_id"]]["world"].get("reference_information")
        for c in r["scores"]["per_constraint"]:
            out.append({"field": c["field"], "status": c["action_status"], "gold_value": c["gold_value"],
                        "itinerary": (r.get("action") or {}).get("itinerary"), "reference": reference})
    return out


def calibrate(rows: List[Dict[str, Any]], out: Path) -> Dict[str, Any]:
    """Compare v2 judgments with the (draft) calibration labels."""
    index = {(r["model"], r["instance_id"], r["turn_id"]): r for r in rows}
    results = []
    for case in V2.load_calibration():
        src = case["source"]
        if case["stage"] == "baseline":
            file = baseline_path(out, src["instance_id"], src["turn_id"])
            if not file.exists():
                continue
            got = case["field"] in json.loads(file.read_text(encoding="utf-8"))["out_of_scope_fields"]
            results.append({"id": case["id"], "expected": "out_of_scope", "got": "out_of_scope" if got else "in_scope", "ok": got})
            continue
        row = index.get((src["model"], src["instance_id"], src["turn_id"]))
        if not row or not row.get("action"):
            continue
        got = row["action"]["status"].get(case["field"], "out_of_scope")
        ok = got == case["expected"]["action_status"]
        if case["expected"].get("unmet_disclosed") is not None:
            disclosed = case["field"] in row["action"]["disclosed"]
            ok = ok and disclosed == case["expected"]["unmet_disclosed"]
            got = f"{got}{' +disclosed' if disclosed else ''}"
        results.append({"id": case["id"], "category": case["category"], "expected": case["expected"], "got": got, "ok": ok})
    return {"cases": results, "correct": sum(r["ok"] for r in results), "total": len(results)}


def tables(metrics: Dict[str, Any], shards: List[str], calibration: Dict[str, Any], quality: Dict[str, Any]) -> str:
    models = [m for m in DISPLAY if m in metrics]
    out = ["### Action level (v2)", "",
           "| 模型 | " + " | ".join(f"{s} Hard Success" for s in shards) + " | Hard Success (turn macro) | Strict (all Must) | Must 满足 | Preferred（仅 Hard Success 轮） | Optional（仅 Hard Success 轮） |",
           "|---|" + "---:|" * (len(shards) + 5)]
    for m in models:
        a = metrics[m]["action"]
        cells = [f"{a['by_shard'][s]['hard_success']}/{a['by_shard'][s]['turns']}" for s in shards]
        out.append(f"| {DISPLAY[m]} | " + " | ".join(cells) + f" | **{frac(a['hard_success'])}** | {frac(a['strict_success'])} | "
                   f"{frac(a['must_satisfaction'])} | {frac(a['preferred_after_hard_success'])} | {frac(a['optional_after_hard_success'])} |")
    out += ["", "### Action gates and diagnostics (v2)", "",
            "| 模型 | 酒店门槛失败 | 其中最低入住 | 其中容量 | budget 满足 | 餐位覆盖缺口 | Must 违反说明率（按约束，含 budget） | 其中 budget 违反说明率 | 存在未说明 Must 违反的轮 | not-feasible 轮 | 无 gold 行程 | judge 失败 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        a, s = metrics[m]["action"], metrics[m]
        out.append(f"| {DISPLAY[m]} | {frac(a['hotel_gate_fail'])} | {frac(a['min_nights_fail'])} | {frac(a['capacity_fail'])} | "
                   f"{frac(a['budget_satisfied'])} | {frac(a['budget_coverage_gap'])} | {frac(a['must_violation_disclosure'])} | {frac(a['budget_violation_disclosure'])} | "
                   f"{frac(a['undisclosed_must_violation'])} | {s['not_feasible_turns']} | {s['gold_plan_missing']} | {s['judge_errors']} |")
    out += ["", "### Intention level (v2, atoms, turn macro)", "",
            "| 模型 | Overall P / R / F1 | micro P / R | Change P / R / F1 | Turn Exact | Priority Accuracy | Priority Turn Exact |",
            "|---|---|---|---|---:|---:|---:|"]
    for m in models:
        i = metrics[m]["intention"]
        out.append(f"| {DISPLAY[m]} | {pct(i['precision']['value'])} / {pct(i['recall']['value'])} / {pct(i['f1']['value'])} | "
                   f"{pct(i['micro_precision'])} / {pct(i['micro_recall'])} | "
                   f"{pct(i['change_precision']['value'])} / {pct(i['change_recall']['value'])} / {pct(i['change_f1']['value'])} "
                   f"(n={i['change_recall']['denominator']}) | {frac(i['turn_exact'])} | {frac(i['priority_accuracy'])} | {frac(i['priority_turn_exact'])} |")
    out += ["", f"### Calibration (draft labels): {calibration['correct']}/{calibration['total']}", "",
            "| case | category | expected | v2 |", "|---|---|---|---|"]
    for c in calibration["cases"]:
        out.append(f"| {c['id']} | {c.get('category', 'scope')} | {json.dumps(c['expected'], ensure_ascii=False)} | {c['got']}{'' if c['ok'] else ' ✗'} |")
    versions = [v for v in ("v1", "v2") if v in quality]
    out += ["", "### Judge quality: unknown rate and agreement with the database verdict (all models pooled)", "",
            "| field | " + " | ".join(f"{v} n / unknown / DB agree (checked)" for v in versions) + " |",
            "|---|" + "---|" * len(versions)]
    for field in (*QUALITY_FIELDS, "(all other fields)", "(all fields)"):
        cells = []
        for v in versions:
            q = quality[v].get(field)
            cells.append("–" if not q else
                         f"{q['n']} / {pct(q['unknown'] / q['n'])} / {pct(q['db_agree'] / q['db_checked']) if q['db_checked'] else '–'} ({q['db_checked']})")
        out.append(f"| {field} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["prepare", "run", "summarize"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-model", default="openai/gpt-6-luna")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--tag", default="", help="Suffix for per-model judge caches, e.g. _r2 for a repeat run.")
    parser.add_argument("--dump-prompts", type=Path, help="Write prompts to this directory instead of calling the judge.")
    parser.add_argument("--compare-v1", type=Path, help="v1 scored_rows JSON to compare judge quality against.")
    parser.add_argument("--parallelism", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=64000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.run_dir, args.gold_dir)
    if args.stage == "summarize":
        summarize_cmd(args, data)
        return 0
    judge = PromptDumper(args.dump_prompts) if args.dump_prompts else Judge(args.out, args.judge_model, args.timeout, args.max_tokens)
    (prepare if args.stage == "prepare" else run)(args, data, judge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
