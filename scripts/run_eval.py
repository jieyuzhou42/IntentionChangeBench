#!/usr/bin/env python3
"""Saved-trajectory eval: prepare, run, summarize for travelplanner or webshop."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval import travelplanner_eval_v2 as I
from eval import webshop_eval as W
from eval.action_scoring import SCORING_VERSION
from eval.baseline import BASELINE_VERSION, fingerprint
from eval.summarize import summarize, tables
from eval.webshop_checks import selection


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Judge:
    def __init__(self, args):
        self.args = args
        self.client = None

    def call(self, path, prompt, validator, meta):
        identity = {**meta, "rules_version": W.RULES_VERSION, "rules_sha256": fingerprint(W.RULES_PATH.read_text(encoding="utf-8")),
                    "prompt_sha256": fingerprint(prompt), "judge_model": self.args.judge_model}
        if path.exists():
            record = read(path)
            if any(record.get(k) != v for k, v in identity.items()):
                raise ValueError("Stale cache; choose a new --out: " + str(path))
            validator(record["judge"])
            return record
        if self.args.dump_prompts:
            out = self.args.dump_prompts / (path.parent.name + "__" + path.stem + ".txt")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(prompt, encoding="utf-8")
            return None
        if self.client is None:
            from common.llm_clients import OpenRouterChatClient
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("OPENROUTER_API_KEY required; use --dump-prompts for offline inspection")
            self.client = OpenRouterChatClient(api_key=key, model=self.args.judge_model,
                timeout=self.args.timeout, max_tokens=self.args.max_tokens)
        error = None
        for _ in range(3):
            try:
                raw = validator(self.client.generate_json(prompt))
                record = {**identity, "judge": raw}
                write(path, record)
                return record
            except Exception as exc:
                error = exc
        raise RuntimeError("Judge failed: " + str(error))


def catalog_records(value):
    if isinstance(value, dict):
        value = value.get("products", value.get("items", value))
        if isinstance(value, dict):
            value = list(value.values())
    if not isinstance(value, list):
        raise ValueError("Catalog must be an array or an ASIN-to-product map")
    result = {}
    for item in value:
        asin = str(item["asin"]).upper()
        if asin in result and result[asin] != item:
            raise ValueError("Conflicting catalog records for " + asin)
        result[asin] = item
    return result


def load_baseline(path, payload):
    record = read(path)
    if (record.get("source_sha256") != fingerprint(payload) or record.get("rules_version") != W.RULES_VERSION
            or record.get("rules_sha256") != fingerprint(W.RULES_PATH.read_text(encoding="utf-8"))):
        raise ValueError("Stale baseline; rerun prepare in a new --out: " + str(path))
    W.validate_baseline(record["judge"], payload)
    return record


def saved_trajectories(value):
    if isinstance(value, dict) and "trajectories" in value:
        return value["trajectories"]
    rows = value.get("rows", []) if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("Expected trajectories or rows array")
    if rows and "turns" in rows[0]:
        return rows
    grouped = {}
    for row in rows:
        grouped.setdefault(row["instance_id"], []).append(row)
    return [{"instance_id": iid, "turns": turns} for iid, turns in grouped.items()]


def run_webshop(args):
    cases = read(args.gold)
    if not isinstance(cases, list):
        raise ValueError("--gold must contain an array of annotated cases")
    catalog = catalog_records(read(args.catalog))
    inputs = {}
    for instance in cases:
        if args.instances and instance["instance_id"] not in args.instances:
            continue
        instance["turns"] = sorted(instance["turns"], key=lambda t: int(t["turn_id"]))
        for index, turn in enumerate(instance["turns"]):
            payload = W.make_input(instance, index, catalog)
            key = (instance["instance_id"], int(turn["turn_id"]))
            if key in inputs:
                raise ValueError("Duplicate Gold turn: " + str(key))
            inputs[key] = payload
    judge = Judge(args)
    if args.stage == "prepare":
        for (iid, tid), payload in inputs.items():
            path = args.out / "baseline" / (iid + "__t" + str(tid) + ".json")
            judge.call(path, W.baseline_prompt(payload), lambda raw: W.validate_baseline(raw, payload),
                {"baseline_version": BASELINE_VERSION, "input": payload, "source_sha256": fingerprint(payload)})
        return
    if not args.trajectory:
        raise ValueError("run/summarize require --trajectory MODEL=PATH (repeatable)")
    rows = []
    seen = set()
    models = set()
    for spec in args.trajectory:
        model, sep, path = spec.partition("=")
        if not sep or not model or any(ch in model for ch in '/\\:') or model in (".", ".."):
            raise ValueError("Use --trajectory safe-model-name=PATH")
        models.add(model)
        for trajectory in saved_trajectories(read(path)):
            iid = trajectory["instance_id"]
            if args.instances and iid not in args.instances:
                continue
            previous = None
            for turn in sorted(trajectory["turns"], key=lambda t: int(t["turn_id"])):
                tid = int(turn["turn_id"])
                if (model, iid, tid) in seen:
                    raise ValueError("Duplicate tested turn")
                seen.add((model, iid, tid))
                payload = inputs[(iid, tid)]
                if turn.get("user_utterance") is not None and turn["user_utterance"] != payload["dialogue"][-1]["user"]:
                    raise ValueError("Tested utterance differs from annotated turn")
                baseline = load_baseline(args.out / "baseline" / (iid + "__t" + str(tid) + ".json"), payload)
                selected = selection(turn)
                items = I.intent_items(turn.get("agent_intention_prediction"))
                ip = {"turn_id": tid, "dialogue_so_far": payload["dialogue"],
                    "gold_atoms": [{"atom_id": a["atom_id"], "field": a["source_field"], "value": a["value"]}
                                   for a in baseline["judge"]["gold_atoms"]],
                    "predicted_items": I.prediction_for_judge(items, indexed=True),
                    "previous_turn_predicted_items": previous}
                first = len(payload["dialogue"]) == 1
                previous = I.prediction_for_judge(items)
                meta = {"model": model, "instance_id": iid, "turn_id": tid,
                        "input_sha256": fingerprint(turn), "baseline_sha256": fingerprint(baseline),
                        "intention_rules_sha256": fingerprint(I.load_rules()["Intention rules"])}
                row = {"model": model, "domain": "webshop", "shard": "all", "instance_id": iid, "turn_id": tid,
                       "action": None, "intention": None, "errors": {}}
                for stage in ("action", "intention"):
                    file = args.out / stage / model / (iid + "__t" + str(tid) + ".json")
                    validator = (lambda raw: W.validate_action(raw, set(payload["gold"]["constraints"]))) if stage == "action" else (
                        lambda raw: I.validate_intention(raw, len(items), {a["atom_id"] for a in baseline["judge"]["gold_atoms"]}, first))
                    try:
                        if args.stage == "run":
                            prompt = W.action_prompt(baseline, selected, catalog) if stage == "action" else I.build_intention_prompt(ip, I.load_rules())
                            judge.call(file, prompt, validator, meta)
                        else:
                            record = read(file)
                            if (any(record.get(k) != v for k, v in meta.items()) or record.get("rules_version") != W.RULES_VERSION
                                    or record.get("rules_sha256") != fingerprint(W.RULES_PATH.read_text(encoding="utf-8"))):
                                raise ValueError("Stale judgment; rerun run in a new --out")
                            raw = validator(record["judge"])
                            if stage == "action":
                                row[stage] = W.score_action(baseline, raw, selected, catalog)
                            else:
                                row[stage] = I.score_intention(gold=payload["gold"], gold_delta=payload["gold_delta"],
                                    baseline=baseline, judgment=raw, items=items, first_turn=first)
                    except Exception as exc:
                        row["errors"][stage] = str(exc)
                rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        for model in sorted(models):
            for iid, tid in inputs:
                if (model, iid, tid) not in seen:
                    rows.append({"model": model, "domain": "webshop", "shard": "all", "instance_id": iid,
                                 "turn_id": tid, "action": None, "intention": None,
                                 "errors": {"trajectory": "missing tested turn"}})
        metrics = summarize(rows)
        write(args.out / "scored_rows.json", {"scoring_version": SCORING_VERSION,
            "intention_scoring_version": I.INTENTION_SCORING_VERSION, "rows": rows})
        write(args.out / "metrics.json", {"scoring_version": SCORING_VERSION,
            "intention_scoring_version": I.INTENTION_SCORING_VERSION, "models": metrics})
        (args.out / "tables.md").write_text(tables(metrics), encoding="utf-8")
        print(tables(metrics))
    else:
        write(args.out / "run_errors.json", [r for r in rows if r["errors"]])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # TravelPlanner retains its existing flags and entry point.
    argv = sys.argv[1:]
    if "--domain=travelplanner" in argv:
        argv = [part for part in argv if part != "--domain=travelplanner"] + ["--domain", "travelplanner"]
    if "--domain" in argv and argv.index("--domain") + 1 < len(argv) and argv[argv.index("--domain") + 1] == "travelplanner":
        index = argv.index("--domain")
        sys.argv = [sys.argv[0]] + argv[:index] + argv[index+2:]
        from run_travelplanner_eval_v2 import main as travel_main
        return travel_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "summarize"])
    parser.add_argument("--domain", choices=["webshop", "travelplanner"], required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--trajectory", action="append", default=[])
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-model", default="openai/gpt-6-luna")
    parser.add_argument("--dump-prompts", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=16000)
    run_webshop(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
