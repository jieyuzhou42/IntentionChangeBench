#!/usr/bin/env python3
"""Run the teammate's constraint extractor without letting it rewrite priority.

`scripts/extract_travelplanner_constraints.py` does two independent things:

  A. drops TRAVEL_CONTEXT_FIELDS (org / dest / start_date / end_date /
     visiting_city_number) so itinerary context stops being scored as a
     constraint, and keeps days / people_number / budget only when the
     utterance grounds them;
  B. calls `classify_instance` at the end, which recomputes `priority` purely
     from mention recency and therefore discards hand-annotated priorities.

B is the generator default for unlabelled data. When the input already carries
reviewed priorities we only want A, so this wrapper neutralises the
`classify_instance` call and restores each turn's original priority afterwards.

Nothing in scripts/ or src/ is modified; the upstream module is imported and
patched in memory for this process only.

Usage:
    python annotation/tools/extract_constraints_keep_priority.py IN.json OUT.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import scripts.extract_travelplanner_constraints as upstream  # noqa: E402


def _priorities_by_turn(instances: List[Dict[str, Any]]) -> Dict[str, Dict[int, Any]]:
    saved: Dict[str, Dict[int, Any]] = {}
    for instance in instances:
        per_turn: Dict[int, Any] = {}
        for index, turn in enumerate(instance.get("turns") or []):
            gold = turn.get("gold_current_intention")
            if isinstance(gold, dict) and gold.get("priority") is not None:
                per_turn[int(turn.get("turn_id", index))] = copy.deepcopy(gold["priority"])
        saved[str(instance.get("instance_id"))] = per_turn
    return saved


def _restore_priorities(
    instances: List[Dict[str, Any]],
    saved: Dict[str, Dict[int, Any]],
) -> Dict[str, int]:
    stats = {"restored": 0, "no_source": 0, "dropped_fields": 0}
    for instance in instances:
        per_turn = saved.get(str(instance.get("instance_id"))) or {}
        for index, turn in enumerate(instance.get("turns") or []):
            gold = turn.get("gold_current_intention")
            if not isinstance(gold, dict):
                continue
            original = per_turn.get(int(turn.get("turn_id", index)))
            if original is None:
                stats["no_source"] += 1
                continue
            # Extraction removes context fields, so a restored priority must not
            # reference fields that no longer exist in constraints.
            active = {
                str(field)
                for field, value in (gold.get("constraints") or {}).items()
                if value is not None
            }
            entities = gold.get("entities")
            if isinstance(entities, dict):
                for entity_id, entity in entities.items():
                    if not isinstance(entity, dict):
                        continue
                    for field, value in (entity.get("constraints") or {}).items():
                        if value is not None:
                            active.add(f"entities.{entity_id}.constraints.{field}")
            if isinstance(original, dict):
                cleaned = {}
                for level in ("high", "medium", "low"):
                    kept = [f for f in (original.get(level) or []) if str(f) in active]
                    stats["dropped_fields"] += len(original.get(level) or []) - len(kept)
                    cleaned[level] = kept
            elif isinstance(original, list):
                kept = [f for f in original if str(f) in active]
                stats["dropped_fields"] += len(original) - len(kept)
                cleaned = kept
            else:
                cleaned = original
            gold["priority"] = cleaned
            stats["restored"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("output must differ from input")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise SystemExit(f"{args.input} is not a JSON list")
    saved = _priorities_by_turn(source)
    print(
        f"保留 {sum(len(v) for v in saved.values())} 轮的人工 priority；"
        f"只应用字段过滤（TRAVEL_CONTEXT_FIELDS={sorted(upstream.TRAVEL_CONTEXT_FIELDS)}）",
        file=sys.stderr,
    )

    # Neutralise B for this process. A (the field filtering) is untouched.
    upstream.classify_instance = lambda instance: None

    sys.argv = [
        "extract_travelplanner_constraints.py",
        str(args.input),
        str(args.output),
        "--parallelism",
        str(args.parallelism),
        "--retries",
        str(args.retries),
        "--timeout",
        str(args.timeout),
    ]
    upstream.main()

    extracted = json.loads(args.output.read_text(encoding="utf-8"))
    stats = _restore_priorities(extracted, saved)
    args.output.write_text(
        json.dumps(extracted, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"priority 已还原：{stats['restored']} 轮"
        f"（无来源 {stats['no_source']} 轮，"
        f"因字段被剔除而移出 priority 的条目 {stats['dropped_fields']} 个）",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
