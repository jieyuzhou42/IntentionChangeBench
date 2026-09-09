#!/usr/bin/env python3
"""Blindly compare WebShop trajectories against intention-change guidelines."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.llm_clients import create_llm_client_from_env


EXCLUSION_REASONS = {
    "none",
    "repeated_initial_requirement",
    "clarification_only",
    "no_change_feedback",
    "ranking_or_format_request",
    "agent_correction",
    "filler_reversal",
    "single_title_attribute_reveal",
    "product_as_target",
    "bundled_product_following",
    "serialized_product_following",
    "other_pseudo_change",
}


def _load(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def _truncate(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact_delta(delta: Any) -> Dict[str, Any]:
    if not isinstance(delta, dict):
        return {}
    compact: Dict[str, Any] = {}
    for field, change in delta.items():
        if not isinstance(change, dict):
            continue
        compact[field] = {
            "op": change.get("op"),
            "old": change.get("old"),
            "new": change.get("new"),
            "rationale": _truncate(change.get("rationale"), 140),
        }
    return compact


def _candidate_evidence(turns: List[Dict[str, Any]], turn_index: int) -> List[Dict[str, Any]]:
    source_turn = turns[max(turn_index - 1, 0)]
    feedback = source_turn.get("env_feedback") or {}
    candidates = feedback.get("candidate_items") or []
    compact = []
    for item in candidates[:5]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "asin": item.get("asin"),
                "title": _truncate(item.get("title"), 150),
                "price": item.get("price"),
            }
        )
    return compact


def _compact_trajectory(instance: Dict[str, Any]) -> Dict[str, Any]:
    turns = instance.get("turns") or []
    compact_turns = []
    for index, turn in enumerate(turns[1:], start=1):
        shift = (turn.get("shift_condition") or {}).get("details") or {}
        sampling = shift.get("candidate_sampling") or {}
        decision = sampling.get("decision_point")
        compact_turns.append(
            {
                "turn_id": turn.get("turn_id", index),
                "utterance": _truncate(turn.get("user_utterance"), 240),
                "delta": _compact_delta(turn.get("gold_delta")),
                "shift_reason": _truncate((turn.get("shift_condition") or {}).get("reason"), 180),
                "candidate_evidence": _candidate_evidence(turns, index),
                "declared_decision_point": decision,
            }
        )
    first_turn = turns[0] if turns else {}
    return {
        "initial_request": _truncate(first_turn.get("user_utterance"), 260),
        "turns": compact_turns,
    }


def _prompt(instance_id: str, trajectories: Dict[str, Dict[str, Any]]) -> str:
    return f"""
You are auditing synthetic shopping trajectories. Compare the anonymous variants for
instance {instance_id}. Judge only the supplied trajectory and candidate evidence.

Required standards:
1. Every kept turn must materially change the final purchase target via add, relax,
   override, or reprioritize.
2. Reject pseudo changes: repeating an initial or previous-turn requirement; merely
   clarifying an existing meaning; no-change feedback; ranking/output-format requests;
   correcting an agent that omitted or misapplied an already explicit requirement;
   unsupported add-then-delete filler; or progressively copying product attributes.
3. Candidate products are trigger evidence, not the new target answer. A kept turn
   may extract one abstract purchase trade-off, but must not adopt a candidate's full
   product form, feature bundle, packaging, accessories, or title as the next goal.
4. Each turn should contain one primary decision trade-off and one substantive target
   change, optionally with reprioritization. Reject bundled following of several product
   attributes even when every attribute is present in a real result.
5. Audit the full trajectory for serialized product following. Limiting each turn to
   one field is not enough: reject successive turns that extract brand, color, material,
   form, packaging, or functions from the same product or near-identical product family.
   Distinct ASINs count as a real comparison only when their meaningful differences can
   plausibly change the search direction; cosmetic variants are insufficient.
6. A good trajectory compares genuinely different products and may revise the search
   direction because another product exposes a reasonable trade-off. The result evidence
   must cause the change rather than merely decorate a predetermined next constraint.
7. The whole trajectory must form a coherent purchase decision process without
   unexplained toggling of color, budget, material, category, product form, or use case.

For each turn, return:
- genuine_change: boolean
- exclusion_reason: one of {sorted(EXCLUSION_REASONS)}
- result_grounded: boolean
- cross_product_tradeoff: boolean
- trajectory_coherent: boolean
- keep: boolean; true only if the turn should remain under all standards
- note: concise reason, at most 25 words

Also return a variant_summary with integers:
- total_turns
- kept_turns
- genuine_change_turns
- result_grounded_turns
- cross_product_tradeoff_turns
- coherent_turns
- pseudo_change_turns
- single_product_anchoring_turns
- serialized_product_following_turns

Use this exact JSON shape:
{{
  "variants": {{
    "A": {{"turns": [{{...}}], "variant_summary": {{...}}}},
    "B": {{"turns": [{{...}}], "variant_summary": {{...}}}},
    "C": {{"turns": [{{...}}], "variant_summary": {{...}}}}
  }},
  "best_variant": "A | B | C | tie",
  "comparison_note": "at most 50 words"
}}

Anonymous trajectories:
{json.dumps(trajectories, ensure_ascii=False, default=str)}
""".strip()


def _validate(result: Dict[str, Any], labels: Iterable[str]) -> None:
    variants = result.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Judge output has no variants object")
    for label in labels:
        item = variants.get(label)
        if not isinstance(item, dict) or not isinstance(item.get("turns"), list):
            raise ValueError(f"Judge output is missing variant {label}")
        for turn in item["turns"]:
            reason = str(turn.get("exclusion_reason") or "")
            if reason not in EXCLUSION_REASONS:
                raise ValueError(f"Unexpected exclusion_reason: {reason}")


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, Counter[str]] = defaultdict(Counter)
    best_counts: Counter[str] = Counter()
    for row in rows:
        label_to_version = row["label_to_version"]
        judged = row["judgment"]
        best_label = judged.get("best_variant")
        best_counts[label_to_version.get(best_label, "tie")] += 1
        for label, item in judged["variants"].items():
            version = label_to_version[label]
            summary = item.get("variant_summary") or {}
            for key, value in summary.items():
                if isinstance(value, int):
                    totals[version][key] += value
            for turn in item.get("turns") or []:
                totals[version][f"exclusion:{turn.get('exclusion_reason')}"] += 1

    report: Dict[str, Any] = {"best_variant_counts": dict(best_counts), "versions": {}}
    for version, counts in totals.items():
        total_turns = counts.get("total_turns", 0)
        report["versions"][version] = {
            **dict(counts),
            "keep_rate": (
                round(counts.get("kept_turns", 0) / total_turns, 4)
                if total_turns
                else 0.0
            ),
            "genuine_change_rate": (
                round(counts.get("genuine_change_turns", 0) / total_turns, 4)
                if total_turns
                else 0.0
            ),
            "result_grounded_rate": (
                round(counts.get("result_grounded_turns", 0) / total_turns, 4)
                if total_turns
                else 0.0
            ),
            "cross_product_tradeoff_rate": (
                round(counts.get("cross_product_tradeoff_turns", 0) / total_turns, 4)
                if total_turns
                else 0.0
            ),
            "coherence_rate": (
                round(counts.get("coherent_turns", 0) / total_turns, 4)
                if total_turns
                else 0.0
            ),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets = {
        "v2": _load(args.v2),
        "v3": _load(args.v3),
        "new": _load(args.new),
    }
    by_version = {
        version: {str(item.get("instance_id")): item for item in items}
        for version, items in datasets.items()
    }
    instance_ids = [str(item.get("instance_id")) for item in datasets["new"]]
    for version, mapping in by_version.items():
        missing = [instance_id for instance_id in instance_ids if instance_id not in mapping]
        if missing:
            raise ValueError(f"{version} is missing instances: {missing}")

    client = create_llm_client_from_env(timeout=300)
    rows: List[Dict[str, Any]] = []
    for position, instance_id in enumerate(instance_ids):
        versions = ["v2", "v3", "new"]
        random.Random(f"blind-order:{instance_id}").shuffle(versions)
        labels = ["A", "B", "C"]
        label_to_version = dict(zip(labels, versions))
        anonymous = {
            label: _compact_trajectory(by_version[version][instance_id])
            for label, version in label_to_version.items()
        }
        judgment = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                judgment = client.generate_json(_prompt(instance_id, anonymous))
                break
            except ValueError as error:
                last_error = error
                print(
                    f"judge retry {attempt}/3 for {instance_id}: {error}",
                    flush=True,
                )
        if judgment is None:
            raise RuntimeError(
                f"Judge failed after 3 attempts for {instance_id}"
            ) from last_error
        _validate(judgment, labels)
        rows.append(
            {
                "instance_id": instance_id,
                "label_to_version": label_to_version,
                "judgment": judgment,
            }
        )
        payload = {"rows": rows, "aggregate": _aggregate(rows)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, args.output)
        print(f"judged {position + 1}/{len(instance_ids)}: {instance_id}", flush=True)

    print(json.dumps(_aggregate(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
