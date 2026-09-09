#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "WebShop", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_full_webshop_real_eval_case as base
from eval.human_annotated_pilot import load_json_list


ANNOTATION_DIR = (
    REPO_ROOT
    / "data"
    / "simulation"
    / "webshop_annotated_21-34"
    / "webshop_v7_full_350_priority_classified_shards"
)


def load_collaborator_annotations():
    shard_paths = sorted(ANNOTATION_DIR.glob("shard_*_human_annotated.json"))
    if len(shard_paths) != 14:
        raise ValueError(
            f"Expected 14 collaborator-annotated WebShop shards, found {len(shard_paths)}"
        )

    instances = []
    for shard_path in shard_paths:
        shard = load_json_list(shard_path)
        if len(shard) != 10:
            raise ValueError(f"Expected 10 cases in {shard_path}, found {len(shard)}")
        instances.extend(shard)

    instance_ids = [str(instance.get("instance_id")) for instance in instances]
    if len(instances) != 140 or len(set(instance_ids)) != 140:
        raise ValueError(
            "Expected 140 unique collaborator-annotated WebShop instances"
        )
    return instances


base.load_all_webshop_annotations = load_collaborator_annotations


if __name__ == "__main__":
    original_main = base.main

    # The shared runner validates against 50 cases. Keep its evaluation behavior
    # while widening only the collaborator dataset's case-index range.
    def main() -> int:
        import argparse
        import json

        from common.llm_clients import BedrockConverseClient
        from eval.human_annotated_pilot import aggregate_scored_rows, atomic_write_json

        parser = argparse.ArgumentParser()
        parser.add_argument("--case-index", type=int, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--agent-model", required=True)
        parser.add_argument("--judge-model", required=True)
        parser.add_argument("--max-internal-steps", type=int, default=12)
        parser.add_argument("--timeout", type=int, default=600)
        args = parser.parse_args()
        if not 0 <= args.case_index < 140:
            raise ValueError("--case-index must be in [0, 139]")

        instances = load_collaborator_annotations()
        agent_client = BedrockConverseClient(
            model=args.agent_model,
            timeout=args.timeout,
            max_tokens=8192,
        )
        judge_client = BedrockConverseClient(
            model=args.judge_model,
            timeout=args.timeout,
            max_tokens=8192,
        )
        rows = base.model_eval._run_webshop(
            instances[args.case_index],
            source_index=args.case_index + 1,
            agent_client=agent_client,
            judge_client=judge_client,
            max_internal_steps=args.max_internal_steps,
        )
        payload = {
            "metadata": {
                "design": (
                    "single evaluated agent; two offline metrics; "
                    "strict public environment"
                ),
                "dataset": "webshop_annotated_21-34",
                "annotation_source": "collaborator_human_annotated",
                "model": args.agent_model,
                "agent_model": args.agent_model,
                "judge_model": args.judge_model,
                "case_index": args.case_index,
                "domain": "webshop",
                "webshop_retrieval": "native WebShop Lucene BM25 search only",
            },
            "rows": rows,
            "aggregate": aggregate_scored_rows(rows),
            "errors": [],
        }
        atomic_write_json(args.output, payload)
        print(json.dumps(payload["aggregate"], indent=2))
        return 0

    raise SystemExit(main())
