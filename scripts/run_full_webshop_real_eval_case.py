#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT / "src",
    REPO_ROOT / "WebShop",
    REPO_ROOT / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_qwen_single_agent_real_eval_case as model_eval
import run_single_agent_real_eval_case as baseline
from common.llm_clients import BedrockConverseClient
from eval.human_annotated_pilot import (
    aggregate_scored_rows,
    atomic_write_json,
    load_json_list,
)


def load_all_webshop_annotations():
    instances = []
    shard_paths = sorted(baseline.SHARD_DIR.glob("*_human_annotated.json"))
    if len(shard_paths) != 5:
        raise ValueError(f"Expected 5 annotated WebShop shards, found {len(shard_paths)}")
    for shard_path in shard_paths:
        shard = load_json_list(shard_path)
        if len(shard) != 10:
            raise ValueError(f"Expected 10 cases in {shard_path}, found {len(shard)}")
        instances.extend(shard)
    if len(instances) != 50:
        raise ValueError(f"Expected 50 annotated WebShop cases, found {len(instances)}")
    return instances


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-internal-steps", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.case_index < 50:
        raise ValueError("--case-index must be in [0, 49]")

    instances = load_all_webshop_annotations()
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
    rows = model_eval._run_webshop(
        instances[args.case_index],
        source_index=args.case_index + 1,
        agent_client=agent_client,
        judge_client=judge_client,
        max_internal_steps=args.max_internal_steps,
    )
    payload = {
        "metadata": {
            "design": "single evaluated agent; two offline metrics; strict public environment",
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


if __name__ == "__main__":
    raise SystemExit(main())
