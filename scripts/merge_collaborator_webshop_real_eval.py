#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.human_annotated_pilot import aggregate_scored_rows, atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [args.cases_dir / f"case_{index}.json" for index in range(140)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing WebShop case outputs: {missing}")

    rows = []
    agent_models = set()
    judge_models = set()
    errors = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("rows") or [])
        errors.extend(payload.get("errors") or [])
        metadata = payload.get("metadata") or {}
        agent_models.add(str(metadata.get("agent_model") or "unknown"))
        judge_models.add(str(metadata.get("judge_model") or "unknown"))
    if errors:
        raise RuntimeError(f"Cannot merge outputs with errors: {errors}")

    instance_ids = {str(row.get("instance_id")) for row in rows}
    if len(instance_ids) != 140:
        raise ValueError(f"Expected 140 WebShop instances, found {len(instance_ids)}")

    rows.sort(key=lambda row: (row["instance_id"], row["turn_id"]))
    output = {
        "metadata": {
            "design": (
                "single evaluated agent; two offline metrics; "
                "strict public environment"
            ),
            "dataset": "webshop_annotated_21-34",
            "annotation_source": "collaborator_human_annotated",
            "domain": "webshop",
            "instances": 140,
            "agent_models": sorted(agent_models),
            "judge_models": sorted(judge_models),
            "webshop_retrieval": "native WebShop Lucene BM25 search only",
        },
        "rows": rows,
        "aggregate": aggregate_scored_rows(rows),
        "errors": [],
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
