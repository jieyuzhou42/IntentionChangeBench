#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.human_annotated_pilot import aggregate_scored_rows, atomic_write_json


def _load(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop", type=Path, required=True)
    parser.add_argument("--travelplanner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [_load(args.webshop), _load(args.travelplanner)]
    errors: List[Any] = []
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get("rows") or [])
        errors.extend(payload.get("errors") or [])
    if errors:
        raise RuntimeError(f"Cannot finalize pilot with errors: {errors}")

    rows.sort(key=lambda row: (row["domain"], row["instance_id"], row["turn_id"]))
    instance_counts = {
        domain: len(
            {
                str(row.get("instance_id"))
                for row in rows
                if row.get("domain") == domain
            }
        )
        for domain in ("webshop", "travelplanner")
    }
    expected = {"webshop": 10, "travelplanner": 6}
    if instance_counts != expected:
        raise ValueError(f"Expected instance counts {expected}, got {instance_counts}")

    model_names = {
        str((payload.get("metadata") or {}).get("model") or "unknown")
        for payload in payloads
    }
    output = {
        "metadata": {
            "design": "blind real-environment two-layer pilot",
            "models": sorted(model_names),
            "instance_counts": instance_counts,
            "webshop_retrieval": "real WebShop search action -> native Lucene BM25 results",
            "travelplanner_execution": "real Search -> NotebookWrite -> Planner tools",
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
