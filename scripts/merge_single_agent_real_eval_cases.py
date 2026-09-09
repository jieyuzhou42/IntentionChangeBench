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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [args.cases_dir / f"case_{index}.json" for index in range(16)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing case outputs: {missing}")

    rows: List[Dict[str, Any]] = []
    errors: List[Any] = []
    models = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("rows") or [])
        errors.extend(payload.get("errors") or [])
        models.add(str((payload.get("metadata") or {}).get("model") or "unknown"))
    if errors:
        raise RuntimeError(f"Cannot merge case outputs with errors: {errors}")

    rows.sort(key=lambda row: (row["domain"], row["instance_id"], row["turn_id"]))
    counts = {
        domain: len(
            {
                str(row.get("instance_id"))
                for row in rows
                if row.get("domain") == domain
            }
        )
        for domain in ("webshop", "travelplanner")
    }
    if counts != {"webshop": 10, "travelplanner": 6}:
        raise ValueError(f"Unexpected instance counts: {counts}")

    output = {
        "metadata": {
            "design": "single evaluated agent; two offline metrics; strict public environment",
            "models": sorted(models),
            "instance_counts": counts,
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
