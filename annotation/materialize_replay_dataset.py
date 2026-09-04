from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from annotation.replay_server import (
    enrich_webshop_constraints_from_metadata,
    load_json,
    set_initial_constraints_must_have,
)


def save_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the exact constraint and priority state initially shown by replay_server."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    instances = load_json(args.source)
    if not isinstance(instances, list):
        raise ValueError(f"Expected a JSON list in {args.source}")
    restored = enrich_webshop_constraints_from_metadata(instances)
    initial_priorities = set_initial_constraints_must_have(instances)
    save_json_atomic(args.output, instances)
    print(
        f"Materialized {len(instances)} trajectories with {restored} restored constraints "
        f"and {initial_priorities} all-Must-have initial turns."
    )


if __name__ == "__main__":
    main()
