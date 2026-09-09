#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    metadata = payload.setdefault("metadata", {})
    metadata["annotation_sources"] = {
        "human_annotated": 44,
        "source_gold": 6,
    }
    temporary = args.path.with_suffix(args.path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
