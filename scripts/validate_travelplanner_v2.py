#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from run_travelplanner_v2_shard import _valid_instances

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--tasks",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    tasks=json.loads(args.tasks.read_text(encoding="utf-8"))
    instances=json.loads(args.output.read_text(encoding="utf-8"))
    expected=[str(task["instance_id"]) for task in tasks]
    actual=[str(item.get("instance_id") or "") for item in instances]
    valid=_valid_instances(instances,expected,7)
    if actual != expected:
        raise ValueError("Output IDs are missing, duplicated, or out of order")
    if len(valid) != len(expected):
        invalid=[item for item in expected if item not in valid]
        raise ValueError("Invalid trajectories: "+",".join(invalid))
    print(f"Validated {len(instances)} TravelPlanner v2 trajectories")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
