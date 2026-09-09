"""Build a representative 10-case TravelPlanner small-scale evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# One case from every days x level category, plus a second 3-day Hard case.
# Middle reference-size quartiles avoid selecting unusually tiny/huge contexts.
TARGETS: List[Tuple[int, str, int, str]] = [
    (3, "easy", 2, "budget_only"),
    (3, "medium", 3, "cuisine"),
    (3, "hard", 2, "cuisine+house rule+transportation"),
    (3, "hard", 3, "house rule+room type+transportation"),
    (5, "easy", 3, "budget_only"),
    (5, "medium", 2, "house rule"),
    (5, "hard", 3, "cuisine+house rule+room type"),
    (7, "easy", 2, "budget_only"),
    (7, "medium", 3, "room type"),
    (7, "hard", 2, "cuisine+room type+transportation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/travelplanner/short_horizon_250_multi_360_tasks.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/travelplanner/small_scale_10_tasks.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/travelplanner/small_scale_10_manifest.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/travelplanner/small_scale_10_report.json"),
    )
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def choose_tasks(tasks: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    used_origins = set()
    used_destinations = set()

    for days, level, quartile, signature in TARGETS:
        candidates = [
            task
            for task in tasks
            if task["instance_id"] not in selected_ids
            and task["travelplanner_query_data"]["days"] == days
            and task["travelplanner_query_data"]["level"] == level
            and task["selection_metadata"]["reference_size_quartile"] == quartile
            and task["selection_metadata"]["constraint_signature"] == signature
        ]
        if not candidates:
            raise ValueError(
                f"No candidate for {days}-day/{level}/Q{quartile}/{signature}"
            )

        for task in candidates:
            task["_small_set_tie_breaker"] = rng.random()

        def score(task: Dict[str, Any]) -> Tuple[int, int, float, int]:
            query = task["travelplanner_query_data"]
            source_row = int(task["source"]["row_number_1_based"])
            return (
                int(query["org"] in used_origins),
                int(query["dest"] in used_destinations),
                task["_small_set_tie_breaker"],
                source_row,
            )

        choice = min(candidates, key=score)
        choice.pop("_small_set_tie_breaker", None)
        selected.append(choice)
        selected_ids.add(choice["instance_id"])
        query = choice["travelplanner_query_data"]
        used_origins.add(query["org"])
        used_destinations.add(query["dest"])

    return selected


def write_manifest(path: Path, tasks: List[Dict[str, Any]]) -> None:
    fields = (
        "case_number",
        "instance_id",
        "days",
        "level",
        "people_number",
        "org",
        "dest",
        "constraint_signature",
        "reference_size_quartile",
        "query",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, task in enumerate(tasks, start=1):
            query = task["travelplanner_query_data"]
            metadata = task["selection_metadata"]
            writer.writerow(
                {
                    "case_number": index,
                    "instance_id": task["instance_id"],
                    "days": query["days"],
                    "level": query["level"],
                    "people_number": query["people_number"],
                    "org": query["org"],
                    "dest": query["dest"],
                    "constraint_signature": metadata["constraint_signature"],
                    "reference_size_quartile": metadata["reference_size_quartile"],
                    "query": query["query"],
                }
            )


def main() -> None:
    args = parse_args()
    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    selected = choose_tasks(tasks, args.seed)
    for case_number, task in enumerate(selected, start=1):
        task["small_set_metadata"] = {
            "case_number": case_number,
            "selection_seed": args.seed,
            "coverage_category": (
                f"{task['travelplanner_query_data']['days']}-day/"
                f"{task['travelplanner_query_data']['level']}"
            ),
        }

    report = {
        "source": str(args.input),
        "selection_seed": args.seed,
        "selected_count": len(selected),
        "policy": "One per days x level category plus one extra 3-day Hard case",
        "category_targets": [
            {
                "days": days,
                "level": level,
                "reference_size_quartile": quartile,
                "constraint_signature": signature,
            }
            for days, level, quartile, signature in TARGETS
        ],
        "instance_ids": [task["instance_id"] for task in selected],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(args.manifest, selected)
    print(f"Selected {len(selected)} cases -> {args.output}")
    print(f"Manifest -> {args.manifest}")
    print(f"Report -> {args.report}")


if __name__ == "__main__":
    main()
