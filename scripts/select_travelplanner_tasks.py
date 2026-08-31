"""Select a reproducible, Hard-heavy TravelPlanner test subset.

The official test CSV keeps local constraints only in the natural-language
query.  This script reconstructs those constraints, aligns the query rows with
the structured ``test_ref_info.jsonl`` records, and performs diversity-aware
sampling across day, level, and reference-size strata.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


LEVEL_QUOTAS = {"easy": 15, "medium": 35, "hard": 70}
CUISINES = (
    "Chinese",
    "American",
    "Italian",
    "Mexican",
    "Indian",
    "Mediterranean",
    "French",
)
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "a": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("TravelPlanner/database/test.csv"),
    )
    parser.add_argument(
        "--reference-info",
        type=Path,
        default=Path("TravelPlanner/database/test_ref_info.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/travelplanner/diverse_360_tasks.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/travelplanner/diverse_360_report.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/travelplanner/diverse_360_manifest.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def first_match(patterns: Iterable[str], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match
    return None


def parse_budget(query: str) -> int:
    match = first_match(
        (
            r"\$\s*([0-9][0-9,]*)",
            r"budget(?:\s+(?:is|of|at|set at|set to|limited to))?\s*([0-9][0-9,]*)",
            r"([0-9][0-9,]*)\s*dollars?",
        ),
        query,
    )
    if not match:
        raise ValueError(f"Could not parse budget: {query}")
    return int(match.group(1).replace(",", ""))


def parse_people(query: str, level: str) -> int:
    text = query.lower()
    if any(phrase in text for phrase in ("solo traveler", "solo trip", "lone traveler")):
        return 1
    match = first_match(
        (
            r"(?:for|party of|group of)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a)\s+(?:people|persons?|travelers?|individuals?)\b",
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons?|travelers?|individuals?)\b",
            r"(?:trip|itinerary|plan|journey)\s+for\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            r"\btrip for (\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            r"\bfor a (?:family|group|party) of (\d+)\b",
            r"\b(pair|couple)(?: of travelers?)?\b",
            r"\b(?:group|party|team|family) of (\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            r"\bthere (?:will|would) be (\d+|one|two|three|four|five|six|seven|eight|nine|ten) of us\b",
            r"\bwe(?:'re| are) (?:a )?(?:group|party|team|family) of (\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            r"\bfor (\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s*[,.]|\s+(?:departing|leaving|starting|traveling|travelling|visiting))",
        ),
        text,
    )
    if match:
        token = match.group(1)
        if token in {"pair", "couple"}:
            return 2
        return int(token) if token.isdigit() else NUMBER_WORDS[token]
    if level == "easy" or "one person" in text or "one individual" in text:
        return 1
    raise ValueError(f"Could not parse party size: {query}")


def parse_local_constraint(query: str) -> Dict[str, Any]:
    text = query.lower()
    cuisine_context = any(
        marker in text
        for marker in ("cuisine", "food", "meal", "dining", "dishes", "restaurants")
    )
    cuisines = [
        name
        for name in CUISINES
        if cuisine_context and re.search(rf"\b{re.escape(name.lower())}\b", text)
    ]

    room_type = None
    if re.search(
        r"(?:not|non)[ -]?shared rooms?|do not include shared|avoid shared|"
        r"rooms?.{0,30}(?:not|aren't|isn't|cannot be|can't be) shared|"
        r"(?:not|never) (?:to )?share (?:our )?(?:rooms?|accommodations?)|"
        r"do not want to share|prefer not to share|won't be sharing|guarantee privacy",
        text,
    ):
        room_type = "not shared room"
    elif re.search(r"entire (?:rooms?|homes?|places?|accommodations?)", text):
        room_type = "entire room"
    elif re.search(r"private rooms?", text):
        room_type = "private room"
    elif re.search(r"(?:prefer|want|require|book|stay in) (?:a )?shared rooms?", text):
        room_type = "shared room"

    house_rule = None
    house_patterns: Sequence[Tuple[str, str]] = (
        ("children under 10", r"children under (?:the age of )?10|child-friendly|young children|traveling with (?:a child|kids)"),
        ("parties", r"allow(?:ed|ing)? (?:us to host )?part(?:y|ies)|permit(?:s|ted|ting)? parties|parties (?:are )?(?:allowed|permitted)|party-friendly|suitable for parties|open to parties|accommodate parties"),
        ("smoking", r"smoking[- ](?:is )?(?:allowed|permitted)|allow(?:ed|ing)? smoking|permit(?:s|ted|ting)? smoking|smoking-friendly|smoking-allowed|smokers?.{0,45}(?:allow|permit) smoking"),
        ("visitors", r"allow(?:ed|ing)? (?:us to have )?visitors|permit(?:s|ted|ting)? visitors|visitors (?:are )?(?:allowed|permitted)|have visitors|welcome visitors|visitor[- ]friendly|visitors[- ]allowed|expecting visitors|entertain visitors|accommodate visitors|open to visitors|allow guests|lodgings must allow guests"),
        ("pets", r"pet-friendly|allow(?:ed|ing)? pets|pets (?:are )?(?:allowed|permitted)|bringing (?:our )?pets|bring pets|travel(?:ing|ling) with (?:our )?pets|pets.{0,45}allow them|accommodate pets"),
    )
    for value, pattern in house_patterns:
        if re.search(pattern, text):
            house_rule = value
            break

    transportation = None
    negative_markers = (
        "avoid", "exclude", "without", "not", "no ", "won't", "will not",
        "do not", "don't", "does not", "doesn't", "forego", "anything but",
        "rather than", "instead of", "other than", "did not involve",
    )
    flight_mentioned = bool(re.search(r"\b(?:flight|flights|flying|fly|air travel)\b", text))
    self_drive_mentioned = bool(
        re.search(r"self[- ]driv|drive ourselves|driving ourselves|drive on our own|planning to drive|own driving", text)
    )
    if (flight_mentioned and any(marker in text for marker in negative_markers)) or "ground transportation" in text:
        transportation = "no flight"
    elif self_drive_mentioned and any(marker in text for marker in negative_markers):
        transportation = "no self-driving"

    return {
        "house rule": house_rule,
        "cuisine": cuisines or None,
        "room type": room_type,
        "transportation": transportation,
    }


def constraint_signature(local_constraint: Dict[str, Any]) -> str:
    names = [key for key, value in local_constraint.items() if value is not None]
    return "+".join(sorted(names)) if names else "budget_only"


def read_rows(csv_path: Path, ref_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    raw_ref_lines = [line for line in ref_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(csv_rows) != len(raw_ref_lines):
        raise ValueError(
            f"CSV/reference row mismatch: {len(csv_rows)} != {len(raw_ref_lines)}"
        )

    candidates: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    for zero_index, (row, raw_ref) in enumerate(zip(csv_rows, raw_ref_lines)):
        query = row["query"].strip()
        level = row["level"].strip().lower()
        try:
            local_constraint = parse_local_constraint(query)
            active_count = sum(value is not None for value in local_constraint.values())
            expected_count = {"easy": 0, "medium": 1, "hard": 3}[level]
            if active_count != expected_count:
                raise ValueError(
                    f"expected {expected_count} local constraint types, parsed {active_count}: "
                    f"{local_constraint}"
                )
            reference = json.loads(raw_ref)
            if not isinstance(reference, dict):
                raise ValueError("reference information is not an object")
            query_data = {
                "org": row["org"].strip(),
                "dest": row["dest"].strip(),
                "days": int(row["days"]),
                "visiting_city_number": {3: 1, 5: 2, 7: 3}[int(row["days"])],
                "date": ast.literal_eval(row["date"]),
                "people_number": parse_people(query, level),
                "local_constraint": local_constraint,
                "budget": parse_budget(query),
                "query": query,
                "level": level,
            }
        except (KeyError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
            parse_errors.append(f"row {zero_index + 1}: {exc}")
            continue

        candidate_count = sum(
            len(value) if isinstance(value, list) else 1
            for value in reference.values()
        )
        candidates.append(
            {
                "source_row": zero_index + 1,
                "query_data": query_data,
                "reference_information": reference,
                "reference_chars": len(raw_ref),
                "reference_items": candidate_count,
                "constraint_signature": constraint_signature(local_constraint),
            }
        )

    return candidates, parse_errors


def assign_reference_bins(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_data = row["query_data"]
        grouped[(query_data["days"], query_data["level"])].append(row)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item["reference_chars"], item["source_row"]))
        for rank, row in enumerate(ordered):
            row["reference_bin"] = min(3, (4 * rank) // len(ordered))


def bin_quotas(total: int) -> List[int]:
    base, remainder = divmod(total, 4)
    return [base + (1 if index < remainder else 0) for index in range(4)]


def select_rows(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    org_counts: Counter[str] = Counter()
    dest_counts: Counter[str] = Counter()
    route_counts: Counter[Tuple[str, str]] = Counter()
    signature_counts: Counter[Tuple[int, str, str]] = Counter()

    for days in (3, 5, 7):
        for level in ("easy", "medium", "hard"):
            quota = LEVEL_QUOTAS[level]
            group = [
                row
                for row in rows
                if row["query_data"]["days"] == days
                and row["query_data"]["level"] == level
            ]
            per_bin = bin_quotas(quota)
            for ref_bin, bin_quota in enumerate(per_bin):
                pool = [row for row in group if row["reference_bin"] == ref_bin]
                if len(pool) < bin_quota:
                    raise ValueError(
                        f"Insufficient rows for {days}-day/{level}/Q{ref_bin + 1}: "
                        f"{len(pool)} < {bin_quota}"
                    )
                for row in pool:
                    row["tie_breaker"] = rng.random()
                for _ in range(bin_quota):
                    def score(item: Dict[str, Any]) -> Tuple[float, float, int]:
                        query_data = item["query_data"]
                        org = query_data["org"]
                        dest = query_data["dest"]
                        route = (org, dest)
                        signature_key = (days, level, item["constraint_signature"])
                        diversity_penalty = (
                            4.0 * route_counts[route]
                            + 1.5 * org_counts[org]
                            + 2.0 * dest_counts[dest]
                            + 1.0 * signature_counts[signature_key]
                        )
                        return diversity_penalty, item["tie_breaker"], item["source_row"]

                    choice = min(pool, key=score)
                    pool.remove(choice)
                    selected.append(choice)
                    query_data = choice["query_data"]
                    org_counts[query_data["org"]] += 1
                    dest_counts[query_data["dest"]] += 1
                    route_counts[(query_data["org"], query_data["dest"])] += 1
                    signature_counts[(days, level, choice["constraint_signature"])] += 1

    return selected


def make_tasks(selected: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    tasks = []
    for row in sorted(selected, key=lambda item: item["source_row"]):
        tasks.append(
            {
                "instance_id": f"travelplanner_test_{row['source_row']:04d}",
                "task_type": "planning",
                "subtype": "travel",
                "source": {
                    "dataset": "osunlp/TravelPlanner",
                    "split": "test",
                    "row_number_1_based": row["source_row"],
                },
                "travelplanner_query_data": row["query_data"],
                "reference_information": row["reference_information"],
                "selection_metadata": {
                    "seed": seed,
                    "reference_chars": row["reference_chars"],
                    "reference_items": row["reference_items"],
                    "reference_size_quartile": row["reference_bin"] + 1,
                    "constraint_signature": row["constraint_signature"],
                },
            }
        )
    return tasks


def nested_distribution(
    rows: Iterable[Dict[str, Any]], fields: Sequence[str]
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for row in rows:
        cursor = output
        for field in fields[:-1]:
            value = str(row["query_data"].get(field, row.get(field)))
            cursor = cursor.setdefault(value, {})
        last = str(row["query_data"].get(fields[-1], row.get(fields[-1])))
        cursor[last] = cursor.get(last, 0) + 1
    return output


def write_manifest(path: Path, tasks: List[Dict[str, Any]]) -> None:
    fieldnames = (
        "instance_id",
        "source_row_1_based",
        "org",
        "dest",
        "days",
        "level",
        "people_number",
        "budget",
        "local_constraint",
        "constraint_signature",
        "reference_size_quartile",
        "reference_chars",
        "reference_items",
        "query",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            query_data = task["travelplanner_query_data"]
            metadata = task["selection_metadata"]
            writer.writerow(
                {
                    "instance_id": task["instance_id"],
                    "source_row_1_based": task["source"]["row_number_1_based"],
                    "org": query_data["org"],
                    "dest": query_data["dest"],
                    "days": query_data["days"],
                    "level": query_data["level"],
                    "people_number": query_data["people_number"],
                    "budget": query_data["budget"],
                    "local_constraint": json.dumps(
                        query_data["local_constraint"], ensure_ascii=False, sort_keys=True
                    ),
                    "constraint_signature": metadata["constraint_signature"],
                    "reference_size_quartile": metadata["reference_size_quartile"],
                    "reference_chars": metadata["reference_chars"],
                    "reference_items": metadata["reference_items"],
                    "query": query_data["query"],
                }
            )


def build_report(
    all_rows: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    seed: int,
    parse_errors: List[str],
) -> Dict[str, Any]:
    selected_orgs = Counter(row["query_data"]["org"] for row in selected)
    selected_dests = Counter(row["query_data"]["dest"] for row in selected)
    selected_routes = Counter(
        (row["query_data"]["org"], row["query_data"]["dest"])
        for row in selected
    )
    signatures = Counter(row["constraint_signature"] for row in selected)
    ref_chars = [row["reference_chars"] for row in selected]
    ref_items = [row["reference_items"] for row in selected]
    return {
        "selection_method": "Hard-heavy stratified diversity sampling",
        "seed": seed,
        "source_rows": len(all_rows) + len(parse_errors),
        "eligible_rows_after_strict_parsing": len(all_rows),
        "rejected_rows_after_strict_parsing": len(parse_errors),
        "rejection_reasons": parse_errors,
        "selected_rows": len(selected),
        "quota_per_days_and_level": LEVEL_QUOTAS,
        "distribution_days_level": nested_distribution(selected, ("days", "level")),
        "distribution_days_level_reference_quartile": nested_distribution(
            selected, ("days", "level", "reference_bin")
        ),
        "constraint_signature_counts": dict(sorted(signatures.items())),
        "reference_chars": {
            "min": min(ref_chars),
            "max": max(ref_chars),
            "mean": round(sum(ref_chars) / len(ref_chars), 2),
        },
        "reference_items": {
            "min": min(ref_items),
            "max": max(ref_items),
            "mean": round(sum(ref_items) / len(ref_items), 2),
        },
        "diversity": {
            "unique_origins": len(selected_orgs),
            "unique_destinations": len(selected_dests),
            "unique_routes": len(selected_routes),
            "max_origin_frequency": max(selected_orgs.values()),
            "max_destination_frequency": max(selected_dests.values()),
            "max_route_frequency": max(selected_routes.values()),
        },
        "selected_source_rows_1_based": sorted(row["source_row"] for row in selected),
    }


def main() -> None:
    args = parse_args()
    rows, parse_errors = read_rows(args.csv, args.reference_info)
    assign_reference_bins(rows)
    selected = select_rows(rows, args.seed)
    if len(selected) != 360:
        raise AssertionError(f"Expected 360 selected rows, got {len(selected)}")
    tasks = make_tasks(selected, args.seed)
    report = build_report(rows, selected, args.seed, parse_errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(args.manifest, tasks)
    print(f"Selected {len(tasks)} tasks -> {args.output}")
    print(f"Report -> {args.report}")
    print(f"Manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
