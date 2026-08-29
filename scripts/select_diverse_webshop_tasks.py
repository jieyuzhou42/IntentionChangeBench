"""Select a diverse, reproducible subset of human WebShop goals.

The WebShop runtime assigns goal indices after scanning the product file in
file order.  This script mirrors that ordering, then performs balanced
round-robin sampling across product families while penalizing near-duplicate
queries and instructions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
PATH_SEPARATOR_RE = re.compile(r"\s*(?:›|>|â€º|\|)\s*")
STOPWORDS = {
    "a", "an", "and", "are", "be", "for", "i", "in", "is", "it",
    "looking", "me", "my", "of", "on", "or", "please", "that", "the",
    "this", "to", "want", "with",
}
CATEGORY_ALIASES = {
    "beauty": "beauty & personal care",
    "fashion": "clothing, shoes & jewelry",
    "grocery": "grocery & gourmet food",
}


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Stream values from a top-level JSON array without loading the file."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        finished = False

        while not finished:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer = buffer[position:] + chunk
                position = 0
            elif position >= len(buffer):
                break

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"Expected a JSON array in {path}")
                    started = True
                    position += 1
                    continue

                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    finished = True
                    position += 1
                    break

                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield value
                position = end

        if not finished:
            raise ValueError(f"Incomplete JSON array in {path}")


def normalize_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip().lower())


def tokens(value: Any) -> Set[str]:
    return {
        token for token in TOKEN_RE.findall(normalize_text(value))
        if token not in STOPWORDS and len(token) > 1
    }


def first_taxonomy_label(product: Dict[str, Any]) -> str:
    raw = normalize_text(product.get("product_category"))
    if raw:
        label = PATH_SEPARATOR_RE.split(raw)[0].strip(" ,-_")
        if label:
            return CATEGORY_ALIASES.get(label, label)
    category = product.get("category")
    if isinstance(category, list):
        category = category[0] if category else ""
    label = normalize_text(category) or "other"
    return CATEGORY_ALIASES.get(label, label)


def query_label(product: Dict[str, Any]) -> str:
    return normalize_text(product.get("query") or product.get("name")) or "unknown"


def build_candidates(
    products_path: Path,
    human_instructions_path: Path,
    num_products: int,
) -> List[Dict[str, Any]]:
    human_instructions = json.loads(human_instructions_path.read_text(encoding="utf-8"))
    candidates: List[Dict[str, Any]] = []
    seen_asins: Set[str] = set()
    goal_index = 0

    for raw_index, product in enumerate(iter_json_array(products_path)):
        if raw_index >= num_products:
            break
        if not isinstance(product, dict):
            continue
        asin = str(product.get("asin") or "")
        if asin == "nan" or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)

        for instruction in human_instructions.get(asin, []):
            attrs = instruction.get("instruction_attributes") or []
            if not attrs:
                continue
            instruction_text = str(instruction.get("instruction") or "").strip().rstrip(".")
            candidate = {
                "goal_index": goal_index,
                "asin": asin,
                "broad_category": first_taxonomy_label(product),
                "query": query_label(product),
                "name": str(product.get("name") or "").strip(),
                "product_category": str(product.get("product_category") or "").strip(),
                "instruction": instruction_text,
                "attributes": list(attrs),
                "options": instruction.get("instruction_options") or [],
            }
            candidate["tokens"] = tokens(
                " ".join(
                    [candidate["query"], candidate["name"], candidate["instruction"]]
                    + [str(value) for value in candidate["attributes"]]
                )
            )
            candidates.append(candidate)
            goal_index += 1

    return candidates


def jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def balanced_quotas(counts: Counter, target: int) -> Dict[str, int]:
    """Allocate nearly equal category quotas, redistributing shortages."""

    quotas = {key: 0 for key in counts}
    remaining = target
    active = set(counts)
    while remaining and active:
        share = max(1, math.ceil(remaining / len(active)))
        progressed = False
        for key in sorted(active):
            capacity = counts[key] - quotas[key]
            take = min(share, capacity, remaining)
            if take:
                quotas[key] += take
                remaining -= take
                progressed = True
            if quotas[key] >= counts[key]:
                active.discard(key)
            if not remaining:
                break
        if not progressed:
            break
    if remaining:
        raise ValueError(f"Only {target - remaining} eligible goals are available")
    return quotas


def select_diverse(
    candidates: Sequence[Dict[str, Any]],
    target: int,
    max_per_query: int,
) -> List[Dict[str, Any]]:
    if target > len(candidates):
        raise ValueError(f"Requested {target} goals, but only {len(candidates)} are eligible")

    for candidate in candidates:
        candidate.pop("selected", None)

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_category[candidate["broad_category"]].append(candidate)
    quotas = balanced_quotas(Counter({k: len(v) for k, v in by_category.items()}), target)

    query_capacity = sum(
        min(max_per_query, count)
        for count in Counter(row["query"] for row in candidates).values()
    )
    if query_capacity < target:
        raise ValueError(
            f"The max-per-query cap permits only {query_capacity} selections; "
            f"increase --max-per-query or reduce --count"
        )

    selected: List[Dict[str, Any]] = []
    query_counts: Counter = Counter()
    selected_tokens: List[Set[str]] = []
    selected_instructions: Set[str] = set()
    selected_asins: Set[str] = set()
    category_counts: Counter = Counter()
    min_similarity = {candidate["goal_index"]: 0.0 for candidate in candidates}

    while len(selected) < target:
        best: Dict[str, Any] | None = None
        best_score: Tuple[float, float, int, int] | None = None
        for category, pool in by_category.items():
            for candidate in pool:
                if candidate.get("selected"):
                    continue
                query_count = query_counts[candidate["query"]]
                if query_count >= max_per_query:
                    continue
                if candidate["asin"] in selected_asins:
                    continue
                normalized_instruction = normalize_text(candidate["instruction"])
                if normalized_instruction in selected_instructions:
                    continue
                similarity = min_similarity[candidate["goal_index"]]
                category_fill = category_counts[category] / max(1, quotas[category])
                score = (
                    category_fill,
                    similarity + 0.18 * query_count,
                    query_count,
                    candidate["goal_index"],
                )
                if best_score is None or score < best_score:
                    best = candidate
                    best_score = score

        if best is None:
            raise ValueError(
                "Could not fill the selection without duplicate instructions "
                "or violating --max-per-query"
            )
        best["selected"] = True
        selected.append(best)
        selected_tokens.append(best["tokens"])
        selected_instructions.add(normalize_text(best["instruction"]))
        selected_asins.add(best["asin"])
        query_counts[best["query"]] += 1
        category_counts[best["broad_category"]] += 1
        for candidate in candidates:
            if not candidate.get("selected"):
                index = candidate["goal_index"]
                min_similarity[index] = max(
                    min_similarity[index], jaccard(candidate["tokens"], best["tokens"])
                )

    return selected


def task_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    index = candidate["goal_index"]
    return {
        "instance_id": f"webshop_goal_{index:05d}",
        "task_type": "transaction",
        "subtype": "shopping",
        "world_state": {
            "domain": "webshop",
            "webshop_goal_index": index,
        },
        "initial_intention": {
            # Keep this empty so WebShop can expose the complete goal text for
            # the pinned index, including its generated price constraint.
            "request": "",
            "constraints": {},
            "priority": [],
        },
        "selection_metadata": {
            key: candidate[key]
            for key in (
                "asin", "broad_category", "query", "name", "product_category",
                "instruction", "attributes", "options",
            )
        },
    }


def selection_report(candidates: Sequence[Dict[str, Any]], selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pairs: List[Tuple[float, int, int]] = []
    for left_index, left in enumerate(selected):
        for right_index in range(left_index):
            score = jaccard(left["tokens"], selected[right_index]["tokens"])
            pairs.append((score, right_index, left_index))
    pairs.sort(reverse=True)
    return {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "unique_asins": len({row["asin"] for row in selected}),
        "unique_queries": len({row["query"] for row in selected}),
        "category_distribution": dict(sorted(Counter(row["broad_category"] for row in selected).items())),
        "most_common_queries": Counter(row["query"] for row in selected).most_common(20),
        "most_similar_pairs": [
            {
                "similarity": round(score, 4),
                "left_goal_index": selected[left]["goal_index"],
                "right_goal_index": selected[right]["goal_index"],
                "left_query": selected[left]["query"],
                "right_query": selected[right]["query"],
            }
            for score, left, right in pairs[:20]
        ],
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=repo_root.parent / "WebShop" / "data" / "items_shuffle.json")
    parser.add_argument("--human-instructions", type=Path, default=repo_root.parent / "WebShop" / "data" / "items_human_ins.json")
    parser.add_argument("--num-products", type=int, default=100000)
    parser.add_argument("--count", type=int, default=350)
    parser.add_argument("--max-per-query", type=int, default=3)
    parser.add_argument("--output", type=Path, default=repo_root / "data" / "webshop" / "diverse_350_tasks.json")
    parser.add_argument("--report", type=Path, default=repo_root / "data" / "webshop" / "diverse_350_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_candidates(args.products, args.human_instructions, args.num_products)
    selected = select_diverse(candidates, args.count, args.max_per_query)
    tasks = [task_payload(candidate) for candidate in selected]
    report = selection_report(candidates, selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
