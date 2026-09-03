from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_DECISION_CANDIDATE_LIMIT = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _tokens(values: Iterable[Any]) -> Set[str]:
    text = " ".join(_normalize_text(value) for value in values if value is not None)
    return {
        token
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _collection_values(value: Any, limit: int = 8) -> List[Any]:
    if isinstance(value, dict):
        flattened: List[Any] = []
        for key, item in list(value.items())[:limit]:
            flattened.extend([key, item])
        return flattened
    if isinstance(value, (list, tuple, set)):
        return list(value)[:limit]
    return [value] if value is not None else []


def _feature_tokens(item: Dict[str, Any]) -> Set[str]:
    values: List[Any] = [
        item.get("title"),
        item.get("query"),
        item.get("category"),
        item.get("product_category"),
    ]
    values.extend(_collection_values(item.get("bullet_points"), limit=4))
    values.extend(_collection_values(item.get("attributes"), limit=8))
    values.extend(_collection_values(item.get("options"), limit=6))
    return _tokens(values)


def _category_tokens(item: Dict[str, Any]) -> Set[str]:
    return _tokens(
        [
            item.get("product_category"),
            item.get("category"),
            item.get("query"),
        ]
    )


def _as_price(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(?:\$\s*)?([0-9]+(?:\.[0-9]+)?)", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _jaccard_distance(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return 1.0 - (len(left & right) / len(union)) if union else 0.0


def candidate_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    """Return a deterministic product-form/price/feature distance in [0, 1]."""

    feature_distance = _jaccard_distance(_feature_tokens(left), _feature_tokens(right))
    left_category = _category_tokens(left)
    right_category = _category_tokens(right)
    category_distance = _jaccard_distance(left_category, right_category)

    left_price = _as_price(left.get("price"))
    right_price = _as_price(right.get("price"))
    if left_price is None or right_price is None:
        price_distance = 0.0
    else:
        denominator = max(abs(left_price), abs(right_price), 1.0)
        price_distance = min(abs(left_price - right_price) / denominator, 1.0)

    return max(
        0.0,
        min(1.0, 0.45 * feature_distance + 0.35 * category_distance + 0.20 * price_distance),
    )


def _difference_axes(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    axes: List[str] = []
    if _jaccard_distance(_category_tokens(left), _category_tokens(right)) >= 0.35:
        axes.append("product_form_or_category")

    left_price = _as_price(left.get("price"))
    right_price = _as_price(right.get("price"))
    if left_price is not None and right_price is not None:
        relative_gap = abs(left_price - right_price) / max(left_price, right_price, 1.0)
        if relative_gap >= 0.20:
            axes.append("price")

    if _jaccard_distance(_feature_tokens(left), _feature_tokens(right)) >= 0.45:
        axes.append("core_features_or_use_case")
    return axes or ["variant_level_only"]


def _deduplicate_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        asin = _normalize_text(item.get("asin"))
        title = _normalize_text(item.get("title"))
        key = f"asin:{asin}" if asin else f"title:{title}"
        if not title and not asin:
            key = f"row:{index}"
        if key in seen:
            continue
        seen.add(key)
        copied = copy.deepcopy(item)
        copied.setdefault("original_rank", item.get("rank") or index)
        unique.append(copied)
    return unique


def select_diverse_candidates(
    candidates: Iterable[Dict[str, Any]],
    *,
    limit: int = DEFAULT_DECISION_CANDIDATE_LIMIT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Greedily retain relevant candidates while avoiding near-identical SKU variants."""

    unique = _deduplicate_candidates(candidates)
    target = min(max(int(limit), 0), len(unique))
    if target == 0:
        return [], {
            "strategy": "rank_relevance_plus_product_distance",
            "source_candidate_count": len(unique),
            "returned_candidate_count": 0,
            "target_range": [3, DEFAULT_DECISION_CANDIDATE_LIMIT],
            "selections": [],
        }

    selected_indices = [0]
    selection_details = [
        {
            "asin": unique[0].get("asin"),
            "original_rank": unique[0].get("original_rank"),
            "novelty_score": 1.0,
            "difference_axes": ["baseline"],
        }
    ]

    while len(selected_indices) < target:
        best_index: Optional[int] = None
        best_score = -math.inf
        best_novelty = 0.0
        best_nearest_index = selected_indices[0]
        for index, item in enumerate(unique):
            if index in selected_indices:
                continue
            distances = [
                (candidate_distance(item, unique[selected_index]), selected_index)
                for selected_index in selected_indices
            ]
            novelty, nearest_index = min(distances, key=lambda pair: pair[0])
            relevance = 1.0 / (1.0 + max(index, 0))
            score = 0.78 * novelty + 0.22 * relevance
            if score > best_score:
                best_index = index
                best_score = score
                best_novelty = novelty
                best_nearest_index = nearest_index

        if best_index is None:
            break
        selected_indices.append(best_index)
        selection_details.append(
            {
                "asin": unique[best_index].get("asin"),
                "original_rank": unique[best_index].get("original_rank"),
                "novelty_score": round(best_novelty, 4),
                "difference_axes": _difference_axes(
                    unique[best_index],
                    unique[best_nearest_index],
                ),
            }
        )

    selected: List[Dict[str, Any]] = []
    for rank, index in enumerate(selected_indices, start=1):
        item = unique[index]
        item["rank"] = rank
        item["diversity"] = copy.deepcopy(selection_details[rank - 1])
        selected.append(item)

    metadata = {
        "strategy": "rank_relevance_plus_product_distance",
        "source_candidate_count": len(unique),
        "returned_candidate_count": len(selected),
        "target_range": [3, DEFAULT_DECISION_CANDIDATE_LIMIT],
        "selections": selection_details,
    }
    return selected, metadata


__all__ = [
    "DEFAULT_DECISION_CANDIDATE_LIMIT",
    "candidate_distance",
    "select_diverse_candidates",
]
