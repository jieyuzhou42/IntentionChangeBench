from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prompt_logging import log_prompt


@dataclass(frozen=True)
class RerankerConfig:
    enable_reranking: bool = True
    rerank_top_n: int = 30
    rerank_return_k: int = 10
    reranker_model: Optional[str] = None
    reranker_debug: bool = False


def compact_candidate_for_reranking(
    candidate: Dict[str, Any],
    *,
    original_rank: int,
    include_description: bool = False,
) -> Dict[str, Any]:
    compact = {
        "asin": candidate.get("asin"),
        "original_rank": original_rank,
        "title": candidate.get("title"),
        "price": candidate.get("price"),
        "product_category": candidate.get("product_category") or candidate.get("category"),
        "query": candidate.get("query"),
        "attributes": _limit_collection(candidate.get("attributes"), 12),
        "options": candidate.get("options") or {},
        "brand": candidate.get("brand"),
        "bullet_points": [
            _clip_text(item, 240)
            for item in list(candidate.get("bullet_points") or [])[:3]
        ],
    }
    if include_description:
        compact["description"] = _clip_text(candidate.get("description"), 300)
    return compact


def rerank_candidates_with_llm(
    *,
    llm_client: Any,
    current_intention: Dict[str, Any],
    gold_delta: Optional[Dict[str, Any]] = None,
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
    model: Optional[str] = None,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not candidates:
        return [], {
            "enabled": True,
            "raw_candidate_count": 0,
            "returned_candidate_count": 0,
            "succeeded": True,
            "fallback_used": False,
            "raw_top_candidates": [],
            "reranked_top_candidates": [],
        }

    target_k = min(max(int(top_k), 0), len(candidates))
    compact_candidates = [
        compact_candidate_for_reranking(
            item,
            original_rank=_original_rank(item, index),
        )
        for index, item in enumerate(candidates, start=1)
    ]
    prompt = build_reranker_prompt(
        current_intention=current_intention,
        gold_delta=gold_delta or {},
        candidates=compact_candidates,
        top_k=target_k,
    )
    log_prompt(
        "executor.reranker",
        prompt,
        metadata={"model": model, "candidate_count": len(candidates), "top_k": target_k},
    )

    raw_output: Any = None
    try:
        parsed, raw_output = _call_reranker_llm(llm_client, prompt)
        reranked_items, reranked_summary = _apply_reranker_output(
            parsed=parsed,
            candidates=candidates,
            top_k=target_k,
        )
    except Exception as exc:
        return _fallback_candidates(
            candidates,
            top_k=target_k,
            error=f"{type(exc).__name__}: {exc}",
            raw_output=raw_output,
            compact_candidates=compact_candidates,
            debug=debug,
        )

    info = {
        "enabled": True,
        "raw_candidate_count": len(candidates),
        "returned_candidate_count": len(reranked_items),
        "succeeded": True,
        "fallback_used": False,
        "raw_top_candidates": [
            {
                "asin": item.get("asin"),
                "original_rank": _original_rank(item, index),
            }
            for index, item in enumerate(candidates, start=1)
        ],
        "reranked_top_candidates": reranked_summary,
    }
    if debug:
        info["compact_reranker_input"] = compact_candidates
        info["raw_reranker_output"] = raw_output
    return reranked_items, info


def build_reranker_prompt(
    *,
    current_intention: Dict[str, Any],
    gold_delta: Optional[Dict[str, Any]] = None,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> str:
    payload = {
        "current_shopping_intention": _reranker_intention_payload(current_intention),
        "gold_delta": copy.deepcopy(gold_delta or {}),
        "candidate_items": candidates,
    }
    return f"""
You are a constraint-aware reranker for WebShop search results.

Return a single JSON object only.

Task:
Given the current shopping intention and a list of candidate items, select the top {top_k} candidates that best match the intention.

Important:
- Use the candidate item fields as evidence.
- Prioritize matching the product type/category first.
- Do not over-trust noisy keyword-stuffed descriptions.
- If an item is clearly the wrong product type, rank it low even if it contains many matching keywords.
- If an exact fine-grained constraint is not explicitly supported but the item is in the correct product family, it can still be kept if better options are unavailable.
- Do not invent attributes that are not supported by the item fields.

Scoring guidance:
- First apply the product family gate:
  - If the candidate is clearly the wrong product family, give it a very low score and rank it below plausible product-family matches.
  - Wrong product-family items should not appear in the final top {top_k} unless fewer than {top_k} plausible candidates exist.
- Then score constraints from current_shopping_intention.constraints.
- Use gold_delta to weight the changed constraint:
  - If a gold_delta entry has op="add", and the candidate clearly satisfies that entry's new value, add 4 points.
  - If the added constraint is missing or contradicted, add 0 points for that added constraint.
  - For every other current constraint that the candidate clearly satisfies, add 2 points.
  - If there is no add entry in gold_delta, score all current constraints as "other current constraints" worth 2 points each.
- Budget/price, size, and color must be checked from reliable item fields when possible:
  - budget_max from price
  - size from options and relevant title/attributes/bullet_points
  - color from options, title, attributes, bullet_points, or explicit color field
- Do not add points for unsupported guesses.
- If evidence is uncertain, mention it briefly and do not award full points for that constraint.

Return this JSON schema:
{{
  "reranked_candidates": [
    {{
      "asin": "candidate asin",
      "original_rank": 1,
      "new_rank": 1,
      "score": 12,
      "product_family_match": "exact | broad | related | wrong",
      "gold_delta_score": 4,
      "other_constraints_score": 8,
      "decision": "keep",
      "matched_constraints": ["..."],
      "missing_or_uncertain_constraints": ["..."],
      "mismatch_reasons": ["..."],
      "brief_reason": "concise reason"
    }}
  ]
}}

Rules:
- Return exactly {top_k} candidates if at least {top_k} candidates are available.
- Preserve candidate ASINs exactly.
- Do not include candidates that are clearly wrong product type unless fewer than {top_k} plausible candidates exist.
- Keep explanations concise.

Input JSON:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _reranker_intention_payload(current_intention: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(current_intention or {})
    payload.pop("gold_search_query", None)
    if "constraints" not in payload or not isinstance(payload.get("constraints"), dict):
        payload["constraints"] = {}
    return payload


def _call_reranker_llm(llm_client: Any, prompt: str) -> Tuple[Dict[str, Any], Any]:
    generate_json_text = getattr(llm_client, "generate_json_text", None)
    if callable(generate_json_text):
        raw_text = generate_json_text(prompt)
        return _parse_json_like(raw_text), raw_text

    generate_json = getattr(llm_client, "generate_json", None)
    if callable(generate_json):
        parsed = generate_json(prompt)
        if not isinstance(parsed, dict):
            raise ValueError("reranker JSON response was not an object")
        return parsed, parsed

    generate_text = getattr(llm_client, "generate_text", None)
    if callable(generate_text):
        raw_text = generate_text(prompt)
        return _parse_json_like(raw_text), raw_text

    raise ValueError("LLM client has no supported JSON/text generation method")


def _apply_reranker_output(
    *,
    parsed: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(parsed, dict):
        raise ValueError("reranker response is not a JSON object")
    rows = parsed.get("reranked_candidates")
    if not isinstance(rows, list):
        raise ValueError("reranker response missing reranked_candidates list")
    if len(rows) < top_k:
        raise ValueError(f"reranker returned {len(rows)} candidates, expected {top_k}")

    by_asin = {
        _asin_key(candidate.get("asin")): candidate
        for candidate in candidates
        if _asin_key(candidate.get("asin"))
    }
    selected: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    seen = set()
    for output_rank, row in enumerate(rows[:top_k], start=1):
        if not isinstance(row, dict):
            raise ValueError("reranker candidate entry is not an object")
        asin = row.get("asin")
        asin_key = _asin_key(asin)
        if not asin_key or asin_key not in by_asin:
            raise ValueError(f"reranker returned unknown ASIN: {asin!r}")
        if asin_key in seen:
            raise ValueError(f"reranker returned duplicate ASIN: {asin!r}")
        seen.add(asin_key)

        item = copy.deepcopy(by_asin[asin_key])
        original_rank = _original_rank(item, None)
        new_rank = _coerce_int(row.get("new_rank"), output_rank) or output_rank
        item["original_rank"] = original_rank
        item["rerank_rank"] = new_rank
        item["rerank_score"] = _coerce_float(row.get("score"))
        item["rerank_product_type_match"] = _coerce_float(row.get("product_type_match"))
        item["rerank_hard_constraint_match"] = _coerce_float(row.get("hard_constraint_match"))
        item["rerank_soft_constraint_match"] = _coerce_float(row.get("soft_constraint_match"))
        item["rerank_evidence_quality"] = _coerce_float(row.get("evidence_quality"))
        item["rerank_product_family_match"] = _clip_text(row.get("product_family_match"), 80)
        item["rerank_gold_delta_score"] = _coerce_float(row.get("gold_delta_score"))
        item["rerank_other_constraints_score"] = _coerce_float(row.get("other_constraints_score"))
        item["rerank_brief_reason"] = _clip_text(row.get("brief_reason"), 300)
        item["rerank_matched_constraints"] = _string_list(row.get("matched_constraints"))
        item["rerank_missing_or_uncertain_constraints"] = _string_list(
            row.get("missing_or_uncertain_constraints")
        )
        item["rerank_mismatch_reasons"] = _string_list(row.get("mismatch_reasons"))
        selected.append(item)
        summary.append(
            {
                "asin": item.get("asin"),
                "original_rank": original_rank,
                "rerank_rank": new_rank,
                "score": item.get("rerank_score"),
            }
        )

    return selected, summary


def _fallback_candidates(
    candidates: List[Dict[str, Any]],
    *,
    top_k: int,
    error: str,
    raw_output: Any,
    compact_candidates: List[Dict[str, Any]],
    debug: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fallback = []
    for index, item in enumerate(candidates[:top_k], start=1):
        copied = copy.deepcopy(item)
        copied["original_rank"] = _original_rank(copied, index)
        copied["rerank_failed"] = True
        copied["rerank_error"] = error
        copied["rerank_fallback"] = "bm25_top10"
        fallback.append(copied)

    info = {
        "enabled": True,
        "raw_candidate_count": len(candidates),
        "returned_candidate_count": len(fallback),
        "succeeded": False,
        "fallback_used": True,
        "rerank_failed": True,
        "rerank_error": error,
        "rerank_fallback": "bm25_top10",
        "raw_top_candidates": [
            {
                "asin": item.get("asin"),
                "original_rank": _original_rank(item, index),
            }
            for index, item in enumerate(candidates, start=1)
        ],
        "reranked_top_candidates": [
            {
                "asin": item.get("asin"),
                "original_rank": item.get("original_rank"),
                "rerank_rank": None,
                "score": None,
            }
            for item in fallback
        ],
    }
    if debug:
        info["compact_reranker_input"] = compact_candidates
        info["raw_reranker_output"] = raw_output
    return fallback, info


def _parse_json_like(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("reranker returned non-text, non-dict output")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("reranker JSON output was not an object")
    return parsed


def _original_rank(item: Dict[str, Any], fallback: Optional[int]) -> Optional[int]:
    return _coerce_int(item.get("original_rank"), None) or _coerce_int(item.get("rank"), fallback)


def _asin_key(value: Any) -> str:
    return str(value or "").strip().upper()


def _clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _limit_collection(value: Any, limit: int) -> Any:
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, dict):
        return dict(list(value.items())[:limit])
    return value


def _coerce_int(value: Any, default: Optional[int]) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_clip_text(item, 180) for item in value if str(item or "").strip()]


__all__ = [
    "RerankerConfig",
    "build_reranker_prompt",
    "compact_candidate_for_reranking",
    "rerank_candidates_with_llm",
]
