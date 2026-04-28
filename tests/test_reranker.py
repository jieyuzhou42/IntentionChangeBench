from __future__ import annotations

import json
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.reranker import (
    build_reranker_prompt,
    compact_candidate_for_reranking,
    rerank_candidates_with_llm,
)


class FakeLLMClient:
    def __init__(self, payload):
        self.payload = payload

    def generate_json_text(self, prompt):
        return json.dumps(self.payload)


class BadLLMClient:
    def generate_json_text(self, prompt):
        return "{not valid json"


def _fake_candidate(index, *, family="dress"):
    return {
        "asin": f"ASIN{index:03d}",
        "rank": index,
        "title": f"Women's green {family} item {index}",
        "price": 24.99 + index,
        "product_category": "Clothing",
        "query": "green dress",
        "attributes": ["machine wash", "polyester", "slim fit"],
        "options": {"color": ["green", "black"], "size": ["M", "L"]},
        "brand": "Example",
        "bullet_points": [
            "Soft fabric",
            "Machine washable",
            "Daily wear",
            "Fourth bullet should not be passed to reranker",
        ],
        "description": "Noisy description " * 80,
    }


def test_compact_candidate_limits_verbose_fields():
    compact = compact_candidate_for_reranking(_fake_candidate(1), original_rank=1)

    assert compact["asin"] == "ASIN001"
    assert compact["original_rank"] == 1
    assert len(compact["bullet_points"]) == 3
    assert "description" not in compact


def test_reranker_prompt_uses_gold_delta_weights_without_gold_search_query():
    prompt = build_reranker_prompt(
        current_intention={
            "gold_search_query": "women green jumpsuit",
            "constraints": {
                "category": "women's jumpsuits, rompers & overalls",
                "budget_max": 60.0,
                "color": "green stripe",
                "size": "large",
            },
        },
        gold_delta={
            "color": {
                "op": "add",
                "old": None,
                "new": "green stripe",
                "rationale": "user added color preference",
            }
        },
        candidates=[compact_candidate_for_reranking(_fake_candidate(1), original_rank=1)],
        top_k=1,
    )

    assert '"gold_delta"' in prompt
    assert '"gold_search_query"' not in prompt
    assert "add 4 points" in prompt
    assert "add 2 points" in prompt
    assert "hard_constraint" not in prompt
    assert "soft_constraint" not in prompt


def test_rerank_candidates_uses_order_and_attaches_metadata():
    candidates = [
        _fake_candidate(i, family="dress" if i > 10 else "phone case")
        for i in range(1, 31)
    ]
    ordered_asins = [f"ASIN{i:03d}" for i in range(30, 20, -1)]
    payload = {
        "reranked_candidates": [
            {
                "asin": asin,
                "original_rank": int(asin[-3:]),
                "new_rank": rank,
                "score": 8.5 - (rank * 0.1),
                "product_type_match": 3,
                "hard_constraint_match": 3,
                "soft_constraint_match": 2,
                "evidence_quality": 0.5,
                "decision": "keep",
                "matched_constraints": ["green", "size L"],
                "missing_or_uncertain_constraints": [],
                "mismatch_reasons": [],
                "brief_reason": "Correct family and constraints.",
            }
            for rank, asin in enumerate(ordered_asins, start=1)
        ]
    }

    reranked, info = rerank_candidates_with_llm(
        llm_client=FakeLLMClient(payload),
        current_intention={"gold_search_query": "green dress size L"},
        candidates=candidates,
        top_k=10,
    )

    assert len(reranked) == 10
    assert [item["asin"] for item in reranked] == ordered_asins
    assert reranked[0]["original_rank"] == 30
    assert reranked[0]["rerank_rank"] == 1
    assert reranked[0]["rerank_score"] == 8.4
    assert reranked[0]["rerank_product_type_match"] == 3
    assert reranked[0]["rerank_matched_constraints"] == ["green", "size L"]
    assert info["succeeded"] is True
    assert info["fallback_used"] is False


def test_rerank_candidates_falls_back_to_bm25_top_10_on_invalid_output():
    candidates = [_fake_candidate(i) for i in range(1, 31)]

    reranked, info = rerank_candidates_with_llm(
        llm_client=BadLLMClient(),
        current_intention={"gold_search_query": "green dress"},
        candidates=candidates,
        top_k=10,
    )

    assert len(reranked) == 10
    assert [item["asin"] for item in reranked] == [f"ASIN{i:03d}" for i in range(1, 11)]
    assert reranked[0]["original_rank"] == 1
    assert reranked[0]["rerank_failed"] is True
    assert reranked[0]["rerank_fallback"] == "bm25_top10"
    assert "rerank_error" in reranked[0]
    assert info["succeeded"] is False
    assert info["fallback_used"] is True
