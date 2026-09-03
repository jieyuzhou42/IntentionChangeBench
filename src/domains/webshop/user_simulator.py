from __future__ import annotations

import copy
import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from domains.webshop.candidate_diversity import candidate_distance
from models import EnvFeedback, ShiftOp
from simulation.simulation.base_user_simulator import (
    REALIZATION_CONTEXT_MARKER,
    SHIFT_CONTEXT_MARKER,
    HumanSimulator,
    ShiftDistributionController,
    _safe_json_dumps,
)


_HIGH_IMPACT_DIMENSION_TERMS = {
    "category",
    "durability",
    "function",
    "manual",
    "material",
    "price",
    "product form",
    "size capacity",
    "stability",
    "use case",
}
_LOW_IMPACT_DIMENSION_TERMS = {
    "color",
    "dimensions",
    "exact size",
    "finish",
    "pattern",
    "style variant",
}


class WebShopUserSimulator(HumanSimulator):
    """Result-driven WebShop simulator that generates decision-level changes."""

    def __init__(self, llm_client: Any):
        super().__init__(llm_client=llm_client)
        self.reset_trajectory()

    def reset_trajectory(self) -> None:
        self._recent_decisions: List[Dict[str, Any]] = []

    def _infer_domain(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> str:
        return "webshop"

    def _build_shift_prompt(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
        current_gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
        distribution_guidance: Optional[Dict[str, Any]] = None,
    ) -> str:
        context: Dict[str, Any] = {
            "intention_timeline": self._serialize_intention_timeline(
                current_intention,
                intention_history,
                current_gold_delta=current_gold_delta,
            ),
            "latest_env_feedback": self._serialize_env_feedback(env_feedback),
            "recent_product_decisions": copy.deepcopy(self._recent_decisions[-2:]),
        }
        if distribution_guidance:
            context["distribution_guidance"] = copy.deepcopy(distribution_guidance)

        instructions = """
Pretend you are a real user shopping on amazon.com.
Return a single JSON object only.

Generation objective:
- Given the current intention and the differentiated candidate products, choose a product difference that could change the user's purchase decision, then generate the next intention.
- This is not a task of gradually revealing attributes from one preselected target SKU.
- The user is dissatisfied, persuaded, or newly aware of a trade-off and MUST make at least one genuine change.

Evidence and decision-point rules:
- Inspect the 3-5 candidate_items and compare at least two distinct ASINs. If fewer are available, use only the real candidates present and never invent a product.
- Choose one consequential decision point grounded in an explicit candidate difference: product form/category, price, core function, durability/material, capacity, adjustment method, or use case.
- A realistic adjacent substitute is allowed when it serves the underlying use case.
- Color, finish, exact dimensions, or a title fragment alone are normally variant-level clarification, not an intention change. Use them only as part of a broader, evidence-backed purchase trade-off.
- Every evidence ASIN must occur in candidate_items. State what each compared option enables or sacrifices.
- Express the user's purchase reason, not a catalog-title phrase. Never copy a full title or mechanically promote successive words from one title into constraints.

Trajectory rules:
- Preserve a coherent path from the current intention and earlier changes; do not toggle repeatedly between two values.
- Consult recent_product_decisions. If the same chosen_asin has driven two consecutive changes, choose a different product or a genuinely new cross-product trade-off when one is supported.
- Prefer a candidate that exposes a new trade-off over extracting another attribute from the previously chosen SKU.

Change rules:
- Allowed conditions: user_preference, real_world_feasibility.
- Allowed categories: add, relax, override, reprioritize.
- One decision may naturally cause several related changes. Include all of them in changes; do not split one thought into mechanical turns.
- A realistic user can change multiple constraints in the same turn when they are consequences of one coherent decision point.
- Examples of coherent compound changes include relaxing from a complete desk to a converter while adding two-monitor capacity, or keeping a complete electric desk while relaxing the budget.
- Reprioritize when the evidence makes an existing factor matter more or less; do not merely append an attribute.
- Do not mention ratings, reviews, stars, or customer scores.
- Preserve a coherent trajectory above every diversity objective.
- Treat distribution_guidance only as a weak tie-breaker between equally natural, evidence-grounded decisions.

Required JSON schema:
{
  "intention_changed": true,
  "condition": "user_preference | real_world_feasibility",
  "decision_point": {
    "dimension": "decision-relevant comparison dimension",
    "options_compared": [
      {"asin": "real candidate ASIN", "option": "plain-language product form", "tradeoff": "what it enables or sacrifices"},
      {"asin": "different real candidate ASIN", "option": "plain-language product form", "tradeoff": "what it enables or sacrifices"}
    ],
    "chosen_option": "the direction the user now prefers",
    "chosen_asin": "supporting candidate ASIN or null",
    "purchase_reason": "why this difference changes the decision"
  },
  "changes": [
    {
      "category": "add | relax | override | reprioritize",
      "field": "constraint field name or null",
      "old_value": "previous value or null",
      "value": "new value or null",
      "priority_update": ["ordered", "priority", "fields"] or null,
      "rationale": "how this change follows from the decision point"
    }
  ],
  "rationale": "short purchase-level explanation",
  "utterance_plan": {
    "style": "explicit | partial | elliptical",
    "directness": "direct | indirect",
    "mention_old_value": true
  }
}

Good example:
{
  "intention_changed": true,
  "condition": "user_preference",
  "decision_point": {
    "dimension": "product form and usable workspace",
    "options_compared": [
      {"asin": "CONVERTER1", "option": "desktop converter", "tradeoff": "lower price but must fit the existing desk"},
      {"asin": "FULLDESK01", "option": "complete electric desk", "tradeoff": "more workspace but costs more"}
    ],
    "chosen_option": "desktop converter with room for two monitors",
    "chosen_asin": "CONVERTER1",
    "purchase_reason": "keeping the existing desk is practical if the work surface is large enough"
  },
  "changes": [
    {"category": "relax", "field": "category", "old_value": "standing desk", "value": "standing desk converter", "priority_update": null, "rationale": "a converter serves the same use case"},
    {"category": "add", "field": "monitor_capacity", "old_value": null, "value": "two monitors", "priority_update": null, "rationale": "the smaller form still needs enough workspace"}
  ],
  "rationale": "A converter is acceptable if it preserves dual-monitor workspace.",
  "utterance_plan": {"style": "explicit", "directness": "direct", "mention_old_value": false}
}
""".strip()
        return f"{instructions}\n\n{SHIFT_CONTEXT_MARKER}\n{_safe_json_dumps(context)}"

    def decide_shift(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
        current_gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
        candidate_samples: int = 1,
        max_candidate_samples: Optional[int] = None,
        prefer_multi: bool = False,
        rng: Optional[random.Random] = None,
        distribution_controller: Optional[ShiftDistributionController] = None,
    ) -> ShiftOp:
        observation = (env_feedback.observation or {}) if env_feedback is not None else {}
        has_comparison_pool = len(observation.get("candidate_items") or []) >= 2
        sample_limit = max_candidate_samples
        if has_comparison_pool:
            sample_limit = max(int(sample_limit or candidate_samples), 3)
        return super().decide_shift(
            current_intention,
            env_feedback=env_feedback,
            intention_history=intention_history,
            current_gold_delta=current_gold_delta,
            candidate_samples=candidate_samples,
            max_candidate_samples=sample_limit,
            prefer_multi=prefer_multi,
            rng=rng,
            distribution_controller=distribution_controller,
        )

    def _postprocess_shift_candidate(
        self,
        llm_output: Dict[str, Any],
        shift: ShiftOp,
        *,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback],
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> ShiftOp:
        if shift.op == "none":
            return shift

        decision_point, validation = self._normalize_decision_point(
            llm_output.get("decision_point"),
            env_feedback,
        )
        available_count = validation["available_candidate_count"]
        # Preserve direct parser callers when no product evidence was supplied.
        # Real result-driven turns with two or more products require a valid
        # cross-product decision point.
        if available_count >= 2 and not validation["valid"]:
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale=f"invalid_decision_point: {validation['reason']}",
                sampling_metadata={"decision_validation": validation},
            )

        shift.sampling_metadata.update(
            {
                "decision_point": decision_point,
                "evidence_asins": validation["evidence_asins"],
                "chosen_asin": validation["chosen_asin"],
                "decision_validation": validation,
                "decision_quality_score": self._decision_quality_score(decision_point),
            }
        )
        return shift

    def _normalize_decision_point(
        self,
        raw_decision_point: Any,
        env_feedback: Optional[EnvFeedback],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        observation = (env_feedback.observation or {}) if env_feedback is not None else {}
        candidates = [
            item
            for item in observation.get("candidate_items") or []
            if isinstance(item, dict)
        ]
        available_asins = {
            str(item.get("asin") or "").strip().upper()
            for item in candidates
            if item.get("asin")
        }
        candidates_by_asin = {
            str(item.get("asin") or "").strip().upper(): item
            for item in candidates
            if item.get("asin")
        }
        validation: Dict[str, Any] = {
            "valid": False,
            "reason": "missing_decision_point",
            "available_candidate_count": len(available_asins),
            "evidence_asins": [],
            "chosen_asin": None,
        }
        if not isinstance(raw_decision_point, dict):
            return None, validation

        dimension = self._clean_text(raw_decision_point.get("dimension"))
        chosen_option = self._clean_text(raw_decision_point.get("chosen_option"))
        purchase_reason = self._clean_text(raw_decision_point.get("purchase_reason"))
        normalized_options: List[Dict[str, Any]] = []
        evidence_asins: List[str] = []
        for option in raw_decision_point.get("options_compared") or []:
            if not isinstance(option, dict):
                continue
            asin = str(option.get("asin") or "").strip().upper()
            if not asin or asin not in available_asins or asin in evidence_asins:
                continue
            evidence_asins.append(asin)
            normalized_options.append(
                {
                    "asin": asin,
                    "option": self._clean_text(option.get("option")),
                    "tradeoff": self._clean_text(option.get("tradeoff")),
                }
            )

        chosen_asin = str(raw_decision_point.get("chosen_asin") or "").strip().upper() or None
        if chosen_asin not in evidence_asins:
            chosen_asin = None

        normalized = {
            "dimension": dimension,
            "options_compared": normalized_options,
            "chosen_option": chosen_option,
            "chosen_asin": chosen_asin,
            "purchase_reason": purchase_reason,
        }
        evidence_distances = [
            candidate_distance(candidates_by_asin[left], candidates_by_asin[right])
            for left_index, left in enumerate(evidence_asins)
            for right in evidence_asins[left_index + 1 :]
        ]
        max_evidence_distance = max(evidence_distances) if evidence_distances else 0.0
        reason = "ok"
        if not dimension or not chosen_option or not purchase_reason:
            reason = "missing_decision_explanation"
        elif len(evidence_asins) < min(2, len(available_asins)):
            reason = "fewer_than_two_distinct_real_evidence_asins"
        elif any(not option["option"] or not option["tradeoff"] for option in normalized_options):
            reason = "missing_compared_option_tradeoff"
        elif max_evidence_distance < 0.12:
            reason = "near_duplicate_evidence_products"
        elif (
            any(term in dimension.lower() for term in _LOW_IMPACT_DIMENSION_TERMS)
            and not any(term in dimension.lower() for term in _HIGH_IMPACT_DIMENSION_TERMS)
        ):
            reason = "variant_level_decision_point"
        else:
            validation["valid"] = True

        validation.update(
            {
                "reason": reason,
                "evidence_asins": evidence_asins,
                "chosen_asin": chosen_asin,
                "max_evidence_product_distance": round(max_evidence_distance, 4),
            }
        )
        return normalized, validation

    def _decision_quality_score(self, decision_point: Optional[Dict[str, Any]]) -> float:
        if not decision_point:
            return 0.0
        dimension = self._clean_text(decision_point.get("dimension")).lower()
        score = 1.0
        if any(term in dimension for term in _HIGH_IMPACT_DIMENSION_TERMS):
            score += 1.0
        if any(term in dimension for term in _LOW_IMPACT_DIMENSION_TERMS):
            score -= 0.75
        score += 0.25 * min(len(decision_point.get("options_compared") or []), 3)
        return round(score, 3)

    def _prepare_shift_selection_pool(
        self,
        candidates: List[ShiftOp],
        *,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback],
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[List[ShiftOp], Dict[str, Any]]:
        if not candidates:
            return candidates, {"strategy": "decision_quality_and_sku_rotation"}

        recent_chosen = [
            str(item.get("chosen_asin") or "").strip().upper()
            for item in self._recent_decisions[-2:]
            if item.get("chosen_asin")
        ]
        dominated_asin = (
            recent_chosen[-1]
            if len(recent_chosen) >= 2 and recent_chosen[-1] == recent_chosen[-2]
            else None
        )
        diagnostics: List[Dict[str, Any]] = []
        non_dominated: List[ShiftOp] = []
        for candidate in candidates:
            metadata = candidate.sampling_metadata or {}
            chosen_asin = str(metadata.get("chosen_asin") or "").strip().upper() or None
            quality = float(metadata.get("decision_quality_score") or 0.0)
            repeated_after_dominance = bool(dominated_asin and chosen_asin == dominated_asin)
            if repeated_after_dominance:
                quality -= 2.0
            else:
                non_dominated.append(candidate)
            diagnostics.append(
                {
                    "chosen_asin": chosen_asin,
                    "evidence_asins": copy.deepcopy(metadata.get("evidence_asins") or []),
                    "single_sku_dominance_penalty": 2.0 if repeated_after_dominance else 0.0,
                    "quality_score_after_rotation_penalty": quality,
                    "repeated_after_two_turn_dominance": repeated_after_dominance,
                }
            )

        filtered = non_dominated if dominated_asin and non_dominated else candidates
        best_quality = max(
            float((candidate.sampling_metadata or {}).get("decision_quality_score") or 0.0)
            for candidate in filtered
        )
        quality_pool = [
            candidate
            for candidate in filtered
            if float((candidate.sampling_metadata or {}).get("decision_quality_score") or 0.0)
            >= best_quality - 0.75
        ]
        return quality_pool or filtered, {
            "strategy": "decision_quality_and_sku_rotation",
            "recent_chosen_asins": recent_chosen,
            "dominated_asin": dominated_asin,
            "filtered_repeated_dominant_candidates": len(candidates) - len(filtered),
            "candidate_diagnostics": diagnostics,
        }

    def _should_resample_shift_candidate(
        self,
        shift: ShiftOp,
        *,
        sampled_count: int,
        sample_limit: int,
    ) -> bool:
        return (
            sampled_count < sample_limit
            and shift.op == "none"
            and shift.rationale.startswith("invalid_decision_point")
        )

    def _on_shift_selected(
        self,
        shift: ShiftOp,
        *,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback],
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> None:
        decision_point = copy.deepcopy((shift.sampling_metadata or {}).get("decision_point"))
        if not decision_point:
            return
        self._recent_decisions.append(
            {
                "dimension": decision_point.get("dimension"),
                "chosen_option": decision_point.get("chosen_option"),
                "chosen_asin": decision_point.get("chosen_asin"),
                "evidence_asins": copy.deepcopy(shift.sampling_metadata.get("evidence_asins") or []),
            }
        )
        self._recent_decisions = self._recent_decisions[-2:]

    def _build_realization_prompt(
        self,
        shift: ShiftOp,
        current_intention: Dict[str, Any],
        style: str,
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
        current_gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        base_prompt = super()._build_realization_prompt(
            shift,
            current_intention,
            style,
            env_feedback=env_feedback,
            intention_history=intention_history,
            current_gold_delta=current_gold_delta,
        )
        instructions, context = base_prompt.split(REALIZATION_CONTEXT_MARKER, 1)
        decision_point = copy.deepcopy((shift.sampling_metadata or {}).get("decision_point"))
        extra = """
WebShop realization rules:
- Express why the selected product difference changes what the user wants; do not recite fields as a checklist.
- Do not copy a candidate title. Never reproduce six or more consecutive words from any title.
- Translate catalog features into a purchase reason or trade-off in ordinary user language.
- Keep related multi-constraint changes together in the same natural utterance.
""".strip()
        decision_context = json.dumps(decision_point, ensure_ascii=False, default=str)
        return (
            f"{instructions.rstrip()}\n\n{extra}\n"
            f"Selected decision point: {decision_context}\n\n"
            f"{REALIZATION_CONTEXT_MARKER}{context}"
        )

    def realize_shift(
        self,
        shift: ShiftOp,
        current_intention: Dict[str, Any],
        style: str,
        env_feedback: Optional[EnvFeedback] = None,
        intention_history: Optional[List[Dict[str, Any]]] = None,
        current_gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        prompt = self._build_realization_prompt(
            shift,
            current_intention,
            style,
            env_feedback=env_feedback,
            intention_history=intention_history,
            current_gold_delta=current_gold_delta,
        )
        utterance = self._call_llm_for_realization(prompt, strict=False)
        if utterance and not self._looks_like_title_copy(utterance, env_feedback):
            return utterance
        if utterance:
            retry_instructions, retry_context = prompt.split(
                REALIZATION_CONTEXT_MARKER,
                1,
            )
            retry_prompt = (
                f"{retry_instructions.rstrip()}\n\n"
                "CRITICAL: Paraphrase the purchase reason. Do not copy wording from a product title.\n\n"
                f"{REALIZATION_CONTEXT_MARKER}{retry_context}"
            )
            retry = self._call_llm_for_realization(retry_prompt, strict=False)
            if retry and not self._looks_like_title_copy(retry, env_feedback):
                return retry
        return self._fallback_realization(shift, style)

    def _looks_like_title_copy(
        self,
        utterance: str,
        env_feedback: Optional[EnvFeedback],
        ngram_size: int = 6,
    ) -> bool:
        utterance_tokens = self._word_tokens(utterance)
        if len(utterance_tokens) < ngram_size:
            return False
        utterance_ngrams = {
            tuple(utterance_tokens[index : index + ngram_size])
            for index in range(len(utterance_tokens) - ngram_size + 1)
        }
        observation = (env_feedback.observation or {}) if env_feedback is not None else {}
        for item in observation.get("candidate_items") or []:
            if not isinstance(item, dict):
                continue
            title_tokens = self._word_tokens(item.get("title"))
            for index in range(len(title_tokens) - ngram_size + 1):
                if tuple(title_tokens[index : index + ngram_size]) in utterance_ngrams:
                    return True
        return False

    @staticmethod
    def _word_tokens(value: Any) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(value or "").lower())

    @staticmethod
    def _clean_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()


__all__ = ["WebShopUserSimulator"]
