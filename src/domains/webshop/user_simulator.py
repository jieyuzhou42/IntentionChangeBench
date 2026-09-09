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
_IDENTITY_FIELDS = {
    "category",
    "core_function",
    "format",
    "product_form",
    "product_type",
    "style",
    "use_case",
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
- Candidate products are trigger evidence, never the new target answer.
- Preserve the user's stable purchase goal, compare the differentiated candidates, and extract exactly one decision-relevant trade-off that changes one purchase criterion.
- Do not adopt a candidate's full product form, feature bundle, packaging, accessories, or title as the next intention.
- This is not a task of gradually revealing attributes from one preselected target SKU.
- The user is dissatisfied, persuaded, or newly aware of a trade-off and MUST make one genuine, focused change.

Evidence and decision-point rules:
- Inspect the 3-5 candidate_items and compare at least two distinct ASINs. If fewer are available, use only the real candidates present and never invent a product.
- State stable_purchase_goal as the underlying job the user is still trying to accomplish; it must not be a candidate title or SKU-specific description.
- Set evidence_role to trigger_only: products justify the trade-off but do not define the new target.
- Choose one consequential decision point grounded in an explicit candidate difference: product form/category, price, core function, durability/material, capacity, adjustment method, or use case.
- A realistic adjacent substitute is allowed when it serves the underlying use case.
- Color, finish, exact dimensions, or a title fragment alone are normally variant-level clarification, not an intention change. Use them only as part of a broader, evidence-backed purchase trade-off.
- Every evidence ASIN must occur in candidate_items. State what each compared option enables or sacrifices.
- Express the user's purchase reason, not a catalog-title phrase. Never copy a full title or mechanically promote successive words from one title into constraints.

Trajectory rules:
- Preserve a coherent path from the current intention and earlier changes; do not toggle repeatedly between two values.
- Inspect the recent gold_delta entries before changing a field. Do not return a field to a recent value, reverse a numeric direction on the next turn, or modify the same field for a third consecutive turn.
- After choosing a product form or category, keep that commitment long enough to compare within it. Do not bounce among adjacent categories merely because each search exposes another plausible product.
- Consult recent_product_decisions. A chosen_asin is supporting evidence only and cannot drive another change within the next two turns.
- Do not add a fresh product attribute immediately after another add. The next turn must revise, relax, override, or reprioritize the current decision before any further addition.
- chosen_option must describe an abstract trade-off direction, never selecting a product or restating its form and feature bundle.
- Prefer a candidate that exposes a new trade-off over extracting another attribute from the previously chosen SKU.

Change rules:
- Allowed conditions: user_preference, real_world_feasibility.
- Allowed categories: add, relax, override, reprioritize.
- Return exactly one primary substantive change in changes. An optional reprioritize entry may accompany it, but do not add secondary product features.
- Change the abstract purchase criterion exposed by the trade-off, not every property of the supporting product.
- If the trade-off is cheaper versus more durable, change budget or durability—not category, material, pack size, and features together.
- Reprioritize when the evidence makes an existing factor matter more or less; do not merely append an attribute.
- Do not mention ratings, reviews, stars, or customer scores.
- Preserve a coherent trajectory above every diversity objective.
- Treat distribution_guidance only as a weak tie-breaker between equally natural, evidence-grounded decisions.

Required JSON schema:
{
  "intention_changed": true,
  "condition": "user_preference | real_world_feasibility",
  "decision_point": {
    "stable_purchase_goal": "underlying task that remains unchanged",
    "evidence_role": "trigger_only",
    "dimension": "one decision-relevant comparison dimension",
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
    "stable_purchase_goal": "a height-adjustable workstation for the existing office",
    "evidence_role": "trigger_only",
    "dimension": "complete desk versus desktop converter",
    "options_compared": [
      {"asin": "CONVERTER1", "option": "desktop converter", "tradeoff": "reuses the existing desk and costs less"},
      {"asin": "FULLDESK01", "option": "complete electric desk", "tradeoff": "adds workspace but replaces the current desk and costs more"}
    ],
    "chosen_option": "accept a converter to reuse the existing desk",
    "chosen_asin": "CONVERTER1",
    "purchase_reason": "keeping the current desk matters more than buying a complete frame"
  },
  "changes": [
    {"category": "relax", "field": "category", "old_value": "standing desk", "value": "standing desk or desktop converter", "priority_update": null, "rationale": "a converter serves the same height-adjustable work goal"}
  ],
  "rationale": "The product comparison supports one broader category trade-off; it does not make every converter feature a new requirement.",
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

        if available_count >= 2:
            tradeoff_validation = self._validate_tradeoff_scope(
                shift,
                decision_point=decision_point,
                intention_history=intention_history,
            )
            if not tradeoff_validation["valid"]:
                return ShiftOp(
                    op="none",
                    intention_changed=False,
                    condition="none",
                    change_category="none",
                    rationale=f"invalid_tradeoff_scope: {tradeoff_validation['reason']}",
                    sampling_metadata={
                        "decision_point": decision_point,
                        "evidence_asins": validation["evidence_asins"],
                        "chosen_asin": validation["chosen_asin"],
                        "decision_validation": validation,
                        "tradeoff_validation": tradeoff_validation,
                    },
                )
        else:
            tradeoff_validation = {
                "valid": True,
                "reason": "no_product_evidence_bypass",
                "substantive_change_count": self._shift_change_count(shift),
                "priority_change_count": 0,
                "primary_field": shift.field,
            }

        trajectory_validation = self._validate_trajectory_candidate(
            shift,
            current_intention=current_intention,
            intention_history=intention_history,
        )
        if not trajectory_validation["valid"]:
            return ShiftOp(
                op="none",
                intention_changed=False,
                condition="none",
                change_category="none",
                rationale=f"invalid_trajectory: {trajectory_validation['reason']}",
                sampling_metadata={
                    "decision_point": decision_point,
                    "evidence_asins": validation["evidence_asins"],
                    "chosen_asin": validation["chosen_asin"],
                    "decision_validation": validation,
                    "trajectory_validation": trajectory_validation,
                },
            )

        shift.sampling_metadata.update(
            {
                "decision_point": decision_point,
                "evidence_asins": validation["evidence_asins"],
                "chosen_asin": validation["chosen_asin"],
                "decision_validation": validation,
                "tradeoff_validation": tradeoff_validation,
                "trajectory_validation": trajectory_validation,
                "decision_quality_score": self._decision_quality_score(decision_point),
            }
        )
        return shift

    def _validate_tradeoff_scope(
        self,
        shift: ShiftOp,
        *,
        decision_point: Optional[Dict[str, Any]],
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        effective_changes = shift.changes or [shift]
        substantive_changes = [
            change
            for change in effective_changes
            if change.field
            and change.op != "reprioritize"
            and change.priority_update is None
        ]
        priority_changes = [
            change
            for change in effective_changes
            if change.op == "reprioritize" or change.priority_update is not None
        ]
        validation: Dict[str, Any] = {
            "valid": True,
            "reason": "ok",
            "substantive_change_count": len(substantive_changes),
            "priority_change_count": len(priority_changes),
            "primary_field": (
                str(substantive_changes[0].field) if substantive_changes else None
            ),
        }
        if len(substantive_changes) > 1:
            return {
                **validation,
                "valid": False,
                "reason": "multiple_substantive_changes_from_one_product_tradeoff",
            }
        if not substantive_changes and not priority_changes:
            return {**validation, "valid": False, "reason": "missing_primary_change"}

        primary_field = validation["primary_field"]
        if isinstance(intention_history, list):
            primary_change = substantive_changes[0] if substantive_changes else None
            if primary_change is not None and primary_change.op == "add":
                previous_delta = (
                    intention_history[-1].get("gold_delta")
                    if intention_history and isinstance(intention_history[-1], dict)
                    else None
                )
                if isinstance(previous_delta, dict) and any(
                    isinstance(change, dict) and change.get("op") == "add"
                    for field, change in previous_delta.items()
                    if field != "priority"
                ):
                    return {
                        **validation,
                        "valid": False,
                        "reason": "consecutive_attribute_addition",
                    }

            chosen_asin = self._clean_text(
                (decision_point or {}).get("chosen_asin")
            ).upper()
            if chosen_asin:
                recent_chosen_asins = []
                for turn in intention_history[-2:]:
                    if not isinstance(turn, dict):
                        continue
                    shift_condition = turn.get("shift_condition")
                    details = (
                        shift_condition.get("details")
                        if isinstance(shift_condition, dict)
                        else {}
                    )
                    sampling = (
                        details.get("candidate_sampling")
                        if isinstance(details, dict)
                        else {}
                    ) or {}
                    prior_asin = self._clean_text(sampling.get("chosen_asin")).upper()
                    if prior_asin:
                        recent_chosen_asins.append(prior_asin)
                if chosen_asin in recent_chosen_asins:
                    return {
                        **validation,
                        "valid": False,
                        "reason": "reused_recent_product_evidence",
                        "chosen_asin": chosen_asin,
                        "recent_chosen_asins": recent_chosen_asins,
                    }

        if primary_field in _IDENTITY_FIELDS and isinstance(intention_history, list):
            recent_identity_changes = []
            for turn in intention_history[-2:]:
                delta = turn.get("gold_delta") if isinstance(turn, dict) else None
                if not isinstance(delta, dict):
                    continue
                recent_identity_changes.extend(
                    field for field in delta if field in _IDENTITY_FIELDS
                )
            if recent_identity_changes:
                return {
                    **validation,
                    "valid": False,
                    "reason": "identity_goal_changed_too_recently",
                    "recent_identity_fields": recent_identity_changes,
                }

        if not decision_point or decision_point.get("evidence_role") != "trigger_only":
            return {
                **validation,
                "valid": False,
                "reason": "products_not_marked_as_trigger_only",
            }
        if not self._clean_text(decision_point.get("stable_purchase_goal")):
            return {
                **validation,
                "valid": False,
                "reason": "missing_stable_purchase_goal",
            }
        return validation

    def _validate_trajectory_candidate(
        self,
        shift: ShiftOp,
        *,
        current_intention: Dict[str, Any],
        intention_history: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        validation: Dict[str, Any] = {
            "valid": True,
            "reason": "ok",
            "field": None,
            "recent_turn_ids": [],
        }
        if not isinstance(intention_history, list) or not intention_history:
            return validation

        recent_history = [
            turn
            for turn in intention_history[-3:]
            if isinstance(turn, dict) and isinstance(turn.get("gold_delta"), dict)
        ]
        constraints = current_intention.get("constraints") or {}
        effective_changes = shift.changes or [shift]
        for change in effective_changes:
            field = str(change.field or "").strip()
            if (
                not field
                or change.op == "reprioritize"
                or change.priority_update is not None
            ):
                continue

            current_value = constraints.get(field)
            new_value = change.value
            if self._values_equal(current_value, new_value):
                return {
                    **validation,
                    "valid": False,
                    "reason": "no_effect_change",
                    "field": field,
                }

            field_history = [
                (turn.get("turn_id"), turn["gold_delta"][field])
                for turn in recent_history
                if isinstance(turn["gold_delta"].get(field), dict)
            ]
            if not field_history:
                continue

            recent_turn_ids = [turn_id for turn_id, _ in field_history]
            recent_old_values = [
                delta.get("old")
                for _, delta in field_history
                if delta.get("old") is not None
            ]
            if any(self._values_equal(new_value, old) for old in recent_old_values):
                return {
                    **validation,
                    "valid": False,
                    "reason": "returns_to_recent_value",
                    "field": field,
                    "recent_turn_ids": recent_turn_ids,
                }

            if new_value is None and any(
                delta.get("old") is None and delta.get("new") is not None
                for _, delta in field_history
            ):
                return {
                    **validation,
                    "valid": False,
                    "reason": "recent_add_then_remove",
                    "field": field,
                    "recent_turn_ids": recent_turn_ids,
                }

            latest_delta = field_history[-1][1]
            previous_direction = self._numeric_direction(
                latest_delta.get("old"),
                latest_delta.get("new"),
            )
            candidate_direction = self._numeric_direction(current_value, new_value)
            if (
                previous_direction
                and candidate_direction
                and previous_direction != candidate_direction
            ):
                return {
                    **validation,
                    "valid": False,
                    "reason": "immediate_numeric_direction_reversal",
                    "field": field,
                    "recent_turn_ids": recent_turn_ids,
                }

            if len(recent_history) >= 2 and all(
                isinstance(turn["gold_delta"].get(field), dict)
                for turn in recent_history[-2:]
            ):
                return {
                    **validation,
                    "valid": False,
                    "reason": "third_consecutive_field_change",
                    "field": field,
                    "recent_turn_ids": [
                        turn.get("turn_id") for turn in recent_history[-2:]
                    ],
                }

        return validation

    @classmethod
    def _values_equal(cls, left: Any, right: Any) -> bool:
        return cls._canonical_value(left) == cls._canonical_value(right)

    @classmethod
    def _canonical_value(cls, value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip().lower()
        if isinstance(value, (list, tuple)):
            return tuple(cls._canonical_value(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key), cls._canonical_value(item))
                    for key, item in value.items()
                )
            )
        return str(value).strip().lower()

    @staticmethod
    def _numeric_direction(old_value: Any, new_value: Any) -> int:
        def parse(value: Any) -> Optional[float]:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if not isinstance(value, str):
                return None
            match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", value.replace(",", ""))
            return float(match.group(0)) if match else None

        old_number = parse(old_value)
        new_number = parse(new_value)
        if old_number is None or new_number is None or old_number == new_number:
            return 0
        return 1 if new_number > old_number else -1

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

        stable_purchase_goal = self._clean_text(
            raw_decision_point.get("stable_purchase_goal")
        )
        evidence_role = self._clean_text(raw_decision_point.get("evidence_role")).lower()
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
            "stable_purchase_goal": stable_purchase_goal,
            "evidence_role": evidence_role,
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
        if not stable_purchase_goal:
            reason = "missing_stable_purchase_goal"
        elif evidence_role != "trigger_only":
            reason = "products_not_marked_as_trigger_only"
        elif not dimension or not chosen_option or not purchase_reason:
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
            and shift.rationale.startswith(
                (
                    "invalid_decision_point",
                    "invalid_tradeoff_scope",
                    "invalid_trajectory",
                )
            )
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
