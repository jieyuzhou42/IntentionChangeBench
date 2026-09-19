from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Sequence

from domains.travelplanner.environment import TravelPlannerEnvAdapter
from domains.travelplanner.executor import TravelPlannerExecutor
from eval.human_annotated_pilot import call_json_with_retries
from eval.intent_schema import INTENT_RULES, INTENT_SCHEMA_JSON, normalize_intent_prediction, environment_intention
from eval.action_policy import TRAVEL_SELECTION_RULES
from models import BaseTask
from simulation.simulation.run_simulation import (
    _public_env_feedback_payload,
    execute_turn,
)


class EvaluationTravelPlannerExecutor(TravelPlannerExecutor):
    """Apply evaluation's concrete-selection policy to the separate planner."""

    def _build_plan_prompt(self, history, user_utterance, observation):
        prompt = super()._build_plan_prompt(history, user_utterance, observation)
        return prompt + "\n\nEvaluation final-selection policy (takes precedence over exact-match guidance):\n" + TRAVEL_SELECTION_RULES


def build_blind_intention_prompt(
    *,
    domain: str,
    user_utterances: Sequence[str],
) -> str:
    return f"""
Infer the user's current cumulative intention from the user utterances only.
The utterances are ordered oldest to newest. A later utterance may add, relax,
replace, remove, or reprioritize an earlier requirement.

Return exactly one JSON object:
{INTENT_SCHEMA_JSON}

Rules:
- Include every active requirement and omit superseded requirements.
- Do not invent requirements.
{INTENT_RULES}
- Use normalized concise field names; semantic equivalence matters more than wording.

DOMAIN: {domain}
USER_UTTERANCES:
{json.dumps(list(user_utterances), ensure_ascii=False, indent=2)}
""".strip()


def normalize_intention_prediction(raw: Dict[str, Any], *, domain: str) -> Dict[str, Any]:
    return normalize_intent_prediction(raw)


def infer_blind_intention(
    client: Any,
    *,
    domain: str,
    user_utterances: Sequence[str],
) -> Dict[str, Any]:
    raw = call_json_with_retries(
        client,
        build_blind_intention_prompt(
            domain=domain,
            user_utterances=user_utterances,
        ),
    )
    return normalize_intention_prediction(raw, domain=domain)


def travel_task_from_instance(instance: Dict[str, Any]) -> BaseTask:
    world_state = copy.deepcopy(instance.get("world_state") or {})
    return BaseTask(
        instance_id=str(instance.get("instance_id")),
        task_type=str(instance.get("task_type") or "planning"),
        subtype=str(instance.get("subtype") or "travel"),
        world_state=world_state,
        # The environment receives the original public query through world_state.
        # Current intention is inferred blindly for each turn before tool execution.
        initial_intention={"constraints": {}, "priority": [], "domain": "travelplanner"},
    )


def replay_travelplanner_instance(
    *,
    instance: Dict[str, Any],
    client: Any,
    max_internal_steps: int = 30,
) -> List[Dict[str, Any]]:
    task = travel_task_from_instance(instance)
    env = TravelPlannerEnvAdapter()
    executor = EvaluationTravelPlannerExecutor(llm_client=client)
    env_observation = env.reset(task)
    history: List[Dict[str, Any]] = []
    utterances: List[str] = []
    rows: List[Dict[str, Any]] = []

    for turn_index, source_turn in enumerate(instance.get("turns") or []):
        user_utterance = str(source_turn.get("user_utterance") or "").strip()
        utterances.append(user_utterance)
        history.append({"role": "user", "content": user_utterance})
        predicted_intention = infer_blind_intention(
            client,
            domain="travelplanner",
            user_utterances=utterances,
        )
        rollout = execute_turn(
            env=env,
            execution_agent=executor,
            history=history,
            user_utterance=user_utterance,
            current_intention=environment_intention(predicted_intention),
            env_observation=env_observation,
            gold_delta={},
            max_internal_steps=max_internal_steps,
            stop_on_candidate_ready=False,
        )
        final_action = rollout.final_action
        final_feedback = rollout.final_env_feedback
        action_dict = (
            {
                "action_type": final_action.action_type,
                "action_payload": copy.deepcopy(final_action.action_payload or {}),
                "rationale": final_action.rationale,
            }
            if final_action is not None
            else None
        )
        gold_intention = copy.deepcopy(source_turn.get("gold_current_intention") or {})
        submitted_plan = (
            (final_feedback.observation or {}).get("submitted_plan")
            if final_feedback is not None
            else None
        )
        gold_plan_eval = (
            env._evaluate_plan(submitted_plan, gold_intention)
            if isinstance(submitted_plan, dict)
            else {}
        )
        rows.append(
            {
                "domain": "travelplanner",
                "instance_id": task.instance_id,
                "turn_id": int(source_turn.get("turn_id", turn_index)),
                "user_utterance": user_utterance,
                "gold_intention": gold_intention,
                "agent_intention_prediction": predicted_intention,
                "agent_action": action_dict,
                "env_feedback": _public_env_feedback_payload(final_feedback),
                "rollout_trace": copy.deepcopy(rollout.rollout_trace),
                "num_internal_steps": rollout.num_internal_steps,
                "stop_reason": rollout.stop_reason,
                "action_evidence": {
                    "final_action": action_dict,
                    "submitted_plan": copy.deepcopy(submitted_plan),
                    "gold_deterministic_plan_eval": gold_plan_eval,
                    "rollout_trace": copy.deepcopy(rollout.rollout_trace),
                },
            }
        )
        env_observation = env.get_observation()
    return rows


def webshop_action_evidence(turn: Dict[str, Any]) -> Dict[str, Any]:
    feedback = turn.get("env_feedback") or {}
    selected = feedback.get("selected_item") or feedback.get("selected_candidate")
    return {
        "final_action": copy.deepcopy(turn.get("agent_action")),
        "selected_product": copy.deepcopy(selected),
        "selected_asin": feedback.get("selected_asin"),
        "gold_deterministic_constraint_eval": {
            "satisfied_constraints": feedback.get("gold_eval_satisfied_constraints") or [],
            "violated_constraints": feedback.get("gold_eval_violated_constraints") or [],
            "constraint_debug": feedback.get("gold_eval_constraint_debug") or {},
        },
        "rollout_trace": copy.deepcopy(turn.get("rollout_trace") or []),
    }
