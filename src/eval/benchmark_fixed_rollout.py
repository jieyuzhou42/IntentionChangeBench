from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import AgentAction, EnvFeedback

DEFAULT_MAX_INTERNAL_STEPS = 12


@dataclass
class FixedTurnRolloutResult:
    final_action: Optional[AgentAction]
    final_env_feedback: Optional[EnvFeedback]
    rollout_trace: List[Dict[str, Any]]
    num_internal_steps: int
    stop_reason: str
    num_search_actions: int = 0
    search_queries: List[str] = field(default_factory=list)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _action_signature(agent_action: Optional[AgentAction]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    if agent_action is None:
        return ("", ())
    payload = agent_action.action_payload or {}
    normalized_payload = tuple(
        sorted((str(key), _normalize_text(value)) for key, value in payload.items())
    )
    return (str(agent_action.action_type or ""), normalized_payload)


def _feedback_state_signature(env_feedback: Optional[EnvFeedback]) -> Optional[Tuple[Any, ...]]:
    if env_feedback is None:
        return None
    observation = env_feedback.observation or {}
    result = env_feedback.result or {}
    selected_options = observation.get("selected_options") or {}
    visible_items = observation.get("visible_items") or []
    visible_asins = tuple(
        str(item.get("asin", "")).strip().upper()
        for item in visible_items[:5]
        if isinstance(item, dict) and item.get("asin")
    )
    normalized_options = tuple(
        sorted((str(key), _normalize_text(value)) for key, value in selected_options.items())
    )
    return (
        env_feedback.status,
        observation.get("page_type"),
        observation.get("selected_asin") or result.get("asin"),
        normalized_options,
        visible_asins,
    )


def _made_useful_progress(
    previous_feedback: Optional[EnvFeedback],
    current_feedback: Optional[EnvFeedback],
) -> bool:
    if current_feedback is None:
        return False
    if previous_feedback is None:
        return current_feedback.status != "error"

    previous_obs = previous_feedback.observation or {}
    current_obs = current_feedback.observation or {}
    if _feedback_state_signature(previous_feedback) != _feedback_state_signature(current_feedback):
        return True
    if current_obs.get("selected_asin") and current_obs.get("selected_asin") != previous_obs.get("selected_asin"):
        return True
    return False


def _history_returned_items(
    env_feedback: Optional[EnvFeedback],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    if env_feedback is None:
        return []
    observation = env_feedback.observation or {}
    items = observation.get("candidate_items") or observation.get("visible_items") or []
    if not isinstance(items, list):
        return []

    returned_items: List[Dict[str, Any]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        returned_items.append(
            {
                "asin": item.get("asin"),
                "rank": item.get("rank"),
                "title": item.get("title"),
                "price": item.get("price"),
            }
        )
    return returned_items


def _maybe_summarize_current_state(env, agent_intention: Dict[str, Any]) -> Optional[EnvFeedback]:
    summarize_current_state = getattr(env, "summarize_current_state", None)
    if not callable(summarize_current_state):
        return None
    return summarize_current_state(agent_intention)


def _build_rollout_trace_entry(
    step_index: int,
    agent_action: AgentAction,
    env_feedback: EnvFeedback,
    *,
    state_changed: bool,
    made_progress: bool,
    stop_reason: Optional[str],
) -> Dict[str, Any]:
    observation = env_feedback.observation or {}
    return {
        "step_index": step_index,
        "action": {
            "action_type": agent_action.action_type,
            "action_payload": dict(agent_action.action_payload or {}),
            "rationale": getattr(agent_action, "rationale", None),
            "predicted_current_intention": copy.deepcopy(
                getattr(agent_action, "predicted_current_intention", None)
            ),
        },
        "page_type": observation.get("page_type"),
        "selected_asin": observation.get("selected_asin"),
        "selected_options": copy.deepcopy(observation.get("selected_options") or {}),
        "state_changed": state_changed,
        "made_progress": made_progress,
        "stop_reason": stop_reason,
    }


def _fixed_stop_reason(
    env_feedback: Optional[EnvFeedback],
    *,
    num_internal_steps: int,
    max_internal_steps: int,
    repeated_action_streak: int,
    stagnant_steps: int,
    env_done: bool,
) -> Optional[str]:
    if env_feedback is None:
        return "no_feedback"
    if env_feedback.status == "error":
        return "error"
    if env_done:
        return "env_done"
    if repeated_action_streak >= 2 or stagnant_steps >= 2:
        return "stuck"
    if num_internal_steps >= max_internal_steps:
        return "step_budget"
    return None


def execute_fixed_user_turn(
    env,
    execution_agent,
    history: List[Dict[str, Any]],
    user_utterance: str,
    env_observation: Dict[str, Any],
    max_internal_steps: int = DEFAULT_MAX_INTERNAL_STEPS,
) -> FixedTurnRolloutResult:
    """
    Benchmark-only rollout for the fixed-user WebShop executor.

    This function intentionally never receives gold_current_intention. The
    environment sees only the agent's own predicted_current_intention, which is
    emitted together with each action. Gold state is reserved for offline eval.
    """

    working_history = copy.deepcopy(history)
    rollout_trace: List[Dict[str, Any]] = []
    current_observation = copy.deepcopy(env_observation)
    final_feedback: Optional[EnvFeedback] = None
    final_action: Optional[AgentAction] = None
    previous_action_signature: Optional[Tuple[str, Tuple[Tuple[str, str], ...]]] = None
    repeated_action_streak = 0
    stagnant_steps = 0
    search_queries: List[str] = []

    previous_feedback = _maybe_summarize_current_state(env, {})

    for step_index in range(1, max_internal_steps + 1):
        agent_action = execution_agent.act(working_history, user_utterance, current_observation)
        if agent_action.action_type == "search":
            search_queries.append(str((agent_action.action_payload or {}).get("query", "")))

        predicted_intention = getattr(agent_action, "predicted_current_intention", None)
        env_user_state = predicted_intention if isinstance(predicted_intention, dict) else {}
        env_feedback = env.step(agent_action, env_user_state)

        action_signature = _action_signature(agent_action)
        if action_signature == previous_action_signature:
            repeated_action_streak += 1
        else:
            repeated_action_streak = 1

        state_changed = _feedback_state_signature(previous_feedback) != _feedback_state_signature(env_feedback)
        made_progress = _made_useful_progress(previous_feedback, env_feedback)
        if made_progress:
            stagnant_steps = 0
        else:
            stagnant_steps += 1

        if agent_action.action_type == "buy":
            stop_reason = "virtual_buy"
        else:
            stop_reason = _fixed_stop_reason(
                env_feedback,
                num_internal_steps=step_index,
                max_internal_steps=max_internal_steps,
                repeated_action_streak=repeated_action_streak,
                stagnant_steps=stagnant_steps,
                env_done=getattr(env, "done", False),
            )

        rollout_trace.append(
            _build_rollout_trace_entry(
                step_index,
                agent_action,
                env_feedback,
                state_changed=state_changed,
                made_progress=made_progress,
                stop_reason=stop_reason,
            )
        )

        working_history.append(
            {
                "role": "assistant",
                "content": {
                    "action_type": agent_action.action_type,
                    "action_payload": dict(agent_action.action_payload or {}),
                    "env_result": copy.deepcopy(env_feedback.result or {}),
                    "page_type": (env_feedback.observation or {}).get("page_type"),
                    "selected_asin": (env_feedback.observation or {}).get("selected_asin"),
                    "selected_options": copy.deepcopy((env_feedback.observation or {}).get("selected_options") or {}),
                    "returned_items": _history_returned_items(env_feedback),
                    "internal_step": step_index,
                },
            }
        )

        final_action = agent_action
        final_feedback = env_feedback
        current_observation = env.get_observation()
        previous_feedback = env_feedback
        previous_action_signature = action_signature

        if stop_reason is not None:
            return FixedTurnRolloutResult(
                final_action=final_action,
                final_env_feedback=final_feedback,
                rollout_trace=rollout_trace,
                num_internal_steps=step_index,
                stop_reason=stop_reason,
                num_search_actions=len(search_queries),
                search_queries=list(search_queries),
            )

    return FixedTurnRolloutResult(
        final_action=final_action,
        final_env_feedback=final_feedback,
        rollout_trace=rollout_trace,
        num_internal_steps=len(rollout_trace),
        stop_reason="step_budget",
        num_search_actions=len(search_queries),
        search_queries=list(search_queries),
    )
