from __future__ import annotations

import argparse
import copy
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.llm_executor import LLMWebShopExecutor
from evaluators.runtime_logger import RuntimeLogger
from models import AgentAction, BaseTask, DialogueInstance, EnvFeedback, TurnRecord
from simulators.human_simulator import HumanSimulator
from simulators.llm_clients import AzureOpenAIChatClient
from envs.webshop_env import WebShopEnvAdapter

STYLE_POOL = ["explicit", "partial", "elliptical"]
DEFAULT_MAX_INTERNAL_STEPS = 6
SELECTABLE_CONSTRAINT_FIELDS = ("color", "size", "brand")
PAGE_TYPE_RANK = {
    "unknown": 0,
    "search": 0,
    "results": 1,
    "item": 2,
}


@dataclass
class TurnRolloutResult:
    final_action: Optional[AgentAction]
    final_env_feedback: Optional[EnvFeedback]
    rollout_trace: List[Dict[str, Any]]
    num_internal_steps: int
    stop_reason: str


def load_local_dotenv(dotenv_path: str | None = None, override: bool = False) -> None:
    """
    Load simple KEY=VALUE pairs from a local `.env` file.

    This keeps the project dependency-free while still supporting local secret
    configuration for the simulator. Existing environment variables are
    preserved by default.
    """

    candidate_paths = []
    if dotenv_path:
        candidate_paths.append(Path(dotenv_path))
    else:
        repo_root = Path(__file__).resolve().parent.parent
        candidate_paths.extend(
            [
                Path.cwd() / ".env",
                repo_root / ".env",
            ]
        )

    seen = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)

        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]

            if override or key not in os.environ:
                os.environ[key] = value


def make_demo_webshop_task(instance_index: int = 1) -> BaseTask:
    return BaseTask(
        instance_id=f"webshop_demo_{instance_index:03d}",
        task_type="transaction",
        subtype="shopping",
        world_state={
            "domain": "webshop",
            "catalog_subset": "demo",
        },
        initial_intention={
            "request": "Find me a black office chair under 120 dollars.",
            "constraints": {
                "category": "office chair",
                "color": "black",
                "budget_max": 120,
                "brand": None,
            },
            "priority": ["category", "budget_max", "color", "brand"],
        },
    )


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _requested_selectable_constraints(current_intention: Dict[str, Any]) -> Dict[str, Any]:
    constraints = current_intention.get("constraints", {}) or {}
    requested: Dict[str, Any] = {}
    for field in SELECTABLE_CONSTRAINT_FIELDS:
        desired = constraints.get(f"{field}_exact")
        if desired is None:
            desired = constraints.get(field)
        if desired is not None:
            requested[field] = desired
    return requested


def _constraint_match_status(env_feedback: Optional[EnvFeedback], field: str) -> Optional[bool]:
    if env_feedback is None:
        return None

    observation = env_feedback.observation or {}
    constraint_debug = observation.get("constraint_debug") or {}
    field_debug = constraint_debug.get(field)
    if isinstance(field_debug, dict):
        matched = field_debug.get("matched")
        if isinstance(matched, bool):
            return matched

    if field in (env_feedback.satisfied_constraints or []):
        return True
    if field in (env_feedback.violated_constraints or []):
        return False
    return None


def _available_option_fields(env_feedback: Optional[EnvFeedback]) -> set[str]:
    if env_feedback is None:
        return set()

    observation = env_feedback.observation or {}
    item_context = observation.get("item_context") or {}
    options = item_context.get("options") or {}
    if not isinstance(options, dict):
        return set()
    return {_normalize_text(field) for field in options.keys() if _normalize_text(field)}


def _all_requested_selectable_constraints_satisfied(
    current_intention: Dict[str, Any],
    env_feedback: Optional[EnvFeedback],
) -> bool:
    requested = _requested_selectable_constraints(current_intention)
    if not requested or env_feedback is None:
        return False

    observation = env_feedback.observation or {}
    if observation.get("page_type") != "item":
        return False
    if not (observation.get("selected_asin") or env_feedback.result.get("asin")):
        return False

    for field in requested:
        if _constraint_match_status(env_feedback, field) is not True:
            return False
    return True


def _selectable_constraints_resolved_for_current_candidate(
    current_intention: Dict[str, Any],
    env_feedback: Optional[EnvFeedback],
) -> bool:
    requested = _requested_selectable_constraints(current_intention)
    if env_feedback is None:
        return False
    if not requested:
        return True

    available_fields = _available_option_fields(env_feedback)
    for field in requested:
        if _normalize_text(field) in available_fields and _constraint_match_status(env_feedback, field) is not True:
            return False
    return True


def _has_candidate_evidence(env_feedback: Optional[EnvFeedback]) -> bool:
    if env_feedback is None:
        return False
    observation = env_feedback.observation or {}
    result = env_feedback.result or {}
    selected_item = observation.get("selected_item") or {}
    return any(
        value is not None and value != ""
        for value in (
            result.get("title"),
            result.get("price"),
            result.get("category"),
            result.get("brand"),
            selected_item.get("title") if isinstance(selected_item, dict) else None,
            selected_item.get("price") if isinstance(selected_item, dict) else None,
        )
    )


def _candidate_ready(current_intention: Dict[str, Any], env_feedback: Optional[EnvFeedback]) -> bool:
    if env_feedback is None:
        return False

    observation = env_feedback.observation or {}
    if observation.get("page_type") != "item":
        return False
    if not (observation.get("selected_asin") or env_feedback.result.get("asin")):
        return False
    if not _has_candidate_evidence(env_feedback):
        return False
    return _selectable_constraints_resolved_for_current_candidate(current_intention, env_feedback)


def _page_type_rank(page_type: Any) -> int:
    return PAGE_TYPE_RANK.get(str(page_type or "").strip().lower(), 0)


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
        sorted(
            (_normalize_text(key), _normalize_text(value))
            for key, value in selected_options.items()
        )
    ) if isinstance(selected_options, dict) else ()
    return (
        str(observation.get("page_type") or ""),
        str(observation.get("selected_asin") or result.get("asin") or ""),
        normalized_options,
        visible_asins,
        _normalize_text(result.get("title")),
        result.get("price"),
        tuple(sorted(_normalize_text(field) for field in env_feedback.satisfied_constraints or [])),
    )


def _made_useful_progress(
    previous_feedback: Optional[EnvFeedback],
    current_feedback: Optional[EnvFeedback],
) -> bool:
    if current_feedback is None:
        return False
    if previous_feedback is None:
        return True

    prev_obs = previous_feedback.observation or {}
    current_obs = current_feedback.observation or {}
    prev_result = previous_feedback.result or {}
    current_result = current_feedback.result or {}

    if _page_type_rank(current_obs.get("page_type")) > _page_type_rank(prev_obs.get("page_type")):
        return True
    if current_obs.get("selected_asin") and current_obs.get("selected_asin") != prev_obs.get("selected_asin"):
        return True

    prev_selected_options = prev_obs.get("selected_options") or {}
    current_selected_options = current_obs.get("selected_options") or {}
    if isinstance(prev_selected_options, dict) and isinstance(current_selected_options, dict):
        if len(current_selected_options) > len(prev_selected_options):
            return True
        if current_selected_options != prev_selected_options:
            return True

    if len(current_feedback.satisfied_constraints or []) > len(previous_feedback.satisfied_constraints or []):
        return True

    prev_visible_asins = {
        str(item.get("asin", "")).strip().upper()
        for item in prev_obs.get("visible_items", []) or []
        if isinstance(item, dict) and item.get("asin")
    }
    current_visible_asins = {
        str(item.get("asin", "")).strip().upper()
        for item in current_obs.get("visible_items", []) or []
        if isinstance(item, dict) and item.get("asin")
    }
    if current_visible_asins and current_visible_asins != prev_visible_asins:
        return True

    if _normalize_text(current_result.get("title")) and _normalize_text(current_result.get("title")) != _normalize_text(prev_result.get("title")):
        return True

    return False


def _compact_constraint_snapshot(env_feedback: Optional[EnvFeedback]) -> Dict[str, Any]:
    if env_feedback is None:
        return {}

    snapshot: Dict[str, Any] = {}
    constraint_debug = (env_feedback.observation or {}).get("constraint_debug") or {}
    for field, details in constraint_debug.items():
        if not isinstance(details, dict):
            continue
        snapshot[field] = {
            "desired": details.get("desired"),
            "actual": details.get("actual"),
            "matched": details.get("matched"),
        }
    return snapshot


def _action_signature(agent_action: Optional[AgentAction]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    if agent_action is None:
        return ("", ())

    payload = agent_action.action_payload or {}
    normalized_payload = tuple(
        sorted((str(key), _normalize_text(value)) for key, value in payload.items())
    )
    return (str(agent_action.action_type or ""), normalized_payload)


def _maybe_summarize_current_state(
    env: WebShopEnvAdapter,
    current_intention: Dict[str, Any],
) -> Optional[EnvFeedback]:
    summarize_current_state = getattr(env, "summarize_current_state", None)
    if not callable(summarize_current_state):
        return None
    return summarize_current_state(current_intention)


def _rollout_stop_reason(
    current_intention: Dict[str, Any],
    env_feedback: Optional[EnvFeedback],
    *,
    num_internal_steps: int,
    max_internal_steps: int,
    repeated_action_streak: int = 0,
    stagnant_steps: int = 0,
    env_done: bool = False,
) -> Optional[str]:
    if env_feedback is None:
        return "no_feedback"
    if env_feedback.status == "error":
        return "error"
    if env_done:
        return "env_done"
    if _all_requested_selectable_constraints_satisfied(current_intention, env_feedback):
        return "requested_options_satisfied"
    if _candidate_ready(current_intention, env_feedback):
        return "candidate_ready"
    if repeated_action_streak >= 2 or stagnant_steps >= 2:
        return "stuck"
    if num_internal_steps >= max_internal_steps:
        return "step_budget"
    return None


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
        },
        "page_type": observation.get("page_type"),
        "selected_asin": observation.get("selected_asin"),
        "selected_options": copy.deepcopy(observation.get("selected_options") or {}),
        "state_changed": state_changed,
        "made_progress": made_progress,
        "satisfied_constraints": list(env_feedback.satisfied_constraints or []),
        "violated_constraints": list(env_feedback.violated_constraints or []),
        "constraint_snapshot": _compact_constraint_snapshot(env_feedback),
        "stop_reason": stop_reason,
    }


def execute_turn(
    env: WebShopEnvAdapter,
    execution_agent,
    history: List[Dict[str, Any]],
    current_intention: Dict[str, Any],
    env_observation: Dict[str, Any],
    max_internal_steps: int = DEFAULT_MAX_INTERNAL_STEPS,
) -> TurnRolloutResult:
    working_history = copy.deepcopy(history)
    rollout_trace: List[Dict[str, Any]] = []
    previous_feedback = _maybe_summarize_current_state(env, current_intention)
    initial_stop_reason = _rollout_stop_reason(
        current_intention,
        previous_feedback,
        num_internal_steps=0,
        max_internal_steps=max_internal_steps,
        env_done=getattr(env, "done", False),
    )
    if initial_stop_reason in {"requested_options_satisfied", "candidate_ready", "env_done"}:
        return TurnRolloutResult(
            final_action=None,
            final_env_feedback=previous_feedback,
            rollout_trace=rollout_trace,
            num_internal_steps=0,
            stop_reason=initial_stop_reason,
        )

    current_observation = copy.deepcopy(env_observation)
    final_feedback = previous_feedback
    final_action: Optional[AgentAction] = None
    previous_action_signature: Optional[Tuple[str, Tuple[Tuple[str, str], ...]]] = None
    repeated_action_streak = 0
    stagnant_steps = 0

    for step_index in range(1, max_internal_steps + 1):
        agent_action = execution_agent.act(working_history, current_intention, current_observation)
        env_feedback = env.step(agent_action, current_intention)
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

        stop_reason = _rollout_stop_reason(
            current_intention,
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
            return TurnRolloutResult(
                final_action=final_action,
                final_env_feedback=final_feedback,
                rollout_trace=rollout_trace,
                num_internal_steps=step_index,
                stop_reason=stop_reason,
            )

    return TurnRolloutResult(
        final_action=final_action,
        final_env_feedback=final_feedback,
        rollout_trace=rollout_trace,
        num_internal_steps=len(rollout_trace),
        stop_reason="step_budget",
    )


def simulate_dialogue_instance(
    task: BaseTask,
    env: WebShopEnvAdapter,
    execution_agent,
    human_simulator: HumanSimulator,
    max_turns: int = 4,
    max_internal_steps: int = DEFAULT_MAX_INTERNAL_STEPS,
    seed: int = 7,
) -> DialogueInstance:
    rng = random.Random(seed)
    turns: List[TurnRecord] = []

    current_intention = copy.deepcopy(task.initial_intention)
    env_obs = env.reset(task)
    real_instruction = env.get_instruction_text()
    parsed_intention = env.parse_instruction_to_intention(real_instruction)
    if parsed_intention is not None:
        current_intention = parsed_intention
    elif real_instruction and real_instruction.strip():
        current_intention["request"] = real_instruction.strip()

    initial_request = current_intention.get("request", "")

    history: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": initial_request,
        }
    ]

    turns.append(
        TurnRecord(
            turn_id=0,
            user_utterance=initial_request,
            agent_action=None,
            env_feedback=None,
            trigger_evidence=None,
            shift_condition=None,
            gold_delta={},
            gold_current_intention=copy.deepcopy(current_intention),
            linguistic_style="explicit",
            action_implication="start_search",
        )
    )

    for turn_id in range(1, max_turns + 1):
        rollout = execute_turn(
            env=env,
            execution_agent=execution_agent,
            history=history,
            current_intention=current_intention,
            env_observation=env_obs,
            max_internal_steps=max_internal_steps,
        )
        agent_action = rollout.final_action
        env_feedback = rollout.final_env_feedback
        style = rng.choice(STYLE_POOL)
        shift = human_simulator.decide_shift(
            current_intention,
            agent_action=agent_action,
            env_feedback=env_feedback,
            history=history,
        )
        new_intention, delta = human_simulator.apply_shift(current_intention, shift)
        user_utt = human_simulator.realize_shift(
            shift,
            current_intention,
            style,
            agent_action=agent_action,
            env_feedback=env_feedback,
            history=history,
        )
        shift_condition = None
        trigger_evidence = {
            "trigger_type": "none",
            "source": "simulator",
            "details": {},
        }
        if shift.op != "none":
            shift_condition = {
                "type": "llm_inferred_shift",
                "reason": shift.rationale,
                "source": "simulator",
                "details": {
                    "op": shift.op,
                    "field": shift.field,
                    "old_value": shift.old_value,
                    "value": shift.value,
                    "priority_update": shift.priority_update,
                },
            }
            trigger_evidence = {
                "trigger_type": "llm_inferred_shift",
                "source": "simulator",
                "details": {
                    "op": shift.op,
                    "field": shift.field,
                    "rationale": shift.rationale,
                },
            }
        current_intention = new_intention
        action_implication = "requery" if shift.op in {"relax", "override"} else "continue"

        turns.append(
            TurnRecord(
                turn_id=turn_id,
                user_utterance=user_utt,
                agent_action=(
                    {
                        "action_type": agent_action.action_type,
                        "action_payload": agent_action.action_payload,
                    }
                    if agent_action is not None
                    else None
                ),
                env_feedback=(
                    {
                        "status": env_feedback.status,
                        "feasible": env_feedback.feasible,
                        "reason": env_feedback.reason,
                        "observation": env_feedback.observation,
                        "result": env_feedback.result,
                        "satisfied_constraints": env_feedback.satisfied_constraints,
                        "violated_constraints": env_feedback.violated_constraints,
                    }
                    if env_feedback is not None
                    else None
                ),
                trigger_evidence=trigger_evidence,
                shift_condition=shift_condition,
                gold_delta=delta,
                gold_current_intention=copy.deepcopy(current_intention),
                linguistic_style=style,
                action_implication=action_implication,
                num_internal_steps=rollout.num_internal_steps,
                stop_reason=rollout.stop_reason,
                rollout_trace=rollout.rollout_trace,
            )
        )

        history.append(
            {
                "role": "assistant",
                "content": {
                    "action_type": agent_action.action_type if agent_action is not None else None,
                    "action_payload": agent_action.action_payload if agent_action is not None else {},
                    "env_result": env_feedback.result if env_feedback is not None else {},
                    "num_internal_steps": rollout.num_internal_steps,
                    "stop_reason": rollout.stop_reason,
                    "rollout_trace": rollout.rollout_trace,
                },
            }
        )
        history.append({"role": "user", "content": user_utt})
        env_obs = env.get_observation()

        if env.done and not delta:
            break

    return DialogueInstance(
        instance_id=task.instance_id,
        task_type=task.task_type,
        subtype=task.subtype,
        world_state=task.world_state,
        turns=turns,
    )


def main():
    load_local_dotenv()
    import gym
    from web_agent_site.envs import WebAgentTextEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=".\IntentionChangeBench\data\webshop_simulated_dataset.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_internal_steps", type=int, default=DEFAULT_MAX_INTERNAL_STEPS)
    parser.add_argument("--num_instances", type=int, default=10)
    parser.add_argument(
        "--azure_api_version",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    args = parser.parse_args()

    if args.num_instances < 1:
        raise ValueError("--num_instances must be at least 1")
    # Gym v0.24 env_checker can behave poorly with older envs; disable it.
    raw_env = gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text",
        num_products=1000,   # 先用 small
        disable_env_checker=True,
    )
    if raw_env is None:
        # Defensive fallback for edge cases where gym.make returns None.
        raw_env = WebAgentTextEnv(observation_mode="text", num_products=1000)
    env = WebShopEnvAdapter(webshop_env=raw_env, action_style="auto")
    llm_client = AzureOpenAIChatClient.from_env(api_version=args.azure_api_version)
    agent = LLMWebShopExecutor(llm_client=llm_client)
    human = HumanSimulator(llm_client=llm_client)
    logger = RuntimeLogger()

    for instance_index in range(1, args.num_instances + 1):
        task = make_demo_webshop_task(instance_index=instance_index)
        instance = simulate_dialogue_instance(
            task=task,
            env=env,
            execution_agent=agent,
            human_simulator=human,
            max_turns=args.max_turns,
            max_internal_steps=args.max_internal_steps,
            seed=args.seed + instance_index - 1,
        )
        logger.log_instance(instance)

    logger.dump_json(args.output)
    print(f"Saved {len(logger.instances)} instances to {args.output}")


if __name__ == "__main__":
    main()
