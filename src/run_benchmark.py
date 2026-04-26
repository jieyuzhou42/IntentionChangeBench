from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluators.runtime_logger import RuntimeLogger
from models import BaseTask, DialogueInstance, TurnRecord
from prompt_logging import get_prompt_log_path
from run_simulation import (
    DEFAULT_MAX_INTERNAL_STEPS,
    _build_runtime_components,
    _history_returned_items,
    _partition_indexed_tasks,
    _public_env_feedback_payload,
    _task_from_payload,
    execute_turn,
    load_local_dotenv,
    parse_instance_ids,
    parse_webshop_num_products,
)


@dataclass
class BenchmarkTask:
    task: BaseTask
    source_instance: Dict[str, Any]


def _load_raw_instances(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Gold trajectory file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("instances"), list):
            instances = payload["instances"]
        elif isinstance(payload.get("tasks"), list):
            instances = payload["tasks"]
        else:
            instances = [payload]
    elif isinstance(payload, list):
        instances = payload
    else:
        raise ValueError(f"Gold trajectory file {path} must contain a JSON object or array")

    normalized_instances: List[Dict[str, Any]] = []
    for index, raw_instance in enumerate(instances, start=1):
        if not isinstance(raw_instance, dict):
            raise ValueError(f"Instance #{index} in {path} must be a JSON object")
        normalized_instances.append(raw_instance)
    return normalized_instances


def load_benchmark_tasks(
    *,
    gold_trajectory_path: str,
    num_instances: Optional[int],
    instance_ids: Optional[List[str]] = None,
) -> List[BenchmarkTask]:
    raw_instances = _load_raw_instances(Path(gold_trajectory_path))

    if instance_ids:
        instance_by_id = {
            str(instance.get("instance_id") or f"webshop_task_{index:03d}"): instance
            for index, instance in enumerate(raw_instances, start=1)
        }
        missing = [instance_id for instance_id in instance_ids if instance_id not in instance_by_id]
        if missing:
            raise ValueError(f"Could not find requested instance_id(s): {', '.join(missing)}")
        selected_instances = [instance_by_id[instance_id] for instance_id in instance_ids]
    else:
        selected_instances = list(raw_instances)

    if num_instances is not None:
        if num_instances < 1:
            raise ValueError("--num_instances must be at least 1")
        if num_instances > len(selected_instances) and not instance_ids:
            raise ValueError(
                f"Requested {num_instances} instances, but {gold_trajectory_path} only contains "
                f"{len(selected_instances)}"
            )
        selected_instances = selected_instances[:num_instances]

    benchmark_tasks: List[BenchmarkTask] = []
    for index, raw_instance in enumerate(selected_instances, start=1):
        benchmark_tasks.append(
            BenchmarkTask(
                task=_task_from_payload(raw_instance, fallback_index=index),
                source_instance=raw_instance,
            )
        )
    return benchmark_tasks


def _resolve_turn_intention(
    source_turn: Dict[str, Any],
    fallback_intention: Dict[str, Any],
) -> Dict[str, Any]:
    raw_intention = source_turn.get("gold_current_intention")
    if isinstance(raw_intention, dict):
        return copy.deepcopy(raw_intention)
    return copy.deepcopy(fallback_intention)


def _resolve_turn_user_utterance(
    source_turn: Dict[str, Any],
    current_intention: Dict[str, Any],
) -> str:
    utterance = source_turn.get("user_utterance")
    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()

    request = current_intention.get("request")
    if isinstance(request, str) and request.strip():
        return request.strip()
    return ""


def replay_dialogue_instance(
    *,
    benchmark_task: BenchmarkTask,
    env,
    execution_agent,
    max_turns: Optional[int],
    max_internal_steps: int,
) -> DialogueInstance:
    task = benchmark_task.task
    source_instance = benchmark_task.source_instance
    source_turns = source_instance.get("turns") or []
    if not isinstance(source_turns, list):
        raise ValueError(f"Instance {task.instance_id} has invalid turns payload")

    replay_turns = source_turns
    if max_turns is not None:
        replay_turns = source_turns[: max_turns + 1]

    turns: List[TurnRecord] = []
    env_obs = env.reset(task)
    current_intention = copy.deepcopy(task.initial_intention)

    for turn_index, raw_turn in enumerate(replay_turns):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"Turn #{turn_index} in {task.instance_id} must be a JSON object")

        current_intention = _resolve_turn_intention(raw_turn, current_intention)
        user_utterance = _resolve_turn_user_utterance(raw_turn, current_intention)
        history: List[Dict[str, Any]] = [{"role": "user", "content": user_utterance}]

        rollout = execute_turn(
            env=env,
            execution_agent=execution_agent,
            history=history,
            user_utterance=user_utterance,
            current_intention=current_intention,
            env_observation=env_obs,
            max_internal_steps=max_internal_steps,
        )
        agent_action = rollout.final_action
        env_feedback = rollout.final_env_feedback

        turns.append(
            TurnRecord(
                turn_id=int(raw_turn.get("turn_id", turn_index)),
                user_utterance=user_utterance,
                agent_action=(
                    {
                        "action_type": agent_action.action_type,
                        "action_payload": agent_action.action_payload,
                    }
                    if agent_action is not None
                    else None
                ),
                env_feedback=_public_env_feedback_payload(env_feedback),
                trigger_evidence=copy.deepcopy(raw_turn.get("trigger_evidence")),
                shift_condition=copy.deepcopy(raw_turn.get("shift_condition")),
                gold_delta=copy.deepcopy(raw_turn.get("gold_delta") or {}),
                gold_current_intention=copy.deepcopy(current_intention),
                linguistic_style=str(raw_turn.get("linguistic_style") or "explicit"),
                action_implication=str(
                    raw_turn.get("action_implication") or ("start_search" if turn_index == 0 else "continue")
                ),
                num_internal_steps=rollout.num_internal_steps,
                num_rollout_search_actions=rollout.num_search_actions,
                rollout_search_queries=list(rollout.search_queries),
                stop_reason=rollout.stop_reason,
                rollout_trace=rollout.rollout_trace,
            )
        )

        env_obs = env.get_observation()

        if env.done:
            break

    return DialogueInstance(
        instance_id=task.instance_id,
        task_type=task.task_type,
        subtype=task.subtype,
        world_state=copy.deepcopy(source_instance.get("world_state") or task.world_state),
        turns=turns,
    )


def _replay_instances_serial(
    *,
    benchmark_tasks: List[BenchmarkTask],
    max_turns: Optional[int],
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
) -> List[DialogueInstance]:
    env, agent, _human, raw_env = _build_runtime_components(
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
    )
    try:
        instances = []
        for benchmark_task in benchmark_tasks:
            instances.append(
                replay_dialogue_instance(
                    benchmark_task=benchmark_task,
                    env=env,
                    execution_agent=agent,
                    max_turns=max_turns,
                    max_internal_steps=max_internal_steps,
                )
            )
        return instances
    finally:
        close_env = getattr(raw_env, "close", None)
        if callable(close_env):
            close_env()


def _replay_task_batch(
    *,
    indexed_tasks: List[Tuple[int, BenchmarkTask]],
    max_turns: Optional[int],
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
) -> Dict[int, DialogueInstance]:
    env, agent, _human, raw_env = _build_runtime_components(
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
    )
    try:
        instances_by_index: Dict[int, DialogueInstance] = {}
        for task_index, benchmark_task in indexed_tasks:
            instances_by_index[task_index] = replay_dialogue_instance(
                benchmark_task=benchmark_task,
                env=env,
                execution_agent=agent,
                max_turns=max_turns,
                max_internal_steps=max_internal_steps,
            )
        return instances_by_index
    finally:
        close_env = getattr(raw_env, "close", None)
        if callable(close_env):
            close_env()


def _replay_instances(
    *,
    benchmark_tasks: List[BenchmarkTask],
    max_turns: Optional[int],
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
    parallelism: int,
) -> List[DialogueInstance]:
    if parallelism <= 1:
        return _replay_instances_serial(
            benchmark_tasks=benchmark_tasks,
            max_turns=max_turns,
            max_internal_steps=max_internal_steps,
            azure_api_version=azure_api_version,
            webshop_num_products=webshop_num_products,
            executor_type=executor_type,
        )

    effective_parallelism = min(parallelism, len(benchmark_tasks))
    task_batches = _partition_indexed_tasks(benchmark_tasks, effective_parallelism)
    instances_by_index: Dict[int, DialogueInstance] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_batch_index = {
            executor.submit(
                _replay_task_batch,
                indexed_tasks=batch,
                max_turns=max_turns,
                max_internal_steps=max_internal_steps,
                azure_api_version=azure_api_version,
                webshop_num_products=webshop_num_products,
                executor_type=executor_type,
            ): batch_index
            for batch_index, batch in enumerate(task_batches, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_batch_index):
            batch_index = future_to_batch_index[future]
            try:
                instances_by_index.update(future.result())
            except Exception as exc:
                raise RuntimeError(f"Failed while replaying benchmark batch #{batch_index}") from exc

    return [instances_by_index[task_index] for task_index in sorted(instances_by_index)]


def main() -> None:
    load_local_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=r".\IntentionChangeBench\data\webshop_benchmark_output.json",
    )
    parser.add_argument(
        "--gold_trajectory_path",
        "--tasks_path",
        dest="gold_trajectory_path",
        type=str,
        default=r".\IntentionChangeBench\data\webshop_simulated_dataset_20_0.json",
        help="Gold trajectory JSON used to replay fixed user utterances.",
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated instance ids to replay, e.g. "
            "webshop_demo_001,webshop_demo_010 or shorthand web1,web10."
        ),
    )
    parser.add_argument("--num_instances", type=int, default=None)
    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Maximum turn index to replay. Omit to use every turn from the gold trajectory.",
    )
    parser.add_argument("--max_internal_steps", type=int, default=DEFAULT_MAX_INTERNAL_STEPS)
    parser.add_argument(
        "--webshop_num_products",
        type=str,
        default=os.getenv("WEBSHOP_NUM_PRODUCTS", "100000"),
        help="WebShop product subset to load: 100, 1000, 100000, or all.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of benchmark instances to replay concurrently.",
    )
    parser.add_argument(
        "--azure_api_version",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    parser.add_argument(
        "--executor_type",
        type=str,
        choices=["llm", "fixed_user"],
        default="fixed_user",
        help="Execution agent to use during replay.",
    )
    args = parser.parse_args()
    print(f"Prompt log path: {get_prompt_log_path()}")

    if args.parallelism < 1:
        raise ValueError("--parallelism must be at least 1")
    if args.max_turns is not None and args.max_turns < 0:
        raise ValueError("--max_turns must be non-negative")

    webshop_num_products = parse_webshop_num_products(args.webshop_num_products)
    instance_ids = parse_instance_ids(args.instance_ids)
    benchmark_tasks = load_benchmark_tasks(
        gold_trajectory_path=args.gold_trajectory_path,
        num_instances=args.num_instances,
        instance_ids=instance_ids,
    )
    effective_parallelism = min(args.parallelism, len(benchmark_tasks))

    logger = RuntimeLogger()
    instances = _replay_instances(
        benchmark_tasks=benchmark_tasks,
        max_turns=args.max_turns,
        max_internal_steps=args.max_internal_steps,
        azure_api_version=args.azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=args.executor_type,
        parallelism=effective_parallelism,
    )
    for instance in instances:
        logger.log_instance(instance)

    logger.dump_json(args.output)
    print(
        f"Saved {len(logger.instances)} benchmark instances to {args.output} "
        f"(parallelism={effective_parallelism}, webshop_num_products={args.webshop_num_products}, "
        f"executor_type={args.executor_type})"
    )


if __name__ == "__main__":
    main()
