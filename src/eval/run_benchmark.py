from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from eval.benchmark_fixed_rollout import execute_fixed_user_turn
from eval.evaluators.constraint_importance_eval import attach_instance_evaluations
from eval.evaluators.runtime_logger import RuntimeLogger
from common.llm_clients import AzureOpenAIChatClient
from envs.webshop_env import WebShopEnvAdapter
from eval.fixed_user_llm_executor import FixedUserLLMWebShopExecutor
from models import BaseTask, DialogueInstance, TurnRecord
from prompt_logging import get_prompt_log_path

DEFAULT_MAX_INTERNAL_STEPS = 12
DEFAULT_WEBSHOP_NUM_PRODUCTS = "100000"


@dataclass
class BenchmarkTask:
    task: BaseTask
    source_instance: Dict[str, Any]


def parse_webshop_num_products(value: Any) -> Optional[int]:
    text = str(value if value is not None else DEFAULT_WEBSHOP_NUM_PRODUCTS).strip().lower()
    if text in {"all", "full", "large", "none"}:
        return None

    try:
        num_products = int(text)
    except ValueError as exc:
        raise ValueError(
            "--webshop_num_products must be one of 100, 1000, 100000, or all"
        ) from exc

    if num_products not in {100, 1000, 100000}:
        raise ValueError(
            "--webshop_num_products must be one of 100, 1000, 100000, or all"
        )
    return num_products


def configure_webshop_dataset(num_products: Optional[int]) -> None:
    dataset_mode = "all" if num_products is None or num_products > 1000 else "small"
    os.environ["WEBSHOP_DATASET"] = dataset_mode

    if dataset_mode != "all":
        return

    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "WebShop" / "data"
    search_index_name = "indexes" if num_products is None else "indexes_100k"
    required_paths = [
        data_dir / "items_shuffle.json",
        data_dir / "items_ins_v2_1000.json",
        repo_root / "WebShop" / "search_engine" / search_index_name,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Full WebShop data files are not present. Download/build the full dataset first.\n"
            f"Missing:\n  - {missing_text}"
        )


def load_local_dotenv(dotenv_path: str | None = None, override: bool = False) -> None:
    candidate_paths = []
    if dotenv_path:
        candidate_paths.append(Path(dotenv_path))
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidate_paths.extend([Path.cwd() / ".env", repo_root / ".env"])

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
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if override or key not in os.environ:
                os.environ[key] = value


def _fallback_initial_intention(request: str) -> Dict[str, Any]:
    return {
        "request": request,
        "constraints": {},
        "priority": [],
        "gold_search_query": request,
    }


def _task_from_payload(raw_task: Dict[str, Any], *, fallback_index: int) -> BaseTask:
    if not isinstance(raw_task, dict):
        raise ValueError(f"Task #{fallback_index} must be a JSON object")

    initial_intention = raw_task.get("initial_intention")
    world_state = copy.deepcopy(raw_task.get("world_state") or {"domain": "webshop"})
    if not isinstance(initial_intention, dict) and isinstance(raw_task.get("turns"), list):
        first_turn = raw_task["turns"][0] if raw_task["turns"] else {}
        if isinstance(first_turn, dict):
            turn_intention = first_turn.get("gold_current_intention")
            if isinstance(turn_intention, dict):
                initial_intention = copy.deepcopy(turn_intention)
            elif first_turn.get("user_utterance"):
                initial_intention = _fallback_initial_intention(str(first_turn["user_utterance"]))

    if isinstance(initial_intention, dict):
        request = initial_intention.get("request")
        if isinstance(request, str) and request.strip():
            world_state.setdefault("webshop_instruction_text", request.strip())

    if not isinstance(initial_intention, dict):
        raise ValueError(f"Task #{fallback_index} is missing a valid initial_intention object")

    return BaseTask(
        instance_id=str(raw_task.get("instance_id") or f"webshop_task_{fallback_index:03d}"),
        task_type=str(raw_task.get("task_type") or "transaction"),
        subtype=str(raw_task.get("subtype") or "shopping"),
        world_state=world_state,
        initial_intention=copy.deepcopy(initial_intention),
    )


def parse_instance_ids(value: Optional[str]) -> Optional[List[str]]:
    if value is None or not str(value).strip():
        return None
    ids = []
    seen = set()
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        normalized = _normalize_instance_id(part)
        if normalized not in seen:
            ids.append(normalized)
            seen.add(normalized)
    return ids or None


def _normalize_instance_id(value: str) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"web(?:shop_demo_)?(\d+)", text, flags=re.IGNORECASE)
    if match:
        number = int(match.group(1))
        if number == 0:
            number = 1
        return f"webshop_demo_{number:03d}"
    match = re.fullmatch(r"(\d+)", text)
    if match:
        number = int(match.group(1))
        if number == 0:
            number = 1
        return f"webshop_demo_{number:03d}"
    return text


def _partition_indexed_tasks(
    tasks: List[BenchmarkTask],
    num_partitions: int,
) -> List[List[Tuple[int, BenchmarkTask]]]:
    partitions: List[List[Tuple[int, BenchmarkTask]]] = [[] for _ in range(num_partitions)]
    for zero_based_index, task in enumerate(tasks):
        partitions[zero_based_index % num_partitions].append((zero_based_index + 1, task))
    return [partition for partition in partitions if partition]


def _build_runtime_components(
    *,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
) -> Tuple[WebShopEnvAdapter, Any, None, Any]:
    if executor_type != "fixed_user":
        raise ValueError("eval/run_benchmark.py only supports executor_type='fixed_user'")

    configure_webshop_dataset(webshop_num_products)

    import gym
    from web_agent_site.envs import WebAgentTextEnv

    raw_env = gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text",
        num_products=webshop_num_products,
        disable_env_checker=True,
    )
    if raw_env is None:
        raw_env = WebAgentTextEnv(
            observation_mode="text",
            num_products=webshop_num_products,
        )

    env = WebShopEnvAdapter(webshop_env=raw_env, action_style="auto")
    llm_client = AzureOpenAIChatClient.from_env(api_version=azure_api_version)
    agent = FixedUserLLMWebShopExecutor(llm_client=llm_client)
    return env, agent, None, raw_env


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
) -> str:
    utterance = source_turn.get("user_utterance")
    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()
    return ""


def _gold_constraint_eval_payload(
    env,
    env_feedback,
    gold_intention: Dict[str, Any],
) -> Dict[str, Any]:
    if env_feedback is None:
        return {
            "satisfied_constraints": [],
            "violated_constraints": [],
            "constraint_debug": {},
        }

    observation = env_feedback.observation or {}
    result = observation.get("extracted_result")
    if not isinstance(result, dict) or not result:
        result = observation.get("selected_candidate")
    if not isinstance(result, dict) or not result:
        result = observation.get("selected_item")

    selected_asin = observation.get("selected_asin")
    if (not isinstance(result, dict) or not result) and selected_asin:
        candidate_lookup = getattr(env, "_candidate_item_for_asin", None)
        if callable(candidate_lookup):
            result = candidate_lookup(selected_asin, observation.get("candidate_items") or [])

    check_constraints = getattr(env, "_check_constraints", None)
    if not callable(check_constraints) or not isinstance(result, dict) or not result:
        return {
            "satisfied_constraints": [],
            "violated_constraints": [],
            "constraint_debug": {},
        }

    satisfied, violated, constraint_debug = check_constraints(
        result,
        gold_intention,
        include_debug=True,
    )
    return {
        "satisfied_constraints": list(satisfied or []),
        "violated_constraints": list(violated or []),
        "constraint_debug": copy.deepcopy(constraint_debug or {}),
    }


def _benchmark_env_feedback_payload(
    env,
    env_feedback,
    gold_intention: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if env_feedback is None:
        return None

    observation = env_feedback.observation or {}
    gold_eval = _gold_constraint_eval_payload(env, env_feedback, gold_intention)
    return {
        "status": env_feedback.status,
        "feedback_type": "candidate_items",
        "page_type": observation.get("page_type"),
        "candidate_items": copy.deepcopy(observation.get("candidate_items") or []),
        "selected_candidate": copy.deepcopy(observation.get("selected_candidate")),
        "selected_item": copy.deepcopy(observation.get("selected_item")),
        "selected_asin": observation.get("selected_asin"),
        "agent_state_satisfied_constraints": list(env_feedback.satisfied_constraints or []),
        "agent_state_violated_constraints": list(env_feedback.violated_constraints or []),
        "agent_state_constraint_debug": copy.deepcopy(observation.get("constraint_debug") or {}),
        "gold_eval_satisfied_constraints": gold_eval["satisfied_constraints"],
        "gold_eval_violated_constraints": gold_eval["violated_constraints"],
        "gold_eval_constraint_debug": gold_eval["constraint_debug"],
        "rerank_info": copy.deepcopy(observation.get("rerank_info")),
    }


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
    gold_current_intention = copy.deepcopy(task.initial_intention)
    history: List[Dict[str, Any]] = []

    for turn_index, raw_turn in enumerate(replay_turns):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"Turn #{turn_index} in {task.instance_id} must be a JSON object")

        gold_current_intention = _resolve_turn_intention(raw_turn, gold_current_intention)
        user_utterance = _resolve_turn_user_utterance(raw_turn)
        history.append({"role": "user", "content": user_utterance})

        rollout = execute_fixed_user_turn(
            env=env,
            execution_agent=execution_agent,
            history=history,
            user_utterance=user_utterance,
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
                env_feedback=_benchmark_env_feedback_payload(env, env_feedback, gold_current_intention),
                trigger_evidence=copy.deepcopy(raw_turn.get("trigger_evidence")),
                shift_condition=copy.deepcopy(raw_turn.get("shift_condition")),
                gold_delta=copy.deepcopy(raw_turn.get("gold_delta") or {}),
                gold_current_intention=copy.deepcopy(gold_current_intention),
                linguistic_style=str(raw_turn.get("linguistic_style") or "explicit"),
                action_implication=str(
                    raw_turn.get("action_implication") or ("start_search" if turn_index == 0 else "continue")
                ),
                num_internal_steps=rollout.num_internal_steps,
                num_rollout_search_actions=rollout.num_search_actions,
                rollout_search_queries=list(rollout.search_queries),
                stop_reason=rollout.stop_reason,
                rollout_trace=rollout.rollout_trace,
                agent_intention_prediction=copy.deepcopy(
                    getattr(agent_action, "predicted_current_intention", None)
                    if agent_action is not None
                    else None
                ),
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
        default=r".\IntentionChangeBench\data\eval\benchmark_eval.json",
    )
    parser.add_argument(
        "--gold_trajectory_path",
        "--tasks_path",
        dest="gold_trajectory_path",
        type=str,
        default=r".\IntentionChangeBench\data\simulation\annotated_dataset.json",
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
        default=os.getenv("WEBSHOP_NUM_PRODUCTS", DEFAULT_WEBSHOP_NUM_PRODUCTS),
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
        choices=["fixed_user"],
        default="fixed_user",
        help="Execution agent to use during replay.",
    )
    args = parser.parse_args()
    print(f"Prompt log path: {get_prompt_log_path()}")

    if args.parallelism < 1:
        raise ValueError("--parallelism must be at least 1")
    if args.max_turns is not None and args.max_turns < 0:
        raise ValueError("--max_turns must be non-negative")
    if args.executor_type != "fixed_user":
        raise ValueError(
            "run_benchmark.py evaluates the original WebShop-style fixed_user executor only. "
            "Use simulation/simulation/run_simulation.py for the gold/direct BM25 executor."
        )

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
        attach_instance_evaluations(instance)
        logger.log_instance(instance)

    logger.dump_json(args.output)
    print(
        f"Saved {len(logger.instances)} benchmark instances to {args.output} "
        f"(parallelism={effective_parallelism}, webshop_num_products={args.webshop_num_products}, "
        f"executor_type={args.executor_type})"
    )


if __name__ == "__main__":
    main()
