from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.fixed_user_llm_executor import FixedUserLLMWebShopExecutor
from agents.executor import WebShopExecutor
from evaluators.runtime_logger import RuntimeLogger
from models import AgentAction, BaseTask, DialogueInstance, EnvFeedback, TurnRecord
from prompt_logging import get_prompt_log_path, log_prompt
from simulators.human_simulator import HumanSimulator
from simulators.llm_clients import AzureOpenAIChatClient
from envs.webshop_env import WebShopEnvAdapter

STYLE_POOL = ["explicit", "partial", "elliptical"]
DEFAULT_MAX_INTERNAL_STEPS = 12
DEFAULT_WEBSHOP_NUM_PRODUCTS = "100000"
ROLLOUT_CONSTRAINT_FIELDS = ("category", "color", "size", "brand")
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
    num_search_actions: int = 0
    search_queries: List[str] = field(default_factory=list)


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
    """
    Select the product JSON/attribute JSON before importing WebShop modules.

    WebShop has two independent switches:
    - data files: 1000-product files vs full files
    - search index: chosen by num_products in WebShop's init_search_engine
    """

    dataset_mode = "all" if num_products is None or num_products > 1000 else "small"
    os.environ["WEBSHOP_DATASET"] = dataset_mode

    if dataset_mode != "all":
        return

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "WebShop" / "data"
    required_files = [
        data_dir / "items_shuffle.json",
        data_dir / "items_ins_v2_1000.json",
    ]
    search_index_name = "indexes" if num_products is None else "indexes_100k"
    required_dirs = [
        repo_root / "WebShop" / "search_engine" / search_index_name,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        missing_text = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Full WebShop data files are not present. Download/build the full dataset first, "
            "then rerun with --webshop_num_products all or 100000.\n"
            f"Missing:\n  - {missing_text}\n"
            "Expected setup: from WebShop/, run `bash setup.sh -d all` or otherwise place "
            "`items_shuffle.json` under WebShop/data and build the matching search index. "
            "The default instruction/attribute file stays on the 1k subset; set "
            "WEBSHOP_ATTR_DATASET=all only if you also want full instructions."
        )


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


def make_webshop_goal_task(goal_index: int) -> BaseTask:
    return BaseTask(
        instance_id=f"webshop_goal_{goal_index:05d}",
        task_type="transaction",
        subtype="shopping",
        world_state={
            "domain": "webshop",
            "webshop_goal_index": goal_index,
        },
        initial_intention={
            "request": "",
            "constraints": {},
            "priority": [],
        },
    )


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


def load_webshop_tasks(
    *,
    tasks_path: Optional[str],
    num_instances: Optional[int],
    goal_indices: Optional[List[int]] = None,
    instance_ids: Optional[List[str]] = None,
) -> List[BaseTask]:
    if goal_indices is not None:
        tasks = [make_webshop_goal_task(goal_index) for goal_index in goal_indices]
        if num_instances is not None:
            tasks = tasks[:num_instances]
        return tasks

    if not tasks_path:
        total = num_instances or 10
        tasks = [make_demo_webshop_task(instance_index=i) for i in range(1, total + 1)]
        return _filter_tasks_by_instance_ids(tasks, instance_ids)

    path = Path(tasks_path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    raw_tasks: List[Dict[str, Any]]
    if path.is_dir():
        raw_tasks = []
        for file_path in sorted(path.glob("*.json")):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
                raw_tasks.extend(payload["tasks"])
            elif isinstance(payload, list):
                raw_tasks.extend(payload)
            elif isinstance(payload, dict):
                raw_tasks.append(payload)
            else:
                raise ValueError(f"Task file {file_path} must contain a JSON object or array")
    elif path.suffix.lower() == ".jsonl":
        raw_tasks = []
        for line_index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_index} in {path} is not a JSON object")
            raw_tasks.append(payload)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            raw_tasks = payload["tasks"]
        elif isinstance(payload, list):
            raw_tasks = payload
        else:
            raise ValueError(
                f"Task file {path} must be a JSON array, a JSONL file, or a JSON object with a 'tasks' array"
            )

    tasks = [
        _task_from_payload(raw_task, fallback_index=index)
        for index, raw_task in enumerate(raw_tasks, start=1)
    ]
    tasks = _filter_tasks_by_instance_ids(tasks, instance_ids)
    if not tasks:
        raise ValueError(f"Task path {path} did not contain any selected tasks")

    if num_instances is None:
        return tasks
    if num_instances < 1:
        raise ValueError("--num_instances must be at least 1")
    if num_instances > len(tasks):
        if instance_ids:
            return tasks
        raise ValueError(
            f"Requested {num_instances} tasks, but {path} only contains {len(tasks)} tasks"
        )
    return tasks[:num_instances]


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


def _filter_tasks_by_instance_ids(tasks: List[BaseTask], instance_ids: Optional[List[str]]) -> List[BaseTask]:
    if not instance_ids:
        return tasks
    wanted = set(instance_ids)
    selected = [task for task in tasks if task.instance_id in wanted]
    found = {task.instance_id for task in selected}
    missing = [instance_id for instance_id in instance_ids if instance_id not in found]
    if missing:
        raise ValueError(f"Could not find requested instance_id(s): {', '.join(missing)}")
    return selected


def parse_goal_indices(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not str(value).strip():
        return None

    indices: List[int] = []
    seen = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"Invalid goal index range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for index in values:
            if index < 0:
                raise ValueError("Goal indices must be non-negative")
            if index not in seen:
                seen.add(index)
                indices.append(index)
    if not indices:
        raise ValueError("--webshop_goal_indices did not contain any indices")
    return indices


def _clean_initial_request(instruction: str) -> str:
    return re.sub(r"^\s*Instruction:\s*", "", instruction or "", flags=re.IGNORECASE).strip()


def _fallback_initial_intention(request: str) -> Dict[str, Any]:
    return {
        "request": request,
        "constraints": {},
        "priority": [],
        "gold_search_query": request,
    }


def _normalize_initial_constraint_key(key: Any) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(key or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    aliases = {
        "max_price": "budget_max",
        "maximum_price": "budget_max",
        "price_max": "budget_max",
        "budget": "budget_max",
        "budget_limit": "budget_max",
        "product_type": "category",
        "item_type": "category",
    }
    return aliases.get(normalized, normalized)


def _normalize_initial_constraint_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        if not value or value.lower() in {"none", "null", "unknown", "not specified"}:
            return None

    if field == "budget_max":
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[0-9]+(?:\.[0-9]+)?", value.replace(",", ""))
            if match:
                return float(match.group(0))
        return None

    return value


def _sanitize_llm_initial_intention(raw_intention: Any, request: str) -> Dict[str, Any]:
    if not isinstance(raw_intention, dict):
        return _fallback_initial_intention(request)

    raw_constraints = raw_intention.get("constraints") or {}
    constraints: Dict[str, Any] = {}
    if isinstance(raw_constraints, dict):
        for raw_field, raw_value in raw_constraints.items():
            field = _normalize_initial_constraint_key(raw_field)
            if not field or field.endswith("_exact"):
                continue
            value = _normalize_initial_constraint_value(field, raw_value)
            if value is not None:
                constraints[field] = value

    raw_priority = raw_intention.get("priority") or []
    priority: List[str] = []
    if isinstance(raw_priority, list):
        for raw_field in raw_priority:
            field = _normalize_initial_constraint_key(raw_field)
            if field in constraints and field not in priority:
                priority.append(field)
    for field in constraints:
        if field not in priority:
            priority.append(field)

    llm_request = raw_intention.get("request")
    if isinstance(llm_request, str) and llm_request.strip():
        request = llm_request.strip()

    gold_search_query = raw_intention.get("gold_search_query")
    if isinstance(gold_search_query, str) and gold_search_query.strip():
        gold_search_query = re.sub(r"\s+", " ", gold_search_query).strip()
    else:
        query_parts: List[str] = []
        category = constraints.get("category")
        if category:
            query_parts.append(str(category))
        for field in ("color", "brand", "size"):
            value = constraints.get(field)
            if value is not None:
                query_parts.append(str(value))
        gold_search_query = re.sub(r"\s+", " ", " ".join(query_parts)).strip() or request

    return {
        "request": request,
        "constraints": constraints,
        "priority": priority,
        "gold_search_query": gold_search_query,
    }


def _llm_initial_intention_from_instruction(
    instruction: str,
    llm_client: Any,
) -> Optional[Dict[str, Any]]:
    request = _clean_initial_request(instruction)
    if not request:
        return None
    if llm_client is None or not hasattr(llm_client, "generate_json"):
        return _fallback_initial_intention(request)

    prompt = f"""
Convert the initial WebShop instruction into benchmark intention JSON.
Return one JSON object only.

Schema:
{{
  "request": "the original user request, cleaned but not rewritten",
  "constraints": {{
    "category": "product category or null",
    "budget_max": "maximum price as a number or null",
    "color": "requested color option or null",
    "brand": "requested brand only if explicitly stated or null",
    "size": "requested size option or null"
  }},
  "priority": ["ordered constraint fields that matter most"],
  "gold_search_query": "concise WebShop/BM25 keyword query for this instruction"
}}

Rules:
- Extract constraints from the instruction semantics, not with regex-style substring guesses.
- Generate gold_search_query as search keywords, not a full sentence. Include product type and positive search-relevant attributes such as color, brand, or size.
- Do not include budget/price limits in gold_search_query unless those words are naturally part of the item name; budget is evaluated after search.
- Preserve option values exactly when they are explicit labels, e.g. color: dusty blush.
- Use budget_max for "price lower than", "under", "below", or similar maximum-price language.
- Set brand only when the instruction explicitly names a brand, uses a brand label, or says by/from a brand.
- Do not infer brand from dimensions, quoted fragments, size strings, or punctuation. For example, 52"w x 54"l is a size, not a brand.
- Omit constraints whose value is unknown instead of inventing them.
- Do not output *_exact fields.

Instruction:
{request}
""".strip()

    log_prompt("initial_intention", prompt)
    try:
        raw_intention = llm_client.generate_json(prompt)
    except Exception:
        return _fallback_initial_intention(request)
    return _sanitize_llm_initial_intention(raw_intention, request)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _requested_rollout_constraints(current_intention: Dict[str, Any]) -> Dict[str, Any]:
    constraints = current_intention.get("constraints", {}) or {}
    requested: Dict[str, Any] = {}
    for field in ROLLOUT_CONSTRAINT_FIELDS:
        desired = constraints.get(field)
        if desired is not None:
            requested[field] = desired
    return requested


def _requested_selectable_constraints(current_intention: Dict[str, Any]) -> Dict[str, Any]:
    constraints = current_intention.get("constraints", {}) or {}
    requested: Dict[str, Any] = {}
    for field in SELECTABLE_CONSTRAINT_FIELDS:
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


def _all_requested_rollout_constraints_satisfied(
    current_intention: Dict[str, Any],
    env_feedback: Optional[EnvFeedback],
) -> bool:
    requested = _requested_rollout_constraints(current_intention)
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
    requested_rollout = _requested_rollout_constraints(current_intention)
    if "category" in requested_rollout and _constraint_match_status(env_feedback, "category") is not True:
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


def _public_observation_payload(observation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = copy.deepcopy(observation or {})
    payload.pop("constraint_debug", None)
    payload.pop("extracted_result", None)
    return payload


def _public_env_feedback_payload(env_feedback: Optional[EnvFeedback]) -> Optional[Dict[str, Any]]:
    if env_feedback is None:
        return None

    observation = env_feedback.observation or {}
    return {
        "status": env_feedback.status,
        "feedback_type": "candidate_items",
        "page_type": observation.get("page_type"),
        "candidate_items": copy.deepcopy(observation.get("candidate_items") or []),
        "selected_candidate": copy.deepcopy(observation.get("selected_candidate")),
    }


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
                key: copy.deepcopy(item.get(key))
                for key in ("rank", "asin", "title", "price", "category", "brand", "color")
                if item.get(key) is not None
            }
        )
    return returned_items


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
    if _all_requested_rollout_constraints_satisfied(current_intention, env_feedback):
        return "rollout_options_satisfied"
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
        "stop_reason": stop_reason,
    }


def execute_turn(
    env: WebShopEnvAdapter,
    execution_agent,
    history: List[Dict[str, Any]],
    user_utterance: str,
    current_intention: Dict[str, Any],
    env_observation: Dict[str, Any],
    max_internal_steps: int = DEFAULT_MAX_INTERNAL_STEPS,
) -> TurnRolloutResult:
    direct_search = getattr(execution_agent, "search", None)
    if callable(direct_search):
        agent_action, env_feedback = direct_search(env, current_intention, user_utterance)
        query = str((agent_action.action_payload or {}).get("query", ""))
        rollout_trace = [
            _build_rollout_trace_entry(
                1,
                agent_action,
                env_feedback,
                state_changed=True,
                made_progress=bool((env_feedback.observation or {}).get("candidate_items")),
                stop_reason="direct_search",
            )
        ]
        return TurnRolloutResult(
            final_action=agent_action,
            final_env_feedback=env_feedback,
            rollout_trace=rollout_trace,
            num_internal_steps=1,
            stop_reason="direct_search",
            num_search_actions=1 if query else 0,
            search_queries=[query] if query else [],
        )

    working_history = copy.deepcopy(history)
    rollout_trace: List[Dict[str, Any]] = []
    previous_feedback = _maybe_summarize_current_state(env, current_intention)

    current_observation = copy.deepcopy(env_observation)
    final_feedback = previous_feedback
    final_action: Optional[AgentAction] = None
    previous_action_signature: Optional[Tuple[str, Tuple[Tuple[str, str], ...]]] = None
    repeated_action_streak = 0
    stagnant_steps = 0
    search_queries: List[str] = []

    for step_index in range(1, max_internal_steps + 1):
        agent_action = execution_agent.act(working_history, user_utterance, current_observation)
        if agent_action.action_type in {"search", "refine"}:
            search_queries.append(str((agent_action.action_payload or {}).get("query", "")))
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

        if agent_action.action_type == "buy":
            stop_reason = "virtual_buy"
        else:
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
            return TurnRolloutResult(
                final_action=final_action,
                final_env_feedback=final_feedback,
                rollout_trace=rollout_trace,
                num_internal_steps=step_index,
                stop_reason=stop_reason,
                num_search_actions=len(search_queries),
                search_queries=list(search_queries),
            )

    return TurnRolloutResult(
        final_action=final_action,
        final_env_feedback=final_feedback,
        rollout_trace=rollout_trace,
        num_internal_steps=len(rollout_trace),
        stop_reason="step_budget",
        num_search_actions=len(search_queries),
        search_queries=list(search_queries),
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
    llm_initial_intention = _llm_initial_intention_from_instruction(
        real_instruction,
        getattr(human_simulator, "llm_client", None),
    )
    if llm_initial_intention is not None:
        current_intention = llm_initial_intention
    elif real_instruction and real_instruction.strip():
        current_intention = _fallback_initial_intention(_clean_initial_request(real_instruction))

    user_utterance = _clean_initial_request(real_instruction)
    gold_delta: Dict[str, Dict[str, Any]] = {}
    trigger_evidence: Optional[Dict[str, Any]] = None
    shift_condition: Optional[Dict[str, Any]] = None
    linguistic_style = "explicit"
    action_implication = "start_search"

    history: List[Dict[str, Any]] = [{"role": "user", "content": user_utterance}]
    intention_history: List[Dict[str, Any]] = [
        {
            "turn_id": 0,
            "user_utterance": user_utterance,
            "gold_intention": copy.deepcopy(current_intention),
        }
    ]

    for turn_id in range(max_turns + 1):
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
                turn_id=turn_id,
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
                trigger_evidence=trigger_evidence,
                shift_condition=shift_condition,
                gold_delta=gold_delta,
                gold_current_intention=copy.deepcopy(current_intention),
                linguistic_style=linguistic_style,
                action_implication=action_implication,
                num_internal_steps=rollout.num_internal_steps,
                num_rollout_search_actions=rollout.num_search_actions,
                rollout_search_queries=list(rollout.search_queries),
                stop_reason=rollout.stop_reason,
                rollout_trace=rollout.rollout_trace,
            )
        )

        env_obs = env.get_observation()

        if turn_id >= max_turns:
            break

        style = rng.choice(STYLE_POOL)
        shift = human_simulator.decide_shift(
            current_intention,
            env_feedback=env_feedback,
            intention_history=intention_history[:-1],
        )
        new_intention, delta = human_simulator.apply_shift(current_intention, shift)
        user_utt = human_simulator.realize_shift(
            shift,
            current_intention,
            style,
            env_feedback=env_feedback,
            intention_history=intention_history[:-1],
        )
        if env.done and not delta:
            break

        shift_condition = None
        trigger_evidence = {
            "trigger_type": "none",
            "source": "simulator",
            "details": {},
        }
        intention_changed = shift.intention_changed if shift.intention_changed is not None else shift.op != "none"
        condition = shift.condition or "none"
        change_category = shift.change_category or (shift.op if shift.op != "none" else "none")
        if intention_changed:
            shift_condition = {
                "type": condition,
                "reason": shift.rationale,
                "source": "simulator",
                "details": {
                    "intention_changed": intention_changed,
                    "condition": condition,
                    "change_category": change_category,
                    "op": shift.op,
                    "field": shift.field,
                    "old_value": shift.old_value,
                    "value": shift.value,
                    "priority_update": shift.priority_update,
                },
            }
            trigger_evidence = {
                "trigger_type": condition,
                "source": "simulator",
                "details": {
                    "change_category": change_category,
                    "op": shift.op,
                    "field": shift.field,
                    "rationale": shift.rationale,
                },
            }

        current_intention = new_intention
        user_utterance = user_utt
        gold_delta = delta
        linguistic_style = style
        action_implication = "continue"

        history = [{"role": "user", "content": user_utterance}]
        intention_history.append(
            {
                "turn_id": turn_id + 1,
                "user_utterance": user_utterance,
                "gold_intention": copy.deepcopy(current_intention),
            }
        )

    return DialogueInstance(
        instance_id=task.instance_id,
        task_type=task.task_type,
        subtype=task.subtype,
        world_state=task.world_state,
        turns=turns,
    )


def _build_runtime_components(
    *,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
) -> Tuple[WebShopEnvAdapter, Any, HumanSimulator, Any]:
    configure_webshop_dataset(webshop_num_products)

    import gym
    from web_agent_site.envs import WebAgentTextEnv

    # Each task gets isolated runtime state so parallel runs do not share a
    # mutable WebShop session or agent history.
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
    if executor_type == "fixed_user":
        agent = FixedUserLLMWebShopExecutor(llm_client=llm_client)
    else:
        agent = WebShopExecutor(llm_client=llm_client)
    human = HumanSimulator(llm_client=llm_client)
    return env, agent, human, raw_env


def _simulate_single_instance(
    *,
    task: BaseTask,
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
) -> DialogueInstance:
    env, agent, human, raw_env = _build_runtime_components(
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
    )
    try:
        return simulate_dialogue_instance(
            task=task,
            env=env,
            execution_agent=agent,
            human_simulator=human,
            max_turns=max_turns,
            max_internal_steps=max_internal_steps,
            seed=seed,
        )
    finally:
        close_env = getattr(raw_env, "close", None)
        if callable(close_env):
            close_env()


def _simulate_instances_serial(
    *,
    tasks: List[BaseTask],
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
) -> List[DialogueInstance]:
    env, agent, human, raw_env = _build_runtime_components(
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
    )
    try:
        instances = []
        for task_index, task in enumerate(tasks, start=1):
            instances.append(
                simulate_dialogue_instance(
                    task=task,
                    env=env,
                    execution_agent=agent,
                    human_simulator=human,
                    max_turns=max_turns,
                    max_internal_steps=max_internal_steps,
                    seed=seed + task_index - 1,
                )
            )
        return instances
    finally:
        close_env = getattr(raw_env, "close", None)
        if callable(close_env):
            close_env()


def _simulate_task_batch(
    *,
    indexed_tasks: List[Tuple[int, BaseTask]],
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
) -> Dict[int, DialogueInstance]:
    env, agent, human, raw_env = _build_runtime_components(
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
    )
    try:
        instances_by_index = {}
        for task_index, task in indexed_tasks:
            instances_by_index[task_index] = simulate_dialogue_instance(
                task=task,
                env=env,
                execution_agent=agent,
                human_simulator=human,
                max_turns=max_turns,
                max_internal_steps=max_internal_steps,
                seed=seed + task_index - 1,
            )
        return instances_by_index
    finally:
        close_env = getattr(raw_env, "close", None)
        if callable(close_env):
            close_env()


def _partition_indexed_tasks(
    tasks: List[BaseTask],
    num_partitions: int,
) -> List[List[Tuple[int, BaseTask]]]:
    partitions: List[List[Tuple[int, BaseTask]]] = [
        [] for _ in range(num_partitions)
    ]
    for zero_based_index, task in enumerate(tasks):
        partitions[zero_based_index % num_partitions].append(
            (zero_based_index + 1, task)
        )
    return [partition for partition in partitions if partition]


def _simulate_instances(
    *,
    tasks: List[BaseTask],
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
    parallelism: int,
) -> List[DialogueInstance]:
    if parallelism <= 1:
        return _simulate_instances_serial(
            tasks=tasks,
            seed=seed,
            max_turns=max_turns,
            max_internal_steps=max_internal_steps,
            azure_api_version=azure_api_version,
            webshop_num_products=webshop_num_products,
            executor_type=executor_type,
        )

    effective_parallelism = min(parallelism, len(tasks))
    task_batches = _partition_indexed_tasks(tasks, effective_parallelism)
    instances_by_index: Dict[int, DialogueInstance] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_batch_index = {
            executor.submit(
                _simulate_task_batch,
                indexed_tasks=batch,
                seed=seed,
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
                raise RuntimeError(
                    f"Failed while simulating task batch #{batch_index}"
                ) from exc

    return [
        instances_by_index[task_index]
        for task_index in sorted(instances_by_index)
    ]


def main():
    load_local_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=r".\IntentionChangeBench\data\webshop_simulated_dataset.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_internal_steps", type=int, default=DEFAULT_MAX_INTERNAL_STEPS)
    parser.add_argument("--tasks_path", type=str, default=None)
    parser.add_argument(
        "--instance_ids",
        type=str,
        default=None,
        help=(
            "Comma-separated instance ids to run from --tasks_path, e.g. "
            "webshop_demo_004,webshop_demo_010 or shorthand web4,web10."
        ),
    )
    parser.add_argument(
        "--webshop_goal_indices",
        type=str,
        default=None,
        help="Comma-separated WebShop goal indices/ranges, e.g. 0,3,10-12.",
    )
    parser.add_argument("--num_instances", type=int, default=20)
    parser.add_argument(
        "--webshop_num_products",
        type=str,
        default=os.getenv("WEBSHOP_NUM_PRODUCTS", DEFAULT_WEBSHOP_NUM_PRODUCTS),
        help=(
            "WebShop product subset to load: 100, 1000, 100000, or all. "
            "Use all after downloading the full WebShop data and building the full search index."
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of tasks to simulate concurrently. Use the same value as the number of selected tasks for one task per worker.",
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
        default="llm",
        help=(
            "Execution agent to use. `fixed_user` only conditions on trajectory "
            "user utterances and ignores assistant/context history."
        ),
    )
    args = parser.parse_args()
    print(f"Prompt log path: {get_prompt_log_path()}")

    if args.parallelism < 1:
        raise ValueError("--parallelism must be at least 1")
    webshop_num_products = parse_webshop_num_products(args.webshop_num_products)
    instance_ids = parse_instance_ids(args.instance_ids)
    goal_indices = parse_goal_indices(args.webshop_goal_indices)

    tasks = load_webshop_tasks(
        tasks_path=args.tasks_path,
        num_instances=args.num_instances,
        goal_indices=goal_indices,
        instance_ids=instance_ids,
    )
    effective_parallelism = min(args.parallelism, len(tasks))
    logger = RuntimeLogger()

    instances = _simulate_instances(
        tasks=tasks,
        seed=args.seed,
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
        f"Saved {len(logger.instances)} instances to {args.output} "
        f"(parallelism={effective_parallelism}, webshop_num_products={args.webshop_num_products}, "
        f"executor_type={args.executor_type})"
    )
    return


if __name__ == "__main__":
    main()
