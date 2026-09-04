from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import json
import os
import random
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.llm_clients import create_llm_client_from_env
from models import AgentAction, BaseTask, DialogueInstance, EnvFeedback, TurnRecord
from prompt_logging import get_prompt_log_path, log_prompt
from simulation.simulation.reranker import RerankerConfig
from simulation.simulation.runtime_logger import RuntimeLogger
from simulation.simulation.base_user_simulator import HumanSimulator, ShiftDistributionController
from domains.travelplanner import (
    TravelPlannerEnvAdapter,
    TravelPlannerExecutor,
    TravelPlannerUserSimulator,
    load_travelplanner_ref_info,
)
from domains.webshop import WebShopEnvAdapter, WebShopExecutor, WebShopUserSimulator

STYLE_POOL = ["explicit", "partial", "elliptical"]


@dataclass(frozen=True)
class ShiftSamplingConfig:
    """Domain-aware shift sampling and distribution-control settings."""

    multi_change_rate: float = 0.0
    multi_candidate_samples: int = 1
    max_multi_candidate_samples: int = 1
    distribution_controller: Optional[ShiftDistributionController] = None


def _balanced_style_schedule(num_shifts: int, rng: random.Random) -> List[str]:
    schedule = [STYLE_POOL[index % len(STYLE_POOL)] for index in range(num_shifts)]
    rng.shuffle(schedule)
    return schedule


def _multi_preference_schedule(
    num_shifts: int,
    rate: float,
    rng: random.Random,
) -> Set[int]:
    """Choose approximately `rate` slots without specifying a change count."""
    if num_shifts <= 0 or rate <= 0:
        return set()
    expected = min(1.0, rate) * num_shifts
    count = int(expected)
    if rng.random() < expected - count:
        count += 1
    return set(rng.sample(range(num_shifts), k=min(count, num_shifts)))


def _distribution_controller_from_baseline(
    baseline_path: str,
    balance_strength: float,
    control_mode: str = "prompt",
    domain: str = "webshop",
) -> ShiftDistributionController:
    with open(baseline_path, "r", encoding="utf-8") as handle:
        instances = json.load(handle)

    category_counts: Dict[str, int] = {}
    condition_counts: Dict[str, int] = {}
    for instance in instances:
        for turn in instance.get("turns", []):
            shift_condition = turn.get("shift_condition") or {}
            if not shift_condition:
                continue
            condition = str(shift_condition.get("type") or "none")
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
            details = shift_condition.get("details") or {}
            changes = details.get("changes") or []
            categories = [
                str(change.get("change_category") or change.get("op") or "none")
                for change in changes
                if isinstance(change, dict)
            ]
            if not categories:
                categories = [str(details.get("change_category") or details.get("op") or "none")]
            for category in categories:
                if domain == "travelplanner" and category == "scope_correction":
                    category = "entity"
                category_counts[category] = category_counts.get(category, 0) + 1

    if domain == "travelplanner":
        controlled_categories = ["add", "relax", "override", "reprioritize", "entity"]
        controlled_conditions = [
            "user_preference",
            "real_world_feasibility",
            "agent_misunderstanding",
        ]
    else:
        controlled_categories = ["add", "relax", "override", "reprioritize"]
        controlled_conditions = ["user_preference", "real_world_feasibility"]

    return ShiftDistributionController(
        category_counts=category_counts,
        condition_counts=condition_counts,
        categories=controlled_categories,
        conditions=controlled_conditions,
        balance_strength=balance_strength,
        control_mode=control_mode,
    )
DEFAULT_MAX_INTERNAL_STEPS = 12
DEFAULT_TRAVELPLANNER_MAX_INTERNAL_STEPS = 30
TRAVELPLANNER_CASE_RETRIES = 3
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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


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

    repo_root = Path(__file__).resolve().parents[4]
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
            "then rerun with --webshop_num_products all.\n"
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
        repo_root = Path(__file__).resolve().parents[3]
        candidate_paths.extend(
            [
                Path.cwd() / ".env",
                repo_root / ".env",
                Path.cwd() / ".env.llm",
                repo_root / ".env.llm",
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
    selection_metadata = raw_task.get("selection_metadata")
    if isinstance(selection_metadata, dict):
        world_state["webshop_selection_metadata"] = copy.deepcopy(selection_metadata)
        selected_instruction = selection_metadata.get("instruction")
        if isinstance(selected_instruction, str) and selected_instruction.strip():
            world_state["webshop_instruction_text"] = selected_instruction.strip()
        selected_price_upper = selection_metadata.get("price_upper")
        if isinstance(selected_price_upper, (int, float)):
            world_state["webshop_fixed_price_upper"] = float(selected_price_upper)
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


def _travelplanner_initial_intention(raw_task: Dict[str, Any]) -> Dict[str, Any]:
    query_data = raw_task.get("travelplanner_query_data") or raw_task
    local_constraint = copy.deepcopy(query_data.get("local_constraint") or {})
    if isinstance(local_constraint, str):
        try:
            local_constraint = ast.literal_eval(local_constraint)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "TravelPlanner local_constraint must be a mapping or a Python/JSON mapping string"
            ) from exc
    if not isinstance(local_constraint, dict):
        raise ValueError("TravelPlanner local_constraint must resolve to a mapping")
    constraints: Dict[str, Any] = {}
    for source, target in (
        ("budget", "budget"),
        ("days", "days"),
        ("people_number", "people_number"),
        ("org", "org"),
        ("dest", "dest"),
        ("visiting_city_number", "visiting_city_number"),
    ):
        if query_data.get(source) is not None:
            constraints[target] = copy.deepcopy(query_data[source])

    raw_dates = query_data.get("date")
    if isinstance(raw_dates, str):
        try:
            raw_dates = ast.literal_eval(raw_dates)
        except (SyntaxError, ValueError):
            raw_dates = None
    if isinstance(raw_dates, (list, tuple)) and raw_dates:
        constraints["start_date"] = copy.deepcopy(raw_dates[0])
        constraints["end_date"] = copy.deepcopy(raw_dates[-1])
    else:
        if query_data.get("start_date") is not None:
            constraints["start_date"] = copy.deepcopy(query_data["start_date"])
        if query_data.get("end_date") is not None:
            constraints["end_date"] = copy.deepcopy(query_data["end_date"])

    for source, target in (
        ("cuisine", "cuisine"),
        ("room type", "room_type"),
        ("room_type", "room_type"),
        ("house rule", "house_rule"),
        ("house_rule", "house_rule"),
        ("transportation", "transportation"),
    ):
        if local_constraint.get(source) is not None:
            constraints[target] = copy.deepcopy(local_constraint[source])

    request = str(query_data.get("query") or raw_task.get("query") or "").strip()
    if constraints.get("people_number") is None:
        inferred_people = _travelplanner_people_number_from_query(request)
        if inferred_people is not None:
            constraints["people_number"] = inferred_people

    priority = [
        field
        for field in (
            "dest",
            "start_date",
            "end_date",
            "days",
            "budget",
            "transportation",
            "cuisine",
            "room_type",
            "house_rule",
        )
        if constraints.get(field) is not None
    ]
    intention = {
        "request": request,
        "constraints": constraints,
        "priority": priority,
    }
    from domains.travelplanner.entity_intention import ensure_entity_state

    return ensure_entity_state(intention)


def _travelplanner_people_number_from_query(query: str) -> Optional[int]:
    text = str(query or "").strip().lower()
    if not text:
        return None
    if any(phrase in text for phrase in ("solo traveler", "solo trip", "lone traveler")):
        return 1
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "a": 1,
    }
    patterns = (
        r"(?:for|party of|group of)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a)\s+(?:people|persons?|travelers?|individuals?)\b",
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons?|travelers?|individuals?)\b",
        r"(?:itinerary|trip|journey|plan)\s+for\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b(?!\s*(?:days?|nights?|cities)\b)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        token = match.group(1)
        if token.isdigit():
            return max(1, int(token))
        return number_words[token]
    return None


def _travelplanner_task_from_payload(
    raw_task: Dict[str, Any],
    *,
    fallback_index: int,
    reference_information: Any = None,
) -> BaseTask:
    if not isinstance(raw_task, dict):
        raise ValueError(f"TravelPlanner task #{fallback_index} must be a JSON object")

    supplied_world_state = raw_task.get("world_state")
    if not isinstance(supplied_world_state, dict):
        supplied_world_state = {}
    query_data = copy.deepcopy(
        raw_task.get("travelplanner_query_data")
        or supplied_world_state.get("travelplanner_query_data")
        or raw_task
    )
    for field in ("local_constraint", "date"):
        value = query_data.get(field)
        if not isinstance(value, str) or not value.strip().startswith(("{", "[")):
            continue
        try:
            query_data[field] = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"TravelPlanner {field} must contain a valid Python/JSON literal"
            ) from exc
    ref_info = raw_task.get("reference_information")
    if ref_info is None:
        ref_info = supplied_world_state.get("reference_information")
    if ref_info is None:
        ref_info = reference_information
    world_state = copy.deepcopy(supplied_world_state)
    world_state.update(
        {
            "domain": "travelplanner",
            "travelplanner_query_data": query_data,
            "reference_information": copy.deepcopy(ref_info),
        }
    )
    initial_intention = raw_task.get("initial_intention")
    # A prior simulation dataset is a convenient, self-contained task source
    # for regression runs. Rebuild its initial state from authoritative query
    # data so stale generated priority/entity state is not carried forward.
    if not isinstance(initial_intention, dict) and isinstance(raw_task.get("turns"), list):
        initial_intention = _travelplanner_initial_intention(query_data)
    if not isinstance(initial_intention, dict):
        initial_intention = _travelplanner_initial_intention(query_data)
    else:
        from domains.travelplanner.entity_intention import ensure_entity_state

        initial_intention = ensure_entity_state(initial_intention)

    return BaseTask(
        instance_id=str(raw_task.get("instance_id") or f"travelplanner_{fallback_index:05d}"),
        task_type=str(raw_task.get("task_type") or "planning"),
        subtype=str(raw_task.get("subtype") or "travel"),
        world_state=world_state,
        initial_intention=copy.deepcopy(initial_intention),
    )


def load_travelplanner_tasks(
    *,
    tasks_path: Optional[str],
    num_instances: Optional[int],
    set_type: str = "validation",
    instance_ids: Optional[List[str]] = None,
) -> List[BaseTask]:
    ref_infos = load_travelplanner_ref_info(set_type)
    raw_tasks: List[Dict[str, Any]] = []

    if tasks_path:
        path = Path(tasks_path)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            for line_index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if line:
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
            elif isinstance(payload, dict):
                raw_tasks = [payload]
            else:
                raise ValueError(f"Task file {path} must contain a JSON object or array")
    else:
        try:
            from datasets import load_dataset

            dataset = load_dataset("osunlp/TravelPlanner", set_type)[set_type]
            limit = num_instances or 10
            raw_tasks = []
            for index in range(min(limit, len(dataset))):
                row = dict(dataset[index])
                # Hugging Face stores reference_information as a large Python-
                # literal string containing rendered tables. Prefer the local
                # structured *_ref_info.jsonl row, which is index-aligned with
                # the query split and is directly searchable by the executor.
                row.pop("reference_information", None)
                raw_tasks.append(row)
        except Exception as exc:
            raise RuntimeError(
                "TravelPlanner tasks require --tasks_path, or an available cached/online "
                "`datasets.load_dataset('osunlp/TravelPlanner', set_type)` source."
            ) from exc

    tasks = [
        _travelplanner_task_from_payload(
            raw_task,
            fallback_index=index,
            reference_information=ref_infos[index - 1] if index - 1 < len(ref_infos) else None,
        )
        for index, raw_task in enumerate(raw_tasks, start=1)
    ]
    tasks = _filter_tasks_by_instance_ids(tasks, instance_ids)
    if num_instances is not None:
        tasks = tasks[:num_instances]
    if not tasks:
        raise ValueError("No TravelPlanner tasks were selected")
    return tasks


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


def _merge_selection_metadata_initial_intention(
    intention: Dict[str, Any],
    selection_metadata: Any,
) -> Dict[str, Any]:
    """Merge authoritative task attributes/options that the LLM schema may omit."""
    merged = copy.deepcopy(intention or {})
    if not isinstance(selection_metadata, dict):
        return merged
    constraints = merged.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        merged["constraints"] = constraints

    query = str(selection_metadata.get("query") or "").strip()
    if query:
        constraints.setdefault("category", query)
    price_upper = selection_metadata.get("price_upper")
    if isinstance(price_upper, (int, float)) and float(price_upper) < 1_000_000:
        constraints.setdefault("budget_max", float(price_upper))
    for raw_key, value in (selection_metadata.get("options") or {}).items():
        key = _normalize_initial_constraint_key(raw_key)
        if key and value not in (None, ""):
            constraints.setdefault(key, copy.deepcopy(value))
    for attribute in selection_metadata.get("attributes") or []:
        key = _normalize_initial_constraint_key(attribute)
        if key:
            constraints.setdefault(key, True)

    priority = merged.get("priority")
    if not isinstance(priority, list):
        priority = []
        merged["priority"] = priority
    for key in constraints:
        if key not in priority:
            priority.append(key)
    return merged


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
  "priority": ["ordered constraint fields that matter most"]
}}

Rules:
- Extract constraints from the instruction semantics, not with regex-style substring guesses.
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
        str(observation.get("tool_name") or ""),
        str(observation.get("tool_argument") or ""),
        observation.get("notebook_size") or len(observation.get("notebook") or []),
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

    if current_obs.get("tool_name") and (
        current_obs.get("tool_name"),
        current_obs.get("tool_argument"),
    ) != (
        prev_obs.get("tool_name"),
        prev_obs.get("tool_argument"),
    ):
        return True

    if len(current_obs.get("notebook") or []) > len(prev_obs.get("notebook") or []):
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
    if str(observation.get("domain") or "").lower() == "travelplanner":
        return {
            "status": env_feedback.status,
            "feedback_type": observation.get("feedback_type") or "travel_search_results",
            "page_type": observation.get("page_type"),
            "search_results": copy.deepcopy(observation.get("search_results") or {}),
            "satisfied_constraints": list(env_feedback.satisfied_constraints or []),
            "violated_constraints": list(env_feedback.violated_constraints or []),
            "constraint_debug": copy.deepcopy(observation.get("constraint_debug") or {}),
        }
    return {
        "status": env_feedback.status,
        "feedback_type": observation.get("feedback_type") or "candidate_items",
        "page_type": observation.get("page_type"),
        "candidate_items": copy.deepcopy(observation.get("candidate_items") or []),
        "candidate_diversity": copy.deepcopy(observation.get("candidate_diversity")),
        "selected_candidate": copy.deepcopy(observation.get("selected_candidate")),
        "rerank_info": copy.deepcopy(observation.get("rerank_info")),
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
    stop_on_candidate_ready: bool = True,
) -> Optional[str]:
    if env_feedback is None:
        return "no_feedback"
    if env_feedback.status == "error":
        return "error"
    if env_done:
        return "env_done"
    if stop_on_candidate_ready and _all_requested_rollout_constraints_satisfied(current_intention, env_feedback):
        return "rollout_options_satisfied"
    if stop_on_candidate_ready and _candidate_ready(current_intention, env_feedback):
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
    action_payload = dict(agent_action.action_payload or {})
    original_argument = str(action_payload.get("argument") or "").strip()
    trace_entry = {
        "step_index": step_index,
        "action": {
            "action_type": agent_action.action_type,
            "action_payload": action_payload,
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
    if str(observation.get("domain") or "").lower() == "travelplanner":
        trace_entry["action"]["original_action"] = f"{agent_action.action_type}[{original_argument}]"
        trace_entry.update(
            {
                "tool_name": observation.get("tool_name"),
                "tool_argument": observation.get("tool_argument"),
                "tool_result": copy.deepcopy(observation.get("tool_result")),
                "notebook_size": observation.get("notebook_size") or len(observation.get("notebook") or []),
            }
        )
    return trace_entry


def execute_turn(
    env,
    execution_agent,
    history: List[Dict[str, Any]],
    user_utterance: str,
    current_intention: Dict[str, Any],
    env_observation: Dict[str, Any],
    gold_delta: Optional[Dict[str, Dict[str, Any]]] = None,
    max_internal_steps: int = DEFAULT_MAX_INTERNAL_STEPS,
    stop_on_candidate_ready: bool = True,
) -> TurnRolloutResult:
    prepare_turn = getattr(env, "prepare_turn", None)
    if callable(prepare_turn):
        prepare_turn(current_intention, user_utterance, gold_delta or {})
        env_observation = env.get_observation()

    direct_execute = getattr(execution_agent, "execute", None)
    if callable(direct_execute):
        agent_action, env_feedback = direct_execute(
            env,
            current_intention,
            user_utterance,
            history=history,
            gold_delta=gold_delta or {},
        )
        rollout_trace = [
            _build_rollout_trace_entry(
                1,
                agent_action,
                env_feedback,
                state_changed=True,
                made_progress=not bool(env_feedback.violated_constraints),
                stop_reason="direct_execute",
            )
        ]
        return TurnRolloutResult(
            final_action=agent_action,
            final_env_feedback=env_feedback,
            rollout_trace=rollout_trace,
            num_internal_steps=1,
            stop_reason="direct_execute",
            num_search_actions=0,
            search_queries=[],
        )

    direct_search = getattr(execution_agent, "search", None)
    if callable(direct_search):
        agent_action, env_feedback = direct_search(
            env,
            current_intention,
            user_utterance,
            gold_delta=gold_delta or {},
        )
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
        elif agent_action.action_type in {
            "FlightSearch",
            "AttractionSearch",
            "AccommodationSearch",
            "RestaurantSearch",
            "CitySearch",
            "GoogleDistanceMatrix",
        }:
            payload = agent_action.action_payload or {}
            query = payload.get("query") or payload.get("argument")
            if not query:
                query = ", ".join(str(value) for value in payload.values() if value is not None)
            search_queries.append(f"{agent_action.action_type}[{query}]")
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
        elif agent_action.action_type == "Planner" and not getattr(env, "done", False):
            # Planner is terminal in the original TravelPlanner workflow. An
            # infeasible plan is evidence for the next user intention change,
            # not a request to repeatedly invoke Planner in the same turn.
            stop_reason = "planner_submitted"
        else:
            stop_reason = _rollout_stop_reason(
                current_intention,
                env_feedback,
                num_internal_steps=step_index,
                max_internal_steps=max_internal_steps,
                repeated_action_streak=repeated_action_streak,
                stagnant_steps=stagnant_steps,
                env_done=getattr(env, "done", False),
                stop_on_candidate_ready=stop_on_candidate_ready,
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

        history_content = {
            "action_type": agent_action.action_type,
            "action_payload": dict(agent_action.action_payload or {}),
            "env_result": copy.deepcopy(env_feedback.result or {}),
            "page_type": (env_feedback.observation or {}).get("page_type"),
            "selected_asin": (env_feedback.observation or {}).get("selected_asin"),
            "selected_options": copy.deepcopy((env_feedback.observation or {}).get("selected_options") or {}),
            "returned_items": _history_returned_items(env_feedback),
            "internal_step": step_index,
        }
        if str((env_feedback.observation or {}).get("domain") or "").lower() == "travelplanner":
            history_content.update(
                {
                    "tool_name": (env_feedback.observation or {}).get("tool_name"),
                    "tool_argument": (env_feedback.observation or {}).get("tool_argument"),
                    "tool_result": copy.deepcopy((env_feedback.observation or {}).get("tool_result")),
                    "notebook_size": (env_feedback.observation or {}).get("notebook_size")
                    or len((env_feedback.observation or {}).get("notebook") or []),
                }
            )
        working_history.append({"role": "assistant", "content": history_content})

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
    env,
    execution_agent,
    human_simulator: HumanSimulator,
    max_turns: int = 4,
    max_internal_steps: int = DEFAULT_MAX_INTERNAL_STEPS,
    seed: int = 7,
    shift_sampling_config: Optional[ShiftSamplingConfig] = None,
) -> DialogueInstance:
    reset_trajectory = getattr(human_simulator, "reset_trajectory", None)
    if callable(reset_trajectory):
        reset_trajectory()
    rng = random.Random(seed)
    sampling_config = shift_sampling_config or ShiftSamplingConfig()
    schedule_rng = random.Random(f"webshop-shift-schedule:{seed}")
    style_schedule = _balanced_style_schedule(max_turns, schedule_rng)
    multi_preferred_turns = _multi_preference_schedule(
        max_turns,
        sampling_config.multi_change_rate,
        schedule_rng,
    )
    turns: List[TurnRecord] = []

    current_intention = copy.deepcopy(task.initial_intention)
    env_obs = env.reset(task)
    real_instruction = env.get_instruction_text()
    domain = (task.world_state or {}).get("domain")
    if domain:
        current_intention.setdefault("domain", domain)
    if domain != "travelplanner":
        llm_initial_intention = _llm_initial_intention_from_instruction(
            real_instruction,
            getattr(human_simulator, "llm_client", None),
        )
        if llm_initial_intention is not None:
            current_intention = llm_initial_intention
        elif real_instruction and real_instruction.strip():
            current_intention = _fallback_initial_intention(_clean_initial_request(real_instruction))
        current_intention = _merge_selection_metadata_initial_intention(
            current_intention,
            (task.world_state or {}).get("webshop_selection_metadata"),
        )
        initial_gold_search_query = human_simulator.generate_gold_search_query_for_intention(
            {**current_intention, "gold_search_query": None}
        )
        if initial_gold_search_query:
            current_intention["gold_search_query"] = initial_gold_search_query

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
            "gold_delta": copy.deepcopy(gold_delta),
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
            gold_delta=gold_delta,
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

        style = style_schedule[turn_id]
        prefer_multi = turn_id in multi_preferred_turns
        use_candidate_pool = (
            sampling_config.distribution_controller is not None
            and (
                domain != "travelplanner"
                or sampling_config.distribution_controller.control_mode
                in {"selection", "hybrid"}
            )
        )
        shift = human_simulator.decide_shift(
            current_intention,
            env_feedback=env_feedback,
            intention_history=intention_history[:-1],
            current_gold_delta=gold_delta,
            candidate_samples=(
                sampling_config.multi_candidate_samples
                if prefer_multi or use_candidate_pool
                else 1
            ),
            max_candidate_samples=(
                sampling_config.max_multi_candidate_samples
                if prefer_multi
                else (
                    sampling_config.multi_candidate_samples
                    if use_candidate_pool
                    else 1
                )
            ),
            prefer_multi=prefer_multi,
            rng=rng,
            distribution_controller=sampling_config.distribution_controller,
        )
        new_intention, delta = human_simulator.apply_shift(current_intention, shift)
        user_utt = human_simulator.realize_shift(
            shift,
            current_intention,
            style,
            env_feedback=env_feedback,
            intention_history=intention_history[:-1],
            current_gold_delta=gold_delta,
        )
        if env.done and not delta and domain != "travelplanner":
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
                    "changes": [asdict(change) for change in shift.changes],
                    "candidate_sampling": copy.deepcopy(shift.sampling_metadata),
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
                    "changes": [asdict(change) for change in shift.changes],
                    "candidate_sampling": copy.deepcopy(shift.sampling_metadata),
                },
            }

        current_intention = new_intention
        user_utterance = user_utt
        gold_delta = delta
        linguistic_style = style
        action_implication = "continue"

        history.append({"role": "user", "content": user_utterance})
        intention_history.append(
            {
                "turn_id": turn_id + 1,
                "user_utterance": user_utterance,
                "gold_intention": copy.deepcopy(current_intention),
                "gold_delta": copy.deepcopy(gold_delta),
                "shift_condition": condition if intention_changed else "none",
                "change_category": change_category if intention_changed else "none",
                "shift_rationale": shift.rationale,
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
    domain: str = "webshop",
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
    reranker_config: Optional[RerankerConfig] = None,
) -> Tuple[Any, Any, HumanSimulator, Any]:
    llm_client = create_llm_client_from_env(azure_api_version=azure_api_version)
    if domain == "travelplanner":
        env = TravelPlannerEnvAdapter()
        agent = TravelPlannerExecutor(llm_client=llm_client)
        human = TravelPlannerUserSimulator(llm_client=llm_client)
        return env, agent, human, env

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
    if executor_type not in {"gold", "llm"}:
        raise ValueError("simulation/simulation/run_simulation.py only supports the gold BM25+reranking executor")
    agent = WebShopExecutor(llm_client=llm_client, reranker_config=reranker_config)
    human = WebShopUserSimulator(llm_client=llm_client)
    return env, agent, human, raw_env


def _simulate_task_with_retries(
    *,
    domain: str,
    task: BaseTask,
    env: Any,
    execution_agent: Any,
    human_simulator: HumanSimulator,
    max_turns: int,
    max_internal_steps: int,
    seed: int,
    shift_sampling_config: Optional[ShiftSamplingConfig] = None,
) -> Optional[DialogueInstance]:
    max_attempts = 1 + (TRAVELPLANNER_CASE_RETRIES if domain == "travelplanner" else 0)
    for attempt in range(1, max_attempts + 1):
        try:
            return simulate_dialogue_instance(
                task=task,
                env=env,
                execution_agent=execution_agent,
                human_simulator=human_simulator,
                max_turns=max_turns,
                max_internal_steps=max_internal_steps,
                seed=seed,
                shift_sampling_config=shift_sampling_config,
            )
        except Exception:
            if domain != "travelplanner":
                raise
            status = "retrying" if attempt < max_attempts else "skipping case"
            print(
                f"TravelPlanner case {task.instance_id!r} failed on attempt "
                f"{attempt}/{max_attempts}; {status}.",
                file=sys.stderr,
            )
            traceback.print_exc()
    return None


def _simulate_single_instance(
    *,
    domain: str,
    task: BaseTask,
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
    reranker_config: RerankerConfig,
    shift_sampling_config: Optional[ShiftSamplingConfig] = None,
) -> DialogueInstance:
    env, agent, human, raw_env = _build_runtime_components(
        domain=domain,
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
        reranker_config=reranker_config,
    )
    try:
        instance = _simulate_task_with_retries(
            domain=domain,
            task=task,
            env=env,
            execution_agent=agent,
            human_simulator=human,
            max_turns=max_turns,
            max_internal_steps=max_internal_steps,
            seed=seed,
            shift_sampling_config=shift_sampling_config,
        )
        if instance is None:
            raise RuntimeError(f"TravelPlanner case {task.instance_id!r} failed after all retries.")
        return instance
    finally:
        close_env = getattr(raw_env, "close", None)
        if callable(close_env):
            close_env()


def _simulate_instances_serial(
    *,
    domain: str,
    tasks: List[BaseTask],
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
    reranker_config: RerankerConfig,
    shift_sampling_config: ShiftSamplingConfig,
) -> List[DialogueInstance]:
    env, agent, human, raw_env = _build_runtime_components(
        domain=domain,
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
        reranker_config=reranker_config,
    )
    try:
        instances = []
        for task_index, task in enumerate(tasks, start=1):
            instance = _simulate_task_with_retries(
                domain=domain,
                task=task,
                env=env,
                execution_agent=agent,
                human_simulator=human,
                max_turns=max_turns,
                max_internal_steps=max_internal_steps,
                seed=seed + task_index - 1,
                shift_sampling_config=shift_sampling_config,
            )
            if instance is not None:
                instances.append(instance)
        return instances
    finally:
        close_env = getattr(raw_env, "close", None)
        if callable(close_env):
            close_env()


def _simulate_task_batch(
    *,
    domain: str,
    indexed_tasks: List[Tuple[int, BaseTask]],
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
    reranker_config: RerankerConfig,
    shift_sampling_config: ShiftSamplingConfig,
) -> Dict[int, DialogueInstance]:
    env, agent, human, raw_env = _build_runtime_components(
        domain=domain,
        azure_api_version=azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=executor_type,
        reranker_config=reranker_config,
    )
    try:
        instances_by_index = {}
        for task_index, task in indexed_tasks:
            instance = _simulate_task_with_retries(
                domain=domain,
                task=task,
                env=env,
                execution_agent=agent,
                human_simulator=human,
                max_turns=max_turns,
                max_internal_steps=max_internal_steps,
                seed=seed + task_index - 1,
                shift_sampling_config=shift_sampling_config,
            )
            if instance is not None:
                instances_by_index[task_index] = instance
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
    domain: str,
    tasks: List[BaseTask],
    seed: int,
    max_turns: int,
    max_internal_steps: int,
    azure_api_version: str,
    webshop_num_products: Optional[int],
    executor_type: str,
    parallelism: int,
    reranker_config: RerankerConfig,
    shift_sampling_config: ShiftSamplingConfig,
) -> List[DialogueInstance]:
    if parallelism <= 1:
        return _simulate_instances_serial(
            domain=domain,
            tasks=tasks,
            seed=seed,
            max_turns=max_turns,
            max_internal_steps=max_internal_steps,
            azure_api_version=azure_api_version,
            webshop_num_products=webshop_num_products,
            executor_type=executor_type,
            reranker_config=reranker_config,
            shift_sampling_config=shift_sampling_config,
        )

    effective_parallelism = min(parallelism, len(tasks))
    task_batches = _partition_indexed_tasks(tasks, effective_parallelism)
    instances_by_index: Dict[int, DialogueInstance] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_batch_index = {
            executor.submit(
                _simulate_task_batch,
                domain=domain,
                indexed_tasks=batch,
                seed=seed,
                max_turns=max_turns,
                max_internal_steps=max_internal_steps,
                azure_api_version=azure_api_version,
                webshop_num_products=webshop_num_products,
                executor_type=executor_type,
                reranker_config=reranker_config,
                shift_sampling_config=shift_sampling_config,
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
    parser.add_argument("--output", type=str, default=r".\IntentionChangeBench\data\simulation\simulated_dataset.json")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["webshop", "travelplanner"],
        default=os.getenv("BENCHMARK_DOMAIN", "webshop"),
        help="Benchmark domain to simulate.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument(
        "--max_internal_steps",
        type=int,
        default=None,
        help="Per-turn action budget. Defaults to 12 for WebShop and the original TravelPlanner value of 30.",
    )
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
    parser.add_argument("--num_instances", type=int, default=10)
    parser.add_argument(
        "--travelplanner_set_type",
        type=str,
        choices=["train", "validation", "test"],
        default=os.getenv("TRAVELPLANNER_SET_TYPE", "validation"),
        help="TravelPlanner split used for reference_information alignment when --domain travelplanner.",
    )
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
        default=2,
        help="Number of tasks to simulate concurrently. Use the same value as the number of selected tasks for one task per worker.",
    )
    parser.add_argument(
        "--multi_change_rate",
        type=float,
        default=0.30,
        help=(
            "Fraction of WebShop shift slots that prefer a naturally sampled multi-change candidate. "
            "This does not put a change count in the LLM prompt."
        ),
    )
    parser.add_argument(
        "--travelplanner_multi_change_rate",
        type=float,
        default=0.30,
        help=(
            "Fraction of TravelPlanner shift slots that softly prefer one coherent "
            "multi-intention update. The prompt never requires an exact change count."
        ),
    )
    parser.add_argument(
        "--multi_candidate_samples",
        type=int,
        default=4,
        help=(
            "Initial independent candidates for a multi-preferred WebShop turn, "
            "or for TravelPlanner when distribution mode is selection/hybrid."
        ),
    )
    parser.add_argument(
        "--max_multi_candidate_samples",
        type=int,
        default=12,
        help="Maximum candidates sampled when a multi-preferred turn has not yet produced a natural multi change.",
    )
    parser.add_argument(
        "--shift_distribution_baseline",
        type=str,
        default=os.getenv("SHIFT_DISTRIBUTION_BASELINE"),
        help=(
            "Optional baseline dataset whose category/condition counts initialize the "
            "domain-specific deficit controller. WebShop balances four change categories "
            "and two conditions; TravelPlanner also balances entity and agent_misunderstanding."
        ),
    )
    parser.add_argument(
        "--distribution_balance_strength",
        type=float,
        default=6.0,
        help="Strength of deficit-weighted candidate selection; 0 chooses uniformly from the eligible pool.",
    )
    parser.add_argument(
        "--distribution_control_mode",
        choices=["prompt", "selection", "hybrid"],
        default="prompt",
        help=(
            "How v1 deficits affect generation: prompt adds soft dynamic guidance, "
            "selection only reweights candidates, and hybrid does both."
        ),
    )
    parser.add_argument(
        "--azure_api_version",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    parser.add_argument(
        "--executor_type",
        type=str,
        choices=["gold", "llm"],
        default="gold",
        help=(
            "Gold trajectory executor. `gold` and legacy alias `llm` both use "
            "BM25 search plus optional LLM reranking over the exposed intention state."
        ),
    )
    parser.add_argument(
        "--enable_reranking",
        type=parse_bool,
        default=parse_bool(os.getenv("ENABLE_RERANKING", "true")),
        help="Whether the llm executor reranks WebShop candidates before returning them.",
    )
    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=int(os.getenv("RERANK_TOP_N", "30")),
        help="Number of raw WebShop candidates to pass to the LLM reranker.",
    )
    parser.add_argument(
        "--rerank_return_k",
        type=int,
        default=int(os.getenv("RERANK_RETURN_K", "10")),
        help="Number of reranked candidates to return to the human simulator.",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default=os.getenv("RERANKER_MODEL"),
        help="Optional reranker model/deployment label for logs and metadata.",
    )
    parser.add_argument(
        "--reranker_debug",
        type=parse_bool,
        default=parse_bool(os.getenv("RERANKER_DEBUG", "false")),
        help="Include compact reranker input and raw LLM output in rerank_info.",
    )
    args = parser.parse_args()
    if args.max_internal_steps is None:
        args.max_internal_steps = (
            DEFAULT_TRAVELPLANNER_MAX_INTERNAL_STEPS
            if args.domain == "travelplanner"
            else DEFAULT_MAX_INTERNAL_STEPS
        )
    print(f"Prompt log path: {get_prompt_log_path()}")

    if args.parallelism < 1:
        raise ValueError("--parallelism must be at least 1")
    if not 0.0 <= args.multi_change_rate <= 1.0:
        raise ValueError("--multi_change_rate must be between 0 and 1")
    if not 0.0 <= args.travelplanner_multi_change_rate <= 1.0:
        raise ValueError("--travelplanner_multi_change_rate must be between 0 and 1")
    if args.multi_candidate_samples < 1:
        raise ValueError("--multi_candidate_samples must be at least 1")
    if args.max_multi_candidate_samples < args.multi_candidate_samples:
        raise ValueError(
            "--max_multi_candidate_samples cannot be smaller than --multi_candidate_samples"
        )
    if args.distribution_balance_strength < 0:
        raise ValueError("--distribution_balance_strength cannot be negative")
    if args.rerank_top_n < 1:
        raise ValueError("--rerank_top_n must be at least 1")
    if args.rerank_return_k < 1:
        raise ValueError("--rerank_return_k must be at least 1")
    if args.rerank_return_k > args.rerank_top_n:
        raise ValueError("--rerank_return_k cannot exceed --rerank_top_n")
    webshop_num_products = parse_webshop_num_products(args.webshop_num_products)
    instance_ids = parse_instance_ids(args.instance_ids)
    goal_indices = parse_goal_indices(args.webshop_goal_indices)

    if args.domain == "travelplanner":
        if goal_indices is not None:
            raise ValueError("--webshop_goal_indices is only valid with --domain webshop")
        tasks = load_travelplanner_tasks(
            tasks_path=args.tasks_path,
            num_instances=args.num_instances,
            set_type=args.travelplanner_set_type,
            instance_ids=instance_ids,
        )
    else:
        tasks = load_webshop_tasks(
            tasks_path=args.tasks_path,
            num_instances=args.num_instances,
            goal_indices=goal_indices,
            instance_ids=instance_ids,
        )
    effective_parallelism = min(args.parallelism, len(tasks))
    logger = RuntimeLogger()
    reranker_config = RerankerConfig(
        enable_reranking=args.enable_reranking,
        rerank_top_n=args.rerank_top_n,
        rerank_return_k=args.rerank_return_k,
        reranker_model=args.reranker_model,
        reranker_debug=args.reranker_debug,
    )
    distribution_controller = None
    if args.shift_distribution_baseline:
        distribution_controller = _distribution_controller_from_baseline(
            args.shift_distribution_baseline,
            balance_strength=args.distribution_balance_strength,
            control_mode=args.distribution_control_mode,
            domain=args.domain,
        )
    shift_sampling_config = ShiftSamplingConfig(
        multi_change_rate=(
            args.multi_change_rate
            if args.domain == "webshop"
            else args.travelplanner_multi_change_rate
        ),
        multi_candidate_samples=(
            args.multi_candidate_samples
            if args.domain == "webshop"
            or (
                distribution_controller is not None
                and distribution_controller.control_mode in {"selection", "hybrid"}
            )
            else 1
        ),
        max_multi_candidate_samples=(
            args.max_multi_candidate_samples
            if args.domain == "webshop"
            or (
                distribution_controller is not None
                and distribution_controller.control_mode in {"selection", "hybrid"}
            )
            else 1
        ),
        distribution_controller=distribution_controller,
    )

    instances = _simulate_instances(
        domain=args.domain,
        tasks=tasks,
        seed=args.seed,
        max_turns=args.max_turns,
        max_internal_steps=args.max_internal_steps,
        azure_api_version=args.azure_api_version,
        webshop_num_products=webshop_num_products,
        executor_type=args.executor_type,
        parallelism=effective_parallelism,
        reranker_config=reranker_config,
        shift_sampling_config=shift_sampling_config,
    )
    for instance in instances:
        logger.log_instance(instance)

    logger.dump_json(args.output)
    print(
        f"Saved {len(logger.instances)} instances to {args.output} "
        f"(domain={args.domain}, parallelism={effective_parallelism}, webshop_num_products={args.webshop_num_products}, "
        f"executor_type={args.executor_type}, multi_change_rate={shift_sampling_config.multi_change_rate})"
    )
    return


if __name__ == "__main__":
    main()
