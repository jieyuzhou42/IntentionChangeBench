from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domains.webshop.environment import WebShopEnvAdapter
from simulation.simulation.run_simulation import _task_from_payload


def test_selected_instruction_is_promoted_to_fixed_world_state():
    task = _task_from_payload(
        {
            "instance_id": "webshop_goal_00042",
            "world_state": {"domain": "webshop", "webshop_goal_index": 42},
            "turns": [{"turn_id": 0, "user_utterance": "fallback instruction"}],
            "selection_metadata": {
                "instruction": "fixed selected instruction, and price lower than 80.00 dollars",
                "price_upper": 80.0,
            },
        },
        fallback_index=1,
    )

    assert task.initial_intention["request"] == "fallback instruction"
    assert task.world_state["webshop_instruction_text"].startswith("fixed selected instruction")
    assert task.world_state["webshop_fixed_price_upper"] == 80.0


def test_fixed_price_upper_is_applied_to_webshop_reward_goal():
    raw_env = SimpleNamespace(
        session="42",
        server=SimpleNamespace(
            user_sessions={"42": {"goal": {"price_upper": 30.0}}}
        ),
    )
    adapter = WebShopEnvAdapter(raw_env)
    task = SimpleNamespace(world_state={"webshop_fixed_price_upper": 80.0})

    adapter._apply_fixed_goal_metadata(task)

    assert raw_env.server.user_sessions["42"]["goal"]["price_upper"] == 80.0
