from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import AgentAction, BaseTask, EnvFeedback, ShiftOp
from run_simulation import execute_turn, simulate_dialogue_instance


def _make_intention() -> dict:
    return {
        "request": "Find a green stripe shirt in large.",
        "constraints": {
            "category": "shirt",
            "color": "green stripe",
            "color_exact": "green stripe",
            "size": "large",
            "size_exact": "large",
            "budget_max": None,
            "brand": None,
            "brand_exact": None,
        },
        "priority": ["category", "color", "size"],
    }


class ScriptedExecutionAgent:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def act(self, history, current_intention, env_observation):
        self.calls.append(
            {
                "history": copy.deepcopy(history),
                "env_observation": copy.deepcopy(env_observation),
            }
        )
        if not self.actions:
            raise AssertionError("Execution agent was asked for more actions than expected.")
        return self.actions.pop(0)


class RecordingHumanSimulator:
    def __init__(self):
        self.decide_calls = []
        self.realize_calls = []

    def decide_shift(self, user_state, agent_action=None, env_feedback=None, history=None):
        self.decide_calls.append(
            {
                "user_state": copy.deepcopy(user_state),
                "agent_action": agent_action,
                "env_feedback": env_feedback,
                "history": copy.deepcopy(history),
            }
        )
        return ShiftOp(op="none", rationale="keep current intention")

    def apply_shift(self, current_intention, shift):
        return copy.deepcopy(current_intention), {}

    def realize_shift(
        self,
        shift,
        current_intention,
        style,
        agent_action=None,
        env_feedback=None,
        history=None,
    ):
        self.realize_calls.append(
            {
                "shift": shift,
                "agent_action": agent_action,
                "env_feedback": env_feedback,
            }
        )
        return "Keep going."


class AddConstraintHumanSimulator(RecordingHumanSimulator):
    def decide_shift(self, user_state, agent_action=None, env_feedback=None, history=None):
        super().decide_shift(
            user_state,
            agent_action=agent_action,
            env_feedback=env_feedback,
            history=history,
        )
        return ShiftOp(
            op="add",
            field="brand",
            old_value=None,
            value="Field & Pine",
            rationale="user adds a brand preference after seeing the item",
        )

    def apply_shift(self, current_intention, shift):
        next_state = copy.deepcopy(current_intention)
        next_state["constraints"]["brand"] = shift.value
        return next_state, {
            "brand": {
                "op": "add",
                "old": None,
                "new": shift.value,
                "rationale": shift.rationale,
            }
        }


class MockRolloutEnv:
    def __init__(self, initial_observation=None):
        self.done = False
        self.step_calls = []
        self._initial_observation = copy.deepcopy(initial_observation or self._search_observation())
        self.observation = copy.deepcopy(self._initial_observation)

    def reset(self, task=None):
        self.done = False
        self.step_calls.clear()
        self.observation = copy.deepcopy(self._initial_observation)
        return copy.deepcopy(self.observation)

    def get_instruction_text(self):
        return ""

    def parse_instruction_to_intention(self, instruction):
        return None

    def get_observation(self):
        return copy.deepcopy(self.observation)

    def summarize_current_state(self, user_state):
        return self._make_feedback(self.observation, user_state, executed_action=None)

    def step(self, agent_action, user_state):
        self.step_calls.append(
            {
                "action_type": agent_action.action_type,
                "action_payload": dict(agent_action.action_payload or {}),
            }
        )

        if agent_action.action_type == "search":
            self.observation = self._results_observation()
            executed_action = f"search[{agent_action.action_payload.get('query', '')}]"
        elif agent_action.action_type == "click" and agent_action.action_payload.get("target") == "B000GREEN1":
            self.observation = self._item_observation()
            executed_action = "click[B000GREEN1]"
        elif agent_action.action_type == "click" and agent_action.action_payload.get("target") == "green stripe":
            self.observation = self._item_observation({"color": "green stripe"})
            executed_action = "click[green stripe]"
        elif agent_action.action_type == "click" and agent_action.action_payload.get("target") == "large":
            self.observation = self._item_observation({"color": "green stripe", "size": "large"})
            executed_action = "click[large]"
        else:
            raise AssertionError(f"Unexpected action: {agent_action}")

        return self._make_feedback(self.observation, user_state, executed_action=executed_action)

    def _search_observation(self):
        return {
            "page_type": "search",
            "instruction": "Find a green stripe shirt in large.",
            "raw_text": "Instruction: Find a green stripe shirt in large. Search",
            "clickables": [],
            "visible_items": [],
            "selected_item": None,
            "selected_asin": None,
            "selected_options": {},
            "item_context": None,
        }

    def _results_observation(self):
        return {
            "page_type": "results",
            "instruction": "Find a green stripe shirt in large.",
            "raw_text": "Results[SEP]B000GREEN1[SEP]Trail Shirt[SEP]$29.99",
            "clickables": ["B000GREEN1"],
            "visible_items": [
                {
                    "asin": "B000GREEN1",
                    "title": "Trail Shirt",
                    "price": 29.99,
                    "click_target": "B000GREEN1",
                }
            ],
            "selected_item": None,
            "selected_asin": None,
            "selected_options": {},
            "item_context": None,
        }

    def _item_observation(self, selected_options=None):
        selected_options = dict(selected_options or {})
        item_context = {
            "asin": "B000GREEN1",
            "title": "Trail Shirt",
            "price": 29.99,
            "pricing": [29.99],
            "category": "shirt",
            "product_category": "shirt",
            "brand": "Field & Pine",
            "color": "red",
            "options": {
                "color": ["red", "green stripe"],
                "size": ["medium", "large"],
            },
            "selected_options": selected_options,
        }
        return {
            "page_type": "item",
            "instruction": "Find a green stripe shirt in large.",
            "raw_text": "Item Page[SEP]Trail Shirt[SEP]Price: $29.99[SEP]Buy Now",
            "clickables": ["green stripe", "large", "Buy Now"],
            "visible_items": [],
            "selected_item": {
                "asin": "B000GREEN1",
                "title": "Trail Shirt",
                "price": 29.99,
                "category": "shirt",
            },
            "selected_asin": "B000GREEN1",
            "selected_options": selected_options,
            "item_context": item_context,
        }

    def _make_feedback(self, observation, user_state, executed_action):
        result = {}
        if observation.get("page_type") == "results" and observation.get("visible_items"):
            result = dict(observation["visible_items"][0])
            result["category"] = "shirt"
        elif observation.get("page_type") == "item":
            item_context = observation.get("item_context") or {}
            result = {
                "asin": item_context.get("asin"),
                "title": item_context.get("title"),
                "price": item_context.get("price"),
                "category": item_context.get("category"),
                "brand": item_context.get("brand"),
                "base_color": item_context.get("color"),
                "color": item_context.get("color"),
                "selected_options": copy.deepcopy(observation.get("selected_options") or {}),
            }
            if "color" in result["selected_options"]:
                result["selected_color"] = result["selected_options"]["color"]
                result["color"] = result["selected_options"]["color"]
            if "size" in result["selected_options"]:
                result["selected_size"] = result["selected_options"]["size"]
                result["size"] = result["selected_options"]["size"]

        satisfied = []
        violated = []
        constraint_debug = {}
        constraints = user_state.get("constraints", {}) or {}
        for field in ("category", "color", "size"):
            desired = constraints.get(f"{field}_exact") or constraints.get(field)
            if desired is None:
                continue
            actual = None
            if field == "category":
                actual = result.get("category")
            elif field == "color":
                actual = (result.get("selected_options") or {}).get("color") or result.get("color")
            elif field == "size":
                actual = (result.get("selected_options") or {}).get("size") or result.get("size")
            matched = None if actual is None else str(actual).lower() == str(desired).lower()
            constraint_debug[field] = {
                "desired": desired,
                "actual": actual,
                "matched": matched,
            }
            if matched is True:
                satisfied.append(field)
            elif matched is False:
                violated.append(field)

        observation_payload = copy.deepcopy(observation)
        observation_payload.update(
            {
                "executed_action": executed_action,
                "reward": None,
                "constraint_debug": constraint_debug,
                "selection_changed": False,
                "extracted_result": copy.deepcopy(result),
            }
        )
        return EnvFeedback(
            status="observed",
            feasible=True,
            reason=None,
            observation=observation_payload,
            result=result,
            satisfied_constraints=satisfied,
            violated_constraints=violated,
        )


class StuckEnv(MockRolloutEnv):
    def step(self, agent_action, user_state):
        self.step_calls.append(
            {
                "action_type": agent_action.action_type,
                "action_payload": dict(agent_action.action_payload or {}),
            }
        )
        return self._make_feedback(self.observation, user_state, executed_action="search[still stuck]")


def test_execute_turn_rolls_search_item_and_option_selection_into_one_benchmark_turn():
    env = MockRolloutEnv()
    history = [{"role": "user", "content": "Find a green stripe shirt in large."}]
    agent = ScriptedExecutionAgent(
        [
            AgentAction("search", {"query": "green stripe shirt large"}),
            AgentAction("click", {"target": "B000GREEN1"}),
            AgentAction("click", {"target": "green stripe"}),
            AgentAction("click", {"target": "large"}),
        ]
    )

    rollout = execute_turn(
        env=env,
        execution_agent=agent,
        history=history,
        current_intention=_make_intention(),
        env_observation=env.reset(),
        max_internal_steps=6,
    )

    assert rollout.num_internal_steps == 4
    assert rollout.stop_reason == "requested_options_satisfied"
    assert [call["action_type"] for call in env.step_calls] == ["search", "click", "click", "click"]
    assert [step["page_type"] for step in rollout.rollout_trace] == ["results", "item", "item", "item"]
    assert rollout.final_env_feedback.observation["selected_asin"] == "B000GREEN1"
    assert rollout.final_env_feedback.observation["selected_options"] == {
        "color": "green stripe",
        "size": "large",
    }
    assert rollout.final_env_feedback.satisfied_constraints == ["category", "color", "size"]


def test_simulator_only_sees_the_final_state_after_rollout():
    env = MockRolloutEnv()
    agent = ScriptedExecutionAgent(
        [
            AgentAction("search", {"query": "green stripe shirt large"}),
            AgentAction("click", {"target": "B000GREEN1"}),
            AgentAction("click", {"target": "green stripe"}),
            AgentAction("click", {"target": "large"}),
        ]
    )
    human = RecordingHumanSimulator()
    task = BaseTask(
        instance_id="webshop_rollout_test",
        task_type="transaction",
        subtype="shopping",
        world_state={"domain": "webshop"},
        initial_intention=_make_intention(),
    )

    instance = simulate_dialogue_instance(
        task=task,
        env=env,
        execution_agent=agent,
        human_simulator=human,
        max_turns=1,
        max_internal_steps=6,
        seed=7,
    )

    assert len(human.decide_calls) == 1
    decide_call = human.decide_calls[0]
    assert decide_call["agent_action"].action_payload == {"target": "large"}
    assert decide_call["env_feedback"].observation["selected_options"] == {
        "color": "green stripe",
        "size": "large",
    }

    turn = instance.turns[1]
    assert turn.num_internal_steps == 4
    assert turn.stop_reason == "requested_options_satisfied"
    assert len(turn.rollout_trace) == 4
    assert turn.env_feedback["observation"]["selected_options"] == {
        "color": "green stripe",
        "size": "large",
    }


def test_execute_turn_stops_early_when_requested_options_are_already_satisfied():
    env = MockRolloutEnv(
        initial_observation=MockRolloutEnv()._item_observation(
            {"color": "green stripe", "size": "large"}
        )
    )
    agent = ScriptedExecutionAgent([AgentAction("buy", {})])

    rollout = execute_turn(
        env=env,
        execution_agent=agent,
        history=[{"role": "user", "content": "Find a green stripe shirt in large."}],
        current_intention=_make_intention(),
        env_observation=env.reset(),
        max_internal_steps=6,
    )

    assert rollout.num_internal_steps == 0
    assert rollout.stop_reason == "requested_options_satisfied"
    assert rollout.final_action is None
    assert agent.calls == []


def test_execute_turn_stops_when_no_progress_repeats():
    env = StuckEnv()
    agent = ScriptedExecutionAgent(
        [
            AgentAction("search", {"query": "green stripe shirt"}),
            AgentAction("search", {"query": "green stripe shirt"}),
        ]
    )

    rollout = execute_turn(
        env=env,
        execution_agent=agent,
        history=[{"role": "user", "content": "Find a green stripe shirt in large."}],
        current_intention=_make_intention(),
        env_observation=env.reset(),
        max_internal_steps=6,
    )

    assert rollout.stop_reason == "stuck"
    assert rollout.num_internal_steps == 2
    assert [step["state_changed"] for step in rollout.rollout_trace] == [False, False]
    assert [step["made_progress"] for step in rollout.rollout_trace] == [False, False]


def test_simulate_dialogue_marks_add_constraint_shift_as_requery():
    env = MockRolloutEnv()
    agent = ScriptedExecutionAgent(
        [
            AgentAction("search", {"query": "green stripe shirt large"}),
            AgentAction("click", {"target": "B000GREEN1"}),
            AgentAction("click", {"target": "green stripe"}),
            AgentAction("click", {"target": "large"}),
        ]
    )
    human = AddConstraintHumanSimulator()
    task = BaseTask(
        instance_id="webshop_add_constraint_test",
        task_type="transaction",
        subtype="shopping",
        world_state={"domain": "webshop"},
        initial_intention=_make_intention(),
    )

    instance = simulate_dialogue_instance(
        task=task,
        env=env,
        execution_agent=agent,
        human_simulator=human,
        max_turns=1,
        max_internal_steps=6,
        seed=7,
    )

    turn = instance.turns[1]
    assert turn.action_implication == "requery"
    assert turn.shift_condition["details"]["op"] == "add"
    assert turn.gold_delta["brand"]["op"] == "add"
    assert turn.gold_current_intention["constraints"]["brand"] == "Field & Pine"
