from __future__ import annotations

import argparse
import copy
import random
from typing import Any, Dict, List

from agents.oracle_executor import OracleWebShopExecutor
from evaluators.runtime_logger import RuntimeLogger
from models import BaseTask, DialogueInstance, TurnRecord
from simulators.human_simulator import HumanSimulator
from simulators.trigger_detector import detect_trigger
import gym
from web_agent_site.envs import WebAgentTextEnv
from envs.webshop_env import WebShopEnvAdapter

STYLE_POOL = ["explicit", "partial", "elliptical"]


def make_demo_webshop_task() -> BaseTask:
    return BaseTask(
        instance_id="webshop_demo_001",
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


def simulate_dialogue_instance(
    task: BaseTask,
    env: WebShopEnvAdapter,
    execution_agent,
    human_simulator: HumanSimulator,
    max_turns: int = 4,
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
        agent_action = execution_agent.act(history, current_intention, env_obs)
        env_feedback = env.step(agent_action, current_intention)

        trigger, evidence = detect_trigger(env_feedback, current_intention)
        style = rng.choice(STYLE_POOL)

        if trigger is not None:
            shift = human_simulator.decide_shift(trigger, current_intention)
            new_intention, delta = human_simulator.apply_shift(current_intention, shift)
            user_utt = human_simulator.realize_shift(shift, current_intention, style)
            shift_condition = {
                "type": trigger.type,
                "reason": trigger.reason,
                "source": trigger.source,
                "details": trigger.details,
            }
            current_intention = new_intention
            action_implication = "requery" if shift.op in {"relax", "override"} else "continue"
        else:
            delta = {}
            user_utt = "Looks good, continue."
            shift_condition = None
            action_implication = "continue"

        turns.append(
            TurnRecord(
                turn_id=turn_id,
                user_utterance=user_utt,
                agent_action={
                    "action_type": agent_action.action_type,
                    "action_payload": agent_action.action_payload,
                },
                env_feedback={
                    "status": env_feedback.status,
                    "feasible": env_feedback.feasible,
                    "reason": env_feedback.reason,
                    "observation": env_feedback.observation,
                    "result": env_feedback.result,
                    "satisfied_constraints": env_feedback.satisfied_constraints,
                    "violated_constraints": env_feedback.violated_constraints,
                },
                trigger_evidence={
                    "trigger_type": evidence.trigger_type,
                    "source": evidence.source,
                    "details": evidence.details,
                },
                shift_condition=shift_condition,
                gold_delta=delta,
                gold_current_intention=copy.deepcopy(current_intention),
                linguistic_style=style,
                action_implication=action_implication,
            )
        )

        history.append(
            {
                "role": "assistant",
                "content": {
                    "action_type": agent_action.action_type,
                    "action_payload": agent_action.action_payload,
                    "env_result": env_feedback.result,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../data/webshop_simulated_dataset.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_turns", type=int, default=4)
    args = parser.parse_args()

    task = make_demo_webshop_task()
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
    agent = OracleWebShopExecutor()
    human = HumanSimulator(seed=args.seed)
    logger = RuntimeLogger()

    instance = simulate_dialogue_instance(
        task=task,
        env=env,
        execution_agent=agent,
        human_simulator=human,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    logger.log_instance(instance)
    logger.dump_json(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
