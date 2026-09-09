from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence

from domains.travelplanner.environment import TravelPlannerEnvAdapter
from domains.travelplanner.executor import TravelPlannerExecutor
from models import AgentAction, BaseTask, EnvFeedback


PUBLIC_TRAVEL_OBSERVATION_FIELDS = {
    "domain",
    "page_type",
    "feedback_type",
    "latest_user_utterance",
    "available_actions",
    "tool_name",
    "tool_argument",
    "tool_result",
    "pending_notebook",
    "notebook",
    "notebook_size",
    "completed_actions",
    "submitted_plan",
}


def public_travel_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in (observation or {}).items()
        if key in PUBLIC_TRAVEL_OBSERVATION_FIELDS
    }


def public_travel_feedback(feedback: Optional[EnvFeedback]) -> Optional[Dict[str, Any]]:
    if feedback is None:
        return None
    return {
        "status": feedback.status,
        "reason": feedback.reason,
        "observation": public_travel_observation(feedback.observation or {}),
        "result": copy.deepcopy(feedback.result or {}),
    }


def user_utterance_history(history: Sequence[Dict[str, Any]]) -> List[str]:
    utterances: List[str] = []
    for entry in history:
        if not isinstance(entry, dict) or str(entry.get("role") or "").lower() != "user":
            continue
        content = entry.get("content")
        if isinstance(content, dict):
            continue
        text = str(content or "").strip()
        if text:
            utterances.append(text)
    return utterances


class SingleAgentTravelPlannerExecutor(TravelPlannerExecutor):
    """One evaluated agent emits both its intention understanding and tool action."""

    def act(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        prompt = self._build_single_agent_prompt(history, env_observation)
        raw_action = self.llm_client.generate_json(prompt)
        action = self._action_from_llm(raw_action)
        if action is None:
            raise ValueError(f"TravelPlanner agent returned an invalid action: {raw_action!r}")
        if not isinstance(action.predicted_current_intention, dict):
            raise ValueError("TravelPlanner agent omitted predicted_current_intention")
        if env_observation.get("pending_notebook") and action.action_type != "NotebookWrite":
            raise ValueError("NotebookWrite is required immediately after a search tool result")
        if action.action_type == "Planner" and not self._payload_has_plan(action.action_payload or {}):
            raise ValueError("Planner action must include the agent-generated plan")
        return self._with_original_argument(action)

    def _build_single_agent_prompt(
        self,
        history: List[Dict[str, Any]],
        observation: Dict[str, Any],
    ) -> str:
        context = {
            "user_utterances": user_utterance_history(history),
            "public_environment_observation": public_travel_observation(observation),
        }
        return f"""
You are the single evaluated TravelPlanner agent.
Infer the user's current cumulative intention and choose exactly one next environment action.
Return one JSON object only.

Required schema:
{{
  "predicted_current_intention": {{
    "constraints": {{"semantic_field": "active value"}},
    "priority": ["most important field", "next field"],
    "explanation": "brief interpretation"
  }},
  "action_type": "FlightSearch | AttractionSearch | AccommodationSearch | RestaurantSearch | CitySearch | GoogleDistanceMatrix | NotebookWrite | Planner",
  "action_payload": {{}},
  "rationale": "brief reason"
}}

Action payloads:
- FlightSearch: {{"origin": "city", "destination": "city", "date": "YYYY-MM-DD"}}
- AttractionSearch: {{"city": "city"}}
- AccommodationSearch: {{"city": "city"}}
- RestaurantSearch: {{"city": "city"}}
- CitySearch: {{"state": "state name"}}
- GoogleDistanceMatrix: {{"origin": "city", "destination": "city", "mode": "self-driving or taxi"}}
- NotebookWrite: {{"description": "short description of the preceding tool result"}}
- Planner: {{"query": "current request", "plan": {{"itinerary": [...]}}}}

Rules:
- Use only user_utterances and public_environment_observation.
- Later utterances may add, replace, remove, relax, or reprioritize earlier requirements.
- Include every active constraint and rank each active field once.
- Use real search tools, write their results to Notebook, then submit a grounded plan.
- If pending_notebook is true, the next action must be NotebookWrite.
- A Planner action must include the complete itinerary in action_payload.plan.
- Never invent tool results.

CONTEXT:
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}
""".strip()


def travel_task_from_annotated_instance(instance: Dict[str, Any]) -> BaseTask:
    source_world = instance.get("world_state") or {}
    return BaseTask(
        instance_id=str(instance.get("instance_id")),
        task_type=str(instance.get("task_type") or "planning"),
        subtype=str(instance.get("subtype") or "travel"),
        world_state={
            "domain": "travelplanner",
            # Private environment database; exposed only via explicit search tools.
            "reference_information": copy.deepcopy(source_world.get("reference_information")),
        },
        initial_intention={"constraints": {}, "priority": [], "domain": "travelplanner"},
    )


def run_single_agent_travel_instance(
    *,
    instance: Dict[str, Any],
    client: Any,
    max_internal_steps: int = 30,
) -> List[Dict[str, Any]]:
    task = travel_task_from_annotated_instance(instance)
    env = TravelPlannerEnvAdapter()
    agent = SingleAgentTravelPlannerExecutor(llm_client=client)
    env.reset(task)
    user_history: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for turn_index, source_turn in enumerate(instance.get("turns") or []):
        utterance = str(source_turn.get("user_utterance") or "").strip()
        user_history.append({"role": "user", "content": utterance})
        env.prepare_turn({}, utterance, {})
        observation = public_travel_observation(env.get_observation())
        rollout_trace: List[Dict[str, Any]] = []
        final_action: Optional[AgentAction] = None
        final_feedback: Optional[EnvFeedback] = None
        stop_reason = "step_budget"

        for step_index in range(1, max_internal_steps + 1):
            action = agent.act(user_history, utterance, observation)
            prediction = copy.deepcopy(action.predicted_current_intention or {})
            feedback = env.step(action, prediction)
            public_feedback = public_travel_feedback(feedback)
            rollout_trace.append(
                {
                    "step_index": step_index,
                    "action": {
                        "action_type": action.action_type,
                        "action_payload": copy.deepcopy(action.action_payload or {}),
                        "rationale": action.rationale,
                        "predicted_current_intention": prediction,
                    },
                    "env_feedback": public_feedback,
                }
            )
            final_action = action
            final_feedback = feedback
            observation = public_travel_observation(env.get_observation())
            if feedback.status == "error":
                stop_reason = "error"
                break
            if action.action_type == "Planner":
                stop_reason = "planner_submitted"
                break

        prediction = copy.deepcopy(
            final_action.predicted_current_intention
            if final_action is not None
            else {}
        )
        action_dict = (
            {
                "action_type": final_action.action_type,
                "action_payload": copy.deepcopy(final_action.action_payload or {}),
                "rationale": final_action.rationale,
            }
            if final_action is not None
            else None
        )
        submitted_plan = (
            (final_feedback.observation or {}).get("submitted_plan")
            if final_feedback is not None
            else None
        )
        gold_intention = copy.deepcopy(source_turn.get("gold_current_intention") or {})
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
                "user_utterance": utterance,
                "gold_intention": gold_intention,
                "agent_intention_prediction": prediction,
                "agent_action": action_dict,
                "env_feedback": public_travel_feedback(final_feedback),
                "rollout_trace": rollout_trace,
                "num_internal_steps": len(rollout_trace),
                "stop_reason": stop_reason,
                "action_evidence": {
                    "final_action": action_dict,
                    "submitted_plan": copy.deepcopy(submitted_plan),
                    "gold_deterministic_plan_eval": gold_plan_eval,
                    "rollout_trace": rollout_trace,
                },
            }
        )
    return rows


__all__ = [
    "SingleAgentTravelPlannerExecutor",
    "public_travel_observation",
    "run_single_agent_travel_instance",
    "travel_task_from_annotated_instance",
    "user_utterance_history",
]
