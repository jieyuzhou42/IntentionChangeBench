from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ShiftCondition:
    type: str
    reason: str
    source: str = "environment"  # environment / result_inspection / stochastic
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShiftOp:
    op: str  # add / relax / override / reprioritize / scope_correction / none
    field: Optional[str] = None
    value: Any = None
    old_value: Any = None
    rationale: str = ""
    priority_update: Optional[List[str]] = None
    utterance_plan: Optional[Dict[str, Any]] = None


@dataclass
class AgentAction:
    action_type: str
    action_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvFeedback:
    status: str  # adapter-level observation status, e.g. observed / error
    feasible: bool  # adapter hint, not the benchmark's final shift decision
    reason: Optional[str] = None
    observation: Dict[str, Any] = field(default_factory=dict)  # raw_text, actions, visible_items, item_context, etc.
    result: Dict[str, Any] = field(default_factory=dict)
    satisfied_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)


@dataclass
class TriggerEvidence:
    trigger_type: str  # infeasible / partial_mismatch / preference_mismatch / none
    source: str        # environment / result_inspection / none
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRecord:
    turn_id: int
    user_utterance: Optional[str]
    agent_action: Optional[Dict[str, Any]]
    env_feedback: Optional[Dict[str, Any]]
    trigger_evidence: Optional[Dict[str, Any]]
    shift_condition: Optional[Dict[str, Any]]
    gold_delta: Dict[str, Dict[str, Any]]
    gold_current_intention: Dict[str, Any]
    linguistic_style: str
    action_implication: str
    num_internal_steps: int = 0
    stop_reason: Optional[str] = None
    rollout_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BaseTask:
    instance_id: str
    task_type: str
    subtype: str
    world_state: Dict[str, Any]
    initial_intention: Dict[str, Any]


@dataclass
class DialogueInstance:
    instance_id: str
    task_type: str
    subtype: str
    world_state: Dict[str, Any]
    turns: List[TurnRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "task_type": self.task_type,
            "subtype": self.subtype,
            "world_state": self.world_state,
            "turns": [asdict(t) for t in self.turns],
        }
