from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class ShiftCondition:
    type: str
    reason: str
    explicitly_mentioned: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShiftOp:
    turn: int
    op: str
    field: str
    value: Any
    style: str
    condition: ShiftCondition

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["condition"] = self.condition.to_dict()
        return data


@dataclass
class TurnRecord:
    turn_id: int
    user_utterance: str
    shift_condition: Dict[str, Any]
    gold_delta: Dict[str, Dict[str, Any]]
    gold_current_intention: Dict[str, Any]
    linguistic_style: str
    action_implication: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaseTask:
    instance_id: str
    task_type: str
    subtype: str
    world_state: Dict[str, Any]
    initial_intention: Dict[str, Any]
    shift_program: List[ShiftOp] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["shift_program"] = [op.to_dict() for op in self.shift_program]
        return data


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
            "turns": [t.to_dict() for t in self.turns],
        }
