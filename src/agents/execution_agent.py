from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from models import AgentAction


class ExecutionAgent(ABC):
    @abstractmethod
    def act(
        self,
        history: List[Dict[str, Any]],
        current_intention: Dict[str, Any],
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        raise NotImplementedError