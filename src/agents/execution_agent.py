from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from models import AgentAction


class ExecutionAgent(ABC):
    @abstractmethod
    def act(
        self,
        history: List[Dict[str, Any]],
        user_utterance: str,
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        raise NotImplementedError
