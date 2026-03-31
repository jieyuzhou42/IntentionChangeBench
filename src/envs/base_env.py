from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from models import EnvFeedback


class BaseEnv(ABC):
    @abstractmethod
    def reset(self, task) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(self, agent_action, user_state) -> EnvFeedback:
        raise NotImplementedError