from __future__ import annotations

import random
from typing import Any, Dict, List

from agents.execution_agent import ExecutionAgent
from models import AgentAction


class NoisyWebShopExecutor(ExecutionAgent):
    def __init__(self, seed: int = 7, mistake_rate: float = 0.3):
        self.rng = random.Random(seed)
        self.mistake_rate = mistake_rate

    def act(
        self,
        history: List[Dict[str, Any]],
        current_intention: Dict[str, Any],
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        page_type = env_observation.get("page_type", "search")
        constraints = current_intention.get("constraints", {})

        if page_type in {"search", "unknown"}:
            category = constraints.get("category", "product")
            color = constraints.get("color")
            query_parts = []
            if self.rng.random() > self.mistake_rate and color:
                query_parts.append(str(color))
            query_parts.append(str(category))
            return AgentAction("search", {"query": " ".join(query_parts)})

        if page_type == "results":
            visible = env_observation.get("visible_items", [])
            if visible:
                idx = 0 if self.rng.random() > self.mistake_rate else min(1, len(visible) - 1)
                target = (
                    visible[idx].get("click_target")
                    or visible[idx].get("asin")
                    or visible[idx].get("title", "item")
                )
                return AgentAction("click", {"target": target})
            return AgentAction("refine", {"query": current_intention.get("request", "product")})

        if page_type == "item":
            if self.rng.random() < self.mistake_rate:
                return AgentAction("search", {"query": current_intention.get("request", "product")})
            return AgentAction("buy", {})

        return AgentAction("search", {"query": current_intention.get("request", "product")})
