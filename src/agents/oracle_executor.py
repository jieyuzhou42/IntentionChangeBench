from __future__ import annotations

from typing import Any, Dict, List

from agents.execution_agent import ExecutionAgent
from models import AgentAction


class OracleWebShopExecutor(ExecutionAgent):
    def act(
        self,
        history: List[Dict[str, Any]],
        current_intention: Dict[str, Any],
        env_observation: Dict[str, Any],
    ) -> AgentAction:
        page_type = env_observation.get("page_type", "search")
        constraints = current_intention.get("constraints", {})

        if page_type in {"search", "unknown"}:
            query_parts = []
            category = constraints.get("category")
            color = constraints.get("color")
            brand = constraints.get("brand")

            if color:
                query_parts.append(str(color))
            if brand:
                query_parts.append(str(brand))
            if category:
                query_parts.append(str(category))

            query = " ".join(query_parts).strip() or current_intention.get("request", "product")
            return AgentAction("search", {"query": query})

        if page_type == "results":
            visible = env_observation.get("visible_items", [])
            if visible:
                target = (
                    visible[0].get("click_target")
                    or visible[0].get("asin")
                    or visible[0].get("title", "item_0")
                )
                return AgentAction("click", {"target": target})
            query = current_intention.get("request", "product")
            return AgentAction("refine", {"query": query})

        if page_type == "item":
            return AgentAction("buy", {})

        return AgentAction("search", {"query": current_intention.get("request", "product")})
