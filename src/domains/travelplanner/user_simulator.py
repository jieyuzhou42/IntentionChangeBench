from __future__ import annotations

from typing import Any, Dict, Optional

from models import EnvFeedback
from simulation.simulation.base_user_simulator import HumanSimulator


class TravelPlannerUserSimulator(HumanSimulator):
    """TravelPlanner-only user simulator with fail-fast LLM behavior."""

    def _infer_domain(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> str:
        return "travelplanner"


__all__ = ["TravelPlannerUserSimulator"]
