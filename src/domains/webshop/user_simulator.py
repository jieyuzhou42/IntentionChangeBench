from __future__ import annotations

from typing import Any, Dict, Optional

from models import EnvFeedback
from simulation.simulation.base_user_simulator import HumanSimulator


class WebShopUserSimulator(HumanSimulator):
    """WebShop-only user simulator preserving the established WebShop path."""

    def _infer_domain(
        self,
        current_intention: Dict[str, Any],
        env_feedback: Optional[EnvFeedback] = None,
    ) -> str:
        return "webshop"


__all__ = ["WebShopUserSimulator"]
