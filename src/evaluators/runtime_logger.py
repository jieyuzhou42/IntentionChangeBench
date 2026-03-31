from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from models import DialogueInstance


class RuntimeLogger:
    def __init__(self):
        self.instances: List[Dict] = []

    def log_instance(self, instance: DialogueInstance):
        self.instances.append(instance.to_dict())

    def dump_json(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.instances, f, ensure_ascii=False, indent=2)