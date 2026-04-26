from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_LOG_LOCK = threading.Lock()


def _default_log_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "prompt_log.jsonl"


def get_prompt_log_path() -> Path:
    configured = os.getenv("PROMPT_LOG_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _default_log_path()


def log_prompt(
    source: str,
    prompt: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    enabled = os.getenv("PROMPT_LOG_ENABLED", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    if not isinstance(prompt, str):
        prompt = str(prompt)

    log_path = get_prompt_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(source or "unknown"),
        "prompt": prompt,
    }
    if metadata:
        record["metadata"] = metadata

    header = f"\n=== PROMPT LOG | {record['source']} | {record['timestamp']} ===\n"

    with _LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
        sys.stdout.write(header)
        sys.stdout.write(prompt)
        if not prompt.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
