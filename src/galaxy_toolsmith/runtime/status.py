from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_status_log_path(raw_path: str) -> Path | None:
    value = str(raw_path).strip()
    if not value:
        return None
    return Path(value).resolve()


def emit_status(payload: dict[str, Any], *, status_log_path: Path | None = None) -> None:
    line = json.dumps(payload)
    print(line, flush=True)
    if status_log_path is None:
        return
    status_log_path.parent.mkdir(parents=True, exist_ok=True)
    with status_log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
