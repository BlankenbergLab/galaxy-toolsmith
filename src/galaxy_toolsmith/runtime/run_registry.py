from __future__ import annotations

import json
import os
import tempfile
import threading
import traceback
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.runtime.environment import collect_environment_snapshot

RUN_STATUSES = {"running", "completed", "failed", "dry-run"}
_MONITOR_REGISTRY_LOCK = threading.RLock()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class MonitorRun:
    run_id: str
    kind: str
    command: list[str]
    status: str
    created_at: str
    updated_at: str
    progress: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _monitor_root(paths: WorkspacePaths) -> Path:
    return paths.runs_root / "monitor"


def _new_run_id(kind: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{kind}-{timestamp}-{os.getpid()}"


def _record_path(paths: WorkspacePaths, run_id: str) -> Path:
    return _monitor_root(paths) / f"{run_id}.json"


def _write_record(paths: WorkspacePaths, record: dict[str, Any]) -> None:
    path = _record_path(paths, str(record["run_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(json.dumps(record, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)


def _read_record(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _status(value: str) -> str:
    status = str(value or "running").strip().lower()
    return status if status in RUN_STATUSES else "running"


def create_monitor_run(
    paths: WorkspacePaths,
    *,
    kind: str,
    command: list[str],
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    status: str = "running",
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    environment_snapshot = environment
    if environment_snapshot is None:
        try:
            environment_snapshot = collect_environment_snapshot(cwd=paths.repo_root)
        except Exception as error:
            environment_snapshot = {
                "cwd": str(paths.repo_root),
                "environment_snapshot_failed": True,
                "environment_snapshot_error": error_payload(error),
            }
    with _MONITOR_REGISTRY_LOCK:
        now = utc_now_iso()
        run = MonitorRun(
            run_id=_new_run_id(kind),
            kind=str(kind).strip() or "unknown",
            command=[str(item) for item in command],
            status=_status(status),
            created_at=now,
            updated_at=now,
            progress=progress or {},
            inputs=inputs or {},
            outputs=outputs or {},
            summary=summary or {},
            environment=environment_snapshot,
        ).to_dict()
        _write_record(paths, run)
        return run


def update_monitor_run(
    paths: WorkspacePaths,
    run_id: str,
    *,
    status: str | None = None,
    progress: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _MONITOR_REGISTRY_LOCK:
        path = _record_path(paths, run_id)
        record = _read_record(path)
        if not record:
            raise FileNotFoundError(f"Monitor run not found: {run_id}")
        if status is not None:
            record["status"] = _status(status)
        if progress is not None:
            record["progress"] = progress
        if inputs is not None:
            record["inputs"] = {**dict(record.get("inputs", {})), **inputs}
        if outputs is not None:
            record["outputs"] = {**dict(record.get("outputs", {})), **outputs}
        if summary is not None:
            record["summary"] = {**dict(record.get("summary", {})), **summary}
        if error is not None:
            record["error"] = error
        record["updated_at"] = utc_now_iso()
        _write_record(paths, record)
        return record


def error_payload(error: BaseException) -> dict[str, str]:
    message = str(error).strip()
    return {
        "type": type(error).__name__,
        "message": message or repr(error),
        "repr": repr(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__, limit=8)
        ).strip(),
    }


class MonitorRunTracker:
    def __init__(self, paths: WorkspacePaths, record: dict[str, Any]):
        self.paths = paths
        self.record = record
        self._lock = threading.RLock()

    @property
    def run_id(self) -> str:
        return str(self.record.get("run_id", ""))

    def update(
        self,
        *,
        status: str | None = None,
        progress: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.record = update_monitor_run(
                self.paths,
                self.run_id,
                status=status,
                progress=progress,
                inputs=inputs,
                outputs=outputs,
                summary=summary,
                error=error,
            )
            return self.record

    def complete(
        self,
        *,
        progress: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        return self.update(status=status, progress=progress, outputs=outputs, summary=summary)

    def fail(self, error: BaseException) -> dict[str, Any]:
        return self.update(status="failed", error=error_payload(error))


def create_monitor_run_tracker(
    paths: WorkspacePaths,
    *,
    kind: str,
    command: list[str],
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    status: str = "running",
    environment: dict[str, Any] | None = None,
) -> MonitorRunTracker:
    return MonitorRunTracker(
        paths,
        create_monitor_run(
            paths,
            kind=kind,
            command=command,
            inputs=inputs,
            outputs=outputs,
            summary=summary,
            progress=progress,
            status=status,
            environment=environment,
        ),
    )


def list_monitor_runs(
    paths: WorkspacePaths,
    *,
    kind: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    root = _monitor_root(paths)
    records = [_read_record(path) for path in root.glob("*.json")] if root.exists() else []
    records = [record for record in records if record]
    if kind:
        records = [record for record in records if str(record.get("kind", "")) == kind]
    records.sort(key=lambda record: str(record.get("updated_at", "")), reverse=True)
    selected = records[: max(limit, 0)]
    summary: dict[str, Any] = {
        "total": len(selected),
        "running": 0,
        "completed": 0,
        "failed": 0,
        "dry-run": 0,
        "unknown": 0,
        "by_kind": {},
    }
    for record in selected:
        status = str(record.get("status", "unknown"))
        if status in RUN_STATUSES:
            summary[status] += 1
        else:
            summary["unknown"] += 1
        record_kind = str(record.get("kind", "unknown")) or "unknown"
        by_kind = summary["by_kind"]
        by_kind[record_kind] = int(by_kind.get(record_kind, 0)) + 1
    return {"summary": summary, "runs": selected}


def get_monitor_run(paths: WorkspacePaths, run_id: str) -> dict[str, Any]:
    record = _read_record(_record_path(paths, run_id))
    if not record:
        raise FileNotFoundError(f"Monitor run not found: {run_id}")
    return record
