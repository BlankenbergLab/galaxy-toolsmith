from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from galaxy_toolsmith.core.manifests import utc_now_iso
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.runtime.progress import make_progress_snapshot


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class DistributedTask:
    task_id: str
    job_id: str
    kind: str
    payload: dict
    status: str = "pending"
    created_at: str = field(default_factory=utc_now_iso)
    lease_until: str = ""
    assigned_worker: str = ""
    started_at: str = ""
    completed_at: str = ""
    result: dict = field(default_factory=dict)
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class DistributedJob:
    job_id: str
    status: str
    task_ids: list[str]
    payload: dict
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    result: dict = field(default_factory=dict)
    error: str = ""
    artifacts: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _distributed_root(paths: WorkspacePaths) -> Path:
    return paths.runs_root / "distributed"


def _jobs_root(paths: WorkspacePaths) -> Path:
    return _distributed_root(paths) / "jobs"


def _tasks_root(paths: WorkspacePaths) -> Path:
    return _distributed_root(paths) / "tasks"


def _job_path(paths: WorkspacePaths, job_id: str) -> Path:
    return _jobs_root(paths) / f"{job_id}.json"


def _task_path(paths: WorkspacePaths, task_id: str) -> Path:
    return _tasks_root(paths) / f"{task_id}.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_training_job(
    paths: WorkspacePaths,
    *,
    profile_name: str,
    dataset_manifest_path: str,
    corpus_jsonl_path: str,
    variant_id: str,
    trainer_command: list[str],
) -> DistributedJob:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    payload = {
        "profile_name": profile_name,
        "dataset_manifest_path": dataset_manifest_path,
        "corpus_jsonl_path": corpus_jsonl_path,
        "variant_id": variant_id,
        "trainer_command": trainer_command,
    }
    task = DistributedTask(
        task_id=task_id,
        job_id=job_id,
        kind="train",
        payload=payload,
    )
    job = DistributedJob(
        job_id=job_id,
        status="queued",
        task_ids=[task_id],
        payload=payload,
    )
    _write_json(_task_path(paths, task_id), asdict(task))
    _write_json(_job_path(paths, job_id), asdict(job))
    return job


def get_job(paths: WorkspacePaths, job_id: str) -> dict:
    return _read_json(_job_path(paths, job_id))


def list_jobs(paths: WorkspacePaths, *, limit: int = 50) -> list[dict]:
    jobs_root = _jobs_root(paths)
    jobs_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    for job_file in jobs_root.glob("job-*.json"):
        jobs.append(_read_json(job_file))
    jobs.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    if limit > 0:
        return jobs[:limit]
    return jobs


def get_tasks_for_job(paths: WorkspacePaths, job_id: str) -> list[dict]:
    job = get_job(paths, job_id)
    tasks: list[dict] = []
    for task_id in job.get("task_ids", []):
        task_file = _task_path(paths, task_id)
        if task_file.exists():
            tasks.append(_read_json(task_file))
    return tasks


def get_job_progress(paths: WorkspacePaths, job_id: str) -> dict:
    job = get_job(paths, job_id)
    tasks = get_tasks_for_job(paths, job_id)
    started_at = str(job.get("created_at", "")).strip() or utc_now_iso()
    total = len(tasks)
    completed = sum(1 for task in tasks if task.get("status") == "completed")
    snapshot = make_progress_snapshot(
        started_at=started_at,
        completed_units=completed,
        total_units=total if total > 0 else None,
    )
    return snapshot.to_dict()


def claim_task(paths: WorkspacePaths, *, worker_id: str, lease_seconds: int) -> dict | None:
    tasks_root = _tasks_root(paths)
    tasks_root.mkdir(parents=True, exist_ok=True)
    now = _utc_now()
    for task_file in sorted(tasks_root.glob("task-*.json")):
        task = _read_json(task_file)
        status = str(task.get("status", "pending"))
        lease_until_raw = str(task.get("lease_until", "")).strip()
        lease_expired = True
        if lease_until_raw:
            lease_expired = _parse_iso(lease_until_raw) <= now
        if status == "pending" or (status == "running" and lease_expired):
            task["status"] = "running"
            task["assigned_worker"] = worker_id
            task["started_at"] = task.get("started_at") or now.isoformat()
            task["lease_until"] = (now + timedelta(seconds=lease_seconds)).isoformat()
            _write_json(task_file, task)
            job = get_job(paths, str(task.get("job_id", "")))
            job["status"] = "running"
            job["updated_at"] = now.isoformat()
            _write_json(_job_path(paths, job["job_id"]), job)
            return task
    return None


def heartbeat_task(
    paths: WorkspacePaths,
    *,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    progress: dict | None = None,
) -> dict:
    task_file = _task_path(paths, task_id)
    task = _read_json(task_file)
    if task.get("assigned_worker") != worker_id:
        raise RuntimeError("worker_mismatch")
    if task.get("status") != "running":
        raise RuntimeError("task_not_running")
    task["lease_until"] = (_utc_now() + timedelta(seconds=lease_seconds)).isoformat()
    if isinstance(progress, dict):
        task["progress"] = progress
    _write_json(task_file, task)
    return task


def complete_task(
    paths: WorkspacePaths,
    *,
    task_id: str,
    worker_id: str,
    success: bool,
    result: dict,
    error: str = "",
) -> dict:
    now = _utc_now()
    task_file = _task_path(paths, task_id)
    task = _read_json(task_file)
    if task.get("assigned_worker") != worker_id:
        raise RuntimeError("worker_mismatch")
    task["status"] = "completed" if success else "failed"
    task["completed_at"] = now.isoformat()
    task["result"] = result
    task["error"] = error
    task["lease_until"] = ""
    _write_json(task_file, task)

    job_id = str(task.get("job_id", ""))
    job_file = _job_path(paths, job_id)
    job = _read_json(job_file)
    tasks = get_tasks_for_job(paths, job_id)
    if any(item.get("status") == "failed" for item in tasks):
        job["status"] = "failed"
        job["error"] = error or "task_failed"
    elif all(item.get("status") == "completed" for item in tasks):
        job["status"] = "completed"
    else:
        job["status"] = "running"
    if success and result.get("artifacts"):
        job["artifacts"] = list(result.get("artifacts", []))
    if success and result.get("training_run"):
        job["result"] = result
    job["updated_at"] = now.isoformat()
    _write_json(job_file, job)
    return job


def list_job_artifacts(paths: WorkspacePaths, job_id: str) -> list[dict]:
    job = get_job(paths, job_id)
    artifacts: list[dict] = []
    for idx, artifact in enumerate(job.get("artifacts", [])):
        abs_path = Path(str(artifact.get("path", ""))).resolve()
        artifact_id = f"a{idx}"
        if abs_path.exists() and abs_path.is_file():
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "name": artifact.get("name", abs_path.name),
                    "path": str(abs_path),
                    "size_bytes": abs_path.stat().st_size,
                }
            )
    return artifacts


def resolve_artifact_path(paths: WorkspacePaths, job_id: str, artifact_id: str) -> Path:
    if not artifact_id.startswith("a"):
        raise RuntimeError("invalid_artifact_id")
    index = int(artifact_id[1:])
    job = get_job(paths, job_id)
    artifacts = job.get("artifacts", [])
    if index < 0 or index >= len(artifacts):
        raise RuntimeError("artifact_not_found")
    artifact_path = Path(str(artifacts[index].get("path", ""))).resolve()
    if not artifact_path.exists():
        raise RuntimeError("artifact_missing")
    return artifact_path
