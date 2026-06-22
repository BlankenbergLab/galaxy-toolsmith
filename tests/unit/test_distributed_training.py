from __future__ import annotations

from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.orchestration.distributed import (
    claim_task,
    complete_task,
    create_training_job,
    get_job,
    get_job_progress,
    list_job_artifacts,
)


def test_distributed_job_lifecycle(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()

    artifact_file = tmp_path / "artifact.bin"
    artifact_file.write_text("x", encoding="utf-8")

    job = create_training_job(
        paths=paths,
        profile_name="proto-qwen25-7b",
        dataset_manifest_path=str(tmp_path / "dataset.manifest.json"),
        corpus_jsonl_path=str(tmp_path / "corpus.jsonl"),
        variant_id="",
        trainer_command=[],
    )
    claimed = claim_task(paths, worker_id="worker-a", lease_seconds=60)
    assert claimed is not None

    updated = complete_task(
        paths,
        task_id=str(claimed["task_id"]),
        worker_id="worker-a",
        success=True,
        result={"artifacts": [{"name": "artifact", "path": str(artifact_file)}]},
    )
    assert updated["status"] == "completed"

    reloaded = get_job(paths, job.job_id)
    assert reloaded["status"] == "completed"
    progress = get_job_progress(paths, job.job_id)
    assert progress["completed_units"] == 1
    assert progress["total_units"] == 1
    artifacts = list_job_artifacts(paths, job.job_id)
    assert len(artifacts) == 1
