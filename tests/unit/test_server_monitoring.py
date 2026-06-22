from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from galaxy_toolsmith.core.manifests import TrainingRunManifest
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.orchestration.distributed import create_training_job
from galaxy_toolsmith.runtime.run_registry import create_monitor_run
from galaxy_toolsmith.server.app import create_app

pytest.importorskip("fastapi")
anyio = pytest.importorskip("anyio")
httpx = pytest.importorskip("httpx")


class _ASGITestClient:
    def __init__(self, app) -> None:
        self._app = app

    def get(self, path: str, **kwargs):
        return anyio.run(self._request, "GET", path, kwargs)

    def post(self, path: str, **kwargs):
        return anyio.run(self._request, "POST", path, kwargs)

    async def _request(self, method: str, path: str, kwargs: dict[str, Any]):
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.request(method, path, **kwargs)
            await response.aread()
            return response


def _build_client(
    repo_root: Path,
    auth_tokens: list[str] | None = None,
    status_log_path: Path | None = None,
    allow_stub_local: bool = False,
) -> _ASGITestClient:
    app = create_app(
        repo_root=repo_root,
        provider="local",
        model="",
        model_variant="server-default",
        temperature=0.1,
        max_tokens=1024,
        max_prompt_help_chars=12000,
        auth_tokens=auth_tokens or [],
        require_generate_auth=False,
        allow_stub_local=allow_stub_local,
        status_log_path=status_log_path,
    )
    return _ASGITestClient(app)


def test_monitor_dashboard_route_served(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    response = client.get("/monitor")
    assert response.status_code == 200
    assert "Galaxy Toolsmith Monitor" in response.text
    assert "/train/local-runs?limit=50" in response.text
    assert "/runs?limit=50" in response.text
    assert "Benchmark Runs" in response.text
    assert "Inference Runs" in response.text


def test_monitor_root_redirects_to_dashboard(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/monitor"


def test_train_jobs_endpoint_returns_summary_and_progress(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    create_training_job(
        paths=paths,
        profile_name="agentic-devstral-24b",
        dataset_manifest_path=str(tmp_path / "dataset.manifest.json"),
        corpus_jsonl_path=str(tmp_path / "corpus.jsonl"),
        variant_id="",
        trainer_command=[],
    )

    client = _build_client(tmp_path)
    response = client.get("/train/jobs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 1
    assert len(payload["jobs"]) == 1
    assert "progress" in payload["jobs"][0]
    assert payload["jobs"][0]["progress"]["completed_units"] == 0
    assert payload["jobs"][0]["progress"]["total_units"] == 1


def test_train_jobs_endpoint_requires_auth_when_tokens_enabled(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    create_training_job(
        paths=paths,
        profile_name="agentic-devstral-24b",
        dataset_manifest_path=str(tmp_path / "dataset.manifest.json"),
        corpus_jsonl_path=str(tmp_path / "corpus.jsonl"),
        variant_id="",
        trainer_command=[],
    )

    client = _build_client(tmp_path, auth_tokens=["secret-token"])
    unauthorized = client.get("/train/jobs")
    assert unauthorized.status_code == 401

    authorized = client.get("/train/jobs", headers={"Authorization": "Bearer secret-token"})
    assert authorized.status_code == 200


def test_local_training_runs_endpoint_returns_direct_runs(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    run_dir = paths.runs_root / "training" / "train-local"
    run_dir.mkdir(parents=True)
    manifest = TrainingRunManifest(
        run_id="train-local",
        profile_name="proto-qwen25-7b",
        backend="axolotl",
        status="running",
        metrics_path=str(run_dir / "metrics.json"),
    )
    (run_dir / "run.manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        '{"status": "running", "backend_impl": "axolotl", "progress": {"completed_units": 0, "total_units": 1}}',
        encoding="utf-8",
    )

    client = _build_client(tmp_path)
    response = client.get("/train/local-runs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 1
    assert payload["runs"][0]["run"]["run_id"] == "train-local"
    assert payload["runs"][0]["progress"]["total_units"] == 1

    detail = client.get("/train/local-runs/train-local")
    assert detail.status_code == 200
    assert detail.json()["status"] == "running"


def test_local_training_runs_endpoint_requires_auth_when_tokens_enabled(tmp_path: Path) -> None:
    client = _build_client(tmp_path, auth_tokens=["secret-token"])
    unauthorized = client.get("/train/local-runs")
    assert unauthorized.status_code == 401

    authorized = client.get("/train/local-runs", headers={"Authorization": "Bearer secret-token"})
    assert authorized.status_code == 200


def test_runs_endpoint_returns_monitor_registry_records(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    run = create_monitor_run(
        paths,
        kind="benchmark",
        command=["gtsm", "benchmark-generate"],
        inputs={"provider": "local"},
        summary={"attempted": 5, "succeeded": 4, "failed": 1},
        environment={"runtime_capabilities": {"recommended_backend": "cuda"}},
        status="completed",
    )

    client = _build_client(tmp_path)
    response = client.get("/runs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["completed"] == 1
    assert payload["summary"]["by_kind"]["benchmark"] == 1
    assert payload["runs"][0]["run_id"] == run["run_id"]

    detail = client.get(f"/runs/{run['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["summary"]["succeeded"] == 4


def test_runs_endpoint_requires_auth_when_tokens_enabled(tmp_path: Path) -> None:
    client = _build_client(tmp_path, auth_tokens=["secret-token"])
    unauthorized = client.get("/runs")
    assert unauthorized.status_code == 401

    authorized = client.get("/runs", headers={"Authorization": "Bearer secret-token"})
    assert authorized.status_code == 200


def test_server_generate_request_writes_inference_run(tmp_path: Path) -> None:
    client = _build_client(tmp_path, allow_stub_local=True)

    response = client.post(
        "/generate",
        json={
            "tool_name": "echo_tool",
            "help_text": "Usage: echo_tool --input FILE",
            "allow_stub_local": True,
        },
    )
    assert response.status_code == 200

    runs = client.get("/runs?kind=inference").json()
    assert runs["summary"]["completed"] == 1
    assert runs["runs"][0]["inputs"]["tool_name"] == "echo_tool"
    assert runs["runs"][0]["summary"]["validation"]["root_is_tool"] is True


def test_server_generate_supports_udt_yaml(tmp_path: Path) -> None:
    client = _build_client(tmp_path, allow_stub_local=True)

    response = client.post(
        "/generate",
        json={
            "tool_name": "echo_tool",
            "help_text": "Usage: echo_tool --input FILE",
            "allow_stub_local": True,
            "artifact_format": "udt-yaml",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_format"] == "udt_yaml"
    assert payload["udt_yaml"].startswith("class: GalaxyUserTool")
    assert payload["validation"]["artifact_valid"] is True


def test_server_monitoring_writes_status_log_when_enabled(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    job = create_training_job(
        paths=paths,
        profile_name="agentic-devstral-24b",
        dataset_manifest_path=str(tmp_path / "dataset.manifest.json"),
        corpus_jsonl_path=str(tmp_path / "corpus.jsonl"),
        variant_id="",
        trainer_command=[],
    )
    status_log_path = tmp_path / "status" / "server-status.jsonl"
    client = _build_client(tmp_path, status_log_path=status_log_path)
    response = client.get("/train/jobs")
    assert response.status_code == 200
    response = client.get(f"/train/jobs/{job.job_id}")
    assert response.status_code == 200
    lines = status_log_path.read_text(encoding="utf-8").splitlines()
    assert any('"status": "server-train-jobs-list-requested"' in line for line in lines)
    assert any('"status": "server-train-job-status-requested"' in line for line in lines)
