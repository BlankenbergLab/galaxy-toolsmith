from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from galaxy_toolsmith.cli import main as cli_main
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.runtime import run_registry
from galaxy_toolsmith.runtime.environment import collect_environment_snapshot


def test_monitor_run_registry_create_update_and_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    monkeypatch.setattr(
        run_registry,
        "collect_environment_snapshot",
        lambda cwd=None: {
            "gtsm_version": "0.1.0",
            "runtime_capabilities": {"recommended_backend": "cuda"},
            "package_versions": {"torch": "2.12.0"},
        },
    )

    tracker = run_registry.create_monitor_run_tracker(
        paths,
        kind="benchmark",
        command=["gtsm", "benchmark-generate"],
        inputs={"provider": "local"},
    )
    tracker.update(progress={"completed_units": 1, "total_units": 5})
    tracker.complete(summary={"attempted": 5, "succeeded": 5, "failed": 0})

    payload = run_registry.list_monitor_runs(paths)

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["completed"] == 1
    assert payload["summary"]["by_kind"]["benchmark"] == 1
    run = payload["runs"][0]
    assert run["kind"] == "benchmark"
    assert run["status"] == "completed"
    assert run["progress"]["completed_units"] == 1
    assert run["summary"]["succeeded"] == 5
    assert run["environment"]["package_versions"]["torch"] == "2.12.0"
    assert run_registry.get_monitor_run(paths, run["run_id"])["run_id"] == run["run_id"]


def test_monitor_run_registry_records_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    monkeypatch.setattr(run_registry, "collect_environment_snapshot", lambda cwd=None: {})
    tracker = run_registry.create_monitor_run_tracker(
        paths,
        kind="inference",
        command=["gtsm", "generate-wrapper"],
    )

    tracker.fail(RuntimeError("generation failed"))
    run = run_registry.get_monitor_run(paths, tracker.run_id)

    assert run["status"] == "failed"
    assert run["error"]["type"] == "RuntimeError"
    assert run["error"]["message"] == "generation failed"


def test_monitor_run_registry_handles_concurrent_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    monkeypatch.setattr(run_registry, "collect_environment_snapshot", lambda cwd=None: {})
    tracker = run_registry.create_monitor_run_tracker(
        paths,
        kind="benchmark",
        command=["gtsm", "benchmark-generate"],
    )

    def update(index: int) -> None:
        if index % 2:
            tracker.update(progress={"completed_units": index, "total_units": 50})
        else:
            tracker.update(summary={f"shard_{index}": {"completed": index}})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(update, range(50)))

    run = run_registry.get_monitor_run(paths, tracker.run_id)
    monitor_path = paths.runs_root / "monitor" / f"{tracker.run_id}.json"

    assert run["run_id"] == tracker.run_id
    assert json.loads(monitor_path.read_text(encoding="utf-8"))["run_id"] == tracker.run_id
    assert run["summary"]["shard_48"]["completed"] == 48
    assert run["progress"]["total_units"] == 50
    assert not list(monitor_path.parent.glob(f".{monitor_path.name}.*.tmp"))


def test_environment_snapshot_handles_missing_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("galaxy_toolsmith.runtime.environment.shutil.which", lambda name: None)

    snapshot = collect_environment_snapshot(cwd=Path("/tmp/example"))

    assert snapshot["cwd"] == "/tmp/example"
    assert snapshot["gtsm_version"]
    assert "python" in snapshot
    assert "runtime_capabilities" in snapshot
    assert snapshot["gpu_summary"]["available"] is False


def test_environment_snapshot_captures_nvidia_smi_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert text is True
        assert capture_output is True
        assert check is False
        assert timeout == 10.0
        if "--query-gpu=index,name,memory.total,memory.used,driver_version" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="0, NVIDIA A100-PCIE-40GB, 40960, 0, 590.48.01\n",
                stderr="",
            )
        if command == ["nvidia-smi", "--version"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "NVIDIA-SMI version  : 590.48.01\n"
                    "NVML version        : 590.48\n"
                    "DRIVER version      : 590.48.01\n"
                    "CUDA Version        : 13.1\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("galaxy_toolsmith.runtime.environment.shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("galaxy_toolsmith.runtime.environment.subprocess.run", fake_run)

    snapshot = collect_environment_snapshot(cwd=Path("/tmp/example"))
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert ["nvidia-smi"] not in calls
    assert snapshot["gpu_summary"]["cuda_version"] == "13.1"
    assert snapshot["gpu_summary"]["gpus"] == [
        {
            "index": "0",
            "name": "NVIDIA A100-PCIE-40GB",
            "memory_total_mib": "40960",
            "memory_used_mib": "0",
            "driver_version": "590.48.01",
        }
    ]
    assert snapshot["gpu_summary"]["nvidia_smi_errors"] == []


def test_environment_snapshot_handles_nvidia_smi_timeout_with_partial_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str],
        *,
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,name,memory.total,memory.used,driver_version" in command:
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output="0, NVIDIA A100-PCIE-40GB, 40960, 512, 590.48.01\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="CUDA Version        : 13.1\n",
            stderr="",
        )

    monkeypatch.setattr("galaxy_toolsmith.runtime.environment.shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("galaxy_toolsmith.runtime.environment.subprocess.run", fake_run)

    snapshot = collect_environment_snapshot(cwd=Path("/tmp/example"))
    summary = snapshot["gpu_summary"]

    assert summary["nvidia_smi_timed_out"] is True
    assert "timed out after 10s" in summary["nvidia_smi_errors"][0]
    assert summary["cuda_version"] == "13.1"
    assert summary["gpus"][0]["memory_used_mib"] == "512"


def test_environment_snapshot_records_nvidia_smi_driver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str],
        *,
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            9,
            stdout=(
                "NVIDIA-SMI has failed because it couldn't communicate with the "
                "NVIDIA driver.\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("galaxy_toolsmith.runtime.environment.shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("galaxy_toolsmith.runtime.environment.subprocess.run", fake_run)

    snapshot = collect_environment_snapshot(cwd=Path("/tmp/example"))
    summary = snapshot["gpu_summary"]

    assert summary["available"] is True
    assert summary["gpus"] == []
    assert summary["cuda_version"] == ""
    assert summary["nvidia_smi_timed_out"] is False
    assert len(summary["nvidia_smi_errors"]) == 2
    assert "couldn't communicate with the NVIDIA driver" in summary["nvidia_smi_errors"][0]


def test_monitor_run_registry_falls_back_when_environment_snapshot_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()

    def fail_snapshot(cwd: Path | None = None) -> dict:
        raise RuntimeError("snapshot boom")

    monkeypatch.setattr(run_registry, "collect_environment_snapshot", fail_snapshot)

    tracker = run_registry.create_monitor_run_tracker(
        paths,
        kind="benchmark",
        command=["gtsm", "benchmark-generate"],
    )
    run = run_registry.get_monitor_run(paths, tracker.run_id)

    assert run["status"] == "running"
    assert run["environment"]["environment_snapshot_failed"] is True
    assert run["environment"]["environment_snapshot_error"]["type"] == "RuntimeError"
    assert run["environment"]["environment_snapshot_error"]["message"] == "snapshot boom"


def test_benchmark_cli_writes_monitor_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_registry, "collect_environment_snapshot", lambda cwd=None: {})
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        '{"tool_name": "echo_tool", "help_text": "Usage: echo_tool --input FILE"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "gtsm",
            "--repo-root",
            str(tmp_path),
            "benchmark-generate",
            "--corpus-jsonl",
            str(corpus_jsonl),
            "--wrappers-dir",
            str(tmp_path / "wrappers"),
            "--generation-records",
            str(tmp_path / "generation.records.json"),
            "--evaluation-report",
            str(tmp_path / "evaluation.summary.json"),
            "--benchmark-summary",
            str(tmp_path / "benchmark.summary.json"),
            "--limit",
            "1",
            "--allow-stub-local",
        ],
    )

    assert cli_main.main() == 0
    runs = run_registry.list_monitor_runs(WorkspacePaths.from_repo_root(tmp_path), kind="benchmark")

    assert runs["summary"]["completed"] == 1
    assert runs["runs"][0]["summary"]["attempted"] == 1
    assert runs["runs"][0]["summary"]["succeeded"] == 1


def test_benchmark_cli_continues_when_monitor_update_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTracker:
        def update(self, **kwargs: object) -> dict:
            raise FileNotFoundError("monitor missing")

        def complete(self, **kwargs: object) -> dict:
            raise FileNotFoundError("monitor missing")

        def fail(self, error: BaseException) -> dict:
            raise FileNotFoundError("monitor missing")

    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        '{"tool_name": "echo_tool", "help_text": "Usage: echo_tool --input FILE"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_main,
        "create_monitor_run_tracker",
        lambda *args, **kwargs: FailingTracker(),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "gtsm",
            "--repo-root",
            str(tmp_path),
            "benchmark-generate",
            "--corpus-jsonl",
            str(corpus_jsonl),
            "--wrappers-dir",
            str(tmp_path / "wrappers"),
            "--generation-records",
            str(tmp_path / "generation.records.json"),
            "--evaluation-report",
            str(tmp_path / "evaluation.summary.json"),
            "--benchmark-summary",
            str(tmp_path / "benchmark.summary.json"),
            "--limit",
            "1",
            "--allow-stub-local",
        ],
    )

    assert cli_main.main() == 0
    summary = json.loads((tmp_path / "benchmark.summary.json").read_text(encoding="utf-8"))

    assert summary["attempted"] == 1
    assert summary["succeeded"] == 1


def test_generate_wrapper_cli_writes_monitor_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_registry, "collect_environment_snapshot", lambda cwd=None: {})
    help_text = tmp_path / "help.txt"
    help_text.write_text("Usage: echo_tool --input FILE\n", encoding="utf-8")
    output = tmp_path / "echo.xml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gtsm",
            "--repo-root",
            str(tmp_path),
            "generate-wrapper",
            "--tool-name",
            "echo_tool",
            "--help-text-file",
            str(help_text),
            "--output",
            str(output),
            "--allow-stub-local",
        ],
    )

    assert cli_main.main() == 0
    runs = run_registry.list_monitor_runs(WorkspacePaths.from_repo_root(tmp_path), kind="inference")

    assert runs["summary"]["completed"] == 1
    assert runs["runs"][0]["inputs"]["tool_name"] == "echo_tool"
    assert runs["runs"][0]["outputs"]["output_xml_path"] == str(output)
    assert runs["runs"][0]["summary"]["validation"]["root_is_tool"] is True


def test_serve_process_discovery_filters_to_matching_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    rows = [
        {
            "pid": 101,
            "command": (
                "python -m galaxy_toolsmith.cli.main --repo-root "
                f"{tmp_path} serve --host 0.0.0.0 --port 8765"
            ),
        },
        {
            "pid": 102,
            "command": "python -m galaxy_toolsmith.cli.main benchmark-generate --port 8765",
        },
        {
            "pid": 103,
            "command": "python -u -m axolotl.cli.train run.yml",
        },
        {
            "pid": 104,
            "command": "squashfuse_ll -f /data/home/example/rootfs",
        },
        {
            "pid": 105,
            "command": "python -m galaxy_toolsmith.cli.main serve-stop --port 8765",
        },
    ]
    monkeypatch.setattr(cli_main, "_ps_process_rows", lambda: rows)

    matches = cli_main._matching_server_processes(paths=paths, host="", port=8765)

    assert [item["pid"] for item in matches] == [101]
    assert matches[0]["host"] == "0.0.0.0"
    assert matches[0]["port"] == 8765


def test_serve_process_discovery_respects_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    other = tmp_path / "other"
    rows = [
        {
            "pid": 101,
            "command": (
                "python -m galaxy_toolsmith.cli.main --repo-root "
                f"{other} serve --host 0.0.0.0 --port 8765"
            ),
        }
    ]
    monkeypatch.setattr(cli_main, "_ps_process_rows", lambda: rows)

    assert cli_main._matching_server_processes(paths=paths, host="", port=8765) == []


def test_stop_serve_processes_dry_run_does_not_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    monkeypatch.setattr(
        cli_main,
        "_ps_process_rows",
        lambda: [
            {
                "pid": 101,
                "command": "python -m galaxy_toolsmith.cli.main serve --host 0.0.0.0 --port 8765",
            }
        ],
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(cli_main.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    result = cli_main.stop_serve_processes(paths=paths, dry_run=True)

    assert result["dry_run"] is True
    assert [item["pid"] for item in result["matched"]] == [101]
    assert signals == []


def test_stop_serve_processes_terminates_matched_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    alive = {101}
    monkeypatch.setattr(
        cli_main,
        "_ps_process_rows",
        lambda: [
            {
                "pid": 101,
                "command": "python -m galaxy_toolsmith.cli.main serve --host 0.0.0.0 --port 8765",
            }
        ],
    )
    monkeypatch.setattr(cli_main, "_pid_alive", lambda pid: pid in alive)
    monkeypatch.setattr(cli_main.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(cli_main, "_mark_stopped_server_runs", lambda paths, pids: None)

    def fake_kill(pid: int, sig: int) -> None:
        if sig == cli_main.signal.SIGTERM:
            alive.discard(pid)

    monkeypatch.setattr(cli_main.os, "kill", fake_kill)

    result = cli_main.stop_serve_processes(paths=paths, timeout_seconds=0.1)

    assert [item["pid"] for item in result["terminated"]] == [101]
    assert result["still_running"] == []


def test_model_cache_info_cli_reports_workspace_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GTSM_MODEL_CACHE_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "gtsm",
            "--repo-root",
            str(tmp_path),
            "model-cache-info",
        ],
    )

    assert cli_main.main() == 0
    payload = json.loads(capsys.readouterr().out)
    expected = str(WorkspacePaths.from_repo_root(tmp_path).models_root / "hf-cache")

    assert payload["cache_root"] == expected
    assert payload["model_source_policy"]["cache_dir_source"] == "workspace-default"
