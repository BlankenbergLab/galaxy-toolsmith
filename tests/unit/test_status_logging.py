from __future__ import annotations

import json
from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.orchestration.benchmark import run_benchmark_generation
from galaxy_toolsmith.runtime.status import emit_status


def test_emit_status_console_only(capsys) -> None:
    emit_status({"status": "test", "value": 1})
    captured = capsys.readouterr()
    assert '"status": "test"' in captured.out


def test_emit_status_writes_file_when_enabled(tmp_path: Path, capsys) -> None:
    status_log = tmp_path / "logs" / "status.jsonl"
    emit_status({"status": "test-file", "value": 2}, status_log_path=status_log)
    captured = capsys.readouterr()
    assert '"status": "test-file"' in captured.out
    lines = status_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "test-file"


def test_benchmark_status_sink_receives_progress_events(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "echo_tool", "help_text": "Usage: echo_tool --input FILE"}) + "\n",
        encoding="utf-8",
    )
    events: list[dict] = []
    run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
        status_sink=lambda payload: events.append(payload),
    )
    assert any(event.get("status") == "benchmark-progress" for event in events)
    assert any(event.get("status") == "benchmark-record-started" for event in events)
    assert any(event.get("status") == "benchmark-model-load-started" for event in events)
    ready_events = [event for event in events if event.get("status") == "benchmark-model-ready"]
    assert ready_events
    assert ready_events[0]["startup"]["backend"] == "local-stub"
