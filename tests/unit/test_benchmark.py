from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.output_diagnostics import diagnose_generated_xml
from galaxy_toolsmith.inference.source_context import source_context_settings
from galaxy_toolsmith.orchestration import benchmark as benchmark_mod
from galaxy_toolsmith.orchestration.benchmark import (
    DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS,
    resolve_benchmark_parallelism,
    run_benchmark_generation,
    run_benchmark_generation_sharded,
)


def test_benchmark_summary_includes_progress(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()

    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "echo_tool", "help_text": "Usage: echo_tool --input FILE"}) + "\n",
        encoding="utf-8",
    )

    summary = run_benchmark_generation(
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
    )

    assert summary.attempted == 1
    assert summary.progress["completed_units"] == 1
    assert summary.progress["total_units"] == 1
    assert summary.startup["backend"] == "local-stub"
    assert summary.startup["model_loaded"] is False
    assert summary.quality["startup"]["backend"] == "local-stub"
    assert summary.quality["validity"]["success_rate"] == 1.0


def test_benchmark_generation_supports_udt_yaml(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "echo_tool", "help_text": "Usage: echo_tool --input FILE"}) + "\n",
        encoding="utf-8",
    )

    summary = run_benchmark_generation(
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
        artifact_format="udt-yaml",
    )

    records = json.loads((tmp_path / "generation_records.json").read_text(encoding="utf-8"))
    output_path = Path(records[0]["output_udt_yaml_path"])
    assert summary.artifact_format == "udt_yaml"
    assert output_path.suffix == ".yml"
    assert output_path.read_text(encoding="utf-8").startswith("class: GalaxyUserTool")
    assert summary.quality["validity"]["artifact_valid_rate"] == 1.0
    assert summary.quality["validity"]["udt_schema_valid_rate"] == 1.0


def test_benchmark_generation_can_record_suite_recommendation(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "samtools", "help_text": "Usage: samtools {view,sort}\n"}) + "\n",
        encoding="utf-8",
    )

    summary = run_benchmark_generation(
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
        suite_generation="recommend",
    )

    records = json.loads((tmp_path / "generation_records.json").read_text(encoding="utf-8"))
    assert summary.suite_generation == "recommend"
    assert summary.suite_count == 1
    assert records[0]["suite_plan"]["suite_recommended"] is True


def test_benchmark_generation_can_generate_suite_bundle(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "samtools", "help_text": "Usage: samtools {view,sort}\n"}) + "\n",
        encoding="utf-8",
    )

    summary = run_benchmark_generation(
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
        suite_generation="generate",
    )

    records = json.loads((tmp_path / "generation_records.json").read_text(encoding="utf-8"))
    generated_files = records[0]["generated_files"]
    assert summary.suite_generation == "generate"
    assert summary.suite_member_count == 2
    assert len(generated_files) == 2
    assert all(Path(item["path"]).exists() for item in generated_files)


def test_benchmark_generation_records_source_context_summary(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    source_root = tmp_path / "source"
    (source_root / "src" / "echo_tool").mkdir(parents=True)
    (source_root / "src" / "echo_tool" / "cli.py").write_text(
        "from argparse import ArgumentParser\n",
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "echo_tool",
                "help_text": "Usage: echo_tool --input FILE",
                "bioconda_sources": [
                    {
                        "package": "echo_tool",
                        "source_checkout": str(source_root),
                        "command_hints": ["echo_tool"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_benchmark_generation(
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
        source_context_settings=source_context_settings(
            mode="snippets",
            max_chars=4000,
            max_files=2,
        ),
    )

    records = json.loads((tmp_path / "generation_records.json").read_text(encoding="utf-8"))
    assert summary.startup["source_context"]["mode"] == "snippets"
    assert records[0]["source_context"]["mode"] == "snippets"
    assert records[0]["source_context"]["included_files"] == 1
    assert records[0]["source_context"]["included_paths"]


def test_resolve_benchmark_parallelism_uses_all_requested_gpu_devices() -> None:
    process_count, devices = resolve_benchmark_parallelism(
        provider="local",
        num_processes=0,
        gpu_devices="0,1,2,3",
    )

    assert process_count == 4
    assert devices == ["0", "1", "2", "3"]


def test_resolve_benchmark_parallelism_caps_auto_processes_for_small_limits() -> None:
    process_count, devices = resolve_benchmark_parallelism(
        provider="local",
        num_processes=0,
        gpu_devices="0,1,2,3",
        total_records=5,
        min_items_per_process=8,
    )

    assert process_count == 1
    assert devices == ["0"]


def test_resolve_benchmark_parallelism_auto_uses_visible_gpus_for_small_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    process_count, devices = resolve_benchmark_parallelism(
        provider="local",
        num_processes=0,
        gpu_devices="",
        total_records=5,
        min_items_per_process=DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS,
    )

    assert process_count == 4
    assert devices == ["0", "1", "2", "3"]


def test_resolve_benchmark_parallelism_honors_explicit_process_count() -> None:
    process_count, devices = resolve_benchmark_parallelism(
        provider="local",
        num_processes=4,
        gpu_devices="0,1,2,3",
        total_records=5,
    )

    assert process_count == 4
    assert devices == ["0", "1", "2", "3"]


def test_resolve_benchmark_parallelism_rejects_more_processes_than_gpus() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_benchmark_parallelism(
            provider="local",
            num_processes=3,
            gpu_devices="0,1",
        )


def test_resolve_benchmark_parallelism_model_parallel_uses_one_process_all_gpus() -> None:
    process_count, devices = resolve_benchmark_parallelism(
        provider="local",
        num_processes=4,
        gpu_devices="0,1,2,3",
        local_gpu_topology="model-parallel",
    )

    assert process_count == 1
    assert devices == ["0", "1", "2", "3"]


def test_resolve_benchmark_parallelism_rejects_model_parallel_for_external_provider() -> None:
    with pytest.raises(ValueError, match="model-parallel"):
        resolve_benchmark_parallelism(
            provider="openai",
            num_processes=1,
            gpu_devices="0,1",
            local_gpu_topology="model-parallel",
        )


def test_sharded_benchmark_merges_records_and_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    records = []
    for index in range(4):
        reference = tmp_path / f"reference-{index}.xml"
        reference.write_text(
            f"""
<tool id="echo_{index}" name="Echo {index}" version="0.1.0">
  <requirements><requirement type="package">coreutils</requirement></requirements>
  <command><![CDATA[echo "$input" > "$output"]]></command>
  <inputs><param name="input" type="data" format="txt"/></inputs>
  <outputs><data name="output" format="txt"/></outputs>
  <tests><test expect_num_outputs="1"><output name="output"/></test></tests>
</tool>
""".strip(),
            encoding="utf-8",
        )
        records.append(
            {
                "tool_name": f"Echo {index}",
                "help_text": "Usage: echo --input FILE",
                "expanded_xml_path": str(reference),
                "primary_command": "echo",
            }
        )
    corpus_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    captured = []
    sleeps = []
    status_events = []

    def fake_shard_process(**kwargs):
        shard = kwargs["shard"]
        command = kwargs["command"]
        env = kwargs["env"]
        captured.append((shard.index, command, env.get("CUDA_VISIBLE_DEVICES", "")))
        shard_records = [
            json.loads(line)
            for line in shard.corpus_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        generated = []
        shard.wrappers_dir.mkdir(parents=True, exist_ok=True)
        for record in shard_records:
            output_path = shard.wrappers_dir / f"{record['_benchmark_corpus_index']}.xml"
            output_path.write_text(
                """
<tool id="echo" name="Echo" version="0.1.0">
  <requirements><requirement type="package">coreutils</requirement></requirements>
  <command><![CDATA[echo "$input" > "$output"]]></command>
  <inputs><param name="input" type="data" format="txt"/></inputs>
  <outputs><data name="output" format="txt"/></outputs>
  <tests><test expect_num_outputs="1"><output name="output"/></test></tests>
</tool>
""".strip(),
                encoding="utf-8",
            )
            generated.append(
                {
                    **benchmark_mod._record_metadata(record),
                    "tool_name": record["tool_name"],
                    "provider": "local-stub",
                    "model_variant": "variant-a",
                    "output_xml_path": str(output_path),
                    "validation": {
                        "xml_well_formed": True,
                        "root_tag": "tool",
                        "root_is_tool": True,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": [],
                    },
                    "repair_attempted": False,
                }
            )
        shard.generation_records_path.parent.mkdir(parents=True, exist_ok=True)
        shard.generation_records_path.write_text(json.dumps(generated), encoding="utf-8")
        shard.benchmark_summary_path.write_text(
            json.dumps(
                {
                    "attempted": len(shard_records),
                    "succeeded": len(generated),
                    "failed": 0,
                    "failures": [],
                    "startup": {"model_load_seconds": float(shard.index + 1)},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(benchmark_mod, "_run_benchmark_shard_process", fake_shard_process)
    monkeypatch.setattr(benchmark_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    summary = run_benchmark_generation_sharded(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation.records.json",
        evaluation_report_path=tmp_path / "evaluation.summary.json",
        benchmark_summary_path=tmp_path / "benchmark.summary.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        num_processes=2,
        gpu_devices="0,1",
        allow_stub_local=True,
        startup_stagger_seconds=0.25,
        status_sink=status_events.append,
    )

    merged_records = json.loads(Path(summary.generation_records_path).read_text(encoding="utf-8"))

    sharded_started = next(
        event for event in status_events if event["status"] == "benchmark-sharded-started"
    )
    assert sharded_started["processes"] == 2
    assert sharded_started["gpu_devices"] == ["0", "1"]
    assert sharded_started["min_items_per_process"] == DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS
    assert [item[2] for item in sorted(captured)] == ["0", "1"]
    assert all("--benchmark-shard-worker" in item[1] for item in captured)
    assert summary.attempted == 4
    assert summary.succeeded == 4
    assert summary.quality["validity"]["success_rate"] == 1.0
    assert summary.quality["reference_fidelity"]["compared_records"] == 4
    assert len(summary.quality["reference_fidelity"]["records"]) == 4
    assert summary.quality["reference_fidelity"]["records"][0]["input_count_abs_error"] == 0
    assert summary.startup["processes"] == 2
    assert summary.startup["model_load_seconds_max"] == 2.0
    assert summary.quality["startup"]["model_load_seconds_mean"] == 1.5
    assert sleeps == [0.25]
    assert [record["corpus_index"] for record in merged_records] == [0, 1, 2, 3]


def test_model_parallel_sharded_benchmark_exposes_all_gpus_to_one_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "Echo", "help_text": "Usage: echo"}) + "\n",
        encoding="utf-8",
    )
    captured = []

    def fake_shard_process(**kwargs):
        shard = kwargs["shard"]
        captured.append(
            {
                "gpu_device": shard.gpu_device,
                "env_cuda": kwargs["env"].get("CUDA_VISIBLE_DEVICES", ""),
                "command": kwargs["command"],
                "timeout": kwargs["record_timeout_seconds"],
            }
        )
        shard.generation_records_path.parent.mkdir(parents=True, exist_ok=True)
        shard.generation_records_path.write_text("[]", encoding="utf-8")
        shard.benchmark_summary_path.write_text(
            json.dumps({"failures": [], "startup": {"model_load_seconds": 0.1}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(benchmark_mod, "_run_benchmark_shard_process", fake_shard_process)

    summary = run_benchmark_generation_sharded(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation.records.json",
        evaluation_report_path=tmp_path / "evaluation.summary.json",
        benchmark_summary_path=tmp_path / "benchmark.summary.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        num_processes=4,
        gpu_devices="0,1,2,3",
        allow_stub_local=True,
        local_gpu_topology="model-parallel",
        local_offload_policy="fail",
        local_gpu_memory_reserve_gib=3.0,
        resume_existing=True,
        record_timeout_seconds=12,
    )

    assert summary.startup["processes"] == 1
    assert summary.startup["gpu_devices"] == ["0", "1", "2", "3"]
    assert captured == [
        {
            "gpu_device": "0,1,2,3",
            "env_cuda": "0,1,2,3",
            "command": captured[0]["command"],
            "timeout": 12,
        }
    ]
    command = captured[0]["command"]
    assert "--local-gpu-topology" in command
    assert "model-parallel" in command
    assert "--local-offload-policy" in command
    assert "fail" in command
    assert "--resume-existing" in command


def test_benchmark_resume_existing_skips_checkpointed_successes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        "".join(
            json.dumps({"tool_name": f"Echo {index}", "help_text": "Usage: echo"}) + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    calls = 0

    class ValidGeneration:
        def __init__(self, output_path: Path, tool_name: str):
            self.output_path = output_path
            self.tool_name = tool_name

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": self.tool_name,
                    "provider": "local-stub",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": True,
                        "root_tag": "tool",
                        "root_is_tool": True,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": [],
                    },
                }
            )

    def valid_generation(*args, **kwargs):
        nonlocal calls
        calls += 1
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<tool id='echo' name='Echo' version='0.1.0'/>", encoding="utf-8")
        return ValidGeneration(output_path, kwargs["tool_name"])

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", valid_generation)
    checkpoint_path = tmp_path / "checkpoint.records.jsonl"

    first = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
        checkpoint_records_path=checkpoint_path,
    )
    assert first.succeeded == 2
    assert calls == 2
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == 2

    def fail_if_called(*args, **kwargs):
        raise AssertionError("checkpointed record should have been skipped")

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", fail_if_called)
    second = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
        resume_existing=True,
        checkpoint_records_path=checkpoint_path,
    )

    assert second.succeeded == 2
    assert second.startup["skipped_existing"] == 2
    assert second.startup["checkpoint_records_loaded"] == 2


def test_benchmark_resume_existing_reruns_partial_xml_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "Echo", "help_text": "Usage: echo"}) + "\n",
        encoding="utf-8",
    )
    wrappers_dir = tmp_path / "wrappers"
    wrappers_dir.mkdir()
    (wrappers_dir / benchmark_mod._safe_wrapper_filename("Echo")).write_text(
        "<tool id='echo'><inputs><param",
        encoding="utf-8",
    )
    calls = 0

    class ValidGeneration:
        def __init__(self, output_path: Path):
            self.output_path = output_path

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": "Echo",
                    "provider": "local-stub",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": True,
                        "root_tag": "tool",
                        "root_is_tool": True,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": [],
                    },
                }
            )

    def valid_generation(*args, **kwargs):
        nonlocal calls
        calls += 1
        output_path = kwargs["output_path"]
        output_path.write_text("<tool id='echo' name='Echo' version='0.1.0'/>", encoding="utf-8")
        return ValidGeneration(output_path)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", valid_generation)
    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=wrappers_dir,
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
        resume_existing=True,
        checkpoint_records_path=tmp_path / "missing-checkpoint.jsonl",
    )

    assert calls == 1
    assert summary.succeeded == 1


def test_benchmark_shard_process_times_out_active_record(tmp_path: Path) -> None:
    shard = benchmark_mod.BenchmarkShard(
        index=0,
        corpus_jsonl=tmp_path / "corpus.jsonl",
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation.records.json",
        evaluation_report_path=tmp_path / "evaluation.summary.json",
        benchmark_summary_path=tmp_path / "benchmark.summary.json",
        checkpoint_records_path=tmp_path / "checkpoint.records.jsonl",
        status_log_path=tmp_path / "status.jsonl",
        stdout_log_path=tmp_path / "stdout.log",
        stderr_log_path=tmp_path / "stderr.log",
        gpu_device="0",
    )
    events: list[dict] = []
    command = [
        sys.executable,
        "-c",
        (
            "import json,time;"
            "print(json.dumps({'status':'benchmark-record-started',"
            "'corpus_index':1,'tool_name':'Slow','output_xml_path':'slow.xml'}), flush=True);"
            "time.sleep(5)"
        ),
    ]

    with pytest.raises(RuntimeError, match="exceeded record timeout"):
        benchmark_mod._run_benchmark_shard_process(
            shard=shard,
            command=command,
            env={"GTSM_REPO_ROOT": str(tmp_path)},
            status_sink=events.append,
            progress_by_shard={0: 0},
            total_units=1,
            started_at=benchmark_mod.utc_now_iso(),
            lock=threading.Lock(),
            record_timeout_seconds=0.1,
        )

    assert any(event.get("status") == "benchmark-record-timeout" for event in events)


def test_reference_fidelity_splits_comma_separated_datatypes(tmp_path: Path) -> None:
    reference = tmp_path / "reference.xml"
    generated = tmp_path / "generated.xml"
    reference.write_text(
        """
<tool id="x" name="X" version="0.1.0">
  <requirements><requirement type="package">abricate</requirement></requirements>
  <command><![CDATA[abricate "$input" > "$output"]]></command>
  <inputs><param name="input" type="data" format="fasta,genbank"/></inputs>
  <outputs><data name="output" format="tabular"/></outputs>
  <tests><test expect_num_outputs="1"><output name="output"/></test></tests>
</tool>
""".strip(),
        encoding="utf-8",
    )
    generated.write_text(
        """
<tool id="x" name="X" version="0.1.0">
  <requirements><requirement type="package">abricate</requirement></requirements>
  <command><![CDATA[abricate "$input" > "$output"]]></command>
  <inputs><param name="input" type="data" format="genbank,fasta"/></inputs>
  <outputs><data name="output" format="tabular"/></outputs>
  <tests><test expect_num_outputs="1"><output name="output"/></test></tests>
</tool>
""".strip(),
        encoding="utf-8",
    )

    fidelity = benchmark_mod._reference_fidelity(
        [
            {
                "tool_name": "ABRicate",
                "corpus_index": 7,
                "package_id": "iuc/abricate",
                "tool_id": "abricate",
                "expanded_xml_path": str(reference),
                "output_xml_path": str(generated),
                "primary_command": "abricate",
            }
        ]
    )

    assert fidelity["compared_records"] == 1
    assert fidelity["input_datatype_jaccard_mean"] == 1.0
    assert fidelity["output_datatype_jaccard_mean"] == 1.0
    assert fidelity["requirement_package_jaccard_mean"] == 1.0
    assert fidelity["primary_command_presence_rate"] == 1.0
    detail = fidelity["records"][0]
    assert detail["corpus_index"] == 7
    assert detail["input_datatypes_reference"] == ["fasta", "genbank"]
    assert detail["input_datatypes_generated"] == ["fasta", "genbank"]
    assert detail["input_datatype_jaccard"] == 1.0
    assert detail["input_count_abs_error"] == 0


def test_with_corpus_indices_preserves_existing_indices() -> None:
    indexed = benchmark_mod._with_corpus_indices(
        [
            {"tool_name": "a", "_benchmark_corpus_index": 5},
            {"tool_name": "b"},
        ]
    )

    assert indexed[0]["_benchmark_corpus_index"] == 5
    assert indexed[1]["_benchmark_corpus_index"] == 1


def test_benchmark_local_provider_requires_real_backend_without_stub_opt_in(
    tmp_path: Path,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "echo_tool", "help_text": "Usage: echo_tool --input FILE"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No real local generator"):
        run_benchmark_generation(
            paths=paths,
            corpus_jsonl=corpus_jsonl,
            wrappers_dir=tmp_path / "wrappers",
            generation_records_path=tmp_path / "generation_records.json",
            evaluation_report_path=tmp_path / "evaluation_report.json",
            provider="local",
            model_variant="missing-variant",
            model="",
            temperature=0.0,
            max_tokens=128,
            max_workers=4,
            limit=None,
            xsd_path=None,
            run_planemo=False,
        )


def test_benchmark_failure_records_nonempty_details_for_empty_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyMessageError(Exception):
        def __str__(self) -> str:
            return ""

    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "bad/tool:name", "help_text": "Usage: bad"}) + "\n",
        encoding="utf-8",
    )

    def fail_generation(*args, **kwargs):
        raise EmptyMessageError()

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", fail_generation)

    summary = run_benchmark_generation(
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
    )

    assert summary.failed == 1
    failure = summary.failures[0]
    assert failure["error"] == "EmptyMessageError()"
    assert failure["error_type"] == "EmptyMessageError"
    assert failure["error_repr"] == "EmptyMessageError()"
    assert "EmptyMessageError" in failure["traceback"]
    failure_name = Path(failure["output_xml_path"]).name
    assert failure_name.startswith("bad_tool_name-")
    assert failure_name.endswith(".xml")


def test_benchmark_malformed_xml_counts_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "abriTAMR",
                "help_text": "Usage: abritamr run",
                "package_id": "iuc/abritamr",
                "tool_id": "abritamr",
                "primary_command": "abritamr",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class MalformedGeneration:
        def __init__(self, output_path: Path):
            self.output_path = output_path

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": "abriTAMR",
                    "provider": "local-peft",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": False,
                        "root_tag": "",
                        "root_is_tool": False,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": ["XML parse error: no element found: line 47, column 61"],
                    },
                }
            )

    def malformed_generation(*args, **kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<tool id='abritamr'>", encoding="utf-8")
        return MalformedGeneration(output_path)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", malformed_generation)

    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
    )

    records = json.loads(Path(summary.generation_records_path).read_text(encoding="utf-8"))

    assert summary.succeeded == 0
    assert summary.failed == 1
    assert records == []
    failure = summary.failures[0]
    assert failure["error_type"] == "InvalidGeneratedXmlError"
    assert "Generated XML is not well formed" in failure["error"]
    assert failure["validation"]["xml_well_formed"] is False
    assert "line 47" in failure["validation"]["notes"][0]


def test_benchmark_well_formed_non_tool_xml_counts_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "abriTAMR",
                "help_text": "Usage: abritamr run",
                "package_id": "iuc/abritamr",
                "tool_id": "abritamr",
                "primary_command": "abritamr",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class NonToolGeneration:
        def __init__(self, output_path: Path):
            self.output_path = output_path

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": "abriTAMR",
                    "provider": "local-peft",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": True,
                        "root_tag": "macros",
                        "root_is_tool": False,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": ["Expected root <tool>; found <macros>."],
                    },
                }
            )

    def non_tool_generation(*args, **kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<macros><xml name='inputs'/></macros>", encoding="utf-8")
        return NonToolGeneration(output_path)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", non_tool_generation)

    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
    )

    records = json.loads(Path(summary.generation_records_path).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(summary.evaluation_report_path).read_text(encoding="utf-8"))

    assert summary.succeeded == 0
    assert summary.failed == 1
    assert records == []
    assert evaluation["total_wrappers"] == 0
    failure = summary.failures[0]
    assert failure["error_type"] == "InvalidGeneratedXmlError"
    assert "root is not <tool>" in failure["error"]
    assert failure["validation"]["xml_well_formed"] is True
    assert failure["validation"]["root_tag"] == "macros"
    assert failure["validation"]["root_is_tool"] is False


def test_benchmark_degenerate_xml_counts_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "ABRicate Summary", "help_text": "Usage: abricate --summary"})
        + "\n",
        encoding="utf-8",
    )

    class DegenerateGeneration:
        def __init__(self, output_path: Path, xml_text: str):
            self.output_path = output_path
            self.xml_text = xml_text

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": "ABRicate Summary",
                    "provider": "local-peft",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": True,
                        "root_tag": "tool",
                        "root_is_tool": True,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": [],
                        "generation_diagnostics": diagnose_generated_xml(self.xml_text).to_dict(),
                    },
                }
            )

    def degenerate_generation(*args, **kwargs):
        output_path = kwargs["output_path"]
        repeated = "\n".join('        <has_text text="100.00"/>' for _ in range(14))
        xml_text = f"""<tool id="abricate_summary" name="ABRicate Summary" version="0.1.0">
    <command>abricate --summary '$input' > '$output'</command>
    <inputs><param name="input" type="data" format="tabular"/></inputs>
    <outputs><data name="output" format="tabular"/></outputs>
    <tests>
        <test expect_num_outputs="1">
            <output name="output">
                <assert_contents>
{repeated}
                </assert_contents>
            </output>
        </test>
    </tests>
</tool>"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_text, encoding="utf-8")
        return DegenerateGeneration(output_path, xml_text)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", degenerate_generation)

    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
        repair_invalid_xml=False,
    )

    assert summary.succeeded == 0
    assert summary.failed == 1
    failure = summary.failures[0]
    assert "appears degenerate" in failure["error"]
    diagnostics = failure["validation"]["generation_diagnostics"]
    assert diagnostics["repeated_xml_lines"] == ['<has_text text="100.00"/>']


def test_benchmark_repairs_invalid_xml_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "abriTAMR",
                "help_text": "Usage: abritamr run",
                "package_id": "iuc/abritamr",
                "tool_id": "abritamr",
                "primary_command": "abritamr",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    attempts: list[dict] = []

    class AttemptGeneration:
        def __init__(self, output_path: Path, valid: bool, kwargs: dict):
            self.output_path = output_path
            self.valid = valid
            self.kwargs = kwargs

        def to_json(self) -> str:
            validation = (
                {
                    "xml_well_formed": True,
                    "root_tag": "tool",
                    "root_is_tool": True,
                    "unknown_datatypes": [],
                    "xsd_status": "not_run",
                    "planemo_status": "not_run",
                    "notes": [],
                }
                if self.valid
                else {
                    "xml_well_formed": False,
                    "root_tag": "",
                    "root_is_tool": False,
                    "unknown_datatypes": [],
                    "xsd_status": "not_run",
                    "planemo_status": "not_run",
                    "notes": ["XML parse error: unclosed token: line 83, column 12"],
                }
            )
            return json.dumps(
                {
                    "tool_name": "abriTAMR",
                    "provider": "local-peft",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": validation,
                    "attempt_count": self.kwargs.get("attempt_count", 1),
                    "repair_attempted": self.kwargs.get("repair_attempted", False),
                    "repair_reason": self.kwargs.get("repair_reason", ""),
                }
            )

    def repairable_generation(*args, **kwargs):
        attempts.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        valid = len(attempts) == 2
        output_path.write_text(
            "<tool id='abritamr' name='abriTAMR' version='0.1.0'></tool>"
            if valid
            else "<tool id='abritamr'><inputs><param",
            encoding="utf-8",
        )
        return AttemptGeneration(output_path, valid, kwargs)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", repairable_generation)

    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
    )

    records = json.loads(Path(summary.generation_records_path).read_text(encoding="utf-8"))

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert len(attempts) == 2
    assert "previous generated wrapper failed" in attempts[1]["repair_context"]
    assert "under 80 lines" in attempts[1]["repair_context"]
    assert "at most six input parameters" in attempts[1]["repair_context"]
    assert "Do not preserve the full interface" in attempts[1]["repair_context"]
    assert (
        attempts[1]["max_prompt_help_chars"]
        == benchmark_mod.TRUNCATION_REPAIR_MAX_PROMPT_HELP_CHARS
    )
    assert attempts[1]["metadata_hints"] == {
        "package_id": "iuc/abritamr",
        "tool_id": "abritamr",
        "primary_command": "abritamr",
    }
    assert records[0]["attempt_count"] == 2
    assert records[0]["repair_attempted"] is True
    assert "Generated XML is not well formed" in records[0]["repair_reason"]


def test_benchmark_uses_compact_fallback_after_truncated_repair_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "ABRicate List",
                "help_text": "Usage: abricate --list",
                "package_id": "iuc/abricate",
                "tool_id": "abricate_list",
                "primary_command": "abricate",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    attempts: list[dict] = []

    class TruncatedGeneration:
        def __init__(self, output_path: Path, kwargs: dict):
            self.output_path = output_path
            self.kwargs = kwargs

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": "ABRicate List",
                    "provider": "local-peft",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": False,
                        "root_tag": "",
                        "root_is_tool": False,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": ["XML parse error: unclosed token: line 22, column 20"],
                        "generation_diagnostics": {
                            "has_problems": True,
                            "missing_closing_tool": True,
                            "ends_mid_tag": True,
                            "unclosed_cdata": False,
                        },
                    },
                    "attempt_count": self.kwargs.get("attempt_count", 1),
                    "repair_attempted": self.kwargs.get("repair_attempted", False),
                    "repair_reason": self.kwargs.get("repair_reason", ""),
                }
            )

    def truncated_generation(*args, **kwargs):
        attempts.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<tool id='abricate_list'><outputs><data", encoding="utf-8")
        return TruncatedGeneration(output_path, kwargs)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", truncated_generation)

    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
        allow_compact_fallback=True,
    )

    records = json.loads(Path(summary.generation_records_path).read_text(encoding="utf-8"))
    output_xml = Path(records[0]["output_xml_path"]).read_text(encoding="utf-8")

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert len(attempts) == 2
    assert records[0]["attempt_count"] == 3
    assert records[0]["repair_attempted"] is True
    assert records[0]["compact_fallback_attempted"] is True
    assert records[0]["provider"] == "local-compact-fallback"
    assert records[0]["validation"]["xml_well_formed"] is True
    assert records[0]["validation"]["root_is_tool"] is True
    assert '<requirement type="package">abricate</requirement>' in output_xml
    assert "abricate --help" in output_xml


def test_benchmark_reports_truncated_repair_failure_without_compact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "ABRicate List",
                "help_text": "Usage: abricate --list",
                "package_id": "iuc/abricate",
                "tool_id": "abricate_list",
                "primary_command": "abricate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class TruncatedGeneration:
        def __init__(self, output_path: Path, kwargs: dict):
            self.output_path = output_path
            self.kwargs = kwargs

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": "ABRicate List",
                    "provider": "local-peft",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": False,
                        "root_tag": "",
                        "root_is_tool": False,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": ["XML parse error: unclosed token: line 22, column 20"],
                        "generation_diagnostics": {
                            "has_problems": True,
                            "missing_closing_tool": True,
                            "ends_mid_tag": True,
                            "unclosed_cdata": False,
                        },
                    },
                    "attempt_count": self.kwargs.get("attempt_count", 1),
                    "repair_attempted": self.kwargs.get("repair_attempted", False),
                    "repair_reason": self.kwargs.get("repair_reason", ""),
                }
            )

    def truncated_generation(*args, **kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<tool id='abricate_list'><outputs><data", encoding="utf-8")
        return TruncatedGeneration(output_path, kwargs)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", truncated_generation)

    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
    )

    records = json.loads(Path(summary.generation_records_path).read_text(encoding="utf-8"))

    assert summary.succeeded == 0
    assert summary.failed == 1
    assert records == []
    assert summary.failures[0]["compact_fallback_skipped"] is True
    assert "Compact fallback is disabled" in summary.failures[0]["compact_fallback_skip_reason"]


def test_benchmark_can_disable_invalid_xml_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps({"tool_name": "abriTAMR", "help_text": "Usage: abritamr run"}) + "\n",
        encoding="utf-8",
    )
    attempts = 0

    class MalformedGeneration:
        def __init__(self, output_path: Path):
            self.output_path = output_path

        def to_json(self) -> str:
            return json.dumps(
                {
                    "tool_name": "abriTAMR",
                    "provider": "local-peft",
                    "model_variant": "variant-a",
                    "output_xml_path": str(self.output_path),
                    "validation": {
                        "xml_well_formed": False,
                        "root_tag": "",
                        "root_is_tool": False,
                        "unknown_datatypes": [],
                        "xsd_status": "not_run",
                        "planemo_status": "not_run",
                        "notes": ["XML parse error: unclosed token: line 83, column 12"],
                    },
                }
            )

    def malformed_generation(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<tool id='abritamr'><inputs><param", encoding="utf-8")
        return MalformedGeneration(output_path)

    monkeypatch.setattr(benchmark_mod, "generate_wrapper_from_content", malformed_generation)

    summary = run_benchmark_generation(
        paths=paths,
        corpus_jsonl=corpus_jsonl,
        wrappers_dir=tmp_path / "wrappers",
        generation_records_path=tmp_path / "generation_records.json",
        evaluation_report_path=tmp_path / "evaluation_report.json",
        provider="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        max_workers=1,
        limit=None,
        xsd_path=None,
        run_planemo=False,
        allow_stub_local=True,
        repair_invalid_xml=False,
    )

    assert summary.succeeded == 0
    assert summary.failed == 1
    assert attempts == 1
    assert summary.failures[0]["attempt_count"] == 1
    assert summary.failures[0]["repair_attempted"] is False
    assert summary.failures[0]["truncation_suspected"] is True


def test_benchmark_sanitizes_wrapper_filenames_for_unsafe_tool_names(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "AdapterRemoval: remove adapter/sequences",
                "help_text": "Usage: AdapterRemoval --file1 R1.fastq",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_benchmark_generation(
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
    )
    records = json.loads(Path(summary.generation_records_path).read_text(encoding="utf-8"))
    output_path = Path(records[0]["output_xml_path"])

    assert summary.succeeded == 1
    assert output_path.exists()
    assert output_path.parent == tmp_path / "wrappers"
    assert output_path.name.startswith("AdapterRemoval_remove_adapter_sequences-")
