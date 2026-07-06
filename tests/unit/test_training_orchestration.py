from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

import galaxy_toolsmith.orchestration.training as training_mod
import galaxy_toolsmith.orchestration.training_estimates as training_estimates_mod
from galaxy_toolsmith.core.manifests import TrainingRunManifest
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.source_context import source_context_settings
from galaxy_toolsmith.models.training import TrainingProfile, load_training_profile
from galaxy_toolsmith.orchestration.training import (
    TrainingProfileOverrides,
    _apply_training_profile_overrides,
    _axolotl_command,
    _axolotl_config,
    _axolotl_subprocess_environment,
    _build_sft_args,
    _build_sft_trainer,
    _corpus_container_help_counts,
    _deepspeed_zero3_config,
    _ensure_deepspeed_available,
    _fsdp_transformer_layer_cls,
    _hf_sft_distributed_launch_command,
    _load_instruction_records,
    _load_instruction_records_with_diagnostics,
    _mlx_lm_config,
    _model_load_kwargs,
    _read_new_log_lines,
    _record_training_help_text,
    _select_backend,
    _validate_training_method,
    _write_axolotl_runtime_compat,
    get_local_training_run,
    list_local_training_runs,
    run_training,
)
from galaxy_toolsmith.orchestration.training_estimates import (
    DEFAULT_SOURCE_BUDGET_LADDER,
    estimate_training_tokens,
    parse_context_lengths,
)
from galaxy_toolsmith.runtime.capabilities import RuntimeCapabilities
from galaxy_toolsmith.runtime.mlx_lm_lora import _merge_tokenizer_config


def _training_profile() -> TrainingProfile:
    return TrainingProfile(
        name="hf-test",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="hf-sft",
        skills_profile="default",
        default_command=[],
        max_seq_length=2048,
        epochs=2,
        per_device_batch_size=3,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        seed=1234,
    )


class _SourcePolicy:
    revision = ""
    cache_dir = ""
    local_files_only = False


class _CachedSourcePolicy:
    revision = "main"
    cache_dir = "/tmp/gtsm-model-cache"
    local_files_only = True

    def to_dict(self) -> dict:
        return {
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }


class _CudaAvailable:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True


class _TorchCuda:
    cuda = _CudaAvailable()
    bfloat16 = "bf16"
    float16 = "fp16"


def _runtime_capabilities(cuda: bool = True, mps: bool = False) -> RuntimeCapabilities:
    return RuntimeCapabilities(
        platform="Darwin" if mps else "Linux",
        machine="arm64" if mps else "x86_64",
        cpu_available=True,
        cuda_available=cuda,
        rocm_available=False,
        mps_available=mps,
        recommended_backend="cuda" if cuda else ("mps" if mps else "cpu"),
    )


def _write_trainable_corpus(tmp_path: Path) -> Path:
    wrapper_path = tmp_path / "wrapper.xml"
    wrapper_path.write_text(
        '<tool id="echo_tool" name="Echo Tool"><command>echo</command></tool>',
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "echo_tool",
                "expanded_xml_path": str(wrapper_path),
                "help_text": "old wrapper help",
                "container_help_text": "Usage: echo_tool --input FILE",
                "documentation": "source notes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return corpus_jsonl


def test_load_training_profile_defaults_training_method_for_old_configs(tmp_path: Path) -> None:
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "old-profile",
                        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                        "provider": "local",
                        "quantization": "none",
                        "backend": "axolotl",
                        "skills_profile": "default",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    profile = load_training_profile(profile_path, "old-profile")

    assert profile.training_method == "lora"


def test_record_training_help_text_includes_container_usage_once() -> None:
    help_text = _record_training_help_text(
        {
            "help_text": "Wrapper help",
            "container_help_text": "Usage: tool --input FILE",
            "container_usage_text": "$ tool\nUsage: tool <command>",
            "container_execution": [
                {
                    "status": "container-command-help",
                    "command": "tool view --help",
                    "probe_role": "subcommand",
                    "stdout": "Usage: tool view --input FILE",
                    "stderr": "",
                }
            ],
        }
    )

    assert "Wrapper help" in help_text
    assert "Command-line help collected from container execution" in help_text
    assert "Usage: tool --input FILE" in help_text
    assert "Command-line usage collected from container execution" in help_text
    assert "$ tool" in help_text
    assert "Structured runtime help probe commands" in help_text
    assert "Command: tool view --help" in help_text
    assert "Probe role: subcommand" in help_text

    deduped = _record_training_help_text(
        {
            "help_text": "Wrapper help\n\nUsage: tool --input FILE",
            "container_help_text": "Usage: tool --input FILE",
            "container_usage_text": "",
        }
    )

    assert deduped.count("Usage: tool --input FILE") == 1


def test_corpus_container_help_counts_include_runtime_and_configfile_fields(
    tmp_path: Path,
) -> None:
    corpus_jsonl = tmp_path / "corpus.jsonl"
    records = [
        {
            "container_help_text": "Usage: alpha",
            "container_usage_text": "$ alpha\nUsage: alpha",
            "container_api_validation": [{"status": "container-api-validation-ok"}],
            "wrapper_source_summary": {
                "api_backed_wrapper": True,
                "configfile_command_doc_count": 1,
            },
        },
        {
            "container_api_validation": [{"status": "container-api-validation-failed"}],
            "wrapper_source_summary": {"api_backed_wrapper": True},
        },
    ]
    corpus_jsonl.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    counts = _corpus_container_help_counts(corpus_jsonl)

    assert counts["corpus_records"] == 2
    assert counts["container_help_records"] == 1
    assert counts["container_usage_records"] == 1
    assert counts["container_api_validation_records"] == 2
    assert counts["container_api_validation_ok_records"] == 1
    assert counts["container_api_validation_failed_records"] == 1
    assert counts["api_backed_wrapper_records"] == 2
    assert counts["configfile_command_doc_records"] == 1


def test_model_load_kwargs_uses_bf16_for_non_quantized_cuda() -> None:
    profile = _training_profile()

    kwargs = _model_load_kwargs(
        profile=profile,
        source_policy=_SourcePolicy(),
        torch_module=_TorchCuda,
        bitsandbytes_config_cls=None,
    )

    assert kwargs["torch_dtype"] == "bf16"
    assert "quantization_config" not in kwargs
    assert "device_map" not in kwargs


def test_model_load_kwargs_uses_bitsandbytes_for_4bit_cuda() -> None:
    class FakeBitsAndBytesConfig:
        def __init__(
            self,
            load_in_4bit=False,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="",
            bnb_4bit_compute_dtype=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    profile = TrainingProfile(
        name="hf-test-4bit",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="hf-sft",
        skills_profile="default",
        default_command=[],
    )

    kwargs = _model_load_kwargs(
        profile=profile,
        source_policy=_SourcePolicy(),
        torch_module=_TorchCuda,
        bitsandbytes_config_cls=FakeBitsAndBytesConfig,
    )

    config = kwargs["quantization_config"]
    assert kwargs["torch_dtype"] == "bf16"
    assert kwargs["device_map"] == "auto"
    assert config.kwargs["load_in_4bit"] is True
    assert config.kwargs["bnb_4bit_quant_type"] == "nf4"
    assert config.kwargs["bnb_4bit_compute_dtype"] == "bf16"


def test_model_load_kwargs_uses_local_rank_for_distributed_4bit_cuda() -> None:
    class FakeBitsAndBytesConfig:
        def __init__(self, load_in_4bit=False, bnb_4bit_use_double_quant=False):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    profile = TrainingProfile(
        name="hf-test-4bit",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="hf-sft",
        skills_profile="default",
        default_command=[],
    )

    kwargs = _model_load_kwargs(
        profile=profile,
        source_policy=_SourcePolicy(),
        torch_module=_TorchCuda,
        bitsandbytes_config_cls=FakeBitsAndBytesConfig,
        distributed_world_size=4,
        local_rank=2,
    )

    assert kwargs["device_map"] == {"": 2}


def test_model_load_kwargs_includes_source_policy_cache_controls() -> None:
    profile = _training_profile()

    kwargs = _model_load_kwargs(
        profile=profile,
        source_policy=_CachedSourcePolicy(),
        torch_module=None,
        bitsandbytes_config_cls=None,
    )

    assert kwargs["revision"] == "main"
    assert kwargs["cache_dir"] == "/tmp/gtsm-model-cache"
    assert kwargs["local_files_only"] is True


def test_select_backend_auto_uses_axolotl_when_profile_requests_it() -> None:
    profile = _training_profile()
    profile = TrainingProfile(**{**profile.__dict__, "backend": "axolotl"})

    selection = _select_backend(
        profile=profile,
        command_override=None,
        capabilities=_runtime_capabilities(cuda=True),
    )

    assert selection.selected_backend == "axolotl"
    assert selection.fallback_reason == ""


def test_select_backend_auto_uses_mlx_when_profile_requests_it() -> None:
    profile = TrainingProfile(
        name="mps-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
    )

    selection = _select_backend(
        profile=profile,
        command_override=None,
        capabilities=_runtime_capabilities(cuda=False, mps=True),
    )

    assert selection.intended_backend == "mlx-lm"
    assert selection.selected_backend == "mlx-lm"
    assert selection.fallback_reason == ""
    assert selection.intended_methodology_supported is True


def test_load_instruction_records_includes_container_help_text(tmp_path: Path) -> None:
    corpus_jsonl = _write_trainable_corpus(tmp_path)

    records = _load_instruction_records(corpus_jsonl, _training_profile())

    assert len(records) == 1
    assert "Usage: echo_tool --input FILE" in records[0]["instruction"]
    assert "old wrapper help" in records[0]["instruction"]
    assert records[0]["output"].startswith('<tool id="echo_tool"')


def test_load_instruction_records_falls_back_to_wrapper_when_expanded_missing(
    tmp_path: Path,
) -> None:
    missing_expanded = tmp_path / "missing.expanded.xml"
    wrapper_path = tmp_path / "wrapper.xml"
    wrapper_path.write_text(
        '<tool id="fallback_tool" name="Fallback"><command>fallback</command></tool>',
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "fallback_tool",
                "expanded_xml_path": str(missing_expanded),
                "wrapper_path": str(wrapper_path),
                "help_text": "fallback help",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl,
        _training_profile(),
        repo_root=tmp_path,
    )

    assert len(records) == 1
    assert records[0]["output"].startswith('<tool id="fallback_tool"')
    assert diagnostics["trainable_samples"] == 1
    assert diagnostics["target_source_counts"] == {"wrapper": 1}
    assert diagnostics["missing_xml_target_count"] == 0


def test_load_instruction_records_rejects_non_tool_xml_primary_targets(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "tool_data_table_conf.xml"
    table_path.write_text(
        "<tables><table name='ref'><columns>value, label, path</columns></table></tables>",
        encoding="utf-8",
    )
    wrapper_path = tmp_path / "wrapper.xml"
    wrapper_path.write_text(
        '<tool id="real_tool" name="Real"><command>real</command></tool>',
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "real_tool",
                "expanded_xml_path": str(table_path),
                "wrapper_path": str(wrapper_path),
                "help_text": "real help",
                "shed_name": "real_tool",
                "shed_owner": "iuc",
                "shed_description": "Real tool repository",
                "shed_categories": ["Sequence Analysis"],
                "is_suite_root": True,
                "suite_members": ["real_tool", "real_tool_extra"],
                "wrapper_sidecar_files": [
                    {
                        "relative_path": "tool_data_table_conf.xml",
                        "role": "tool_data_table_conf",
                        "root_tag": "tables",
                        "byte_count": 80,
                        "content": table_path.read_text(encoding="utf-8"),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl,
        _training_profile(),
        repo_root=tmp_path,
    )

    assert len(records) == 1
    assert records[0]["output"].startswith('<tool id="real_tool"')
    assert records[0]["output"].lstrip().startswith("<tables") is False
    assert "Tool Shed repository metadata" in records[0]["instruction"]
    assert "name: real_tool" in records[0]["instruction"]
    assert "Companion Galaxy sidecar artifacts:" in records[0]["instruction"]
    assert "tool_data_table_conf.xml" in records[0]["instruction"]
    assert "Related suite tools:" in records[0]["instruction"]
    assert diagnostics["skipped_non_tool_xml_target_count"] == 1
    assert diagnostics["records_with_shed_metadata"] == 1
    assert diagnostics["records_with_suite_metadata"] == 1
    assert diagnostics["records_with_repository_sidecars"] == 1
    assert diagnostics["target_source_counts"] == {"wrapper": 1}


def test_load_instruction_records_rebases_stale_absolute_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "galaxy-toolsmith"
    expanded_path = repo_root / ".gtsm-cache" / "datasets" / "expanded" / "pkg" / "tool.xml"
    expanded_path.parent.mkdir(parents=True)
    expanded_path.write_text(
        '<tool id="rebased_tool" name="Rebased"><command>rebased</command></tool>',
        encoding="utf-8",
    )
    stale_path = Path("/old/home/blanked2/git/galaxy-toolsmith") / expanded_path.relative_to(
        repo_root
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "rebased_tool",
                "expanded_xml_path": str(stale_path),
                "help_text": "rebased help",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl,
        _training_profile(),
        repo_root=repo_root,
    )

    assert len(records) == 1
    assert records[0]["output"].startswith('<tool id="rebased_tool"')
    assert diagnostics["target_source_counts"] == {"expanded_rebased": 1}
    assert diagnostics["missing_xml_target_count"] == 0


def test_load_instruction_records_can_include_source_context(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "src" / "sourcetool").mkdir(parents=True)
    (source_root / "tests").mkdir()
    (source_root / "src" / "sourcetool" / "cli.py").write_text(
        """
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(prog="sourcetool")
    parser.add_argument("--input")
""".strip(),
        encoding="utf-8",
    )
    (source_root / "tests" / "test_cli.py").write_text(
        "SOURCE_CONTEXT_TEST_SHOULD_NOT_APPEAR = True\n",
        encoding="utf-8",
    )
    wrapper_path = tmp_path / "wrapper.xml"
    wrapper_path.write_text(
        '<tool id="sourcetool" name="Source Tool"><command>sourcetool</command></tool>',
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "sourcetool",
                "wrapper_path": str(wrapper_path),
                "help_text": "Usage: sourcetool --input reads.fq",
                "bioconda_sources": [
                    {
                        "package": "sourcetool",
                        "source_url": "https://example.org/sourcetool.tar.gz",
                        "source_checkout": str(source_root),
                        "command_hints": ["sourcetool"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl,
        _training_profile(),
        repo_root=tmp_path,
        source_context_settings=source_context_settings(
            mode="snippets",
            max_chars=5000,
            max_files=3,
        ),
    )

    assert len(records) == 1
    prompt = records[0]["instruction"]
    assert "Source metadata:" in prompt
    assert "Source file: src/sourcetool/cli.py" in prompt
    assert "ArgumentParser" in prompt
    assert "SOURCE_CONTEXT_TEST_SHOULD_NOT_APPEAR" not in prompt
    assert diagnostics["source_context_mode"] == "snippets"
    assert diagnostics["source_context_records"] == 1
    assert diagnostics["source_context_files"] >= 1
    assert diagnostics["source_context_chars"] > 0


def test_load_instruction_records_uses_training_data_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_records = []
    for index in range(2):
        wrapper_path = tmp_path / f"wrapper-{index}.xml"
        wrapper_path.write_text(
            (
                f'<tool id="worker_tool_{index}" name="Worker {index}">'
                f"<command>worker-{index}</command></tool>"
            ),
            encoding="utf-8",
        )
        corpus_records.append(
            {
                "tool_name": f"worker_tool_{index}",
                "wrapper_path": str(wrapper_path),
                "help_text": f"worker help {index}",
            }
        )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        "\n".join(json.dumps(record) for record in corpus_records) + "\n",
        encoding="utf-8",
    )
    progress: list[dict[str, object]] = []
    monkeypatch.setenv("GTSM_TRAIN_DATA_WORKERS", "2")

    records, diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl,
        _training_profile(),
        repo_root=tmp_path,
        progress_callback=progress.append,
    )

    assert [record["output"] for record in records] == [
        '<tool id="worker_tool_0" name="Worker 0"><command>worker-0</command></tool>',
        '<tool id="worker_tool_1" name="Worker 1"><command>worker-1</command></tool>',
    ]
    assert diagnostics["training_data_workers"] == 2
    assert diagnostics["target_source_counts"] == {"wrapper": 2}
    assert any(item.get("worker_count") == 2 for item in progress)


def test_load_instruction_records_supports_udt_yaml_targets(tmp_path: Path) -> None:
    udt_path = tmp_path / "tool.yml"
    udt_path.write_text(
        """
class: GalaxyUserTool
id: echo_tool
version: "0.1.0"
name: Echo Tool
container: busybox
shell_command: echo '$(inputs.text)' > output.txt
inputs:
  - name: text
    type: text
outputs:
  - name: output
    type: data
    format: txt
    from_work_dir: output.txt
""".strip(),
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "echo_tool",
                "udt_yaml_path": str(udt_path),
                "help_text": "echo help",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl,
        _training_profile(),
        repo_root=tmp_path,
        artifact_format="udt-yaml",
    )

    assert len(records) == 1
    assert "Galaxy User-Defined Tool YAML" in records[0]["instruction"]
    assert records[0]["output"].startswith("class: GalaxyUserTool")
    assert diagnostics["artifact_format"] == "udt_yaml"
    assert diagnostics["target_source_counts"] == {"udt_yaml_generate:udt_yaml": 1}


def test_load_instruction_records_mixed_adds_conversion_pair(tmp_path: Path) -> None:
    xml_path = tmp_path / "tool.xml"
    xml_path.write_text(
        '<tool id="echo_tool" name="Echo Tool"><command>echo</command></tool>',
        encoding="utf-8",
    )
    udt_path = tmp_path / "tool.yml"
    udt_path.write_text(
        """
class: GalaxyUserTool
id: echo_tool
version: "0.1.0"
name: Echo Tool
container: busybox
shell_command: echo hi > output.txt
outputs:
  - name: output
    type: data
    format: txt
    from_work_dir: output.txt
""".strip(),
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "echo_tool",
                "expanded_xml_path": str(xml_path),
                "udt_yaml_path": str(udt_path),
                "help_text": "echo help",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl,
        _training_profile(),
        repo_root=tmp_path,
        artifact_format="mixed",
    )

    assert len(records) == 3
    assert any(record["output"].startswith("<tool") for record in records)
    assert any(record["output"].startswith("class: GalaxyUserTool") for record in records)
    assert any(
        "Convert the following Galaxy User-Defined Tool YAML" in record["instruction"]
        for record in records
    )
    assert diagnostics["target_source_counts"] == {
        "expanded": 1,
        "udt_yaml_generate:udt_yaml": 1,
        "udt_to_xml:paired": 1,
    }


def test_estimate_training_tokens_reports_mixed_tasks_and_thresholds(tmp_path: Path) -> None:
    xml_path = tmp_path / "tool.xml"
    xml_path.write_text(
        '<tool id="echo_tool" name="Echo Tool"><command>echo</command></tool>',
        encoding="utf-8",
    )
    udt_path = tmp_path / "tool.yml"
    udt_path.write_text(
        """
class: GalaxyUserTool
id: echo_tool
version: "0.1.0"
name: Echo Tool
container: busybox
shell_command: echo hi > output.txt
outputs:
  - name: output
    type: data
    format: txt
    from_work_dir: output.txt
""".strip(),
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "echo_tool",
                "expanded_xml_path": str(xml_path),
                "udt_yaml_path": str(udt_path),
                "help_text": "echo help",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = estimate_training_tokens(
        profile=_training_profile(),
        corpus_jsonl_path=corpus_jsonl,
        repo_root=tmp_path,
        artifact_format="mixed",
        max_seq_lengths=(16, 4096),
        chars_per_token=1.0,
    )

    estimate = result["estimates"][0]
    assert estimate["samples"] == 3
    assert estimate["by_task"] == {
        "udt_to_xml": 1,
        "udt_yaml_generate": 1,
        "xml_generate": 1,
    }
    assert estimate["thresholds"][0]["max_seq_length"] == 16
    assert estimate["thresholds"][0]["over_max_seq_length"] > 0
    assert estimate["thresholds"][1]["max_seq_length"] == 4096
    assert estimate["thresholds"][1]["over_max_seq_length"] == 0
    assert result["recommendation"]["max_seq_length"] == 4096


def test_estimate_training_tokens_builds_source_variants_once_per_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "cli.py").write_text("from argparse import ArgumentParser\n", encoding="utf-8")
    xml_path = tmp_path / "tool.xml"
    xml_path.write_text(
        '<tool id="echo_tool" name="Echo Tool"><command>echo</command></tool>',
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        json.dumps(
            {
                "tool_name": "echo_tool",
                "expanded_xml_path": str(xml_path),
                "help_text": "echo help",
                "bioconda_sources": [{"source_checkout": str(source_root), "package": "echo"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[int] = []
    original = training_estimates_mod.build_source_context_variants_from_record

    def wrapped(record, settings):
        calls.append(len(settings))
        return original(record, settings)

    monkeypatch.setattr(
        training_estimates_mod,
        "build_source_context_variants_from_record",
        wrapped,
    )

    result = estimate_training_tokens(
        profile=_training_profile(),
        corpus_jsonl_path=corpus_jsonl,
        repo_root=tmp_path,
        artifact_format="xml",
        source_context_settings=source_context_settings(mode="all-filtered"),
        source_context_modes=("all-filtered",),
        max_seq_lengths=(4096, 8192),
        source_context_budget_ladder=True,
        chars_per_token=1.0,
        workers=1,
    )

    assert calls == [2]
    assert len(result["estimates"]) == 2


def test_estimate_training_tokens_parallel_matches_serial(tmp_path: Path) -> None:
    xml_path = tmp_path / "tool.xml"
    xml_path.write_text(
        '<tool id="echo_tool" name="Echo Tool"><command>echo</command></tool>',
        encoding="utf-8",
    )
    corpus_jsonl = tmp_path / "corpus.jsonl"
    corpus_jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "tool_name": f"echo_tool_{index}",
                    "expanded_xml_path": str(xml_path),
                    "help_text": "echo help",
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    serial = estimate_training_tokens(
        profile=_training_profile(),
        corpus_jsonl_path=corpus_jsonl,
        repo_root=tmp_path,
        artifact_format="xml",
        max_seq_lengths=(4096, 8192),
        chars_per_token=1.0,
        workers=1,
    )
    parallel = estimate_training_tokens(
        profile=_training_profile(),
        corpus_jsonl_path=corpus_jsonl,
        repo_root=tmp_path,
        artifact_format="xml",
        max_seq_lengths=(4096, 8192),
        chars_per_token=1.0,
        workers=2,
    )

    assert serial["estimates"] == parallel["estimates"]
    assert serial["recommendation"] == parallel["recommendation"]


def test_parse_context_lengths_accepts_k_suffix() -> None:
    assert parse_context_lengths("12k,16384,24k") == (12288, 16384, 24576)


def test_source_budget_ladder_includes_low_context_fallbacks() -> None:
    assert DEFAULT_SOURCE_BUDGET_LADDER[8192] == (12000, 64)
    assert DEFAULT_SOURCE_BUDGET_LADDER[4096] == (6000, 32)
    assert DEFAULT_SOURCE_BUDGET_LADDER[2048] == (3000, 16)


def test_axolotl_command_adds_accelerate_processes(tmp_path: Path) -> None:
    command = _axolotl_command(tmp_path / "axolotl.yml", num_processes=4)

    assert command == [
        "axolotl",
        "train",
        str(tmp_path / "axolotl.yml"),
        "--launcher",
        "accelerate",
        "--",
        "--num_processes",
        "4",
    ]


def test_axolotl_config_uses_qlora_for_4bit(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="axolotl-test",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )

    config = _axolotl_config(
        profile=profile,
        train_jsonl_path=tmp_path / "train.jsonl",
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "output",
        source_policy=_CachedSourcePolicy(),
    )

    assert config["adapter"] == "qlora"
    assert config["pad_to_sequence_len"] is True
    assert config["load_in_4bit"] is True
    assert config["bnb_4bit_quant_type"] == "nf4"
    assert config["cache_dir"] == "/tmp/gtsm-model-cache"
    assert config["datasets"][0]["type"] == "alpaca"


def test_axolotl_config_uses_lora_for_non_quantized_override(tmp_path: Path) -> None:
    profile = _apply_training_profile_overrides(
        TrainingProfile(
            name="axolotl-test",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            provider="local",
            quantization="none",
            backend="axolotl",
            skills_profile="default",
            default_command=[],
            max_seq_length=8192,
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
        ),
        TrainingProfileOverrides(
            max_seq_length=4096,
            pad_to_sequence_len=False,
            attn_implementation="xformers",
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
        ),
    )

    config = _axolotl_config(
        profile=profile,
        train_jsonl_path=tmp_path / "train.jsonl",
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "output",
    )

    assert config["adapter"] == "lora"
    assert config["sequence_len"] == 4096
    assert config["pad_to_sequence_len"] is False
    assert config["attn_implementation"] == "xformers"
    assert config["micro_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 2
    assert "load_in_4bit" not in config
    assert "bnb_4bit_quant_type" not in config


def test_axolotl_config_omits_adapter_for_full_training(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="full-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        training_method="full",
        learning_rate=2e-5,
    )

    config = _axolotl_config(
        profile=profile,
        train_jsonl_path=tmp_path / "train.jsonl",
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "output",
    )

    assert "adapter" not in config
    assert "lora_r" not in config
    assert "lora_target_linear" not in config
    assert config["learning_rate"] == 2e-5


def test_training_method_validation_rejects_invalid_combinations() -> None:
    full_quantized = TrainingProfile(
        name="bad-full",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        training_method="full",
    )
    qlora_mlx = TrainingProfile(
        name="bad-mlx",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        training_method="qlora",
    )

    with pytest.raises(ValueError, match="full requires quantization=none"):
        _validate_training_method(full_quantized, "axolotl")
    with pytest.raises(ValueError, match="not supported by the mlx-lm"):
        _validate_training_method(qlora_mlx, "mlx-lm")


def test_fsdp_transformer_layer_inference_maps_common_families() -> None:
    assert (
        _fsdp_transformer_layer_cls(
            TrainingProfile(
                name="agentic-devstral-24b",
                base_model="mistralai/Devstral-Small-2505",
                provider="local",
                quantization="none",
                backend="axolotl",
                skills_profile="default",
                default_command=[],
            )
        )
        == "MistralDecoderLayer"
    )
    assert (
        _fsdp_transformer_layer_cls(
            TrainingProfile(
                name="proto-qwen25-7b",
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                provider="local",
                quantization="none",
                backend="axolotl",
                skills_profile="default",
                default_command=[],
            )
        )
        == "Qwen2DecoderLayer"
    )


def test_axolotl_config_adds_fsdp_strategy(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="agentic-devstral-24b",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )

    config = _axolotl_config(
        profile=profile,
        train_jsonl_path=tmp_path / "train.jsonl",
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "output",
        distributed_strategy="fsdp",
    )

    assert config["fsdp"] == ["full_shard", "auto_wrap"]
    assert config["fsdp_config"]["auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert config["fsdp_config"]["transformer_layer_cls_to_wrap"] == "MistralDecoderLayer"
    assert config["fsdp_config"]["cpu_ram_efficient_loading"] is True
    assert config["fsdp_config"]["activation_checkpointing"] is True
    assert "gradient_checkpointing" not in config
    assert "deepspeed" not in config


def test_axolotl_config_adds_deepspeed_strategy(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="agentic-devstral-24b",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )
    deepspeed_config = tmp_path / "zero3.json"

    config = _axolotl_config(
        profile=profile,
        train_jsonl_path=tmp_path / "train.jsonl",
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "output",
        distributed_strategy="deepspeed-zero3",
        deepspeed_config_path=deepspeed_config,
    )

    assert config["deepspeed"] == str(deepspeed_config)
    assert "fsdp" not in config


def test_axolotl_config_adds_max_steps_for_probe_runs(tmp_path: Path) -> None:
    config = _axolotl_config(
        profile=_training_profile(),
        train_jsonl_path=tmp_path / "train.jsonl",
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "output",
        max_steps=5,
    )

    assert config["max_steps"] == 5


def test_deepspeed_zero3_config_can_enable_cpu_offload() -> None:
    config = _deepspeed_zero3_config(offload=True)

    assert config["zero_optimization"]["stage"] == 3
    assert config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True
    assert config["zero_optimization"]["stage3_param_persistence_threshold"] == 0
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert config["zero_optimization"]["offload_param"]["device"] == "cpu"


def test_axolotl_runtime_compat_injects_mistral_sitecustomize(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="agentic-devstral-24b",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )

    sitecustomize_path = _write_axolotl_runtime_compat(tmp_path / "run", profile)

    assert sitecustomize_path is not None
    assert sitecustomize_path.name == "sitecustomize.py"
    assert "save_jinja_files" in sitecustomize_path.read_text(encoding="utf-8")
    env = _axolotl_subprocess_environment(
        source_policy=_SourcePolicy(),
        compat_sitecustomize_path=sitecustomize_path,
    )
    assert env["PATH"].split(os.pathsep)[0] == str(Path(training_mod.sys.executable).parent)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(sitecustomize_path.parent)


def test_axolotl_runtime_compat_skips_non_mistral_profiles(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="proto-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )

    assert _write_axolotl_runtime_compat(tmp_path / "run", profile) is None


def test_deepspeed_preflight_fails_clearly_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(training_mod, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match="DeepSpeed is required"):
        _ensure_deepspeed_available("deepspeed-zero3")


def test_hf_sft_distributed_launch_command_uses_torchrun(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    command = _hf_sft_distributed_launch_command(
        paths=paths,
        profile=_training_profile(),
        dataset_manifest_path=tmp_path / "dataset.manifest.json",
        corpus_jsonl_path=tmp_path / "corpus.jsonl",
        run_id="train-fixed",
        num_processes=4,
        variant_id="variant-ddp",
        profile_overrides=TrainingProfileOverrides(
            max_seq_length=4096,
            pad_to_sequence_len=False,
            attn_implementation="xformers",
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            training_method="full",
        ),
        status_log_path=tmp_path / "train.status.jsonl",
        status_interval_seconds=15,
        stream_logs=True,
        log_tail_lines=7,
        max_steps=5,
    )

    assert command[:4] == [command[0], "-m", "torch.distributed.run", "--standalone"]
    assert "--nproc_per_node" in command
    assert "4" in command
    assert "--internal-run-id" in command
    assert "train-fixed" in command
    assert "--internal-distributed-child" in command
    assert "--variant-id" in command
    assert "--max-seq-length" in command
    assert "--no-pad-to-sequence-len" in command
    assert "--attn-implementation" in command
    assert "xformers" in command
    assert "--per-device-batch-size" in command
    assert "--gradient-accumulation-steps" in command
    assert "--learning-rate" in command
    assert "2e-05" in command
    assert "--training-method" in command
    assert "full" in command
    assert "--max-steps" in command
    assert "5" in command
    assert "--status-log" in command
    assert str(tmp_path / "train.status.jsonl") in command
    assert "--stream-logs" in command
    assert command[-2:] == ["--log-tail-lines", "7"]


def test_hf_sft_args_use_profile_overrides(tmp_path: Path) -> None:
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir,
            num_train_epochs,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            logging_steps,
            save_steps,
            max_steps,
            seed,
            report_to,
            dataset_text_field,
            max_length,
            gradient_checkpointing,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    profile = _apply_training_profile_overrides(
        _training_profile(),
        TrainingProfileOverrides(
            max_seq_length=4096,
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
        ),
    )

    args = _build_sft_args(
        training_arguments_cls=object,
        sft_config_cls=FakeSFTConfig,
        profile=profile,
        checkpoints_dir=tmp_path / "checkpoints",
        max_steps=5,
    )

    assert args.kwargs["max_length"] == 4096
    assert args.kwargs["per_device_train_batch_size"] == 1
    assert args.kwargs["gradient_accumulation_steps"] == 2
    assert args.kwargs["max_steps"] == 5


def test_build_sft_trainer_supports_legacy_trl_kwargs(tmp_path: Path) -> None:
    class FakeTrainingArguments:
        def __init__(
            self,
            output_dir,
            num_train_epochs,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            logging_steps,
            save_steps,
            max_steps,
            seed,
            report_to,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    class LegacySFTTrainer:
        def __init__(
            self,
            model,
            train_dataset,
            tokenizer,
            args,
            peft_config,
            dataset_text_field,
            max_seq_length,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    profile = _training_profile()
    args = _build_sft_args(
        training_arguments_cls=FakeTrainingArguments,
        sft_config_cls=None,
        profile=profile,
        checkpoints_dir=tmp_path / "checkpoints",
    )
    trainer = _build_sft_trainer(
        LegacySFTTrainer,
        model="model",
        train_dataset="dataset",
        tokenizer="tokenizer",
        args=args,
        peft_config="peft",
        profile=profile,
    )

    assert isinstance(args, FakeTrainingArguments)
    assert args.kwargs["output_dir"] == str(tmp_path / "checkpoints")
    assert args.kwargs["num_train_epochs"] == 2
    assert trainer.kwargs["tokenizer"] == "tokenizer"
    assert trainer.kwargs["dataset_text_field"] == "text"
    assert trainer.kwargs["max_seq_length"] == 2048


def test_build_sft_trainer_supports_current_trl_kwargs(tmp_path: Path) -> None:
    class FakeSFTConfig:
        def __init__(
            self,
            output_dir,
            num_train_epochs,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            logging_steps,
            save_steps,
            max_steps,
            seed,
            report_to,
            dataset_text_field,
            max_length,
            bf16,
            gradient_checkpointing,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    class CurrentSFTTrainer:
        def __init__(
            self,
            model,
            args,
            train_dataset=None,
            processing_class=None,
            peft_config=None,
        ):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    profile = _training_profile()
    args = _build_sft_args(
        training_arguments_cls=object,
        sft_config_cls=FakeSFTConfig,
        profile=profile,
        checkpoints_dir=tmp_path / "checkpoints",
        torch_module=_TorchCuda,
    )
    trainer = _build_sft_trainer(
        CurrentSFTTrainer,
        model="model",
        train_dataset="dataset",
        tokenizer="tokenizer",
        args=args,
        peft_config="peft",
        profile=profile,
    )

    assert isinstance(args, FakeSFTConfig)
    assert args.kwargs["dataset_text_field"] == "text"
    assert args.kwargs["max_length"] == 2048
    assert args.kwargs["bf16"] is True
    assert args.kwargs["gradient_checkpointing"] is True
    assert trainer.kwargs["processing_class"] == "tokenizer"
    assert trainer.kwargs["train_dataset"] == "dataset"
    assert "tokenizer" not in trainer.kwargs


def test_build_sft_trainer_omits_peft_config_for_full_training(tmp_path: Path) -> None:
    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FullSFTTrainer:
        def __init__(self, model, args, train_dataset=None, processing_class=None):
            self.kwargs = dict(locals())
            self.kwargs.pop("self")

    profile = _training_profile()
    args = _build_sft_args(
        training_arguments_cls=FakeTrainingArguments,
        sft_config_cls=None,
        profile=profile,
        checkpoints_dir=tmp_path / "checkpoints",
    )
    trainer = _build_sft_trainer(
        FullSFTTrainer,
        model="model",
        train_dataset="dataset",
        tokenizer="tokenizer",
        args=args,
        peft_config=None,
        profile=profile,
    )

    assert "peft_config" not in trainer.kwargs
    assert trainer.kwargs["processing_class"] == "tokenizer"


def test_run_training_command_backend_creates_variant_manifest(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()

    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_id": "dset-1"}), encoding="utf-8")

    profile = TrainingProfile(
        name="cmd-test",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )
    run = run_training(
        paths=paths,
        profile=profile,
        dataset_manifest_path=dataset_manifest,
        command_override=["echo", "ok"],
        variant_id="variant-cmd",
        corpus_jsonl_path=paths.datasets_root / "missing.jsonl",
    )

    assert run.status == "completed"
    assert run.model_variant_path
    assert Path(run.model_variant_path).exists()
    metrics_path = Path(run.metrics_path)
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "source_quantization" in metrics
    assert "intended_backend" in metrics
    assert "intended_methodology_supported" in metrics
    assert metrics["requested_training_method"] == "lora"
    assert metrics["effective_training_method"] == "qlora"
    assert metrics["artifact_kind"] == "unknown"
    assert "progress" in metrics
    assert metrics["progress"]["total_units"] == 1
    variant = json.loads(Path(run.model_variant_path).read_text(encoding="utf-8"))
    assert variant["training_method"] == "lora"
    assert variant["effective_training_method"] == "qlora"
    assert variant["artifact_kind"] == "unknown"


def test_run_training_axolotl_dry_run_writes_config_and_dataset(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = _write_trainable_corpus(tmp_path)

    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_id": "dset-1"}), encoding="utf-8")

    profile = TrainingProfile(
        name="axolotl-test",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )
    run = run_training(
        paths=paths,
        profile=profile,
        dataset_manifest_path=dataset_manifest,
        command_override=None,
        variant_id="variant-axolotl",
        corpus_jsonl_path=corpus_jsonl,
        backend_override="axolotl",
        num_processes=4,
        dry_run_backend=True,
    )

    assert run.status == "dry-run"
    assert run.model_variant_path == ""
    metrics = json.loads(Path(run.metrics_path).read_text(encoding="utf-8"))
    assert metrics["backend_impl"] == "axolotl"
    assert metrics["status"] == "dry-run"
    assert metrics["command"][-2:] == ["--num_processes", "4"]
    dataset_path = Path(metrics["dataset_path"])
    config_path = Path(metrics["config_path"])
    assert dataset_path.exists()
    assert config_path.exists()
    dataset_record = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    assert dataset_record["output"].startswith('<tool id="echo_tool"')
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["load_in_4bit"] is True
    assert config["datasets"][0]["path"] == str(dataset_path)
    assert config["cache_dir"] == str(paths.models_root / "hf-cache")
    assert metrics["model_source_policy"]["cache_dir"] == str(paths.models_root / "hf-cache")
    assert metrics["training_data_diagnostics"]["total_corpus_records"] == 1
    assert metrics["training_data_diagnostics"]["trainable_samples"] == 1
    assert metrics["training_data_diagnostics"]["target_source_counts"] == {"expanded": 1}


def test_run_training_mlx_dry_run_writes_config_and_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = _write_trainable_corpus(tmp_path)
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_id": "dset-1"}), encoding="utf-8")
    monkeypatch.setattr(
        training_mod,
        "detect_runtime_capabilities",
        lambda: _runtime_capabilities(cuda=False, mps=True),
    )
    profile = TrainingProfile(
        name="mps-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        max_seq_length=4096,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    )

    run = run_training(
        paths=paths,
        profile=profile,
        dataset_manifest_path=dataset_manifest,
        command_override=None,
        variant_id="variant-mlx",
        corpus_jsonl_path=corpus_jsonl,
        dry_run_backend=True,
    )

    assert run.status == "dry-run"
    metrics = json.loads(Path(run.metrics_path).read_text(encoding="utf-8"))
    assert metrics["backend_impl"] == "mlx-lm"
    assert metrics["intended_backend"] == "mlx-lm"
    assert metrics["intended_methodology_supported"] is True
    assert metrics["samples"] == 1
    assert metrics["command"][1:3] == ["-m", "galaxy_toolsmith.runtime.mlx_lm_lora"]
    train_record = json.loads(Path(metrics["train_jsonl_path"]).read_text(encoding="utf-8"))
    assert "Usage: echo_tool --input FILE" in train_record["prompt"]
    assert train_record["completion"].startswith('<tool id="echo_tool"')
    config = yaml.safe_load(Path(metrics["config_path"]).read_text(encoding="utf-8"))
    assert config["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config["train"] is True
    assert config["fine_tune_type"] == "lora"
    assert config["adapter_path"] == metrics["adapter_path"]
    assert config["data"] == metrics["data_dir"]
    assert config["max_seq_length"] == 4096
    assert config["batch_size"] == 1
    assert config["grad_accumulation_steps"] == 2
    assert config["mask_prompt"] is True
    assert config["lora_parameters"]["rank"] == 16
    assert "tokenizer_config" not in config


def test_mlx_lm_config_uses_finetuning_mode_for_mistral_common(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="mps-devstral-24b",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
    )

    config = _mlx_lm_config(
        profile=profile,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        sample_count=1,
    )

    assert config["mask_prompt"] is False
    assert config["tokenizer_config"] == {"mode": "finetuning"}


def test_mlx_lm_lora_wrapper_merges_tokenizer_config() -> None:
    merged = _merge_tokenizer_config(
        {"trust_remote_code": True, "mode": "test"},
        {"mode": "finetuning"},
    )

    assert merged == {"trust_remote_code": True, "mode": "finetuning"}


def test_mlx_lm_config_supports_full_training(tmp_path: Path) -> None:
    profile = TrainingProfile(
        name="mps-qwen25-7b-full",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        training_method="full",
    )

    config = _mlx_lm_config(
        profile=profile,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        sample_count=4,
    )

    assert config["fine_tune_type"] == "full"
    assert config["num_layers"] == 0
    assert "lora_parameters" not in config


def test_mlx_lm_config_uses_max_steps_for_probe_runs(tmp_path: Path) -> None:
    config = _mlx_lm_config(
        profile=_training_profile(),
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        sample_count=100,
        max_steps=7,
    )

    assert config["iters"] == 7


def test_run_training_axolotl_dry_run_applies_non_quantized_overrides(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = _write_trainable_corpus(tmp_path)

    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_id": "dset-1"}), encoding="utf-8")

    profile = TrainingProfile(
        name="proto-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
    )
    run = run_training(
        paths=paths,
        profile=profile,
        dataset_manifest_path=dataset_manifest,
        command_override=None,
        variant_id="variant-axolotl-nonquant",
        corpus_jsonl_path=corpus_jsonl,
        backend_override="axolotl",
        num_processes=4,
        dry_run_backend=True,
        profile_overrides=TrainingProfileOverrides(
            max_seq_length=4096,
            pad_to_sequence_len=False,
            attn_implementation="xformers",
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
        ),
    )

    assert run.status == "dry-run"
    metrics = json.loads(Path(run.metrics_path).read_text(encoding="utf-8"))
    assert metrics["source_quantization"] == "none"
    assert metrics["training_profile_overrides"] == {
        "max_seq_length": 4096,
        "pad_to_sequence_len": False,
        "attn_implementation": "xformers",
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
    }
    assert metrics["effective_training_profile"] == {
        "max_seq_length": 4096,
        "pad_to_sequence_len": False,
        "attn_implementation": "xformers",
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "distributed_strategy": "ddp",
    }
    config = yaml.safe_load(Path(metrics["config_path"]).read_text(encoding="utf-8"))
    assert config["adapter"] == "lora"
    assert config["sequence_len"] == 4096
    assert config["pad_to_sequence_len"] is False
    assert config["attn_implementation"] == "xformers"
    assert config["micro_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 2
    assert config["cache_dir"] == str(paths.models_root / "hf-cache")
    assert "load_in_4bit" not in config


def test_run_training_axolotl_dry_run_writes_deepspeed_config(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = _write_trainable_corpus(tmp_path)

    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_id": "dset-1"}), encoding="utf-8")

    profile = TrainingProfile(
        name="agentic-devstral-24b",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )
    run = run_training(
        paths=paths,
        profile=profile,
        dataset_manifest_path=dataset_manifest,
        command_override=None,
        variant_id="variant-axolotl-zero3",
        corpus_jsonl_path=corpus_jsonl,
        backend_override="axolotl",
        num_processes=4,
        dry_run_backend=True,
        distributed_strategy="deepspeed-zero3-offload",
    )

    metrics = json.loads(Path(run.metrics_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(metrics["config_path"]).read_text(encoding="utf-8"))
    deepspeed_config_path = Path(metrics["deepspeed_config_path"])
    deepspeed_config = json.loads(deepspeed_config_path.read_text(encoding="utf-8"))

    assert run.status == "dry-run"
    assert metrics["distributed_strategy"] == "deepspeed-zero3-offload"
    assert config["deepspeed"] == str(deepspeed_config_path)
    assert deepspeed_config["zero_optimization"]["stage"] == 3
    assert deepspeed_config["zero_optimization"]["stage3_param_persistence_threshold"] == 0
    assert deepspeed_config["zero_optimization"]["offload_param"]["device"] == "cpu"


def test_list_and_get_local_training_runs_include_progress_and_log_tails(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    run_dir = paths.runs_root / "training" / "train-local"
    run_dir.mkdir(parents=True)
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    stdout_log.write_text("a\nb\nc\n", encoding="utf-8")
    stderr_log.write_text("warn\nerr\n", encoding="utf-8")
    progress_log = run_dir / "progress.jsonl"
    progress_log.write_text(
        json.dumps({"completed_units": 1, "total_units": 3}) + "\n",
        encoding="utf-8",
    )
    manifest = TrainingRunManifest(
        run_id="train-local",
        profile_name="proto-qwen25-7b",
        backend="axolotl",
        status="running",
        metrics_path=str(run_dir / "metrics.json"),
    )
    (run_dir / "run.manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "running",
                "backend_impl": "axolotl",
                "pid": 999999999,
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
                "progress_log_path": str(progress_log),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    runs = list_local_training_runs(paths, limit=5)
    status = get_local_training_run(paths, "latest", tail_lines=2)

    assert runs["summary"]["total"] == 1
    assert runs["summary"]["running"] == 1
    assert runs["runs"][0]["run"]["run_id"] == "train-local"
    assert status["progress"]["completed_units"] == 1
    assert status["logs"]["stdout_tail"] == "b\nc"
    assert status["logs"]["stderr_tail"] == "warn\nerr"
    assert status["process"]["running"] is False


def test_list_local_training_runs_skips_archived_duplicate_run_dirs(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    run_id = "train-local"
    run_dir = paths.runs_root / "training" / run_id
    archived_dir = paths.runs_root / "training" / f"{run_id}.failed.20260618T221500Z"
    run_dir.mkdir(parents=True)
    archived_dir.mkdir(parents=True)
    manifest = TrainingRunManifest(
        run_id=run_id,
        profile_name="proto-qwen25-7b",
        backend="axolotl",
        status="running",
        metrics_path=str(run_dir / "metrics.json"),
    )
    archived_manifest = TrainingRunManifest(
        run_id=run_id,
        profile_name="proto-qwen25-7b",
        backend="axolotl",
        status="failed",
        metrics_path=str(archived_dir / "metrics.json"),
    )
    (run_dir / "run.manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    (archived_dir / "run.manifest.json").write_text(
        archived_manifest.to_json(),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text('{"status": "running"}', encoding="utf-8")
    (archived_dir / "metrics.json").write_text('{"status": "failed"}', encoding="utf-8")

    runs = list_local_training_runs(paths, limit=10)

    assert runs["summary"]["total"] == 1
    assert runs["summary"]["running"] == 1
    assert runs["summary"]["failed"] == 0
    assert [row["run"]["run_id"] for row in runs["runs"]] == [run_id]
    assert runs["runs"][0]["run_dir"] == str(run_dir)


def test_read_new_log_lines_zero_tail_advances_offset_without_chunk(tmp_path: Path) -> None:
    log_path = tmp_path / "stdout.log"
    log_path.write_text("started\nloading\n", encoding="utf-8")

    offset, chunk, last_line = _read_new_log_lines(log_path, offset=0, max_lines=0)

    assert offset == len("started\nloading\n")
    assert chunk == ""
    assert last_line == "loading"


def test_run_training_axolotl_managed_process_writes_live_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    corpus_jsonl = _write_trainable_corpus(tmp_path)
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_id": "dset-1"}), encoding="utf-8")

    class FakePopen:
        def __init__(self, command, text, stdout, stderr, cwd, env):
            self.command = command
            self.text = text
            self.stdout = stdout
            self.stderr = stderr
            self.cwd = cwd
            self.env = env
            self.pid = 4321
            self.poll_count = 0
            assert env["HF_HOME"] == str(paths.models_root / "hf-cache")
            assert env["PATH"].split(os.pathsep)[0] == str(
                Path(training_mod.sys.executable).parent
            )
            stdout.write("started\nloading\n")
            stderr.write("warning\n")
            stdout.flush()
            stderr.flush()

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                return None
            return 0

    monkeypatch.setattr(training_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        training_mod,
        "detect_runtime_capabilities",
        lambda: _runtime_capabilities(cuda=True),
    )
    status_log = tmp_path / "train.status.jsonl"
    profile = TrainingProfile(
        name="axolotl-test",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
    )

    run = run_training(
        paths=paths,
        profile=profile,
        dataset_manifest_path=dataset_manifest,
        command_override=None,
        variant_id="variant-managed",
        corpus_jsonl_path=corpus_jsonl,
        backend_override="axolotl",
        status_log_path=status_log,
        status_interval_seconds=0.1,
        stream_logs=True,
        log_tail_lines=1,
    )

    metrics = json.loads(Path(run.metrics_path).read_text(encoding="utf-8"))
    assert run.status == "completed"
    assert metrics["status"] == "completed"
    assert metrics["pid"] == 4321
    assert metrics["process_running"] is False
    assert metrics["last_stdout_line"] == "loading"
    assert metrics["last_stderr_line"] == "warning"
    assert Path(metrics["stdout_log_path"]).read_text(encoding="utf-8") == "started\nloading\n"
    assert Path(metrics["stderr_log_path"]).read_text(encoding="utf-8") == "warning\n"
    assert Path(metrics["config_path"]).exists()
    status_events = [
        json.loads(line) for line in status_log.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["status"] == "training-backend-started" for event in status_events)
    log_events = [event for event in status_events if event["status"] == "training-log"]
    assert log_events
    assert log_events[0]["stdout"] == "loading"
    progress_events = [event for event in status_events if event["status"] == "training-progress"]
    assert progress_events[-1]["last_stdout_line"] == "loading"
    assert progress_events[-1]["last_stderr_line"] == "warning"
    assert progress_events[-1]["process_running"] is False


def test_run_training_command_backend_failure_includes_stderr(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()

    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text(json.dumps({"dataset_id": "dset-1"}), encoding="utf-8")

    profile = TrainingProfile(
        name="cmd-test",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="hf-sft",
        skills_profile="default",
        default_command=[],
    )
    run = run_training(
        paths=paths,
        profile=profile,
        dataset_manifest_path=dataset_manifest,
        command_override=["sh", "-c", "echo trainer failed >&2; exit 7"],
        variant_id="variant-cmd",
        corpus_jsonl_path=paths.datasets_root / "missing.jsonl",
    )

    assert run.status == "failed"
    assert "code 7" in run.error
    assert "trainer failed" in run.error
    metrics = json.loads(Path(run.metrics_path).read_text(encoding="utf-8"))
    assert metrics["returncode"] == 7
    assert metrics["stderr"] == "trainer failed"
