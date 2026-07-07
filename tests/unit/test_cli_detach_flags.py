from __future__ import annotations

from galaxy_toolsmith.cli.main import (
    _build_parser,
    _source_context_settings_from_args,
    _suite_generation_help_text_from_args,
)
from galaxy_toolsmith.inference.runtime_discovery import RuntimeDiscoveryResult


def test_serve_detach_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(["serve", "--detach", "--detach-log", "/tmp/serve.log"])
    assert args.detach is True
    assert args.detach_log == "/tmp/serve.log"


def test_serve_stop_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(["serve", "--stop", "--port", "8765", "--dry-run", "--force"])
    assert args.stop is True
    assert args.port == 8765
    assert args.dry_run is True
    assert args.force is True


def test_serve_stop_command_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["serve-stop", "--port", "8765", "--all-ports", "--timeout-seconds", "2"]
    )
    assert args.command == "serve-stop"
    assert args.port == 8765
    assert args.all_ports is True
    assert args.timeout_seconds == 2


def test_train_worker_detach_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-worker", "--detach", "--detach-log", "/tmp/worker.log"])
    assert args.detach is True
    assert args.detach_log == "/tmp/worker.log"


def test_train_backend_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train",
            "--backend",
            "axolotl",
            "--num-processes",
            "4",
            "--distributed-strategy",
            "fsdp",
            "--dry-run-backend",
            "--max-seq-length",
            "4096",
            "--max-steps",
            "5",
            "--no-pad-to-sequence-len",
            "--attn-implementation",
            "xformers",
            "--source-context-mode",
            "snippets",
            "--source-context-max-chars",
            "6000",
            "--source-context-max-files",
            "5",
            "--include-source-tests",
            "--test-context-max-chars",
            "1200",
            "--test-context-max-files",
            "3",
            "--test-context-max-file-bytes",
            "32KiB",
            "--source-root",
            "/tmp/source-root",
            "--source-file",
            "/tmp/source.py",
            "--per-device-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "2",
            "--learning-rate",
            "2e-5",
            "--training-method",
            "full",
            "--status-log",
            "/tmp/train.status.jsonl",
            "--status-interval-seconds",
            "15",
            "--stream-logs",
            "--log-tail-lines",
            "7",
        ]
    )
    assert args.backend == "axolotl"
    assert args.num_processes == 4
    assert args.distributed_strategy == "fsdp"
    assert args.dry_run_backend is True
    assert args.max_seq_length == 4096
    assert args.max_steps == 5
    assert args.pad_to_sequence_len is False
    assert args.attn_implementation == "xformers"
    assert args.source_context_mode == "snippets"
    assert args.source_context_max_chars == 6000
    assert args.source_context_max_files == 5
    assert args.include_source_tests is True
    assert args.test_context_mode == "none"
    assert args.test_context_max_chars == 1200
    assert args.test_context_max_files == 3
    assert args.test_context_max_file_bytes == 32 * 1024
    source_context = _source_context_settings_from_args(args)
    assert source_context.test_context_mode == "snippets"
    assert source_context.test_context_max_chars == 1200
    assert source_context.test_context_max_files == 3
    assert source_context.test_context_max_file_bytes == 32 * 1024
    assert args.source_root == "/tmp/source-root"
    assert args.source_file == "/tmp/source.py"
    assert args.per_device_batch_size == 1
    assert args.gradient_accumulation_steps == 2
    assert args.learning_rate == 2e-5
    assert args.training_method == "full"
    assert args.status_log == "/tmp/train.status.jsonl"
    assert args.status_interval_seconds == 15
    assert args.stream_logs is True
    assert args.log_tail_lines == 7


def test_generate_wrapper_source_archive_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-wrapper",
            "--tool-name",
            "my_tool",
            "--tool-id",
            "my_tool_safe_id",
            "--help-text-file",
            "help.txt",
            "--source-archive",
            "https://example.org/my_tool.tar.gz",
            "--source-archive-max-bytes",
            "1GiB",
            "--source-archive-timeout-seconds",
            "17",
            "--source-context-mode",
            "all-filtered",
            "--output",
            "my_tool.xml",
        ]
    )

    assert args.command == "generate-wrapper"
    assert args.tool_id == "my_tool_safe_id"
    assert args.source_archive == "https://example.org/my_tool.tar.gz"
    assert args.source_archive_max_bytes == 1024**3
    assert args.source_archive_timeout_seconds == 17
    assert args.source_context_mode == "all-filtered"


def test_generate_wrapper_logging_repair_and_sidecar_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-wrapper",
            "--tool-name",
            "my_tool",
            "--help-text-file",
            "help.txt",
            "--tool-granularity",
            "subcommands",
            "--stream-output",
            "--raw-response-log",
            "/tmp/raw.log",
            "--no-repair-invalid-xml",
            "--generate-sidecars",
            "--sidecar-output-dir",
            "/tmp/sidecars",
            "--output",
            "my_tool.xml",
        ]
    )

    assert args.tool_granularity == "subcommands"
    assert args.stream_output is True
    assert args.raw_response_log == "/tmp/raw.log"
    assert args.repair_invalid_xml is False
    assert args.generate_sidecars is True
    assert args.sidecar_output_dir == "/tmp/sidecars"
    assert args.include_toolsmith_citation is True
    assert args.datatype_scaffold is True


def test_generate_wrapper_postprocess_opt_out_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-wrapper",
            "--tool-name",
            "my tool",
            "--help-text-file",
            "help.txt",
            "--no-toolsmith-citation",
            "--no-datatype-scaffold",
            "--output",
            "my_tool.xml",
        ]
    )

    assert args.include_toolsmith_citation is False
    assert args.datatype_scaffold is False


def test_generate_wrapper_runtime_discovery_flags_parse_without_help_file() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-wrapper",
            "--tool-name",
            "minibwa",
            "--discovery-mode",
            "conda",
            "--discovery-package",
            "minibwa=0.2.0",
            "--discovery-command",
            "minibwa",
            "--discovery-source-download-max-bytes",
            "1GiB",
            "--source-context-mode",
            "snippets",
            "--output",
            "minibwa.xml",
        ]
    )

    assert args.help_text_file == ""
    assert args.discovery_mode == "conda"
    assert args.discovery_package == ["minibwa=0.2.0"]
    assert args.discovery_command == "minibwa"
    assert args.discovery_source_download_max_bytes == 1024**3
    assert args.discover_subcommands is True


def test_generate_suite_test_context_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-suite",
            "--tool-name",
            "minibwa",
            "--help-text-file",
            "help.txt",
            "--source-context-mode",
            "all-filtered",
            "--test-context-mode",
            "fixtures",
            "--test-context-max-chars",
            "2400",
            "--test-context-max-files",
            "8",
            "--test-context-max-file-bytes",
            "64KB",
            "--output-dir",
            "/tmp/minibwa-suite",
        ]
    )

    source_context = _source_context_settings_from_args(args)
    assert source_context.mode == "all-filtered"
    assert source_context.test_context_mode == "fixtures"
    assert source_context.test_context_max_chars == 2400
    assert source_context.test_context_max_files == 8
    assert source_context.test_context_max_file_bytes == 64_000


def test_generate_wrapper_source_archive_conflicts_with_source_root() -> None:
    parser = _build_parser()
    try:
        parser.parse_args(
            [
                "generate-wrapper",
                "--tool-name",
                "my_tool",
                "--help-text-file",
                "help.txt",
                "--source-root",
                "/tmp/source",
                "--source-archive",
                "source.tar.gz",
                "--output",
                "my_tool.xml",
            ]
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("--source-root and --source-archive should conflict")


def test_estimate_training_tokens_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "estimate-training-tokens",
            "--profile",
            "agentic-devstral-24b",
            "--artifact-format",
            "mixed",
            "--max-seq-lengths",
            "12k,16k",
            "--source-context-mode",
            "all-filtered",
            "--compare-source-context-modes",
            "all-filtered,all-raw",
            "--source-context-budget-ladder",
            "--limit",
            "10",
            "--exact-tokenizer",
            "--workers",
            "4",
            "--longest-sample-count",
            "7",
        ]
    )

    assert args.command == "estimate-training-tokens"
    assert args.profile == "agentic-devstral-24b"
    assert args.artifact_format == "mixed"
    assert args.max_seq_lengths == "12k,16k"
    assert args.source_context_mode == "all-filtered"
    assert args.compare_source_context_modes == "all-filtered,all-raw"
    assert args.source_context_budget_ladder is True
    assert args.limit == 10
    assert args.exact_tokenizer is True
    assert args.workers == 4
    assert args.longest_sample_count == 7


def test_train_mlx_backend_aliases_parse() -> None:
    parser = _build_parser()
    for backend in ("mlx-lm", "mlx", "mps"):
        args = parser.parse_args(["train", "--backend", backend, "--dry-run-backend"])
        assert args.backend == backend
        assert args.dry_run_backend is True


def test_convert_adapter_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "convert-adapter",
            "--from",
            "mlx",
            "--to",
            "peft",
            "--base-model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--adapter-dir",
            "/tmp/mlx",
            "--output-dir",
            "/tmp/peft",
        ]
    )

    assert args.command == "convert-adapter"
    assert args.from_format == "mlx"
    assert args.to_format == "peft"
    assert args.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert args.adapter_dir == "/tmp/mlx"
    assert args.output_dir == "/tmp/peft"


def test_train_runs_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(["train-runs", "--limit", "10", "--status-log", "/tmp/runs.jsonl"])
    assert args.limit == 10
    assert args.status_log == "/tmp/runs.jsonl"


def test_train_status_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train-status",
            "--run-id",
            "train-abc",
            "--tail",
            "40",
            "--status-log",
            "/tmp/status.jsonl",
        ]
    )
    assert args.run_id == "train-abc"
    assert args.tail == 40
    assert args.status_log == "/tmp/status.jsonl"


def test_benchmark_startup_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "benchmark-generate",
            "--gpu-devices",
            "0,1,2,3",
            "--min-items-per-process",
            "5",
            "--startup-stagger-seconds",
            "0.5",
            "--local-gpu-topology",
            "model-parallel",
            "--local-offload-policy",
            "fail",
            "--local-gpu-memory-reserve-gib",
            "3.5",
            "--resume-existing",
            "--checkpoint-records",
            "/tmp/checkpoint.jsonl",
            "--record-timeout-seconds",
            "42",
            "--ollama-context-tokens",
            "16384",
            "--source-context-mode",
            "metadata",
            "--source-context-max-chars",
            "3000",
            "--source-context-max-files",
            "4",
            "--source-root",
            "/tmp/benchmark-source",
        ]
    )

    assert args.num_processes == 0
    assert args.gpu_devices == "0,1,2,3"
    assert args.min_items_per_process == 5
    assert args.startup_stagger_seconds == 0.5
    assert args.local_gpu_topology == "model-parallel"
    assert args.local_offload_policy == "fail"
    assert args.local_gpu_memory_reserve_gib == 3.5
    assert args.resume_existing is True
    assert args.checkpoint_records == "/tmp/checkpoint.jsonl"
    assert args.record_timeout_seconds == 42
    assert args.ollama_context_tokens == 16384
    assert args.source_context_mode == "metadata"
    assert args.source_context_max_chars == 3000
    assert args.source_context_max_files == 4
    assert args.source_root == "/tmp/benchmark-source"


def test_generate_wrapper_source_context_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-wrapper",
            "--tool-name",
            "tool",
            "--help-text-file",
            "/tmp/help.txt",
            "--source-file",
            "/tmp/tool.py",
            "--source-root",
            "/tmp/source",
            "--source-context-mode",
            "all-filtered",
            "--source-context-max-chars",
            "7000",
            "--source-context-max-files",
            "8",
            "--ollama-context-tokens",
            "12288",
            "--output",
            "/tmp/tool.xml",
        ]
    )

    assert args.source_file == "/tmp/tool.py"
    assert args.source_root == "/tmp/source"
    assert args.source_context_mode == "all-filtered"
    assert args.source_context_max_chars == 7000
    assert args.source_context_max_files == 8
    assert args.ollama_context_tokens == 12288


def test_generate_wrapper_repository_output_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-wrapper",
            "--tool-name",
            "tool",
            "--help-text-file",
            "/tmp/help.txt",
            "--repository-output-dir",
            "/tmp/repo",
            "--shed-owner",
            "iuc",
            "--shed-category",
            "Sequence Analysis",
        ]
    )

    assert args.repository_output_dir == "/tmp/repo"
    assert args.output == ""
    assert args.shed_owner == "iuc"
    assert args.shed_category == ["Sequence Analysis"]


def test_generate_suite_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-suite",
            "--tool-name",
            "samtools",
            "--help-text-file",
            "/tmp/help.txt",
            "--output-dir",
            "/tmp/repo",
            "--max-suite-tools",
            "3",
            "--ollama-context-tokens",
            "32768",
            "--local-offload-policy",
            "fail",
            "--local-gpu-memory-reserve-gib",
            "3.5",
            "--raw-response-logs",
            "--shed-owner",
            "iuc",
        ]
    )

    assert args.command == "generate-suite"
    assert args.max_suite_tools == 3
    assert args.ollama_context_tokens == 32768
    assert args.local_offload_policy == "fail"
    assert args.local_gpu_memory_reserve_gib == 3.5
    assert args.raw_response_logs is True
    assert args.shed_owner == "iuc"


def test_compare_generation_runs_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "compare-generation-runs",
            "--left-run-dir",
            "/tmp/q4",
            "--right-run-dir",
            "/tmp/full",
            "--output",
            "/tmp/comparison.json",
        ]
    )

    assert args.command == "compare-generation-runs"
    assert args.left_run_dir == "/tmp/q4"
    assert args.right_run_dir == "/tmp/full"
    assert args.output == "/tmp/comparison.json"


def test_generate_suite_runtime_discovery_flags_parse_without_help_file() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-suite",
            "--tool-name",
            "samtools",
            "--discovery-mode",
            "auto",
            "--discovery-package",
            "samtools",
            "--discovery-container-runtime",
            "singularity",
            "--no-discover-subcommands",
            "--output-dir",
            "/tmp/repo",
        ]
    )

    assert args.command == "generate-suite"
    assert args.help_text_file == ""
    assert args.discovery_mode == "auto"
    assert args.discovery_package == ["samtools"]
    assert args.discovery_container_runtime == "singularity"
    assert args.discover_subcommands is False


def test_generate_suite_runtime_discovery_uses_compact_top_level_help() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "generate-suite",
            "--tool-name",
            "minibwa",
            "--discovery-mode",
            "conda",
            "--output-dir",
            "/tmp/repo",
        ]
    )
    discovery = RuntimeDiscoveryResult(
        mode="conda",
        selected_runtime="conda",
        top_level_help="$ minibwa --help\nUsage: minibwa {index,map}",
        subcommand_help={"minibwa map": "$ minibwa map --help\nLOTS OF MAP HELP"},
        combined_help_text=(
            "Runtime-discovered top-level command help:\n\n"
            "$ minibwa --help\nUsage: minibwa {index,map}\n\n"
            "Runtime-discovered subcommand help for `minibwa map`:\n\n"
            "$ minibwa map --help\nLOTS OF MAP HELP"
        ),
    )

    help_text = _suite_generation_help_text_from_args(args, discovery)

    assert "Usage: minibwa {index,map}" in help_text
    assert "LOTS OF MAP HELP" not in help_text


def test_benchmark_suite_generation_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "benchmark-generate",
            "--suite-generation",
            "generate",
            "--max-suite-tools",
            "5",
        ]
    )

    assert args.suite_generation == "generate"
    assert args.max_suite_tools == 5


def test_extract_corpus_container_flags_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "extract-corpus",
            "--resolve-containers",
            "--execute-containers",
            "--container-runtime",
            "apptainer",
            "--container-cache-dir",
            "/tmp/gtsm-containers",
            "--container-help-probe-mode",
            "safe",
            "--source-workers",
            "7",
            "--container-prepare-workers",
            "2",
            "--container-probe-workers",
            "5",
            "--container-image-timeout-seconds",
            "900",
            "--container-image-quarantine-seconds",
            "3600",
            "--source-download-timeout-seconds",
            "45",
            "--source-download-max-bytes",
            "1048576",
            "--singularity-depot-url",
            "https://example.org/singularity",
            "--status-log",
            "/tmp/extract.status.jsonl",
            "--docker-use-sudo",
            "--bioconda-checkout-sources",
            "--wrapper-source-max-bytes",
            "1234",
            "--wrapper-configfile-max-bytes",
            "5678",
            "--retry-manifest",
            "/tmp/retry.json",
            "--restart",
        ]
    )
    assert args.resolve_containers is True
    assert args.execute_containers is True
    assert args.container_runtime == "apptainer"
    assert args.container_cache_dir == "/tmp/gtsm-containers"
    assert args.container_help_probe_mode == "safe"
    assert args.source_workers == 7
    assert args.container_prepare_workers == 2
    assert args.container_probe_workers == 5
    assert args.container_image_timeout_seconds == 900
    assert args.container_image_quarantine_seconds == 3600
    assert args.source_download_timeout_seconds == 45
    assert args.source_download_max_bytes == 1048576
    assert args.singularity_depot_url == "https://example.org/singularity"
    assert args.status_log == "/tmp/extract.status.jsonl"
    assert args.docker_use_sudo is True
    assert args.bioconda_checkout_sources is True
    assert args.wrapper_source_max_bytes == 1234
    assert args.wrapper_configfile_max_bytes == 5678
    assert args.retry_manifest == "/tmp/retry.json"
    assert args.restart is True


def test_extract_corpus_source_download_max_bytes_human_sizes_parse() -> None:
    parser = _build_parser()
    args = parser.parse_args(["extract-corpus"])
    assert args.source_download_max_bytes == 0
    cases = {
        "0": 0,
        "1048576": 1048576,
        "1KB": 1000,
        "1KiB": 1024,
        "1.5MB": 1500000,
        "1gb": 1000**3,
        "1 GiB": 1024**3,
        "1GB": 1000**3,
        "1GiB": 1024**3,
    }
    for value, expected in cases.items():
        args = parser.parse_args(["extract-corpus", "--source-download-max-bytes", value])
        assert args.source_download_max_bytes == expected
