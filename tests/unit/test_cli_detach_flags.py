from __future__ import annotations

from galaxy_toolsmith.cli.main import _build_parser


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
            "--no-pad-to-sequence-len",
            "--attn-implementation",
            "xformers",
            "--source-context-mode",
            "snippets",
            "--source-context-max-chars",
            "6000",
            "--source-context-max-files",
            "5",
            "--source-root",
            "/tmp/source-root",
            "--source-file",
            "/tmp/source.py",
            "--per-device-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "2",
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
    assert args.pad_to_sequence_len is False
    assert args.attn_implementation == "xformers"
    assert args.source_context_mode == "snippets"
    assert args.source_context_max_chars == 6000
    assert args.source_context_max_files == 5
    assert args.source_root == "/tmp/source-root"
    assert args.source_file == "/tmp/source.py"
    assert args.per_device_batch_size == 1
    assert args.gradient_accumulation_steps == 2
    assert args.status_log == "/tmp/train.status.jsonl"
    assert args.status_interval_seconds == 15
    assert args.stream_logs is True
    assert args.log_tail_lines == 7


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
            "--output",
            "/tmp/tool.xml",
        ]
    )

    assert args.source_file == "/tmp/tool.py"
    assert args.source_root == "/tmp/source"
    assert args.source_context_mode == "all-filtered"
    assert args.source_context_max_chars == 7000
    assert args.source_context_max_files == 8


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
            "--restart",
        ]
    )
    assert args.resolve_containers is True
    assert args.execute_containers is True
    assert args.container_runtime == "apptainer"
    assert args.container_cache_dir == "/tmp/gtsm-containers"
    assert args.container_help_probe_mode == "safe"
    assert args.singularity_depot_url == "https://example.org/singularity"
    assert args.status_log == "/tmp/extract.status.jsonl"
    assert args.docker_use_sudo is True
    assert args.bioconda_checkout_sources is True
    assert args.wrapper_source_max_bytes == 1234
    assert args.wrapper_configfile_max_bytes == 5678
    assert args.restart is True
