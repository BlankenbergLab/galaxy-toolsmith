from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from galaxy_toolsmith.cache.sources import sync_galaxy_skills, sync_tools_iuc
from galaxy_toolsmith.cache.xsd import sync_galaxy_xsd
from galaxy_toolsmith.client.remote import (
    fetch_training_artifacts_parallel,
    request_remote_generation,
    request_remote_json,
)
from galaxy_toolsmith.core.config import write_default_config
from galaxy_toolsmith.core.manifests import DatasetManifest, ModelVariantManifest, SourceRef
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.data.corpus import (
    DEFAULT_WRAPPER_CONFIGFILE_MAX_BYTES,
    DEFAULT_WRAPPER_SOURCE_MAX_BYTES,
    ExtractionSettings,
    extract_tools_corpus,
    rebuild_execution_report_from_jsonl,
    write_corpus_diagnostics,
)
from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    format_cli_value,
    normalize_artifact_format,
    normalize_training_artifact_format,
)
from galaxy_toolsmith.inference.evaluation import evaluate_wrapper_paths
from galaxy_toolsmith.inference.generation import generate_wrapper
from galaxy_toolsmith.inference.postprocess import (
    datatype_scaffold_dir_for_output,
    write_datatype_scaffold,
)
from galaxy_toolsmith.inference.prompt_context import DEFAULT_MAX_PROMPT_HELP_CHARS
from galaxy_toolsmith.inference.repository import (
    build_tool_shed_metadata,
    safe_repository_name,
    safe_tool_id,
    write_gtsm_json,
    write_shed_yml,
)
from galaxy_toolsmith.inference.runtime_discovery import (
    DEFAULT_DISCOVERY_CHANNELS,
    DISCOVERY_MODES,
    RuntimeDiscoveryResult,
    RuntimeDiscoverySettings,
    discover_runtime_context,
    normalize_discovery_mode,
)
from galaxy_toolsmith.inference.source_archives import (
    DEFAULT_SOURCE_ARCHIVE_MAX_BYTES,
    DEFAULT_SOURCE_ARCHIVE_TIMEOUT_SECONDS,
    resolve_source_archive,
)
from galaxy_toolsmith.inference.source_context import (
    DEFAULT_SOURCE_CONTEXT_MAX_CHARS,
    DEFAULT_SOURCE_CONTEXT_MAX_FILES,
    DEFAULT_TEST_CONTEXT_MAX_CHARS,
    DEFAULT_TEST_CONTEXT_MAX_FILE_BYTES,
    DEFAULT_TEST_CONTEXT_MAX_FILES,
    SOURCE_CONTEXT_MODES,
    TEST_CONTEXT_MODES,
    source_context_settings,
)
from galaxy_toolsmith.inference.suite import (
    compare_generation_run_dirs,
    generate_suite,
    plan_suite_from_content,
)
from galaxy_toolsmith.inference.udt import udt_yaml_to_tool_xml, validate_udt_yaml
from galaxy_toolsmith.inference.validation import PlanemoTestOptions
from galaxy_toolsmith.models.training import (
    TRAINING_METHODS,
    load_training_profile,
    write_default_training_profiles,
)
from galaxy_toolsmith.orchestration.adapter_conversion import convert_mlx_lora_to_peft
from galaxy_toolsmith.orchestration.benchmark import (
    DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS,
    run_benchmark_generation,
    run_benchmark_generation_sharded,
)
from galaxy_toolsmith.orchestration.export import (
    create_ollama_model,
    export_model_artifacts,
    normalize_ollama_model_name,
    update_variant_ollama_metadata,
    write_ollama_modelfile,
)
from galaxy_toolsmith.orchestration.promotion import (
    PromotionPolicy,
    decide_promotion,
    load_promotion_policy,
    write_default_promotion_policies,
)
from galaxy_toolsmith.orchestration.training import (
    DISTRIBUTED_TRAINING_STRATEGIES,
    TrainingProfileOverrides,
    get_local_training_run,
    list_local_training_runs,
    run_training,
)
from galaxy_toolsmith.orchestration.training_estimates import (
    DEFAULT_CONTEXT_LENGTHS,
    estimate_training_tokens,
    parse_context_lengths,
    parse_source_context_modes,
)
from galaxy_toolsmith.runtime.capabilities import detect_runtime_capabilities
from galaxy_toolsmith.runtime.estimates import model_estimates_json
from galaxy_toolsmith.runtime.model_source import model_cache_info
from galaxy_toolsmith.runtime.run_registry import (
    create_monitor_run_tracker,
    list_monitor_runs,
    update_monitor_run,
)
from galaxy_toolsmith.runtime.status import emit_status, resolve_status_log_path
from galaxy_toolsmith.server.app import serve

PLANEMO_ENGINE_CHOICES = ("galaxy", "docker_galaxy", "cwltool", "toil", "external_galaxy")
_BYTE_CAP_RE = re.compile(r"^\s*(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>b|[kmgtp](?:i?b?)?)?\s*$", re.I)
_BYTE_CAP_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "ki": 1024,
    "kib": 1024,
    "m": 1000**2,
    "mb": 1000**2,
    "mi": 1024**2,
    "mib": 1024**2,
    "g": 1000**3,
    "gb": 1000**3,
    "gi": 1024**3,
    "gib": 1024**3,
    "t": 1000**4,
    "tb": 1000**4,
    "ti": 1024**4,
    "tib": 1024**4,
    "p": 1000**5,
    "pb": 1000**5,
    "pi": 1024**5,
    "pib": 1024**5,
}


def _parse_optional_byte_cap(value: str) -> int:
    match = _BYTE_CAP_RE.match(value)
    if not match:
        raise argparse.ArgumentTypeError(
            "expected a non-negative byte count, or a size like 512MB/1GiB; use 0 for unlimited"
        )
    unit = (match.group("unit") or "").lower()
    multiplier = _BYTE_CAP_MULTIPLIERS[unit]
    parsed = Decimal(match.group("size")) * multiplier
    return int(parsed)


def _add_planemo_test_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-planemo-tests",
        action="store_true",
        help="Run 'planemo test' for each wrapper during evaluation.",
    )
    parser.add_argument(
        "--planemo-test-output-dir",
        default="",
        help="Directory for planemo test reports (default: next to evaluation report).",
    )
    parser.add_argument(
        "--planemo-test-timeout",
        type=int,
        default=0,
        help="Planemo per-test timeout in seconds; 0 leaves Planemo default unchanged.",
    )
    parser.add_argument(
        "--planemo-galaxy-root",
        default="",
        help="Optional Galaxy root passed to planemo test --galaxy_root.",
    )
    parser.add_argument(
        "--planemo-install-galaxy",
        action="store_true",
        help="Pass --install_galaxy to planemo test.",
    )
    parser.add_argument(
        "--planemo-engine",
        choices=PLANEMO_ENGINE_CHOICES,
        default="",
        help="Optional Planemo test engine.",
    )
    parser.add_argument(
        "--planemo-conda-prefix",
        default="",
        help="Optional conda prefix passed to planemo test --conda_prefix.",
    )
    parser.add_argument(
        "--planemo-test-data",
        default="",
        help="Optional test-data directory passed to planemo test --test_data.",
    )
    parser.add_argument(
        "--planemo-extra-tools",
        action="append",
        default=[],
        help="Extra tool source passed to planemo test --extra_tools; repeat as needed.",
    )
    parser.add_argument(
        "--planemo-no-dependency-resolution",
        action="store_true",
        help="Pass --no_dependency_resolution to planemo test.",
    )


def _optional_path(value: str) -> Path | None:
    return Path(value).resolve() if value else None


def _add_source_context_args(
    parser: argparse.ArgumentParser,
    *,
    include_source_root: bool = False,
    include_source_file: bool = False,
    source_root_target: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup | None = None,
) -> None:
    parser.add_argument(
        "--source-context-mode",
        choices=SOURCE_CONTEXT_MODES,
        default="none",
        help=(
            "Optional underlying source-code context mode: none, metadata, snippets, "
            "all-filtered, or all-raw."
        ),
    )
    parser.add_argument(
        "--source-context-max-chars",
        type=int,
        default=DEFAULT_SOURCE_CONTEXT_MAX_CHARS,
        help="Maximum source-context characters included in each prompt.",
    )
    parser.add_argument(
        "--source-context-max-files",
        type=int,
        default=DEFAULT_SOURCE_CONTEXT_MAX_FILES,
        help="Maximum source files included in each prompt.",
    )
    parser.add_argument(
        "--include-source-tests",
        action="store_true",
        help=(
            "Include small upstream test/example source files as optional sidecar context. "
            "Equivalent to --test-context-mode snippets unless a test mode is set."
        ),
    )
    parser.add_argument(
        "--test-context-mode",
        choices=TEST_CONTEXT_MODES,
        default="none",
        help=(
            "Optional upstream test/example context mode: none, metadata, snippets, or fixtures. "
            "Fixtures includes small expected-output/data files from test/example directories."
        ),
    )
    parser.add_argument(
        "--test-context-max-chars",
        type=int,
        default=DEFAULT_TEST_CONTEXT_MAX_CHARS,
        help="Maximum test/example context characters included in each prompt.",
    )
    parser.add_argument(
        "--test-context-max-files",
        type=int,
        default=DEFAULT_TEST_CONTEXT_MAX_FILES,
        help="Maximum test/example files included in each prompt.",
    )
    parser.add_argument(
        "--test-context-max-file-bytes",
        type=_parse_optional_byte_cap,
        default=DEFAULT_TEST_CONTEXT_MAX_FILE_BYTES,
        help="Maximum bytes per test/example context file; 0 disables this per-file cap.",
    )
    if include_source_root:
        target = source_root_target or parser
        target.add_argument(
            "--source-root",
            default="",
            help="Optional source tree to scan for source context.",
        )
    if include_source_file:
        parser.add_argument(
            "--source-file",
            default="",
            help="Optional source code file to include in source context.",
        )


def _source_context_settings_from_args(
    args: argparse.Namespace,
    *,
    source_root: Path | None = None,
):
    resolved_source_root = (
        source_root if source_root is not None else _optional_path(getattr(args, "source_root", ""))
    )
    test_context_mode = getattr(args, "test_context_mode", "none")
    if getattr(args, "include_source_tests", False) and test_context_mode == "none":
        test_context_mode = "snippets"
    return source_context_settings(
        mode=args.source_context_mode,
        max_chars=args.source_context_max_chars,
        max_files=args.source_context_max_files,
        source_root=resolved_source_root,
        source_file=_optional_path(getattr(args, "source_file", "")),
        test_context_mode=test_context_mode,
        test_context_max_chars=getattr(
            args,
            "test_context_max_chars",
            DEFAULT_TEST_CONTEXT_MAX_CHARS,
        ),
        test_context_max_files=getattr(
            args,
            "test_context_max_files",
            DEFAULT_TEST_CONTEXT_MAX_FILES,
        ),
        test_context_max_file_bytes=getattr(
            args,
            "test_context_max_file_bytes",
            DEFAULT_TEST_CONTEXT_MAX_FILE_BYTES,
        ),
    )


def _add_runtime_discovery_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--discovery-mode",
        choices=DISCOVERY_MODES,
        default="none",
        help=(
            "Optionally discover runtime help/source before generation using a Bioconda "
            "conda environment, Biocontainers, or conda then Biocontainers fallback."
        ),
    )
    parser.add_argument(
        "--discovery-package",
        action="append",
        default=[],
        help=(
            "Conda package spec to install and inspect, such as minibwa or minibwa=0.2.0. "
            "Repeat for tools requiring multiple packages. Defaults to the discovery command "
            "or tool name when discovery is enabled."
        ),
    )
    parser.add_argument(
        "--discovery-command",
        default="",
        help="Executable command to probe for help. Defaults to the first discovery package.",
    )
    parser.add_argument(
        "--discovery-channel",
        action="append",
        default=[],
        help="Conda channel for discovery environment creation; repeat to set order.",
    )
    parser.add_argument("--discovery-env-dir", default="", help="Reusable conda env prefix.")
    parser.add_argument(
        "--discovery-cache-dir",
        default="",
        help="Cache directory for discovery conda environments and metadata.",
    )
    parser.add_argument(
        "--discovery-conda-executable",
        default="",
        help="Explicit mamba/micromamba/conda executable for discovery env creation.",
    )
    parser.add_argument(
        "--discovery-conda-timeout-seconds",
        type=int,
        default=900,
        help="Timeout for creating the discovery conda environment.",
    )
    parser.add_argument(
        "--discover-subcommands",
        dest="discover_subcommands",
        action="store_true",
        default=True,
        help="Probe detected subcommands for focused runtime help.",
    )
    parser.add_argument(
        "--no-discover-subcommands",
        dest="discover_subcommands",
        action="store_false",
        help="Only probe top-level runtime help.",
    )
    parser.add_argument(
        "--max-discovered-subcommands",
        type=int,
        default=8,
        help="Maximum subcommands to probe during runtime discovery.",
    )
    parser.add_argument(
        "--discovery-container-runtime",
        choices=["auto", "singularity", "apptainer", "docker"],
        default="auto",
        help="Container runtime for Biocontainers discovery; auto tries local supported runtimes.",
    )
    parser.add_argument(
        "--discovery-container-cache-dir",
        default="",
        help="Container cache directory for Biocontainers discovery.",
    )
    parser.add_argument(
        "--discovery-docker-use-sudo",
        action="store_true",
        help="Use sudo for Docker commands during Biocontainers discovery.",
    )
    parser.add_argument(
        "--discovery-container-help-probe-mode",
        choices=["safe", "exploratory"],
        default="exploratory",
        help="Help probe breadth for runtime discovery.",
    )
    parser.add_argument(
        "--discovery-container-timeout-seconds",
        type=int,
        default=120,
        help="Per-command timeout for runtime help probes.",
    )
    parser.add_argument(
        "--discovery-bioconda-ref",
        default="master",
        help="Bioconda-recipes git ref used when resolving source code.",
    )
    parser.add_argument(
        "--discovery-source-download-max-bytes",
        type=_parse_optional_byte_cap,
        default=0,
        help=(
            "Maximum bytes for automatic source downloads during discovery "
            "(default: 0, unlimited; accepts sizes like 1GB or 1GiB)."
        ),
    )
    parser.add_argument(
        "--discovery-source-download-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for automatic source downloads during discovery.",
    )


def _runtime_discovery_settings_from_args(
    args: argparse.Namespace,
    *,
    paths: WorkspacePaths,
) -> RuntimeDiscoverySettings | None:
    mode = normalize_discovery_mode(getattr(args, "discovery_mode", "none"))
    if mode == "none":
        return None
    command = str(getattr(args, "discovery_command", "") or "").strip()
    package_specs = tuple(str(value).strip() for value in getattr(args, "discovery_package", []) if str(value).strip())
    if not command:
        command = package_specs[0].split("=", 1)[0] if package_specs else str(args.tool_name)
    if not package_specs:
        package_specs = (command,)
    return RuntimeDiscoverySettings(
        mode=mode,
        package_specs=package_specs,
        command=command,
        channels=tuple(getattr(args, "discovery_channel", []) or DEFAULT_DISCOVERY_CHANNELS),
        cache_dir=Path(args.discovery_cache_dir).resolve()
        if str(getattr(args, "discovery_cache_dir", "") or "").strip()
        else paths.cache_root / "generation" / "runtime-discovery",
        env_dir=Path(args.discovery_env_dir).resolve()
        if str(getattr(args, "discovery_env_dir", "") or "").strip()
        else None,
        conda_executable=str(getattr(args, "discovery_conda_executable", "") or ""),
        conda_timeout_seconds=int(getattr(args, "discovery_conda_timeout_seconds", 900)),
        discover_subcommands=bool(getattr(args, "discover_subcommands", True)),
        max_discovered_subcommands=int(getattr(args, "max_discovered_subcommands", 8)),
        container_runtime=str(getattr(args, "discovery_container_runtime", "auto") or "auto"),
        container_cache_dir=Path(args.discovery_container_cache_dir).resolve()
        if str(getattr(args, "discovery_container_cache_dir", "") or "").strip()
        else paths.cache_root / "containers",
        docker_use_sudo=bool(getattr(args, "discovery_docker_use_sudo", False)),
        container_help_probe_mode=str(
            getattr(args, "discovery_container_help_probe_mode", "exploratory") or "exploratory"
        ),
        container_timeout_seconds=int(getattr(args, "discovery_container_timeout_seconds", 120)),
        bioconda_ref=str(getattr(args, "discovery_bioconda_ref", "master") or "master"),
        source_download_max_bytes=int(getattr(args, "discovery_source_download_max_bytes", 0)),
        source_download_timeout_seconds=int(
            getattr(args, "discovery_source_download_timeout_seconds", 60)
        ),
    )


def _run_runtime_discovery_from_args(
    args: argparse.Namespace,
    *,
    paths: WorkspacePaths,
) -> RuntimeDiscoveryResult | None:
    settings = _runtime_discovery_settings_from_args(args, paths=paths)
    if settings is None:
        return None
    print(
        f"Runtime discovery: mode={settings.mode} command={settings.command} "
        f"packages={','.join(settings.package_specs)}",
        file=sys.stderr,
    )
    result = discover_runtime_context(paths=paths, settings=settings)
    if result.has_help:
        print(
            f"Runtime discovery accepted help from {result.selected_runtime or result.mode}.",
            file=sys.stderr,
        )
    elif result.errors:
        print(
            "Runtime discovery did not collect accepted help: "
            + "; ".join(result.errors[:3]),
            file=sys.stderr,
        )
    return result


def _generation_help_text_from_args(
    args: argparse.Namespace,
    discovery: RuntimeDiscoveryResult | None,
) -> str:
    parts: list[str] = []
    help_text_file = str(getattr(args, "help_text_file", "") or "").strip()
    if help_text_file:
        parts.append(Path(help_text_file).resolve().read_text(encoding="utf-8"))
    if discovery is not None and discovery.combined_help_text.strip():
        parts.append(discovery.combined_help_text)
    if not parts:
        raise ValueError("--help-text-file is required unless --discovery-mode produces help.")
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _suite_generation_help_text_from_args(
    args: argparse.Namespace,
    discovery: RuntimeDiscoveryResult | None,
) -> str:
    parts: list[str] = []
    help_text_file = str(getattr(args, "help_text_file", "") or "").strip()
    if help_text_file:
        parts.append(Path(help_text_file).resolve().read_text(encoding="utf-8"))
    if discovery is not None and discovery.top_level_help.strip():
        parts.append(
            "Runtime-discovered top-level command help:\n\n"
            + discovery.top_level_help.strip()
        )
    elif discovery is not None and discovery.combined_help_text.strip():
        parts.append(discovery.combined_help_text)
    if not parts:
        raise ValueError("--help-text-file is required unless --discovery-mode produces help.")
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _help_text_path_from_args(args: argparse.Namespace) -> Path | None:
    help_text_file = str(getattr(args, "help_text_file", "") or "").strip()
    return Path(help_text_file).resolve() if help_text_file else None


def _source_root_override(
    args: argparse.Namespace,
    *,
    source_archive_root: Path | None,
    discovery: RuntimeDiscoveryResult | None,
) -> Path | None:
    if source_archive_root is not None:
        return source_archive_root
    if str(getattr(args, "source_root", "") or "").strip():
        return None
    if discovery is not None and discovery.source_root:
        return Path(discovery.source_root).resolve()
    return None


def _add_shed_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shed-name", default="", help="Tool Shed repository name override.")
    parser.add_argument("--shed-owner", default="", help="Tool Shed owner metadata.")
    parser.add_argument("--shed-description", default="", help="Tool Shed repository description.")
    parser.add_argument(
        "--shed-category",
        action="append",
        default=[],
        help="Tool Shed category; repeat to include multiple categories.",
    )
    parser.add_argument(
        "--shed-homepage-url",
        default="",
        help="Tool Shed homepage_url metadata.",
    )
    parser.add_argument(
        "--shed-remote-repository-url",
        default="",
        help="Tool Shed remote_repository_url metadata.",
    )
    parser.add_argument(
        "--no-shed-yml",
        action="store_true",
        help="Do not write .shed.yml in repository bundle output modes.",
    )


def _shed_metadata_from_args(
    args: argparse.Namespace,
    *,
    default_name: str,
    default_description: str,
    suite: bool = False,
    repositories: list[str] | tuple[str, ...] | None = None,
):
    return build_tool_shed_metadata(
        name=str(getattr(args, "shed_name", "") or "").strip() or default_name,
        owner=str(getattr(args, "shed_owner", "") or "").strip(),
        description=str(getattr(args, "shed_description", "") or "").strip()
        or default_description,
        homepage_url=str(getattr(args, "shed_homepage_url", "") or "").strip(),
        remote_repository_url=str(getattr(args, "shed_remote_repository_url", "") or "").strip(),
        categories=list(getattr(args, "shed_category", []) or []),
        suite=suite,
        repositories=repositories or (),
    )


def _repository_output_path(
    *,
    repository_output_dir: Path | None,
    output_value: str,
    tool_name: str,
) -> Path:
    if repository_output_dir is None:
        if not str(output_value).strip():
            raise ValueError("--output is required unless --repository-output-dir is provided.")
        return Path(output_value).resolve()
    repository_output_dir = repository_output_dir.resolve()
    output_text = str(output_value or "").strip()
    if not output_text:
        return repository_output_dir / f"{safe_tool_id(tool_name)}.xml"
    output_path = Path(output_text)
    if output_path.is_absolute():
        raise ValueError("--output must be relative when --repository-output-dir is used.")
    return repository_output_dir / output_path


def _planemo_test_options_from_args(args: argparse.Namespace) -> PlanemoTestOptions:
    return PlanemoTestOptions(
        output_dir=_optional_path(args.planemo_test_output_dir),
        timeout_seconds=max(0, int(args.planemo_test_timeout or 0)),
        galaxy_root=_optional_path(args.planemo_galaxy_root),
        install_galaxy=bool(args.planemo_install_galaxy),
        engine=str(args.planemo_engine or ""),
        conda_prefix=_optional_path(args.planemo_conda_prefix),
        test_data=_optional_path(args.planemo_test_data),
        extra_tools=tuple(Path(path).resolve() for path in args.planemo_extra_tools or [] if path),
        no_dependency_resolution=bool(args.planemo_no_dependency_resolution),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gtsm",
        description="Galaxy Toolsmith command line interface.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root for path resolution (default: current directory).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Print resolved workspace paths.")
    subparsers.add_parser("init-config", help="Write default config file.")
    subparsers.add_parser(
        "init-workspace", help="Create cache/config dirs and seed manifest files."
    )
    subparsers.add_parser("list-train-profiles", help="List configured training profiles.")
    subparsers.add_parser("list-model-variants", help="List known model variant manifests.")
    subparsers.add_parser(
        "estimate-model-resources",
        help="Print resource/cost estimate tiers for model profiles.",
    )
    estimate_tokens_parser = subparsers.add_parser(
        "estimate-training-tokens",
        help="Estimate training sample sizes across context-length candidates.",
    )
    estimate_tokens_parser.add_argument(
        "--profile",
        default="agentic-devstral-24b",
        help="Training profile name from config/training.profiles.json.",
    )
    estimate_tokens_parser.add_argument(
        "--corpus-jsonl",
        default=".gtsm-cache/datasets/tools-iuc-corpus.jsonl",
        help="Training corpus JSONL path.",
    )
    estimate_tokens_parser.add_argument(
        "--artifact-format",
        choices=["xml", "udt-yaml", "mixed"],
        default="xml",
        help="Training target format to estimate.",
    )
    estimate_tokens_parser.add_argument(
        "--max-seq-lengths",
        default=",".join(str(value) for value in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated context lengths to check; k suffix means 1024 tokens.",
    )
    _add_source_context_args(
        estimate_tokens_parser,
        include_source_root=True,
        include_source_file=True,
    )
    estimate_tokens_parser.add_argument(
        "--compare-source-context-modes",
        default="",
        help="Comma-separated source-context modes to compare, e.g. all-filtered,all-raw.",
    )
    estimate_tokens_parser.add_argument(
        "--source-context-budget-ladder",
        action="store_true",
        help="Use the built-in source char/file budget ladder for each context length.",
    )
    estimate_tokens_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional corpus-record limit for quick estimates; 0 means the full corpus.",
    )
    estimate_tokens_parser.add_argument(
        "--exact-tokenizer",
        action="store_true",
        help="Use the profile tokenizer from the local model cache instead of char estimates.",
    )
    estimate_tokens_parser.add_argument(
        "--chars-per-token",
        type=float,
        default=3.7,
        help="Approximate characters per token when not using --exact-tokenizer.",
    )
    estimate_tokens_parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print estimator progress every N corpus records; 0 disables progress.",
    )
    estimate_tokens_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Estimator worker threads. 0 uses an automatic bounded worker count.",
    )
    estimate_tokens_parser.add_argument(
        "--longest-sample-count",
        type=int,
        default=25,
        help="Number of longest training samples to include per source-context estimate.",
    )
    estimate_tokens_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output. This is currently the default output format.",
    )
    subparsers.add_parser(
        "list-promotion-policies", help="List configured promotion policy profiles."
    )
    subparsers.add_parser("runtime-detect", help="Detect CPU/CUDA/ROCm/MPS runtime capabilities.")
    subparsers.add_parser(
        "model-cache-info", help="Print resolved Hugging Face model cache settings."
    )

    sync_parser = subparsers.add_parser(
        "sync-tools-iuc", help="Clone/fetch tools-iuc into source cache."
    )
    sync_parser.add_argument("--ref", default="main", help="Git ref to checkout (default: main).")
    sync_skills_parser = subparsers.add_parser(
        "sync-galaxy-skills",
        help="Clone/fetch galaxyproject-skills into source cache.",
    )
    sync_skills_parser.add_argument(
        "--ref", default="main", help="Git ref to checkout (default: main)."
    )
    sync_xsd_parser = subparsers.add_parser(
        "sync-galaxy-xsd",
        help="Download/cache Galaxy tool schema (galaxy.xsd).",
    )
    sync_xsd_parser.add_argument(
        "--ref", default="dev", help="Galaxy git ref for raw XSD (default: dev)."
    )

    extract_parser = subparsers.add_parser(
        "extract-corpus",
        help="Extract tool wrapper/test/datatype corpus from a tools-iuc checkout.",
    )
    extract_parser.add_argument(
        "--tools-root",
        help="Path to tools-iuc/tools. Defaults to cached tools-iuc/tools.",
    )
    extract_parser.add_argument(
        "--output",
        default=".gtsm-cache/datasets/tools-iuc-corpus.jsonl",
        help="Output JSONL path.",
    )
    extract_parser.add_argument(
        "--checkpoint",
        default=".gtsm-cache/datasets/tools-iuc-corpus.checkpoint",
        help="Checkpoint path for resumable extraction.",
    )
    extract_parser.add_argument(
        "--restart",
        action="store_true",
        help="Archive existing corpus artifacts and start extraction from scratch.",
    )
    extract_parser.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for extraction status events (disabled by default).",
    )
    extract_parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers.")
    extract_parser.add_argument(
        "--source-workers",
        type=int,
        default=0,
        help=(
            "Maximum concurrent source checkout/download operations "
            "(default: min(8, --max-workers))."
        ),
    )
    extract_parser.add_argument(
        "--container-prepare-workers",
        type=int,
        default=2,
        help="Maximum concurrent container image prepare/build/pull operations.",
    )
    extract_parser.add_argument(
        "--container-probe-workers",
        type=int,
        default=4,
        help="Maximum concurrent container help/API probe workers.",
    )
    extract_parser.add_argument(
        "--retries", type=int, default=3, help="Retries per tool on parse failures."
    )
    extract_parser.add_argument(
        "--no-fetch-docs",
        action="store_true",
        help="Disable GitHub README fetching from .shed.yml homepage_url.",
    )
    extract_parser.add_argument(
        "--resolve-containers",
        action="store_true",
        help="Resolve container candidates from explicit containers, mulled/BioContainers names, and package metadata.",
    )
    extract_parser.add_argument(
        "--execute-containers",
        action="store_true",
        help="Run resolved containers and collect command help output (opt-in).",
    )
    extract_parser.add_argument(
        "--container-runtime",
        choices=("auto", "singularity", "apptainer", "docker"),
        default="auto",
        help="Container runtime for help extraction (default: auto; prefers Singularity/Apptainer, then Docker).",
    )
    extract_parser.add_argument(
        "--container-cache-dir",
        default="",
        help="Directory for cached Singularity/Apptainer images (default: .gtsm-cache/containers).",
    )
    extract_parser.add_argument(
        "--container-sif-exec-mode",
        choices=("auto", "sif", "sandbox"),
        default="auto",
        help=(
            "How Singularity/Apptainer executes cached SIF images: auto uses direct SIF "
            "mounting when user FUSE is available and otherwise reuses a persistent "
            "sandbox cache; sif preserves direct runtime behavior; sandbox always "
            "materializes a reusable sandbox."
        ),
    )
    extract_parser.add_argument(
        "--container-help-probe-mode",
        choices=("safe", "exploratory"),
        default="exploratory",
        help="Container help probe strategy (default: exploratory; tries more safe fallbacks).",
    )
    extract_parser.add_argument(
        "--container-image-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for container image download/build/pull operations.",
    )
    extract_parser.add_argument(
        "--container-image-quarantine-seconds",
        type=int,
        default=86400,
        help="Seconds to skip an image after a prepare timeout/failure quarantine.",
    )
    extract_parser.add_argument(
        "--container-image-quarantine-file",
        default="",
        help=(
            "JSON file for persistent container image quarantine state "
            "(default: <container-cache-dir>/image-quarantine.json)."
        ),
    )
    extract_parser.add_argument(
        "--source-download-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for upstream source archive download requests.",
    )
    extract_parser.add_argument(
        "--source-download-max-bytes",
        type=_parse_optional_byte_cap,
        default=0,
        help=(
            "Maximum bytes to download for one upstream source archive "
            "(default: 0, unlimited; accepts sizes like 512MB or 1GiB)."
        ),
    )
    extract_parser.add_argument(
        "--singularity-depot-url",
        default="https://depot.galaxyproject.org/singularity",
        help="Galaxy Singularity depot URL used before docker:// fallback.",
    )
    extract_parser.add_argument(
        "--docker-use-sudo",
        action="store_true",
        help="Use 'sudo docker' when Docker fallback is selected.",
    )
    extract_parser.add_argument(
        "--no-remove-images",
        action="store_true",
        help="Keep pulled images after execution (default removes after final use).",
    )
    extract_parser.add_argument(
        "--bioconda-checkout-sources",
        action="store_true",
        help="Resolve Bioconda recipe and checkout upstream source for requirement packages.",
    )
    extract_parser.add_argument(
        "--bioconda-ref",
        default="master",
        help="Bioconda recipes git ref for recipe/source resolution.",
    )
    extract_parser.add_argument(
        "--synthesize-udt-yaml",
        action="store_true",
        help="Write deterministic Galaxy User-Defined Tool YAML targets for each extracted XML wrapper.",
    )
    extract_parser.add_argument(
        "--retry-manifest",
        default="",
        help=(
            "Read an existing retry manifest to target listed wrappers, then write an updated "
            "retry/failure manifest to the same path (default: alongside output JSONL)."
        ),
    )
    extract_parser.add_argument(
        "--wrapper-source-max-bytes",
        type=int,
        default=DEFAULT_WRAPPER_SOURCE_MAX_BYTES,
        help=(
            "Maximum bytes for wrapper-local helper source files captured as context "
            f"(default: {DEFAULT_WRAPPER_SOURCE_MAX_BYTES})."
        ),
    )
    extract_parser.add_argument(
        "--wrapper-configfile-max-bytes",
        type=int,
        default=DEFAULT_WRAPPER_CONFIGFILE_MAX_BYTES,
        help=(
            "Maximum stored bytes for each inline wrapper configfile context block "
            f"(default: {DEFAULT_WRAPPER_CONFIGFILE_MAX_BYTES})."
        ),
    )

    rebuild_report_parser = subparsers.add_parser(
        "rebuild-execution-report",
        help="Rebuild extract-corpus execution report from a corpus JSONL file.",
    )
    rebuild_report_parser.add_argument(
        "--corpus-jsonl",
        default=".gtsm-cache/datasets/tools-iuc-corpus.jsonl",
        help="Corpus JSONL path.",
    )
    rebuild_report_parser.add_argument(
        "--output",
        default="",
        help="Execution report output path (default: corpus path with .execution.json suffix).",
    )

    diagnose_corpus_parser = subparsers.add_parser(
        "diagnose-corpus",
        help="Write QA diagnostics for an extract-corpus execution report.",
    )
    diagnose_corpus_parser.add_argument(
        "--execution-report",
        default=".gtsm-cache/datasets/tools-iuc-corpus.execution.json",
        help="Execution report JSON path.",
    )
    diagnose_corpus_parser.add_argument(
        "--corpus-jsonl",
        default="",
        help="Corpus JSONL path (default: inferred from execution report).",
    )
    diagnose_corpus_parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path (default: inferred from execution report).",
    )
    diagnose_corpus_parser.add_argument(
        "--current-run",
        default="",
        help="Current run pointer path (default: execution report directory/current).",
    )
    diagnose_corpus_parser.add_argument(
        "--diagnostics-dir",
        default=".gtsm-cache/diagnostics",
        help="Directory for diagnostic output files.",
    )
    diagnose_corpus_parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Maximum number of non-help records to include in samples.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Run profile-based training orchestration and persist run metadata.",
    )
    train_parser.add_argument(
        "--profile",
        default="agentic-devstral-24b",
        help="Training profile name from config/training.profiles.json.",
    )
    train_parser.add_argument(
        "--dataset-manifest",
        default="config/dataset.manifest.json",
        help="Path to dataset manifest JSON.",
    )
    train_parser.add_argument(
        "--variant-id",
        help="Optional explicit model variant id.",
    )
    train_parser.add_argument(
        "--command",
        nargs="+",
        dest="trainer_command",
        help="Optional trainer command override.",
    )
    train_parser.add_argument(
        "--corpus-jsonl",
        default=".gtsm-cache/datasets/tools-iuc-corpus.jsonl",
        help="Training corpus JSONL path.",
    )
    train_parser.add_argument(
        "--artifact-format",
        choices=["xml", "udt-yaml", "mixed"],
        default="xml",
        help="Training target format: xml, udt-yaml, or mixed real targets.",
    )
    train_parser.add_argument(
        "--backend",
        choices=["auto", "axolotl", "hf-sft", "command", "mlx-lm", "mlx", "mps"],
        default="auto",
        help="Training backend override (default: auto, using the profile backend).",
    )
    train_parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of local training processes for Axolotl/torchrun launch.",
    )
    train_parser.add_argument(
        "--distributed-strategy",
        choices=sorted(DISTRIBUTED_TRAINING_STRATEGIES),
        default="",
        help="Distributed training strategy for Axolotl (default: profile setting, usually ddp).",
    )
    train_parser.add_argument(
        "--dry-run-backend",
        action="store_true",
        help="Prepare backend inputs and print metadata without launching training.",
    )
    train_parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Override the profile max sequence length for this run.",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop training after this many optimizer steps; useful for memory probes.",
    )
    pad_group = train_parser.add_mutually_exclusive_group()
    pad_group.add_argument(
        "--pad-to-sequence-len",
        dest="pad_to_sequence_len",
        action="store_true",
        default=None,
        help="Pad training samples to the full configured sequence length.",
    )
    pad_group.add_argument(
        "--no-pad-to-sequence-len",
        dest="pad_to_sequence_len",
        action="store_false",
        help="Do not pad training samples to the full configured sequence length.",
    )
    train_parser.add_argument(
        "--attn-implementation",
        choices=[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "flex_attention",
            "xformers",
            "sage",
            "fp8",
        ],
        help="Override the Axolotl attention backend for this run.",
    )
    _add_source_context_args(
        train_parser,
        include_source_root=True,
        include_source_file=True,
    )
    train_parser.add_argument(
        "--per-device-batch-size",
        type=int,
        help="Override the profile per-device batch size for this run.",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Override the profile gradient accumulation steps for this run.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the profile learning rate for this run.",
    )
    train_parser.add_argument(
        "--training-method",
        choices=sorted(TRAINING_METHODS),
        help="Override training method: lora, qlora, or full.",
    )
    train_parser.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for training status events.",
    )
    train_parser.add_argument(
        "--status-interval-seconds",
        type=float,
        default=30.0,
        help="Seconds between live training status updates.",
    )
    train_parser.add_argument(
        "--stream-logs",
        action="store_true",
        help="Emit incremental training log chunks as status events while the backend runs.",
    )
    train_parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=40,
        help="Maximum new stdout/stderr lines per streamed training log event.",
    )
    train_parser.add_argument(
        "--internal-run-id",
        default="",
        help=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--internal-distributed-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    train_parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Reserved for local/distributed checkpoint resume integration.",
    )
    train_parser.add_argument(
        "--post-export-quantizations",
        default="",
        help="Optional comma-separated quantizations to export after successful training.",
    )
    train_parser.add_argument(
        "--post-ollama-model-name",
        default="",
        help="Optional Ollama model name to generate Modelfile for after training.",
    )
    train_parser.add_argument(
        "--post-ollama-create",
        action="store_true",
        help="Run `ollama create` after generating post-training Modelfile.",
    )
    train_runs_parser = subparsers.add_parser(
        "train-runs",
        help="List local direct training runs.",
    )
    train_runs_parser.add_argument("--limit", type=int, default=20)
    train_runs_parser.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for status events (disabled by default).",
    )
    train_status_parser = subparsers.add_parser(
        "train-status",
        help="Read status for a local direct training run.",
    )
    train_status_parser.add_argument("--run-id", default="latest")
    train_status_parser.add_argument("--tail", type=int, default=80)
    train_status_parser.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for status events (disabled by default).",
    )
    export_parser = subparsers.add_parser(
        "export-model",
        help="Export model artifacts for a trained variant.",
    )
    export_parser.add_argument(
        "--variant-id", required=True, help="Variant id (from models/variants)."
    )
    export_parser.add_argument(
        "--format",
        default="all",
        choices=["all", "merged", "gguf"],
        help="Export format.",
    )
    export_parser.add_argument(
        "--quantizations",
        default="q4_k_m",
        help="Comma-separated GGUF quantization methods (used when format is gguf/all).",
    )
    export_ollama_parser = subparsers.add_parser(
        "export-ollama-model",
        help="Generate Ollama Modelfile and optionally create Ollama model.",
    )
    export_ollama_parser.add_argument("--variant-id", required=True)
    export_ollama_parser.add_argument("--model-name", required=True)
    export_ollama_parser.add_argument("--from-quantization", default="q4_k_m")
    export_ollama_parser.add_argument("--create", action="store_true")

    convert_adapter_parser = subparsers.add_parser(
        "convert-adapter",
        help="Convert supported adapter artifacts between local formats.",
    )
    convert_adapter_parser.add_argument(
        "--from", dest="from_format", choices=["mlx"], required=True
    )
    convert_adapter_parser.add_argument("--to", dest="to_format", choices=["peft"], required=True)
    convert_adapter_parser.add_argument(
        "--base-model",
        required=True,
        help="Base model id/path used to select the architecture mapping.",
    )
    convert_adapter_parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Input MLX adapter directory containing adapter_config.json and adapters.safetensors.",
    )
    convert_adapter_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the PEFT adapter.",
    )

    generate_parser = subparsers.add_parser(
        "generate-wrapper",
        help="Generate a Galaxy tool artifact from help text and optional source code.",
    )
    generate_parser.add_argument("--tool-name", required=True, help="Tool identifier/name.")
    generate_parser.add_argument(
        "--tool-id",
        default="",
        help="Optional Galaxy tool id. Defaults to a safe id derived from --tool-name.",
    )
    generate_parser.add_argument("--help-text-file", default="", help="Path to help text file.")
    _add_runtime_discovery_args(generate_parser)
    generate_parser.add_argument(
        "--artifact-format",
        choices=["xml", "udt-yaml"],
        default="xml",
        help="Artifact format to generate.",
    )
    generate_parser.add_argument(
        "--tool-granularity",
        choices=["single", "subcommands", "suite", "auto"],
        default="auto",
        help=(
            "Generation granularity hint. Current generate-wrapper output remains one primary "
            "tool XML; suite/subcommand hints are recorded for prompt and monitor metadata."
        ),
    )
    generate_parser.add_argument(
        "--source-file", help="Optional source code file to provide extra context."
    )
    source_root_group = generate_parser.add_mutually_exclusive_group()
    _add_source_context_args(
        generate_parser,
        include_source_root=True,
        source_root_target=source_root_group,
    )
    source_root_group.add_argument(
        "--source-archive",
        default="",
        help=(
            "Optional local source archive path or HTTP(S)/FTP URL to download, "
            "extract, and scan for source context."
        ),
    )
    generate_parser.add_argument(
        "--source-archive-max-bytes",
        type=_parse_optional_byte_cap,
        default=DEFAULT_SOURCE_ARCHIVE_MAX_BYTES,
        help=(
            "Maximum bytes for one manual source archive "
            "(default: 1GB; accepts sizes like 512MB or 1GiB; use 0 for unlimited)."
        ),
    )
    generate_parser.add_argument(
        "--source-archive-timeout-seconds",
        type=int,
        default=DEFAULT_SOURCE_ARCHIVE_TIMEOUT_SECONDS,
        help="Timeout for manual source archive URL downloads.",
    )
    generate_parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "copilot", "ollama"],
        default="local",
        help="Generation provider.",
    )
    generate_parser.add_argument(
        "--model",
        default="",
        help="Provider model name override.",
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for external providers.",
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens for external providers.",
    )
    generate_parser.add_argument(
        "--ollama-context-tokens",
        type=int,
        default=None,
        help=(
            "Ollama runtime context tokens (num_ctx). "
            "Unset uses GTSM_OLLAMA_CONTEXT_TOKENS or Ollama defaults; 0 disables num_ctx."
        ),
    )
    generate_parser.add_argument(
        "--max-prompt-help-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_HELP_CHARS,
        help="Maximum help-text characters included in the generation prompt.",
    )
    generate_parser.add_argument(
        "--model-variant",
        default="bootstrap-variant",
        help="Model variant identifier used for metadata.",
    )
    generate_parser.add_argument(
        "--skills-profile",
        default="default",
        help="Prompt skills profile name (default: default).",
    )
    generate_parser.add_argument(
        "--allow-stub-local",
        action="store_true",
        help="Allow the local provider to return canned starter XML when no real local model is configured.",
    )
    generate_parser.add_argument(
        "--repair-invalid-xml",
        dest="repair_invalid_xml",
        action="store_true",
        default=True,
        help="Retry malformed, non-tool, truncated, or degenerate XML once with a stricter repair prompt.",
    )
    generate_parser.add_argument(
        "--no-repair-invalid-xml",
        dest="repair_invalid_xml",
        action="store_false",
        help="Disable automatic one-shot repair for invalid generated XML.",
    )
    generate_parser.add_argument(
        "--stream-output",
        action="store_true",
        help="For local HF/PEFT generation, stream decoded model text to stderr while generating.",
    )
    generate_parser.add_argument(
        "--raw-response-log",
        default="",
        help="Optional path for the full unstripped raw model response.",
    )
    generate_parser.add_argument(
        "--generate-sidecars",
        action="store_true",
        help="Write companion sidecar XML blocks from the raw response alongside the primary tool XML.",
    )
    generate_parser.add_argument(
        "--no-toolsmith-citation",
        dest="include_toolsmith_citation",
        action="store_false",
        default=True,
        help="Do not add the deterministic Galaxy Toolsmith citation to generated artifacts.",
    )
    generate_parser.add_argument(
        "--no-datatype-scaffold",
        dest="datatype_scaffold",
        action="store_false",
        default=True,
        help="Do not write datatype placeholder scaffolds for unknown Galaxy datatypes.",
    )
    generate_parser.add_argument(
        "--sidecar-output-dir",
        default="",
        help="Directory for generated sidecar files (default: <output>.sidecars/).",
    )
    generate_parser.add_argument(
        "--repository-output-dir",
        default="",
        help=(
            "Optional Tool Shed-style repository directory. When set, the generated XML, "
            ".shed.yml, sidecars, and .gtsm metadata are written into this directory."
        ),
    )
    _add_shed_metadata_args(generate_parser)
    generate_parser.add_argument(
        "--output",
        default="",
        help="Output artifact path. Required unless --repository-output-dir is provided.",
    )

    plan_suite_parser = subparsers.add_parser(
        "plan-suite",
        help="Inspect help/source context and produce a proposed multi-tool suite plan.",
    )
    plan_suite_parser.add_argument("--tool-name", required=True, help="Tool identifier/name.")
    plan_suite_parser.add_argument("--help-text-file", default="", help="Path to help text file.")
    _add_runtime_discovery_args(plan_suite_parser)
    plan_suite_parser.add_argument("--source-file", default="", help="Optional source code file.")
    plan_suite_parser.add_argument(
        "--max-suite-tools",
        type=int,
        default=8,
        help="Maximum suite members to include in the proposed plan.",
    )
    plan_suite_parser.add_argument(
        "--force-suite",
        action="store_true",
        help="Return a suite-style plan even when only one tool is detected.",
    )

    suite_parser = subparsers.add_parser(
        "generate-suite",
        help="Generate a Tool Shed-style repository bundle containing multiple tool wrappers.",
    )
    suite_parser.add_argument("--tool-name", required=True, help="Tool/package name.")
    suite_parser.add_argument("--help-text-file", default="", help="Path to help text file.")
    _add_runtime_discovery_args(suite_parser)
    suite_parser.add_argument("--source-file", default="", help="Optional source code file.")
    suite_source_root_group = suite_parser.add_mutually_exclusive_group()
    _add_source_context_args(
        suite_parser,
        include_source_root=True,
        source_root_target=suite_source_root_group,
    )
    suite_source_root_group.add_argument(
        "--source-archive",
        default="",
        help="Optional local source archive path or HTTP(S)/FTP URL to extract and scan.",
    )
    suite_parser.add_argument(
        "--source-archive-max-bytes",
        type=_parse_optional_byte_cap,
        default=DEFAULT_SOURCE_ARCHIVE_MAX_BYTES,
        help="Maximum bytes for one manual source archive (default: 1GB; use 0 for unlimited).",
    )
    suite_parser.add_argument(
        "--source-archive-timeout-seconds",
        type=int,
        default=DEFAULT_SOURCE_ARCHIVE_TIMEOUT_SECONDS,
        help="Timeout for manual source archive URL downloads.",
    )
    suite_parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "copilot", "ollama"],
        default="local",
        help="Generation provider.",
    )
    suite_parser.add_argument("--model", default="", help="Provider model name override.")
    suite_parser.add_argument("--model-variant", default="suite-variant")
    suite_parser.add_argument("--skills-profile", default="default")
    suite_parser.add_argument("--temperature", type=float, default=0.1)
    suite_parser.add_argument("--max-tokens", type=int, default=4096)
    suite_parser.add_argument(
        "--ollama-context-tokens",
        type=int,
        default=None,
        help=(
            "Ollama runtime context tokens (num_ctx). "
            "Unset uses GTSM_OLLAMA_CONTEXT_TOKENS or Ollama defaults; 0 disables num_ctx."
        ),
    )
    suite_parser.add_argument("--max-prompt-help-chars", type=int, default=DEFAULT_MAX_PROMPT_HELP_CHARS)
    suite_parser.add_argument("--allow-stub-local", action="store_true")
    suite_parser.add_argument(
        "--local-offload-policy",
        choices=["allow", "fail"],
        default="allow",
        help="Whether local PEFT suite generation may use CPU/disk/model offload.",
    )
    suite_parser.add_argument(
        "--local-gpu-memory-reserve-gib",
        type=float,
        default=2.0,
        help="GiB to reserve per visible GPU when computing local PEFT max_memory.",
    )
    suite_parser.add_argument("--output-dir", required=True, help="Repository bundle output directory.")
    suite_parser.add_argument("--max-suite-tools", type=int, default=8)
    suite_parser.add_argument("--generate-sidecars", action="store_true", default=True)
    suite_parser.add_argument("--no-generate-sidecars", dest="generate_sidecars", action="store_false")
    suite_parser.add_argument(
        "--no-toolsmith-citation",
        dest="include_toolsmith_citation",
        action="store_false",
        default=True,
        help="Do not add the deterministic Galaxy Toolsmith citation to generated tools.",
    )
    suite_parser.add_argument(
        "--no-datatype-scaffold",
        dest="datatype_scaffold",
        action="store_false",
        default=True,
        help="Do not write datatype placeholder scaffolds for unknown Galaxy datatypes.",
    )
    suite_parser.add_argument("--raw-response-logs", action="store_true")
    suite_parser.add_argument("--stream-output", action="store_true")
    suite_parser.add_argument("--repair-invalid-xml", dest="repair_invalid_xml", action="store_true", default=True)
    suite_parser.add_argument("--no-repair-invalid-xml", dest="repair_invalid_xml", action="store_false")
    _add_shed_metadata_args(suite_parser)

    compare_generation_parser = subparsers.add_parser(
        "compare-generation-runs",
        help="Compare two generated-suite run directories using their generation records.",
    )
    compare_generation_parser.add_argument("--left-run-dir", required=True)
    compare_generation_parser.add_argument("--right-run-dir", required=True)
    compare_generation_parser.add_argument(
        "--output",
        default="",
        help="Optional comparison JSON path; defaults to LEFT/.gtsm/generation-comparison.json.",
    )

    server_parser = subparsers.add_parser(
        "serve",
        help="Run optional HTTP inference server for remote generation.",
    )
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8765)
    server_parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop matching running gtsm serve processes instead of starting a server.",
    )
    server_parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "copilot", "ollama"],
        default="local",
    )
    server_parser.add_argument("--model", default="")
    server_parser.add_argument("--model-variant", default="server-default")
    server_parser.add_argument("--temperature", type=float, default=0.1)
    server_parser.add_argument("--max-tokens", type=int, default=4096)
    server_parser.add_argument(
        "--ollama-context-tokens",
        type=int,
        default=None,
        help=(
            "Ollama runtime context tokens (num_ctx). "
            "Unset uses GTSM_OLLAMA_CONTEXT_TOKENS or Ollama defaults; 0 disables num_ctx."
        ),
    )
    server_parser.add_argument(
        "--max-prompt-help-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_HELP_CHARS,
        help="Maximum help-text characters included in each generation prompt.",
    )
    server_parser.add_argument(
        "--auth-token-env",
        default="GTSM_SERVER_AUTH_TOKEN",
        help="Environment variable containing optional bearer auth token.",
    )
    server_parser.add_argument(
        "--auth-token",
        action="append",
        default=[],
        help="Optional bearer token value (repeat for multiple tokens).",
    )
    server_parser.add_argument(
        "--auth-tokens-file",
        default="",
        help="Optional file path with one token per line.",
    )
    server_parser.add_argument(
        "--require-generate-auth",
        action="store_true",
        help="Require auth for /generate when tokens are configured.",
    )
    server_parser.add_argument(
        "--allow-stub-local",
        action="store_true",
        help="Allow /generate to return canned starter XML for local provider requests with no real local model.",
    )
    server_parser.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for status events (disabled by default).",
    )
    server_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --stop, list matching server processes without stopping them.",
    )
    server_parser.add_argument(
        "--force",
        action="store_true",
        help="With --stop, send SIGKILL to matching server processes still alive after SIGTERM.",
    )
    server_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="With --stop, seconds to wait after SIGTERM before reporting or forcing.",
    )

    serve_stop_parser = subparsers.add_parser(
        "serve-stop",
        help="Stop running gtsm serve processes for this checkout.",
    )
    serve_stop_parser.add_argument("--host", default="", help="Optional host filter.")
    serve_stop_parser.add_argument("--port", type=int, default=8765, help="Server port filter.")
    serve_stop_parser.add_argument(
        "--all-ports",
        action="store_true",
        help="Ignore --port and stop all matching gtsm serve processes for this checkout.",
    )
    serve_stop_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching server processes without stopping them.",
    )
    serve_stop_parser.add_argument(
        "--force",
        action="store_true",
        help="Send SIGKILL to matching server processes still alive after SIGTERM.",
    )
    serve_stop_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait after SIGTERM before reporting or forcing.",
    )

    remote_parser = subparsers.add_parser(
        "generate-wrapper-remote",
        help="Generate wrapper using optional remote server endpoint.",
    )
    remote_parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    remote_parser.add_argument(
        "--auth-token-env",
        default="GTSM_SERVER_AUTH_TOKEN",
        help="Environment variable containing optional bearer auth token.",
    )
    remote_parser.add_argument("--tool-name", required=True)
    remote_parser.add_argument(
        "--tool-id",
        default="",
        help="Optional Galaxy tool id. Defaults to a safe id derived from --tool-name.",
    )
    remote_parser.add_argument("--help-text-file", required=True)
    remote_parser.add_argument("--source-file")
    remote_parser.add_argument(
        "--artifact-format",
        choices=["xml", "udt-yaml"],
        default="xml",
        help="Artifact format to generate.",
    )
    remote_parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "copilot", "ollama"],
        default="local",
    )
    remote_parser.add_argument("--model", default="")
    remote_parser.add_argument("--model-variant", default="remote-variant")
    remote_parser.add_argument("--skills-profile", default="default")
    remote_parser.add_argument("--temperature", type=float, default=0.1)
    remote_parser.add_argument("--max-tokens", type=int, default=4096)
    remote_parser.add_argument(
        "--max-prompt-help-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_HELP_CHARS,
        help="Maximum help-text characters included in the remote generation prompt.",
    )
    remote_parser.add_argument(
        "--no-toolsmith-citation",
        dest="include_toolsmith_citation",
        action="store_false",
        default=True,
    )
    remote_parser.add_argument(
        "--no-datatype-scaffold",
        dest="datatype_scaffold",
        action="store_false",
        default=True,
    )
    remote_parser.add_argument("--output", required=True)

    remote_suite_parser = subparsers.add_parser(
        "generate-suite-remote",
        help="Generate a repository bundle using the remote server suite endpoint.",
    )
    remote_suite_parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    remote_suite_parser.add_argument(
        "--auth-token-env",
        default="GTSM_SERVER_AUTH_TOKEN",
        help="Environment variable containing optional bearer auth token.",
    )
    remote_suite_parser.add_argument("--tool-name", required=True)
    remote_suite_parser.add_argument("--help-text-file", required=True)
    remote_suite_parser.add_argument("--source-file", default="")
    remote_suite_parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "copilot", "ollama"],
        default="local",
    )
    remote_suite_parser.add_argument("--model", default="")
    remote_suite_parser.add_argument("--model-variant", default="remote-suite-variant")
    remote_suite_parser.add_argument("--skills-profile", default="default")
    remote_suite_parser.add_argument("--temperature", type=float, default=0.1)
    remote_suite_parser.add_argument("--max-tokens", type=int, default=4096)
    remote_suite_parser.add_argument(
        "--max-prompt-help-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_HELP_CHARS,
    )
    remote_suite_parser.add_argument("--max-suite-tools", type=int, default=8)
    remote_suite_parser.add_argument("--output-dir", required=True)
    remote_suite_parser.add_argument("--generate-sidecars", action="store_true", default=True)
    remote_suite_parser.add_argument("--no-generate-sidecars", dest="generate_sidecars", action="store_false")
    remote_suite_parser.add_argument(
        "--no-toolsmith-citation",
        dest="include_toolsmith_citation",
        action="store_false",
        default=True,
    )
    remote_suite_parser.add_argument(
        "--no-datatype-scaffold",
        dest="datatype_scaffold",
        action="store_false",
        default=True,
    )
    remote_suite_parser.add_argument("--repair-invalid-xml", dest="repair_invalid_xml", action="store_true", default=True)
    remote_suite_parser.add_argument("--no-repair-invalid-xml", dest="repair_invalid_xml", action="store_false")
    _add_shed_metadata_args(remote_suite_parser)

    convert_udt_parser = subparsers.add_parser(
        "convert-udt",
        help="Convert Galaxy User-Defined Tool YAML to standard Galaxy tool XML.",
    )
    convert_udt_parser.add_argument("--input", required=True, help="Input UDT YAML path.")
    convert_udt_parser.add_argument("--output", required=True, help="Output Galaxy XML path.")
    convert_udt_parser.add_argument(
        "--report",
        default="",
        help="Optional JSON report path for validation and conversion notes.",
    )
    convert_udt_parser.add_argument(
        "--allow-lossy-conversion",
        action="store_true",
        help="Permit XML output when unsupported UDT expressions must be preserved with notes.",
    )

    train_remote_submit = subparsers.add_parser(
        "train-remote-submit",
        help="Submit training job to server-coordinated worker pool.",
    )
    train_remote_submit.add_argument("--server-url", default="http://127.0.0.1:8765")
    train_remote_submit.add_argument(
        "--auth-token-env",
        default="GTSM_SERVER_AUTH_TOKEN",
        help="Environment variable containing optional bearer auth token.",
    )
    train_remote_submit.add_argument("--profile", default="agentic-devstral-24b")
    train_remote_submit.add_argument("--dataset-manifest", default="config/dataset.manifest.json")
    train_remote_submit.add_argument(
        "--corpus-jsonl", default=".gtsm-cache/datasets/tools-iuc-corpus.jsonl"
    )
    train_remote_submit.add_argument("--variant-id", default="")
    train_remote_submit.add_argument("--command", nargs="+", dest="trainer_command")
    train_remote_submit.add_argument("--learning-rate", type=float)
    train_remote_submit.add_argument(
        "--training-method",
        choices=sorted(TRAINING_METHODS),
        help="Override training method: lora, qlora, or full.",
    )

    train_remote_status = subparsers.add_parser(
        "train-remote-status",
        help="Read status of a remote training job.",
    )
    train_remote_status.add_argument("--server-url", default="http://127.0.0.1:8765")
    train_remote_status.add_argument(
        "--auth-token-env",
        default="GTSM_SERVER_AUTH_TOKEN",
        help="Environment variable containing optional bearer auth token.",
    )
    train_remote_status.add_argument("--job-id", required=True)
    train_remote_status.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for status events (disabled by default).",
    )

    train_worker = subparsers.add_parser(
        "train-worker",
        help="Run training worker that claims tasks from coordinator server.",
    )
    train_worker.add_argument("--server-url", default="http://127.0.0.1:8765")
    train_worker.add_argument(
        "--auth-token-env",
        default="GTSM_SERVER_AUTH_TOKEN",
        help="Environment variable containing optional bearer auth token.",
    )
    train_worker.add_argument("--worker-id", default="")
    train_worker.add_argument("--poll-seconds", type=float, default=2.0)
    train_worker.add_argument("--lease-seconds", type=int, default=180)
    train_worker.add_argument("--max-jobs", type=int, default=0, help="0 means unlimited.")
    train_worker.add_argument("--once", action="store_true", help="Exit after first claim attempt.")
    train_worker.add_argument(
        "--detach", action="store_true", help="Run worker as detached background process."
    )
    train_worker.add_argument(
        "--detach-log",
        default="",
        help="Optional stdout/stderr log path for detached process.",
    )
    train_worker.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for status events (disabled by default).",
    )
    server_parser.add_argument(
        "--detach", action="store_true", help="Run server as detached background process."
    )
    server_parser.add_argument(
        "--detach-log",
        default="",
        help="Optional stdout/stderr log path for detached process.",
    )

    train_artifacts_fetch = subparsers.add_parser(
        "train-artifacts-fetch",
        help="Fetch training artifacts from server in parallel.",
    )
    train_artifacts_fetch.add_argument("--server-url", default="http://127.0.0.1:8765")
    train_artifacts_fetch.add_argument(
        "--auth-token-env",
        default="GTSM_SERVER_AUTH_TOKEN",
        help="Environment variable containing optional bearer auth token.",
    )
    train_artifacts_fetch.add_argument("--job-id", required=True)
    train_artifacts_fetch.add_argument(
        "--output-dir",
        default=".gtsm-cache/models/remote-artifacts",
        help="Directory to write downloaded artifacts.",
    )
    train_artifacts_fetch.add_argument("--max-workers", type=int, default=4)

    eval_parser = subparsers.add_parser(
        "evaluate-wrappers",
        help="Run validation/evaluation summary over generated wrapper XML files.",
    )
    eval_parser.add_argument(
        "--wrappers",
        nargs="+",
        required=True,
        help="Wrapper/artifact file paths.",
    )
    eval_parser.add_argument(
        "--artifact-format",
        choices=["xml", "udt-yaml"],
        default="xml",
        help="Artifact format to evaluate.",
    )
    eval_parser.add_argument(
        "--report",
        default=".gtsm-cache/runs/evaluation/summary.json",
        help="Path to write evaluation summary JSON report.",
    )
    eval_parser.add_argument(
        "--xsd",
        help="Optional Galaxy tool XSD file path for xmllint schema validation.",
    )
    eval_parser.add_argument(
        "--run-planemo",
        action="store_true",
        help="Run 'planemo lint' for each wrapper when planemo is available.",
    )
    _add_planemo_test_args(eval_parser)

    bench_parser = subparsers.add_parser(
        "benchmark-generate",
        help="Run corpus-scale generation + evaluation for benchmarking.",
    )
    bench_parser.add_argument(
        "--corpus-jsonl",
        default=".gtsm-cache/datasets/tools-iuc-corpus.jsonl",
        help="Input corpus JSONL from extract-corpus.",
    )
    bench_parser.add_argument(
        "--wrappers-dir",
        default=".gtsm-cache/runs/benchmark/wrappers",
        help="Directory for generated wrapper/artifact files.",
    )
    bench_parser.add_argument(
        "--artifact-format",
        choices=["xml", "udt-yaml"],
        default="xml",
        help="Artifact format to generate during benchmarking.",
    )
    bench_parser.add_argument(
        "--generation-records",
        default=".gtsm-cache/runs/benchmark/generation.records.json",
        help="Output JSON file for per-tool generation records.",
    )
    bench_parser.add_argument(
        "--evaluation-report",
        default=".gtsm-cache/runs/benchmark/evaluation.summary.json",
        help="Output JSON file for evaluation summary.",
    )
    bench_parser.add_argument(
        "--benchmark-summary",
        default=".gtsm-cache/runs/benchmark/benchmark.summary.json",
        help="Output JSON file for benchmark summary.",
    )
    bench_parser.add_argument(
        "--provider",
        choices=["local", "openai", "anthropic", "copilot", "ollama"],
        default="local",
        help="Generation provider.",
    )
    bench_parser.add_argument(
        "--model-variant", default="benchmark-variant", help="Model variant label."
    )
    bench_parser.add_argument("--model", default="", help="Provider model override.")
    bench_parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature."
    )
    bench_parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens.")
    bench_parser.add_argument(
        "--ollama-context-tokens",
        type=int,
        default=None,
        help=(
            "Ollama runtime context tokens (num_ctx). "
            "Unset uses GTSM_OLLAMA_CONTEXT_TOKENS or Ollama defaults; 0 disables num_ctx."
        ),
    )
    bench_parser.add_argument(
        "--max-workers", type=int, default=4, help="Parallel workers for generation."
    )
    bench_parser.add_argument(
        "--num-processes",
        type=int,
        default=0,
        help="Native benchmark worker processes for local generation; 0 selects automatically.",
    )
    bench_parser.add_argument(
        "--gpu-devices",
        default="",
        help="Comma-separated GPU ids to bind benchmark worker processes, e.g. 0,1,2,3.",
    )
    bench_parser.add_argument(
        "--min-items-per-process",
        type=int,
        default=DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS,
        help=(
            "Minimum benchmark items per auto-selected worker process. "
            "Only applies when --num-processes is 0."
        ),
    )
    bench_parser.add_argument(
        "--startup-stagger-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between launching native benchmark worker processes.",
    )
    bench_parser.add_argument(
        "--local-gpu-topology",
        choices=["per-process", "model-parallel"],
        default="per-process",
        help="GPU binding topology for local benchmark workers.",
    )
    bench_parser.add_argument(
        "--local-offload-policy",
        choices=["allow", "fail"],
        default="allow",
        help="Whether local PEFT inference may use CPU/disk/model offload.",
    )
    bench_parser.add_argument(
        "--local-gpu-memory-reserve-gib",
        type=float,
        default=2.0,
        help="GiB to reserve per visible GPU when computing local PEFT max_memory.",
    )
    bench_parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Skip successful checkpointed benchmark records with existing wrapper XML.",
    )
    bench_parser.add_argument(
        "--checkpoint-records",
        default="",
        help="JSONL path for per-record benchmark checkpoints.",
    )
    bench_parser.add_argument(
        "--record-timeout-seconds",
        type=float,
        default=0.0,
        help="Parent-enforced timeout for an active benchmark record; 0 disables.",
    )
    bench_parser.add_argument(
        "--max-prompt-help-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_HELP_CHARS,
        help="Maximum help-text characters included in each generation prompt.",
    )
    _add_source_context_args(
        bench_parser,
        include_source_root=True,
        include_source_file=True,
    )
    bench_parser.add_argument(
        "--suite-generation",
        choices=["single", "recommend", "generate"],
        default="single",
        help=(
            "Suite handling for benchmark records: single keeps one wrapper, recommend records "
            "a suite plan, generate writes repository bundles and evaluates all generated XMLs."
        ),
    )
    bench_parser.add_argument(
        "--max-suite-tools",
        type=int,
        default=8,
        help="Maximum suite members when --suite-generation is recommend or generate.",
    )
    bench_parser.add_argument(
        "--repair-invalid-xml",
        dest="repair_invalid_xml",
        action="store_true",
        default=True,
        help="Retry malformed, non-tool, or degenerate benchmark XML once with a stricter repair prompt.",
    )
    bench_parser.add_argument(
        "--no-repair-invalid-xml",
        dest="repair_invalid_xml",
        action="store_false",
        help="Disable automatic repair retries for invalid benchmark XML.",
    )
    bench_parser.add_argument(
        "--allow-compact-fallback",
        action="store_true",
        help=(
            "After repair fails on truncated XML, write a minimal placeholder wrapper "
            "instead of reporting the record as failed. Intended for smoke tests, not "
            "quality benchmarks."
        ),
    )
    bench_parser.add_argument("--limit", type=int, help="Optional max number of corpus records.")
    bench_parser.add_argument("--xsd", help="Optional Galaxy tool XSD file path.")
    bench_parser.add_argument(
        "--run-planemo", action="store_true", help="Run planemo lint during evaluation."
    )
    _add_planemo_test_args(bench_parser)
    bench_parser.add_argument(
        "--allow-stub-local",
        action="store_true",
        help="Allow benchmark-generate to use canned local starter XML for smoke tests.",
    )
    bench_parser.add_argument(
        "--status-log",
        default="",
        help="Optional JSONL file for status events (disabled by default).",
    )
    bench_parser.add_argument(
        "--benchmark-shard-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    bench_summary_parser = subparsers.add_parser(
        "benchmark-summary",
        help="Print a compact summary of a benchmark-generate summary JSON.",
    )
    bench_summary_parser.add_argument(
        "--summary",
        default=".gtsm-cache/runs/benchmark/benchmark.summary.json",
        help="Benchmark summary JSON path.",
    )

    promote_parser = subparsers.add_parser(
        "promote-candidate",
        help="Apply promotion quality gates to candidate benchmark outputs.",
    )
    promote_parser.add_argument(
        "--candidate-summary",
        required=True,
        help="Candidate benchmark summary JSON from benchmark-generate.",
    )
    promote_parser.add_argument(
        "--baseline-summary",
        help="Optional baseline benchmark summary JSON for regression checks.",
    )
    promote_parser.add_argument(
        "--decision-out",
        default=".gtsm-cache/runs/promotion/decision.json",
        help="Output file for promotion decision JSON.",
    )
    promote_parser.add_argument(
        "--policy",
        default="staging",
        help="Promotion policy profile from config/promotion.policies.json.",
    )
    promote_parser.add_argument("--min-generation-success-rate", type=float)
    promote_parser.add_argument("--min-xml-well-formed-rate", type=float)
    promote_parser.add_argument("--max-unknown-datatype-rate", type=float)
    promote_parser.add_argument("--require-xsd-pass", action="store_true")
    promote_parser.add_argument("--require-planemo-pass", action="store_true")
    promote_parser.add_argument("--require-planemo-test-pass", action="store_true")
    promote_parser.add_argument("--baseline-tolerance", type=float)
    return parser


def _paths_table(paths: WorkspacePaths) -> str:
    rows = [
        ("repo_root", paths.repo_root),
        ("cache_root", paths.cache_root),
        ("source_cache", paths.source_cache),
        ("datasets_root", paths.datasets_root),
        ("runs_root", paths.runs_root),
        ("models_root", paths.models_root),
        ("xsd_root", paths.xsd_root),
        ("configs_root", paths.configs_root),
    ]
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"{name.ljust(width)} : {path}" for name, path in rows)


def _print_progress_status(
    progress: dict,
    *,
    label: str,
    status_log_path: Path | None = None,
) -> None:
    if not progress:
        return
    emit_status(
        {
            "status": label,
            "completed": progress.get("completed_units"),
            "total": progress.get("total_units"),
            "elapsed_seconds": progress.get("elapsed_seconds"),
            "units_per_second": progress.get("units_per_second"),
            "eta_seconds": progress.get("eta_seconds"),
            "eta_timestamp": progress.get("eta_timestamp"),
        },
        status_log_path=status_log_path,
    )


def _resolve_cli_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _format_rate(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _format_seconds(value: object) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "0.0s"


def _compact_benchmark_summary(summary: dict) -> str:
    quality = summary.get("quality", {})
    quality = quality if isinstance(quality, dict) else {}
    throughput = quality.get("throughput", {})
    throughput = throughput if isinstance(throughput, dict) else {}
    validity = quality.get("validity", {})
    validity = validity if isinstance(validity, dict) else {}
    repair = quality.get("repair", {})
    repair = repair if isinstance(repair, dict) else {}
    fidelity = quality.get("reference_fidelity", {})
    fidelity = fidelity if isinstance(fidelity, dict) else {}
    startup = summary.get("startup", {})
    startup = startup if isinstance(startup, dict) else {}
    records = fidelity.get("records", [])
    records = records if isinstance(records, list) else []
    failures = summary.get("failures", [])
    failures = failures if isinstance(failures, list) else []

    lines = [
        "Benchmark summary",
        (
            f"attempted={summary.get('attempted', 0)} "
            f"succeeded={summary.get('succeeded', 0)} "
            f"failed={summary.get('failed', 0)}"
        ),
        (
            "throughput="
            f"{_format_rate(throughput.get('wrappers_per_minute'))} wrappers/min "
            f"({_format_seconds(throughput.get('seconds_per_attempted_wrapper'))}/wrapper)"
        ),
        (
            "validity="
            f"success {_format_rate(validity.get('success_rate'))} "
            f"xml {_format_rate(validity.get('xml_well_formed_rate'))} "
            f"tool-root {_format_rate(validity.get('tool_root_rate'))}"
        ),
        (
            "repair="
            f"attempt {_format_rate(repair.get('repair_attempt_rate'))} "
            f"success {_format_rate(repair.get('repair_success_rate'))} "
            f"truncation-failure {_format_rate(repair.get('truncation_failure_rate'))}"
        ),
        (
            "startup="
            f"processes={startup.get('processes', 1)} "
            f"model-load-max={_format_seconds(startup.get('model_load_seconds_max'))} "
            f"model-load-mean={_format_seconds(startup.get('model_load_seconds_mean'))}"
        ),
        (
            "fidelity="
            f"compared={fidelity.get('compared_records', 0)} "
            f"input-error={_format_rate(fidelity.get('avg_input_count_abs_error'))} "
            f"output-error={_format_rate(fidelity.get('avg_output_count_abs_error'))} "
            f"input-dtype={_format_rate(fidelity.get('input_datatype_jaccard_mean'))} "
            f"output-dtype={_format_rate(fidelity.get('output_datatype_jaccard_mean'))} "
            f"command={_format_rate(fidelity.get('primary_command_presence_rate'))}"
        ),
    ]

    if records:
        lines.extend(["", "Per-tool fidelity:"])
        for record in records:
            if not isinstance(record, dict):
                continue
            lines.append(
                " - "
                f"{record.get('tool_name', '')}: "
                f"inputs +-{record.get('input_count_abs_error', 0)}, "
                f"outputs +-{record.get('output_count_abs_error', 0)}, "
                f"input_dtype={_format_rate(record.get('input_datatype_jaccard'))}, "
                f"output_dtype={_format_rate(record.get('output_datatype_jaccard'))}, "
                f"command={bool(record.get('primary_command_present'))}, "
                f"xml={record.get('output_xml_path', '')}"
            )

    if failures:
        lines.extend(["", "Failures:"])
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            lines.append(
                " - "
                f"{failure.get('tool_name', '')}: "
                f"{failure.get('error_type', '')}: {failure.get('error', '')} "
                f"xml={failure.get('output_xml_path', '')}"
            )

    return "\n".join(lines)


def _strip_flag(argv: list[str], flag: str, takes_value: bool) -> list[str]:
    result: list[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == flag:
            i += 2 if takes_value else 1
            continue
        result.append(item)
        i += 1
    return result


def _spawn_detached(argv: list[str], detach_log: Path) -> dict:
    detach_log.parent.mkdir(parents=True, exist_ok=True)
    handle = detach_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "galaxy_toolsmith.cli.main", *argv],
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=handle,
        start_new_session=True,
        close_fds=True,
    )
    return {"pid": process.pid, "log_path": str(detach_log)}


def _safe_pid_file_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return safe.strip("._-") or "host"


def _server_pid_dir(paths: WorkspacePaths) -> Path:
    return paths.cache_root / "server"


def _server_pid_file(paths: WorkspacePaths, *, host: str, port: int) -> Path:
    return _server_pid_dir(paths) / f"serve-{_safe_pid_file_part(host)}-{int(port)}.pid"


def _write_server_pid_file(paths: WorkspacePaths, *, host: str, port: int, pid: int) -> Path:
    path = _server_pid_file(paths, host=host, port=port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "pid": int(pid),
                "host": host,
                "port": int(port),
                "repo_root": str(paths.repo_root),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _remove_server_pid_file(path: Path, *, pid: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if int(payload.get("pid") or -1) == int(pid):
        with suppress(OSError):
            path.unlink()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _ps_process_rows() -> list[dict]:
    user = pwd.getpwuid(os.getuid()).pw_name
    completed = subprocess.run(
        ["ps", "-u", user, "-o", "pid=", "-o", "args="],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    rows: list[dict] = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        pid_text, _, command = value.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        rows.append({"pid": pid, "command": command.strip()})
    return rows


def _token_value(tokens: list[str], flag: str) -> str:
    try:
        index = tokens.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(tokens):
        return ""
    return tokens[index + 1]


def _server_candidate_from_process(
    *,
    row: dict,
    paths: WorkspacePaths,
    host: str,
    port: int | None,
) -> dict | None:
    pid = int(row.get("pid") or -1)
    command = str(row.get("command") or "")
    if pid <= 0 or pid == os.getpid():
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if "galaxy_toolsmith.cli.main" not in tokens or "serve" not in tokens:
        return None
    if "serve-stop" in tokens or "--stop" in tokens:
        return None
    repo_root_text = _token_value(tokens, "--repo-root")
    if repo_root_text and Path(repo_root_text).resolve() != paths.repo_root:
        return None
    command_host = _token_value(tokens, "--host") or "127.0.0.1"
    if host and command_host != host:
        return None
    command_port_text = _token_value(tokens, "--port")
    try:
        command_port = int(command_port_text) if command_port_text else 8765
    except ValueError:
        command_port = 8765
    if port is not None and command_port != int(port):
        return None
    return {
        "pid": pid,
        "command": command,
        "host": command_host,
        "port": command_port,
    }


def _matching_server_processes(
    *,
    paths: WorkspacePaths,
    host: str,
    port: int | None,
) -> list[dict]:
    candidates: dict[int, dict] = {}
    for row in _ps_process_rows():
        candidate = _server_candidate_from_process(
            row=row,
            paths=paths,
            host=host,
            port=port,
        )
        if candidate is not None:
            candidates[int(candidate["pid"])] = candidate

    pid_root = _server_pid_dir(paths)
    if pid_root.exists():
        for path in pid_root.glob("serve-*.pid"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                pid = int(payload.get("pid") or -1)
            except (TypeError, ValueError):
                continue
            if pid in candidates:
                candidates[pid]["pid_file"] = str(path)
    return sorted(candidates.values(), key=lambda item: int(item["pid"]))


def _wait_for_pids_to_exit(pids: list[int], timeout_seconds: float) -> list[int]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    remaining = [int(pid) for pid in pids]
    while remaining and time.monotonic() < deadline:
        remaining = [pid for pid in remaining if _pid_alive(pid)]
        if remaining:
            time.sleep(0.2)
    return [pid for pid in remaining if _pid_alive(pid)]


def _mark_stopped_server_runs(paths: WorkspacePaths, *, pids: set[int]) -> None:
    if not pids:
        return
    try:
        runs = list_monitor_runs(paths, kind="server", limit=500).get("runs", [])
    except Exception:
        return
    for run in runs:
        if str(run.get("status")) != "running":
            continue
        summary = dict(run.get("summary", {}))
        try:
            run_pid = int(summary.get("pid") or -1)
        except (TypeError, ValueError):
            continue
        if run_pid not in pids:
            continue
        try:
            update_monitor_run(
                paths,
                str(run.get("run_id", "")),
                status="completed",
                summary={"stopped": True, "pid": run_pid},
            )
        except Exception:
            continue


def stop_serve_processes(
    *,
    paths: WorkspacePaths,
    host: str = "",
    port: int | None = 8765,
    dry_run: bool = False,
    force: bool = False,
    timeout_seconds: float = 10.0,
) -> dict:
    candidates = _matching_server_processes(paths=paths, host=host, port=port)
    matched = [dict(candidate) for candidate in candidates]
    if dry_run:
        return {
            "matched": matched,
            "terminated": [],
            "still_running": [],
            "skipped": [],
            "dry_run": True,
        }

    terminated: list[dict] = []
    skipped: list[dict] = []
    for candidate in candidates:
        pid = int(candidate["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append({**candidate, "signal": "SIGTERM"})
        except ProcessLookupError:
            terminated.append({**candidate, "signal": "already-exited"})
        except PermissionError as error:
            skipped.append({**candidate, "reason": str(error) or "permission denied"})

    waiting_pids = [int(item["pid"]) for item in terminated if item.get("signal") == "SIGTERM"]
    still_running = _wait_for_pids_to_exit(waiting_pids, timeout_seconds)
    if force and still_running:
        for pid in list(still_running):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                skipped.append({"pid": pid, "reason": str(error) or "permission denied"})
        still_running = _wait_for_pids_to_exit(still_running, 2.0)

    stopped_pids = {int(item["pid"]) for item in terminated} - set(still_running)
    _mark_stopped_server_runs(paths, pids=stopped_pids)
    for candidate in candidates:
        pid_file = str(candidate.get("pid_file") or "")
        if pid_file and int(candidate["pid"]) in stopped_pids:
            _remove_server_pid_file(Path(pid_file), pid=int(candidate["pid"]))

    return {
        "matched": matched,
        "terminated": terminated,
        "still_running": [
            candidate for candidate in matched if int(candidate["pid"]) in set(still_running)
        ],
        "skipped": skipped,
        "dry_run": False,
    }


def _monitor_command() -> list[str]:
    return ["gtsm", *sys.argv[1:]]


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    paths = WorkspacePaths.from_repo_root(Path(args.repo_root))

    if args.command == "doctor":
        print(_paths_table(paths))
        return 0

    if args.command == "init-config":
        paths.create_directories()
        config_file = write_default_config(paths)
        print(f"Wrote default config to {config_file}")
        return 0

    if args.command == "init-workspace":
        paths.create_directories()
        config_file = write_default_config(paths)

        dataset_manifest = DatasetManifest(
            dataset_id="bootstrap-dataset",
            sources=[
                SourceRef(
                    name="tools-iuc",
                    url="https://github.com/galaxyproject/tools-iuc",
                    ref="HEAD",
                )
            ],
            transforms=["bootstrap"],
            includes_tests=True,
            includes_datatype_report=True,
        )
        model_manifest = ModelVariantManifest(
            variant_id="bootstrap-variant",
            base_model="unset",
            quantization="unset",
            training_dataset_id=dataset_manifest.dataset_id,
            provider="unset",
            skills_profile="unset",
        )

        dataset_manifest_path = paths.configs_root / "dataset.manifest.json"
        model_manifest_path = paths.configs_root / "model.manifest.json"
        training_profiles_path = paths.configs_root / "training.profiles.json"
        promotion_policies_path = paths.configs_root / "promotion.policies.json"
        dataset_manifest_path.write_text(dataset_manifest.to_json(), encoding="utf-8")
        model_manifest_path.write_text(model_manifest.to_json(), encoding="utf-8")
        write_default_training_profiles(training_profiles_path)
        write_default_promotion_policies(promotion_policies_path)

        print(
            json.dumps(
                {
                    "config": str(config_file),
                    "dataset_manifest": str(dataset_manifest_path),
                    "model_manifest": str(model_manifest_path),
                    "training_profiles": str(training_profiles_path),
                    "promotion_policies": str(promotion_policies_path),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "list-train-profiles":
        profile_path = paths.configs_root / "training.profiles.json"
        if not profile_path.exists():
            write_default_training_profiles(profile_path)
        profiles = json.loads(profile_path.read_text(encoding="utf-8")).get("profiles", [])
        capabilities = detect_runtime_capabilities()
        enriched: list[dict] = []
        for profile in profiles:
            backend = str(profile.get("backend", "")).lower()
            source_quant = str(profile.get("quantization", "none"))
            quant_state = "pre-quantized" if source_quant != "none" else "non-quantized"
            if backend in {"axolotl", "cuda", "rocm"}:
                supported = capabilities.cuda_available or capabilities.rocm_available
            elif backend in {"mlx-lm", "mlx", "mps"}:
                supported = capabilities.mps_available
            else:
                supported = True
            enriched.append(
                {
                    **profile,
                    "source_quantization_state": quant_state,
                    "intended_methodology_supported": supported,
                    "profile_tier": (
                        "opt_in_evaluation"
                        if "deepseek" in str(profile.get("name", "")).lower()
                        else "default"
                    ),
                    "selection_guidance": (
                        "DeepSeek coding/distilled profile: benchmark against primary defaults before promotion"
                        if "deepseek" in str(profile.get("name", "")).lower()
                        else "Primary default profile set"
                    ),
                    "recommended_flow": (
                        "fine-tune non-quantized first, export quantized variants"
                        if source_quant == "none"
                        else "pre-quantized tuning path (accessibility-first)"
                    ),
                }
            )
        print(json.dumps({"profiles": enriched}, indent=2))
        return 0

    if args.command == "list-model-variants":
        variants_dir = paths.models_root / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        variants = sorted(path.name for path in variants_dir.glob("*.manifest.json"))
        print(json.dumps({"variants": variants}, indent=2))
        return 0

    if args.command == "estimate-model-resources":
        print(model_estimates_json())
        return 0

    if args.command == "estimate-training-tokens":
        profile_path = paths.configs_root / "training.profiles.json"
        if not profile_path.exists():
            write_default_training_profiles(profile_path)
        profile = load_training_profile(profile_path, args.profile)
        source_context = _source_context_settings_from_args(args)
        context_lengths = parse_context_lengths(args.max_seq_lengths)
        source_modes = parse_source_context_modes(
            args.compare_source_context_modes,
            default=source_context.mode,
        )
        result = estimate_training_tokens(
            profile=profile,
            corpus_jsonl_path=Path(args.corpus_jsonl).resolve(),
            repo_root=paths.repo_root,
            artifact_format=args.artifact_format,
            source_context_settings=source_context,
            source_context_modes=source_modes,
            max_seq_lengths=context_lengths,
            source_context_budget_ladder=bool(args.source_context_budget_ladder),
            limit=int(args.limit) if int(args.limit or 0) > 0 else None,
            exact_tokenizer=bool(args.exact_tokenizer),
            chars_per_token=float(args.chars_per_token),
            progress_interval=max(0, int(args.progress_interval)),
            workers=int(args.workers),
            longest_sample_count=max(0, int(args.longest_sample_count)),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "list-promotion-policies":
        policy_path = paths.configs_root / "promotion.policies.json"
        if not policy_path.exists():
            write_default_promotion_policies(policy_path)
        policies = json.loads(policy_path.read_text(encoding="utf-8"))
        print(json.dumps(policies, indent=2))
        return 0

    if args.command == "runtime-detect":
        capabilities = detect_runtime_capabilities()
        print(json.dumps(capabilities.to_dict(), indent=2))
        return 0

    if args.command == "model-cache-info":
        print(json.dumps(model_cache_info(paths), indent=2))
        return 0

    if args.command == "serve-stop":
        port_filter = None if args.all_ports else int(args.port)
        result = stop_serve_processes(
            paths=paths,
            host=args.host,
            port=port_filter,
            dry_run=bool(args.dry_run),
            force=bool(args.force),
            timeout_seconds=float(args.timeout_seconds),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "sync-tools-iuc":
        paths.create_directories()
        result = sync_tools_iuc(paths=paths, ref=args.ref)
        print(
            json.dumps(
                {
                    "name": result.name,
                    "path": str(result.path),
                    "revision": result.revision,
                    "cloned": result.cloned,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "sync-galaxy-skills":
        paths.create_directories()
        result = sync_galaxy_skills(paths=paths, ref=args.ref)
        print(
            json.dumps(
                {
                    "name": result.name,
                    "path": str(result.path),
                    "revision": result.revision,
                    "cloned": result.cloned,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "sync-galaxy-xsd":
        paths.create_directories()
        result = sync_galaxy_xsd(paths=paths, ref=args.ref)
        print(
            json.dumps(
                {
                    "url": result.url,
                    "path": str(result.path),
                    "bytes_written": result.bytes_written,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "extract-corpus":
        status_log_path = resolve_status_log_path(args.status_log)
        if args.tools_root:
            tools_root = Path(args.tools_root).resolve()
        else:
            tools_root = (paths.source_cache / "tools-iuc" / "tools").resolve()
        output_jsonl = Path(args.output).resolve()
        checkpoint_file = Path(args.checkpoint).resolve()
        tracker = create_monitor_run_tracker(
            paths,
            kind="extract",
            command=_monitor_command(),
            inputs={
                "tools_root": str(tools_root),
                "max_workers": args.max_workers,
                "source_workers": args.source_workers,
                "container_prepare_workers": args.container_prepare_workers,
                "container_probe_workers": args.container_probe_workers,
                "container_image_timeout_seconds": args.container_image_timeout_seconds,
                "container_image_quarantine_seconds": args.container_image_quarantine_seconds,
                "container_image_quarantine_file": args.container_image_quarantine_file,
                "container_sif_exec_mode": args.container_sif_exec_mode,
                "fetch_documentation": not args.no_fetch_docs,
                "resolve_containers": args.resolve_containers,
                "execute_containers": args.execute_containers,
                "synthesize_udt_yaml": args.synthesize_udt_yaml,
                "source_download_max_bytes": args.source_download_max_bytes,
                "wrapper_source_max_bytes": args.wrapper_source_max_bytes,
                "wrapper_configfile_max_bytes": args.wrapper_configfile_max_bytes,
                "restart": args.restart,
            },
            outputs={"output": str(output_jsonl), "checkpoint": str(checkpoint_file)},
        )

        settings = ExtractionSettings(
            max_workers=args.max_workers,
            source_workers=args.source_workers,
            container_prepare_workers=args.container_prepare_workers,
            container_probe_workers=args.container_probe_workers,
            retries=args.retries,
            fetch_documentation=not args.no_fetch_docs,
            resolve_containers=args.resolve_containers,
            execute_containers=args.execute_containers,
            container_runtime=args.container_runtime,
            container_cache_dir=Path(args.container_cache_dir).resolve()
            if args.container_cache_dir
            else None,
            container_sif_exec_mode=args.container_sif_exec_mode,
            container_help_probe_mode=args.container_help_probe_mode,
            container_image_timeout_seconds=args.container_image_timeout_seconds,
            container_image_quarantine_seconds=args.container_image_quarantine_seconds,
            container_image_quarantine_file=Path(args.container_image_quarantine_file).resolve()
            if args.container_image_quarantine_file
            else None,
            source_download_timeout_seconds=args.source_download_timeout_seconds,
            source_download_max_bytes=args.source_download_max_bytes,
            singularity_depot_url=args.singularity_depot_url,
            docker_use_sudo=args.docker_use_sudo,
            remove_images_after_use=not args.no_remove_images,
            bioconda_checkout_sources=args.bioconda_checkout_sources,
            bioconda_ref=args.bioconda_ref,
            synthesize_udt_yaml=args.synthesize_udt_yaml,
            wrapper_source_max_bytes=args.wrapper_source_max_bytes,
            wrapper_configfile_max_bytes=args.wrapper_configfile_max_bytes,
            cache_root=paths.source_cache,
            restart=args.restart,
            status_log_path=status_log_path,
            retry_manifest_path=Path(args.retry_manifest).resolve()
            if args.retry_manifest
            else None,
        )
        try:
            result = extract_tools_corpus(
                tools_root=tools_root,
                output_jsonl=output_jsonl,
                checkpoint_file=checkpoint_file,
                settings=settings,
            )
            result["tools_root"] = str(tools_root)
            result["output"] = str(output_jsonl)
            result["checkpoint"] = str(checkpoint_file)
            tracker.complete(
                progress={
                    "completed_units": result.get("processed_now", 0),
                    "total_units": result.get("total_wrappers", result.get("total_records", 0)),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                },
                summary=result,
            )
            print(json.dumps(result, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "rebuild-execution-report":
        corpus_jsonl = Path(args.corpus_jsonl).resolve()
        output_path = Path(args.output).resolve() if args.output else None
        tracker = create_monitor_run_tracker(
            paths,
            kind="diagnostics",
            command=_monitor_command(),
            inputs={"corpus_jsonl": str(corpus_jsonl)},
            outputs={"execution_report_path": str(output_path) if output_path else ""},
        )
        try:
            report_path = rebuild_execution_report_from_jsonl(corpus_jsonl, output_path)
            payload = {"execution_report_path": str(report_path)}
            tracker.complete(outputs=payload, summary=payload)
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "diagnose-corpus":
        diagnostics_dir = Path(args.diagnostics_dir)
        tracker = create_monitor_run_tracker(
            paths,
            kind="diagnostics",
            command=_monitor_command(),
            inputs={
                "execution_report_path": str(Path(args.execution_report)),
                "corpus_jsonl": str(Path(args.corpus_jsonl)) if args.corpus_jsonl else "",
                "sample_limit": args.sample_limit,
            },
            outputs={"diagnostics_dir": str(diagnostics_dir)},
        )
        try:
            result = write_corpus_diagnostics(
                execution_report_path=Path(args.execution_report),
                diagnostics_dir=diagnostics_dir,
                corpus_jsonl=Path(args.corpus_jsonl) if args.corpus_jsonl else None,
                checkpoint_file=Path(args.checkpoint) if args.checkpoint else None,
                current_run_path=Path(args.current_run) if args.current_run else None,
                sample_limit=args.sample_limit,
            )
            tracker.complete(
                outputs={"diagnostics_dir": result.get("diagnostics_dir", "")}, summary=result
            )
            print(json.dumps(result, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "train-runs":
        status_log_path = resolve_status_log_path(args.status_log)
        result = list_local_training_runs(paths, limit=args.limit)
        emit_status(
            {
                "status": "local-training-runs",
                "runs": result.get("summary", {}).get("total", 0),
            },
            status_log_path=status_log_path,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "train-status":
        status_log_path = resolve_status_log_path(args.status_log)
        result = get_local_training_run(paths, args.run_id, tail_lines=args.tail)
        _print_progress_status(
            dict(result.get("progress", {})),
            label="local-training-progress",
            status_log_path=status_log_path,
        )
        emit_status(
            {
                "status": "local-training-status",
                "run_id": dict(result.get("run", {})).get("run_id", ""),
                "run_status": result.get("status", ""),
            },
            status_log_path=status_log_path,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "train":
        training_artifact_format = normalize_training_artifact_format(args.artifact_format)
        status_log_path = resolve_status_log_path(args.status_log)
        source_context = _source_context_settings_from_args(args)
        profile_path = paths.configs_root / "training.profiles.json"
        if not profile_path.exists():
            write_default_training_profiles(profile_path)
        profile = load_training_profile(profile_path, args.profile)
        should_track = not (
            args.internal_distributed_child and os.environ.get("RANK", "0") not in {"", "0"}
        )
        tracker = (
            create_monitor_run_tracker(
                paths,
                kind="training",
                command=_monitor_command(),
                inputs={
                    "profile": args.profile,
                    "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
                    "corpus_jsonl": str(Path(args.corpus_jsonl).resolve()),
                    "backend": args.backend,
                    "artifact_format": format_cli_value(training_artifact_format),
                    "num_processes": args.num_processes,
                    "distributed_strategy": args.distributed_strategy
                    or profile.distributed_strategy,
                    "training_method": args.training_method or profile.training_method,
                    "learning_rate": (
                        args.learning_rate
                        if args.learning_rate is not None
                        else profile.learning_rate
                    ),
                    "max_steps": args.max_steps,
                    "variant_id": args.variant_id,
                    "source_context": source_context.to_dict(),
                },
            )
            if should_track
            else None
        )
        try:
            run = run_training(
                paths=paths,
                profile=profile,
                dataset_manifest_path=Path(args.dataset_manifest).resolve(),
                command_override=args.trainer_command,
                variant_id=args.variant_id,
                corpus_jsonl_path=Path(args.corpus_jsonl).resolve(),
                backend_override=args.backend,
                num_processes=args.num_processes,
                dry_run_backend=args.dry_run_backend,
                artifact_format=training_artifact_format,
                run_id_override=args.internal_run_id or None,
                distributed_child=args.internal_distributed_child,
                profile_overrides=TrainingProfileOverrides(
                    max_seq_length=args.max_seq_length,
                    pad_to_sequence_len=args.pad_to_sequence_len,
                    attn_implementation=args.attn_implementation,
                    per_device_batch_size=args.per_device_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    learning_rate=args.learning_rate,
                    training_method=args.training_method,
                ),
                distributed_strategy=args.distributed_strategy or None,
                status_log_path=status_log_path,
                status_interval_seconds=args.status_interval_seconds,
                stream_logs=args.stream_logs,
                log_tail_lines=args.log_tail_lines,
                source_context_settings=source_context,
                max_steps=args.max_steps,
            )
        except Exception as error:
            if tracker is not None:
                tracker.fail(error)
            raise
        should_print = not (
            args.internal_distributed_child and os.environ.get("RANK", "0") not in {"", "0"}
        )
        if run.status == "completed" and (
            args.post_export_quantizations or args.post_ollama_model_name
        ):
            variant_id = Path(str(run.model_variant_path)).name.replace(".manifest.json", "")
            hook_summary: dict = {}
            if args.post_export_quantizations:
                quantizations = [
                    item.strip()
                    for item in str(args.post_export_quantizations).split(",")
                    if item.strip()
                ]
                export_result = export_model_artifacts(
                    paths=paths,
                    variant_id=variant_id,
                    export_format="all",
                    quantizations=quantizations,
                )
                hook_summary["export"] = json.loads(export_result.to_json())
            if args.post_ollama_model_name:
                normalized_ollama_name = normalize_ollama_model_name(args.post_ollama_model_name)
                modelfile_path = write_ollama_modelfile(
                    paths=paths,
                    variant_id=variant_id,
                    model_name=args.post_ollama_model_name,
                    from_quantization="q4_k_m",
                )
                update_variant_ollama_metadata(
                    paths=paths,
                    variant_id=variant_id,
                    ollama_model_name=normalized_ollama_name.effective,
                    ollama_modelfile_path=str(modelfile_path),
                    requested_ollama_model_name=normalized_ollama_name.requested,
                    export_quantizations=quantizations if args.post_export_quantizations else None,
                )
                hook_summary["ollama_modelfile"] = str(modelfile_path)
                hook_summary["ollama_model_name"] = normalized_ollama_name.effective
                if normalized_ollama_name.changed:
                    hook_summary["requested_ollama_model_name"] = normalized_ollama_name.requested
                if args.post_ollama_create:
                    hook_summary["ollama_create"] = create_ollama_model(
                        modelfile_path=modelfile_path,
                        model_name=args.post_ollama_model_name,
                    )
            metrics_path = Path(run.metrics_path)
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics["post_training_hooks"] = hook_summary
                metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics_path = Path(run.metrics_path)
        if should_print and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            _print_progress_status(dict(metrics.get("progress", {})), label="training-progress")
        if tracker is not None:
            metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {}
            )
            run_payload = json.loads(run.to_json())
            tracker.complete(
                status=run.status
                if run.status in {"completed", "failed", "dry-run"}
                else "completed",
                progress=dict(metrics.get("progress", {})) if isinstance(metrics, dict) else {},
                outputs={
                    "output_dir": run_payload.get("output_dir", ""),
                    "checkpoints_dir": run_payload.get("checkpoints_dir", ""),
                    "metrics_path": run_payload.get("metrics_path", ""),
                    "model_variant_path": run_payload.get("model_variant_path", ""),
                },
                summary={
                    "run_id": run_payload.get("run_id", ""),
                    "profile_name": run_payload.get("profile_name", ""),
                    "backend": run_payload.get("backend", ""),
                    "status": run_payload.get("status", ""),
                    "error": run_payload.get("error", ""),
                },
            )
        if should_print:
            print(run.to_json())
        return 0 if run.status in {"completed", "dry-run"} else 1

    if args.command == "train-remote-submit":
        auth_token = os.getenv(args.auth_token_env)
        payload = {
            "profile_name": args.profile,
            "dataset_manifest_path": str(Path(args.dataset_manifest).resolve()),
            "corpus_jsonl_path": str(Path(args.corpus_jsonl).resolve()),
            "variant_id": args.variant_id,
            "trainer_command": list(args.trainer_command or []),
            "learning_rate": args.learning_rate,
            "training_method": args.training_method,
        }
        result = request_remote_json(
            server_url=args.server_url,
            endpoint="/train/jobs",
            method="POST",
            auth_token=auth_token,
            payload=payload,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "train-remote-status":
        status_log_path = resolve_status_log_path(args.status_log)
        auth_token = os.getenv(args.auth_token_env)
        result = request_remote_json(
            server_url=args.server_url,
            endpoint=f"/train/jobs/{args.job_id}",
            method="GET",
            auth_token=auth_token,
        )
        _print_progress_status(
            dict(result.get("progress", {})),
            label="remote-training-progress",
            status_log_path=status_log_path,
        )
        emit_status(
            {
                "status": "remote-training-status",
                "job_id": args.job_id,
                "job_status": dict(result.get("job", {})).get("status", ""),
                "tasks": len(list(result.get("tasks", []))),
            },
            status_log_path=status_log_path,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "train-worker":
        if args.detach:
            cleaned = _strip_flag(sys.argv[1:], "--detach", takes_value=False)
            cleaned = _strip_flag(cleaned, "--detach-log", takes_value=True)
            detach_log = (
                Path(args.detach_log).resolve()
                if str(args.detach_log).strip()
                else (paths.cache_root / "logs" / "train-worker.detach.log")
            )
            print(json.dumps(_spawn_detached(cleaned, detach_log), indent=2))
            return 0
        status_log_path = resolve_status_log_path(args.status_log)
        auth_token = os.getenv(args.auth_token_env)
        worker_id = args.worker_id.strip() or f"{socket.gethostname()}-{os.getpid()}"
        profile_path = paths.configs_root / "training.profiles.json"
        if not profile_path.exists():
            write_default_training_profiles(profile_path)
        handled = 0
        while True:
            claim = request_remote_json(
                server_url=args.server_url,
                endpoint="/train/tasks/claim",
                method="POST",
                auth_token=auth_token,
                payload={"worker_id": worker_id, "lease_seconds": args.lease_seconds},
            )
            task = claim.get("task")
            if not task:
                if args.once:
                    break
                time.sleep(max(0.2, float(args.poll_seconds)))
                continue
            task_id = str(task.get("task_id", ""))
            task_payload = dict(task.get("payload", {}))
            emit_status(
                {
                    "status": "worker-task-claimed",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "job_id": task.get("job_id", ""),
                },
                status_log_path=status_log_path,
            )
            success = False
            error = ""
            result: dict = {}
            try:
                profile = load_training_profile(profile_path, str(task_payload["profile_name"]))
                run = run_training(
                    paths=paths,
                    profile=profile,
                    dataset_manifest_path=Path(
                        str(task_payload["dataset_manifest_path"])
                    ).resolve(),
                    command_override=list(task_payload.get("trainer_command", [])),
                    variant_id=str(task_payload.get("variant_id", "")).strip() or None,
                    corpus_jsonl_path=Path(str(task_payload["corpus_jsonl_path"])).resolve(),
                    profile_overrides=TrainingProfileOverrides(
                        learning_rate=(
                            float(task_payload["learning_rate"])
                            if task_payload.get("learning_rate") is not None
                            else None
                        ),
                        training_method=(
                            str(task_payload["training_method"])
                            if task_payload.get("training_method")
                            else None
                        ),
                    ),
                )
                run_data = json.loads(run.to_json())
                artifacts: list[dict] = []
                for key in ("model_variant_path", "output_dir", "checkpoints_dir", "metrics_path"):
                    value = str(run_data.get(key, "")).strip()
                    if value and Path(value).exists():
                        artifacts.append({"name": key, "path": str(Path(value).resolve())})
                result = {
                    "training_run": run_data,
                    "artifacts": artifacts,
                }
                metrics_path = Path(str(run_data.get("metrics_path", "")).strip())
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    progress = dict(metrics.get("progress", {}))
                    result["progress"] = progress
                    _print_progress_status(
                        progress,
                        label="worker-training-progress",
                        status_log_path=status_log_path,
                    )
                success = run_data.get("status") == "completed"
                if not success:
                    error = str(run_data.get("error", "training_failed"))
            except Exception as exc:  # pragma: no cover - runtime failure path
                error = str(exc)
                result = {}
                success = False
            request_remote_json(
                server_url=args.server_url,
                endpoint=f"/train/tasks/{task_id}/complete",
                method="POST",
                auth_token=auth_token,
                payload={
                    "worker_id": worker_id,
                    "success": success,
                    "result": result,
                    "error": error,
                },
            )
            emit_status(
                {
                    "status": "worker-task-completed",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "success": success,
                    "error": error,
                },
                status_log_path=status_log_path,
            )
            handled += 1
            if args.max_jobs > 0 and handled >= int(args.max_jobs):
                break
            if args.once:
                break
        emit_status(
            {"status": "worker-finished", "worker_id": worker_id, "jobs_handled": handled},
            status_log_path=status_log_path,
        )
        print(json.dumps({"worker_id": worker_id, "jobs_handled": handled}, indent=2))
        return 0

    if args.command == "train-artifacts-fetch":
        auth_token = os.getenv(args.auth_token_env)
        output_dir = Path(args.output_dir).resolve()
        tracker = create_monitor_run_tracker(
            paths,
            kind="export",
            command=_monitor_command(),
            inputs={
                "server_url": args.server_url,
                "job_id": args.job_id,
                "max_workers": args.max_workers,
            },
            outputs={"output_dir": str(output_dir)},
        )
        try:
            summary = fetch_training_artifacts_parallel(
                server_url=args.server_url,
                job_id=args.job_id,
                output_dir=output_dir,
                auth_token=auth_token,
                max_workers=args.max_workers,
            )
            tracker.complete(outputs={"output_dir": str(output_dir)}, summary=summary)
            print(json.dumps(summary, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "export-model":
        paths.create_directories()
        quantizations = [item.strip() for item in args.quantizations.split(",") if item.strip()]
        tracker = create_monitor_run_tracker(
            paths,
            kind="export",
            command=_monitor_command(),
            inputs={
                "variant_id": args.variant_id,
                "format": args.format,
                "quantizations": quantizations,
            },
        )
        try:
            result = export_model_artifacts(
                paths=paths,
                variant_id=args.variant_id,
                export_format=args.format,
                quantizations=quantizations,
            )
            payload = json.loads(result.to_json())
            tracker.complete(
                outputs={
                    "merged_path": payload.get("merged_path", ""),
                    "gguf_path": payload.get("gguf_path", ""),
                    "ollama_modelfile_path": payload.get("ollama_modelfile_path", ""),
                },
                summary={
                    "variant_id": payload.get("variant_id", args.variant_id),
                    "status": payload.get("status", ""),
                    "quantizations": payload.get("quantizations", []),
                    "notes": payload.get("notes", []),
                },
            )
            print(result.to_json())
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "export-ollama-model":
        paths.create_directories()
        normalized_ollama_name = normalize_ollama_model_name(args.model_name)
        tracker = create_monitor_run_tracker(
            paths,
            kind="export",
            command=_monitor_command(),
            inputs={
                "variant_id": args.variant_id,
                "model_name": normalized_ollama_name.effective,
                "requested_ollama_model_name": normalized_ollama_name.requested,
                "from_quantization": args.from_quantization,
                "create": args.create,
            },
        )
        try:
            modelfile_path = write_ollama_modelfile(
                paths=paths,
                variant_id=args.variant_id,
                model_name=args.model_name,
                from_quantization=args.from_quantization,
            )
            update_variant_ollama_metadata(
                paths=paths,
                variant_id=args.variant_id,
                ollama_model_name=normalized_ollama_name.effective,
                ollama_modelfile_path=str(modelfile_path),
                requested_ollama_model_name=normalized_ollama_name.requested,
                export_quantizations=[args.from_quantization],
            )
            payload: dict = {
                "variant_id": args.variant_id,
                "model_name": normalized_ollama_name.effective,
                "modelfile_path": str(modelfile_path),
            }
            if normalized_ollama_name.changed:
                payload["requested_ollama_model_name"] = normalized_ollama_name.requested
            if args.create:
                payload["create"] = create_ollama_model(
                    modelfile_path=modelfile_path, model_name=args.model_name
                )
            tracker.complete(
                outputs={"modelfile_path": str(modelfile_path)},
                summary={
                    "variant_id": args.variant_id,
                    "model_name": normalized_ollama_name.effective,
                    "requested_ollama_model_name": normalized_ollama_name.requested,
                    "created": bool(args.create),
                },
            )
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "convert-adapter":
        paths.create_directories()
        tracker = create_monitor_run_tracker(
            paths,
            kind="export",
            command=_monitor_command(),
            inputs={
                "from": args.from_format,
                "to": args.to_format,
                "base_model": args.base_model,
                "adapter_dir": str(Path(args.adapter_dir).resolve()),
                "output_dir": str(Path(args.output_dir).resolve()),
            },
        )
        try:
            if args.from_format == "mlx" and args.to_format == "peft":
                result = convert_mlx_lora_to_peft(
                    base_model=args.base_model,
                    adapter_dir=Path(args.adapter_dir).resolve(),
                    output_dir=Path(args.output_dir).resolve(),
                )
            else:  # pragma: no cover - parser choices currently prevent this.
                raise ValueError(
                    f"Unsupported adapter conversion: {args.from_format} -> {args.to_format}"
                )
            payload = json.loads(result.to_json())
            tracker.complete(
                outputs={"output_dir": payload.get("output_dir", "")},
                summary={
                    "status": payload.get("status", ""),
                    "converted_tensors": payload.get("converted_tensors", 0),
                    "architecture": payload.get("architecture", ""),
                },
            )
            print(result.to_json())
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "generate-wrapper":
        paths.create_directories()
        artifact_format = normalize_artifact_format(args.artifact_format)
        source_path = Path(args.source_file).resolve() if args.source_file else None
        source_archive_resolution = None
        source_archive_root = None
        if str(getattr(args, "source_archive", "") or "").strip():
            source_archive_resolution = resolve_source_archive(
                str(args.source_archive),
                cache_root=paths.cache_root / "manual-sources" / "archives",
                max_bytes=int(args.source_archive_max_bytes),
                timeout_seconds=int(args.source_archive_timeout_seconds),
            )
            source_archive_root = Path(source_archive_resolution.extracted_root).resolve()
        discovery = _run_runtime_discovery_from_args(args, paths=paths)
        help_text = _generation_help_text_from_args(args, discovery)
        source_context = _source_context_settings_from_args(
            args,
            source_root=_source_root_override(
                args,
                source_archive_root=source_archive_root,
                discovery=discovery,
            ),
        )
        repository_output_dir = (
            Path(args.repository_output_dir).resolve()
            if str(args.repository_output_dir).strip()
            else None
        )
        if repository_output_dir is not None and artifact_format != ARTIFACT_FORMAT_XML:
            raise ValueError("--repository-output-dir currently supports XML generation only.")
        output_path = _repository_output_path(
            repository_output_dir=repository_output_dir,
            output_value=args.output,
            tool_name=args.tool_name,
        )
        raw_response_log_path = (
            Path(args.raw_response_log).resolve()
            if str(args.raw_response_log).strip()
            else (
                (
                    repository_output_dir / ".gtsm" / "raw" / f"{output_path.stem}.log"
                    if repository_output_dir is not None
                    else output_path.with_name(f"{output_path.name}.raw-response.log")
                )
                if args.stream_output
                else None
            )
        )
        sidecar_output_dir = (
            Path(args.sidecar_output_dir).resolve()
            if str(args.sidecar_output_dir).strip()
            else (repository_output_dir if repository_output_dir is not None else None)
        )
        tracker = create_monitor_run_tracker(
            paths,
            kind="inference",
            command=_monitor_command(),
            inputs={
                "tool_name": args.tool_name,
                "tool_id": args.tool_id,
                "help_text_file": str(_help_text_path_from_args(args) or ""),
                "source_file": str(source_path) if source_path else "",
                "source_archive": str(getattr(args, "source_archive", "") or ""),
                "source_archive_resolution": (
                    source_archive_resolution.to_dict() if source_archive_resolution else {}
                ),
                "runtime_discovery": discovery.to_dict() if discovery else {},
                "repository_output_dir": str(repository_output_dir or ""),
                "source_root": str(source_context.source_root or ""),
                "source_context": source_context.to_dict(),
                "provider": args.provider,
                "model": args.model,
                "model_variant": args.model_variant,
                "skills_profile": args.skills_profile,
                "artifact_format": format_cli_value(artifact_format),
                "tool_granularity": args.tool_granularity,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "ollama_context_tokens": args.ollama_context_tokens,
                "max_prompt_help_chars": args.max_prompt_help_chars,
                "repair_invalid_xml": args.repair_invalid_xml,
                "stream_output": args.stream_output,
                "raw_response_log_path": str(raw_response_log_path or ""),
                "generate_sidecars": args.generate_sidecars,
                "sidecar_output_dir": str(sidecar_output_dir or ""),
                "include_toolsmith_citation": bool(args.include_toolsmith_citation),
                "datatype_scaffold": bool(args.datatype_scaffold),
            },
            outputs={
                "output_path": str(output_path),
                "output_xml_path": str(output_path)
                if artifact_format == ARTIFACT_FORMAT_XML
                else "",
                "output_udt_yaml_path": str(output_path)
                if artifact_format == ARTIFACT_FORMAT_UDT_YAML
                else "",
                "raw_response_log_path": str(raw_response_log_path or ""),
                "sidecar_output_dir": str(sidecar_output_dir or ""),
                "shed_yml_path": str((repository_output_dir / ".shed.yml") if repository_output_dir else ""),
            },
        )
        try:
            record = generate_wrapper(
                paths=paths,
                tool_name=args.tool_name,
                help_text_path=_help_text_path_from_args(args),
                help_text=help_text,
                source_path=source_path,
                output_path=output_path,
                provider_name=args.provider,
                model_variant=args.model_variant,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ollama_context_tokens=args.ollama_context_tokens,
                skills_profile=args.skills_profile,
                allow_stub_local=args.allow_stub_local,
                max_prompt_help_chars=args.max_prompt_help_chars,
                artifact_format=artifact_format,
                source_context_settings=source_context,
                repair_invalid_xml=args.repair_invalid_xml,
                raw_response_log_path=raw_response_log_path,
                stream_output=args.stream_output,
                generate_sidecars=args.generate_sidecars,
                sidecar_output_dir=sidecar_output_dir,
                tool_granularity=args.tool_granularity,
                tool_id=args.tool_id,
                tool_display_name=args.tool_name,
                include_toolsmith_citation=bool(args.include_toolsmith_citation),
                toolsmith_citation_mode=(
                    "macro"
                    if repository_output_dir is not None and args.generate_sidecars
                    else "direct"
                ),
                datatype_scaffold=bool(args.datatype_scaffold),
                datatype_scaffold_dir=repository_output_dir,
                datatype_scaffold_repository_style=repository_output_dir is not None,
            )
            record_payload = json.loads(record.to_json())
            shed_yml_path = ""
            if repository_output_dir is not None:
                metadata = _shed_metadata_from_args(
                    args,
                    default_name=safe_repository_name(args.tool_name),
                    default_description=f"Generated Galaxy Toolsmith repository for {args.tool_name}",
                    suite=False,
                )
                if not args.no_shed_yml:
                    shed_yml_path = str(write_shed_yml(repository_output_dir / ".shed.yml", metadata))
                if discovery is not None:
                    write_gtsm_json(
                        repository_output_dir / ".gtsm" / "runtime-discovery.json",
                        discovery.to_dict(),
                    )
                write_gtsm_json(
                    repository_output_dir / ".gtsm" / "generation-record.json",
                    {
                        "record": record_payload,
                        "shed_metadata": metadata.to_dict(),
                        "shed_yml_path": shed_yml_path,
                        "runtime_discovery": discovery.to_dict() if discovery else {},
                    },
                )
            tracker.complete(
                outputs={
                    "output_path": record_payload.get("output_path", str(output_path)),
                    "output_xml_path": record_payload.get("output_xml_path", str(output_path)),
                    "output_udt_yaml_path": record_payload.get("output_udt_yaml_path", ""),
                    "report_path": record_payload.get("report_path", ""),
                    "raw_response_log_path": record_payload.get("raw_response_log_path", ""),
                    "sidecar_artifacts": record_payload.get("sidecar_artifacts", []),
                    "datatype_scaffold": record_payload.get("datatype_scaffold", {}),
                    "shed_yml_path": shed_yml_path,
                },
                summary={
                    "provider": record_payload.get("provider", args.provider),
                    "model_variant": record_payload.get("model_variant", args.model_variant),
                    "validation": record_payload.get("validation", {}),
                },
            )
            print(record.to_json())
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "plan-suite":
        paths.create_directories()
        discovery = _run_runtime_discovery_from_args(args, paths=paths)
        help_text = _suite_generation_help_text_from_args(args, discovery)
        source_code = (
            Path(args.source_file).resolve().read_text(encoding="utf-8")
            if str(args.source_file).strip()
            else ""
        )
        plan = plan_suite_from_content(
            tool_name=args.tool_name,
            help_text=help_text,
            source_code=source_code,
            max_suite_tools=args.max_suite_tools,
            force_suite=args.force_suite,
        )
        print(plan.to_json())
        return 0

    if args.command == "generate-suite":
        paths.create_directories()
        source_archive_resolution = None
        source_archive_root = None
        if str(getattr(args, "source_archive", "") or "").strip():
            source_archive_resolution = resolve_source_archive(
                str(args.source_archive),
                cache_root=paths.cache_root / "manual-sources" / "archives",
                max_bytes=int(args.source_archive_max_bytes),
                timeout_seconds=int(args.source_archive_timeout_seconds),
            )
            source_archive_root = Path(source_archive_resolution.extracted_root).resolve()
        discovery = _run_runtime_discovery_from_args(args, paths=paths)
        help_text = _suite_generation_help_text_from_args(args, discovery)
        source_context = _source_context_settings_from_args(
            args,
            source_root=_source_root_override(
                args,
                source_archive_root=source_archive_root,
                discovery=discovery,
            ),
        )
        output_dir = Path(args.output_dir).resolve()
        metadata = _shed_metadata_from_args(
            args,
            default_name=f"suite_{safe_repository_name(args.tool_name)}",
            default_description=f"Generated Galaxy Toolsmith suite for {args.tool_name}",
            suite=True,
            repositories=[],
        )
        tracker = create_monitor_run_tracker(
            paths,
            kind="inference",
            command=_monitor_command(),
            inputs={
                "tool_name": args.tool_name,
                "help_text_file": str(_help_text_path_from_args(args) or ""),
                "source_file": str(Path(args.source_file).resolve()) if args.source_file else "",
                "source_archive": str(getattr(args, "source_archive", "") or ""),
                "source_archive_resolution": (
                    source_archive_resolution.to_dict() if source_archive_resolution else {}
                ),
                "runtime_discovery": discovery.to_dict() if discovery else {},
                "source_context": source_context.to_dict(),
                "provider": args.provider,
                "model": args.model,
                "model_variant": args.model_variant,
                "skills_profile": args.skills_profile,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "ollama_context_tokens": args.ollama_context_tokens,
                "local_offload_policy": args.local_offload_policy,
                "local_gpu_memory_reserve_gib": args.local_gpu_memory_reserve_gib,
                "max_suite_tools": args.max_suite_tools,
                "generate_sidecars": args.generate_sidecars,
                "include_toolsmith_citation": bool(args.include_toolsmith_citation),
                "datatype_scaffold": bool(args.datatype_scaffold),
                "raw_response_logs": args.raw_response_logs,
                "stream_output": args.stream_output,
                "repair_invalid_xml": args.repair_invalid_xml,
                "write_shed": not args.no_shed_yml,
            },
            outputs={"output_dir": str(output_dir)},
        )
        try:
            record = generate_suite(
                paths=paths,
                tool_name=args.tool_name,
                help_text_path=_help_text_path_from_args(args),
                help_text=help_text,
                source_path=Path(args.source_file).resolve() if args.source_file else None,
                output_dir=output_dir,
                provider_name=args.provider,
                model_variant=args.model_variant,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ollama_context_tokens=args.ollama_context_tokens,
                skills_profile=args.skills_profile,
                allow_stub_local=args.allow_stub_local,
                max_prompt_help_chars=args.max_prompt_help_chars,
                local_offload_policy=args.local_offload_policy,
                local_gpu_memory_reserve_gib=args.local_gpu_memory_reserve_gib,
                source_context_settings=source_context,
                max_suite_tools=args.max_suite_tools,
                generate_sidecars=args.generate_sidecars,
                raw_response_logs=args.raw_response_logs,
                stream_output=args.stream_output,
                repair_invalid_xml=args.repair_invalid_xml,
                shed_metadata=metadata,
                write_shed=not args.no_shed_yml,
                subcommand_help=dict(discovery.subcommand_help) if discovery else None,
                include_toolsmith_citation=bool(args.include_toolsmith_citation),
                datatype_scaffold=bool(args.datatype_scaffold),
            )
            if discovery is not None:
                write_gtsm_json(output_dir / ".gtsm" / "runtime-discovery.json", discovery.to_dict())
            payload = record.to_dict()
            tracker.complete(
                outputs={
                    "output_dir": payload.get("output_dir", str(output_dir)),
                    "shed_yml_path": payload.get("shed_yml_path", ""),
                    "generated_files": payload.get("generated_files", []),
                    "manifest_path": payload.get("manifest_path", ""),
                },
                summary={
                    "suite_plan": payload.get("suite_plan", {}),
                    "generated_file_count": len(payload.get("generated_files", [])),
                    "warnings": payload.get("warnings", []),
                },
            )
            print(record.to_json())
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "compare-generation-runs":
        left_run_dir = Path(args.left_run_dir).resolve()
        right_run_dir = Path(args.right_run_dir).resolve()
        output_path = Path(args.output).resolve() if str(args.output).strip() else None
        result = compare_generation_run_dirs(
            left_run_dir=left_run_dir,
            right_run_dir=right_run_dir,
            output_path=output_path,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "serve":
        if args.stop:
            host_filter = args.host if "--host" in sys.argv else ""
            result = stop_serve_processes(
                paths=paths,
                host=host_filter,
                port=int(args.port),
                dry_run=bool(args.dry_run),
                force=bool(args.force),
                timeout_seconds=float(args.timeout_seconds),
            )
            print(json.dumps(result, indent=2))
            return 0
        if args.detach:
            cleaned = _strip_flag(sys.argv[1:], "--detach", takes_value=False)
            cleaned = _strip_flag(cleaned, "--detach-log", takes_value=True)
            detach_log = (
                Path(args.detach_log).resolve()
                if str(args.detach_log).strip()
                else (paths.cache_root / "logs" / "serve.detach.log")
            )
            detached = _spawn_detached(cleaned, detach_log)
            pid_file = _write_server_pid_file(
                paths,
                host=args.host,
                port=int(args.port),
                pid=int(detached["pid"]),
            )
            print(json.dumps({**detached, "pid_file": str(pid_file)}, indent=2))
            return 0
        status_log_path = resolve_status_log_path(args.status_log)
        auth_tokens: list[str] = []
        env_token = os.getenv(args.auth_token_env, "").strip()
        if env_token:
            auth_tokens.append(env_token)
        auth_tokens.extend([token for token in args.auth_token if str(token).strip()])
        if args.auth_tokens_file:
            token_file = Path(args.auth_tokens_file).resolve()
            if token_file.exists():
                for line in token_file.read_text(encoding="utf-8").splitlines():
                    value = line.strip()
                    if value:
                        auth_tokens.append(value)
        unique_tokens = sorted(set(auth_tokens))
        tracker = create_monitor_run_tracker(
            paths,
            kind="server",
            command=_monitor_command(),
            inputs={
                "host": args.host,
                "port": args.port,
                "provider": args.provider,
                "model": args.model,
                "model_variant": args.model_variant,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "ollama_context_tokens": args.ollama_context_tokens,
                "max_prompt_help_chars": args.max_prompt_help_chars,
                "auth_enabled": bool(unique_tokens),
                "generate_auth_required": bool(args.require_generate_auth),
            },
        )
        pid_file = _write_server_pid_file(
            paths, host=args.host, port=int(args.port), pid=os.getpid()
        )
        tracker.update(
            outputs={"pid_file": str(pid_file)},
            summary={"pid": os.getpid(), "host": args.host, "port": args.port},
        )
        try:
            serve(
                host=args.host,
                port=args.port,
                provider=args.provider,
                model=args.model,
                model_variant=args.model_variant,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ollama_context_tokens=args.ollama_context_tokens,
                max_prompt_help_chars=args.max_prompt_help_chars,
                auth_tokens=unique_tokens,
                require_generate_auth=bool(args.require_generate_auth),
                allow_stub_local=args.allow_stub_local,
                repo_root=paths.repo_root,
                status_log_path=status_log_path,
            )
            tracker.complete(summary={"host": args.host, "port": args.port})
            return 0
        except Exception as error:
            tracker.fail(error)
            raise
        finally:
            _remove_server_pid_file(pid_file, pid=os.getpid())

    if args.command == "generate-wrapper-remote":
        artifact_format = normalize_artifact_format(args.artifact_format)
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        help_text = Path(args.help_text_file).resolve().read_text(encoding="utf-8")
        source_code = (
            Path(args.source_file).resolve().read_text(encoding="utf-8") if args.source_file else ""
        )
        auth_token = os.getenv(args.auth_token_env)
        tracker = create_monitor_run_tracker(
            paths,
            kind="inference",
            command=_monitor_command(),
            inputs={
                "tool_name": args.tool_name,
                "tool_id": args.tool_id,
                "server_url": args.server_url,
                "provider": args.provider,
                "model": args.model,
                "model_variant": args.model_variant,
                "skills_profile": args.skills_profile,
                "artifact_format": format_cli_value(artifact_format),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_prompt_help_chars": args.max_prompt_help_chars,
                "include_toolsmith_citation": bool(args.include_toolsmith_citation),
                "datatype_scaffold": bool(args.datatype_scaffold),
            },
            outputs={
                "output_path": str(output_path),
                "output_xml_path": str(output_path)
                if artifact_format == ARTIFACT_FORMAT_XML
                else "",
                "output_udt_yaml_path": str(output_path)
                if artifact_format == ARTIFACT_FORMAT_UDT_YAML
                else "",
            },
        )
        try:
            result = request_remote_generation(
                server_url=args.server_url,
                auth_token=auth_token,
                payload={
                    "tool_name": args.tool_name,
                    "tool_id": args.tool_id,
                    "help_text": help_text,
                    "source_code": source_code,
                    "provider": args.provider,
                    "model": args.model,
                    "model_variant": args.model_variant,
                    "skills_profile": args.skills_profile,
                    "artifact_format": artifact_format,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "max_prompt_help_chars": args.max_prompt_help_chars,
                    "include_toolsmith_citation": bool(args.include_toolsmith_citation),
                },
            )
            artifact_text = str(
                result.get("artifact_text")
                or result.get("xml_wrapper")
                or result.get("udt_yaml")
                or ""
            )
            output_path.write_text(artifact_text, encoding="utf-8")

            request_id = f"remote-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
            report_path = paths.runs_root / "generation-remote" / request_id / "report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result.get("validation", {}), indent=2), encoding="utf-8"
            )
            payload = {
                "request_id": request_id,
                "server_url": args.server_url,
                "tool_name": args.tool_name,
                "provider": result.get("provider", args.provider),
                "model_variant": result.get("model_variant", args.model_variant),
                "skills_profile": result.get("skills_profile", args.skills_profile),
                "artifact_format": result.get("artifact_format", artifact_format),
                "output_path": str(output_path),
                "output_xml_path": str(output_path)
                if artifact_format == ARTIFACT_FORMAT_XML
                else "",
                "output_udt_yaml_path": str(output_path)
                if artifact_format == ARTIFACT_FORMAT_UDT_YAML
                else "",
                "report_path": str(report_path),
                "validation": result.get("validation", {}),
            }
            if args.datatype_scaffold:
                payload["datatype_scaffold"] = write_datatype_scaffold(
                    datatype_scaffold_dir_for_output(output_path),
                    [payload],
                    repository_style=False,
                )
            tracker.complete(
                outputs={
                    "output_path": str(output_path),
                    "output_xml_path": payload["output_xml_path"],
                    "output_udt_yaml_path": payload["output_udt_yaml_path"],
                    "report_path": str(report_path),
                    "datatype_scaffold": payload.get("datatype_scaffold", {}),
                },
                summary={
                    "provider": payload["provider"],
                    "model_variant": payload["model_variant"],
                    "validation": payload["validation"],
                },
            )
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "generate-suite-remote":
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        help_text = Path(args.help_text_file).resolve().read_text(encoding="utf-8")
        source_code = (
            Path(args.source_file).resolve().read_text(encoding="utf-8")
            if args.source_file
            else ""
        )
        auth_token = os.getenv(args.auth_token_env)
        metadata = _shed_metadata_from_args(
            args,
            default_name=f"suite_{safe_repository_name(args.tool_name)}",
            default_description=f"Generated Galaxy Toolsmith suite for {args.tool_name}",
            suite=True,
            repositories=[],
        )
        tracker = create_monitor_run_tracker(
            paths,
            kind="inference",
            command=_monitor_command(),
            inputs={
                "tool_name": args.tool_name,
                "server_url": args.server_url,
                "provider": args.provider,
                "model": args.model,
                "model_variant": args.model_variant,
                "skills_profile": args.skills_profile,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_prompt_help_chars": args.max_prompt_help_chars,
                "max_suite_tools": args.max_suite_tools,
                "generate_sidecars": args.generate_sidecars,
                "repair_invalid_xml": args.repair_invalid_xml,
                "include_toolsmith_citation": bool(args.include_toolsmith_citation),
                "datatype_scaffold": bool(args.datatype_scaffold),
            },
            outputs={"output_dir": str(output_dir)},
        )
        try:
            result = request_remote_json(
                server_url=args.server_url,
                endpoint="/generate-suite",
                method="POST",
                auth_token=auth_token,
                payload={
                    "tool_name": args.tool_name,
                    "help_text": help_text,
                    "source_code": source_code,
                    "provider": args.provider,
                    "model": args.model,
                    "model_variant": args.model_variant,
                    "skills_profile": args.skills_profile,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "max_prompt_help_chars": args.max_prompt_help_chars,
                    "max_suite_tools": args.max_suite_tools,
                    "generate_sidecars": args.generate_sidecars,
                    "repair_invalid_xml": args.repair_invalid_xml,
                    "include_toolsmith_citation": bool(args.include_toolsmith_citation),
                    "datatype_scaffold": bool(args.datatype_scaffold),
                    "shed_metadata": metadata.to_dict(),
                    "no_shed_yml": args.no_shed_yml,
                },
                timeout=900.0,
            )
            written: list[dict] = []
            for artifact in result.get("artifacts", []):
                if not isinstance(artifact, dict):
                    continue
                relative_path = str(artifact.get("relative_path") or "").strip()
                if not relative_path or relative_path.startswith("../") or "/../" in relative_path:
                    continue
                target = output_dir / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(artifact.get("content") or ""), encoding="utf-8")
                written.append({"relative_path": relative_path, "path": str(target)})
            manifest_path = output_dir / ".gtsm" / "remote-suite-result.json"
            write_gtsm_json(manifest_path, {**result, "local_written": written})
            tracker.complete(
                outputs={
                    "output_dir": str(output_dir),
                    "written": written,
                    "manifest_path": str(manifest_path),
                },
                summary={
                    "suite_plan": result.get("suite_plan", {}),
                    "generated_file_count": len(result.get("generated_files", [])),
                },
            )
            print(json.dumps({"output_dir": str(output_dir), "written": written}, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "convert-udt":
        input_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve()
        report_path = Path(args.report).resolve() if args.report else None
        tracker = create_monitor_run_tracker(
            paths,
            kind="inference",
            command=_monitor_command(),
            inputs={
                "input": str(input_path),
                "allow_lossy_conversion": bool(args.allow_lossy_conversion),
            },
            outputs={"output": str(output_path), "report": str(report_path or "")},
        )
        try:
            udt_yaml = input_path.read_text(encoding="utf-8")
            validation = validate_udt_yaml(udt_yaml, check_conversion=True).to_dict()
            result = udt_yaml_to_tool_xml(
                udt_yaml,
                allow_lossy_conversion=bool(args.allow_lossy_conversion),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.xml, encoding="utf-8")
            payload = {
                "input": str(input_path),
                "output": str(output_path),
                "validation": validation,
                "conversion": result.to_dict(),
            }
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tracker.complete(
                outputs={"output": str(output_path), "report": str(report_path or "")},
                summary=payload,
            )
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "evaluate-wrappers":
        artifact_format = normalize_artifact_format(args.artifact_format)
        wrapper_paths = [Path(path).resolve() for path in args.wrappers]
        xsd_path = Path(args.xsd).resolve() if args.xsd else None
        report_path = Path(args.report).resolve()
        planemo_test_options = _planemo_test_options_from_args(args)
        tracker = create_monitor_run_tracker(
            paths,
            kind="evaluation",
            command=_monitor_command(),
            inputs={
                "wrappers": [str(path) for path in wrapper_paths],
                "artifact_format": format_cli_value(artifact_format),
                "xsd": str(xsd_path) if xsd_path else "",
                "run_planemo": args.run_planemo,
                "run_planemo_tests": args.run_planemo_tests,
                "planemo_test_output_dir": str(planemo_test_options.output_dir or ""),
                "planemo_test_timeout": planemo_test_options.timeout_seconds,
            },
            outputs={"report": str(report_path)},
        )
        try:
            summary = evaluate_wrapper_paths(
                wrapper_paths=wrapper_paths,
                output_report=report_path,
                xsd_path=xsd_path,
                run_planemo=args.run_planemo,
                run_planemo_tests=args.run_planemo_tests,
                planemo_test_options=planemo_test_options,
                artifact_format=artifact_format,
            )
            payload = summary.to_dict()
            tracker.complete(outputs={"report": str(report_path)}, summary=payload)
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    if args.command == "benchmark-summary":
        summary_path = _resolve_cli_path(paths.repo_root, args.summary).resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(_compact_benchmark_summary(summary))
        return 0

    if args.command == "benchmark-generate":
        artifact_format = normalize_artifact_format(args.artifact_format)
        status_log_path = resolve_status_log_path(args.status_log)
        paths.create_directories()
        xsd_path = Path(args.xsd).resolve() if args.xsd else None
        corpus_jsonl = Path(args.corpus_jsonl).resolve()
        wrappers_dir = Path(args.wrappers_dir).resolve()
        generation_records = Path(args.generation_records).resolve()
        evaluation_report = Path(args.evaluation_report).resolve()
        benchmark_summary_path = Path(args.benchmark_summary).resolve()
        planemo_test_options = _planemo_test_options_from_args(args)
        source_context = _source_context_settings_from_args(args)
        checkpoint_records_path = (
            Path(args.checkpoint_records).resolve() if args.checkpoint_records else None
        )

        if args.benchmark_shard_worker:
            summary = run_benchmark_generation(
                paths=paths,
                corpus_jsonl=corpus_jsonl,
                wrappers_dir=wrappers_dir,
                generation_records_path=generation_records,
                evaluation_report_path=evaluation_report,
                provider=args.provider,
                model_variant=args.model_variant,
                model=args.model,
                artifact_format=artifact_format,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ollama_context_tokens=args.ollama_context_tokens,
                max_workers=args.max_workers,
                limit=args.limit,
                xsd_path=xsd_path,
                run_planemo=args.run_planemo,
                run_planemo_tests=False,
                planemo_test_options=planemo_test_options,
                allow_stub_local=args.allow_stub_local,
                status_sink=lambda payload: emit_status(payload, status_log_path=status_log_path),
                repair_invalid_xml=args.repair_invalid_xml,
                allow_compact_fallback=args.allow_compact_fallback,
                max_prompt_help_chars=args.max_prompt_help_chars,
                local_gpu_topology=args.local_gpu_topology,
                local_offload_policy=args.local_offload_policy,
                local_gpu_memory_reserve_gib=args.local_gpu_memory_reserve_gib,
                resume_existing=args.resume_existing,
                checkpoint_records_path=checkpoint_records_path,
                source_context_settings=source_context,
                suite_generation=args.suite_generation,
                max_suite_tools=args.max_suite_tools,
            )
            benchmark_summary_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark_summary_path.write_text(summary.to_json(), encoding="utf-8")
            print(summary.to_json())
            return 0

        tracker = create_monitor_run_tracker(
            paths,
            kind="benchmark",
            command=_monitor_command(),
            inputs={
                "corpus_jsonl": str(corpus_jsonl),
                "provider": args.provider,
                "model": args.model,
                "model_variant": args.model_variant,
                "artifact_format": format_cli_value(artifact_format),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "ollama_context_tokens": args.ollama_context_tokens,
                "max_workers": args.max_workers,
                "num_processes": args.num_processes,
                "gpu_devices": args.gpu_devices,
                "min_items_per_process": args.min_items_per_process,
                "startup_stagger_seconds": args.startup_stagger_seconds,
                "local_gpu_topology": args.local_gpu_topology,
                "local_offload_policy": args.local_offload_policy,
                "local_gpu_memory_reserve_gib": args.local_gpu_memory_reserve_gib,
                "resume_existing": args.resume_existing,
                "checkpoint_records_path": str(checkpoint_records_path or ""),
                "record_timeout_seconds": args.record_timeout_seconds,
                "max_prompt_help_chars": args.max_prompt_help_chars,
                "source_context": source_context.to_dict(),
                "suite_generation": args.suite_generation,
                "max_suite_tools": args.max_suite_tools,
                "repair_invalid_xml": args.repair_invalid_xml,
                "allow_compact_fallback": args.allow_compact_fallback,
                "limit": args.limit,
                "run_planemo": args.run_planemo,
                "run_planemo_tests": args.run_planemo_tests,
                "planemo_test_output_dir": str(planemo_test_options.output_dir or ""),
                "planemo_test_timeout": planemo_test_options.timeout_seconds,
            },
            outputs={
                "wrappers_dir": str(wrappers_dir),
                "generation_records_path": str(generation_records),
                "evaluation_report_path": str(evaluation_report),
                "benchmark_summary_path": str(benchmark_summary_path),
            },
        )

        def benchmark_monitor_warning(error: BaseException) -> None:
            emit_status(
                {
                    "status": "benchmark-monitor-warning",
                    "warning_type": type(error).__name__,
                    "warning": str(error) or repr(error),
                },
                status_log_path=status_log_path,
            )

        def benchmark_monitor_update(**kwargs: object) -> None:
            try:
                tracker.update(**kwargs)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
                benchmark_monitor_warning(error)

        def benchmark_monitor_complete(**kwargs: object) -> None:
            try:
                tracker.complete(**kwargs)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
                benchmark_monitor_warning(error)

        def benchmark_monitor_fail(error: BaseException) -> None:
            try:
                tracker.fail(error)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as monitor_error:
                benchmark_monitor_warning(monitor_error)

        def benchmark_status_sink(payload: dict) -> None:
            emit_status(payload, status_log_path=status_log_path)
            status = payload.get("status")
            if status == "benchmark-progress":
                progress = payload.get("progress", {})
                if isinstance(progress, dict):
                    benchmark_monitor_update(progress=progress)
            elif status in {
                "benchmark-sharded-started",
                "benchmark-model-load-started",
                "benchmark-model-ready",
                "benchmark-first-generation-started",
            }:
                phase_summary = {"phase": status}
                if isinstance(payload.get("startup"), dict):
                    phase_summary["startup"] = payload["startup"]
                else:
                    phase_summary["startup"] = {
                        key: payload[key]
                        for key in (
                            "processes",
                            "gpu_devices",
                            "total",
                            "min_items_per_process",
                            "startup_stagger_seconds",
                            "local_gpu_topology",
                            "local_offload_policy",
                            "local_gpu_memory_reserve_gib",
                            "resume_existing",
                            "checkpoint_records_path",
                            "record_timeout_seconds",
                        )
                        if key in payload
                    }
                benchmark_monitor_update(summary=phase_summary)

        try:
            summary = run_benchmark_generation_sharded(
                paths=paths,
                corpus_jsonl=corpus_jsonl,
                wrappers_dir=wrappers_dir,
                generation_records_path=generation_records,
                evaluation_report_path=evaluation_report,
                benchmark_summary_path=benchmark_summary_path,
                provider=args.provider,
                model_variant=args.model_variant,
                model=args.model,
                artifact_format=artifact_format,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ollama_context_tokens=args.ollama_context_tokens,
                max_workers=args.max_workers,
                limit=args.limit,
                xsd_path=xsd_path,
                run_planemo=args.run_planemo,
                run_planemo_tests=args.run_planemo_tests,
                planemo_test_options=planemo_test_options,
                num_processes=args.num_processes,
                gpu_devices=args.gpu_devices,
                allow_stub_local=args.allow_stub_local,
                status_sink=benchmark_status_sink,
                repair_invalid_xml=args.repair_invalid_xml,
                allow_compact_fallback=args.allow_compact_fallback,
                max_prompt_help_chars=args.max_prompt_help_chars,
                min_items_per_process=args.min_items_per_process,
                startup_stagger_seconds=args.startup_stagger_seconds,
                local_gpu_topology=args.local_gpu_topology,
                local_offload_policy=args.local_offload_policy,
                local_gpu_memory_reserve_gib=args.local_gpu_memory_reserve_gib,
                resume_existing=args.resume_existing,
                checkpoint_records_path=checkpoint_records_path,
                record_timeout_seconds=args.record_timeout_seconds,
                source_context_settings=source_context,
                suite_generation=args.suite_generation,
                max_suite_tools=args.max_suite_tools,
            )
            benchmark_summary_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark_summary_path.write_text(summary.to_json(), encoding="utf-8")
            _print_progress_status(
                dict(summary.progress),
                label="benchmark-progress",
                status_log_path=status_log_path,
            )
            emit_status(
                {
                    "status": "benchmark-completed",
                    "attempted": summary.attempted,
                    "succeeded": summary.succeeded,
                    "failed": summary.failed,
                },
                status_log_path=status_log_path,
            )
            benchmark_monitor_complete(
                progress=dict(summary.progress),
                summary={
                    "attempted": summary.attempted,
                    "succeeded": summary.succeeded,
                    "failed": summary.failed,
                    "quality": summary.quality,
                    "startup": summary.startup,
                },
            )
            print(summary.to_json())
            return 0
        except Exception as error:
            benchmark_monitor_fail(error)
            raise

    if args.command == "promote-candidate":
        policy_path = paths.configs_root / "promotion.policies.json"
        if not policy_path.exists():
            write_default_promotion_policies(policy_path)
        selected_policy = load_promotion_policy(policy_path, args.policy)
        policy = PromotionPolicy(
            min_generation_success_rate=(
                args.min_generation_success_rate
                if args.min_generation_success_rate is not None
                else selected_policy.min_generation_success_rate
            ),
            min_xml_well_formed_rate=(
                args.min_xml_well_formed_rate
                if args.min_xml_well_formed_rate is not None
                else selected_policy.min_xml_well_formed_rate
            ),
            max_unknown_datatype_rate=(
                args.max_unknown_datatype_rate
                if args.max_unknown_datatype_rate is not None
                else selected_policy.max_unknown_datatype_rate
            ),
            require_xsd_pass=args.require_xsd_pass or selected_policy.require_xsd_pass,
            require_planemo_pass=args.require_planemo_pass or selected_policy.require_planemo_pass,
            require_planemo_test_pass=(
                args.require_planemo_test_pass or selected_policy.require_planemo_test_pass
            ),
            baseline_tolerance=(
                args.baseline_tolerance
                if args.baseline_tolerance is not None
                else selected_policy.baseline_tolerance
            ),
        )
        baseline = Path(args.baseline_summary).resolve() if args.baseline_summary else None
        candidate_summary = Path(args.candidate_summary).resolve()
        decision_out = Path(args.decision_out).resolve()
        tracker = create_monitor_run_tracker(
            paths,
            kind="promotion",
            command=_monitor_command(),
            inputs={
                "candidate_summary": str(candidate_summary),
                "baseline_summary": str(baseline) if baseline else "",
                "policy": args.policy,
            },
            outputs={"decision_out": str(decision_out)},
        )
        try:
            decision = decide_promotion(
                candidate_summary_path=candidate_summary,
                baseline_summary_path=baseline,
                policy=policy,
            )
            decision_out.parent.mkdir(parents=True, exist_ok=True)
            decision_out.write_text(decision.to_json(), encoding="utf-8")
            decision_payload = json.loads(decision.to_json())
            tracker.complete(outputs={"decision_out": str(decision_out)}, summary=decision_payload)
            print(decision.to_json())
            return 0
        except Exception as error:
            tracker.fail(error)
            raise

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
