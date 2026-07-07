from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml

from galaxy_toolsmith.core.manifests import ModelVariantManifest, TrainingRunManifest, utc_now_iso
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    TRAINING_ARTIFACT_FORMAT_MIXED,
    normalize_training_artifact_format,
)
from galaxy_toolsmith.inference.source_context import (
    SourceContextResult,
    SourceContextSettings,
    build_source_context_from_record,
)
from galaxy_toolsmith.models.training import TrainingProfile, normalize_training_method
from galaxy_toolsmith.prompts import render_prompt_template
from galaxy_toolsmith.runtime.capabilities import RuntimeCapabilities, detect_runtime_capabilities
from galaxy_toolsmith.runtime.model_source import (
    apply_model_source_environment,
    merged_model_source_environment,
    model_source_load_kwargs,
    resolve_model_source_policy,
)
from galaxy_toolsmith.runtime.progress import make_progress_snapshot
from galaxy_toolsmith.runtime.status import emit_status


def _load_dataset_id(dataset_manifest_path: Path) -> str:
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    return str(dataset.get("dataset_id", "unknown-dataset"))


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_container_help_counts(corpus_jsonl_path: Path) -> dict[str, int]:
    if not corpus_jsonl_path.exists():
        return {
            "corpus_records": 0,
            "container_help_records": 0,
            "container_usage_records": 0,
            "container_api_validation_records": 0,
            "container_api_validation_ok_records": 0,
            "container_api_validation_failed_records": 0,
            "api_backed_wrapper_records": 0,
            "configfile_command_doc_records": 0,
        }
    records = 0
    with_container_help = 0
    with_container_usage = 0
    with_api_validation = 0
    with_api_validation_ok = 0
    with_api_validation_failed = 0
    api_backed_wrappers = 0
    with_configfile_command_docs = 0
    with corpus_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("container_help_text", "")).strip():
                with_container_help += 1
            if str(record.get("container_usage_text", "")).strip():
                with_container_usage += 1
            api_validation = record.get("container_api_validation", [])
            if isinstance(api_validation, list) and api_validation:
                with_api_validation += 1
                has_ok = False
                has_failed = False
                for event in api_validation:
                    if not isinstance(event, dict):
                        continue
                    status = str(event.get("status", "") or "")
                    if status == "container-api-validation-ok":
                        has_ok = True
                    elif status == "container-api-validation-failed":
                        has_failed = True
                if has_ok:
                    with_api_validation_ok += 1
                if has_failed:
                    with_api_validation_failed += 1
            wrapper_summary = record.get("wrapper_source_summary", {})
            if not isinstance(wrapper_summary, dict):
                wrapper_summary = {}
            if wrapper_summary.get("api_backed_wrapper"):
                api_backed_wrappers += 1
            if int(wrapper_summary.get("configfile_command_doc_count", 0) or 0) > 0:
                with_configfile_command_docs += 1
    return {
        "corpus_records": records,
        "container_help_records": with_container_help,
        "container_usage_records": with_container_usage,
        "container_api_validation_records": with_api_validation,
        "container_api_validation_ok_records": with_api_validation_ok,
        "container_api_validation_failed_records": with_api_validation_failed,
        "api_backed_wrapper_records": api_backed_wrappers,
        "configfile_command_doc_records": with_configfile_command_docs,
    }


def _training_command_error(error: subprocess.CalledProcessError) -> str:
    detail = ((error.stderr or "").strip() or (error.stdout or "").strip())[-2000:]
    if detail:
        return f"Training command failed with code {error.returncode}: {detail}"
    return f"Training command failed with code {error.returncode}"


@dataclass(frozen=True)
class BackendResult:
    status: str
    metrics: dict


@dataclass(frozen=True)
class DistributedTrainingContext:
    is_child: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


DEFAULT_DISTRIBUTED_CONTEXT = DistributedTrainingContext()


DISTRIBUTED_TRAINING_STRATEGIES = {
    "ddp",
    "fsdp",
    "deepspeed-zero3",
    "deepspeed-zero3-offload",
    "auto",
}

MLX_LM_BACKEND_ALIASES = {"mlx-lm", "mlx", "mps"}
KBIT_QUANTIZATIONS = {"4bit", "int4", "bnb-4bit", "8bit", "int8", "bnb-8bit"}
FOUR_BIT_QUANTIZATIONS = {"4bit", "int4", "bnb-4bit"}


def _normalize_quantization(profile: TrainingProfile) -> str:
    return profile.quantization.strip().lower()


def _effective_training_method(
    profile: TrainingProfile, selected_backend: str | None = None
) -> str:
    method = normalize_training_method(profile.training_method)
    quantization = _normalize_quantization(profile)
    backend = str(selected_backend or profile.backend).strip().lower()
    if (
        method == "lora"
        and quantization in FOUR_BIT_QUANTIZATIONS
        and backend not in MLX_LM_BACKEND_ALIASES
    ):
        return "qlora"
    return method


def _validate_training_method(profile: TrainingProfile, selected_backend: str | None = None) -> str:
    method = normalize_training_method(profile.training_method)
    effective = _effective_training_method(profile, selected_backend)
    quantization = _normalize_quantization(profile)
    backend = str(selected_backend or profile.backend).strip().lower()
    if method == "full" and quantization in KBIT_QUANTIZATIONS:
        raise ValueError("training_method=full requires quantization=none.")
    if method == "qlora":
        if quantization not in FOUR_BIT_QUANTIZATIONS:
            raise ValueError("training_method=qlora requires a 4-bit quantized profile.")
        if backend in MLX_LM_BACKEND_ALIASES:
            raise ValueError("training_method=qlora is not supported by the mlx-lm backend.")
    if backend in MLX_LM_BACKEND_ALIASES and effective == "qlora":
        raise ValueError("QLoRA is not supported by the mlx-lm backend.")
    return effective


def _artifact_kind_for_backend(*, backend: str, effective_training_method: str) -> str:
    normalized_backend = backend.strip().lower()
    if normalized_backend in MLX_LM_BACKEND_ALIASES:
        return "mlx_full_weights" if effective_training_method == "full" else "mlx_adapter"
    if normalized_backend not in {HFSFTTrainingBackend.name, AxolotlTrainingBackend.name}:
        return "unknown"
    if effective_training_method == "full":
        return "hf_full_model"
    if effective_training_method in {"lora", "qlora"}:
        return "peft_adapter"
    return "unknown"


def _normalize_distributed_strategy(strategy: str | None) -> str:
    normalized = str(strategy or "").strip().lower().replace("_", "-")
    if not normalized:
        normalized = "ddp"
    aliases = {
        "deepspeed": "deepspeed-zero3",
        "zero3": "deepspeed-zero3",
        "ds-zero3": "deepspeed-zero3",
        "deepspeed-zero-3": "deepspeed-zero3",
        "deepspeed-zero3-cpu": "deepspeed-zero3-offload",
        "deepspeed-zero3-cpu-offload": "deepspeed-zero3-offload",
        "zero3-offload": "deepspeed-zero3-offload",
        "fsdp-full-shard": "fsdp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in DISTRIBUTED_TRAINING_STRATEGIES:
        choices = ", ".join(sorted(DISTRIBUTED_TRAINING_STRATEGIES))
        raise ValueError(
            f"Unsupported distributed strategy '{strategy}'. Expected one of: {choices}."
        )
    return normalized


def _model_size_billions(profile: TrainingProfile) -> float | None:
    import re

    for value in (profile.name, profile.base_model):
        match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z])", value.lower())
        if match:
            return float(match.group(1))
    return None


def _resolve_distributed_strategy(
    *,
    profile: TrainingProfile,
    requested_strategy: str | None,
    selected_backend: str,
    num_processes: int,
) -> str:
    requested = _normalize_distributed_strategy(
        requested_strategy if requested_strategy is not None else profile.distributed_strategy
    )
    if requested == "auto":
        if (
            selected_backend == AxolotlTrainingBackend.name
            and num_processes > 1
            and profile.quantization.strip().lower() == "none"
            and (_model_size_billions(profile) or 0.0) >= 14.0
        ):
            return "fsdp"
        return "ddp"
    if selected_backend != AxolotlTrainingBackend.name and requested != "ddp":
        raise ValueError("--distributed-strategy is only supported by the Axolotl backend.")
    return requested


@dataclass(frozen=True)
class TrainingProfileOverrides:
    max_seq_length: int | None = None
    pad_to_sequence_len: bool | None = None
    attn_implementation: str | None = None
    per_device_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    training_method: str | None = None

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return {
            key: value
            for key, value in {
                "max_seq_length": self.max_seq_length,
                "pad_to_sequence_len": self.pad_to_sequence_len,
                "attn_implementation": self.attn_implementation,
                "per_device_batch_size": self.per_device_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "training_method": self.training_method,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class BackendSelection:
    backend: TrainingBackend
    intended_backend: str
    selected_backend: str
    fallback_reason: str
    intended_methodology_supported: bool


class TrainingBackend:
    name = "base"

    def run(
        self,
        paths: WorkspacePaths,
        profile: TrainingProfile,
        run_dir: Path,
        checkpoints_dir: Path,
        output_dir: Path,
        command_override: list[str] | None,
        corpus_jsonl_path: Path,
        num_processes: int = 1,
        dry_run_backend: bool = False,
        distributed_context: DistributedTrainingContext = DEFAULT_DISTRIBUTED_CONTEXT,
        metrics_path: Path | None = None,
        base_metrics: dict | None = None,
        status_log_path: Path | None = None,
        status_interval_seconds: float = 30.0,
        stream_logs: bool = False,
        log_tail_lines: int = 40,
        distributed_strategy: str = "ddp",
        artifact_format: str = ARTIFACT_FORMAT_XML,
        source_context_settings: SourceContextSettings | None = None,
        max_steps: int | None = None,
    ) -> BackendResult:
        raise NotImplementedError


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _training_data_worker_count() -> int:
    return max(1, _env_int("GTSM_TRAIN_DATA_WORKERS", 1))


def _distributed_context(distributed_child: bool) -> DistributedTrainingContext:
    if not distributed_child:
        return DistributedTrainingContext()
    return DistributedTrainingContext(
        is_child=True,
        rank=_env_int("RANK", 0),
        local_rank=_env_int("LOCAL_RANK", 0),
        world_size=max(1, _env_int("WORLD_SIZE", 1)),
    )


def _positive_override(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    return value


def _positive_float_override(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"--{name.replace('_', '-')} must be greater than 0.")
    return value


def _apply_training_profile_overrides(
    profile: TrainingProfile,
    overrides: TrainingProfileOverrides | None,
) -> TrainingProfile:
    if overrides is None:
        return profile
    max_seq_length = _positive_override(overrides.max_seq_length, "max_seq_length")
    per_device_batch_size = _positive_override(
        overrides.per_device_batch_size,
        "per_device_batch_size",
    )
    gradient_accumulation_steps = _positive_override(
        overrides.gradient_accumulation_steps,
        "gradient_accumulation_steps",
    )
    learning_rate = _positive_float_override(overrides.learning_rate, "learning_rate")
    training_method = (
        normalize_training_method(overrides.training_method)
        if overrides.training_method is not None
        else profile.training_method
    )
    return replace(
        profile,
        max_seq_length=max_seq_length or profile.max_seq_length,
        pad_to_sequence_len=(
            overrides.pad_to_sequence_len
            if overrides.pad_to_sequence_len is not None
            else profile.pad_to_sequence_len
        ),
        attn_implementation=overrides.attn_implementation or profile.attn_implementation,
        per_device_batch_size=per_device_batch_size or profile.per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps
        or profile.gradient_accumulation_steps,
        learning_rate=learning_rate or profile.learning_rate,
        training_method=training_method,
    )


def _training_run_manifest_from_path(path: Path) -> TrainingRunManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(TrainingRunManifest)}
    return TrainingRunManifest(**{key: value for key, value in payload.items() if key in allowed})


def _tail_text(value: str, limit: int = 4000) -> str:
    return value.strip()[-limit:]


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tail_file(path: Path, lines: int = 80) -> str:
    if lines <= 0 or not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _last_nonempty_line(path: Path) -> str:
    if not path.is_file():
        return ""
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _read_new_log_lines(path: Path, offset: int, max_lines: int) -> tuple[int, str, str]:
    if not path.is_file():
        return offset, "", ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        new_offset = handle.tell()
    lines = text.splitlines()
    last_line = ""
    for line in reversed(lines):
        if line.strip():
            last_line = line.strip()
            break
    lines = [] if max_lines <= 0 else lines[-max_lines:]
    return new_offset, "\n".join(lines), last_line


def _pid_running(pid: object) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _latest_progress_from_log(progress_path: Path) -> dict:
    if not progress_path.is_file():
        return {}
    latest: dict = {}
    for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            latest = json.loads(line)
        except json.JSONDecodeError:
            continue
    return latest


def _progress_for_local_run(run_dir: Path, metrics: dict) -> dict:
    progress_log_raw = str(metrics.get("progress_log_path", "")).strip()
    progress_log_path = Path(progress_log_raw) if progress_log_raw else run_dir / "progress.jsonl"
    progress_from_log = _latest_progress_from_log(progress_log_path)
    if progress_from_log:
        return progress_from_log
    progress = metrics.get("progress", {})
    return progress if isinstance(progress, dict) else {}


def _run_created_sort_key(run_dir: Path, manifest: dict) -> str:
    created = str(manifest.get("created_at", "")).strip()
    if created:
        return created
    return f"{run_dir.stat().st_mtime:.6f}"


def _local_training_run_dirs(paths: WorkspacePaths) -> list[Path]:
    runs_root = paths.runs_root / "training"
    if not runs_root.exists():
        return []
    run_records: list[tuple[Path, dict]] = []
    for path in runs_root.iterdir():
        manifest_path = path / "run.manifest.json"
        if not path.is_dir() or not manifest_path.exists():
            continue
        manifest = _read_json_file(manifest_path)
        if str(manifest.get("run_id", "")).strip() != path.name:
            continue
        run_records.append((path, manifest))
    run_records.sort(
        key=lambda item: _run_created_sort_key(item[0], item[1]),
        reverse=True,
    )
    return [path for path, _manifest in run_records]


def _local_training_run_summary(run_dir: Path, *, tail_lines: int = 0) -> dict:
    manifest = _read_json_file(run_dir / "run.manifest.json")
    metrics_path_raw = str(manifest.get("metrics_path", "")).strip()
    metrics_path = Path(metrics_path_raw) if metrics_path_raw else run_dir / "metrics.json"
    metrics = _read_json_file(metrics_path)
    status = str(metrics.get("status") or manifest.get("status") or "unknown")
    progress = _progress_for_local_run(run_dir, metrics)
    stdout_log_raw = str(metrics.get("stdout_log_path", "")).strip()
    stderr_log_raw = str(metrics.get("stderr_log_path", "")).strip()
    stdout_log_path = Path(stdout_log_raw) if stdout_log_raw else Path()
    stderr_log_path = Path(stderr_log_raw) if stderr_log_raw else Path()
    pid = metrics.get("pid")
    process_running = _pid_running(pid)
    summary = {
        "run": manifest,
        "metrics": metrics,
        "status": status,
        "progress": progress,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path) if str(metrics_path) else "",
        "process": {
            "pid": pid,
            "running": process_running,
        },
        "logs": {
            "stdout_log_path": stdout_log_raw,
            "stderr_log_path": stderr_log_raw,
        },
    }
    if tail_lines > 0:
        summary["logs"]["stdout_tail"] = _tail_file(stdout_log_path, tail_lines)
        summary["logs"]["stderr_tail"] = _tail_file(stderr_log_path, tail_lines)
    return summary


def list_local_training_runs(paths: WorkspacePaths, *, limit: int = 20) -> dict:
    rows = [
        _local_training_run_summary(run_dir)
        for run_dir in _local_training_run_dirs(paths)[: max(limit, 0)]
    ]
    summary = {
        "total": len(rows),
        "running": 0,
        "completed": 0,
        "failed": 0,
        "dry-run": 0,
        "unknown": 0,
    }
    for row in rows:
        status = str(row.get("status", "unknown")).lower()
        if status not in summary:
            summary["unknown"] += 1
        else:
            summary[status] += 1
    return {"summary": summary, "runs": rows}


def get_local_training_run(
    paths: WorkspacePaths, run_id: str = "latest", *, tail_lines: int = 80
) -> dict:
    run_id = str(run_id or "latest").strip()
    run_dirs = _local_training_run_dirs(paths)
    if run_id == "latest":
        if not run_dirs:
            raise FileNotFoundError("No local training runs found.")
        return _local_training_run_summary(run_dirs[0], tail_lines=tail_lines)
    run_dir = paths.runs_root / "training" / run_id
    if not (run_dir / "run.manifest.json").exists():
        raise FileNotFoundError(f"Local training run not found: {run_id}")
    return _local_training_run_summary(run_dir, tail_lines=tail_lines)


def _write_live_training_metrics(
    metrics_path: Path | None,
    base_metrics: dict | None,
    backend_metrics: dict,
) -> None:
    if metrics_path is None:
        return
    _write_json_file(metrics_path, {**(base_metrics or {}), **backend_metrics})


def _record_training_runtime_probe_text(record: dict) -> str:
    accepted_statuses = {
        "container-command-help",
        "container-command-help-degraded",
        "container-command-usage-degraded",
    }
    sections: list[str] = []
    for event in (record.get("container_execution", []) or [])[:64]:
        if not isinstance(event, dict):
            continue
        status = str(event.get("status", "") or "")
        if status not in accepted_statuses:
            continue
        command = str(event.get("command", "") or "").strip()
        if not command:
            continue
        output = "\n".join(
            part.strip()
            for part in (
                str(event.get("stdout", "") or ""),
                str(event.get("stderr", "") or ""),
            )
            if part and str(part).strip()
        ).strip()
        metadata = [
            f"Command: {command}",
            f"Probe role: {event.get('probe_role', '') or 'unknown'}",
            f"Status: {status}",
        ]
        if output:
            metadata.extend(["Output excerpt:", output[:2500]])
        sections.append("\n".join(metadata))
        if len(sections) >= 12:
            break
    if not sections:
        return ""
    return "Structured runtime help probes:\n\n" + "\n\n".join(sections)


def _record_training_help_text(record: dict) -> str:
    help_text = str(record.get("help_text", "")).strip()
    runtime_sections = [
        (
            "Command-line help collected from container execution",
            str(record.get("container_help_text", "")).strip(),
        ),
        (
            "Command-line usage collected from container execution",
            str(record.get("container_usage_text", "")).strip(),
        ),
        ("Structured runtime help probe commands", _record_training_runtime_probe_text(record)),
    ]
    sections: list[str] = []
    if help_text:
        sections.append(help_text)
    for label, text in runtime_sections:
        if not text or text in help_text:
            continue
        sections.append(f"{label}:\n\n{text}")
    return "\n\n".join(sections)


def _dedupe_paths(paths: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[str] = set()
    deduped: list[tuple[Path, str]] = []
    for path, source in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((path, source))
    return deduped


def _rebased_absolute_paths(path: Path, repo_root: Path) -> list[Path]:
    if not path.is_absolute():
        return []
    candidates: list[Path] = []
    parts = path.parts
    repo_name = repo_root.name
    for index, part in enumerate(parts):
        if part == repo_name and index + 1 < len(parts):
            candidates.append(repo_root.joinpath(*parts[index + 1 :]))
    for marker in (".gtsm-cache", "config", "src", "tests"):
        if marker in parts:
            index = parts.index(marker)
            candidates.append(repo_root.joinpath(*parts[index:]))
    return candidates


def _xml_target_candidates(
    raw_path: str,
    *,
    source: str,
    repo_root: Path,
) -> list[tuple[Path, str]]:
    raw_path = raw_path.strip()
    if not raw_path:
        return []
    path = Path(raw_path).expanduser()
    candidates: list[tuple[Path, str]] = []
    if path.is_absolute():
        candidates.append((path, source))
        candidates.extend(
            (candidate, f"{source}_rebased")
            for candidate in _rebased_absolute_paths(path, repo_root)
        )
    else:
        candidates.append((path, source))
        candidates.append((repo_root / path, f"{source}_rebased"))
    return _dedupe_paths(candidates)


def _xml_root_tag(text: str) -> str:
    try:
        return str(ET.fromstring(text).tag)
    except ET.ParseError:
        return ""


def _record_training_sidecar_context(record: dict) -> str:
    lines: list[str] = []
    shed_payload: dict[str, Any] = {}
    shed_name = str(record.get("shed_name") or "").strip()
    shed_owner = str(record.get("shed_owner") or "").strip()
    shed_description = str(record.get("shed_description") or "").strip()
    shed_homepage_url = str(record.get("shed_homepage_url") or "").strip()
    shed_remote_repository_url = str(record.get("shed_remote_repository_url") or "").strip()
    shed_categories = record.get("shed_categories", [])
    suite_members = record.get("suite_members", [])
    if shed_name:
        shed_payload["name"] = shed_name
    if shed_owner:
        shed_payload["owner"] = shed_owner
    if shed_description:
        shed_payload["description"] = shed_description
    if shed_homepage_url:
        shed_payload["homepage_url"] = shed_homepage_url
    if shed_remote_repository_url:
        shed_payload["remote_repository_url"] = shed_remote_repository_url
    if isinstance(shed_categories, list) and shed_categories:
        shed_payload["categories"] = [str(item) for item in shed_categories if str(item).strip()]
    if record.get("is_suite_root"):
        shed_payload["type"] = "suite_repository"
        if isinstance(suite_members, list) and suite_members:
            shed_payload["repositories"] = [
                {"name": str(member)} for member in suite_members if str(member).strip()
            ]
    if shed_payload:
        shed_text = yaml.safe_dump(shed_payload, sort_keys=False).strip()
        lines.extend(
            [
                "Tool Shed repository metadata (.shed.yml context):",
                "This metadata describes the repository and is not the primary generated wrapper.",
                "```yaml",
                shed_text[:4000],
                "```",
            ]
        )
    raw_sidecars = record.get("wrapper_sidecar_files", [])
    sidecars = raw_sidecars if isinstance(raw_sidecars, list) else []
    if sidecars:
        if lines:
            lines.append("")
        lines.extend(
            [
                "Companion Galaxy sidecar artifacts:",
                "These files may be referenced by the primary <tool>, but they are not the primary wrapper.",
            ]
        )
    for item in sidecars[:12]:
        if not isinstance(item, dict):
            continue
        relpath = str(item.get("relative_path") or item.get("path") or "").strip()
        role = str(item.get("role") or "sidecar").strip()
        root_tag = str(item.get("root_tag") or "").strip()
        byte_count = str(item.get("byte_count") or "").strip()
        if not relpath:
            continue
        details = [f"role={role}"]
        if root_tag:
            details.append(f"root=<{root_tag}>")
        if byte_count:
            details.append(f"bytes={byte_count}")
        lines.append(f"- {relpath} ({'; '.join(details)})")
        content = str(item.get("content") or "").strip()
        if content:
            lines.append("```")
            lines.append(content[:4000])
            lines.append("```")
    if isinstance(suite_members, list) and len(suite_members) > 1:
        members = ", ".join(str(member) for member in suite_members[:20] if str(member).strip())
        if members:
            if lines:
                lines.append("")
            lines.extend(
                [
                    "Related suite tools:",
                    members,
                    "Treat sibling suite members as separate primary Galaxy tools, not as sidecars.",
                ]
            )
    return "\n".join(lines).strip()


def _training_data_diagnostics() -> dict[str, Any]:
    return {
        "total_corpus_records": 0,
        "invalid_json_records": 0,
        "missing_tool_name_count": 0,
        "artifact_format": ARTIFACT_FORMAT_XML,
        "missing_xml_path_count": 0,
        "missing_xml_target_count": 0,
        "empty_xml_target_count": 0,
        "skipped_non_tool_xml_target_count": 0,
        "missing_udt_path_count": 0,
        "missing_udt_target_count": 0,
        "empty_udt_target_count": 0,
        "trainable_samples": 0,
        "target_source_counts": {},
        "source_context_mode": "none",
        "source_context_records": 0,
        "source_context_chars": 0,
        "source_context_files": 0,
        "source_context_truncated_records": 0,
        "source_context_error_records": 0,
        "test_context_mode": "none",
        "test_context_records": 0,
        "test_context_chars": 0,
        "test_context_files": 0,
        "test_context_truncated_records": 0,
        "records_with_shed_metadata": 0,
        "records_with_suite_metadata": 0,
        "records_with_repository_sidecars": 0,
        "skipped_non_primary_repository_metadata_targets": 0,
        "records_with_runtime_help": 0,
        "records_with_degraded_help": 0,
        "records_with_missing_effective_help": 0,
        "records_with_help_only_command": 0,
        "records_with_optional_only_inputs": 0,
        "records_with_repair_feedback": 0,
        "repair_training_samples": 0,
        "missing_xml_target_examples": [],
        "empty_xml_target_examples": [],
        "skipped_non_tool_xml_target_examples": [],
        "missing_udt_target_examples": [],
        "empty_udt_target_examples": [],
        "help_only_command_examples": [],
        "optional_only_input_examples": [],
        "missing_effective_help_examples": [],
    }


def _add_limited_example(diagnostics: dict[str, Any], key: str, example: dict[str, str]) -> None:
    examples = diagnostics.setdefault(key, [])
    if isinstance(examples, list) and len(examples) < 10:
        examples.append(example)


def _resolve_training_xml_target(
    record: dict,
    *,
    repo_root: Path,
    diagnostics: dict[str, Any],
) -> tuple[Path | None, str, str]:
    expanded_raw = str(record.get("expanded_xml_path", "")).strip()
    wrapper_raw = str(record.get("wrapper_path", "")).strip()
    candidates = [
        *_xml_target_candidates(expanded_raw, source="expanded", repo_root=repo_root),
        *_xml_target_candidates(wrapper_raw, source="wrapper", repo_root=repo_root),
    ]
    candidates = _dedupe_paths(candidates)
    if not candidates:
        diagnostics["missing_xml_path_count"] += 1
        return None, "", ""

    tool_name = str(record.get("tool_name", "")).strip()
    existing_empty: list[Path] = []
    for path, source in candidates:
        if not path.exists():
            continue
        xml_target = path.read_text(encoding="utf-8").strip()
        if not xml_target:
            existing_empty.append(path)
            continue
        root_tag = _xml_root_tag(xml_target)
        if root_tag != "tool":
            diagnostics["skipped_non_tool_xml_target_count"] += 1
            _add_limited_example(
                diagnostics,
                "skipped_non_tool_xml_target_examples",
                {
                    "tool_name": tool_name,
                    "expanded_xml_path": expanded_raw,
                    "wrapper_path": wrapper_raw,
                    "first_non_tool_path": str(path),
                    "first_non_tool_root": root_tag or "parse_error",
                },
            )
            continue
        return path, source, xml_target

    if existing_empty:
        diagnostics["empty_xml_target_count"] += 1
        _add_limited_example(
            diagnostics,
            "empty_xml_target_examples",
            {
                "tool_name": tool_name,
                "expanded_xml_path": expanded_raw,
                "wrapper_path": wrapper_raw,
                "first_empty_path": str(existing_empty[0]),
            },
        )
        return None, "", ""

    diagnostics["missing_xml_target_count"] += 1
    _add_limited_example(
        diagnostics,
        "missing_xml_target_examples",
        {
            "tool_name": tool_name,
            "expanded_xml_path": expanded_raw,
            "wrapper_path": wrapper_raw,
            "first_attempted_path": str(candidates[0][0]),
        },
    )
    return None, "", ""


def _resolve_training_udt_target(
    record: dict,
    *,
    repo_root: Path,
    diagnostics: dict[str, Any],
) -> tuple[Path | None, str, str]:
    udt_raw = str(record.get("udt_yaml_path", "") or record.get("udt_path", "")).strip()
    candidates = _dedupe_paths(
        _xml_target_candidates(udt_raw, source="udt_yaml", repo_root=repo_root)
    )
    if not candidates:
        diagnostics["missing_udt_path_count"] += 1
        return None, "", ""

    existing_empty: list[Path] = []
    for path, source in candidates:
        if not path.exists():
            continue
        udt_target = path.read_text(encoding="utf-8").strip()
        if not udt_target:
            existing_empty.append(path)
            continue
        return path, source, udt_target

    tool_name = str(record.get("tool_name", "")).strip()
    if existing_empty:
        diagnostics["empty_udt_target_count"] += 1
        _add_limited_example(
            diagnostics,
            "empty_udt_target_examples",
            {
                "tool_name": tool_name,
                "udt_yaml_path": udt_raw,
                "first_empty_path": str(existing_empty[0]),
            },
        )
        return None, "", ""

    diagnostics["missing_udt_target_count"] += 1
    _add_limited_example(
        diagnostics,
        "missing_udt_target_examples",
        {
            "tool_name": tool_name,
            "udt_yaml_path": udt_raw,
            "first_attempted_path": str(candidates[0][0]),
        },
    )
    return None, "", ""


def _append_training_sample(
    records: list[dict[str, str]],
    *,
    diagnostics: dict[str, Any],
    target_source: str,
    task: str,
    profile: TrainingProfile,
    record: dict,
    tool_name: str,
    output: str,
    source_code: str = "",
) -> None:
    target_source_counts = diagnostics.setdefault("target_source_counts", {})
    source_key = target_source if task == "xml_generate" else f"{task}:{target_source}"
    target_source_counts[source_key] = int(target_source_counts.get(source_key, 0)) + 1
    prompt = render_prompt_template(
        task=task,
        skills_profile=profile.skills_profile,
        context={
            "tool_name": tool_name,
            "help_text": _record_training_help_text(record),
            "source_code": source_code,
            "skills_profile": profile.skills_profile,
            "generate_sidecars": False,
        },
    )
    records.append(
        {
            "instruction": prompt,
            "input": "",
            "output": output,
        }
    )


def _xml_command_text(xml_target: str) -> str:
    try:
        root = ET.fromstring(xml_target)
    except ET.ParseError:
        return ""
    command = root.find("command")
    if command is None:
        return ""
    return " ".join(part.strip() for part in command.itertext() if part and part.strip())


_HELP_TOKEN_RE = re.compile(r"(?<![\w-])(?:--help|-h|help)(?![\w-])", re.I)


def _xml_command_looks_help_only(xml_target: str) -> bool:
    command = _xml_command_text(xml_target)
    if not command or not _HELP_TOKEN_RE.search(command):
        return False
    lowered = command.lower()
    if re.search(r"\$|\binput\b|\boutput\b|>\s*\S|<\s*\S", lowered):
        return False
    tokens = re.findall(r"[A-Za-z0-9_./:+-]+", command)
    return len(tokens) <= 6


def _xml_inputs_are_optional_only(xml_target: str) -> bool:
    try:
        root = ET.fromstring(xml_target)
    except ET.ParseError:
        return False
    params = [
        param
        for param in root.findall(".//inputs//param")
        if str(param.get("type", "")).strip().lower() != "hidden"
    ]
    if not params:
        return False
    for param in params:
        optional = str(param.get("optional", "")).strip().lower()
        if optional not in {"true", "1", "yes"}:
            return False
    return True


def _record_has_runtime_help(record: dict) -> bool:
    if str(record.get("container_help_text", "")).strip():
        return True
    if str(record.get("container_usage_text", "")).strip():
        return True
    for event in record.get("container_execution", []) or []:
        if not isinstance(event, dict):
            continue
        status = str(event.get("status", "") or "")
        if status.startswith("container-command-help") or status.startswith(
            "container-command-usage"
        ):
            return True
    return False


def _record_has_degraded_help(record: dict) -> bool:
    for event in record.get("container_execution", []) or []:
        if isinstance(event, dict) and "degraded" in str(event.get("status", "") or ""):
            return True
    return False


def _update_training_quality_diagnostics(
    diagnostics: dict[str, Any],
    record: dict,
    *,
    xml_target: str = "",
) -> None:
    tool_name = str(record.get("tool_name", "")).strip()
    if _record_has_runtime_help(record):
        diagnostics["records_with_runtime_help"] += 1
    if _record_has_degraded_help(record):
        diagnostics["records_with_degraded_help"] += 1
    if not _record_training_help_text(record).strip():
        diagnostics["records_with_missing_effective_help"] += 1
        _add_limited_example(
            diagnostics,
            "missing_effective_help_examples",
            {"tool_name": tool_name},
        )
    if xml_target and _xml_command_looks_help_only(xml_target):
        diagnostics["records_with_help_only_command"] += 1
        _add_limited_example(
            diagnostics,
            "help_only_command_examples",
            {
                "tool_name": tool_name,
                "command": _xml_command_text(xml_target)[:500],
            },
        )
    if xml_target and _xml_inputs_are_optional_only(xml_target):
        diagnostics["records_with_optional_only_inputs"] += 1
        _add_limited_example(
            diagnostics,
            "optional_only_input_examples",
            {"tool_name": tool_name},
        )


def _repair_feedback_items(record: dict) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for key in (
        "repair_feedback",
        "generation_repair_feedback",
        "wrapper_repair_feedback",
        "failed_generation_feedback",
    ):
        raw_value = record.get(key, [])
        raw_items = raw_value if isinstance(raw_value, list) else [raw_value]
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            generated = str(
                raw_item.get("generated_xml")
                or raw_item.get("invalid_xml")
                or raw_item.get("candidate_xml")
                or raw_item.get("raw_output")
                or raw_item.get("raw_response")
                or ""
            ).strip()
            feedback = str(
                raw_item.get("feedback")
                or raw_item.get("validation_feedback")
                or raw_item.get("validation_errors")
                or raw_item.get("repair_reason")
                or raw_item.get("error")
                or ""
            ).strip()
            if generated and feedback:
                items.append({"generated": generated[:12000], "feedback": feedback[:4000]})
    return items


def _append_repair_training_sample(
    records: list[dict[str, str]],
    *,
    diagnostics: dict[str, Any],
    profile: TrainingProfile,
    record: dict,
    tool_name: str,
    xml_target: str,
    source_code: str = "",
) -> None:
    feedback_items = _repair_feedback_items(record)
    if not feedback_items:
        return
    diagnostics["records_with_repair_feedback"] += 1
    target_source_counts = diagnostics.setdefault("target_source_counts", {})
    source_key = "xml_repair:feedback"
    item = feedback_items[0]
    target_source_counts[source_key] = int(target_source_counts.get(source_key, 0)) + 1
    context_parts = [
        "Repair the following generated Galaxy tool XML using the validation feedback, "
        "tool help, and any source context. Return exactly one corrected <tool> document.",
        f"Tool name: {tool_name}",
        f"Skills profile: {profile.skills_profile}",
    ]
    help_text = _record_training_help_text(record)
    if help_text:
        context_parts.extend(["Tool help:", help_text])
    if source_code:
        context_parts.extend(["Source and sidecar context:", source_code])
    context_parts.extend(
        [
            "Validation feedback:",
            item["feedback"],
            "Generated XML to repair:",
            item["generated"],
        ]
    )
    records.append(
        {
            "instruction": "\n\n".join(context_parts),
            "input": "",
            "output": xml_target,
        }
    )
    diagnostics["repair_training_samples"] += 1


def _update_source_context_diagnostics(
    diagnostics: dict[str, Any],
    source_context: SourceContextResult,
) -> None:
    diagnostics["source_context_mode"] = source_context.mode
    diagnostics["test_context_mode"] = source_context.test_context_mode
    if source_context.included_test_files or source_context.included_test_chars:
        diagnostics["test_context_records"] += 1
        diagnostics["test_context_chars"] += source_context.included_test_chars
        diagnostics["test_context_files"] += source_context.included_test_files
        if source_context.test_context_truncated:
            diagnostics["test_context_truncated_records"] += 1
    if not source_context.text.strip():
        if source_context.errors:
            diagnostics["source_context_error_records"] += 1
        return
    diagnostics["source_context_records"] += 1
    diagnostics["source_context_chars"] += len(source_context.text)
    diagnostics["source_context_files"] += source_context.included_files
    if source_context.truncated:
        diagnostics["source_context_truncated_records"] += 1
    if source_context.errors:
        diagnostics["source_context_error_records"] += 1


def _append_conversion_training_sample(
    records: list[dict[str, str]],
    *,
    diagnostics: dict[str, Any],
    profile: TrainingProfile,
    tool_name: str,
    udt_target: str,
    xml_target: str,
) -> None:
    target_source_counts = diagnostics.setdefault("target_source_counts", {})
    source_key = "udt_to_xml:paired"
    target_source_counts[source_key] = int(target_source_counts.get(source_key, 0)) + 1
    instruction = (
        "Convert the following Galaxy User-Defined Tool YAML into one standard Galaxy "
        "tool XML document. Return exactly one complete <tool> document.\n\n"
        f"Tool name: {tool_name}\n"
        f"Skills profile: {profile.skills_profile}\n\n"
        f"UDT YAML:\n{udt_target}"
    )
    records.append(
        {
            "instruction": instruction,
            "input": "",
            "output": xml_target,
        }
    )


def _merge_training_data_diagnostics(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    for key, value in source.items():
        if key == "target_source_counts" and isinstance(value, dict):
            target_counts = target.setdefault("target_source_counts", {})
            for source_key, count in value.items():
                target_counts[source_key] = int(target_counts.get(source_key, 0)) + int(count)
            continue
        if key.endswith("_examples") and isinstance(value, list):
            target_examples = target.setdefault(key, [])
            if isinstance(target_examples, list):
                remaining = max(0, 10 - len(target_examples))
                target_examples.extend(value[:remaining])
            continue
        if key in {"artifact_format", "source_context_mode", "test_context_mode"}:
            target[key] = value
            continue
        if isinstance(value, int):
            target[key] = int(target.get(key, 0) or 0) + value
            continue
        if key not in target:
            target[key] = value


def _training_samples_from_record(
    record: dict,
    *,
    profile: TrainingProfile,
    repo_root: Path,
    artifact_format: str,
    source_context_settings: SourceContextSettings,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    records: list[dict[str, str]] = []
    diagnostics = _training_data_diagnostics()
    diagnostics["total_corpus_records"] = 1
    diagnostics["artifact_format"] = artifact_format
    diagnostics["source_context_mode"] = source_context_settings.mode
    diagnostics["test_context_mode"] = source_context_settings.test_context_mode
    tool_name = str(record.get("tool_name", "")).strip()
    if not tool_name:
        diagnostics["missing_tool_name_count"] += 1
        return records, diagnostics
    if any(
        str(record.get(key) or "").strip()
        for key in (
            "shed_name",
            "shed_owner",
            "shed_description",
            "shed_homepage_url",
            "shed_remote_repository_url",
        )
    ) or record.get("shed_categories"):
        diagnostics["records_with_shed_metadata"] += 1
    suite_members = record.get("suite_members", [])
    if record.get("is_suite_root") or (isinstance(suite_members, list) and len(suite_members) > 1):
        diagnostics["records_with_suite_metadata"] += 1
    sidecars = record.get("wrapper_sidecar_files", [])
    if isinstance(sidecars, list) and sidecars:
        diagnostics["records_with_repository_sidecars"] += 1

    xml_path: Path | None = None
    xml_source = ""
    xml_target = ""
    udt_path: Path | None = None
    udt_source = ""
    udt_target = ""
    if artifact_format in {ARTIFACT_FORMAT_XML, TRAINING_ARTIFACT_FORMAT_MIXED}:
        xml_path, xml_source, xml_target = _resolve_training_xml_target(
            record,
            repo_root=repo_root,
            diagnostics=diagnostics,
        )
    if artifact_format in {ARTIFACT_FORMAT_UDT_YAML, TRAINING_ARTIFACT_FORMAT_MIXED}:
        udt_path, udt_source, udt_target = _resolve_training_udt_target(
            record,
            repo_root=repo_root,
            diagnostics=diagnostics,
        )
    _update_training_quality_diagnostics(diagnostics, record, xml_target=xml_target)

    has_trainable_target = (
        artifact_format in {ARTIFACT_FORMAT_XML, TRAINING_ARTIFACT_FORMAT_MIXED}
        and xml_path is not None
        and bool(xml_target)
    ) or (
        artifact_format in {ARTIFACT_FORMAT_UDT_YAML, TRAINING_ARTIFACT_FORMAT_MIXED}
        and udt_path is not None
        and bool(udt_target)
    )
    source_code = str(record.get("documentation", ""))
    if has_trainable_target:
        source_context = build_source_context_from_record(record, source_context_settings)
        _update_source_context_diagnostics(diagnostics, source_context)
        if source_context.text.strip():
            source_code = source_context.text
        sidecar_context = _record_training_sidecar_context(record)
        if sidecar_context:
            source_code = "\n\n".join(part for part in (source_code, sidecar_context) if part)

    if (
        artifact_format in {ARTIFACT_FORMAT_XML, TRAINING_ARTIFACT_FORMAT_MIXED}
        and xml_path is not None
        and xml_target
    ):
        _append_training_sample(
            records,
            diagnostics=diagnostics,
            target_source=xml_source,
            task="xml_generate",
            profile=profile,
            record=record,
            tool_name=tool_name,
            output=xml_target,
            source_code=source_code,
        )
        _append_repair_training_sample(
            records,
            diagnostics=diagnostics,
            profile=profile,
            record=record,
            tool_name=tool_name,
            xml_target=xml_target,
            source_code=source_code,
        )
    if (
        artifact_format in {ARTIFACT_FORMAT_UDT_YAML, TRAINING_ARTIFACT_FORMAT_MIXED}
        and udt_path is not None
        and udt_target
    ):
        _append_training_sample(
            records,
            diagnostics=diagnostics,
            target_source=udt_source,
            task="udt_yaml_generate",
            profile=profile,
            record=record,
            tool_name=tool_name,
            output=udt_target,
            source_code=source_code,
        )
    if (
        artifact_format == TRAINING_ARTIFACT_FORMAT_MIXED
        and udt_path is not None
        and udt_target
        and xml_path is not None
        and xml_target
    ):
        _append_conversion_training_sample(
            records,
            diagnostics=diagnostics,
            profile=profile,
            tool_name=tool_name,
            udt_target=udt_target,
            xml_target=xml_target,
        )
    diagnostics["trainable_samples"] = len(records)
    return records, diagnostics


def _load_instruction_records_with_diagnostics(
    corpus_jsonl_path: Path,
    profile: TrainingProfile,
    *,
    repo_root: Path | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
    limit: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not corpus_jsonl_path.exists():
        raise FileNotFoundError(f"Corpus JSONL not found: {corpus_jsonl_path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided.")

    repo_root = (repo_root or Path.cwd()).resolve()
    records: list[dict[str, str]] = []
    artifact_format = normalize_training_artifact_format(artifact_format)
    source_context_settings = (source_context_settings or SourceContextSettings()).normalized()
    diagnostics = _training_data_diagnostics()
    diagnostics["artifact_format"] = artifact_format
    diagnostics["source_context_mode"] = source_context_settings.mode
    diagnostics["test_context_mode"] = source_context_settings.test_context_mode
    parsed_records: list[dict[str, Any]] = []
    seen_corpus_records = 0
    with corpus_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            seen_corpus_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                diagnostics["total_corpus_records"] += 1
                diagnostics["invalid_json_records"] += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "records_seen": diagnostics["total_corpus_records"],
                            "trainable_samples": len(records),
                        }
                    )
                continue
            parsed_records.append(record)
            if limit is not None and seen_corpus_records >= limit:
                break

    worker_count = min(_training_data_worker_count(), max(1, len(parsed_records)))
    diagnostics["training_data_workers"] = worker_count

    def process(record: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        return _training_samples_from_record(
            record,
            profile=profile,
            repo_root=repo_root,
            artifact_format=artifact_format,
            source_context_settings=source_context_settings,
        )

    completed_records = 0
    if worker_count <= 1:
        for record in parsed_records:
            sample_records, record_diagnostics = process(record)
            records.extend(sample_records)
            _merge_training_data_diagnostics(diagnostics, record_diagnostics)
            completed_records += int(record_diagnostics.get("total_corpus_records", 0) or 0)
            if progress_callback is not None:
                progress_callback(
                    {
                        "records_seen": diagnostics["total_corpus_records"],
                        "trainable_samples": len(records),
                        "completed_valid_records": completed_records,
                        "worker_count": worker_count,
                    }
                )
    else:
        ordered_results: dict[int, tuple[list[dict[str, str]], dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(process, record): index
                for index, record in enumerate(parsed_records)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                sample_records, record_diagnostics = future.result()
                ordered_results[index] = (sample_records, record_diagnostics)
                _merge_training_data_diagnostics(diagnostics, record_diagnostics)
                completed_records += int(record_diagnostics.get("total_corpus_records", 0) or 0)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "records_seen": diagnostics["total_corpus_records"],
                            "trainable_samples": int(diagnostics.get("trainable_samples", 0) or 0),
                            "completed_valid_records": completed_records,
                            "worker_count": worker_count,
                        }
                    )
        for index in range(len(parsed_records)):
            sample_records, _record_diagnostics = ordered_results[index]
            records.extend(sample_records)
    diagnostics["trainable_samples"] = len(records)
    if not records:
        raise RuntimeError("No trainable samples found in corpus JSONL.")
    return records, diagnostics


def _load_instruction_records(
    corpus_jsonl_path: Path,
    profile: TrainingProfile,
    *,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
) -> list[dict[str, str]]:
    records, _diagnostics = _load_instruction_records_with_diagnostics(
        corpus_jsonl_path,
        profile,
        artifact_format=artifact_format,
        source_context_settings=source_context_settings,
    )
    return records


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class CommandTrainingBackend(TrainingBackend):
    name = "command"

    def run(
        self,
        paths: WorkspacePaths,
        profile: TrainingProfile,
        run_dir: Path,
        checkpoints_dir: Path,
        output_dir: Path,
        command_override: list[str] | None,
        corpus_jsonl_path: Path,
        num_processes: int = 1,
        dry_run_backend: bool = False,
        distributed_context: DistributedTrainingContext = DEFAULT_DISTRIBUTED_CONTEXT,
        metrics_path: Path | None = None,
        base_metrics: dict | None = None,
        status_log_path: Path | None = None,
        status_interval_seconds: float = 30.0,
        stream_logs: bool = False,
        log_tail_lines: int = 40,
        distributed_strategy: str = "ddp",
        artifact_format: str = ARTIFACT_FORMAT_XML,
        source_context_settings: SourceContextSettings | None = None,
        max_steps: int | None = None,
    ) -> BackendResult:
        del (
            checkpoints_dir,
            output_dir,
            corpus_jsonl_path,
            num_processes,
            distributed_context,
            distributed_strategy,
            metrics_path,
            base_metrics,
            status_log_path,
            status_interval_seconds,
            stream_logs,
            log_tail_lines,
            artifact_format,
            source_context_settings,
            max_steps,
        )
        started_at = utc_now_iso()
        command = command_override if command_override else list(profile.default_command)
        source_policy = resolve_model_source_policy(paths)
        if not command:
            command = [
                "echo",
                "No trainer command configured; bootstrap training orchestration only.",
            ]
        if dry_run_backend:
            return BackendResult(
                status="dry-run",
                metrics={
                    "backend": self.name,
                    "command": command,
                    "model_source_policy": source_policy.to_dict(),
                    "progress": make_progress_snapshot(
                        started_at=started_at,
                        completed_units=0,
                        total_units=1,
                    ).to_dict(),
                },
            )

        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            cwd=paths.repo_root,
            env=merged_model_source_environment(None, source_policy),
        )
        return BackendResult(
            status="completed",
            metrics={
                "backend": self.name,
                "command": command,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "returncode": completed.returncode,
                "model_source_policy": source_policy.to_dict(),
                "progress": make_progress_snapshot(
                    started_at=started_at,
                    completed_units=1,
                    total_units=1,
                ).to_dict(),
            },
        )


def _callable_parameters(callable_obj: object) -> tuple[set[str], bool]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set(), False
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    return set(parameters), accepts_kwargs


def _supported_kwargs(callable_obj: object, kwargs: dict) -> dict:
    parameter_names, accepts_kwargs = _callable_parameters(callable_obj)
    if accepts_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameter_names}


def _accepts_kwarg(callable_obj: object, name: str) -> bool:
    parameter_names, accepts_kwargs = _callable_parameters(callable_obj)
    return accepts_kwargs or name in parameter_names


def _cuda_available(torch_module: object | None) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


def _cuda_bf16_supported(torch_module: object | None) -> bool:
    if not _cuda_available(torch_module):
        return False
    cuda = getattr(torch_module, "cuda", None)
    is_supported = getattr(cuda, "is_bf16_supported", None)
    if callable(is_supported):
        return bool(is_supported())
    return hasattr(torch_module, "bfloat16")


def _preferred_cuda_dtype(torch_module: object | None) -> object | None:
    if not _cuda_available(torch_module):
        return None
    if _cuda_bf16_supported(torch_module):
        return getattr(torch_module, "bfloat16", None)
    return getattr(torch_module, "float16", None)


def _training_precision_kwargs(torch_module: object | None) -> dict:
    if not _cuda_available(torch_module):
        return {}
    if _cuda_bf16_supported(torch_module):
        return {"bf16": True}
    return {"fp16": True}


def _training_args_kwargs(
    profile: TrainingProfile,
    checkpoints_dir: Path,
    torch_module: object | None = None,
    max_steps: int | None = None,
) -> dict:
    kwargs = {
        "output_dir": str(checkpoints_dir),
        "num_train_epochs": profile.epochs,
        "per_device_train_batch_size": profile.per_device_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "learning_rate": profile.learning_rate,
        "logging_steps": 10,
        "save_steps": 100,
        "max_steps": int(max_steps) if max_steps is not None else -1,
        "seed": profile.seed,
        "report_to": [],
        "gradient_checkpointing": True,
    }
    kwargs.update(_training_precision_kwargs(torch_module))
    return kwargs


def _source_policy_load_kwargs(source_policy: object) -> dict:
    return model_source_load_kwargs(source_policy)


def _model_load_kwargs(
    *,
    profile: TrainingProfile,
    source_policy: object,
    torch_module: object | None,
    bitsandbytes_config_cls: type | None,
    distributed_world_size: int = 1,
    local_rank: int | None = None,
) -> dict:
    load_kwargs = _source_policy_load_kwargs(source_policy)
    dtype = _preferred_cuda_dtype(torch_module)
    quantization = profile.quantization.strip().lower()
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    if quantization in {"4bit", "int4", "bnb-4bit"}:
        if bitsandbytes_config_cls is None:
            raise RuntimeError("4-bit training requires transformers BitsAndBytesConfig.")
        config_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }
        if dtype is not None:
            config_kwargs["bnb_4bit_compute_dtype"] = dtype
        load_kwargs["quantization_config"] = bitsandbytes_config_cls(
            **_supported_kwargs(bitsandbytes_config_cls, config_kwargs)
        )
        load_kwargs["device_map"] = (
            {"": local_rank} if distributed_world_size > 1 and local_rank is not None else "auto"
        )
    elif quantization in {"8bit", "int8", "bnb-8bit"}:
        if bitsandbytes_config_cls is None:
            raise RuntimeError("8-bit training requires transformers BitsAndBytesConfig.")
        load_kwargs["quantization_config"] = bitsandbytes_config_cls(load_in_8bit=True)
        load_kwargs["device_map"] = (
            {"": local_rank} if distributed_world_size > 1 and local_rank is not None else "auto"
        )
    return load_kwargs


def _build_sft_args(
    training_arguments_cls: type,
    sft_config_cls: type | None,
    profile: TrainingProfile,
    checkpoints_dir: Path,
    torch_module: object | None = None,
    max_steps: int | None = None,
) -> object:
    kwargs = _training_args_kwargs(
        profile=profile,
        checkpoints_dir=checkpoints_dir,
        torch_module=torch_module,
        max_steps=max_steps,
    )
    if sft_config_cls is None:
        return training_arguments_cls(**_supported_kwargs(training_arguments_cls, kwargs))

    sft_kwargs = dict(kwargs)
    if _accepts_kwarg(sft_config_cls, "dataset_text_field"):
        sft_kwargs["dataset_text_field"] = "text"
    if _accepts_kwarg(sft_config_cls, "max_length"):
        sft_kwargs["max_length"] = profile.max_seq_length
    elif _accepts_kwarg(sft_config_cls, "max_seq_length"):
        sft_kwargs["max_seq_length"] = profile.max_seq_length
    return sft_config_cls(**_supported_kwargs(sft_config_cls, sft_kwargs))


def _build_sft_trainer(
    sft_trainer_cls: type,
    *,
    model: object,
    train_dataset: object,
    tokenizer: object,
    args: object,
    peft_config: object | None,
    profile: TrainingProfile,
) -> object:
    kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": args,
    }
    if peft_config is not None:
        kwargs["peft_config"] = peft_config
    if _accepts_kwarg(sft_trainer_cls, "tokenizer"):
        kwargs["tokenizer"] = tokenizer
    elif _accepts_kwarg(sft_trainer_cls, "processing_class"):
        kwargs["processing_class"] = tokenizer
    if _accepts_kwarg(sft_trainer_cls, "dataset_text_field"):
        kwargs["dataset_text_field"] = "text"
    if _accepts_kwarg(sft_trainer_cls, "max_seq_length"):
        kwargs["max_seq_length"] = profile.max_seq_length
    elif _accepts_kwarg(sft_trainer_cls, "max_length"):
        kwargs["max_length"] = profile.max_seq_length
    return sft_trainer_cls(**_supported_kwargs(sft_trainer_cls, kwargs))


class HFSFTTrainingBackend(TrainingBackend):
    name = "hf-sft"

    def _load_pairs(
        self,
        corpus_jsonl_path: Path,
        profile: TrainingProfile,
        *,
        repo_root: Path,
        artifact_format: str = ARTIFACT_FORMAT_XML,
        source_context_settings: SourceContextSettings | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        records, diagnostics = _load_instruction_records_with_diagnostics(
            corpus_jsonl_path,
            profile,
            repo_root=repo_root,
            artifact_format=artifact_format,
            source_context_settings=source_context_settings,
        )
        return [
            {"text": f"{record['instruction']}\n\n{record['output']}"} for record in records
        ], diagnostics

    def run(
        self,
        paths: WorkspacePaths,
        profile: TrainingProfile,
        run_dir: Path,
        checkpoints_dir: Path,
        output_dir: Path,
        command_override: list[str] | None,
        corpus_jsonl_path: Path,
        num_processes: int = 1,
        dry_run_backend: bool = False,
        distributed_context: DistributedTrainingContext = DEFAULT_DISTRIBUTED_CONTEXT,
        metrics_path: Path | None = None,
        base_metrics: dict | None = None,
        status_log_path: Path | None = None,
        status_interval_seconds: float = 30.0,
        stream_logs: bool = False,
        log_tail_lines: int = 40,
        distributed_strategy: str = "ddp",
        artifact_format: str = ARTIFACT_FORMAT_XML,
        source_context_settings: SourceContextSettings | None = None,
        max_steps: int | None = None,
    ) -> BackendResult:
        del (
            command_override,
            num_processes,
            distributed_strategy,
            metrics_path,
            base_metrics,
            status_log_path,
            status_interval_seconds,
            stream_logs,
            log_tail_lines,
        )
        samples, training_data_diagnostics = self._load_pairs(
            corpus_jsonl_path=corpus_jsonl_path,
            profile=profile,
            repo_root=paths.repo_root,
            artifact_format=artifact_format,
            source_context_settings=source_context_settings,
        )
        started_at = utc_now_iso()
        if dry_run_backend:
            effective_method = _validate_training_method(profile, self.name)
            return BackendResult(
                status="dry-run",
                metrics={
                    "backend": self.name,
                    "effective_training_method": effective_method,
                    "artifact_kind": _artifact_kind_for_backend(
                        backend=self.name,
                        effective_training_method=effective_method,
                    ),
                    "samples": len(samples),
                    "training_data_diagnostics": training_data_diagnostics,
                    "progress": make_progress_snapshot(
                        started_at=started_at,
                        completed_units=0,
                        total_units=0,
                    ).to_dict(),
                },
            )

        try:
            import torch
            from datasets import Dataset
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainerCallback,
                TrainingArguments,
            )

            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                BitsAndBytesConfig = None
            from trl import SFTTrainer

            try:
                from trl import SFTConfig
            except ImportError:
                SFTConfig = None
        except Exception as error:
            raise RuntimeError(
                "HF SFT backend requires optional training dependencies: datasets, transformers, trl."
            ) from error

        effective_method = _validate_training_method(profile, self.name)
        LoraConfig = None
        prepare_model_for_kbit_training = None
        if effective_method in {"lora", "qlora"}:
            try:
                from peft import LoraConfig

                try:
                    from peft import prepare_model_for_kbit_training
                except ImportError:
                    prepare_model_for_kbit_training = None
            except Exception as error:
                raise RuntimeError(
                    "HF SFT LoRA/QLoRA training requires optional dependency: peft."
                ) from error

        if distributed_context.world_size > 1 and _cuda_available(torch):
            set_device = getattr(torch.cuda, "set_device", None)
            if callable(set_device):
                set_device(distributed_context.local_rank)

        dataset = Dataset.from_list(samples)

        source_policy = resolve_model_source_policy(paths)
        apply_model_source_environment(source_policy)
        load_kwargs = _model_load_kwargs(
            profile=profile,
            source_policy=source_policy,
            torch_module=torch,
            bitsandbytes_config_cls=BitsAndBytesConfig,
            distributed_world_size=distributed_context.world_size,
            local_rank=distributed_context.local_rank,
        )

        progress_file = run_dir / "progress.jsonl"

        class _ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                del args, control, logs, kwargs
                if not distributed_context.is_rank_zero:
                    return
                total = int(state.max_steps) if int(state.max_steps) > 0 else None
                completed = int(state.global_step)
                snapshot = make_progress_snapshot(
                    started_at=started_at,
                    completed_units=completed,
                    total_units=total,
                )
                with progress_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(snapshot.to_dict()) + "\n")

        tokenizer = AutoTokenizer.from_pretrained(profile.base_model, use_fast=True, **load_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(profile.base_model, **load_kwargs)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if _normalize_quantization(profile) in KBIT_QUANTIZATIONS and (
            prepare_model_for_kbit_training is not None
        ):
            model = prepare_model_for_kbit_training(model)

        peft_config = None
        if effective_method in {"lora", "qlora"}:
            peft_config = LoraConfig(
                r=profile.lora_rank,
                lora_alpha=profile.lora_alpha,
                lora_dropout=profile.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        uses_legacy_sft_kwargs = _accepts_kwarg(SFTTrainer, "dataset_text_field") or _accepts_kwarg(
            SFTTrainer, "max_seq_length"
        )
        training_args = _build_sft_args(
            training_arguments_cls=TrainingArguments,
            sft_config_cls=None if uses_legacy_sft_kwargs else SFTConfig,
            profile=profile,
            checkpoints_dir=checkpoints_dir,
            torch_module=torch,
            max_steps=max_steps,
        )

        trainer = _build_sft_trainer(
            SFTTrainer,
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
            peft_config=peft_config,
            profile=profile,
        )
        trainer.add_callback(_ProgressCallback())
        train_result = trainer.train()
        if distributed_context.is_rank_zero:
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

        max_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
        final_snapshot = make_progress_snapshot(
            started_at=started_at,
            completed_units=int(getattr(train_result, "global_step", 0) or 0),
            total_units=max_steps if max_steps > 0 else None,
        )
        return BackendResult(
            status="completed",
            metrics={
                "backend": self.name,
                "effective_training_method": effective_method,
                "artifact_kind": _artifact_kind_for_backend(
                    backend=self.name,
                    effective_training_method=effective_method,
                ),
                "samples": len(samples),
                "training_data_diagnostics": training_data_diagnostics,
                "global_step": int(getattr(train_result, "global_step", 0) or 0),
                "training_loss": float(getattr(train_result, "training_loss", 0.0) or 0.0),
                "artifact_dir": str(output_dir),
                "progress": final_snapshot.to_dict(),
                "progress_log_path": str(progress_file),
                "model_source_policy": source_policy.to_dict(),
            },
        )


def _axolotl_command(config_path: Path, num_processes: int) -> list[str]:
    command = ["axolotl", "train", str(config_path), "--launcher", "accelerate"]
    if num_processes > 1:
        command.extend(["--", "--num_processes", str(num_processes)])
    return command


def _fsdp_transformer_layer_cls(profile: TrainingProfile) -> str:
    model_text = f"{profile.name} {profile.base_model}".lower()
    if "qwen" in model_text:
        return "Qwen2DecoderLayer"
    if "mistral" in model_text or "devstral" in model_text:
        return "MistralDecoderLayer"
    return ""


def _profile_uses_mistral_common_tokenizer(profile: TrainingProfile) -> bool:
    model_text = f"{profile.name} {profile.base_model}".lower()
    return "mistral" in model_text or "devstral" in model_text


_AXOLOTL_MISTRAL_COMPAT_SITECUSTOMIZE = '''\
"""GTSM Axolotl runtime compatibility patches."""

try:
    from transformers.tokenization_mistral_common import MistralCommonBackend

    _gtsm_original_mistral_save_pretrained = MistralCommonBackend.save_pretrained

    if not getattr(_gtsm_original_mistral_save_pretrained, "_gtsm_drops_save_jinja_files", False):

        def _gtsm_mistral_save_pretrained(self, save_directory, *args, **kwargs):
            kwargs.pop("save_jinja_files", None)
            return _gtsm_original_mistral_save_pretrained(
                self,
                save_directory,
                *args,
                **kwargs,
            )

        _gtsm_mistral_save_pretrained._gtsm_drops_save_jinja_files = True
        MistralCommonBackend.save_pretrained = _gtsm_mistral_save_pretrained
except Exception:
    pass
'''


def _write_axolotl_runtime_compat(run_dir: Path, profile: TrainingProfile) -> Path | None:
    if not _profile_uses_mistral_common_tokenizer(profile):
        return None
    compat_dir = run_dir / "axolotl" / "runtime_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = compat_dir / "sitecustomize.py"
    sitecustomize_path.write_text(_AXOLOTL_MISTRAL_COMPAT_SITECUSTOMIZE, encoding="utf-8")
    return sitecustomize_path


def _prepend_env_path(env: dict[str, str], key: str, value: Path) -> None:
    current = env.get(key, "")
    env[key] = str(value) if not current else f"{value}{os.pathsep}{current}"


def _axolotl_subprocess_environment(
    *,
    source_policy: object,
    compat_sitecustomize_path: Path | None,
) -> dict[str, str]:
    env = merged_model_source_environment(None, source_policy)
    _prepend_env_path(env, "PATH", Path(sys.executable).resolve().parent)
    if compat_sitecustomize_path is not None:
        _prepend_env_path(env, "PYTHONPATH", compat_sitecustomize_path.parent)
    return env


def _ensure_deepspeed_available(strategy: str) -> None:
    if strategy not in {"deepspeed-zero3", "deepspeed-zero3-offload"}:
        return
    if find_spec("deepspeed") is not None:
        return
    raise RuntimeError(
        "DeepSpeed is required for distributed strategy "
        f"'{strategy}'. Install it in the active training environment, for example: "
        "DS_BUILD_OPS=0 python -m pip install --no-build-isolation 'deepspeed>=0.14,<0.18'."
    )


def _deepspeed_zero3_config(*, offload: bool = False) -> dict[str, Any]:
    zero_optimization: dict[str, Any] = {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
    if offload:
        zero_optimization.update(
            {
                "stage3_param_persistence_threshold": 0,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
            }
        )
    return {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": False},
        "zero_optimization": zero_optimization,
    }


def _axolotl_config(
    *,
    profile: TrainingProfile,
    train_jsonl_path: Path,
    prepared_dir: Path,
    output_dir: Path,
    source_policy: object | None = None,
    distributed_strategy: str = "ddp",
    deepspeed_config_path: Path | None = None,
    max_steps: int | None = None,
) -> dict:
    quantization = profile.quantization.strip().lower()
    is_4bit = quantization in {"4bit", "int4", "bnb-4bit"}
    is_8bit = quantization in {"8bit", "int8", "bnb-8bit"}
    effective_method = _validate_training_method(profile, AxolotlTrainingBackend.name)
    strategy = _normalize_distributed_strategy(distributed_strategy)
    config = {
        "base_model": profile.base_model,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "trust_remote_code": True,
        "datasets": [
            {
                "path": str(train_jsonl_path),
                "type": "alpaca",
            }
        ],
        "dataset_prepared_path": str(prepared_dir),
        "output_dir": str(output_dir),
        "sequence_len": profile.max_seq_length,
        "sample_packing": False,
        "pad_to_sequence_len": profile.pad_to_sequence_len,
        "micro_batch_size": profile.per_device_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "num_epochs": profile.epochs,
        "learning_rate": profile.learning_rate,
        "optimizer": "paged_adamw_8bit" if is_4bit or is_8bit else "adamw_torch",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "fp16": False,
        "tf32": True,
        "ddp_find_unused_parameters": False,
        "save_safetensors": True,
        "save_total_limit": 2,
        "seed": profile.seed,
        "wandb_mode": "disabled",
        "strict": False,
    }
    if max_steps is not None:
        config["max_steps"] = int(max_steps)
    if effective_method in {"lora", "qlora"}:
        config.update(
            {
                "adapter": "qlora" if effective_method == "qlora" else "lora",
                "lora_r": profile.lora_rank,
                "lora_alpha": profile.lora_alpha,
                "lora_dropout": profile.lora_dropout,
                "lora_target_linear": True,
            }
        )
    if profile.attn_implementation:
        config["attn_implementation"] = profile.attn_implementation
    if strategy == "fsdp":
        fsdp_config: dict[str, Any] = {
            "state_dict_type": "SHARDED_STATE_DICT",
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
            "use_orig_params": True,
            "limit_all_gathers": True,
            "offload_params": False,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "activation_checkpointing": True,
        }
        transformer_layer = _fsdp_transformer_layer_cls(profile)
        if transformer_layer:
            fsdp_config["transformer_layer_cls_to_wrap"] = transformer_layer
        config.update(
            {
                "fsdp": ["full_shard", "auto_wrap"],
                "fsdp_config": fsdp_config,
            }
        )
    elif strategy in {"deepspeed-zero3", "deepspeed-zero3-offload"}:
        if deepspeed_config_path is None:
            raise ValueError("deepspeed_config_path is required for DeepSpeed strategies.")
        config["deepspeed"] = str(deepspeed_config_path)
        config["gradient_checkpointing"] = True
    else:
        config["gradient_checkpointing"] = True
    if source_policy is not None:
        cache_dir = str(getattr(source_policy, "cache_dir", "") or "")
        if cache_dir:
            config["cache_dir"] = cache_dir
    if is_4bit:
        config.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            }
        )
    elif is_8bit:
        config["load_in_8bit"] = True
    return config


class AxolotlTrainingBackend(TrainingBackend):
    name = "axolotl"

    def run(
        self,
        paths: WorkspacePaths,
        profile: TrainingProfile,
        run_dir: Path,
        checkpoints_dir: Path,
        output_dir: Path,
        command_override: list[str] | None,
        corpus_jsonl_path: Path,
        num_processes: int = 1,
        dry_run_backend: bool = False,
        distributed_context: DistributedTrainingContext = DEFAULT_DISTRIBUTED_CONTEXT,
        metrics_path: Path | None = None,
        base_metrics: dict | None = None,
        status_log_path: Path | None = None,
        status_interval_seconds: float = 30.0,
        stream_logs: bool = False,
        log_tail_lines: int = 40,
        distributed_strategy: str = "ddp",
        artifact_format: str = ARTIFACT_FORMAT_XML,
        source_context_settings: SourceContextSettings | None = None,
        max_steps: int | None = None,
    ) -> BackendResult:
        del checkpoints_dir, command_override, distributed_context
        started_at = utc_now_iso()
        last_data_progress_emit = 0.0
        total_data_records = None
        if base_metrics is not None:
            total_data_records = int(base_metrics.get("corpus_records", 0) or 0) or None

        def training_data_progress(progress: dict[str, Any]) -> None:
            nonlocal last_data_progress_emit
            now = time.monotonic()
            if now - last_data_progress_emit < max(0.1, float(status_interval_seconds)):
                return
            last_data_progress_emit = now
            records_seen = int(progress.get("records_seen", 0) or 0)
            progress_snapshot = make_progress_snapshot(
                started_at=started_at,
                completed_units=records_seen,
                total_units=total_data_records,
            ).to_dict()
            payload = {
                "status": "training-data-progress",
                "run_id": run_dir.name,
                "backend": self.name,
                "total_records": total_data_records,
                "records_seen": records_seen,
                "trainable_samples": int(progress.get("trainable_samples", 0) or 0),
                "completed_valid_records": int(
                    progress.get("completed_valid_records", records_seen) or 0
                ),
                "worker_count": int(progress.get("worker_count", 1) or 1),
                "progress": progress_snapshot,
            }
            _write_live_training_metrics(
                metrics_path,
                base_metrics,
                {
                    "status": "preparing-training-data",
                    "training_data_progress": payload,
                    "progress": progress_snapshot,
                },
            )
            emit_status(payload, status_log_path=status_log_path)

        samples, training_data_diagnostics = _load_instruction_records_with_diagnostics(
            corpus_jsonl_path,
            profile,
            repo_root=paths.repo_root,
            artifact_format=artifact_format,
            source_context_settings=source_context_settings,
            progress_callback=training_data_progress,
        )
        train_jsonl_path = _write_jsonl(run_dir / "axolotl" / "train.jsonl", samples)
        config_path = run_dir / "axolotl" / "axolotl.yml"
        strategy = _normalize_distributed_strategy(distributed_strategy)
        deepspeed_config_path: Path | None = None
        if strategy in {"deepspeed-zero3", "deepspeed-zero3-offload"}:
            deepspeed_config_path = run_dir / "axolotl" / f"{strategy}.json"
            _write_json_file(
                deepspeed_config_path,
                _deepspeed_zero3_config(offload=strategy == "deepspeed-zero3-offload"),
            )
        stdout_log_path = run_dir / "axolotl" / "stdout.log"
        stderr_log_path = run_dir / "axolotl" / "stderr.log"
        source_policy = resolve_model_source_policy(paths)
        compat_sitecustomize_path = _write_axolotl_runtime_compat(run_dir, profile)
        effective_method = _validate_training_method(profile, self.name)
        config = _axolotl_config(
            profile=profile,
            train_jsonl_path=train_jsonl_path,
            prepared_dir=run_dir / "axolotl" / "prepared",
            output_dir=output_dir,
            source_policy=source_policy,
            distributed_strategy=strategy,
            deepspeed_config_path=deepspeed_config_path,
            max_steps=max_steps,
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = _axolotl_command(config_path=config_path, num_processes=num_processes)
        metrics = {
            "backend": self.name,
            "effective_training_method": effective_method,
            "artifact_kind": _artifact_kind_for_backend(
                backend=self.name,
                effective_training_method=effective_method,
            ),
            "samples": len(samples),
            "dataset_path": str(train_jsonl_path),
            "config_path": str(config_path),
            "command": command,
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
            "distributed_strategy": strategy,
            "deepspeed_config_path": str(deepspeed_config_path or ""),
            "runtime_compat_sitecustomize_path": str(compat_sitecustomize_path or ""),
            "model_source_policy": source_policy.to_dict(),
            "training_data_diagnostics": training_data_diagnostics,
        }
        if dry_run_backend:
            return BackendResult(
                status="dry-run",
                metrics={
                    **metrics,
                    "progress": make_progress_snapshot(
                        started_at=started_at,
                        completed_units=0,
                        total_units=0,
                    ).to_dict(),
                },
            )

        _ensure_deepspeed_available(strategy)
        subprocess_env = _axolotl_subprocess_environment(
            source_policy=source_policy,
            compat_sitecustomize_path=compat_sitecustomize_path,
        )
        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        status_interval_seconds = max(0.1, float(status_interval_seconds))
        log_tail_lines = max(0, int(log_tail_lines))
        last_status_emit = 0.0
        stdout_offset = 0
        stderr_offset = 0
        last_stdout_line = ""
        last_stderr_line = ""
        with (
            stdout_log_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_log_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            process = subprocess.Popen(
                command,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=paths.repo_root,
                env=subprocess_env,
            )
            emit_status(
                {
                    "status": "training-backend-started",
                    "run_id": run_dir.name,
                    "backend": self.name,
                    "pid": process.pid,
                    "command": command,
                    "config_path": str(config_path),
                },
                status_log_path=status_log_path,
            )
            while True:
                returncode = process.poll()
                done = returncode is not None
                stdout_handle.flush()
                stderr_handle.flush()
                stdout_offset, stdout_chunk, stdout_last = _read_new_log_lines(
                    stdout_log_path, stdout_offset, log_tail_lines
                )
                stderr_offset, stderr_chunk, stderr_last = _read_new_log_lines(
                    stderr_log_path, stderr_offset, log_tail_lines
                )
                if stdout_last:
                    last_stdout_line = stdout_last
                if stderr_last:
                    last_stderr_line = stderr_last
                if stream_logs and (stdout_chunk or stderr_chunk):
                    emit_status(
                        {
                            "status": "training-log",
                            "run_id": run_dir.name,
                            "backend": self.name,
                            "pid": process.pid,
                            "stdout": stdout_chunk,
                            "stderr": stderr_chunk,
                            "stdout_log_path": str(stdout_log_path),
                            "stderr_log_path": str(stderr_log_path),
                        },
                        status_log_path=status_log_path,
                    )
                run_status = "running"
                completed_units = 0
                if done:
                    run_status = "completed" if returncode == 0 else "failed"
                    completed_units = 1 if returncode == 0 else 0
                progress = make_progress_snapshot(
                    started_at=started_at,
                    completed_units=completed_units,
                    total_units=1,
                ).to_dict()
                live_metrics = {
                    **metrics,
                    "status": run_status,
                    "pid": process.pid,
                    "process_running": not done,
                    "returncode": returncode,
                    "artifact_dir": str(output_dir),
                    "progress": progress,
                    "last_stdout_line": last_stdout_line,
                    "last_stderr_line": last_stderr_line,
                }
                _write_live_training_metrics(metrics_path, base_metrics, live_metrics)
                now = time.monotonic()
                if done or now - last_status_emit >= status_interval_seconds:
                    emit_status(
                        {
                            "status": "training-progress",
                            "run_id": run_dir.name,
                            "run_status": run_status,
                            "backend": self.name,
                            "pid": process.pid,
                            "progress": progress,
                            "stdout_log_path": str(stdout_log_path),
                            "stderr_log_path": str(stderr_log_path),
                            "last_stdout_line": last_stdout_line,
                            "last_stderr_line": last_stderr_line,
                            "process_running": not done,
                        },
                        status_log_path=status_log_path,
                    )
                    last_status_emit = now
                if done:
                    break
                time.sleep(min(status_interval_seconds, 1.0))

        stdout_tail = _tail_file(stdout_log_path, 120)
        stderr_tail = _tail_file(stderr_log_path, 120)
        if returncode != 0:
            raise subprocess.CalledProcessError(
                int(returncode or 1),
                command,
                output=stdout_tail,
                stderr=stderr_tail,
            )
        return BackendResult(
            status="completed",
            metrics={
                **metrics,
                "stdout": stdout_tail,
                "stderr": stderr_tail,
                "pid": process.pid,
                "process_running": False,
                "returncode": returncode,
                "artifact_dir": str(output_dir),
                "last_stdout_line": _last_nonempty_line(stdout_log_path),
                "last_stderr_line": _last_nonempty_line(stderr_log_path),
                "progress": make_progress_snapshot(
                    started_at=started_at,
                    completed_units=1,
                    total_units=1,
                ).to_dict(),
            },
        )


def _mlx_lm_training_iterations(profile: TrainingProfile, sample_count: int) -> int:
    batch_units = max(1, profile.per_device_batch_size * profile.gradient_accumulation_steps)
    return max(1, math.ceil(max(1, sample_count) * max(1, profile.epochs) / batch_units))


def _mlx_lm_dataset_records(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "prompt": str(sample.get("instruction", "")),
            "completion": str(sample.get("output", "")),
        }
        for sample in samples
    ]


def _write_mlx_lm_dataset(data_dir: Path, samples: list[dict[str, str]]) -> dict[str, Path]:
    records = _mlx_lm_dataset_records(samples)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = _write_jsonl(data_dir / "train.jsonl", records)
    valid_count = min(max(1, len(records) // 20), 100)
    valid_path = _write_jsonl(data_dir / "valid.jsonl", records[:valid_count])
    return {"data_dir": data_dir, "train_path": train_path, "valid_path": valid_path}


def _mlx_lm_config(
    *,
    profile: TrainingProfile,
    data_dir: Path,
    output_dir: Path,
    sample_count: int,
    max_steps: int | None = None,
) -> dict:
    iters = (
        int(max_steps)
        if max_steps is not None
        else _mlx_lm_training_iterations(profile, sample_count)
    )
    effective_method = _validate_training_method(profile, MLXLMTrainingBackend.name)
    uses_mistral_common = _profile_uses_mistral_common_tokenizer(profile)
    config = {
        "model": profile.base_model,
        "train": True,
        "fine_tune_type": effective_method,
        "data": str(data_dir),
        "adapter_path": str(output_dir),
        "seed": profile.seed,
        "num_layers": 0 if effective_method == "full" else 16,
        "batch_size": profile.per_device_batch_size,
        "iters": iters,
        "val_batches": -1,
        "learning_rate": profile.learning_rate,
        "steps_per_report": max(1, min(10, iters)),
        "steps_per_eval": max(1, iters),
        "save_every": max(1, iters),
        "max_seq_length": profile.max_seq_length,
        "grad_checkpoint": True,
        "grad_accumulation_steps": profile.gradient_accumulation_steps,
        "mask_prompt": not uses_mistral_common,
        "optimizer": "adamw",
        "trust_remote_code": True,
    }
    if uses_mistral_common:
        config["tokenizer_config"] = {"mode": "finetuning"}
    if effective_method == "lora":
        config["lora_parameters"] = {
            "rank": profile.lora_rank,
            "dropout": profile.lora_dropout,
            "scale": profile.lora_alpha,
        }
    return config


def _mlx_lm_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "galaxy_toolsmith.runtime.mlx_lm_lora",
        "--config",
        str(config_path),
    ]


class MLXLMTrainingBackend(TrainingBackend):
    name = "mlx-lm"

    def run(
        self,
        paths: WorkspacePaths,
        profile: TrainingProfile,
        run_dir: Path,
        checkpoints_dir: Path,
        output_dir: Path,
        command_override: list[str] | None,
        corpus_jsonl_path: Path,
        num_processes: int = 1,
        dry_run_backend: bool = False,
        distributed_context: DistributedTrainingContext = DEFAULT_DISTRIBUTED_CONTEXT,
        metrics_path: Path | None = None,
        base_metrics: dict | None = None,
        status_log_path: Path | None = None,
        status_interval_seconds: float = 30.0,
        stream_logs: bool = False,
        log_tail_lines: int = 40,
        distributed_strategy: str = "ddp",
        artifact_format: str = ARTIFACT_FORMAT_XML,
        source_context_settings: SourceContextSettings | None = None,
        max_steps: int | None = None,
    ) -> BackendResult:
        del checkpoints_dir, command_override, distributed_context, distributed_strategy
        if num_processes != 1:
            raise ValueError("mlx-lm training currently supports --num-processes 1 only.")
        started_at = utc_now_iso()
        samples, training_data_diagnostics = _load_instruction_records_with_diagnostics(
            corpus_jsonl_path,
            profile,
            repo_root=paths.repo_root,
            artifact_format=artifact_format,
            source_context_settings=source_context_settings,
        )
        mlx_dir = run_dir / "mlx-lm"
        dataset_paths = _write_mlx_lm_dataset(mlx_dir / "data", samples)
        config_path = mlx_dir / "lora.yml"
        stdout_log_path = mlx_dir / "stdout.log"
        stderr_log_path = mlx_dir / "stderr.log"
        source_policy = resolve_model_source_policy(paths)
        config = _mlx_lm_config(
            profile=profile,
            data_dir=dataset_paths["data_dir"],
            output_dir=output_dir,
            sample_count=len(samples),
            max_steps=max_steps,
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = _mlx_lm_command(config_path)
        effective_method = _validate_training_method(profile, self.name)
        metrics = {
            "backend": self.name,
            "effective_training_method": effective_method,
            "artifact_kind": _artifact_kind_for_backend(
                backend=self.name,
                effective_training_method=effective_method,
            ),
            "samples": len(samples),
            "data_dir": str(dataset_paths["data_dir"]),
            "train_jsonl_path": str(dataset_paths["train_path"]),
            "valid_jsonl_path": str(dataset_paths["valid_path"]),
            "config_path": str(config_path),
            "command": command,
            "stdout_log_path": str(stdout_log_path),
            "stderr_log_path": str(stderr_log_path),
            "adapter_path": str(output_dir),
            "artifact_dir": str(output_dir),
            "model_source_policy": source_policy.to_dict(),
            "training_data_diagnostics": training_data_diagnostics,
        }
        if dry_run_backend:
            return BackendResult(
                status="dry-run",
                metrics={
                    **metrics,
                    "progress": make_progress_snapshot(
                        started_at=started_at,
                        completed_units=0,
                        total_units=0,
                    ).to_dict(),
                },
            )

        stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess_env = merged_model_source_environment(None, source_policy)
        status_interval_seconds = max(0.1, float(status_interval_seconds))
        log_tail_lines = max(0, int(log_tail_lines))
        last_status_emit = 0.0
        stdout_offset = 0
        stderr_offset = 0
        last_stdout_line = ""
        last_stderr_line = ""
        with (
            stdout_log_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_log_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            process = subprocess.Popen(
                command,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=paths.repo_root,
                env=subprocess_env,
            )
            emit_status(
                {
                    "status": "training-backend-started",
                    "run_id": run_dir.name,
                    "backend": self.name,
                    "pid": process.pid,
                    "command": command,
                    "config_path": str(config_path),
                },
                status_log_path=status_log_path,
            )
            while True:
                returncode = process.poll()
                done = returncode is not None
                stdout_handle.flush()
                stderr_handle.flush()
                stdout_offset, stdout_chunk, stdout_last = _read_new_log_lines(
                    stdout_log_path, stdout_offset, log_tail_lines
                )
                stderr_offset, stderr_chunk, stderr_last = _read_new_log_lines(
                    stderr_log_path, stderr_offset, log_tail_lines
                )
                if stdout_last:
                    last_stdout_line = stdout_last
                if stderr_last:
                    last_stderr_line = stderr_last
                if stream_logs and (stdout_chunk or stderr_chunk):
                    emit_status(
                        {
                            "status": "training-log",
                            "run_id": run_dir.name,
                            "backend": self.name,
                            "pid": process.pid,
                            "stdout": stdout_chunk,
                            "stderr": stderr_chunk,
                            "stdout_log_path": str(stdout_log_path),
                            "stderr_log_path": str(stderr_log_path),
                        },
                        status_log_path=status_log_path,
                    )
                run_status = "running"
                completed_units = 0
                if done:
                    run_status = "completed" if returncode == 0 else "failed"
                    completed_units = 1 if returncode == 0 else 0
                progress = make_progress_snapshot(
                    started_at=started_at,
                    completed_units=completed_units,
                    total_units=1,
                ).to_dict()
                live_metrics = {
                    **metrics,
                    "status": run_status,
                    "pid": process.pid,
                    "process_running": not done,
                    "returncode": returncode,
                    "progress": progress,
                    "last_stdout_line": last_stdout_line,
                    "last_stderr_line": last_stderr_line,
                }
                _write_live_training_metrics(metrics_path, base_metrics, live_metrics)
                now = time.monotonic()
                if done or now - last_status_emit >= status_interval_seconds:
                    emit_status(
                        {
                            "status": "training-progress",
                            "run_id": run_dir.name,
                            "run_status": run_status,
                            "backend": self.name,
                            "pid": process.pid,
                            "progress": progress,
                            "stdout_log_path": str(stdout_log_path),
                            "stderr_log_path": str(stderr_log_path),
                            "last_stdout_line": last_stdout_line,
                            "last_stderr_line": last_stderr_line,
                            "process_running": not done,
                        },
                        status_log_path=status_log_path,
                    )
                    last_status_emit = now
                if done:
                    break
                time.sleep(min(status_interval_seconds, 1.0))

        stdout_tail = _tail_file(stdout_log_path, 120)
        stderr_tail = _tail_file(stderr_log_path, 120)
        if returncode != 0:
            raise subprocess.CalledProcessError(
                int(returncode or 1),
                command,
                output=stdout_tail,
                stderr=stderr_tail,
            )
        return BackendResult(
            status="completed",
            metrics={
                **metrics,
                "stdout": stdout_tail,
                "stderr": stderr_tail,
                "pid": process.pid,
                "process_running": False,
                "returncode": returncode,
                "last_stdout_line": _last_nonempty_line(stdout_log_path),
                "last_stderr_line": _last_nonempty_line(stderr_log_path),
                "progress": make_progress_snapshot(
                    started_at=started_at,
                    completed_units=1,
                    total_units=1,
                ).to_dict(),
            },
        )


def _supports_intended_backend(profile: TrainingProfile, capabilities: RuntimeCapabilities) -> bool:
    backend = profile.backend.strip().lower()
    if backend in {"axolotl", "cuda", "rocm"}:
        return capabilities.cuda_available or capabilities.rocm_available
    if backend in MLX_LM_BACKEND_ALIASES:
        return capabilities.mps_available
    return backend in {"cpu", "hf-sft"}


def _select_backend(
    profile: TrainingProfile,
    command_override: list[str] | None,
    capabilities: RuntimeCapabilities,
    backend_override: str = "auto",
    dry_run_backend: bool = False,
) -> BackendSelection:
    requested_backend = (backend_override or "auto").strip().lower()
    intended_backend = (
        str(profile.backend).strip().lower() if requested_backend == "auto" else requested_backend
    )
    if (
        command_override
        or (requested_backend in {"auto", "command"} and profile.default_command)
        or requested_backend == "command"
    ):
        return BackendSelection(
            backend=CommandTrainingBackend(),
            intended_backend=intended_backend,
            selected_backend=CommandTrainingBackend.name,
            fallback_reason="",
            intended_methodology_supported=_supports_intended_backend(profile, capabilities),
        )
    supported_profile = _supports_intended_backend(
        replace(profile, backend=intended_backend), capabilities
    )
    if intended_backend == "axolotl":
        if supported_profile or dry_run_backend:
            return BackendSelection(
                backend=AxolotlTrainingBackend(),
                intended_backend=intended_backend,
                selected_backend=AxolotlTrainingBackend.name,
                fallback_reason="",
                intended_methodology_supported=supported_profile,
            )
        return BackendSelection(
            backend=HFSFTTrainingBackend(),
            intended_backend=intended_backend,
            selected_backend=HFSFTTrainingBackend.name,
            fallback_reason="intended_backend=axolotl unsupported on this runtime; falling back to hf-sft",
            intended_methodology_supported=False,
        )
    if intended_backend in MLX_LM_BACKEND_ALIASES:
        return BackendSelection(
            backend=MLXLMTrainingBackend(),
            intended_backend=intended_backend,
            selected_backend=MLXLMTrainingBackend.name,
            fallback_reason="",
            intended_methodology_supported=supported_profile,
        )
    if intended_backend in {"hf-sft", "cpu"}:
        return BackendSelection(
            backend=HFSFTTrainingBackend(),
            intended_backend=intended_backend,
            selected_backend=HFSFTTrainingBackend.name,
            fallback_reason="",
            intended_methodology_supported=supported_profile,
        )

    fallback_reason = f"intended_backend={intended_backend} not directly implemented; using hf-sft"
    return BackendSelection(
        backend=HFSFTTrainingBackend(),
        intended_backend=intended_backend,
        selected_backend=HFSFTTrainingBackend.name,
        fallback_reason=fallback_reason,
        intended_methodology_supported=supported_profile,
    )


def _hf_sft_distributed_launch_command(
    *,
    paths: WorkspacePaths,
    profile: TrainingProfile,
    dataset_manifest_path: Path,
    corpus_jsonl_path: Path,
    run_id: str,
    num_processes: int,
    variant_id: str | None,
    profile_overrides: TrainingProfileOverrides | None = None,
    status_log_path: Path | None = None,
    status_interval_seconds: float = 30.0,
    stream_logs: bool = False,
    log_tail_lines: int = 40,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
    max_steps: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(num_processes),
        "-m",
        "galaxy_toolsmith.cli.main",
        "--repo-root",
        str(paths.repo_root),
        "train",
        "--profile",
        profile.name,
        "--dataset-manifest",
        str(dataset_manifest_path),
        "--corpus-jsonl",
        str(corpus_jsonl_path),
        "--artifact-format",
        artifact_format.replace("_", "-"),
        "--backend",
        "hf-sft",
        "--internal-run-id",
        run_id,
        "--internal-distributed-child",
    ]
    if variant_id:
        command.extend(["--variant-id", variant_id])
    overrides = profile_overrides or TrainingProfileOverrides()
    if overrides.max_seq_length is not None:
        command.extend(["--max-seq-length", str(overrides.max_seq_length)])
    if overrides.pad_to_sequence_len is not None:
        command.append(
            "--pad-to-sequence-len" if overrides.pad_to_sequence_len else "--no-pad-to-sequence-len"
        )
    if overrides.attn_implementation is not None:
        command.extend(["--attn-implementation", overrides.attn_implementation])
    if overrides.per_device_batch_size is not None:
        command.extend(["--per-device-batch-size", str(overrides.per_device_batch_size)])
    if overrides.gradient_accumulation_steps is not None:
        command.extend(
            [
                "--gradient-accumulation-steps",
                str(overrides.gradient_accumulation_steps),
            ]
        )
    if overrides.learning_rate is not None:
        command.extend(["--learning-rate", str(overrides.learning_rate)])
    if overrides.training_method is not None:
        command.extend(["--training-method", overrides.training_method])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    source_context_settings = (source_context_settings or SourceContextSettings()).normalized()
    if source_context_settings.mode != "none":
        command.extend(["--source-context-mode", source_context_settings.mode])
        command.extend(["--source-context-max-chars", str(source_context_settings.max_chars)])
        command.extend(["--source-context-max-files", str(source_context_settings.max_files)])
        if source_context_settings.source_root is not None:
            command.extend(["--source-root", str(source_context_settings.source_root)])
        if source_context_settings.source_file is not None:
            command.extend(["--source-file", str(source_context_settings.source_file)])
    if source_context_settings.test_context_mode != "none":
        command.extend(["--test-context-mode", source_context_settings.test_context_mode])
        command.extend(
            ["--test-context-max-chars", str(source_context_settings.test_context_max_chars)]
        )
        command.extend(
            ["--test-context-max-files", str(source_context_settings.test_context_max_files)]
        )
        command.extend(
            [
                "--test-context-max-file-bytes",
                str(source_context_settings.test_context_max_file_bytes),
            ]
        )
    if status_log_path is not None:
        command.extend(["--status-log", str(status_log_path)])
    command.extend(["--status-interval-seconds", str(status_interval_seconds)])
    if stream_logs:
        command.append("--stream-logs")
    command.extend(["--log-tail-lines", str(log_tail_lines)])
    return command


def run_training(
    paths: WorkspacePaths,
    profile: TrainingProfile,
    dataset_manifest_path: Path,
    command_override: list[str] | None,
    variant_id: str | None = None,
    corpus_jsonl_path: Path | None = None,
    backend_override: str = "auto",
    num_processes: int = 1,
    dry_run_backend: bool = False,
    run_id_override: str | None = None,
    distributed_child: bool = False,
    profile_overrides: TrainingProfileOverrides | None = None,
    distributed_strategy: str | None = None,
    status_log_path: Path | None = None,
    status_interval_seconds: float = 30.0,
    stream_logs: bool = False,
    log_tail_lines: int = 40,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
    max_steps: int | None = None,
) -> TrainingRunManifest:
    artifact_format = normalize_training_artifact_format(artifact_format)
    if num_processes < 1:
        raise ValueError("--num-processes must be at least 1.")
    if max_steps is not None and max_steps < 1:
        raise ValueError("--max-steps must be at least 1 when provided.")
    if status_interval_seconds <= 0:
        raise ValueError("--status-interval-seconds must be greater than 0.")
    if log_tail_lines < 0:
        raise ValueError("--log-tail-lines must be at least 0.")
    profile_overrides = profile_overrides or TrainingProfileOverrides()
    profile = _apply_training_profile_overrides(profile, profile_overrides)
    source_context_settings = (source_context_settings or SourceContextSettings()).normalized()
    effective_max_steps = int(max_steps) if max_steps is not None else None
    distributed_context = _distributed_context(distributed_child)
    writes_metadata = distributed_context.is_rank_zero
    run_id = run_id_override or f"train-{uuid.uuid4().hex[:12]}"
    run_dir = paths.runs_root / "training" / run_id
    checkpoints_dir = run_dir / "checkpoints"
    output_dir = run_dir / "output"
    metrics_path = run_dir / "metrics.json"
    run_manifest_path = run_dir / "run.manifest.json"
    variants_dir = paths.models_root / "variants"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = _load_dataset_id(dataset_manifest_path)
    corpus_path = corpus_jsonl_path or (paths.datasets_root / "tools-iuc-corpus.jsonl")
    command = command_override if command_override else list(profile.default_command)
    capabilities = detect_runtime_capabilities()
    corpus_help_counts = _corpus_container_help_counts(corpus_path)
    backend_selection = _select_backend(
        profile=profile,
        command_override=command_override,
        capabilities=capabilities,
        backend_override=backend_override,
        dry_run_backend=dry_run_backend,
    )
    backend = backend_selection.backend
    effective_training_method = _validate_training_method(
        profile,
        backend_selection.selected_backend,
    )
    artifact_kind = _artifact_kind_for_backend(
        backend=backend_selection.selected_backend,
        effective_training_method=effective_training_method,
    )
    effective_distributed_strategy = _resolve_distributed_strategy(
        profile=profile,
        requested_strategy=distributed_strategy,
        selected_backend=backend_selection.selected_backend,
        num_processes=num_processes,
    )

    run = TrainingRunManifest(
        run_id=run_id,
        profile_name=profile.name,
        backend=backend_selection.selected_backend,
        provider=profile.provider,
        base_model=profile.base_model,
        quantization=profile.quantization,
        training_method=profile.training_method,
        effective_training_method=effective_training_method,
        artifact_kind=artifact_kind,
        dataset_manifest_path=str(dataset_manifest_path),
        dataset_id=dataset_id,
        command=command,
        status="running",
        output_dir=str(output_dir),
        checkpoints_dir=str(checkpoints_dir),
        metrics_path=str(metrics_path),
    )
    if writes_metadata:
        run_manifest_path.write_text(run.to_json(), encoding="utf-8")

    base_metrics = {
        "status": "running",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "backend_impl": backend_selection.selected_backend,
        "intended_backend": backend_selection.intended_backend,
        "intended_methodology_supported": backend_selection.intended_methodology_supported,
        "backend_fallback_reason": backend_selection.fallback_reason,
        "source_quantization": profile.quantization,
        "requested_training_method": profile.training_method,
        "effective_training_method": effective_training_method,
        "artifact_kind": artifact_kind,
        "artifact_format": artifact_format,
        "source_context": source_context_settings.to_dict(),
        "training_data_manifest": {
            "schema_version": 1,
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_manifest_sha256": _file_sha256(dataset_manifest_path),
            "corpus_jsonl_path": str(corpus_path),
            "corpus_jsonl_sha256": _file_sha256(corpus_path),
            "artifact_format": artifact_format,
            "source_context": source_context_settings.to_dict(),
        },
        "requested_distributed_strategy": distributed_strategy or profile.distributed_strategy,
        "distributed_strategy": effective_distributed_strategy,
        "training_profile_overrides": profile_overrides.to_dict(),
        "effective_training_profile": {
            "max_seq_length": profile.max_seq_length,
            "pad_to_sequence_len": profile.pad_to_sequence_len,
            "attn_implementation": profile.attn_implementation,
            "per_device_batch_size": profile.per_device_batch_size,
            "gradient_accumulation_steps": profile.gradient_accumulation_steps,
            "distributed_strategy": effective_distributed_strategy,
        },
        "corpus_jsonl_path": str(corpus_path),
        **corpus_help_counts,
        "runtime_capabilities": capabilities.to_dict(),
    }
    if effective_max_steps is not None:
        base_metrics["max_steps"] = effective_max_steps
        base_metrics["effective_training_profile"]["max_steps"] = effective_max_steps
    if writes_metadata:
        _write_json_file(metrics_path, base_metrics)
        emit_status(
            {
                "status": "training-started",
                "run_id": run_id,
                "backend": backend_selection.selected_backend,
                "distributed_strategy": effective_distributed_strategy,
                "profile": profile.name,
                "metrics_path": str(metrics_path),
            },
            status_log_path=status_log_path,
        )

    if (
        backend_selection.selected_backend == HFSFTTrainingBackend.name
        and num_processes > 1
        and not distributed_child
        and not dry_run_backend
    ):
        launch_command = _hf_sft_distributed_launch_command(
            paths=paths,
            profile=profile,
            dataset_manifest_path=dataset_manifest_path,
            corpus_jsonl_path=corpus_path,
            run_id=run_id,
            num_processes=num_processes,
            variant_id=variant_id,
            profile_overrides=profile_overrides,
            status_log_path=status_log_path,
            status_interval_seconds=status_interval_seconds,
            stream_logs=stream_logs,
            log_tail_lines=log_tail_lines,
            artifact_format=artifact_format,
            source_context_settings=source_context_settings,
            max_steps=effective_max_steps,
        )
        try:
            completed = subprocess.run(
                launch_command,
                check=True,
                text=True,
                capture_output=True,
                cwd=paths.repo_root,
            )
            if run_manifest_path.exists():
                distributed_run = _training_run_manifest_from_path(run_manifest_path)
            else:
                distributed_run = replace(run, status="completed")
            parent_metrics_path = run_dir / "distributed-launch.json"
            parent_metrics_path.write_text(
                json.dumps(
                    {
                        "command": launch_command,
                        "returncode": completed.returncode,
                        "stdout": _tail_text(completed.stdout),
                        "stderr": _tail_text(completed.stderr),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return distributed_run
        except subprocess.CalledProcessError as error:
            metrics = {
                **base_metrics,
                "status": "failed",
                "backend_impl": backend_selection.selected_backend,
                "intended_backend": backend_selection.intended_backend,
                "command": launch_command,
                "stdout": _tail_text(error.stdout or ""),
                "stderr": _tail_text(error.stderr or ""),
                "returncode": error.returncode,
                **corpus_help_counts,
            }
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            failed = replace(run, status="failed", error=_training_command_error(error))
            run_manifest_path.write_text(failed.to_json(), encoding="utf-8")
            return failed

    try:
        backend_result = backend.run(
            paths=paths,
            profile=profile,
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            output_dir=output_dir,
            command_override=command_override,
            corpus_jsonl_path=corpus_path,
            num_processes=num_processes,
            dry_run_backend=dry_run_backend,
            distributed_context=distributed_context,
            metrics_path=metrics_path if writes_metadata else None,
            base_metrics=base_metrics,
            status_log_path=status_log_path,
            status_interval_seconds=status_interval_seconds,
            stream_logs=stream_logs,
            log_tail_lines=log_tail_lines,
            distributed_strategy=effective_distributed_strategy,
            artifact_format=artifact_format,
            source_context_settings=source_context_settings,
            max_steps=effective_max_steps,
        )
        metrics = {
            **base_metrics,
            "status": backend_result.status,
            **backend_result.metrics,
        }
        if writes_metadata:
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            emit_status(
                {
                    "status": "training-finished",
                    "run_id": run_id,
                    "run_status": backend_result.status,
                    "backend": backend_selection.selected_backend,
                    "metrics_path": str(metrics_path),
                },
                status_log_path=status_log_path,
            )
    except subprocess.CalledProcessError as error:
        metrics = {
            **base_metrics,
            "status": "failed",
            "backend_impl": backend.name,
            "stdout": (error.stdout or "").strip(),
            "stderr": (error.stderr or "").strip(),
            "returncode": error.returncode,
            **corpus_help_counts,
        }
        if writes_metadata:
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            emit_status(
                {
                    "status": "training-finished",
                    "run_id": run_id,
                    "run_status": "failed",
                    "backend": backend.name,
                    "metrics_path": str(metrics_path),
                },
                status_log_path=status_log_path,
            )
        failed = replace(run, status="failed", error=_training_command_error(error))
        if writes_metadata:
            run_manifest_path.write_text(failed.to_json(), encoding="utf-8")
        return failed
    except Exception as error:
        metrics = {
            **base_metrics,
            "status": "failed",
            "backend_impl": backend.name,
            "error": str(error),
        }
        if writes_metadata:
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            emit_status(
                {
                    "status": "training-finished",
                    "run_id": run_id,
                    "run_status": "failed",
                    "backend": backend.name,
                    "metrics_path": str(metrics_path),
                },
                status_log_path=status_log_path,
            )
        failed = replace(run, status="failed", error=str(error))
        if writes_metadata:
            run_manifest_path.write_text(failed.to_json(), encoding="utf-8")
        return failed

    if not writes_metadata:
        return replace(run, status=backend_result.status)

    if backend_result.status != "completed":
        incomplete_run = replace(run, status=backend_result.status, error="")
        run_manifest_path.write_text(incomplete_run.to_json(), encoding="utf-8")
        return incomplete_run

    resolved_variant_id = variant_id or f"{dataset_id}-{profile.name}".replace("/", "-")
    variant_manifest = ModelVariantManifest(
        variant_id=resolved_variant_id,
        base_model=profile.base_model,
        quantization=profile.quantization,
        training_dataset_id=dataset_id,
        provider=profile.provider,
        skills_profile=profile.skills_profile,
        backend=backend_selection.selected_backend,
        training_method=profile.training_method,
        effective_training_method=effective_training_method,
        artifact_kind=artifact_kind,
        artifact_dir=str(output_dir),
    )
    variant_path = variants_dir / f"{resolved_variant_id}.manifest.json"
    variant_path.write_text(variant_manifest.to_json(), encoding="utf-8")

    completed_run = replace(
        run,
        status="completed",
        model_variant_path=str(variant_path),
        error="",
    )
    run_manifest_path.write_text(completed_run.to_json(), encoding="utf-8")
    return completed_run
