from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    normalize_artifact_format,
    output_path_from_record,
    output_suffix_for_artifact_format,
)
from galaxy_toolsmith.inference.evaluation import evaluate_wrapper_paths
from galaxy_toolsmith.inference.generation import (
    _get_provider,
    _validation_with_generation_diagnostics,
    generate_wrapper_from_content,
)
from galaxy_toolsmith.inference.prompt_context import DEFAULT_MAX_PROMPT_HELP_CHARS
from galaxy_toolsmith.inference.source_context import (
    SourceContextSettings,
    build_source_context_from_record,
)
from galaxy_toolsmith.inference.validation import PlanemoTestOptions
from galaxy_toolsmith.runtime.progress import make_progress_snapshot
from galaxy_toolsmith.runtime.status import emit_status

REPAIR_MAX_PROMPT_HELP_CHARS = 6000
TRUNCATION_REPAIR_MAX_PROMPT_HELP_CHARS = 1500
DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS = 1
LOCAL_GPU_TOPOLOGIES = {"per-process", "model-parallel"}
LOCAL_OFFLOAD_POLICIES = {"allow", "fail"}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_since(started_perf: float) -> float:
    return round(time.perf_counter() - started_perf, 4)


@dataclass(frozen=True)
class BenchmarkSummary:
    created_at: str
    corpus_path: str
    provider: str
    model_variant: str
    attempted: int
    succeeded: int
    failed: int
    wrappers_dir: str
    generation_records_path: str
    evaluation_report_path: str
    artifact_format: str = ARTIFACT_FORMAT_XML
    progress: dict = field(default_factory=dict)
    startup: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    quality: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class BenchmarkShard:
    index: int
    corpus_jsonl: Path
    wrappers_dir: Path
    generation_records_path: Path
    evaluation_report_path: Path
    benchmark_summary_path: Path
    checkpoint_records_path: Path
    status_log_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    gpu_device: str = ""


def _load_corpus_records(corpus_jsonl: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with corpus_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            records.append(json.loads(value))
            if limit is not None and len(records) >= limit:
                break
    return records


def _write_corpus_records(corpus_jsonl: Path, records: list[dict]) -> None:
    corpus_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with corpus_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _with_corpus_indices(records: list[dict]) -> list[dict]:
    indexed: list[dict] = []
    for index, record in enumerate(records):
        item = dict(record)
        if "_benchmark_corpus_index" not in item:
            item["_benchmark_corpus_index"] = index
        indexed.append(item)
    return indexed


def _record_metadata(record: dict) -> dict:
    return {
        "corpus_index": int(record.get("_benchmark_corpus_index", -1)),
        "package_id": str(record.get("package_id", "")),
        "tool_id": str(record.get("tool_id", "")),
        "wrapper_path": str(record.get("wrapper_path", "")),
        "expanded_xml_path": str(record.get("expanded_xml_path", "")),
        "primary_command": str(record.get("primary_command", "")),
    }


def _annotate_generation_record(generated: dict, record: dict) -> dict:
    generated.update(_record_metadata(record))
    return generated


def _annotate_failure(failure: dict, record: dict) -> dict:
    failure.update(_record_metadata(record))
    return failure


def _prompt_metadata_hints(record: dict) -> dict:
    return {
        "package_id": str(record.get("package_id", "")),
        "tool_id": str(record.get("tool_id", "")),
        "primary_command": str(record.get("primary_command", "")),
    }


def _safe_wrapper_filename(tool_name: str, artifact_format: str = ARTIFACT_FORMAT_XML) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", tool_name.strip()).strip("._-")
    if not stem:
        stem = "unknown_tool"
    stem = stem[:80].rstrip("._-") or "unknown_tool"
    digest = sha1(tool_name.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{stem}-{digest}{output_suffix_for_artifact_format(artifact_format)}"


def _safe_xml_id(value: str, *, fallback: str = "generated_tool") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")
    if not cleaned:
        cleaned = fallback
    if not re.match(r"^[A-Za-z_]", cleaned):
        cleaned = f"tool_{cleaned}"
    return cleaned[:80].rstrip("_") or fallback


def _fallback_requirement(record: dict) -> str:
    package_id = str(record.get("package_id", "")).strip()
    if "/" in package_id:
        package_id = package_id.rsplit("/", 1)[-1]
    candidate = package_id or str(record.get("primary_command", "")).strip()
    candidate = re.sub(r"[^A-Za-z0-9_.+-]+", "-", candidate.lower()).strip("-")
    return candidate


def _xml_attr(value: str) -> str:
    return escape(value, {'"': "&quot;"})


def _compact_fallback_xml(*, tool_name: str, record: dict) -> str:
    tool_id = _safe_xml_id(str(record.get("tool_id") or tool_name))
    command_name = str(record.get("primary_command", "")).strip() or tool_id
    command_name = command_name.replace("]]>", "]] >")
    requirement = _fallback_requirement(record)
    requirement_block = (
        f"""
    <requirements>
        <requirement type="package">{escape(requirement)}</requirement>
    </requirements>"""
        if requirement
        else ""
    )
    label = _xml_attr(tool_name.strip() or tool_id)
    command_text = _xml_attr(command_name)
    return f"""<tool id="{tool_id}" name="{label}" version="0.1.0" profile="25.0">
    <description>compact generated wrapper fallback</description>{requirement_block}
    <command detect_errors="aggressive"><![CDATA[
{command_name} --help > '$out_file'
    ]]></command>
    <inputs>
        <param name="input" type="data" format="txt" optional="true" label="Optional input file"/>
    </inputs>
    <outputs>
        <data name="out_file" format="txt"/>
    </outputs>
    <tests>
        <test expect_num_outputs="1">
            <output name="out_file">
                <assert_contents>
                    <has_text text="{command_text}"/>
                </assert_contents>
            </output>
        </test>
    </tests>
    <help><![CDATA[
Compact fallback wrapper generated because model output was truncated.
Review command options and datatypes before production use.
    ]]></help>
</tool>"""


def _compact_fallback_generation_record(
    *,
    paths: WorkspacePaths,
    tool_name: str,
    record: dict,
    output_xml: Path,
    provider: str,
    model_variant: str,
    repair_reason: str,
    initial_failure: dict,
    repaired_failure: dict,
) -> dict:
    xml = _compact_fallback_xml(tool_name=tool_name, record=record)
    _atomic_write_text(output_xml, xml)
    validation = _validation_with_generation_diagnostics(xml)
    request_id = f"gen-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    report_path = paths.runs_root / "generation" / request_id / "report.json"
    _atomic_write_text(report_path, json.dumps(validation, indent=2))
    return {
        "request_id": request_id,
        "created_at": utc_now_iso(),
        "tool_name": tool_name,
        "provider": f"{provider}-compact-fallback",
        "model_variant": model_variant,
        "skills_profile": "default",
        "artifact_format": ARTIFACT_FORMAT_XML,
        "output_path": str(output_xml),
        "output_xml_path": str(output_xml),
        "output_udt_yaml_path": "",
        "report_path": str(report_path),
        "validation": validation,
        "prompt_help": {},
        "attempt_count": 3,
        "repair_attempted": True,
        "repair_reason": repair_reason,
        "compact_fallback_attempted": True,
        "initial_failure": {
            "error": initial_failure.get("error", ""),
            "error_type": initial_failure.get("error_type", ""),
            "validation": initial_failure.get("validation", {}),
        },
        "repaired_failure": {
            "error": repaired_failure.get("error", ""),
            "error_type": repaired_failure.get("error_type", ""),
            "validation": repaired_failure.get("validation", {}),
        },
    }


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _path_if_exists(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def _default_checkpoint_path(generation_records_path: Path) -> Path:
    return generation_records_path.with_suffix(".checkpoint.jsonl")


def _load_successful_checkpoint_records(
    path: Path,
    *,
    artifact_format: str = ARTIFACT_FORMAT_XML,
) -> list[dict]:
    records_by_index: dict[int, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        try:
            corpus_index = int(record.get("corpus_index"))
        except (TypeError, ValueError):
            continue
        if not _artifact_validation_passed(record.get("validation"), artifact_format):
            continue
        if output_path_from_record(record, artifact_format) is None:
            continue
        records_by_index[corpus_index] = record
    return [records_by_index[index] for index in sorted(records_by_index)]


def _append_checkpoint_record(path: Path, record: dict, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _write_checkpoint_records(path: Path, records: list[dict]) -> None:
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    _atomic_write_text(path, text)


def _xml_features(path: Path) -> dict | None:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None
    input_datatypes: set[str] = set()
    output_datatypes: set[str] = set()
    requirement_packages: set[str] = set()
    command_parts: list[str] = []

    input_count = 0
    for param in root.findall(".//inputs//param"):
        input_count += 1
        value = param.attrib.get("format") or param.attrib.get("ext")
        _add_xml_datatypes(input_datatypes, value)

    output_count = 0
    for output in [*root.findall(".//outputs//data"), *root.findall(".//outputs//collection")]:
        output_count += 1
        value = output.attrib.get("format") or output.attrib.get("type") or output.attrib.get("ext")
        _add_xml_datatypes(output_datatypes, value)

    for requirement in root.findall(".//requirements//requirement"):
        if (requirement.attrib.get("type") or "package") != "package":
            continue
        if requirement.text and requirement.text.strip():
            requirement_packages.add(requirement.text.strip())

    for command in root.findall(".//command"):
        if command.text:
            command_parts.append(command.text)

    return {
        "input_count": input_count,
        "output_count": output_count,
        "input_datatypes": input_datatypes,
        "output_datatypes": output_datatypes,
        "requirement_packages": requirement_packages,
        "has_tests": bool(root.findall(".//tests//test")),
        "command_text": "\n".join(command_parts).lower(),
    }


def _udt_features(path: Path) -> dict | None:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    inputs = data.get("inputs") or []
    outputs = data.get("outputs") or []
    input_datatypes: set[str] = set()
    output_datatypes: set[str] = set()
    if not isinstance(inputs, list):
        inputs = []
    if not isinstance(outputs, list):
        outputs = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"data", "data_collection"}:
            value = item.get("format") or ",".join(str(v) for v in item.get("extensions", []) or [])
            _add_xml_datatypes(input_datatypes, str(value) if value else None)
    for item in outputs:
        if not isinstance(item, dict):
            continue
        value = item.get("format") or item.get("format_source")
        _add_xml_datatypes(output_datatypes, str(value) if value else None)
    requirement_packages = {str(data.get("container", "") or "").strip()} - {""}
    return {
        "input_count": len(inputs),
        "output_count": len(outputs),
        "input_datatypes": input_datatypes,
        "output_datatypes": output_datatypes,
        "requirement_packages": requirement_packages,
        "has_tests": bool(data.get("tests")),
        "command_text": str(data.get("shell_command", "") or "").lower(),
    }


def _add_xml_datatypes(datatypes: set[str], value: str | None) -> None:
    if not value:
        return
    datatypes.update(dtype.strip() for dtype in value.split(",") if dtype.strip())


def _reference_path_for_record(record: dict) -> Path | None:
    return _path_if_exists(record.get("expanded_xml_path")) or _path_if_exists(
        record.get("wrapper_path")
    )


def _reference_fidelity(
    generation_records: list[dict],
    *,
    artifact_format: str = ARTIFACT_FORMAT_XML,
) -> dict:
    compared = 0
    input_errors: list[float] = []
    output_errors: list[float] = []
    input_jaccards: list[float] = []
    output_jaccards: list[float] = []
    requirement_jaccards: list[float] = []
    test_matches = 0
    command_checks = 0
    command_matches = 0
    record_reports: list[dict] = []

    for record in generation_records:
        reference_path = _reference_path_for_record(record)
        generated_path = output_path_from_record(record, artifact_format)
        if reference_path is None or generated_path is None:
            continue
        reference = _xml_features(reference_path)
        generated = (
            _udt_features(generated_path)
            if normalize_artifact_format(artifact_format) == ARTIFACT_FORMAT_UDT_YAML
            else _xml_features(generated_path)
        )
        if reference is None or generated is None:
            continue

        compared += 1
        input_error = abs(generated["input_count"] - reference["input_count"])
        output_error = abs(generated["output_count"] - reference["output_count"])
        input_jaccard = _jaccard(generated["input_datatypes"], reference["input_datatypes"])
        output_jaccard = _jaccard(generated["output_datatypes"], reference["output_datatypes"])
        requirement_jaccard = _jaccard(
            generated["requirement_packages"], reference["requirement_packages"]
        )
        tests_match = generated["has_tests"] == reference["has_tests"]
        input_errors.append(input_error)
        output_errors.append(output_error)
        input_jaccards.append(input_jaccard)
        output_jaccards.append(output_jaccard)
        requirement_jaccards.append(requirement_jaccard)
        if tests_match:
            test_matches += 1

        primary_command = str(record.get("primary_command", "")).strip().lower()
        primary_command_present = False
        if primary_command:
            command_checks += 1
            if primary_command in generated["command_text"]:
                command_matches += 1
                primary_command_present = True

        record_reports.append(
            {
                "tool_name": str(record.get("tool_name", "")),
                "corpus_index": int(record.get("corpus_index", -1)),
                "package_id": str(record.get("package_id", "")),
                "tool_id": str(record.get("tool_id", "")),
                "artifact_format": artifact_format,
                "output_path": str(generated_path),
                "output_xml_path": str(generated_path)
                if artifact_format == ARTIFACT_FORMAT_XML
                else "",
                "output_udt_yaml_path": str(generated_path)
                if artifact_format == ARTIFACT_FORMAT_UDT_YAML
                else "",
                "reference_xml_path": str(reference_path),
                "input_count_reference": int(reference["input_count"]),
                "input_count_generated": int(generated["input_count"]),
                "input_count_abs_error": int(input_error),
                "output_count_reference": int(reference["output_count"]),
                "output_count_generated": int(generated["output_count"]),
                "output_count_abs_error": int(output_error),
                "input_datatypes_reference": sorted(reference["input_datatypes"]),
                "input_datatypes_generated": sorted(generated["input_datatypes"]),
                "input_datatype_jaccard": round(input_jaccard, 4),
                "output_datatypes_reference": sorted(reference["output_datatypes"]),
                "output_datatypes_generated": sorted(generated["output_datatypes"]),
                "output_datatype_jaccard": round(output_jaccard, 4),
                "requirement_packages_reference": sorted(reference["requirement_packages"]),
                "requirement_packages_generated": sorted(generated["requirement_packages"]),
                "requirement_package_jaccard": round(requirement_jaccard, 4),
                "has_tests_reference": bool(reference["has_tests"]),
                "has_tests_generated": bool(generated["has_tests"]),
                "test_presence_matches": tests_match,
                "primary_command": primary_command,
                "primary_command_present": primary_command_present,
            }
        )

    return {
        "compared_records": compared,
        "avg_input_count_abs_error": _mean(input_errors),
        "avg_output_count_abs_error": _mean(output_errors),
        "input_datatype_jaccard_mean": _mean(input_jaccards),
        "output_datatype_jaccard_mean": _mean(output_jaccards),
        "requirement_package_jaccard_mean": _mean(requirement_jaccards),
        "test_presence_match_rate": _rate(test_matches, compared),
        "primary_command_checks": command_checks,
        "primary_command_presence_rate": _rate(command_matches, command_checks),
        "records": record_reports,
    }


def _benchmark_quality(
    *,
    attempted: int,
    generation_records: list[dict],
    failures: list[dict],
    evaluation_report: dict,
    progress: dict,
    startup: dict | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
) -> dict:
    artifact_format = normalize_artifact_format(artifact_format)
    all_records = [*generation_records, *failures]
    xml_well_formed = sum(
        1
        for record in all_records
        if isinstance(record.get("validation"), dict)
        and record["validation"].get("xml_well_formed") is True
    )
    tool_root = sum(
        1
        for record in all_records
        if isinstance(record.get("validation"), dict)
        and record["validation"].get("root_is_tool") is True
    )
    yaml_well_formed = sum(
        1
        for record in all_records
        if isinstance(record.get("validation"), dict)
        and record["validation"].get("yaml_well_formed") is True
    )
    udt_schema_valid = sum(
        1
        for record in all_records
        if isinstance(record.get("validation"), dict)
        and record["validation"].get("schema_valid") is True
    )
    user_tool_root = sum(
        1
        for record in all_records
        if isinstance(record.get("validation"), dict)
        and record["validation"].get("root_is_user_tool") is True
    )
    artifact_valid = sum(
        1
        for record in all_records
        if isinstance(record.get("validation"), dict)
        and _artifact_validation_passed(record["validation"], artifact_format)
    )
    repair_attempts = sum(1 for record in all_records if record.get("repair_attempted") is True)
    repair_successes = sum(
        1 for record in generation_records if record.get("repair_attempted") is True
    )
    truncation_failures = sum(
        1 for record in failures if record.get("truncation_suspected") is True
    )
    structural_scores = [
        float(report.get("structural", {}).get("structural_score", 0.0))
        for report in evaluation_report.get("wrapper_reports", [])
        if isinstance(report, dict)
    ]
    elapsed_seconds = float(progress.get("elapsed_seconds") or 0.0)
    succeeded = len(generation_records)

    return {
        "throughput": {
            "elapsed_seconds": round(elapsed_seconds, 4),
            "wrappers_per_minute": round((attempted / elapsed_seconds) * 60, 4)
            if elapsed_seconds > 0
            else 0.0,
            "seconds_per_attempted_wrapper": round(elapsed_seconds / attempted, 4)
            if attempted
            else 0.0,
        },
        "validity": {
            "artifact_format": artifact_format,
            "success_rate": _rate(succeeded, attempted),
            "artifact_valid_rate": _rate(artifact_valid, attempted),
            "xml_well_formed_rate": _rate(xml_well_formed, attempted),
            "tool_root_rate": _rate(tool_root, attempted),
            "yaml_well_formed_rate": _rate(yaml_well_formed, attempted),
            "udt_schema_valid_rate": _rate(udt_schema_valid, attempted),
            "user_tool_root_rate": _rate(user_tool_root, attempted),
        },
        "repair": {
            "repair_attempt_rate": _rate(repair_attempts, attempted),
            "repair_success_rate": _rate(repair_successes, repair_attempts),
            "truncation_failure_rate": _rate(truncation_failures, attempted),
        },
        "structural": {
            "mean_structural_score_successes": _mean(structural_scores),
            "effective_mean_structural_score_all_attempted": round(
                sum(structural_scores) / attempted, 4
            )
            if attempted
            else 0.0,
        },
        "reference_fidelity": _reference_fidelity(
            generation_records,
            artifact_format=artifact_format,
        ),
        "startup": startup or {},
    }


def _read_json_file(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _benchmark_failure(
    *,
    tool_name: str,
    output_xml: Path,
    provider: str,
    model_variant: str,
    error: Exception,
    artifact_format: str = ARTIFACT_FORMAT_XML,
) -> dict:
    artifact_format = normalize_artifact_format(artifact_format)
    message = str(error).strip()
    error_repr = repr(error)
    return {
        "tool_name": tool_name,
        "artifact_format": artifact_format,
        "output_path": str(output_xml),
        "output_xml_path": str(output_xml) if artifact_format == ARTIFACT_FORMAT_XML else "",
        "output_udt_yaml_path": str(output_xml)
        if artifact_format == ARTIFACT_FORMAT_UDT_YAML
        else "",
        "provider": provider,
        "model_variant": model_variant,
        "error": message or error_repr,
        "error_type": type(error).__name__,
        "error_repr": error_repr,
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__, limit=8)
        ).strip(),
    }


class InvalidGeneratedXmlError(RuntimeError):
    """Raised when a generated wrapper fails benchmark XML validation."""


def _xml_validation_failure(
    *,
    generated: dict,
    tool_name: str,
    output_xml: Path,
    provider: str,
    model_variant: str,
) -> dict:
    validation = generated.get("validation")
    notes = validation.get("notes", []) if isinstance(validation, dict) else []
    detail = "; ".join(str(note) for note in notes if str(note).strip())
    if not isinstance(validation, dict):
        message = "Generated XML lacks validation metadata."
    elif validation.get("xml_well_formed") is not True:
        message = "Generated XML is not well formed."
    elif validation.get("root_is_tool") is not True:
        root_tag = str(validation.get("root_tag") or "unknown")
        message = f"Generated XML root is not <tool>; found <{root_tag}>."
    elif _generation_diagnostics_failed(validation):
        message = "Generated XML appears degenerate."
    else:
        message = "Generated XML failed benchmark validation."
    if detail:
        message = f"{message} {detail}"
    failure = _benchmark_failure(
        tool_name=tool_name,
        output_xml=output_xml,
        provider=provider,
        model_variant=model_variant,
        error=InvalidGeneratedXmlError(message),
        artifact_format=ARTIFACT_FORMAT_XML,
    )
    failure["validation"] = validation if isinstance(validation, dict) else {}
    failure["attempt_count"] = int(generated.get("attempt_count") or 1)
    failure["repair_attempted"] = bool(generated.get("repair_attempted", False))
    failure["truncation_suspected"] = _validation_suggests_truncation(validation)
    return failure


def _artifact_validation_failure(
    *,
    generated: dict,
    tool_name: str,
    output_artifact: Path,
    provider: str,
    model_variant: str,
    artifact_format: str,
) -> dict:
    artifact_format = normalize_artifact_format(artifact_format)
    if artifact_format == ARTIFACT_FORMAT_XML:
        return _xml_validation_failure(
            generated=generated,
            tool_name=tool_name,
            output_xml=output_artifact,
            provider=provider,
            model_variant=model_variant,
        )
    validation = generated.get("validation")
    notes = validation.get("notes", []) if isinstance(validation, dict) else []
    detail = "; ".join(str(note) for note in notes if str(note).strip())
    if not isinstance(validation, dict):
        message = "Generated UDT YAML lacks validation metadata."
    elif validation.get("yaml_well_formed") is not True:
        message = "Generated UDT YAML is not well formed."
    elif validation.get("root_is_user_tool") is not True:
        root_class = str(validation.get("root_class") or "unknown")
        message = f"Generated UDT YAML root class is not GalaxyUserTool; found {root_class}."
    elif validation.get("schema_valid") is not True:
        message = "Generated UDT YAML failed schema validation."
    else:
        message = "Generated UDT YAML failed benchmark validation."
    if detail:
        message = f"{message} {detail}"
    failure = _benchmark_failure(
        tool_name=tool_name,
        output_xml=output_artifact,
        provider=provider,
        model_variant=model_variant,
        error=InvalidGeneratedXmlError(message),
        artifact_format=artifact_format,
    )
    failure["validation"] = validation if isinstance(validation, dict) else {}
    failure["attempt_count"] = int(generated.get("attempt_count") or 1)
    failure["repair_attempted"] = bool(generated.get("repair_attempted", False))
    failure["truncation_suspected"] = False
    return failure


def _artifact_validation_passed(validation: object, artifact_format: str) -> bool:
    artifact_format = normalize_artifact_format(artifact_format)
    if artifact_format == ARTIFACT_FORMAT_UDT_YAML:
        return (
            isinstance(validation, dict)
            and validation.get("yaml_well_formed") is True
            and validation.get("schema_valid") is True
            and validation.get("root_is_user_tool") is True
        )
    return _xml_validation_passed(validation)


def _xml_validation_passed(validation: object) -> bool:
    return (
        isinstance(validation, dict)
        and validation.get("xml_well_formed") is True
        and validation.get("root_is_tool") is True
        and not _generation_diagnostics_failed(validation)
    )


def _generation_diagnostics_failed(validation: object) -> bool:
    if not isinstance(validation, dict):
        return False
    diagnostics = validation.get("generation_diagnostics")
    return isinstance(diagnostics, dict) and diagnostics.get("has_problems") is True


def _validation_suggests_truncation(validation: object) -> bool:
    if not isinstance(validation, dict):
        return False
    diagnostics = validation.get("generation_diagnostics")
    if isinstance(diagnostics, dict) and any(
        diagnostics.get(key) is True
        for key in ("missing_closing_tool", "ends_mid_tag", "unclosed_cdata")
    ):
        return True
    notes = validation.get("notes", [])
    if not isinstance(notes, list):
        return False
    note_text = " ".join(str(note).lower() for note in notes)
    return any(
        marker in note_text
        for marker in (
            "unclosed token",
            "unclosed cdata",
            "missing closing",
            "does not contain a closing </tool>",
            "end mid-tag",
        )
    )


def _failure_suggests_truncation(failure: dict) -> bool:
    return bool(failure.get("truncation_suspected")) or _validation_suggests_truncation(
        failure.get("validation")
    )


def _repair_prompt_help_chars(max_prompt_help_chars: int, *, failure: dict | None = None) -> int:
    max_chars = min(max(0, int(max_prompt_help_chars)), REPAIR_MAX_PROMPT_HELP_CHARS)
    if failure is not None and _failure_suggests_truncation(failure):
        return min(max_chars, TRUNCATION_REPAIR_MAX_PROMPT_HELP_CHARS)
    return max_chars


def _repair_context(*, tool_name: str, failure: dict) -> str:
    lines = [
        "The previous generated wrapper failed benchmark validation.",
        f"Tool name: {tool_name}",
        f"Failure: {failure.get('error', 'invalid generated XML')}",
        "Return a shorter complete Galaxy <tool> XML document.",
        "Do not continue or complete a long select option list.",
        "Replace uncertain or long option lists with text parameters.",
    ]
    if _failure_suggests_truncation(failure):
        lines.extend(
            [
                "The previous output appears truncated before XML closure.",
                "Do not preserve the full interface; produce a minimal valid wrapper.",
                "Use a compact wrapper skeleton instead of an exhaustive wrapper.",
                "Use exactly one <command>, one <inputs>, one <outputs>, one <tests>, and one <help> section.",
                "Keep the complete XML under 80 lines.",
                "Keep <command><![CDATA[...]]></command> under 20 lines.",
                "Keep <help><![CDATA[...]]></help> under 10 lines.",
                "Include at most six input parameters and at most two outputs.",
                "Preserve required/core CLI options; skip advanced optional options.",
                "Include at most one minimal <test>.",
                "Inside a test output, include at most three <has_text> assertions.",
                "Do not copy long examples, long help blocks, or long output descriptions.",
                "Do not repeat identical XML lines or Cheetah assignments.",
                "Do not create long CDATA sections.",
                "Close every tag and every CDATA section.",
            ]
        )
    lines.append("Stop immediately after the closing </tool> tag.")
    return "\n".join(lines)


def run_benchmark_generation(
    paths: WorkspacePaths,
    corpus_jsonl: Path,
    wrappers_dir: Path,
    generation_records_path: Path,
    evaluation_report_path: Path,
    provider: str,
    model_variant: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_workers: int,
    limit: int | None,
    xsd_path: Path | None,
    run_planemo: bool,
    allow_stub_local: bool = False,
    status_sink: Callable[[dict], None] | None = None,
    repair_invalid_xml: bool = True,
    allow_compact_fallback: bool = False,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    local_gpu_topology: str = "per-process",
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    resume_existing: bool = False,
    checkpoint_records_path: Path | None = None,
    run_planemo_tests: bool = False,
    planemo_test_options: PlanemoTestOptions | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
) -> BenchmarkSummary:
    artifact_format = normalize_artifact_format(artifact_format)
    source_context_settings = (source_context_settings or SourceContextSettings()).normalized()
    if local_gpu_topology not in LOCAL_GPU_TOPOLOGIES:
        raise ValueError(f"Unsupported local GPU topology: {local_gpu_topology}")
    if local_offload_policy not in LOCAL_OFFLOAD_POLICIES:
        raise ValueError(f"Unsupported local offload policy: {local_offload_policy}")
    records = _with_corpus_indices(_load_corpus_records(corpus_jsonl, limit=limit))
    wrappers_dir.mkdir(parents=True, exist_ok=True)
    generation_records_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_records_path = checkpoint_records_path or _default_checkpoint_path(
        generation_records_path
    )

    generated_wrappers: list[Path] = []
    checkpointed_records = (
        _load_successful_checkpoint_records(
            checkpoint_records_path,
            artifact_format=artifact_format,
        )
        if resume_existing
        else []
    )
    checkpointed_by_index = {
        int(record.get("corpus_index")): record for record in checkpointed_records
    }
    generation_records: list[dict] = list(checkpointed_records)
    generated_wrappers.extend(
        path
        for record in checkpointed_records
        if (path := output_path_from_record(record, artifact_format)) is not None
    )
    if not resume_existing and checkpoint_records_path.exists():
        checkpoint_records_path.unlink()
    records_to_run = [
        record
        for record in records
        if int(record.get("_benchmark_corpus_index", -1)) not in checkpointed_by_index
    ]
    checkpoint_lock = threading.Lock()
    failures: list[dict] = []
    started_at = utc_now_iso()
    run_started_perf = time.perf_counter()
    startup: dict = {
        "started_at": started_at,
        "artifact_format": artifact_format,
        "source_context": source_context_settings.to_dict(),
        "local_gpu_topology": local_gpu_topology,
        "local_offload_policy": local_offload_policy,
        "local_gpu_memory_reserve_gib": local_gpu_memory_reserve_gib,
        "resume_existing": resume_existing,
        "checkpoint_records_path": str(checkpoint_records_path),
        "checkpoint_records_loaded": len(checkpointed_records),
        "skipped_existing": len(checkpointed_records),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "allow_compact_fallback": allow_compact_fallback,
    }
    effective_max_workers = max_workers
    provider_instance = None
    if provider == "local" and records_to_run:
        effective_max_workers = 1
        provider_instance = _get_provider(
            provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            paths=paths,
            allow_stub_local=allow_stub_local,
            local_offload_policy=local_offload_policy,
            local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        )
        model_load_started_at = utc_now_iso()
        model_load_started_perf = time.perf_counter()
        _status_emit(
            status_sink,
            {
                "status": "benchmark-model-load-started",
                "provider": provider,
                "model_variant": model_variant,
            },
        )
        ensure_ready = getattr(provider_instance, "ensure_ready", None)
        if ensure_ready is not None:
            ensure_ready(model_variant)
        load_info = {}
        ensure_loaded = getattr(provider_instance, "ensure_loaded", None)
        if ensure_loaded is not None:
            loaded = ensure_loaded(model_variant)
            if isinstance(loaded, dict):
                load_info = loaded
        model_ready_at = utc_now_iso()
        startup.update(
            {
                "model_load_started_at": model_load_started_at,
                "model_ready_at": model_ready_at,
                "model_load_seconds": _elapsed_since(model_load_started_perf),
                "startup_seconds": _elapsed_since(run_started_perf),
                **load_info,
            }
        )
        _status_emit(
            status_sink,
            {
                "status": "benchmark-model-ready",
                "provider": provider,
                "model_variant": model_variant,
                "startup": dict(startup),
            },
        )

    def _generate_one(
        *,
        tool_name: str,
        help_text: str,
        output_artifact: Path,
        attempt_count: int,
        repair_attempted: bool = False,
        repair_reason: str = "",
        repair_context: str = "",
        prompt_help_chars: int | None = None,
        metadata_hints: dict | None = None,
        source_code: str = "",
        source_context_summary: dict | None = None,
    ) -> dict:
        generation = generate_wrapper_from_content(
            paths=paths,
            tool_name=tool_name,
            help_text=help_text,
            source_code=source_code,
            output_path=output_artifact,
            provider_name=provider,
            model_variant=model_variant,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            allow_stub_local=allow_stub_local,
            provider_instance=provider_instance,
            max_prompt_help_chars=max_prompt_help_chars
            if prompt_help_chars is None
            else prompt_help_chars,
            repair_context=repair_context,
            attempt_count=attempt_count,
            repair_attempted=repair_attempted,
            repair_reason=repair_reason,
            metadata_hints=metadata_hints,
            artifact_format=artifact_format,
            source_context_summary=source_context_summary,
        )
        return json.loads(generation.to_json())

    def _one(record: dict) -> tuple[dict | None, dict | None, Path | None]:
        tool_name = record.get("tool_name", "unknown_tool")
        help_text = record.get("help_text", "")
        output_artifact = wrappers_dir / _safe_wrapper_filename(str(tool_name), artifact_format)
        source_context = build_source_context_from_record(record, source_context_settings)
        source_context_summary = source_context.to_dict()
        record_started_at = utc_now_iso()
        output_keys = {
            "output_path": str(output_artifact),
            "output_xml_path": str(output_artifact)
            if artifact_format == ARTIFACT_FORMAT_XML
            else "",
            "output_udt_yaml_path": str(output_artifact)
            if artifact_format == ARTIFACT_FORMAT_UDT_YAML
            else "",
        }
        _status_emit(
            status_sink,
            {
                "status": "benchmark-record-started",
                "provider": provider,
                "model_variant": model_variant,
                "artifact_format": artifact_format,
                "corpus_index": int(record.get("_benchmark_corpus_index", -1)),
                "tool_name": str(tool_name),
                **output_keys,
                "record_started_at": record_started_at,
            },
        )
        try:
            generated = _generate_one(
                tool_name=str(tool_name),
                help_text=str(help_text),
                output_artifact=output_artifact,
                attempt_count=1,
                metadata_hints=_prompt_metadata_hints(record),
                source_code=source_context.text,
                source_context_summary=source_context_summary,
            )
            generated = _annotate_generation_record(generated, record)
            validation = generated.get("validation")
            if _artifact_validation_passed(validation, artifact_format):
                return generated, None, output_artifact

            initial_failure = _artifact_validation_failure(
                generated=generated,
                tool_name=str(tool_name),
                output_artifact=output_artifact,
                provider=provider,
                model_variant=model_variant,
                artifact_format=artifact_format,
            )
            initial_failure = _annotate_failure(initial_failure, record)
            if repair_invalid_xml and artifact_format == ARTIFACT_FORMAT_XML:
                repair_reason = str(initial_failure.get("error", "")).strip()
                repaired = _generate_one(
                    tool_name=str(tool_name),
                    help_text=str(help_text),
                    output_artifact=output_artifact,
                    attempt_count=2,
                    repair_attempted=True,
                    repair_reason=repair_reason,
                    repair_context=_repair_context(
                        tool_name=str(tool_name),
                        failure=initial_failure,
                    ),
                    prompt_help_chars=_repair_prompt_help_chars(
                        max_prompt_help_chars,
                        failure=initial_failure,
                    ),
                    metadata_hints=_prompt_metadata_hints(record),
                    source_code=source_context.text,
                    source_context_summary=source_context_summary,
                )
                repaired = _annotate_generation_record(repaired, record)
                if _artifact_validation_passed(repaired.get("validation"), artifact_format):
                    return repaired, None, output_artifact
                repaired_failure = _xml_validation_failure(
                    generated=repaired,
                    tool_name=str(tool_name),
                    output_xml=output_artifact,
                    provider=provider,
                    model_variant=model_variant,
                )
                repaired_failure = _annotate_failure(repaired_failure, record)
                repaired_failure["repair_attempted"] = True
                repaired_failure["repair_reason"] = repair_reason
                repaired_failure["initial_failure"] = {
                    "error": initial_failure.get("error", ""),
                    "error_type": initial_failure.get("error_type", ""),
                    "validation": initial_failure.get("validation", {}),
                }
                if allow_compact_fallback and _failure_suggests_truncation(repaired_failure):
                    fallback = _compact_fallback_generation_record(
                        paths=paths,
                        tool_name=str(tool_name),
                        record=record,
                        output_xml=output_artifact,
                        provider=provider,
                        model_variant=model_variant,
                        repair_reason=repair_reason,
                        initial_failure=initial_failure,
                        repaired_failure=repaired_failure,
                    )
                    fallback = _annotate_generation_record(fallback, record)
                    if _xml_validation_passed(fallback.get("validation")):
                        return fallback, None, output_artifact
                    repaired_failure["compact_fallback_attempted"] = True
                    repaired_failure["compact_fallback_validation"] = fallback.get("validation", {})
                elif _failure_suggests_truncation(repaired_failure):
                    repaired_failure["compact_fallback_skipped"] = True
                    repaired_failure["compact_fallback_skip_reason"] = (
                        "Compact fallback is disabled; malformed/truncated generations "
                        "are reported as benchmark failures."
                    )
                return None, repaired_failure, None

            return None, initial_failure, None
        except Exception as error:
            return (
                None,
                _benchmark_failure(
                    tool_name=str(tool_name),
                    output_xml=output_artifact,
                    provider=provider,
                    model_variant=model_variant,
                    error=error,
                    artifact_format=artifact_format,
                )
                | _record_metadata(record),
                None,
            )

    if records_to_run:
        startup["first_generation_started_at"] = utc_now_iso()
        startup["time_to_first_generation_seconds"] = _elapsed_since(run_started_perf)
        _status_emit(
            status_sink,
            {
                "status": "benchmark-first-generation-started",
                "provider": provider,
                "model_variant": model_variant,
                "total": len(records),
                "remaining": len(records_to_run),
                "startup": dict(startup),
            },
        )

    with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
        futures = [executor.submit(_one, record) for record in records_to_run]
        total_units = len(records)
        completed_base = len(checkpointed_records)
        for completed_delta, future in enumerate(as_completed(futures), start=1):
            generated, failed, wrapper_path = future.result()
            if generated is not None:
                generation_records.append(generated)
                _append_checkpoint_record(checkpoint_records_path, generated, checkpoint_lock)
            if failed is not None:
                failures.append(failed)
            if wrapper_path is not None:
                generated_wrappers.append(wrapper_path)
            completed_units = completed_base + completed_delta
            snapshot = make_progress_snapshot(
                started_at=started_at,
                completed_units=completed_units,
                total_units=total_units,
            )
            payload = {
                "status": "benchmark-progress",
                "completed": completed_units,
                "total": total_units,
                "progress": snapshot.to_dict(),
            }
            if status_sink:
                status_sink(payload)
            else:
                emit_status(payload)

    generation_records.sort(key=lambda item: int(item.get("corpus_index", -1)))
    _atomic_write_text(generation_records_path, json.dumps(generation_records, indent=2))
    evaluation_summary = evaluate_wrapper_paths(
        wrapper_paths=generated_wrappers,
        output_report=evaluation_report_path,
        xsd_path=xsd_path,
        run_planemo=run_planemo,
        run_planemo_tests=run_planemo_tests,
        planemo_test_options=planemo_test_options,
        artifact_format=artifact_format,
    ).to_dict()

    final_progress = make_progress_snapshot(
        started_at=started_at,
        completed_units=len(records),
        total_units=len(records),
    ).to_dict()

    summary = BenchmarkSummary(
        created_at=utc_now_iso(),
        corpus_path=str(corpus_jsonl),
        provider=provider,
        model_variant=model_variant,
        attempted=len(records),
        succeeded=len(generation_records),
        failed=len(failures),
        wrappers_dir=str(wrappers_dir),
        generation_records_path=str(generation_records_path),
        evaluation_report_path=str(evaluation_report_path),
        artifact_format=artifact_format,
        progress=final_progress,
        startup=startup,
        failures=failures,
        quality=_benchmark_quality(
            attempted=len(records),
            generation_records=generation_records,
            failures=failures,
            evaluation_report=evaluation_summary,
            progress=final_progress,
            startup=startup,
            artifact_format=artifact_format,
        ),
    )
    return summary


def resolve_benchmark_parallelism(
    *,
    provider: str,
    num_processes: int,
    gpu_devices: str,
    total_records: int | None = None,
    min_items_per_process: int = DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS,
    local_gpu_topology: str = "per-process",
) -> tuple[int, list[str]]:
    if local_gpu_topology not in LOCAL_GPU_TOPOLOGIES:
        raise ValueError(f"Unsupported local GPU topology: {local_gpu_topology}")
    devices = [item.strip() for item in str(gpu_devices or "").split(",") if item.strip()]
    if len(set(devices)) != len(devices):
        raise ValueError("--gpu-devices contains duplicate entries.")

    requested_process_count = int(num_processes or 0)
    if requested_process_count < 0:
        raise ValueError("--num-processes must be at least 0.")

    visible_devices = [
        item.strip() for item in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()
    ]

    if local_gpu_topology == "model-parallel":
        if provider != "local":
            raise ValueError(
                "--local-gpu-topology model-parallel is only supported for local provider."
            )
        return 1, devices or visible_devices

    if devices:
        if requested_process_count > len(devices):
            raise ValueError("--num-processes cannot exceed the number of --gpu-devices.")
        process_count = requested_process_count if requested_process_count else len(devices)
    elif requested_process_count > 0:
        process_count = requested_process_count
        if len(visible_devices) >= process_count:
            devices = visible_devices[:process_count]
    elif provider == "local" and len(visible_devices) > 1:
        process_count = len(visible_devices)
        devices = visible_devices
    else:
        process_count = 1

    if requested_process_count == 0:
        process_count = _cap_processes_for_workload(
            process_count=process_count,
            total_records=total_records,
            min_items_per_process=min_items_per_process,
        )
        if devices:
            devices = devices[:process_count]
    elif devices:
        devices = devices[:process_count]

    if process_count > 1 and provider != "local":
        raise ValueError("Native benchmark multiprocessing is only supported for --provider local.")
    return process_count, devices


def _cap_processes_for_workload(
    *,
    process_count: int,
    total_records: int | None,
    min_items_per_process: int,
) -> int:
    if total_records is None or total_records <= 0 or min_items_per_process <= 0:
        return process_count
    required = max(1, (total_records + min_items_per_process - 1) // min_items_per_process)
    return max(1, min(process_count, required))


def _shard_records(records: list[dict], shard_count: int) -> list[list[dict]]:
    shards = [[] for _ in range(shard_count)]
    for index, record in enumerate(records):
        shards[index % shard_count].append(record)
    return shards


def _shard_root(generation_records_path: Path) -> Path:
    return generation_records_path.parent / f"{generation_records_path.stem}.shards"


def _build_shard_command(
    *,
    paths: WorkspacePaths,
    shard: BenchmarkShard,
    provider: str,
    model_variant: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_workers: int,
    xsd_path: Path | None,
    run_planemo: bool,
    allow_stub_local: bool,
    repair_invalid_xml: bool,
    allow_compact_fallback: bool,
    max_prompt_help_chars: int,
    local_gpu_topology: str,
    local_offload_policy: str,
    local_gpu_memory_reserve_gib: float,
    resume_existing: bool,
    checkpoint_records_path: Path,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "galaxy_toolsmith.cli.main",
        "--repo-root",
        str(paths.repo_root),
        "benchmark-generate",
        "--benchmark-shard-worker",
        "--corpus-jsonl",
        str(shard.corpus_jsonl),
        "--wrappers-dir",
        str(shard.wrappers_dir),
        "--generation-records",
        str(shard.generation_records_path),
        "--evaluation-report",
        str(shard.evaluation_report_path),
        "--benchmark-summary",
        str(shard.benchmark_summary_path),
        "--provider",
        provider,
        "--model-variant",
        model_variant,
        "--artifact-format",
        artifact_format.replace("_", "-"),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--max-workers",
        str(max_workers),
        "--max-prompt-help-chars",
        str(max_prompt_help_chars),
        "--local-gpu-topology",
        local_gpu_topology,
        "--local-offload-policy",
        local_offload_policy,
        "--local-gpu-memory-reserve-gib",
        str(local_gpu_memory_reserve_gib),
        "--checkpoint-records",
        str(checkpoint_records_path),
        "--status-log",
        str(shard.status_log_path),
    ]
    if model:
        command.extend(["--model", model])
    if xsd_path is not None:
        command.extend(["--xsd", str(xsd_path)])
    if run_planemo:
        command.append("--run-planemo")
    if allow_stub_local:
        command.append("--allow-stub-local")
    if allow_compact_fallback:
        command.append("--allow-compact-fallback")
    if resume_existing:
        command.append("--resume-existing")
    command.append("--repair-invalid-xml" if repair_invalid_xml else "--no-repair-invalid-xml")
    source_context_settings = (source_context_settings or SourceContextSettings()).normalized()
    if source_context_settings.mode != "none":
        command.extend(["--source-context-mode", source_context_settings.mode])
        command.extend(["--source-context-max-chars", str(source_context_settings.max_chars)])
        command.extend(["--source-context-max-files", str(source_context_settings.max_files)])
        if source_context_settings.source_root is not None:
            command.extend(["--source-root", str(source_context_settings.source_root)])
    return command


def _status_emit(status_sink: Callable[[dict], None] | None, payload: dict) -> None:
    if status_sink:
        status_sink(payload)
    else:
        emit_status(payload)


def _tail_text(path: Path, limit: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _terminate_process(process: subprocess.Popen[str], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def _run_benchmark_shard_process(
    *,
    shard: BenchmarkShard,
    command: list[str],
    env: dict[str, str],
    status_sink: Callable[[dict], None] | None,
    progress_by_shard: dict[int, int],
    total_units: int,
    started_at: str,
    lock: threading.Lock,
    record_timeout_seconds: float = 0.0,
) -> None:
    shard.stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    shard.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    _status_emit(
        status_sink,
        {
            "status": "benchmark-shard-started",
            "shard_index": shard.index,
            "gpu_device": shard.gpu_device,
            "command": command,
        },
    )
    with (
        shard.stdout_log_path.open("w", encoding="utf-8") as stdout_log,
        shard.stderr_log_path.open("w", encoding="utf-8") as stderr_log,
    ):
        process = subprocess.Popen(
            command,
            cwd=str(env.get("GTSM_REPO_ROOT") or Path.cwd()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            text=True,
        )
        assert process.stdout is not None
        line_queue: queue.Queue[dict | None] = queue.Queue()

        def _read_stdout() -> None:
            for line in process.stdout:
                stdout_log.write(line)
                stdout_log.flush()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    line_queue.put(payload)
            line_queue.put(None)

        reader = threading.Thread(target=_read_stdout, daemon=True)
        reader.start()
        active_record: dict | None = None
        active_record_started_perf = 0.0
        stdout_closed = False
        while not stdout_closed or process.poll() is None:
            try:
                payload = line_queue.get(timeout=0.5)
            except queue.Empty:
                payload = None
            if payload is None:
                if not line_queue.empty():
                    continue
                stdout_closed = process.poll() is not None
                if (
                    record_timeout_seconds > 0
                    and active_record is not None
                    and time.perf_counter() - active_record_started_perf > record_timeout_seconds
                ):
                    timeout_payload = {
                        "status": "benchmark-record-timeout",
                        "shard_index": shard.index,
                        "gpu_device": shard.gpu_device,
                        "record_timeout_seconds": record_timeout_seconds,
                        **active_record,
                    }
                    _status_emit(status_sink, timeout_payload)
                    _terminate_process(process)
                    raise RuntimeError(
                        f"Benchmark shard {shard.index} exceeded "
                        f"record timeout {record_timeout_seconds}s for "
                        f"{active_record.get('tool_name', 'unknown tool')}"
                    )
                continue
            if payload.get("status") != "benchmark-progress":
                if payload.get("status"):
                    forwarded = dict(payload)
                    forwarded.setdefault("shard_index", shard.index)
                    forwarded.setdefault("gpu_device", shard.gpu_device)
                    _status_emit(status_sink, forwarded)
                    if payload.get("status") == "benchmark-record-started":
                        active_record = {
                            "corpus_index": payload.get("corpus_index", -1),
                            "tool_name": payload.get("tool_name", ""),
                            "output_path": payload.get("output_path", ""),
                            "output_xml_path": payload.get("output_xml_path", ""),
                            "output_udt_yaml_path": payload.get("output_udt_yaml_path", ""),
                            "record_started_at": payload.get("record_started_at", ""),
                        }
                        active_record_started_perf = time.perf_counter()
                continue
            with lock:
                progress_by_shard[shard.index] = int(payload.get("completed") or 0)
                active_record = None
                completed_units = sum(progress_by_shard.values())
                progress = make_progress_snapshot(
                    started_at=started_at,
                    completed_units=completed_units,
                    total_units=total_units,
                ).to_dict()
                _status_emit(
                    status_sink,
                    {
                        "status": "benchmark-progress",
                        "completed": completed_units,
                        "total": total_units,
                        "progress": progress,
                    },
                )
        returncode = process.wait()
        reader.join(timeout=2.0)
    if returncode != 0:
        raise RuntimeError(
            f"Benchmark shard {shard.index} failed with exit code {returncode}: "
            f"{_tail_text(shard.stderr_log_path)}"
        )
    _status_emit(
        status_sink,
        {
            "status": "benchmark-shard-completed",
            "shard_index": shard.index,
            "gpu_device": shard.gpu_device,
            "summary_path": str(shard.benchmark_summary_path),
        },
    )


def run_benchmark_generation_sharded(
    *,
    paths: WorkspacePaths,
    corpus_jsonl: Path,
    wrappers_dir: Path,
    generation_records_path: Path,
    evaluation_report_path: Path,
    benchmark_summary_path: Path,
    provider: str,
    model_variant: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_workers: int,
    limit: int | None,
    xsd_path: Path | None,
    run_planemo: bool,
    num_processes: int,
    gpu_devices: str,
    allow_stub_local: bool = False,
    status_sink: Callable[[dict], None] | None = None,
    repair_invalid_xml: bool = True,
    allow_compact_fallback: bool = False,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    min_items_per_process: int = DEFAULT_BENCHMARK_MIN_ITEMS_PER_PROCESS,
    startup_stagger_seconds: float = 0.0,
    local_gpu_topology: str = "per-process",
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    resume_existing: bool = False,
    checkpoint_records_path: Path | None = None,
    record_timeout_seconds: float = 0.0,
    run_planemo_tests: bool = False,
    planemo_test_options: PlanemoTestOptions | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
) -> BenchmarkSummary:
    artifact_format = normalize_artifact_format(artifact_format)
    source_context_settings = (source_context_settings or SourceContextSettings()).normalized()
    if local_gpu_topology not in LOCAL_GPU_TOPOLOGIES:
        raise ValueError(f"Unsupported local GPU topology: {local_gpu_topology}")
    if local_offload_policy not in LOCAL_OFFLOAD_POLICIES:
        raise ValueError(f"Unsupported local offload policy: {local_offload_policy}")
    checkpoint_records_path = checkpoint_records_path or _default_checkpoint_path(
        generation_records_path
    )
    process_count, devices = resolve_benchmark_parallelism(
        provider=provider,
        num_processes=num_processes,
        gpu_devices=gpu_devices,
        total_records=limit,
        min_items_per_process=min_items_per_process,
        local_gpu_topology=local_gpu_topology,
    )
    force_child_process = local_gpu_topology == "model-parallel" or record_timeout_seconds > 0
    if process_count <= 1 and not force_child_process:
        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ",".join(devices) if local_gpu_topology == "model-parallel" else devices[0]
            )
        try:
            return run_benchmark_generation(
                paths=paths,
                corpus_jsonl=corpus_jsonl,
                wrappers_dir=wrappers_dir,
                generation_records_path=generation_records_path,
                evaluation_report_path=evaluation_report_path,
                provider=provider,
                model_variant=model_variant,
                model=model,
                artifact_format=artifact_format,
                temperature=temperature,
                max_tokens=max_tokens,
                max_workers=max_workers,
                limit=limit,
                xsd_path=xsd_path,
                run_planemo=run_planemo,
                allow_stub_local=allow_stub_local,
                status_sink=status_sink,
                repair_invalid_xml=repair_invalid_xml,
                allow_compact_fallback=allow_compact_fallback,
                max_prompt_help_chars=max_prompt_help_chars,
                local_gpu_topology=local_gpu_topology,
                local_offload_policy=local_offload_policy,
                local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
                resume_existing=resume_existing,
                checkpoint_records_path=checkpoint_records_path,
                run_planemo_tests=run_planemo_tests,
                planemo_test_options=planemo_test_options,
                source_context_settings=source_context_settings,
            )
        finally:
            if devices:
                if original_cuda_visible_devices is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

    records = _with_corpus_indices(_load_corpus_records(corpus_jsonl, limit=limit))
    shards_records = _shard_records(records, process_count)
    shard_root = _shard_root(generation_records_path)
    started_at = utc_now_iso()
    total_units = len(records)
    progress_by_shard = dict.fromkeys(range(process_count), 0)
    lock = threading.Lock()

    shards: list[BenchmarkShard] = []
    for index, shard_records in enumerate(shards_records):
        shard_gpu_device = ""
        if local_gpu_topology == "model-parallel":
            shard_gpu_device = ",".join(devices)
        elif index < len(devices):
            shard_gpu_device = devices[index]
        shard = BenchmarkShard(
            index=index,
            corpus_jsonl=shard_root / f"shard-{index:03d}" / "corpus.jsonl",
            wrappers_dir=wrappers_dir / f"shard-{index:03d}",
            generation_records_path=shard_root / f"shard-{index:03d}" / "generation.records.json",
            evaluation_report_path=shard_root / f"shard-{index:03d}" / "evaluation.summary.json",
            benchmark_summary_path=shard_root / f"shard-{index:03d}" / "benchmark.summary.json",
            checkpoint_records_path=shard_root / f"shard-{index:03d}" / "checkpoint.records.jsonl",
            status_log_path=shard_root / f"shard-{index:03d}" / "status.jsonl",
            stdout_log_path=shard_root / f"shard-{index:03d}" / "stdout.log",
            stderr_log_path=shard_root / f"shard-{index:03d}" / "stderr.log",
            gpu_device=shard_gpu_device,
        )
        _write_corpus_records(shard.corpus_jsonl, shard_records)
        shards.append(shard)

    _status_emit(
        status_sink,
        {
            "status": "benchmark-sharded-started",
            "processes": process_count,
            "gpu_devices": devices,
            "total": total_units,
            "artifact_format": artifact_format,
            "shard_root": str(shard_root),
            "min_items_per_process": min_items_per_process,
            "startup_stagger_seconds": startup_stagger_seconds,
            "local_gpu_topology": local_gpu_topology,
            "local_offload_policy": local_offload_policy,
            "local_gpu_memory_reserve_gib": local_gpu_memory_reserve_gib,
            "resume_existing": resume_existing,
            "checkpoint_records_path": str(checkpoint_records_path),
            "record_timeout_seconds": record_timeout_seconds,
            "run_planemo_tests": run_planemo_tests,
            "source_context": source_context_settings.to_dict(),
        },
    )

    with ThreadPoolExecutor(max_workers=process_count) as executor:
        futures = []
        for shard in shards:
            if futures and startup_stagger_seconds > 0:
                time.sleep(startup_stagger_seconds)
            env = os.environ.copy()
            env["GTSM_REPO_ROOT"] = str(paths.repo_root)
            if shard.gpu_device:
                env["CUDA_VISIBLE_DEVICES"] = shard.gpu_device
            command = _build_shard_command(
                paths=paths,
                shard=shard,
                provider=provider,
                model_variant=model_variant,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_workers=max_workers,
                xsd_path=xsd_path,
                run_planemo=run_planemo,
                allow_stub_local=allow_stub_local,
                repair_invalid_xml=repair_invalid_xml,
                allow_compact_fallback=allow_compact_fallback,
                max_prompt_help_chars=max_prompt_help_chars,
                local_gpu_topology=local_gpu_topology,
                local_offload_policy=local_offload_policy,
                local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
                resume_existing=resume_existing,
                checkpoint_records_path=shard.checkpoint_records_path,
                artifact_format=artifact_format,
                source_context_settings=source_context_settings,
            )
            futures.append(
                executor.submit(
                    _run_benchmark_shard_process,
                    shard=shard,
                    command=command,
                    env=env,
                    status_sink=status_sink,
                    progress_by_shard=progress_by_shard,
                    total_units=total_units,
                    started_at=started_at,
                    lock=lock,
                    record_timeout_seconds=record_timeout_seconds,
                )
            )
        for future in as_completed(futures):
            future.result()

    generation_records: list[dict] = []
    failures: list[dict] = []
    shard_startups: list[dict] = []
    for shard in shards:
        generation_records.extend(_read_json_file(shard.generation_records_path, default=[]))
        shard_summary = _read_json_file(shard.benchmark_summary_path, default={})
        if isinstance(shard_summary, dict):
            failures.extend(shard_summary.get("failures", []))
            startup = shard_summary.get("startup")
            if isinstance(startup, dict):
                shard_startups.append(
                    {
                        "shard_index": shard.index,
                        "gpu_device": shard.gpu_device,
                        **startup,
                    }
                )

    generation_records.sort(key=lambda item: int(item.get("corpus_index", -1)))
    failures.sort(key=lambda item: int(item.get("corpus_index", -1)))
    generation_records_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(generation_records_path, json.dumps(generation_records, indent=2))
    _write_checkpoint_records(checkpoint_records_path, generation_records)

    generated_wrappers = [
        output_path
        for record in generation_records
        if (output_path := output_path_from_record(record, artifact_format)) is not None
    ]
    evaluation_summary = evaluate_wrapper_paths(
        wrapper_paths=generated_wrappers,
        output_report=evaluation_report_path,
        xsd_path=xsd_path,
        run_planemo=run_planemo,
        run_planemo_tests=run_planemo_tests,
        planemo_test_options=planemo_test_options,
        artifact_format=artifact_format,
    ).to_dict()
    final_progress = make_progress_snapshot(
        started_at=started_at,
        completed_units=total_units,
        total_units=total_units,
    ).to_dict()
    model_load_seconds = [
        float(item.get("model_load_seconds") or 0.0)
        for item in shard_startups
        if "model_load_seconds" in item
    ]
    startup = {
        "started_at": started_at,
        "artifact_format": artifact_format,
        "source_context": source_context_settings.to_dict(),
        "processes": process_count,
        "gpu_devices": devices,
        "min_items_per_process": min_items_per_process,
        "startup_stagger_seconds": startup_stagger_seconds,
        "local_gpu_topology": local_gpu_topology,
        "local_offload_policy": local_offload_policy,
        "local_gpu_memory_reserve_gib": local_gpu_memory_reserve_gib,
        "resume_existing": resume_existing,
        "checkpoint_records_path": str(checkpoint_records_path),
        "record_timeout_seconds": record_timeout_seconds,
        "shards": shard_startups,
        "model_load_seconds_max": round(max(model_load_seconds), 4) if model_load_seconds else 0.0,
        "model_load_seconds_mean": _mean(model_load_seconds),
    }
    summary = BenchmarkSummary(
        created_at=utc_now_iso(),
        corpus_path=str(corpus_jsonl),
        provider=provider,
        model_variant=model_variant,
        attempted=total_units,
        succeeded=len(generation_records),
        failed=len(failures),
        wrappers_dir=str(wrappers_dir),
        generation_records_path=str(generation_records_path),
        evaluation_report_path=str(evaluation_report_path),
        artifact_format=artifact_format,
        progress=final_progress,
        startup=startup,
        failures=failures,
        quality=_benchmark_quality(
            attempted=total_units,
            generation_records=generation_records,
            failures=failures,
            evaluation_report=evaluation_summary,
            progress=final_progress,
            startup=startup,
            artifact_format=artifact_format,
        ),
    )
    benchmark_summary_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_summary_path.write_text(summary.to_json(), encoding="utf-8")
    return summary
