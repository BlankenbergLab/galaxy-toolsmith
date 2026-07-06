from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.artifacts import ARTIFACT_FORMAT_XML, normalize_artifact_format
from galaxy_toolsmith.inference.generation import (
    _generation_validation_passed,
    _get_provider,
    _merge_generation_attempt_records,
    _repair_context,
    _repair_mode_for_validation,
    _source_code_for_repair,
    generate_wrapper_from_content,
)
from galaxy_toolsmith.inference.postprocess import (
    write_datatype_scaffold,
    write_macro_opportunities,
    write_toolsmith_macros_file,
)
from galaxy_toolsmith.inference.prompt_context import DEFAULT_MAX_PROMPT_HELP_CHARS
from galaxy_toolsmith.inference.repository import (
    ToolShedMetadata,
    build_tool_shed_metadata,
    safe_repository_name,
    safe_tool_id,
    write_gtsm_json,
    write_shed_yml,
)
from galaxy_toolsmith.inference.source_context import SourceContextSettings

_COMMAND_SECTION_RE = re.compile(
    r"(?im)^\s*(?:available\s+)?(?:commands|subcommands)\s*:?\s*$"
)
_COMMAND_LINE_RE = re.compile(r"^\s{0,8}([A-Za-z][A-Za-z0-9_.-]{1,40})\s{2,}(.+?)\s*$")
_USAGE_SUBCOMMAND_RE = re.compile(
    r"(?im)^\s*usage:\s+([A-Za-z0-9_.-]+)\s+(?:\[[^\]]+\]\s+)*\{([^}]+)\}"
)
_NOISE_COMMANDS = {
    "options",
    "optional",
    "required",
    "usage",
    "help",
    "version",
    "examples",
    "arguments",
}


@dataclass(frozen=True)
class SuiteToolPlan:
    tool_id: str
    name: str
    command_focus: str = ""
    prompt_focus: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuitePlan:
    suite_recommended: bool
    reason: str
    suite_name: str
    repository_name: str
    tools: tuple[SuiteToolPlan, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tools"] = [tool.to_dict() for tool in self.tools]
        payload["warnings"] = list(self.warnings)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class SuiteGenerationRecord:
    created_at: str
    tool_name: str
    output_dir: str
    suite_plan: dict[str, Any]
    shed_metadata: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)
    shed_yml_path: str = ""
    generation_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    generated_files: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    sidecar_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    manifest_path: str = ""
    records_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generation_records"] = [dict(record) for record in self.generation_records]
        payload["generated_files"] = [dict(item) for item in self.generated_files]
        payload["sidecar_artifacts"] = [dict(item) for item in self.sidecar_artifacts]
        payload["warnings"] = list(self.warnings)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value.strip())
    return deduped


def _dedupe_artifact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = (
            str(record.get("path") or ""),
            str(record.get("role") or ""),
            str(record.get("root_tag") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _candidate_subcommands_from_usage(help_text: str) -> list[str]:
    candidates: list[str] = []
    for match in _USAGE_SUBCOMMAND_RE.finditer(help_text):
        for value in match.group(2).split(","):
            candidate = value.strip()
            if candidate and candidate.lower() not in _NOISE_COMMANDS:
                candidates.append(candidate)
    return candidates


def _candidate_subcommands_from_sections(help_text: str) -> list[str]:
    lines = help_text.splitlines()
    candidates: list[str] = []
    for index, line in enumerate(lines):
        if not _COMMAND_SECTION_RE.match(line):
            continue
        for subline in lines[index + 1 : index + 40]:
            if not subline.strip():
                if candidates:
                    break
                continue
            if subline.lstrip() == subline and not _COMMAND_LINE_RE.match(subline):
                break
            match = _COMMAND_LINE_RE.match(subline)
            if not match:
                continue
            candidate = match.group(1).strip()
            if candidate.lower() in _NOISE_COMMANDS:
                continue
            candidates.append(candidate)
    return candidates


def infer_suite_tool_plans(
    *,
    tool_name: str,
    help_text: str,
    max_suite_tools: int = 8,
    force_suite: bool = False,
) -> tuple[SuiteToolPlan, ...]:
    command_name = safe_tool_id(tool_name)
    subcommands = _dedupe_preserve_order(
        _candidate_subcommands_from_usage(help_text)
        + _candidate_subcommands_from_sections(help_text)
    )
    if not subcommands:
        return (
            SuiteToolPlan(
                tool_id=command_name,
                name=tool_name or command_name,
                command_focus=tool_name or command_name,
                prompt_focus="Generate a focused Galaxy wrapper for the primary command.",
            ),
        )
    limited = subcommands[: max(1, max_suite_tools)]
    plans: list[SuiteToolPlan] = []
    for subcommand in limited:
        tool_id = safe_tool_id(f"{command_name}_{subcommand}")
        plans.append(
            SuiteToolPlan(
                tool_id=tool_id,
                name=f"{tool_name} {subcommand}".strip(),
                command_focus=f"{tool_name} {subcommand}".strip(),
                prompt_focus=(
                    f"Generate only the Galaxy wrapper for the '{subcommand}' subcommand. "
                    "Do not merge sibling subcommands into this wrapper."
                ),
            )
        )
    if not force_suite and len(plans) <= 1:
        return (
            SuiteToolPlan(
                tool_id=command_name,
                name=tool_name or command_name,
                command_focus=tool_name or command_name,
                prompt_focus="Generate a focused Galaxy wrapper for the primary command.",
            ),
        )
    return tuple(plans)


def plan_suite_from_content(
    *,
    tool_name: str,
    help_text: str,
    source_code: str = "",
    max_suite_tools: int = 8,
    force_suite: bool = False,
) -> SuitePlan:
    del source_code
    tools = infer_suite_tool_plans(
        tool_name=tool_name,
        help_text=help_text,
        max_suite_tools=max_suite_tools,
        force_suite=force_suite,
    )
    recommended = force_suite or len(tools) > 1
    repository_name = safe_repository_name(tool_name)
    suite_name = safe_repository_name(
        repository_name if repository_name.startswith("suite_") else f"suite_{repository_name}"
    )
    if recommended:
        reason = "Multiple subcommands or an explicit suite request were detected."
    else:
        reason = "No strong multi-tool suite signal was detected; single-tool output is preferred."
    warnings: list[str] = []
    if len(tools) >= max(1, max_suite_tools):
        warnings.append(f"Suite plan was capped at {max_suite_tools} tools.")
    return SuitePlan(
        suite_recommended=recommended,
        reason=reason,
        suite_name=suite_name,
        repository_name=repository_name,
        tools=tools,
        warnings=tuple(warnings),
    )


def _focused_source_context(source_code: str, plan: SuitePlan, tool_plan: SuiteToolPlan) -> str:
    suite_context = [
        "Suite generation context:",
        f"Repository: {plan.repository_name}",
        f"Suite: {plan.suite_name}",
        f"Current tool: {tool_plan.tool_id}",
        f"Command focus: {tool_plan.command_focus}",
        tool_plan.prompt_focus,
        "Sibling suite members should be separate primary Galaxy tools, not sidecars.",
    ]
    return "\n\n".join(part for part in (source_code, "\n".join(suite_context)) if part)


def _suite_member_output_contract(plan: SuitePlan, tool_plan: SuiteToolPlan) -> str:
    sibling_commands = [
        sibling.command_focus
        for sibling in plan.tools
        if sibling.tool_id != tool_plan.tool_id and sibling.command_focus
    ]
    sibling_line = ""
    if sibling_commands:
        sibling_line = "Sibling commands not to generate here: " + ", ".join(
            sibling_commands[:16]
        )
    return "\n".join(
        part
        for part in [
            "Suite member output contract:",
            "Generate exactly one Galaxy <tool> XML document for this suite member.",
            f"Current tool id: {tool_plan.tool_id}",
            f"Current command: {tool_plan.command_focus}",
            "Do not generate XML for sibling commands or other suite members.",
            "Do not append a second <tool>, <macros>, <tables>, or other XML document after this tool.",
            "Stop immediately after the closing </tool> for this tool.",
            sibling_line,
        ]
        if part
    )


def _normalized_command_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _focused_runtime_help(
    subcommand_help: Mapping[str, str] | None,
    command_focus: str,
) -> str:
    if not subcommand_help:
        return ""
    focus = _normalized_command_text(command_focus)
    focus_tail = _normalized_command_text(" ".join(command_focus.split()[1:]))
    for command, help_text in subcommand_help.items():
        normalized = _normalized_command_text(command)
        if normalized == focus or (focus_tail and normalized == focus_tail):
            return f"Focused runtime help for `{command}`:\n\n{help_text.strip()}"
    for command, help_text in subcommand_help.items():
        normalized = _normalized_command_text(command)
        if normalized and (focus.startswith(normalized) or normalized.startswith(focus)):
            return f"Focused runtime help for `{command}`:\n\n{help_text.strip()}"
    return ""


def _attempt_raw_log_path(raw_dir: Path | None, tool_id: str, attempt: int = 1) -> Path | None:
    if raw_dir is None:
        return None
    suffix = "" if attempt <= 1 else f".attempt-{attempt}"
    return raw_dir / f"{tool_id}{suffix}.log"


def _record_xml_path(record: Mapping[str, Any]) -> str:
    return str(record.get("output_xml_path") or record.get("output_path") or "")


def _record_has_feedback(record: Mapping[str, Any]) -> bool:
    validation = record.get("validation")
    initial_validation = record.get("initial_validation")
    candidate_artifacts = record.get("candidate_artifacts")
    return (
        record.get("repair_attempted") is True
        or _validation_has_generation_problems(validation)
        or (
            isinstance(initial_validation, Mapping)
            and bool(initial_validation)
            and not _generation_validation_passed(initial_validation, ARTIFACT_FORMAT_XML)
        )
        or int(record.get("selected_candidate_attempt") or 0) != 1
        or int(record.get("selected_candidate_index") or 0) != 1
        or (isinstance(candidate_artifacts, list) and len(candidate_artifacts) > 1)
    )


def _validation_has_generation_problems(validation: object) -> bool:
    if not isinstance(validation, Mapping):
        return False
    diagnostics = validation.get("generation_diagnostics")
    return isinstance(diagnostics, Mapping) and diagnostics.get("has_problems") is True


def _candidate_feedback(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attempt": candidate.get("attempt"),
        "candidate_index": candidate.get("candidate_index"),
        "selected": candidate.get("selected"),
        "score": candidate.get("score"),
        "score_reasons": candidate.get("score_reasons") or [],
        "features": candidate.get("features") or {},
        "validation": candidate.get("validation") or {},
        "postprocess": candidate.get("postprocess") or {},
        "path": candidate.get("path") or "",
    }


def write_repair_feedback(output_dir: Path, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    gtsm_dir = output_dir / ".gtsm"
    gtsm_dir.mkdir(parents=True, exist_ok=True)
    path = gtsm_dir / "repair-feedback.jsonl"
    feedback_records: list[dict[str, Any]] = []
    for record in records:
        if not _record_has_feedback(record):
            continue
        candidates = record.get("candidate_artifacts")
        feedback_records.append(
            {
                "tool_id": record.get("tool_id") or "",
                "tool_name": record.get("tool_name") or "",
                "tool_display_name": record.get("tool_display_name") or "",
                "provider": record.get("provider") or "",
                "model_variant": record.get("model_variant") or "",
                "generation_settings": record.get("generation_settings") or {},
                "repair_attempted": record.get("repair_attempted") is True,
                "repair_mode": record.get("repair_mode") or "",
                "repair_no_progress": record.get("repair_no_progress") is True,
                "repair_reason": record.get("repair_reason") or "",
                "initial_validation": record.get("initial_validation") or {},
                "final_validation": record.get("validation") or {},
                "selected_candidate_attempt": record.get("selected_candidate_attempt") or 0,
                "selected_candidate_index": record.get("selected_candidate_index") or 0,
                "postprocess": record.get("postprocess") or {},
                "raw_response_log_path": record.get("raw_response_log_path") or "",
                "initial_raw_response_log_path": record.get("initial_raw_response_log_path") or "",
                "candidate_artifacts": [
                    _candidate_feedback(candidate)
                    for candidate in (candidates if isinstance(candidates, list) else [])
                    if isinstance(candidate, Mapping)
                ],
            }
        )
    text = "\n".join(json.dumps(item, sort_keys=True) for item in feedback_records)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return {
        "path": str(path),
        "record_count": len(feedback_records),
        "bytes": path.stat().st_size,
    }


def compare_generation_run_dirs(
    *,
    left_run_dir: Path,
    right_run_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    left_records = _load_generation_records(left_run_dir)
    right_records = _load_generation_records(right_run_dir)
    left_by_key = {_record_compare_key(record): record for record in left_records}
    right_by_key = {_record_compare_key(record): record for record in right_records}
    keys = sorted(set(left_by_key) | set(right_by_key))
    records = [
        _compare_generation_records(key, left_by_key.get(key), right_by_key.get(key))
        for key in keys
    ]
    payload = {
        "generated_by": "galaxy-toolsmith",
        "created_at": utc_now_iso(),
        "left_run_dir": str(left_run_dir),
        "right_run_dir": str(right_run_dir),
        "summary": {
            "left_record_count": len(left_records),
            "right_record_count": len(right_records),
            "common_record_count": sum(
                1 for item in records if item["left"]["present"] and item["right"]["present"]
            ),
            "left_valid_count": sum(1 for item in records if item["left"].get("valid") is True),
            "right_valid_count": sum(1 for item in records if item["right"].get("valid") is True),
            "left_repair_count": sum(
                1 for item in records if item["left"].get("repair_attempted") is True
            ),
            "right_repair_count": sum(
                1 for item in records if item["right"].get("repair_attempted") is True
            ),
        },
        "records": records,
    }
    target = output_path or left_run_dir / ".gtsm" / "generation-comparison.json"
    write_gtsm_json(target, payload)
    payload["path"] = str(target)
    return payload


def _load_generation_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / ".gtsm" / "generation-records.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    records = payload.get("records") if isinstance(payload, Mapping) else []
    return [dict(item) for item in records if isinstance(item, Mapping)]


def _record_compare_key(record: Mapping[str, Any]) -> str:
    return str(
        record.get("tool_id")
        or record.get("tool_name")
        or record.get("tool_display_name")
        or record.get("output_path")
        or ""
    )


def _compare_generation_records(
    key: str,
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    left_summary = _record_comparison_summary(left)
    right_summary = _record_comparison_summary(right)
    return {
        "key": key,
        "left": left_summary,
        "right": right_summary,
        "differences": {
            "valid": left_summary.get("valid") != right_summary.get("valid"),
            "repair_attempted": left_summary.get("repair_attempted")
            != right_summary.get("repair_attempted"),
            "diagnostic_problems": left_summary.get("diagnostic_problems")
            != right_summary.get("diagnostic_problems"),
            "unknown_datatypes": left_summary.get("unknown_datatypes")
            != right_summary.get("unknown_datatypes"),
            "selected_candidate": (
                left_summary.get("selected_candidate_attempt"),
                left_summary.get("selected_candidate_index"),
            )
            != (
                right_summary.get("selected_candidate_attempt"),
                right_summary.get("selected_candidate_index"),
            ),
        },
    }


def _record_comparison_summary(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"present": False}
    validation = record.get("validation")
    diagnostics = (
        validation.get("generation_diagnostics")
        if isinstance(validation, Mapping)
        else {}
    )
    output_path_text = str(record.get("output_path") or record.get("output_xml_path") or "")
    output_path = Path(output_path_text) if output_path_text else None
    return {
        "present": True,
        "tool_id": record.get("tool_id") or "",
        "tool_name": record.get("tool_name") or "",
        "tool_display_name": record.get("tool_display_name") or "",
        "provider": record.get("provider") or "",
        "model_variant": record.get("model_variant") or "",
        "generation_settings": record.get("generation_settings") or {},
        "valid": _generation_validation_passed(validation, ARTIFACT_FORMAT_XML)
        if isinstance(validation, Mapping)
        else False,
        "diagnostic_problems": diagnostics.get("problems", [])
        if isinstance(diagnostics, Mapping)
        else [],
        "unknown_datatypes": validation.get("unknown_datatypes", [])
        if isinstance(validation, Mapping)
        else [],
        "repair_attempted": record.get("repair_attempted") is True,
        "repair_mode": record.get("repair_mode") or "",
        "repair_no_progress": record.get("repair_no_progress") is True,
        "selected_candidate_attempt": record.get("selected_candidate_attempt") or 0,
        "selected_candidate_index": record.get("selected_candidate_index") or 0,
        "candidate_count": len(record.get("candidate_artifacts") or []),
        "raw_response_chars": record.get("raw_response_chars") or 0,
        "output_path": output_path_text,
        "output_bytes": output_path.stat().st_size if output_path is not None and output_path.is_file() else 0,
        "postprocess": record.get("postprocess") or {},
        "sidecar_count": len(record.get("sidecar_artifacts") or []),
    }


def generate_suite_from_content(
    *,
    paths: WorkspacePaths,
    tool_name: str,
    help_text: str,
    source_code: str,
    output_dir: Path,
    provider_name: str,
    model_variant: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider_instance: Any | None = None,
    skills_profile: str = "default",
    allow_stub_local: bool = False,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    ollama_context_tokens: int | None = None,
    source_context_summary: Mapping[str, Any] | None = None,
    max_suite_tools: int = 8,
    force_suite: bool = True,
    generate_sidecars: bool = True,
    raw_response_logs: bool = False,
    stream_output: bool = False,
    repair_invalid_xml: bool = True,
    shed_metadata: ToolShedMetadata | None = None,
    write_shed: bool = True,
    subcommand_help: Mapping[str, str] | None = None,
    include_toolsmith_citation: bool = True,
    datatype_scaffold: bool = True,
) -> SuiteGenerationRecord:
    artifact_format = normalize_artifact_format(ARTIFACT_FORMAT_XML)
    if artifact_format != ARTIFACT_FORMAT_XML:
        raise ValueError("Suite generation currently supports XML only.")
    output_dir.mkdir(parents=True, exist_ok=True)
    gtsm_dir = output_dir / ".gtsm"
    raw_dir = gtsm_dir / "raw" if raw_response_logs or stream_output else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_suite_from_content(
        tool_name=tool_name,
        help_text=help_text,
        source_code=source_code,
        max_suite_tools=max_suite_tools,
        force_suite=force_suite,
    )
    provider_instance = provider_instance or _get_provider(
        provider_name,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        paths=paths,
        allow_stub_local=allow_stub_local,
        local_offload_policy=local_offload_policy,
        local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        ollama_context_tokens=ollama_context_tokens,
    )

    generation_records: list[dict[str, Any]] = []
    generated_files: list[dict[str, Any]] = []
    sidecar_artifacts: list[dict[str, Any]] = []
    warnings: list[str] = list(plan.warnings)
    citation_mode = "macro" if generate_sidecars else "direct"
    for tool_index, tool_plan in enumerate(plan.tools, start=1):
        output_path = output_dir / f"{tool_plan.tool_id}.xml"
        focused_runtime_help = _focused_runtime_help(subcommand_help, tool_plan.command_focus)
        suite_member_contract = _suite_member_output_contract(plan, tool_plan)
        focused_help = "\n\n".join(
            part
            for part in [
                help_text,
                focused_runtime_help,
                "Suite member focus:",
                f"Tool id: {tool_plan.tool_id}",
                f"Command focus: {tool_plan.command_focus}",
                tool_plan.prompt_focus,
            ]
            if part
        )
        if stream_output:
            print(
                (
                    f"\n[gtsm] suite member {tool_index}/{len(plan.tools)}: "
                    f"{tool_plan.tool_id} ({tool_plan.command_focus})"
                ),
                file=sys.stderr,
                flush=True,
            )
        focused_source_code = _focused_source_context(source_code, plan, tool_plan)
        first = generate_wrapper_from_content(
            paths=paths,
            tool_name=tool_plan.tool_id,
            help_text=focused_help,
            source_code=focused_source_code,
            output_path=output_path,
            provider_name=provider_name,
            model_variant=model_variant,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            skills_profile=skills_profile,
            allow_stub_local=allow_stub_local,
            provider_instance=provider_instance,
            max_prompt_help_chars=max_prompt_help_chars,
            local_offload_policy=local_offload_policy,
            local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
            repair_context=suite_member_contract,
            artifact_format=ARTIFACT_FORMAT_XML,
            source_context_summary=source_context_summary,
            raw_response_log_path=_attempt_raw_log_path(raw_dir, tool_plan.tool_id),
            stream_output=stream_output,
            generate_sidecars=generate_sidecars,
            sidecar_output_dir=output_dir,
            tool_id=tool_plan.tool_id,
            tool_display_name=tool_plan.name,
            include_toolsmith_citation=include_toolsmith_citation,
            toolsmith_citation_mode=citation_mode,
            datatype_scaffold=False,
        )
        record = json.loads(first.to_json())
        if repair_invalid_xml and not _generation_validation_passed(
            first.validation,
            ARTIFACT_FORMAT_XML,
        ):
            repair_mode = _repair_mode_for_validation(first.validation)
            second = generate_wrapper_from_content(
                paths=paths,
                tool_name=tool_plan.tool_id,
                help_text=focused_help,
                source_code=_source_code_for_repair(focused_source_code, repair_mode),
                output_path=output_path,
                provider_name=provider_name,
                model_variant=model_variant,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                skills_profile=skills_profile,
                allow_stub_local=allow_stub_local,
                provider_instance=provider_instance,
                max_prompt_help_chars=max_prompt_help_chars,
                local_offload_policy=local_offload_policy,
                local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
                repair_context="\n\n".join(
                    [
                        suite_member_contract,
                        _repair_context(tool_plan.tool_id, first.validation),
                    ]
                ),
                attempt_count=2,
                repair_attempted=True,
                repair_mode=repair_mode,
                repair_reason=str(first.validation.get("notes") or "invalid generated XML"),
                artifact_format=ARTIFACT_FORMAT_XML,
                source_context_summary=source_context_summary,
                raw_response_log_path=_attempt_raw_log_path(raw_dir, tool_plan.tool_id, attempt=2),
                stream_output=stream_output,
                generate_sidecars=generate_sidecars,
                sidecar_output_dir=output_dir,
                initial_validation=first.validation,
                initial_raw_response_log_path=first.raw_response_log_path,
                tool_id=tool_plan.tool_id,
                tool_display_name=tool_plan.name,
                include_toolsmith_citation=include_toolsmith_citation,
                toolsmith_citation_mode=citation_mode,
                datatype_scaffold=False,
            )
            second = _merge_generation_attempt_records(
                first=first,
                second=second,
                output_path=output_path,
            )
            record = json.loads(second.to_json())
        generation_records.append(record)
        generated_files.append(
            {
                "path": _record_xml_path(record),
                "role": "tool_xml",
                "tool_id": tool_plan.tool_id,
                "tool_name": tool_plan.name,
            }
        )
        sidecar_artifacts.extend(record.get("sidecar_artifacts", []))

    if include_toolsmith_citation and citation_mode == "macro":
        macros_path = write_toolsmith_macros_file(output_dir)
        if not any(item.get("path") == str(macros_path) for item in sidecar_artifacts):
            sidecar_artifacts.append(
                {
                    "path": str(macros_path),
                    "role": "macros",
                    "root_tag": "macros",
                    "bytes": macros_path.stat().st_size,
                }
            )
    macro_opportunities_path = write_macro_opportunities(output_dir, generation_records)
    sidecar_artifacts.append(
        {
            "path": str(macro_opportunities_path),
            "role": "macro_opportunities",
            "root_tag": "",
            "bytes": macro_opportunities_path.stat().st_size,
        }
    )
    if datatype_scaffold:
        scaffold = write_datatype_scaffold(
            output_dir,
            generation_records,
            repository_style=True,
        )
        if scaffold.get("metadata_path"):
            sidecar_artifacts.append(
                {
                    "path": str(scaffold["metadata_path"]),
                    "role": "datatype_scaffold",
                    "root_tag": "",
                    "bytes": Path(str(scaffold["metadata_path"])).stat().st_size,
                }
            )
    repair_feedback = write_repair_feedback(output_dir, generation_records)
    sidecar_artifacts.append(
        {
            "path": str(repair_feedback["path"]),
            "role": "repair_feedback",
            "root_tag": "",
            "bytes": int(repair_feedback["bytes"]),
            "record_count": int(repair_feedback["record_count"]),
        }
    )
    sidecar_artifacts = _dedupe_artifact_records(sidecar_artifacts)

    member_validations = [
        record.get("validation", {})
        for record in generation_records
        if isinstance(record.get("validation"), dict)
    ]
    valid_member_count = sum(
        1
        for validation in member_validations
        if _generation_validation_passed(validation, ARTIFACT_FORMAT_XML)
    )
    aggregate_validation = {
        "artifact_valid": bool(member_validations)
        and valid_member_count == len(member_validations),
        "xml_well_formed": bool(member_validations)
        and all(validation.get("xml_well_formed") is True for validation in member_validations),
        "root_is_tool": bool(member_validations)
        and all(validation.get("root_is_tool") is True for validation in member_validations),
        "suite_member_count": len(member_validations),
        "valid_suite_member_count": valid_member_count,
    }

    if shed_metadata is not None and shed_metadata.suite and not shed_metadata.repositories:
        metadata = build_tool_shed_metadata(
            name=shed_metadata.name,
            owner=shed_metadata.owner,
            description=shed_metadata.description,
            homepage_url=shed_metadata.homepage_url,
            remote_repository_url=shed_metadata.remote_repository_url,
            categories=shed_metadata.categories,
            suite=True,
            repositories=[tool.tool_id for tool in plan.tools],
        )
    else:
        metadata = shed_metadata or build_tool_shed_metadata(
            name=plan.suite_name if len(plan.tools) > 1 else plan.repository_name,
            description=f"Generated Galaxy Toolsmith repository for {tool_name}",
            suite=len(plan.tools) > 1,
            repositories=[tool.tool_id for tool in plan.tools],
        )
    shed_path = output_dir / ".shed.yml"
    shed_yml_path = str(write_shed_yml(shed_path, metadata)) if write_shed else ""
    plan_path = write_gtsm_json(gtsm_dir / "suite-plan.json", plan.to_dict())
    records_path = write_gtsm_json(
        gtsm_dir / "generation-records.json",
        {"records": generation_records},
    )
    manifest = SuiteGenerationRecord(
        created_at=utc_now_iso(),
        tool_name=tool_name,
        output_dir=str(output_dir),
        suite_plan=plan.to_dict(),
        shed_metadata=metadata.to_dict(),
        validation=aggregate_validation,
        shed_yml_path=shed_yml_path,
        generation_records=tuple(generation_records),
        generated_files=tuple(generated_files),
        sidecar_artifacts=tuple(sidecar_artifacts),
        warnings=tuple(warnings),
        manifest_path=str(plan_path),
        records_path=str(records_path),
    )
    write_gtsm_json(gtsm_dir / "suite-generation.json", manifest.to_dict())
    return manifest


def generate_suite(
    *,
    paths: WorkspacePaths,
    tool_name: str,
    help_text_path: Path | None,
    source_path: Path | None,
    output_dir: Path,
    provider_name: str,
    model_variant: str,
    model: str,
    temperature: float,
    max_tokens: int,
    skills_profile: str = "default",
    allow_stub_local: bool = False,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    ollama_context_tokens: int | None = None,
    source_context_settings: SourceContextSettings | None = None,
    max_suite_tools: int = 8,
    generate_sidecars: bool = True,
    raw_response_logs: bool = False,
    stream_output: bool = False,
    repair_invalid_xml: bool = True,
    shed_metadata: ToolShedMetadata | None = None,
    write_shed: bool = True,
    help_text: str | None = None,
    subcommand_help: Mapping[str, str] | None = None,
    include_toolsmith_citation: bool = True,
    datatype_scaffold: bool = True,
) -> SuiteGenerationRecord:
    from galaxy_toolsmith.inference.source_context import build_source_context_from_paths

    source_context = build_source_context_from_paths(
        settings=source_context_settings,
        source_file=source_path,
    )
    if help_text is None:
        if help_text_path is None:
            raise ValueError("help_text_path is required when help_text is not provided.")
        help_text = help_text_path.read_text(encoding="utf-8")
    return generate_suite_from_content(
        paths=paths,
        tool_name=tool_name,
        help_text=help_text,
        source_code=source_context.text,
        output_dir=output_dir,
        provider_name=provider_name,
        model_variant=model_variant,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        skills_profile=skills_profile,
        allow_stub_local=allow_stub_local,
        max_prompt_help_chars=max_prompt_help_chars,
        local_offload_policy=local_offload_policy,
        local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        ollama_context_tokens=ollama_context_tokens,
        source_context_summary=source_context.to_dict(),
        max_suite_tools=max_suite_tools,
        force_suite=True,
        generate_sidecars=generate_sidecars,
        raw_response_logs=raw_response_logs,
        stream_output=stream_output,
        repair_invalid_xml=repair_invalid_xml,
        shed_metadata=shed_metadata,
        write_shed=write_shed,
        subcommand_help=subcommand_help,
        include_toolsmith_citation=include_toolsmith_citation,
        datatype_scaffold=datatype_scaffold,
    )
