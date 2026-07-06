from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    normalize_artifact_format,
)
from galaxy_toolsmith.inference.output_diagnostics import diagnose_generated_xml
from galaxy_toolsmith.inference.postprocess import (
    datatype_scaffold_dir_for_output,
    postprocess_generated_artifact,
    write_datatype_scaffold,
    write_toolsmith_macros_file,
)
from galaxy_toolsmith.inference.prompt_context import (
    DEFAULT_MAX_PROMPT_HELP_CHARS,
    extract_interface_hints,
    shape_help_text,
)
from galaxy_toolsmith.inference.repository import safe_tool_id
from galaxy_toolsmith.inference.source_context import (
    SourceContextSettings,
    build_source_context_from_paths,
)
from galaxy_toolsmith.inference.udt import validate_udt_yaml
from galaxy_toolsmith.inference.validation import validate_wrapper
from galaxy_toolsmith.providers.base import (
    GenerationInput,
    Provider,
    extract_complete_tool_xml_candidates,
)
from galaxy_toolsmith.providers.external import AnthropicProvider, CopilotProvider, OpenAIProvider
from galaxy_toolsmith.providers.local import LocalProvider
from galaxy_toolsmith.providers.ollama import OllamaProvider


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class GenerationRecord:
    request_id: str
    created_at: str
    tool_name: str
    provider: str
    model_variant: str
    skills_profile: str
    output_xml_path: str
    report_path: str
    tool_id: str = ""
    tool_display_name: str = ""
    artifact_format: str = ARTIFACT_FORMAT_XML
    output_path: str = ""
    output_udt_yaml_path: str = ""
    validation: dict = field(default_factory=dict)
    prompt_help: dict = field(default_factory=dict)
    source_context: dict = field(default_factory=dict)
    raw_response_log_path: str = ""
    raw_response_chars: int = 0
    generation_settings: dict = field(default_factory=dict)
    sidecar_artifacts: list[dict] = field(default_factory=list)
    candidate_artifacts: list[dict] = field(default_factory=list)
    selected_candidate_attempt: int = 0
    selected_candidate_index: int = 0
    postprocess: dict = field(default_factory=dict)
    datatype_scaffold: dict = field(default_factory=dict)
    attempt_count: int = 1
    repair_attempted: bool = False
    repair_mode: str = ""
    repair_no_progress: bool = False
    repair_reason: str = ""
    initial_validation: dict = field(default_factory=dict)
    initial_raw_response_log_path: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _load_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _attempt_raw_response_path(path: Path | None, attempt: int) -> Path | None:
    if path is None:
        return None
    if attempt <= 1:
        return path
    suffix = path.suffix or ".log"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.attempt-{attempt}{suffix}")


def _xml_generation_passed(validation: Mapping[str, object]) -> bool:
    diagnostics = validation.get("generation_diagnostics")
    return (
        validation.get("xml_well_formed") is True
        and validation.get("root_is_tool") is True
        and not (
            isinstance(diagnostics, Mapping)
            and diagnostics.get("has_problems") is True
        )
    )


def _generation_validation_passed(validation: Mapping[str, object], artifact_format: str) -> bool:
    if normalize_artifact_format(artifact_format) == ARTIFACT_FORMAT_UDT_YAML:
        return (
            validation.get("yaml_well_formed") is True
            and validation.get("schema_valid") is True
            and validation.get("root_is_user_tool") is True
        )
    return _xml_generation_passed(validation)


def _repair_reason(validation: Mapping[str, object]) -> str:
    notes = validation.get("notes")
    if isinstance(notes, list):
        joined = "; ".join(str(note) for note in notes if str(note).strip())
        if joined:
            return joined
    root_tag = str(validation.get("root_tag", "") or "").strip()
    if root_tag and root_tag != "tool":
        return f"Expected root <tool>; found <{root_tag}>."
    return "Generated XML failed validation."


def _repair_context(tool_name: str, validation: Mapping[str, object]) -> str:
    root_tag = str(validation.get("root_tag", "") or "").strip()
    repair_mode = _repair_mode_for_validation(validation)
    lines = [
        "The previous generated artifact failed validation.",
        f"Tool name: {tool_name}",
        f"Failure: {_repair_reason(validation)}",
        "Return exactly one complete primary Galaxy <tool> XML document.",
        "Do not return macros.xml, tool_data_table_conf.xml, <tables>, or <macros> as the primary artifact.",
        "If tables or macros are needed, keep the primary wrapper valid and reference/generate sidecars separately.",
        "Stop immediately after the closing </tool> tag.",
    ]
    if root_tag in {"tables", "macros"}:
        lines.append(
            f"The previous response was a sidecar root <{root_tag}>. Generate the wrapper <tool> first."
        )
    diagnostics = validation.get("generation_diagnostics")
    if isinstance(diagnostics, Mapping) and diagnostics.get("missing_closing_tool") is True:
        lines.append("The previous output did not contain a closing </tool> tag.")
    if repair_mode == "repetition_compaction":
        lines.extend(
            [
                "The previous output repeated XML blocks or Cheetah output fragments.",
                "Regenerate a compact wrapper instead of continuing or copying the prior structure.",
                "Include exactly one minimal <test> element.",
                "Do not repeat <conditional>, <output>, <assert_contents>, or Cheetah $out_* fragments.",
                "Keep one command branch and one output declaration per real output format.",
            ]
        )
    return "\n".join(lines)


def _repair_mode_for_validation(validation: Mapping[str, object]) -> str:
    diagnostics = validation.get("generation_diagnostics")
    if isinstance(diagnostics, Mapping) and (
        int(diagnostics.get("repeated_xml_line_count") or 0) > 0
        or int(diagnostics.get("repeated_cheetah_fragments") or 0) > 0
        or diagnostics.get("too_many_tests") is True
        or int(diagnostics.get("test_count") or 0) > 1
    ):
        return "repetition_compaction"
    return "generic"


def _source_code_for_repair(source_code: str, repair_mode: str) -> str:
    if repair_mode == "repetition_compaction":
        return ""
    return source_code


def _sidecar_filename(sidecar: Mapping[str, object], index: int) -> str:
    role = str(sidecar.get("role") or "").strip()
    root_tag = str(sidecar.get("root_tag") or "").strip()
    if role == "macros" or root_tag == "macros":
        return "macros.xml" if index == 1 else f"macros-{index}.xml"
    if role == "tool_data_table_conf" or root_tag == "tables":
        return "tool_data_table_conf.xml" if index == 1 else f"tool_data_table_conf-{index}.xml"
    return f"sidecar-{index}.xml"


def _write_sidecar_artifacts(
    sidecars: tuple[dict, ...],
    *,
    output_path: Path,
    sidecar_output_dir: Path | None,
) -> list[dict]:
    if not sidecars:
        return []
    target_dir = sidecar_output_dir or output_path.with_name(f"{output_path.name}.sidecars")
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    counts: dict[str, int] = {}
    for sidecar in sidecars:
        root_tag = str(sidecar.get("root_tag") or "sidecar")
        counts[root_tag] = counts.get(root_tag, 0) + 1
        filename = _sidecar_filename(sidecar, counts[root_tag])
        path = target_dir / filename
        content = str(sidecar.get("content") or "")
        _atomic_write_text(path, content)
        written.append(
            {
                "path": str(path),
                "role": str(sidecar.get("role") or ""),
                "root_tag": root_tag,
                "bytes": len(content.encode("utf-8", errors="replace")),
            }
        )
    return written


def _candidate_output_dir(output_path: Path) -> Path:
    return output_path.parent / ".gtsm" / "candidates" / output_path.stem


def _xml_candidate_texts(*, raw_response_text: str, artifact_text: str) -> tuple[str, ...]:
    source = raw_response_text or artifact_text
    candidates = list(extract_complete_tool_xml_candidates(source))
    if not candidates and artifact_text.strip():
        candidates = list(extract_complete_tool_xml_candidates(artifact_text))
    if not candidates and artifact_text.strip():
        candidates = [artifact_text.strip()]
    return tuple(candidates)


def _xml_candidate_features(artifact_text: str) -> dict[str, object]:
    features: dict[str, object] = {
        "has_command": False,
        "has_inputs": False,
        "has_outputs": False,
        "has_requirements": False,
        "has_tests": False,
        "has_help": False,
        "command_chars": 0,
        "input_count": 0,
        "output_count": 0,
        "requirement_count": 0,
        "test_count": 0,
    }
    try:
        root = ET.fromstring(artifact_text)
    except ET.ParseError:
        return features
    if root.tag != "tool":
        return features

    command = root.find("command")
    command_text = "".join(command.itertext()).strip() if command is not None else ""
    input_count = len(root.findall(".//inputs//param")) + len(root.findall(".//inputs//conditional"))
    output_count = len(root.findall(".//outputs//data")) + len(root.findall(".//outputs//collection"))
    requirement_count = len(root.findall(".//requirements//requirement"))
    test_count = len(root.findall(".//tests//test"))
    help_node = root.find("help")
    features.update(
        {
            "has_command": bool(command_text),
            "has_inputs": input_count > 0,
            "has_outputs": output_count > 0,
            "has_requirements": requirement_count > 0,
            "has_tests": test_count > 0,
            "has_help": help_node is not None and bool("".join(help_node.itertext()).strip()),
            "command_chars": len(command_text),
            "input_count": input_count,
            "output_count": output_count,
            "requirement_count": requirement_count,
            "test_count": test_count,
        }
    )
    return features


def _score_xml_candidate(
    artifact_text: str,
    *,
    validation: Mapping[str, object],
) -> dict[str, object]:
    score = 0
    reasons: list[str] = []
    features = _xml_candidate_features(artifact_text)

    if validation.get("xml_well_formed") is True:
        score += 100
        reasons.append("xml_well_formed:+100")
    else:
        score -= 1000
        reasons.append("xml_parse_failed:-1000")

    if validation.get("root_is_tool") is True:
        score += 100
        reasons.append("root_is_tool:+100")
    else:
        score -= 500
        reasons.append("root_not_tool:-500")

    diagnostics = validation.get("generation_diagnostics")
    if isinstance(diagnostics, Mapping) and diagnostics.get("has_problems") is True:
        problem_count = len(diagnostics.get("problems") or [])
        diagnostics_penalty = max(150, 150 * problem_count)
        score -= diagnostics_penalty
        reasons.append(f"generation_diagnostics:-{diagnostics_penalty}")
        extra_tests = max(0, int(diagnostics.get("test_count") or 0) - 1)
        if extra_tests:
            penalty = min(300, 25 * extra_tests)
            score -= penalty
            reasons.append(f"extra_tests:-{penalty}")
        repeated_xml_lines = int(diagnostics.get("repeated_xml_line_count") or 0)
        if repeated_xml_lines:
            penalty = min(300, 10 * repeated_xml_lines)
            score -= penalty
            reasons.append(f"repeated_xml_lines:-{penalty}")
        repeated_cheetah_fragments = int(diagnostics.get("repeated_cheetah_fragments") or 0)
        if repeated_cheetah_fragments:
            penalty = min(300, 10 * repeated_cheetah_fragments)
            score -= penalty
            reasons.append(f"repeated_cheetah_fragments:-{penalty}")
    else:
        score += 50
        reasons.append("no_generation_diagnostics:+50")

    if features["has_command"]:
        score += 80
        reasons.append("has_command:+80")
    else:
        score -= 120
        reasons.append("missing_command:-120")

    if features["has_outputs"]:
        score += 60
        reasons.append("has_outputs:+60")
    else:
        score -= 80
        reasons.append("missing_outputs:-80")

    if features["has_inputs"]:
        score += 30
        reasons.append("has_inputs:+30")
    else:
        score -= 10
        reasons.append("missing_inputs:-10")

    if features["has_requirements"]:
        score += 15
        reasons.append("has_requirements:+15")
    if features["has_tests"]:
        score += 15
        reasons.append("has_tests:+15")
    if features["has_help"]:
        score += 10
        reasons.append("has_help:+10")

    unknown_datatypes = validation.get("unknown_datatypes")
    if isinstance(unknown_datatypes, list) and unknown_datatypes:
        penalty = min(50, 5 * len(unknown_datatypes))
        score -= penalty
        reasons.append(f"unknown_datatypes:-{penalty}")

    return {
        "score": score,
        "reasons": reasons,
        "features": features,
    }


def _postprocessed_xml_candidates(
    *,
    output_artifact_text: str,
    raw_response_text: str,
    tool_id: str,
    tool_name: str,
    include_toolsmith_citation: bool,
    toolsmith_citation_mode: str,
) -> list[dict]:
    candidates: list[dict] = []
    for index, candidate_text in enumerate(
        _xml_candidate_texts(
            raw_response_text=raw_response_text,
            artifact_text=output_artifact_text,
        ),
        start=1,
    ):
        postprocess = postprocess_generated_artifact(
            candidate_text,
            artifact_format=ARTIFACT_FORMAT_XML,
            tool_id=tool_id,
            tool_name=tool_name,
            include_toolsmith_citation=include_toolsmith_citation,
            citation_mode=toolsmith_citation_mode,
        )
        artifact_text = postprocess.artifact_text
        validation = _validation_with_generation_diagnostics(
            artifact_text,
            artifact_format=ARTIFACT_FORMAT_XML,
        )
        scoring = _score_xml_candidate(artifact_text, validation=validation)
        candidates.append(
            {
                "candidate_index": index,
                "artifact_text": artifact_text,
                "validation": validation,
                "postprocess": postprocess.metadata,
                "score": scoring["score"],
                "score_reasons": scoring["reasons"],
                "features": scoring["features"],
            }
        )
    return candidates


def _select_xml_candidate(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    selected = candidates[0]
    for candidate in candidates[1:]:
        if _candidate_rank(candidate) > _candidate_rank(selected):
            selected = candidate
    return selected


def _write_xml_candidate_manifest(
    *,
    output_path: Path,
    candidate_artifacts: list[dict],
    selected_candidate_attempt: int,
    selected_candidate_index: int,
) -> None:
    if not candidate_artifacts:
        return
    candidate_dir = _candidate_output_dir(output_path)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        candidate_dir / "manifest.json",
        json.dumps(
            {
                "generated_by": "galaxy-toolsmith",
                "selected_candidate_attempt": selected_candidate_attempt,
                "selected_candidate_index": selected_candidate_index,
                "primary_output_path": str(output_path),
                "candidates": candidate_artifacts,
            },
            indent=2,
        ),
    )


def _write_xml_candidate_artifacts(
    candidates: list[dict],
    *,
    output_path: Path,
    attempt: int,
    selected_candidate_index: int,
) -> list[dict]:
    if not candidates:
        return []
    candidate_dir = _candidate_output_dir(output_path)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []
    manifest_candidates: list[dict] = []
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        artifact_text = str(candidate.get("artifact_text") or "")
        filename = f"candidate-{index}.xml" if attempt <= 1 else f"candidate-attempt-{attempt}-{index}.xml"
        path = candidate_dir / filename
        _atomic_write_text(path, artifact_text)
        payload = {
            "attempt": attempt,
            "candidate_index": index,
            "path": str(path),
            "selected": index == selected_candidate_index and attempt > 0,
            "score": int(candidate.get("score", 0)),
            "score_reasons": list(candidate.get("score_reasons") or []),
            "features": dict(candidate.get("features") or {}),
            "validation": dict(candidate.get("validation") or {}),
            "postprocess": dict(candidate.get("postprocess") or {}),
            "bytes": len(artifact_text.encode("utf-8", errors="replace")),
        }
        artifacts.append(payload)
        manifest_candidates.append(payload)
    _write_xml_candidate_manifest(
        output_path=output_path,
        candidate_artifacts=manifest_candidates,
        selected_candidate_attempt=attempt,
        selected_candidate_index=selected_candidate_index,
    )
    return artifacts


def _select_candidate_artifact(candidate_artifacts: list[dict]) -> dict | None:
    if not candidate_artifacts:
        return None
    selected = candidate_artifacts[0]
    for candidate in candidate_artifacts[1:]:
        if _candidate_rank(candidate) > _candidate_rank(selected):
            selected = candidate
    return selected


def _candidate_rank(candidate: Mapping[str, object]) -> tuple[int, int]:
    validation = candidate.get("validation")
    passed = _xml_generation_passed(validation) if isinstance(validation, Mapping) else False
    return (1 if passed else 0, int(candidate.get("score") or 0))


def _normalized_artifact_text(value: str) -> str:
    return "".join(str(value or "").split())


def _selected_artifact_text(record: GenerationRecord) -> str:
    selected = _select_candidate_artifact(
        [dict(candidate) for candidate in record.candidate_artifacts]
    )
    if selected is None:
        path_text = record.output_path or record.output_xml_path
    else:
        path_text = str(selected.get("path") or "")
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _repair_no_progress(first: GenerationRecord, second: GenerationRecord) -> bool:
    first_text = _normalized_artifact_text(_selected_artifact_text(first))
    second_text = _normalized_artifact_text(_selected_artifact_text(second))
    return bool(first_text and second_text and first_text == second_text)


def _merge_generation_attempt_records(
    *,
    first: GenerationRecord,
    second: GenerationRecord,
    output_path: Path,
) -> GenerationRecord:
    candidate_artifacts = [
        dict(candidate)
        for candidate in [*first.candidate_artifacts, *second.candidate_artifacts]
    ]
    selected = _select_candidate_artifact(candidate_artifacts)
    if selected is None:
        return second

    selected_attempt = int(selected.get("attempt") or 0)
    selected_index = int(selected.get("candidate_index") or 0)
    for candidate in candidate_artifacts:
        candidate["selected"] = (
            int(candidate.get("attempt") or 0) == selected_attempt
            and int(candidate.get("candidate_index") or 0) == selected_index
        )

    selected_path = Path(str(selected.get("path") or ""))
    if selected_path.is_file():
        _atomic_write_text(output_path, selected_path.read_text(encoding="utf-8"))
    _write_xml_candidate_manifest(
        output_path=output_path,
        candidate_artifacts=candidate_artifacts,
        selected_candidate_attempt=selected_attempt,
        selected_candidate_index=selected_index,
    )

    selected_validation = dict(selected.get("validation") or second.validation)
    report_path = Path(second.report_path)
    if str(report_path):
        _atomic_write_text(report_path, json.dumps(selected_validation, indent=2))
    repair_no_progress = _repair_no_progress(first, second)

    return replace(
        second,
        validation=selected_validation,
        postprocess=dict(selected.get("postprocess") or second.postprocess),
        candidate_artifacts=candidate_artifacts,
        selected_candidate_attempt=selected_attempt,
        selected_candidate_index=selected_index,
        initial_validation=first.validation,
        initial_raw_response_log_path=first.raw_response_log_path,
        repair_no_progress=repair_no_progress,
    )


def _generate_record(
    paths: WorkspacePaths,
    tool_name: str,
    help_text: str,
    source_code: str,
    output_path: Path,
    provider_name: str,
    model_variant: str,
    skills_profile: str,
    model: str,
    temperature: float,
    max_tokens: int,
    allow_stub_local: bool = False,
    provider_instance: Provider | None = None,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    ollama_context_tokens: int | None = None,
    repair_context: str = "",
    attempt_count: int = 1,
    repair_attempted: bool = False,
    repair_mode: str = "",
    repair_no_progress: bool = False,
    repair_reason: str = "",
    metadata_hints: Mapping[str, object] | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_summary: Mapping[str, object] | None = None,
    raw_response_log_path: Path | None = None,
    stream_output: bool = False,
    generate_sidecars: bool = False,
    sidecar_output_dir: Path | None = None,
    initial_validation: Mapping[str, object] | None = None,
    initial_raw_response_log_path: str = "",
    tool_id: str = "",
    tool_display_name: str = "",
    include_toolsmith_citation: bool = True,
    toolsmith_citation_mode: str = "direct",
    datatype_scaffold: bool = True,
    datatype_scaffold_dir: Path | None = None,
    datatype_scaffold_repository_style: bool = False,
) -> GenerationRecord:
    artifact_format = normalize_artifact_format(artifact_format)
    provider = provider_instance or _get_provider(
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
    shaped_help = shape_help_text(help_text, max_chars=max_prompt_help_chars)
    interface_hints = extract_interface_hints(help_text, metadata=metadata_hints)
    request = GenerationInput(
        tool_name=tool_name,
        help_text=shaped_help.text,
        source_code=source_code,
        model_variant=model_variant,
        skills_profile=skills_profile,
        repair_context=repair_context,
        interface_hints=interface_hints.text,
        artifact_format=artifact_format,
        raw_response_log_path=str(raw_response_log_path or ""),
        stream_output=stream_output,
        generate_sidecars=generate_sidecars,
    )
    output = provider.generate_wrapper(request)
    generated_tool_id = safe_tool_id(tool_id or tool_name)
    generated_tool_name = str(tool_display_name or tool_name or generated_tool_id)
    candidate_artifacts: list[dict] = []
    selected_candidate_index = 0
    if artifact_format == ARTIFACT_FORMAT_XML:
        candidates = _postprocessed_xml_candidates(
            output_artifact_text=output.artifact_text,
            raw_response_text=output.raw_response_text,
            tool_id=generated_tool_id,
            tool_name=generated_tool_name,
            include_toolsmith_citation=include_toolsmith_citation,
            toolsmith_citation_mode=toolsmith_citation_mode,
        )
        selected_candidate = _select_xml_candidate(candidates)
        if selected_candidate is not None:
            artifact_text = str(selected_candidate["artifact_text"])
            validation = dict(selected_candidate["validation"])
            postprocess_metadata = dict(selected_candidate["postprocess"])
            selected_candidate_index = int(selected_candidate["candidate_index"])
        else:
            postprocess = postprocess_generated_artifact(
                output.artifact_text,
                artifact_format=artifact_format,
                tool_id=generated_tool_id,
                tool_name=generated_tool_name,
                include_toolsmith_citation=include_toolsmith_citation,
                citation_mode=toolsmith_citation_mode,
            )
            artifact_text = postprocess.artifact_text
            validation = _validation_with_generation_diagnostics(
                artifact_text,
                artifact_format=artifact_format,
            )
            postprocess_metadata = postprocess.metadata
    else:
        postprocess = postprocess_generated_artifact(
            output.artifact_text,
            artifact_format=artifact_format,
            tool_id=generated_tool_id,
            tool_name=generated_tool_name,
            include_toolsmith_citation=include_toolsmith_citation,
            citation_mode=toolsmith_citation_mode,
        )
        artifact_text = postprocess.artifact_text
        validation = _validation_with_generation_diagnostics(
            artifact_text,
            artifact_format=artifact_format,
        )
        postprocess_metadata = postprocess.metadata
    if raw_response_log_path is not None and output.raw_response_text:
        _atomic_write_text(raw_response_log_path, output.raw_response_text)
    _atomic_write_text(output_path, artifact_text)
    if artifact_format == ARTIFACT_FORMAT_XML and selected_candidate_index:
        candidate_artifacts = _write_xml_candidate_artifacts(
            candidates,
            output_path=output_path,
            attempt=attempt_count,
            selected_candidate_index=selected_candidate_index,
        )
    sidecar_artifacts = (
        _write_sidecar_artifacts(
            output.sidecar_artifacts,
            output_path=output_path,
            sidecar_output_dir=sidecar_output_dir,
        )
        if generate_sidecars and artifact_format == ARTIFACT_FORMAT_XML
        else []
    )
    if (
        artifact_format == ARTIFACT_FORMAT_XML
        and include_toolsmith_citation
        and toolsmith_citation_mode == "macro"
    ):
        macros_path = write_toolsmith_macros_file(sidecar_output_dir or output_path.parent)
        if not any(item.get("path") == str(macros_path) for item in sidecar_artifacts):
            sidecar_artifacts.append(
                {
                    "path": str(macros_path),
                    "role": "macros",
                    "root_tag": "macros",
                    "bytes": macros_path.stat().st_size,
                }
            )

    scaffold_payload: dict = {"enabled": False}
    if datatype_scaffold:
        scaffold_target = datatype_scaffold_dir or datatype_scaffold_dir_for_output(output_path)
        scaffold_payload = write_datatype_scaffold(
            scaffold_target,
            [
                {
                    "tool_name": generated_tool_name,
                    "tool_id": generated_tool_id,
                    "output_path": str(output_path),
                    "output_xml_path": str(output_path)
                    if artifact_format == ARTIFACT_FORMAT_XML
                    else "",
                    "validation": validation,
                }
            ],
            repository_style=datatype_scaffold_repository_style,
        )
    request_id = f"gen-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    report_path = paths.runs_root / "generation" / request_id / "report.json"
    _atomic_write_text(report_path, json.dumps(validation, indent=2))

    return GenerationRecord(
        request_id=request_id,
        created_at=utc_now_iso(),
        tool_name=tool_name,
        tool_id=generated_tool_id,
        tool_display_name=generated_tool_name,
        provider=output.provider,
        model_variant=output.model_variant,
        skills_profile=skills_profile,
        report_path=str(report_path),
        artifact_format=artifact_format,
        output_path=str(output_path),
        output_xml_path=str(output_path) if artifact_format == ARTIFACT_FORMAT_XML else "",
        output_udt_yaml_path=str(output_path)
        if artifact_format == ARTIFACT_FORMAT_UDT_YAML
        else "",
        validation=validation,
        prompt_help={**shaped_help.to_dict(), "interface_hints": interface_hints.to_dict()},
        source_context=dict(source_context_summary or {}),
        raw_response_log_path=str(raw_response_log_path or ""),
        raw_response_chars=len(output.raw_response_text),
        generation_settings=_provider_generation_settings(
            provider,
            provider_name=provider_name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        sidecar_artifacts=sidecar_artifacts,
        candidate_artifacts=candidate_artifacts,
        selected_candidate_attempt=attempt_count if selected_candidate_index else 0,
        selected_candidate_index=selected_candidate_index,
        postprocess=postprocess_metadata,
        datatype_scaffold=scaffold_payload,
        attempt_count=attempt_count,
        repair_attempted=repair_attempted,
        repair_mode=repair_mode,
        repair_no_progress=repair_no_progress,
        repair_reason=repair_reason,
        initial_validation=dict(initial_validation or {}),
        initial_raw_response_log_path=initial_raw_response_log_path,
    )


def _get_provider(
    name: str,
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    paths: WorkspacePaths | None = None,
    allow_stub_local: bool = False,
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    ollama_context_tokens: int | None = None,
):
    if name == "local":
        return LocalProvider(
            paths=paths,
            model=model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            allow_stub=allow_stub_local,
            local_offload_policy=local_offload_policy,
            local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        )
    if name == "openai":
        return OpenAIProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if name == "anthropic":
        return AnthropicProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if name == "copilot":
        return CopilotProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    if name == "ollama":
        return OllamaProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_tokens=ollama_context_tokens,
        )
    raise ValueError("Unsupported provider. Use one of: local, openai, anthropic, copilot, ollama.")


def _provider_generation_settings(
    provider: object,
    *,
    provider_name: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "provider": provider_name,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "ollama_context_tokens": getattr(provider, "context_tokens", None),
    }


def generate_xml_from_content(
    tool_name: str,
    help_text: str,
    source_code: str,
    provider_name: str,
    model_variant: str,
    model: str,
    temperature: float,
    max_tokens: int,
    skills_profile: str = "default",
    paths: WorkspacePaths | None = None,
    allow_stub_local: bool = False,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    ollama_context_tokens: int | None = None,
    metadata_hints: Mapping[str, object] | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    tool_id: str = "",
    tool_display_name: str = "",
    include_toolsmith_citation: bool = True,
    toolsmith_citation_mode: str = "direct",
) -> dict:
    artifact_format = normalize_artifact_format(artifact_format)
    provider = _get_provider(
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
    shaped_help = shape_help_text(help_text, max_chars=max_prompt_help_chars)
    interface_hints = extract_interface_hints(help_text, metadata=metadata_hints)
    request = GenerationInput(
        tool_name=tool_name,
        help_text=shaped_help.text,
        source_code=source_code,
        model_variant=model_variant,
        skills_profile=skills_profile,
        interface_hints=interface_hints.text,
        artifact_format=artifact_format,
    )
    output = provider.generate_wrapper(request)
    generated_tool_id = safe_tool_id(tool_id or tool_name)
    generated_tool_name = str(tool_display_name or tool_name or generated_tool_id)
    candidate_artifacts: list[dict] = []
    selected_candidate_attempt = 0
    selected_candidate_index = 0
    if artifact_format == ARTIFACT_FORMAT_XML:
        candidates = _postprocessed_xml_candidates(
            output_artifact_text=output.artifact_text,
            raw_response_text=output.raw_response_text,
            tool_id=generated_tool_id,
            tool_name=generated_tool_name,
            include_toolsmith_citation=include_toolsmith_citation,
            toolsmith_citation_mode=toolsmith_citation_mode,
        )
        selected_candidate = _select_xml_candidate(candidates)
        if selected_candidate is not None:
            artifact_text = str(selected_candidate["artifact_text"])
            validation = dict(selected_candidate["validation"])
            postprocess_metadata = dict(selected_candidate["postprocess"])
            selected_candidate_attempt = 1
            selected_candidate_index = int(selected_candidate["candidate_index"])
            for candidate in candidates:
                index = int(candidate["candidate_index"])
                candidate_artifacts.append(
                    {
                        "attempt": 1,
                        "candidate_index": index,
                        "selected": index == selected_candidate_index,
                        "score": int(candidate.get("score", 0)),
                        "score_reasons": list(candidate.get("score_reasons") or []),
                        "features": dict(candidate.get("features") or {}),
                        "validation": dict(candidate.get("validation") or {}),
                        "postprocess": dict(candidate.get("postprocess") or {}),
                        "artifact_text": str(candidate.get("artifact_text") or ""),
                    }
                )
        else:
            postprocess = postprocess_generated_artifact(
                output.artifact_text,
                artifact_format=artifact_format,
                tool_id=generated_tool_id,
                tool_name=generated_tool_name,
                include_toolsmith_citation=include_toolsmith_citation,
                citation_mode=toolsmith_citation_mode,
            )
            artifact_text = postprocess.artifact_text
            validation = _validation_with_generation_diagnostics(
                artifact_text,
                artifact_format=artifact_format,
            )
            postprocess_metadata = postprocess.metadata
    else:
        postprocess = postprocess_generated_artifact(
            output.artifact_text,
            artifact_format=artifact_format,
            tool_id=generated_tool_id,
            tool_name=generated_tool_name,
            include_toolsmith_citation=include_toolsmith_citation,
            citation_mode=toolsmith_citation_mode,
        )
        artifact_text = postprocess.artifact_text
        validation = _validation_with_generation_diagnostics(
            artifact_text,
            artifact_format=artifact_format,
        )
        postprocess_metadata = postprocess.metadata
    result = {
        "tool_name": tool_name,
        "tool_id": generated_tool_id,
        "tool_display_name": generated_tool_name,
        "provider": output.provider,
        "model_variant": output.model_variant,
        "skills_profile": skills_profile,
        "artifact_format": artifact_format,
        "artifact_text": artifact_text,
        "validation": validation,
        "postprocess": postprocess_metadata,
        "candidate_artifacts": candidate_artifacts,
        "selected_candidate_attempt": selected_candidate_attempt,
        "selected_candidate_index": selected_candidate_index,
        "prompt_help": {**shaped_help.to_dict(), "interface_hints": interface_hints.to_dict()},
        "generation_settings": _provider_generation_settings(
            provider,
            provider_name=provider_name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    }
    if artifact_format == ARTIFACT_FORMAT_XML:
        result["xml_wrapper"] = artifact_text
    elif artifact_format == ARTIFACT_FORMAT_UDT_YAML:
        result["udt_yaml"] = artifact_text
    return result


def generate_wrapper(
    paths: WorkspacePaths,
    tool_name: str,
    help_text_path: Path | None,
    source_path: Path | None,
    output_path: Path,
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
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
    repair_invalid_xml: bool = True,
    raw_response_log_path: Path | None = None,
    stream_output: bool = False,
    generate_sidecars: bool = False,
    sidecar_output_dir: Path | None = None,
    tool_granularity: str = "auto",
    help_text: str | None = None,
    tool_id: str = "",
    tool_display_name: str = "",
    include_toolsmith_citation: bool = True,
    toolsmith_citation_mode: str = "direct",
    datatype_scaffold: bool = True,
    datatype_scaffold_dir: Path | None = None,
    datatype_scaffold_repository_style: bool = False,
) -> GenerationRecord:
    artifact_format = normalize_artifact_format(artifact_format)
    source_context = build_source_context_from_paths(
        settings=source_context_settings,
        source_file=source_path,
    )
    provider = _get_provider(
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
    if help_text is None:
        if help_text_path is None:
            raise ValueError("help_text_path is required when help_text is not provided.")
        help_text = _load_text(help_text_path)
    source_code = source_context.text
    if tool_granularity:
        source_code = "\n\n".join(
            part
            for part in (
                source_code,
                (
                    "Generation granularity hint:\n"
                    f"{tool_granularity}. If the software exposes separable commands or "
                    "subcommands, prefer focused Galaxy tools over one merged interface."
                ),
            )
            if part
        )
    first_raw_log = _attempt_raw_response_path(raw_response_log_path, 1)
    first = _generate_record(
        paths=paths,
        tool_name=tool_name,
        help_text=help_text,
        source_code=source_code,
        output_path=output_path,
        provider_name=provider_name,
        model_variant=model_variant,
        skills_profile=skills_profile,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_stub_local=allow_stub_local,
        provider_instance=provider,
        max_prompt_help_chars=max_prompt_help_chars,
        local_offload_policy=local_offload_policy,
        local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        artifact_format=artifact_format,
        source_context_summary=source_context.to_dict(),
        raw_response_log_path=first_raw_log,
        stream_output=stream_output,
        generate_sidecars=generate_sidecars,
        sidecar_output_dir=sidecar_output_dir,
        tool_id=tool_id,
        tool_display_name=tool_display_name,
        include_toolsmith_citation=include_toolsmith_citation,
        toolsmith_citation_mode=toolsmith_citation_mode,
        datatype_scaffold=datatype_scaffold,
        datatype_scaffold_dir=datatype_scaffold_dir,
        datatype_scaffold_repository_style=datatype_scaffold_repository_style,
    )
    if (
        artifact_format != ARTIFACT_FORMAT_XML
        or not repair_invalid_xml
        or _generation_validation_passed(first.validation, artifact_format)
    ):
        return first

    reason = _repair_reason(first.validation)
    repair_mode = _repair_mode_for_validation(first.validation)
    repair_source_code = _source_code_for_repair(source_code, repair_mode)
    second_raw_log = _attempt_raw_response_path(raw_response_log_path, 2)
    second = _generate_record(
        paths=paths,
        tool_name=tool_name,
        help_text=help_text,
        source_code=repair_source_code,
        output_path=output_path,
        provider_name=provider_name,
        model_variant=model_variant,
        skills_profile=skills_profile,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_stub_local=allow_stub_local,
        provider_instance=provider,
        max_prompt_help_chars=max_prompt_help_chars,
        local_offload_policy=local_offload_policy,
        local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        repair_context=_repair_context(tool_name, first.validation),
        attempt_count=2,
        repair_attempted=True,
        repair_mode=repair_mode,
        repair_reason=reason,
        artifact_format=artifact_format,
        source_context_summary=source_context.to_dict(),
        raw_response_log_path=second_raw_log,
        stream_output=stream_output,
        generate_sidecars=generate_sidecars,
        sidecar_output_dir=sidecar_output_dir,
        initial_validation=first.validation,
        initial_raw_response_log_path=first.raw_response_log_path,
        tool_id=tool_id,
        tool_display_name=tool_display_name,
        include_toolsmith_citation=include_toolsmith_citation,
        toolsmith_citation_mode=toolsmith_citation_mode,
        datatype_scaffold=datatype_scaffold,
        datatype_scaffold_dir=datatype_scaffold_dir,
        datatype_scaffold_repository_style=datatype_scaffold_repository_style,
    )
    return _merge_generation_attempt_records(first=first, second=second, output_path=output_path)


def generate_wrapper_from_content(
    paths: WorkspacePaths,
    tool_name: str,
    help_text: str,
    source_code: str,
    output_path: Path,
    provider_name: str,
    model_variant: str,
    model: str,
    temperature: float,
    max_tokens: int,
    skills_profile: str = "default",
    allow_stub_local: bool = False,
    provider_instance: Provider | None = None,
    max_prompt_help_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
    local_offload_policy: str = "allow",
    local_gpu_memory_reserve_gib: float = 2.0,
    ollama_context_tokens: int | None = None,
    repair_context: str = "",
    attempt_count: int = 1,
    repair_attempted: bool = False,
    repair_mode: str = "",
    repair_no_progress: bool = False,
    repair_reason: str = "",
    metadata_hints: Mapping[str, object] | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_summary: Mapping[str, object] | None = None,
    raw_response_log_path: Path | None = None,
    stream_output: bool = False,
    generate_sidecars: bool = False,
    sidecar_output_dir: Path | None = None,
    initial_validation: Mapping[str, object] | None = None,
    initial_raw_response_log_path: str = "",
    tool_id: str = "",
    tool_display_name: str = "",
    include_toolsmith_citation: bool = True,
    toolsmith_citation_mode: str = "direct",
    datatype_scaffold: bool = True,
    datatype_scaffold_dir: Path | None = None,
    datatype_scaffold_repository_style: bool = False,
) -> GenerationRecord:
    return _generate_record(
        paths=paths,
        tool_name=tool_name,
        help_text=help_text,
        source_code=source_code,
        output_path=output_path,
        provider_name=provider_name,
        model_variant=model_variant,
        skills_profile=skills_profile,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_stub_local=allow_stub_local,
        provider_instance=provider_instance,
        max_prompt_help_chars=max_prompt_help_chars,
        local_offload_policy=local_offload_policy,
        local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        ollama_context_tokens=ollama_context_tokens,
        repair_context=repair_context,
        attempt_count=attempt_count,
        repair_attempted=repair_attempted,
        repair_mode=repair_mode,
        repair_no_progress=repair_no_progress,
        repair_reason=repair_reason,
        metadata_hints=metadata_hints,
        artifact_format=artifact_format,
        source_context_summary=source_context_summary,
        raw_response_log_path=raw_response_log_path,
        stream_output=stream_output,
        generate_sidecars=generate_sidecars,
        sidecar_output_dir=sidecar_output_dir,
        initial_validation=initial_validation,
        initial_raw_response_log_path=initial_raw_response_log_path,
        tool_id=tool_id,
        tool_display_name=tool_display_name,
        include_toolsmith_citation=include_toolsmith_citation,
        toolsmith_citation_mode=toolsmith_citation_mode,
        datatype_scaffold=datatype_scaffold,
        datatype_scaffold_dir=datatype_scaffold_dir,
        datatype_scaffold_repository_style=datatype_scaffold_repository_style,
    )


def _validation_with_generation_diagnostics(
    artifact_text: str,
    *,
    artifact_format: str = ARTIFACT_FORMAT_XML,
) -> dict:
    artifact_format = normalize_artifact_format(artifact_format)
    if artifact_format == ARTIFACT_FORMAT_UDT_YAML:
        validation = validate_udt_yaml(artifact_text, check_conversion=True).to_dict()
        validation["artifact_format"] = artifact_format
        return validation

    validation = validate_wrapper(artifact_text).to_dict()
    validation["artifact_format"] = artifact_format
    diagnostics = diagnose_generated_xml(artifact_text)
    if diagnostics.has_problems:
        validation["notes"] = [*validation.get("notes", []), *diagnostics.problems]
        validation["generation_diagnostics"] = diagnostics.to_dict()
    return validation
