from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    normalize_artifact_format,
)
from galaxy_toolsmith.inference.output_diagnostics import diagnose_generated_xml
from galaxy_toolsmith.inference.prompt_context import (
    DEFAULT_MAX_PROMPT_HELP_CHARS,
    extract_interface_hints,
    shape_help_text,
)
from galaxy_toolsmith.inference.source_context import (
    SourceContextSettings,
    build_source_context_from_paths,
)
from galaxy_toolsmith.inference.udt import validate_udt_yaml
from galaxy_toolsmith.inference.validation import validate_wrapper
from galaxy_toolsmith.providers.base import GenerationInput, Provider
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
    artifact_format: str = ARTIFACT_FORMAT_XML
    output_path: str = ""
    output_udt_yaml_path: str = ""
    validation: dict = field(default_factory=dict)
    prompt_help: dict = field(default_factory=dict)
    source_context: dict = field(default_factory=dict)
    attempt_count: int = 1
    repair_attempted: bool = False
    repair_reason: str = ""

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
    repair_context: str = "",
    attempt_count: int = 1,
    repair_attempted: bool = False,
    repair_reason: str = "",
    metadata_hints: Mapping[str, object] | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_summary: Mapping[str, object] | None = None,
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
    )
    output = provider.generate_wrapper(request)
    artifact_text = output.artifact_text
    _atomic_write_text(output_path, artifact_text)

    validation = _validation_with_generation_diagnostics(
        artifact_text,
        artifact_format=artifact_format,
    )
    request_id = f"gen-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    report_path = paths.runs_root / "generation" / request_id / "report.json"
    _atomic_write_text(report_path, json.dumps(validation, indent=2))

    return GenerationRecord(
        request_id=request_id,
        created_at=utc_now_iso(),
        tool_name=tool_name,
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
        attempt_count=attempt_count,
        repair_attempted=repair_attempted,
        repair_reason=repair_reason,
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
        return OllamaProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    raise ValueError("Unsupported provider. Use one of: local, openai, anthropic, copilot, ollama.")


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
    metadata_hints: Mapping[str, object] | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
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
    artifact_text = output.artifact_text
    validation = _validation_with_generation_diagnostics(
        artifact_text,
        artifact_format=artifact_format,
    )
    result = {
        "tool_name": tool_name,
        "provider": output.provider,
        "model_variant": output.model_variant,
        "skills_profile": skills_profile,
        "artifact_format": artifact_format,
        "artifact_text": artifact_text,
        "validation": validation,
        "prompt_help": {**shaped_help.to_dict(), "interface_hints": interface_hints.to_dict()},
    }
    if artifact_format == ARTIFACT_FORMAT_XML:
        result["xml_wrapper"] = output.xml_wrapper
    elif artifact_format == ARTIFACT_FORMAT_UDT_YAML:
        result["udt_yaml"] = artifact_text
    return result


def generate_wrapper(
    paths: WorkspacePaths,
    tool_name: str,
    help_text_path: Path,
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
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
) -> GenerationRecord:
    source_context = build_source_context_from_paths(
        settings=source_context_settings,
        source_file=source_path,
    )
    return _generate_record(
        paths=paths,
        tool_name=tool_name,
        help_text=_load_text(help_text_path),
        source_code=source_context.text,
        output_path=output_path,
        provider_name=provider_name,
        model_variant=model_variant,
        skills_profile=skills_profile,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_stub_local=allow_stub_local,
        max_prompt_help_chars=max_prompt_help_chars,
        local_offload_policy=local_offload_policy,
        local_gpu_memory_reserve_gib=local_gpu_memory_reserve_gib,
        artifact_format=artifact_format,
        source_context_summary=source_context.to_dict(),
    )


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
    repair_context: str = "",
    attempt_count: int = 1,
    repair_attempted: bool = False,
    repair_reason: str = "",
    metadata_hints: Mapping[str, object] | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_summary: Mapping[str, object] | None = None,
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
        repair_context=repair_context,
        attempt_count=attempt_count,
        repair_attempted=repair_attempted,
        repair_reason=repair_reason,
        metadata_hints=metadata_hints,
        artifact_format=artifact_format,
        source_context_summary=source_context_summary,
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
