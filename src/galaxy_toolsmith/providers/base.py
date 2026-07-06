from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree as ET

from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    normalize_artifact_format,
    prompt_task_for_artifact_format,
)


@dataclass(frozen=True)
class GenerationInput:
    tool_name: str
    help_text: str
    source_code: str
    model_variant: str
    skills_profile: str = "default"
    repair_context: str = ""
    interface_hints: str = ""
    artifact_format: str = ARTIFACT_FORMAT_XML
    raw_response_log_path: str = ""
    stream_output: bool = False
    generate_sidecars: bool = False


@dataclass(frozen=True)
class GenerationOutput:
    xml_wrapper: str = ""
    provider: str = ""
    model_variant: str = ""
    artifact_text: str = ""
    artifact_format: str = ARTIFACT_FORMAT_XML
    raw_response_text: str = ""
    sidecar_artifacts: tuple[dict, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        artifact_format = normalize_artifact_format(self.artifact_format)
        artifact_text = self.artifact_text or self.xml_wrapper
        xml_wrapper = self.xml_wrapper
        if artifact_format == ARTIFACT_FORMAT_XML and not xml_wrapper:
            xml_wrapper = artifact_text
        object.__setattr__(self, "artifact_format", artifact_format)
        object.__setattr__(self, "artifact_text", artifact_text)
        object.__setattr__(self, "xml_wrapper", xml_wrapper)


class Provider(Protocol):
    name: str

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        ...


def strip_markdown_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1]
    if value.endswith("```"):
        value = value.rsplit("```", 1)[0]
    return value.strip()


def write_raw_response_log(request: GenerationInput, text: str) -> None:
    if not request.raw_response_log_path:
        return
    path = Path(request.raw_response_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_TOOL_START_RE = re.compile(r"<tool(?:\s|>|/)", re.IGNORECASE)


def extract_complete_tool_xml_candidates(text: str) -> tuple[str, ...]:
    value = strip_markdown_fences(text)
    candidates: list[str] = []
    position = 0
    while True:
        start_match = _TOOL_START_RE.search(value, position)
        if start_match is None:
            break
        start = start_match.start()
        end = value.find("</tool>", start)
        if end == -1:
            break
        candidate = value[start : end + len("</tool>")].strip()
        if candidate:
            candidates.append(candidate)
        position = end + len("</tool>")
    return tuple(candidates)


def extract_complete_tool_xml(text: str) -> str:
    value = strip_markdown_fences(text)
    candidates = extract_complete_tool_xml_candidates(value)
    if candidates:
        return candidates[0]
    start_match = _TOOL_START_RE.search(value)
    if start_match is None:
        return value
    start = start_match.start()
    end = value.find("</tool>", start)
    if end == -1:
        return value[start:].strip()
    return value[start : end + len("</tool>")].strip()


_SIDECAR_XML_RE = re.compile(
    r"<(?P<root>macros|tables)\b(?P<body>.*?)</(?P=root)>",
    re.IGNORECASE | re.DOTALL,
)


def extract_xml_sidecar_artifacts(text: str) -> tuple[dict, ...]:
    value = strip_markdown_fences(text)
    tool_xml = extract_complete_tool_xml(value)
    if not tool_xml.startswith("<tool"):
        return ()
    tool_start = value.find(tool_xml)
    tool_end = tool_start + len(tool_xml) if tool_start >= 0 else -1
    sidecars: list[dict] = []
    seen_ranges: set[tuple[int, int]] = set()
    for match in _SIDECAR_XML_RE.finditer(value):
        block = match.group(0).strip()
        if block == tool_xml:
            continue
        span = match.span()
        if tool_start >= 0 and span[0] >= tool_start and span[1] <= tool_end:
            continue
        if span in seen_ranges:
            continue
        seen_ranges.add(span)
        root = _xml_root_tag(block)
        if root not in {"macros", "tables"}:
            continue
        sidecars.append(
            {
                "role": "macros" if root == "macros" else "tool_data_table_conf",
                "root_tag": root,
                "content": block,
            }
        )
    return tuple(sidecars)


def _xml_root_tag(text: str) -> str:
    try:
        return str(ET.fromstring(text).tag)
    except ET.ParseError:
        return ""


def extract_generated_artifact(text: str, artifact_format: str) -> str:
    normalized = normalize_artifact_format(artifact_format)
    if normalized == ARTIFACT_FORMAT_UDT_YAML:
        from galaxy_toolsmith.inference.udt import extract_udt_yaml

        return extract_udt_yaml(text)
    return extract_complete_tool_xml(text)


def generation_output_from_response(
    *,
    response_text: str,
    request: GenerationInput,
    provider: str,
) -> GenerationOutput:
    write_raw_response_log(request, response_text)
    artifact = extract_generated_artifact(response_text, request.artifact_format)
    sidecars = (
        extract_xml_sidecar_artifacts(response_text)
        if normalize_artifact_format(request.artifact_format) == ARTIFACT_FORMAT_XML
        else ()
    )
    return GenerationOutput(
        artifact_text=artifact,
        artifact_format=request.artifact_format,
        provider=provider,
        model_variant=request.model_variant,
        raw_response_text=response_text,
        sidecar_artifacts=sidecars,
    )


def generation_prompt_task(request: GenerationInput) -> str:
    return prompt_task_for_artifact_format(request.artifact_format)
