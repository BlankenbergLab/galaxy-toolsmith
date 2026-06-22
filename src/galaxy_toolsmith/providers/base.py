from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_XML,
    ARTIFACT_FORMAT_UDT_YAML,
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


@dataclass(frozen=True)
class GenerationOutput:
    xml_wrapper: str = ""
    provider: str = ""
    model_variant: str = ""
    artifact_text: str = ""
    artifact_format: str = ARTIFACT_FORMAT_XML

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


def extract_complete_tool_xml(text: str) -> str:
    value = strip_markdown_fences(text)
    start = value.find("<tool")
    if start == -1:
        return value
    end = value.find("</tool>", start)
    if end == -1:
        return value[start:].strip()
    return value[start : end + len("</tool>")].strip()


def extract_generated_artifact(text: str, artifact_format: str) -> str:
    normalized = normalize_artifact_format(artifact_format)
    if normalized == ARTIFACT_FORMAT_UDT_YAML:
        from galaxy_toolsmith.inference.udt import extract_udt_yaml

        return extract_udt_yaml(text)
    return extract_complete_tool_xml(text)


def generation_prompt_task(request: GenerationInput) -> str:
    return prompt_task_for_artifact_format(request.artifact_format)
