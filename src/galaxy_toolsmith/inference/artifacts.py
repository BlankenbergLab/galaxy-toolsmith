from __future__ import annotations

from pathlib import Path

ARTIFACT_FORMAT_XML = "xml"
ARTIFACT_FORMAT_UDT_YAML = "udt_yaml"
ARTIFACT_FORMATS = {ARTIFACT_FORMAT_XML, ARTIFACT_FORMAT_UDT_YAML}
TRAINING_ARTIFACT_FORMAT_MIXED = "mixed"
TRAINING_ARTIFACT_FORMATS = {
    ARTIFACT_FORMAT_XML,
    ARTIFACT_FORMAT_UDT_YAML,
    TRAINING_ARTIFACT_FORMAT_MIXED,
}


def normalize_artifact_format(value: str | None) -> str:
    text = str(value or ARTIFACT_FORMAT_XML).strip().lower().replace("-", "_")
    aliases = {
        "": ARTIFACT_FORMAT_XML,
        "xml": ARTIFACT_FORMAT_XML,
        "galaxy_xml": ARTIFACT_FORMAT_XML,
        "udt": ARTIFACT_FORMAT_UDT_YAML,
        "udt_yaml": ARTIFACT_FORMAT_UDT_YAML,
        "udt_yml": ARTIFACT_FORMAT_UDT_YAML,
        "user_defined_tool": ARTIFACT_FORMAT_UDT_YAML,
        "user_defined_tool_yaml": ARTIFACT_FORMAT_UDT_YAML,
    }
    normalized = aliases.get(text, text)
    if normalized not in ARTIFACT_FORMATS:
        choices = ", ".join(sorted(format_cli_value(item) for item in ARTIFACT_FORMATS))
        raise ValueError(f"Unsupported artifact format '{value}'. Expected one of: {choices}.")
    return normalized


def normalize_training_artifact_format(value: str | None) -> str:
    text = str(value or ARTIFACT_FORMAT_XML).strip().lower().replace("-", "_")
    if text == TRAINING_ARTIFACT_FORMAT_MIXED:
        return TRAINING_ARTIFACT_FORMAT_MIXED
    normalized = normalize_artifact_format(text)
    if normalized not in TRAINING_ARTIFACT_FORMATS:
        choices = ", ".join(sorted(format_cli_value(item) for item in TRAINING_ARTIFACT_FORMATS))
        raise ValueError(f"Unsupported training artifact format '{value}'. Expected one of: {choices}.")
    return normalized


def format_cli_value(artifact_format: str) -> str:
    return str(artifact_format).replace("_", "-")


def prompt_task_for_artifact_format(artifact_format: str) -> str:
    normalized = normalize_artifact_format(artifact_format)
    if normalized == ARTIFACT_FORMAT_UDT_YAML:
        return "udt_yaml_generate"
    return "xml_generate"


def output_key_for_artifact_format(artifact_format: str) -> str:
    normalized = normalize_artifact_format(artifact_format)
    if normalized == ARTIFACT_FORMAT_UDT_YAML:
        return "output_udt_yaml_path"
    return "output_xml_path"


def output_suffix_for_artifact_format(artifact_format: str) -> str:
    normalized = normalize_artifact_format(artifact_format)
    if normalized == ARTIFACT_FORMAT_UDT_YAML:
        return ".yml"
    return ".xml"


def output_path_from_record(record: dict, artifact_format: str) -> Path | None:
    for key in ("output_path", output_key_for_artifact_format(artifact_format), "output_xml_path"):
        value = str(record.get(key, "") or "").strip()
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path
    return None
