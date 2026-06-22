from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml
from jsonschema import Draft202012Validator

from galaxy_toolsmith.providers.base import strip_markdown_fences

UDT_SCHEMA_RESOURCE = "schemas/user_tool_source.schema.json"
UDT_SCHEMA_URI = "https://schema.galaxyproject.org/customTool.json"
_UDT_EXPRESSION_RE = re.compile(r"\$\((.*?)\)", re.DOTALL)
_SIMPLE_REFERENCE_RE = re.compile(r"^(inputs|outputs)\.([A-Za-z_][A-Za-z0-9_]*)(?:\.path)?$")
_SUPPORTED_INPUT_TYPES = {
    "boolean",
    "color",
    "data",
    "data_collection",
    "float",
    "integer",
    "select",
    "text",
}
_SUPPORTED_OUTPUT_TYPES = {"boolean", "data", "float", "integer", "text"}


@dataclass(frozen=True)
class UdtValidationReport:
    yaml_well_formed: bool
    schema_valid: bool
    root_class: str
    root_is_user_tool: bool
    missing_required: list[str]
    schema_errors: list[dict[str, str]] = field(default_factory=list)
    conversion_supported: bool = True
    artifact_valid: bool = False
    xsd_status: str = "not_run"
    planemo_status: str = "not_run"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UdtConversionResult:
    xml: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UdtConversionError(RuntimeError):
    """Raised when UDT YAML cannot be converted conservatively to Galaxy XML."""


def schema_path() -> Path:
    return Path(str(files("galaxy_toolsmith").joinpath(UDT_SCHEMA_RESOURCE)))


@lru_cache(maxsize=1)
def load_udt_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = load_udt_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def extract_udt_yaml(text: str) -> str:
    value = strip_markdown_fences(text)
    if _is_user_tool_document(value):
        return value.strip()
    match = re.search(r"(?m)^class:\s*GalaxyUserTool\s*$", value)
    if match:
        candidate = value[match.start() :].strip()
        if _is_user_tool_document(candidate):
            return candidate
        return candidate
    return value.strip()


def validate_udt_yaml(udt_yaml: str, *, check_conversion: bool = False) -> UdtValidationReport:
    try:
        parsed = yaml.safe_load(udt_yaml)
    except yaml.YAMLError as error:
        return UdtValidationReport(
            yaml_well_formed=False,
            schema_valid=False,
            root_class="",
            root_is_user_tool=False,
            missing_required=[],
            conversion_supported=False,
            artifact_valid=False,
            notes=[f"YAML parse error: {error}"],
        )

    if not isinstance(parsed, dict):
        return UdtValidationReport(
            yaml_well_formed=True,
            schema_valid=False,
            root_class="",
            root_is_user_tool=False,
            missing_required=[],
            conversion_supported=False,
            artifact_valid=False,
            notes=["Expected UDT YAML root to be a mapping."],
        )

    schema = load_udt_schema()
    required = [str(item) for item in schema.get("required", [])]
    missing_required = [
        key for key in required if key not in parsed or parsed.get(key) in (None, "")
    ]
    root_class = str(parsed.get("class", "") or "")
    root_is_user_tool = root_class == "GalaxyUserTool"
    schema_errors = [_format_schema_error(error) for error in _schema_validator().iter_errors(parsed)]
    schema_errors.sort(key=lambda item: item["path"])
    notes = [f"schema: {item['path']}: {item['message']}" for item in schema_errors[:20]]
    if len(schema_errors) > 20:
        notes.append(f"schema: {len(schema_errors) - 20} additional validation errors omitted.")

    conversion_notes: list[str] = []
    if check_conversion:
        conversion_notes = conversion_warnings(parsed)
        notes.extend(conversion_notes)

    schema_valid = not schema_errors
    artifact_valid = schema_valid and root_is_user_tool
    return UdtValidationReport(
        yaml_well_formed=True,
        schema_valid=schema_valid,
        root_class=root_class,
        root_is_user_tool=root_is_user_tool,
        missing_required=missing_required,
        schema_errors=schema_errors,
        conversion_supported=not conversion_notes,
        artifact_valid=artifact_valid,
        notes=notes,
    )


def udt_yaml_to_tool_xml(udt_yaml: str, *, allow_lossy_conversion: bool = False) -> UdtConversionResult:
    data = _parse_valid_udt(udt_yaml)
    notes: list[str] = []
    command = _convert_template_expressions(
        str(data.get("shell_command", "")),
        notes=notes,
        allow_lossy_conversion=allow_lossy_conversion,
    )

    tool_id = _xml_id(str(data.get("id", "") or data.get("name", "") or "user_defined_tool"))
    name = str(data.get("name", "") or tool_id)
    version = str(data.get("version", "") or "0.1.0")
    description = str(data.get("description", "") or "").strip()

    lines = [
        f'<tool id="{_xml_attr(tool_id)}" name="{_xml_attr(name)}" version="{_xml_attr(version)}" profile="25.0">',
    ]
    if description:
        lines.append(f"  <description>{escape(description)}</description>")
    lines.extend(_requirements_xml(data, notes=notes))
    lines.extend(_configfiles_xml(data, notes=notes, allow_lossy_conversion=allow_lossy_conversion))
    lines.append('  <command detect_errors="aggressive"><![CDATA[')
    lines.append(command.replace("]]>", "]] >"))
    lines.append("  ]]></command>")
    lines.append("  <inputs>")
    for item in data.get("inputs") or []:
        lines.extend(_input_xml(item, notes=notes))
    lines.append("  </inputs>")
    lines.append("  <outputs>")
    for item in data.get("outputs") or []:
        lines.extend(_output_xml(item, notes=notes))
    lines.append("  </outputs>")
    test_lines = _tests_xml(data.get("tests") or [], notes=notes)
    if test_lines:
        lines.append("  <tests>")
        lines.extend(test_lines)
        lines.append("  </tests>")
    help_text = _help_text(data)
    if help_text:
        lines.append('  <help format="markdown"><![CDATA[')
        lines.append(help_text.replace("]]>", "]] >"))
        lines.append("  ]]></help>")
    citation_lines = _citations_xml(data.get("citations") or [])
    if citation_lines:
        lines.append("  <citations>")
        lines.extend(citation_lines)
        lines.append("  </citations>")
    lines.append("</tool>")
    return UdtConversionResult(xml="\n".join(lines), notes=sorted(set(notes)))


def conversion_warnings(data: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    try:
        _convert_template_expressions(
            str(data.get("shell_command", "")),
            notes=notes,
            allow_lossy_conversion=False,
        )
    except UdtConversionError as error:
        notes.append(str(error))
    for item in data.get("inputs") or []:
        if isinstance(item, dict):
            input_type = str(item.get("type", "") or "")
            if input_type and input_type not in _SUPPORTED_INPUT_TYPES:
                notes.append(f"converter: unsupported input type '{input_type}' for {item.get('name', '')}.")
    for item in data.get("outputs") or []:
        if isinstance(item, dict):
            output_type = str(item.get("type", "data") or "data")
            if output_type and output_type not in _SUPPORTED_OUTPUT_TYPES:
                notes.append(
                    f"converter: unsupported output type '{output_type}' for {item.get('name', '')}."
                )
    return sorted(set(note for note in notes if note))


def udt_structural_report(udt_yaml: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(udt_yaml)
    except yaml.YAMLError:
        data = None
    if not isinstance(data, dict):
        return {
            "input_count": 0,
            "output_count": 0,
            "test_count": 0,
            "has_help": False,
            "has_command": False,
            "has_citations": False,
            "structural_score": 0.0,
        }
    input_count = len(data.get("inputs") or [])
    output_count = len(data.get("outputs") or [])
    test_count = len(data.get("tests") or [])
    has_help = bool(data.get("help"))
    has_command = bool(str(data.get("shell_command", "")).strip())
    has_citations = bool(data.get("citations"))
    score = 0.0
    score += 0.25 if input_count > 0 else 0.0
    score += 0.25 if output_count > 0 else 0.0
    score += 0.20 if test_count > 0 else 0.0
    score += 0.15 if has_help else 0.0
    score += 0.10 if has_command else 0.0
    score += 0.05 if has_citations else 0.0
    return {
        "input_count": input_count,
        "output_count": output_count,
        "test_count": test_count,
        "has_help": has_help,
        "has_command": has_command,
        "has_citations": has_citations,
        "structural_score": round(score, 4),
    }


def _is_user_tool_document(value: str) -> bool:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return False
    return isinstance(parsed, dict) and parsed.get("class") == "GalaxyUserTool"


def _parse_valid_udt(udt_yaml: str) -> dict[str, Any]:
    report = validate_udt_yaml(udt_yaml)
    if not report.artifact_valid:
        detail = "; ".join(report.notes) or "UDT YAML failed schema validation."
        raise UdtConversionError(detail)
    parsed = yaml.safe_load(udt_yaml)
    if not isinstance(parsed, dict):
        raise UdtConversionError("UDT YAML root must be a mapping.")
    return parsed


def _format_schema_error(error: Any) -> dict[str, str]:
    path = ".".join(str(item) for item in error.absolute_path)
    return {"path": path or "$", "message": str(error.message)}


def _convert_template_expressions(
    text: str,
    *,
    notes: list[str],
    allow_lossy_conversion: bool,
) -> str:
    unsupported: list[str] = []

    def replace(match: re.Match[str]) -> str:
        expression = " ".join(match.group(1).strip().split())
        simple = _SIMPLE_REFERENCE_RE.match(expression)
        if simple:
            return f"${simple.group(2)}"
        unsupported.append(expression)
        return match.group(0)

    converted = _UDT_EXPRESSION_RE.sub(replace, text)
    if unsupported:
        detail = ", ".join(sorted(set(unsupported)))
        message = f"converter: unsupported UDT expression(s): {detail}"
        notes.append(message)
        if not allow_lossy_conversion:
            raise UdtConversionError(message)
    return converted


def _requirements_xml(data: dict[str, Any], *, notes: list[str]) -> list[str]:
    container = str(data.get("container", "") or "").strip()
    if not container:
        return []
    requirements = ["  <requirements>", f'    <container type="docker">{escape(container)}</container>']
    for item in data.get("requirements") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "container":
            notes.append("converter: ignored extra UDT container requirement; top-level container was used.")
        elif item.get("type"):
            notes.append(f"converter: ignored UDT requirement type '{item.get('type')}'.")
    requirements.append("  </requirements>")
    return requirements


def _configfiles_xml(
    data: dict[str, Any],
    *,
    notes: list[str],
    allow_lossy_conversion: bool,
) -> list[str]:
    configfiles = data.get("configfiles") or []
    if not configfiles:
        return []
    lines = ["  <configfiles>"]
    for index, configfile in enumerate(configfiles, start=1):
        if not isinstance(configfile, dict):
            notes.append("converter: ignored non-object configfile entry.")
            continue
        name = str(configfile.get("name") or configfile.get("filename") or f"config_{index}")
        content = _convert_template_expressions(
            str(configfile.get("content", "")),
            notes=notes,
            allow_lossy_conversion=allow_lossy_conversion,
        )
        lines.append(f'    <configfile name="{_xml_attr(name)}"><![CDATA[')
        lines.append(content.replace("]]>", "]] >"))
        lines.append("    ]]></configfile>")
    lines.append("  </configfiles>")
    return lines


def _input_xml(item: Any, *, notes: list[str]) -> list[str]:
    if not isinstance(item, dict):
        notes.append("converter: ignored non-object input entry.")
        return []
    input_type = str(item.get("type", "") or "").strip()
    if input_type not in _SUPPORTED_INPUT_TYPES:
        raise UdtConversionError(f"converter: unsupported input type '{input_type}' for {item.get('name', '')}.")
    name = str(item.get("name", "") or "").strip()
    if not name:
        raise UdtConversionError("converter: input entry is missing name.")
    attrs = {
        "name": name,
        "type": "data_collection" if input_type == "data_collection" else input_type,
    }
    if input_type in {"data", "data_collection"}:
        fmt = item.get("format") or ",".join(str(value) for value in item.get("extensions", []) or [])
        if fmt:
            attrs["format"] = str(fmt)
        if item.get("collection_type"):
            attrs["collection_type"] = str(item["collection_type"])
    if input_type == "select" and item.get("multiple") is True:
        attrs["multiple"] = "true"
    elif input_type == "data" and item.get("multiple") is True:
        attrs["multiple"] = "true"
    if item.get("optional") is True:
        attrs["optional"] = "true"
    for key in ("label", "help", "value", "truevalue", "falsevalue"):
        if item.get(key) not in (None, ""):
            attrs[key] = str(item[key])
    if input_type == "select" and item.get("options"):
        lines = [f"    <param{_attrs(attrs)}>"]
        for option in item.get("options") or []:
            if isinstance(option, dict):
                value = str(option.get("value", option.get("label", "")))
                label = str(option.get("label", value))
            else:
                value = str(option)
                label = value
            lines.append(f'      <option value="{_xml_attr(value)}">{escape(label)}</option>')
        lines.append("    </param>")
        return lines
    return [f"    <param{_attrs(attrs)}/>"]


def _output_xml(item: Any, *, notes: list[str]) -> list[str]:
    if not isinstance(item, dict):
        notes.append("converter: ignored non-object output entry.")
        return []
    output_type = str(item.get("type", "data") or "data")
    if output_type not in _SUPPORTED_OUTPUT_TYPES:
        raise UdtConversionError(f"converter: unsupported output type '{output_type}' for {item.get('name', '')}.")
    name = str(item.get("name", "") or "").strip()
    if not name:
        raise UdtConversionError("converter: output entry is missing name.")
    attrs = {"name": name}
    if output_type == "data":
        if item.get("format"):
            attrs["format"] = str(item["format"])
        if item.get("format_source"):
            attrs["format_source"] = str(item["format_source"])
    else:
        attrs["format"] = "txt"
        notes.append(f"converter: represented scalar UDT output '{name}' as data format txt.")
    for key in ("from_work_dir", "label"):
        if item.get(key) not in (None, ""):
            attrs[key] = str(item[key])
    if item.get("hidden") is True:
        attrs["hidden"] = "true"
    return [f"    <data{_attrs(attrs)}/>"]


def _tests_xml(tests: list[Any], *, notes: list[str]) -> list[str]:
    lines: list[str] = []
    for test in tests:
        if not isinstance(test, dict):
            notes.append("converter: ignored non-object test entry.")
            continue
        lines.append('    <test expect_num_outputs="1">')
        for name, value in (test.get("inputs") or {}).items():
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            lines.append(f'      <param name="{_xml_attr(str(name))}" value="{_xml_attr(str(value))}"/>')
        for name, value in (test.get("outputs") or {}).items():
            lines.extend(_test_output_xml(str(name), value, notes=notes))
        lines.append("    </test>")
    return lines


def _test_output_xml(name: str, value: Any, *, notes: list[str]) -> list[str]:
    if isinstance(value, str):
        return [f'      <output name="{_xml_attr(name)}" file="{_xml_attr(value)}"/>']
    if not isinstance(value, dict):
        notes.append(f"converter: ignored unsupported test output for '{name}'.")
        return [f'      <output name="{_xml_attr(name)}"/>']
    attrs = {"name": name}
    for key in ("file", "path", "ftype", "compare", "checksum"):
        if value.get(key) not in (None, ""):
            attrs[key] = str(value[key])
    asserts = value.get("asserts")
    if not asserts:
        return [f"      <output{_attrs(attrs)}/>"]
    lines = [f"      <output{_attrs(attrs)}>", "        <assert_contents>"]
    assert_items = asserts if isinstance(asserts, list) else [asserts]
    for item in assert_items:
        if not isinstance(item, dict):
            continue
        for assertion_name, assertion_value in item.items():
            if assertion_name not in {"has_line", "has_text"}:
                notes.append(f"converter: ignored test assertion '{assertion_name}'.")
                continue
            if isinstance(assertion_value, dict):
                attr_name = "line" if assertion_name == "has_line" else "text"
                text = str(assertion_value.get(attr_name, "") or assertion_value.get("text", ""))
            else:
                attr_name = "text"
                text = str(assertion_value)
            lines.append(f'          <{assertion_name} {attr_name}="{_xml_attr(text)}"/>')
    lines.extend(["        </assert_contents>", "      </output>"])
    return lines


def _citations_xml(citations: list[Any]) -> list[str]:
    lines: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        citation_type = str(citation.get("type", "") or "").strip()
        content = str(citation.get("content", "") or "").strip()
        if citation_type and content:
            lines.append(f'    <citation type="{_xml_attr(citation_type)}">{escape(content)}</citation>')
    return lines


def _help_text(data: dict[str, Any]) -> str:
    help_value = data.get("help")
    if isinstance(help_value, str):
        return help_value.strip()
    if isinstance(help_value, dict):
        for key in ("content", "text", "markdown"):
            if help_value.get(key):
                return str(help_value[key]).strip()
    return str(data.get("description", "") or "").strip()


def _xml_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")
    if not cleaned:
        cleaned = "user_defined_tool"
    if not re.match(r"^[A-Za-z_]", cleaned):
        cleaned = f"tool_{cleaned}"
    return cleaned[:255].rstrip("_") or "user_defined_tool"


def _xml_attr(value: str) -> str:
    return escape(str(value), {'"': "&quot;"})


def _attrs(values: dict[str, str]) -> str:
    return "".join(f' {key}="{_xml_attr(value)}"' for key, value in values.items() if value != "")
