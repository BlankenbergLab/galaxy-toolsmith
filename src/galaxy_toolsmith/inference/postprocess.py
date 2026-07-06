from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml

from galaxy_toolsmith.inference.artifacts import ARTIFACT_FORMAT_UDT_YAML, ARTIFACT_FORMAT_XML
from galaxy_toolsmith.inference.repository import write_gtsm_json

TOOLSMITH_CITATION_URL = "https://github.com/BlankenbergLab/galaxy-toolsmith"
TOOLSMITH_CITATION_MACRO_NAME = "gtsm_toolsmith_citation"
TOOLSMITH_CITATION_BIBTEX = (
    "@misc{galaxy_toolsmith,\n"
    "  title = {Galaxy Toolsmith},\n"
    f"  howpublished = {{\\url{{{TOOLSMITH_CITATION_URL}}}}},\n"
    f"  url = {{{TOOLSMITH_CITATION_URL}}}\n"
    "}"
)


@dataclass
class PostprocessResult:
    artifact_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def postprocess_generated_artifact(
    artifact_text: str,
    *,
    artifact_format: str,
    tool_id: str = "",
    tool_name: str = "",
    include_toolsmith_citation: bool = True,
    citation_mode: str = "direct",
) -> PostprocessResult:
    if artifact_format == ARTIFACT_FORMAT_UDT_YAML:
        return postprocess_udt_yaml(
            artifact_text,
            tool_id=tool_id,
            tool_name=tool_name,
            include_toolsmith_citation=include_toolsmith_citation,
        )
    if artifact_format == ARTIFACT_FORMAT_XML:
        return postprocess_xml_tool(
            artifact_text,
            tool_id=tool_id,
            tool_name=tool_name,
            include_toolsmith_citation=include_toolsmith_citation,
            citation_mode=citation_mode,
        )
    return PostprocessResult(artifact_text, {"artifact_format": artifact_format, "changed": False})


def postprocess_xml_tool(
    xml_text: str,
    *,
    tool_id: str = "",
    tool_name: str = "",
    include_toolsmith_citation: bool = True,
    citation_mode: str = "direct",
) -> PostprocessResult:
    metadata: dict[str, Any] = {
        "artifact_format": ARTIFACT_FORMAT_XML,
        "citation_mode": citation_mode,
        "citation_added": False,
        "tool_id_applied": False,
        "tool_name_applied": False,
        "macros_import_added": False,
        "tests_pruned": False,
        "test_count_before": 0,
        "test_count_after": 0,
        "changed": False,
    }
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        metadata["error"] = f"XML parse error: {error}"
        return PostprocessResult(xml_text, metadata)
    if root.tag != "tool":
        metadata["skipped"] = f"Expected root <tool>; found <{root.tag}>."
        return PostprocessResult(xml_text, metadata)

    changed = False
    if tool_id and root.attrib.get("id") != tool_id:
        root.set("id", tool_id)
        metadata["tool_id_applied"] = True
        changed = True
    if tool_name and root.attrib.get("name") != tool_name:
        root.set("name", tool_name)
        metadata["tool_name_applied"] = True
        changed = True

    tests_pruned, test_count_before, test_count_after = prune_extra_generated_tests(root)
    metadata["tests_pruned"] = tests_pruned
    metadata["test_count_before"] = test_count_before
    metadata["test_count_after"] = test_count_after
    if tests_pruned:
        changed = True

    if include_toolsmith_citation and not _has_toolsmith_citation(root):
        citations = root.find("citations")
        if citations is None:
            citations = ET.SubElement(root, "citations")
        if citation_mode == "macro":
            metadata["macros_import_added"] = ensure_macros_import(root)
            ET.SubElement(citations, "expand", {"macro": TOOLSMITH_CITATION_MACRO_NAME})
        else:
            citation = ET.SubElement(citations, "citation", {"type": "bibtex"})
            citation.text = TOOLSMITH_CITATION_BIBTEX
        metadata["citation_added"] = True
        changed = True

    if changed:
        ET.indent(root, space="  ")
        xml_text = ET.tostring(root, encoding="unicode")
    metadata["changed"] = changed
    return PostprocessResult(xml_text, metadata)


def prune_extra_generated_tests(root: ET.Element) -> tuple[bool, int, int]:
    tests = root.find("tests")
    if tests is None:
        return False, 0, 0
    direct_tests = [child for child in list(tests) if child.tag == "test"]
    test_count_before = len(direct_tests)
    if test_count_before <= 1:
        return False, test_count_before, test_count_before
    for test in direct_tests[1:]:
        tests.remove(test)
    return True, test_count_before, 1


def postprocess_udt_yaml(
    yaml_text: str,
    *,
    tool_id: str = "",
    tool_name: str = "",
    include_toolsmith_citation: bool = True,
) -> PostprocessResult:
    metadata: dict[str, Any] = {
        "artifact_format": ARTIFACT_FORMAT_UDT_YAML,
        "citation_added": False,
        "tool_id_applied": False,
        "tool_name_applied": False,
        "changed": False,
    }
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as error:
        metadata["error"] = f"YAML parse error: {error}"
        return PostprocessResult(yaml_text, metadata)
    if not isinstance(parsed, dict):
        metadata["skipped"] = "Expected UDT YAML root to be a mapping."
        return PostprocessResult(yaml_text, metadata)

    changed = False
    if tool_id and parsed.get("id") != tool_id:
        parsed["id"] = tool_id
        metadata["tool_id_applied"] = True
        changed = True
    if tool_name and parsed.get("name") != tool_name:
        parsed["name"] = tool_name
        metadata["tool_name_applied"] = True
        changed = True
    if include_toolsmith_citation and TOOLSMITH_CITATION_URL not in yaml_text:
        citations = parsed.get("citations")
        if not isinstance(citations, list):
            citations = []
            parsed["citations"] = citations
        citations.append({"type": "bibtex", "content": TOOLSMITH_CITATION_BIBTEX})
        metadata["citation_added"] = True
        changed = True

    if changed:
        yaml_text = yaml.safe_dump(parsed, sort_keys=False)
    metadata["changed"] = changed
    return PostprocessResult(yaml_text, metadata)


def ensure_macros_import(root: ET.Element) -> bool:
    macros = root.find("macros")
    changed = False
    if macros is None:
        macros = ET.Element("macros")
        insert_index = _tool_child_insert_index(root)
        root.insert(insert_index, macros)
        changed = True
    imports = [item for item in macros.findall("import") if (item.text or "").strip()]
    if not any((item.text or "").strip() == "macros.xml" for item in imports):
        import_node = ET.SubElement(macros, "import")
        import_node.text = "macros.xml"
        changed = True
    return changed


def write_toolsmith_macros_file(output_dir: Path) -> Path:
    path = output_dir / "macros.xml"
    if path.exists():
        try:
            root = ET.fromstring(path.read_text(encoding="utf-8"))
        except ET.ParseError:
            return path
        if _macros_file_has_toolsmith_macro(root):
            return path
        macro = ET.SubElement(root, "xml", {"name": TOOLSMITH_CITATION_MACRO_NAME})
        citation = ET.SubElement(macro, "citation", {"type": "bibtex"})
        citation.text = TOOLSMITH_CITATION_BIBTEX
    else:
        root = ET.Element("macros")
        macro = ET.SubElement(root, "xml", {"name": TOOLSMITH_CITATION_MACRO_NAME})
        citation = ET.SubElement(macro, "citation", {"type": "bibtex"})
        citation.text = TOOLSMITH_CITATION_BIBTEX
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return path


def write_macro_opportunities(output_dir: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    requirement_blocks: dict[str, list[str]] = {}
    for record in records:
        xml_path = str(record.get("output_xml_path") or record.get("output_path") or "").strip()
        if not xml_path:
            continue
        try:
            root = ET.fromstring(Path(xml_path).read_text(encoding="utf-8"))
        except (OSError, ET.ParseError):
            continue
        requirements = root.find("requirements")
        if requirements is not None:
            block = ET.tostring(requirements, encoding="unicode")
            requirement_blocks.setdefault(block, []).append(xml_path)
    repeated_requirements = [
        {"tool_paths": paths, "count": len(paths)}
        for paths in requirement_blocks.values()
        if len(paths) > 1
    ]
    payload = {
        "generated_by": "galaxy-toolsmith",
        "opportunities": [
            {
                "kind": "toolsmith_citation",
                "macro": TOOLSMITH_CITATION_MACRO_NAME,
                "status": "implemented_when_citation_mode_is_macro",
            },
            {
                "kind": "requirements",
                "status": "review_suggested",
                "repeated_blocks": repeated_requirements,
            },
        ],
    }
    return write_gtsm_json(output_dir / ".gtsm" / "macro-opportunities.json", payload)


def write_datatype_scaffold(
    target_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    repository_style: bool,
) -> dict[str, Any]:
    datatype_records = _datatype_records(records)
    datatypes = sorted({item["datatype"] for item in datatype_records})
    if repository_style:
        metadata_path = target_dir / ".gtsm" / "unknown-datatypes.json"
    else:
        metadata_path = target_dir / "unknown-datatypes.json"
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_by": "galaxy-toolsmith",
        "datatypes": datatypes,
        "records": datatype_records,
        "notes": [
            "This scaffold is intentionally conservative.",
            "Review each extension and replace Text with a more specific Galaxy datatype when appropriate.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    datatypes_path = target_dir / "datatypes_conf.xml.sample"
    datatypes_path.write_text(_datatypes_conf_xml(datatypes), encoding="utf-8")
    readme_path = target_dir / "README.datatypes.md"
    readme_path.write_text(_datatypes_readme(datatypes), encoding="utf-8")
    return {
        "enabled": True,
        "unknown_datatypes": datatypes,
        "metadata_path": str(metadata_path),
        "datatypes_conf_path": str(datatypes_path),
        "readme_path": str(readme_path),
    }


def datatype_scaffold_dir_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.gtsm")


def _has_toolsmith_citation(root: ET.Element) -> bool:
    xml_text = ET.tostring(root, encoding="unicode")
    return TOOLSMITH_CITATION_URL in xml_text or TOOLSMITH_CITATION_MACRO_NAME in xml_text


def _macros_file_has_toolsmith_macro(root: ET.Element) -> bool:
    return any(item.attrib.get("name") == TOOLSMITH_CITATION_MACRO_NAME for item in root.findall("xml"))


def _tool_child_insert_index(root: ET.Element) -> int:
    for index, child in enumerate(list(root)):
        if child.tag not in {"description"}:
            return index
    return len(list(root))


def _datatype_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for record in records:
        validation = record.get("validation", {})
        if not isinstance(validation, Mapping):
            continue
        for datatype in _unknown_datatypes(validation):
            collected.append(
                {
                    "datatype": datatype,
                    "tool_name": str(record.get("tool_name") or ""),
                    "tool_id": str(record.get("tool_id") or ""),
                    "output_path": str(record.get("output_path") or record.get("output_xml_path") or ""),
                }
            )
    return collected


def _unknown_datatypes(validation: Mapping[str, Any]) -> Iterable[str]:
    values = validation.get("unknown_datatypes", [])
    if not isinstance(values, list):
        return []
    return sorted(str(value).strip() for value in values if str(value).strip())


def _datatypes_conf_xml(datatypes: Sequence[str]) -> str:
    lines = [
        "<datatypes>",
        "  <!-- Review these placeholders before installing into a Galaxy instance. -->",
    ]
    for datatype in datatypes:
        escaped = _xml_attr(datatype)
        lines.append(
            f'  <datatype extension="{escaped}" type="galaxy.datatypes.data:Text" '
            'display_in_upload="true"/>'
        )
    lines.append("</datatypes>")
    return "\n".join(lines) + "\n"


def _datatypes_readme(datatypes: Sequence[str]) -> str:
    items = "\n".join(f"- `{datatype}`" for datatype in datatypes) or "- None detected"
    return (
        "# Datatype Scaffold\n\n"
        "Galaxy Toolsmith detected output or input extensions that are not in the packaged "
        "Galaxy datatype list used by validation.\n\n"
        f"{items}\n\n"
        "`datatypes_conf.xml.sample` maps each unknown extension to `Text` as a review "
        "placeholder. Replace it with a specific datatype class when the format semantics "
        "are known.\n"
    )


def _xml_attr(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
