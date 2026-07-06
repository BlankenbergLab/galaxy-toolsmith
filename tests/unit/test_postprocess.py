from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from galaxy_toolsmith.inference.postprocess import (
    TOOLSMITH_CITATION_MACRO_NAME,
    TOOLSMITH_CITATION_URL,
    postprocess_udt_yaml,
    postprocess_xml_tool,
    write_datatype_scaffold,
    write_toolsmith_macros_file,
)


def test_postprocess_xml_adds_toolsmith_citation_and_preserves_display_name() -> None:
    result = postprocess_xml_tool(
        "<tool id='raw' name='raw'><command>echo ok</command></tool>",
        tool_id="minibwa_index",
        tool_name="minibwa index",
    )

    root = ET.fromstring(result.artifact_text)

    assert root.attrib["id"] == "minibwa_index"
    assert root.attrib["name"] == "minibwa index"
    assert TOOLSMITH_CITATION_URL in result.artifact_text
    assert result.metadata["citation_added"] is True


def test_postprocess_xml_can_use_macro_citation() -> None:
    result = postprocess_xml_tool(
        "<tool id='x' name='x'><command>echo ok</command></tool>",
        include_toolsmith_citation=True,
        citation_mode="macro",
    )

    root = ET.fromstring(result.artifact_text)

    assert root.find("macros/import").text == "macros.xml"
    assert root.find("citations/expand").attrib["macro"] == TOOLSMITH_CITATION_MACRO_NAME


def test_postprocess_xml_does_not_duplicate_toolsmith_citation() -> None:
    first = postprocess_xml_tool(
        "<tool id='x' name='x'><command>echo ok</command></tool>",
        include_toolsmith_citation=True,
    )
    second = postprocess_xml_tool(first.artifact_text, include_toolsmith_citation=True)

    root = ET.fromstring(second.artifact_text)

    assert len(root.findall("citations/citation")) == 1
    assert second.metadata["citation_added"] is False


def test_postprocess_xml_prunes_extra_generated_tests() -> None:
    result = postprocess_xml_tool(
        (
            "<tool id='x' name='x'>"
            "<command>echo ok</command>"
            "<tests>"
            "<test><param name='a' value='1'/></test>"
            "<test><param name='a' value='2'/></test>"
            "<test><param name='a' value='3'/></test>"
            "</tests>"
            "</tool>"
        ),
        include_toolsmith_citation=False,
    )

    root = ET.fromstring(result.artifact_text)

    assert len(root.findall("tests/test")) == 1
    assert result.metadata["tests_pruned"] is True
    assert result.metadata["test_count_before"] == 3
    assert result.metadata["test_count_after"] == 1


def test_postprocess_udt_adds_citation_and_name_fields() -> None:
    result = postprocess_udt_yaml(
        "class: GalaxyUserTool\nid: raw\nname: raw\nshell_command: echo ok\n",
        tool_id="echo_tool",
        tool_name="Echo Tool",
    )
    payload = yaml.safe_load(result.artifact_text)

    assert payload["id"] == "echo_tool"
    assert payload["name"] == "Echo Tool"
    assert TOOLSMITH_CITATION_URL in payload["citations"][0]["content"]


def test_write_toolsmith_macros_file(tmp_path: Path) -> None:
    path = write_toolsmith_macros_file(tmp_path)
    text = path.read_text(encoding="utf-8")

    assert path.name == "macros.xml"
    assert TOOLSMITH_CITATION_MACRO_NAME in text
    assert TOOLSMITH_CITATION_URL in text


def test_write_datatype_scaffold(tmp_path: Path) -> None:
    payload = write_datatype_scaffold(
        tmp_path,
        [
            {
                "tool_name": "tool one",
                "tool_id": "tool_one",
                "output_path": "tool_one.xml",
                "validation": {"unknown_datatypes": ["mbw", "pac"]},
            }
        ],
        repository_style=True,
    )
    metadata = json.loads(Path(payload["metadata_path"]).read_text(encoding="utf-8"))

    assert metadata["datatypes"] == ["mbw", "pac"]
    assert "extension=\"mbw\"" in (tmp_path / "datatypes_conf.xml.sample").read_text(
        encoding="utf-8"
    )
