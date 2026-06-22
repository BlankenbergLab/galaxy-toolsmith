from __future__ import annotations

from pathlib import Path

from galaxy_toolsmith.inference.datatypes import known_galaxy_datatypes
from galaxy_toolsmith.inference.validation import validate_wrapper


def test_known_datatypes_expanded_registry_size() -> None:
    known = known_galaxy_datatypes()
    assert len(known) >= 100
    assert "bam" in known
    assert "tabular" in known
    assert "xml" in known


def test_known_datatypes_supports_extra_file(tmp_path: Path) -> None:
    extra = tmp_path / "extra-types.txt"
    extra.write_text("# comment\nmy_custom_type\n\n", encoding="utf-8")
    known = known_galaxy_datatypes(extra_file=extra)
    assert "my_custom_type" in known


def test_validation_uses_shared_known_registry() -> None:
    xml = (
        '<tool id="x" name="x" version="0.1">'
        '<inputs><param name="i" type="data" format="bam"/></inputs>'
        '<outputs><data name="o" format="tabular"/></outputs>'
        "</tool>"
    )
    report = validate_wrapper(xml)
    assert report.xml_well_formed is True
    assert report.root_tag == "tool"
    assert report.root_is_tool is True
    assert report.unknown_datatypes == []


def test_validation_splits_comma_separated_datatypes() -> None:
    xml = (
        '<tool id="x" name="x" version="0.1">'
        '<inputs><param name="i" type="data" format="bam,tabular"/></inputs>'
        '<outputs><data name="o" format="xml"/></outputs>'
        "</tool>"
    )
    report = validate_wrapper(xml)
    assert report.xml_well_formed is True
    assert report.unknown_datatypes == []


def test_validation_reports_well_formed_non_tool_root() -> None:
    report = validate_wrapper("<macros><xml name='inputs'/></macros>")

    assert report.xml_well_formed is True
    assert report.root_tag == "macros"
    assert report.root_is_tool is False
    assert "Expected root <tool>" in report.notes[0]


def test_validation_reports_malformed_xml_without_root() -> None:
    report = validate_wrapper("<tool id='x'>")

    assert report.xml_well_formed is False
    assert report.root_tag == ""
    assert report.root_is_tool is False
    assert "XML parse error:" in report.notes[0]
