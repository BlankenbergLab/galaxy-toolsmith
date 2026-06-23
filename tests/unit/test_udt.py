from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from galaxy_toolsmith.cli import main as cli_main
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.generation import generate_wrapper_from_content
from galaxy_toolsmith.inference.udt import (
    UDT_SCHEMA_URI,
    UdtConversionError,
    extract_udt_yaml,
    load_udt_schema,
    schema_path,
    udt_yaml_to_tool_xml,
    validate_udt_yaml,
)

CAT_UDT = """class: GalaxyUserTool
id: cat_user_defined
version: "0.1"
name: cat_user_defined
description: concatenates a file
container: busybox
shell_command: cat '$(inputs.input1.path)' > output.txt
inputs:
  - name: input1
    type: data
    format: txt
outputs:
  - name: output1
    type: data
    format: txt
    from_work_dir: output.txt
tests:
  - inputs:
      input1: simple_line.txt
    outputs:
      output1: simple_line.txt
"""


def test_udt_schema_is_packaged_and_valid() -> None:
    assert schema_path().exists()
    schema = load_udt_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["class"]["const"] == "GalaxyUserTool"
    assert UDT_SCHEMA_URI.endswith("/customTool.json")


def test_validate_udt_yaml_accepts_simple_user_tool() -> None:
    report = validate_udt_yaml(CAT_UDT)

    assert report.yaml_well_formed is True
    assert report.schema_valid is True
    assert report.root_is_user_tool is True
    assert report.artifact_valid is True
    assert report.notes == []


def test_validate_udt_yaml_reports_schema_errors() -> None:
    report = validate_udt_yaml("class: GalaxyTool\nname: not enough\n")

    assert report.yaml_well_formed is True
    assert report.schema_valid is False
    assert report.root_is_user_tool is False
    assert "id" in report.missing_required
    assert report.schema_errors


def test_extract_udt_yaml_strips_markdown_and_preamble() -> None:
    output = """Here is the tool:
```yaml
class: GalaxyUserTool
id: echo_text
version: "0.1.0"
name: Echo Text
container: busybox
shell_command: echo '$(inputs.text)' > output.txt
inputs:
  - name: text
    type: text
outputs:
  - name: output
    type: data
    format: txt
    from_work_dir: output.txt
```
"""

    extracted = extract_udt_yaml(output)

    assert extracted.startswith("class: GalaxyUserTool")
    assert "shell_command:" in extracted


def test_udt_to_xml_converts_simple_user_tool() -> None:
    result = udt_yaml_to_tool_xml(CAT_UDT)

    assert result.notes == []
    assert result.xml.startswith('<tool id="cat_user_defined"')
    assert '<container type="docker">busybox</container>' in result.xml
    assert "cat '$input1' > output.txt" in result.xml
    assert '<param name="input1" type="data" format="txt"/>' in result.xml
    assert '<data name="output1" format="txt" from_work_dir="output.txt"/>' in result.xml


def test_udt_to_xml_preserves_configfile_filename() -> None:
    udt = """class: GalaxyUserTool
id: config_tool
version: "0.1"
name: config_tool
container: busybox
shell_command: cat '$(inputs.input1.path)' > output.txt
configfiles:
  - name: settings
    filename: settings.json
    content: |
      {"threads": 1}
inputs:
  - name: input1
    type: data
    format: txt
outputs:
  - name: output1
    type: data
    format: txt
    from_work_dir: output.txt
"""

    result = udt_yaml_to_tool_xml(udt)

    assert '<configfile name="settings" filename="settings.json"><![CDATA[' in result.xml
    assert '{"threads": 1}' in result.xml


def test_udt_to_xml_rejects_unsupported_js_expression_by_default() -> None:
    multi_udt = CAT_UDT.replace(
        "cat '$(inputs.input1.path)' > output.txt",
        "cat $(inputs.datasets.map((input) => `'${input.path}'`).join(' ')) > output.txt",
    ).replace("name: input1", "name: datasets")

    with pytest.raises(UdtConversionError, match="unsupported UDT expression"):
        udt_yaml_to_tool_xml(multi_udt)


def test_galaxy_example_udts_validate_when_local_cache_exists() -> None:
    examples_root = (
        Path(__file__).resolve().parents[2]
        / ".gtsm-cache"
        / "planemo"
        / "galaxy"
        / "test"
        / "functional"
        / "tools"
    )
    examples = sorted(examples_root.glob("*user_defined*.yml"))
    if not examples:
        pytest.skip("local Galaxy UDT examples are not cached")

    for path in examples:
        assert validate_udt_yaml(path.read_text(encoding="utf-8")).artifact_valid is True


def test_generate_wrapper_from_content_writes_udt_record(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    output_path = tmp_path / "echo.yml"

    record = generate_wrapper_from_content(
        paths=paths,
        tool_name="echo_tool",
        help_text="Usage: echo_tool --input FILE",
        source_code="",
        output_path=output_path,
        provider_name="local",
        model_variant="stub",
        model="",
        temperature=0.0,
        max_tokens=128,
        allow_stub_local=True,
        artifact_format="udt-yaml",
    )

    payload = record.to_json()
    assert output_path.read_text(encoding="utf-8").startswith("class: GalaxyUserTool")
    assert '"artifact_format": "udt_yaml"' in payload
    assert record.output_path == str(output_path)
    assert record.output_xml_path == ""
    assert record.output_udt_yaml_path == str(output_path)
    assert record.validation["artifact_valid"] is True


def test_convert_udt_cli_writes_xml_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "tool.yml"
    output_path = tmp_path / "tool.xml"
    report_path = tmp_path / "report.json"
    input_path.write_text(CAT_UDT, encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "gtsm",
            "--repo-root",
            str(tmp_path),
            "convert-udt",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert cli_main.main() == 0
    payload = capsys.readouterr().out

    assert output_path.read_text(encoding="utf-8").startswith('<tool id="cat_user_defined"')
    assert report_path.exists()
    assert '"artifact_valid": true' in payload
