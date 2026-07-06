from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.repository import build_tool_shed_metadata, write_shed_yml
from galaxy_toolsmith.inference.suite import generate_suite_from_content, plan_suite_from_content
from galaxy_toolsmith.providers.base import GenerationInput, GenerationOutput


class _CaptureProvider:
    name = "capture"

    def __init__(self) -> None:
        self.requests: list[GenerationInput] = []

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        self.requests.append(request)
        return GenerationOutput(
            xml_wrapper=(
                f'<tool id="{request.tool_name}" name="{request.tool_name}">'
                "<command>echo ok</command></tool>"
            ),
            provider=self.name,
            model_variant=request.model_variant,
        )


def test_write_shed_yml_for_single_tool_repository(tmp_path: Path) -> None:
    metadata = build_tool_shed_metadata(
        name="samtools_view",
        owner="iuc",
        description="View alignments",
        categories=["Sequence Analysis"],
    )

    path = write_shed_yml(tmp_path / ".shed.yml", metadata)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["name"] == "samtools_view"
    assert payload["owner"] == "iuc"
    assert payload["categories"] == ["Sequence Analysis"]
    assert "type" not in payload
    assert "repositories" not in payload


def test_write_shed_yml_for_suite_repository(tmp_path: Path) -> None:
    metadata = build_tool_shed_metadata(
        name="samtools",
        owner="iuc",
        description="Samtools suite",
        suite=True,
        repositories=["samtools_view", "samtools_sort"],
    )

    path = write_shed_yml(tmp_path / ".shed.yml", metadata)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["name"] == "suite_samtools"
    assert payload["type"] == "suite_repository"
    assert payload["repositories"] == [{"name": "samtools_view"}, {"name": "samtools_sort"}]


def test_plan_suite_detects_usage_subcommands() -> None:
    plan = plan_suite_from_content(
        tool_name="samtools",
        help_text="Usage: samtools [options] {view,sort,index}\n",
        max_suite_tools=2,
    )

    assert plan.suite_recommended is True
    assert [tool.tool_id for tool in plan.tools] == ["samtools_view", "samtools_sort"]


def test_plan_suite_skips_version_as_wrapper_command() -> None:
    plan = plan_suite_from_content(
        tool_name="minibwa",
        help_text=(
            "Usage: minibwa <command> <arguments>\n"
            "Commands:\n"
            "  index      index reference FASTA\n"
            "  map        read alignment\n"
            "  version    print the version number\n"
        ),
        max_suite_tools=8,
        force_suite=True,
    )

    assert [tool.tool_id for tool in plan.tools] == ["minibwa_index", "minibwa_map"]


def test_generate_suite_from_content_writes_repository_bundle(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    output_dir = tmp_path / "repo"

    record = generate_suite_from_content(
        paths=paths,
        tool_name="samtools",
        help_text="Usage: samtools [options] {view,sort}\n",
        source_code="int main(int argc, char** argv) { return 0; }",
        output_dir=output_dir,
        provider_name="local",
        model_variant="",
        model="",
        temperature=0.0,
        max_tokens=128,
        allow_stub_local=True,
        max_suite_tools=4,
        generate_sidecars=True,
        raw_response_logs=True,
    )

    payload = record.to_dict()
    shed = yaml.safe_load((output_dir / ".shed.yml").read_text(encoding="utf-8"))
    records_payload = json.loads(
        (output_dir / ".gtsm" / "generation-records.json").read_text(encoding="utf-8")
    )

    assert payload["suite_plan"]["suite_recommended"] is True
    assert (output_dir / "samtools_view.xml").exists()
    assert (output_dir / "samtools_sort.xml").exists()
    assert (output_dir / "macros.xml").exists()
    assert (output_dir / ".gtsm" / "macro-opportunities.json").exists()
    assert (output_dir / ".gtsm" / "unknown-datatypes.json").exists()
    assert 'name="samtools view"' in (output_dir / "samtools_view.xml").read_text(
        encoding="utf-8"
    )
    assert shed["type"] == "suite_repository"
    assert shed["repositories"] == [{"name": "samtools_view"}, {"name": "samtools_sort"}]
    assert len(records_payload["records"]) == 2
    macros_sidecars = [
        item
        for item in payload["sidecar_artifacts"]
        if item["role"] == "macros" and item["path"] == str(output_dir / "macros.xml")
    ]
    assert len(macros_sidecars) == 1


def test_generate_suite_includes_focused_runtime_help_for_subcommands(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    provider = _CaptureProvider()

    generate_suite_from_content(
        paths=paths,
        tool_name="samtools",
        help_text="Usage: samtools [options] {view,sort}\n",
        source_code="",
        output_dir=tmp_path / "repo",
        provider_name="capture",
        model_variant="variant",
        model="",
        temperature=0.0,
        max_tokens=128,
        provider_instance=provider,
        max_suite_tools=2,
        generate_sidecars=False,
        repair_invalid_xml=False,
        subcommand_help={
            "samtools view": "$ samtools view --help\nUsage: samtools view --input FILE",
        },
    )

    view_request = next(
        request for request in provider.requests if request.tool_name == "samtools_view"
    )
    sort_request = next(
        request for request in provider.requests if request.tool_name == "samtools_sort"
    )
    assert "Focused runtime help for `samtools view`" in view_request.help_text
    assert "Usage: samtools view --input FILE" in view_request.help_text
    assert "Suite member output contract:" not in view_request.help_text
    assert "Suite member output contract:" in view_request.repair_context
    assert (
        "Generate exactly one Galaxy <tool> XML document for this suite member."
        in view_request.repair_context
    )
    assert "Current tool id: samtools_view" in view_request.repair_context
    assert "Current command: samtools view" in view_request.repair_context
    assert (
        "Do not generate XML for sibling commands or other suite members."
        in view_request.repair_context
    )
    assert "Stop immediately after the closing </tool> for this tool." in view_request.repair_context
    assert "Sibling commands not to generate here: samtools sort" in view_request.repair_context
    assert "Focused runtime help" not in sort_request.help_text
    assert "Sibling commands not to generate here: samtools view" in sort_request.repair_context


def test_generate_suite_stream_output_prints_member_boundaries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    provider = _CaptureProvider()

    generate_suite_from_content(
        paths=paths,
        tool_name="samtools",
        help_text="Usage: samtools [options] {view,sort}\n",
        source_code="",
        output_dir=tmp_path / "repo",
        provider_name="capture",
        model_variant="variant",
        model="",
        temperature=0.0,
        max_tokens=128,
        provider_instance=provider,
        max_suite_tools=2,
        generate_sidecars=False,
        repair_invalid_xml=False,
        stream_output=True,
    )

    stderr = capsys.readouterr().err
    assert "[gtsm] suite member 1/2: samtools_view (samtools view)" in stderr
    assert "[gtsm] suite member 2/2: samtools_sort (samtools sort)" in stderr
