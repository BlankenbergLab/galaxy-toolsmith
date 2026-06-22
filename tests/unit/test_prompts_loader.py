from __future__ import annotations

from pathlib import Path

import pytest

from galaxy_toolsmith.prompts.loader import render_prompt_template


def test_default_xml_generate_prompt_renders_required_fields() -> None:
    rendered = render_prompt_template(
        task="xml_generate",
        skills_profile="default",
        context={
            "tool_name": "samtools_view",
            "help_text": "samtools view --help output",
            "source_code": "int main() { return 0; }",
            "skills_profile": "default",
            "interface_hints": "Core options:\n- --input FILE\nDatatype cues:\n- fasta",
        },
    )
    assert "Tool name: samtools_view" in rendered
    assert "samtools view --help output" in rendered
    assert "int main() { return 0; }" in rendered
    assert "exactly one complete Galaxy tool XML document" in rendered
    assert "must be <tool" in rendered
    assert "must be </tool>" in rendered
    assert "macros.xml content by itself" in rendered
    assert "Prefer a bounded, valid, interface-faithful wrapper" in rendered
    assert "Interface hints:" in rendered
    assert "Core options:" in rendered
    assert "Datatype cues:" in rendered
    assert "Do not invent long select option lists" in rendered
    assert "For unknown, unbounded, or long choice sets" in rendered
    assert "Keep the complete XML under 120 lines" in rendered
    assert "Include at most one minimal <test>" in rendered
    assert "Inside a test output, include at most three <has_text> assertions" in rendered
    assert "Do not repeat the same XML line" in rendered
    assert "preserve required/core options" in rendered
    assert "Choose output datatypes deliberately" in rendered
    assert "html only for actual HTML" in rendered


def test_default_udt_yaml_generate_prompt_renders_required_fields() -> None:
    rendered = render_prompt_template(
        task="udt_yaml_generate",
        skills_profile="default",
        context={
            "tool_name": "samtools_view",
            "help_text": "samtools view --help output",
            "source_code": "",
            "skills_profile": "default",
            "interface_hints": "Core options:\n- --input FILE\nDatatype cues:\n- bam",
        },
    )
    assert "Galaxy User-Defined Tool YAML" in rendered
    assert "class: GalaxyUserTool" in rendered
    assert "https://schema.galaxyproject.org/customTool.json" in rendered
    assert "Tool name: samtools_view" in rendered
    assert "Interface hints:" in rendered
    assert "Use only supported UDT input types" in rendered
    assert "Do not include unsupported Galaxy XML concepts" in rendered


def test_prompt_loader_falls_back_to_default_profile() -> None:
    rendered = render_prompt_template(
        task="xml_generate",
        skills_profile="does-not-exist",
        context={
            "tool_name": "tool_a",
            "help_text": "help",
            "source_code": "",
            "skills_profile": "does-not-exist",
        },
    )
    assert "Tool name: tool_a" in rendered


def test_prompt_loader_raises_for_missing_task() -> None:
    with pytest.raises(FileNotFoundError):
        render_prompt_template(
            task="missing_task",
            skills_profile="default",
            context={"tool_name": "t", "help_text": "", "source_code": "", "skills_profile": "default"},
        )


def test_prompt_templates_are_packaged() -> None:
    templates_root = Path(__file__).resolve().parents[2] / "src" / "galaxy_toolsmith" / "prompts" / "templates"
    assert (templates_root / "default" / "xml_generate.txt").exists()
    assert (templates_root / "default" / "udt_yaml_generate.txt").exists()
