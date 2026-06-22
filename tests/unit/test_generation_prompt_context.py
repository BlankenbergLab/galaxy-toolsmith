from __future__ import annotations

from galaxy_toolsmith.inference.prompt_context import extract_interface_hints, shape_help_text


def test_shape_help_text_caps_long_help_and_preserves_options() -> None:
    help_text = "\n".join(
        [
            "Usage: tool run [OPTIONS]",
            "intro " * 200,
            "--input FILE",
            "--threads INT",
            "Examples:",
            "tool run --input reads.fq",
            *[f"noise line {index}" for index in range(200)],
        ]
    )

    shaped = shape_help_text(help_text, max_chars=700)

    assert shaped.truncated is True
    assert shaped.shaped_chars <= 700
    assert "Usage: tool run" in shaped.text
    assert "--input FILE" in shaped.text
    assert "--threads INT" in shaped.text
    assert "Toolsmith note: help text was shortened" in shaped.text


def test_shape_help_text_collapses_repeated_lines() -> None:
    shaped = shape_help_text("same\nsame\nsame\nsame\nunique", max_chars=1000)

    assert shaped.omitted_repeated_lines == 2
    assert shaped.text.splitlines() == ["same", "same", "unique"]


def test_extract_interface_hints_summarizes_cli_interface() -> None:
    hints = extract_interface_hints(
        "\n".join(
            [
                "Usage: tool run --input reads.fasta --output results.tsv",
                "Required input file in FASTA format.",
                "--input FILE      input assembly",
                "--output FILE     write tabular report",
                "--threads INT     worker count",
                "--advanced VALUE  optional tuning",
                "Results are written to an output TSV report.",
            ]
        )
    )

    assert "Usage:" in hints.text
    assert "Usage: tool run" in hints.text
    assert "Core options:" in hints.text
    assert "--input FILE" in hints.text
    assert "--output FILE" in hints.text
    assert "Required/input cues:" in hints.text
    assert "Output cues:" in hints.text
    assert "Datatype cues:" in hints.text
    assert "fasta" in hints.datatype_terms
    assert "tsv" in hints.datatype_terms


def test_extract_interface_hints_includes_safe_metadata_and_extra_datatypes() -> None:
    hints = extract_interface_hints(
        "Usage: abricate --db card input.embl > report.html\n"
        "Accepts fasta.bz2 and genbank.gz compressed inputs.\n",
        metadata={
            "package_id": "iuc/abricate",
            "tool_id": "abricate",
            "primary_command": "abricate",
            "expanded_xml_path": "/tmp/reference.xml",
        },
    )

    assert "Metadata cues:" in hints.text
    assert "Package: iuc/abricate" in hints.text
    assert "Tool id: abricate" in hints.text
    assert "Primary command: abricate" in hints.text
    assert "/tmp/reference.xml" not in hints.text
    assert "embl" in hints.datatype_terms
    assert "fasta.bz2" in hints.datatype_terms
    assert "genbank.gz" in hints.datatype_terms
    assert "html" in hints.datatype_terms
