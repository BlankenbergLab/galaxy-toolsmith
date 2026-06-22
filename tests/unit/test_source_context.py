from __future__ import annotations

from pathlib import Path

from galaxy_toolsmith.inference.source_context import (
    build_source_context_from_paths,
    build_source_context_from_record,
    source_context_settings,
)


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "src" / "mytool").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "data").mkdir()
    (root / "pyproject.toml").write_text(
        """
[project.scripts]
mytool = "mytool.cli:main"
""".strip(),
        encoding="utf-8",
    )
    (root / "src" / "mytool" / "cli.py").write_text(
        """
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(prog="mytool")
    parser.add_argument("--input")
    return parser.parse_args()
""".strip(),
        encoding="utf-8",
    )
    (root / "src" / "mytool" / "core.py").write_text(
        "def run(input_path):\n    return input_path\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_cli.py").write_text(
        "def test_cli():\n    assert True\n",
        encoding="utf-8",
    )
    (root / "data" / "reads.fastq").write_text("@r\nACGT\n+\n!!!!\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG\x00")
    return root


def _record(root: Path) -> dict:
    return {
        "tool_name": "mytool",
        "bioconda_sources": [
            {
                "package": "mytool",
                "required_version": "1.0",
                "source_url": "https://example.org/mytool.tar.gz",
                "source_checkout": str(root),
                "command_hints": ["mytool"],
            }
        ],
    }


def test_metadata_mode_includes_source_metadata_without_files(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    result = build_source_context_from_record(
        _record(root),
        source_context_settings(mode="metadata", max_chars=2000, max_files=3),
    )

    assert "https://example.org/mytool.tar.gz" in result.text
    assert "command_hints: mytool" in result.text
    assert "ArgumentParser" not in result.text
    assert result.included_files == 0
    assert result.metadata_sources == 1


def test_snippets_mode_ranks_cli_files_and_excludes_tests(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    result = build_source_context_from_record(
        _record(root),
        source_context_settings(mode="snippets", max_chars=5000, max_files=4),
    )

    assert "Source file: src/mytool/cli.py" in result.text
    assert "ArgumentParser" in result.text
    assert "tests/test_cli.py" not in result.text
    assert "reads.fastq" not in result.text
    assert result.included_files >= 1


def test_all_filtered_excludes_tests_while_all_raw_can_include_them(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    filtered = build_source_context_from_record(
        _record(root),
        source_context_settings(mode="all-filtered", max_chars=8000, max_files=10),
    )
    raw = build_source_context_from_record(
        _record(root),
        source_context_settings(mode="all-raw", max_chars=8000, max_files=10),
    )

    assert "tests/test_cli.py" not in filtered.text
    assert "Source file: tests/test_cli.py" in raw.text
    assert "reads.fastq" not in raw.text


def test_none_mode_preserves_manual_source_file_behavior(tmp_path: Path) -> None:
    source_file = tmp_path / "tool.py"
    source_file.write_text("print('legacy source')\n", encoding="utf-8")

    result = build_source_context_from_paths(
        settings=source_context_settings(mode="none"),
        source_file=source_file,
    )

    assert result.text == "print('legacy source')\n"
    assert result.included_files == 1
