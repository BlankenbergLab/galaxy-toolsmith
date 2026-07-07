from __future__ import annotations

from pathlib import Path

from galaxy_toolsmith.inference.source_context import (
    build_source_context_from_paths,
    build_source_context_from_record,
    build_source_context_variants_from_record,
    source_context_settings,
)


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "src" / "mytool").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "tests" / "data").mkdir()
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
        "def test_cli():\n    assert run_fixture('tests/data/reads.fastq')\n",
        encoding="utf-8",
    )
    (root / "tests" / "data" / "reads.fastq").write_text("@r\nACGT\n+\n!!!!\n", encoding="utf-8")
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


def test_metadata_mode_includes_source_provider_fallback_fields(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    record = _record(root)
    record["bioconda_sources"][0].update(
        {
            "source_provider_package": "htslib",
            "source_provider_required_version": "1.22.1",
            "source_provider_recipe_package": "htslib",
            "source_provider_source_url": "https://example.org/htslib-1.22.1.tar.bz2",
            "source_provider_reason": "source_less_run_dependency",
        }
    )

    result = build_source_context_from_record(
        record,
        source_context_settings(mode="metadata", max_chars=3000, max_files=3),
    )

    assert "source_provider_package: htslib" in result.text
    assert "source_provider_source_url: https://example.org/htslib-1.22.1.tar.bz2" in result.text
    assert "source_provider_reason: source_less_run_dependency" in result.text


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


def test_snippets_mode_includes_wrapper_helpers_before_upstream_source(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    helper = tmp_path / "helper.py"
    helper.write_text("print('wrapper helper')\n", encoding="utf-8")
    record = _record(root)
    record["wrapper_helper_files"] = [
        {
            "path": str(helper),
            "relative_path": "helper.py",
            "extension": ".py",
            "byte_count": helper.stat().st_size,
            "sha256": "abc123",
            "role_hint": "command_reference",
        }
    ]
    record["wrapper_configfiles"] = [
        {
            "name": "script",
            "filename": "generated.py",
            "extension": ".py",
            "language": "python",
            "byte_count": 24,
            "sha256": "def456",
            "role_hint": "script_template",
            "template_kind": "script_template",
            "referenced_by_command": True,
            "content": "print('config helper')\n",
        }
    ]
    record["wrapper_source_summary"] = {
        "helper_file_count": 1,
        "configfile_count": 1,
        "skipped_file_count": 0,
        "skip_reasons": {},
    }

    result = build_source_context_from_record(
        record,
        source_context_settings(mode="snippets", max_chars=8000, max_files=4),
    )

    helper_index = result.text.index("Existing wrapper helper file: helper.py")
    config_index = result.text.index("Existing wrapper configfile: generated.py")
    source_index = result.text.index("Source file: src/mytool/cli.py")
    assert helper_index < source_index
    assert config_index < source_index
    assert "wrapper helper" in result.text
    assert "config helper" in result.text
    assert result.included_files >= 3


def test_metadata_mode_summarizes_wrapper_sources_without_content(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    helper = tmp_path / "helper.py"
    helper.write_text("print('wrapper helper')\n", encoding="utf-8")
    record = _record(root)
    record["wrapper_helper_files"] = [
        {
            "path": str(helper),
            "relative_path": "helper.py",
            "extension": ".py",
            "byte_count": helper.stat().st_size,
            "sha256": "abc123456789",
            "role_hint": "command_reference",
        }
    ]
    record["wrapper_configfiles"] = [
        {
            "name": "script",
            "filename": "generated.py",
            "extension": ".py",
            "language": "python",
            "byte_count": 24,
            "sha256": "def456789012",
            "role_hint": "script_template",
            "template_kind": "script_template",
            "referenced_by_command": True,
            "content": "print('config helper')\n",
        }
    ]
    record["wrapper_source_summary"] = {
        "helper_file_count": 1,
        "configfile_count": 1,
        "skipped_file_count": 0,
        "skip_reasons": {},
    }

    result = build_source_context_from_record(
        record,
        source_context_settings(mode="metadata", max_chars=3000, max_files=4),
    )

    assert "Wrapper source metadata:" in result.text
    assert "helper.py" in result.text
    assert "generated.py" in result.text
    assert "kind=script_template language=python referenced_by_command=true" in result.text
    assert "wrapper helper" not in result.text
    assert "config helper" not in result.text
    assert result.included_files == 0


def test_source_context_marks_truncated_wrapper_configfiles(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    record = _record(root)
    record["wrapper_configfiles"] = [
        {
            "name": "large",
            "filename": "large.txt",
            "extension": ".txt",
            "language": "text",
            "byte_count": 300000,
            "stored_byte_count": 256000,
            "sha256": "abc123456789",
            "role_hint": "config_template",
            "template_kind": "config_template",
            "referenced_by_command": False,
            "content_truncated": True,
            "content": "x = 1\n[truncated configfile content]\n",
        }
    ]
    record["wrapper_source_summary"] = {
        "helper_file_count": 0,
        "configfile_count": 1,
        "truncated_configfile_count": 1,
        "skipped_file_count": 0,
        "skip_reasons": {},
    }

    result = build_source_context_from_record(
        record,
        source_context_settings(mode="snippets", max_chars=3000, max_files=1),
    )

    assert "truncated_configfile_count: 1" in result.text
    assert "content_truncated=true stored_bytes=256000" in result.text
    assert "[truncated configfile content]" in result.text


def test_truncated_source_context_closes_markdown_fence(tmp_path: Path) -> None:
    source_file = tmp_path / "tool.py"
    source_file.write_text("def main():\n" + "    print('x')\n" * 100, encoding="utf-8")

    result = build_source_context_from_paths(
        settings=source_context_settings(mode="snippets", max_chars=180, max_files=1),
        source_file=source_file,
    )

    assert result.truncated is True
    assert "[truncated source context]" in result.text
    assert result.text.count("```") % 2 == 0
    assert result.text.rfind("```") < result.text.rfind("[truncated source context]")


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
    assert "Source file: data/reads.fastq" not in raw.text
    assert "Source file: tests/data/reads.fastq" not in raw.text


def test_test_context_metadata_lists_tests_without_content(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    result = build_source_context_from_record(
        _record(root),
        source_context_settings(
            mode="all-filtered",
            max_chars=8000,
            max_files=10,
            test_context_mode="metadata",
            test_context_max_chars=1000,
            test_context_max_files=4,
        ),
    )

    assert "Source test/example context:" in result.text
    assert "tests/test_cli.py role=test" in result.text
    assert "assert run_fixture" not in result.text
    assert result.included_test_files >= 1
    assert result.test_context_mode == "metadata"


def test_test_context_snippets_include_test_code_not_fixtures(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    result = build_source_context_from_record(
        _record(root),
        source_context_settings(
            mode="all-filtered",
            max_chars=8000,
            max_files=10,
            test_context_mode="snippets",
            test_context_max_chars=2000,
            test_context_max_files=4,
        ),
    )

    assert "Source test/example file: tests/test_cli.py" in result.text
    assert "assert run_fixture" in result.text
    assert "Source test/example fixture: tests/data/reads.fastq" not in result.text
    assert "Source file: tests/test_cli.py" not in result.text
    assert result.included_test_files >= 1


def test_test_context_fixtures_include_small_test_data(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    result = build_source_context_from_record(
        _record(root),
        source_context_settings(
            mode="all-filtered",
            max_chars=8000,
            max_files=10,
            test_context_mode="fixtures",
            test_context_max_chars=3000,
            test_context_max_files=6,
        ),
    )

    assert "Source test/example fixture: tests/data/reads.fastq" in result.text
    assert "@r\nACGT" in result.text
    assert "Source test/example fixture: data/reads.fastq" not in result.text


def test_source_context_variants_match_individual_builds(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    record = _record(root)
    settings = [
        source_context_settings(mode="all-filtered", max_chars=3000, max_files=1),
        source_context_settings(mode="all-filtered", max_chars=8000, max_files=10),
        source_context_settings(mode="all-raw", max_chars=8000, max_files=10),
        source_context_settings(
            mode="all-filtered",
            max_chars=8000,
            max_files=10,
            test_context_mode="fixtures",
            test_context_max_chars=3000,
            test_context_max_files=6,
        ),
    ]

    variants = build_source_context_variants_from_record(record, settings)

    assert variants == tuple(
        build_source_context_from_record(record, setting) for setting in settings
    )


def test_all_filtered_skips_weak_source_roots_but_keeps_metadata(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    record = _record(root)
    record["bioconda_sources"][0].update(
        {
            "recipe_version": "1.9",
            "recipe_selection_reason": "closest_major",
            "source_confidence": "weak",
            "source_version_match": "mismatch",
        }
    )

    filtered = build_source_context_from_record(
        record,
        source_context_settings(mode="all-filtered", max_chars=8000, max_files=10),
    )
    raw = build_source_context_from_record(
        record,
        source_context_settings(mode="all-raw", max_chars=8000, max_files=10),
    )

    assert "source_confidence: weak" in filtered.text
    assert "source_version_match: mismatch" in filtered.text
    assert "Source file: src/mytool/cli.py" not in filtered.text
    assert "Source file: src/mytool/cli.py" in raw.text


def test_none_mode_preserves_manual_source_file_behavior(tmp_path: Path) -> None:
    source_file = tmp_path / "tool.py"
    source_file.write_text("print('legacy source')\n", encoding="utf-8")

    result = build_source_context_from_paths(
        settings=source_context_settings(mode="none"),
        source_file=source_file,
    )

    assert result.text == "print('legacy source')\n"
    assert result.included_files == 1
