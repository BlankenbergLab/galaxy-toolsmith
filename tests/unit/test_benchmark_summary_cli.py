from __future__ import annotations

import json
from pathlib import Path

import pytest

from galaxy_toolsmith.cli import main as cli_main


def test_benchmark_summary_cli_prints_compact_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "benchmark.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "attempted": 2,
                "succeeded": 1,
                "failed": 1,
                "startup": {
                    "processes": 4,
                    "model_load_seconds_max": 30.0,
                    "model_load_seconds_mean": 29.5,
                },
                "failures": [
                    {
                        "tool_name": "Bad Tool",
                        "error_type": "InvalidGeneratedXmlError",
                        "error": "Generated XML is not well formed.",
                        "output_xml_path": "/tmp/bad.xml",
                    }
                ],
                "quality": {
                    "throughput": {
                        "wrappers_per_minute": 1.2,
                        "seconds_per_attempted_wrapper": 50.0,
                    },
                    "validity": {
                        "success_rate": 0.5,
                        "xml_well_formed_rate": 0.5,
                        "tool_root_rate": 0.5,
                    },
                    "repair": {
                        "repair_attempt_rate": 0.5,
                        "repair_success_rate": 0.0,
                        "truncation_failure_rate": 0.5,
                    },
                    "reference_fidelity": {
                        "compared_records": 1,
                        "avg_input_count_abs_error": 1.0,
                        "avg_output_count_abs_error": 2.0,
                        "input_datatype_jaccard_mean": 0.25,
                        "output_datatype_jaccard_mean": 0.5,
                        "primary_command_presence_rate": 1.0,
                        "records": [
                            {
                                "tool_name": "Good Tool",
                                "input_count_abs_error": 1,
                                "output_count_abs_error": 2,
                                "input_datatype_jaccard": 0.25,
                                "output_datatype_jaccard": 0.5,
                                "primary_command_present": True,
                                "output_xml_path": "/tmp/good.xml",
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "gtsm",
            "--repo-root",
            str(tmp_path),
            "benchmark-summary",
            "--summary",
            str(summary_path),
        ],
    )

    assert cli_main.main() == 0
    output = capsys.readouterr().out

    assert "Benchmark summary" in output
    assert "attempted=2 succeeded=1 failed=1" in output
    assert "1.200 wrappers/min" in output
    assert "Per-tool fidelity:" in output
    assert "Good Tool" in output
    assert "Failures:" in output
    assert "Bad Tool" in output


def test_benchmark_summary_cli_handles_missing_fidelity_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_dir = tmp_path / ".gtsm-cache" / "runs" / "benchmark"
    summary_dir.mkdir(parents=True)
    (summary_dir / "benchmark.summary.json").write_text(
        json.dumps({"attempted": 0, "succeeded": 0, "failed": 0, "quality": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "gtsm",
            "--repo-root",
            str(tmp_path),
            "benchmark-summary",
        ],
    )

    assert cli_main.main() == 0
    output = capsys.readouterr().out

    assert "attempted=0 succeeded=0 failed=0" in output
    assert "Per-tool fidelity:" not in output
