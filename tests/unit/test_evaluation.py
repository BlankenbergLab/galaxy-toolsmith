from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from galaxy_toolsmith.inference import evaluation as evaluation_mod
from galaxy_toolsmith.inference import validation as validation_mod
from galaxy_toolsmith.inference.evaluation import evaluate_wrapper_paths
from galaxy_toolsmith.inference.validation import PlanemoTestOptions


def _write_tool_wrapper(path: Path) -> None:
    path.write_text(
        """
<tool id="x" name="x" version="0.1">
  <command><![CDATA[echo hi]]></command>
  <inputs>
    <param name="i" type="data" format="bam"/>
  </inputs>
  <outputs>
    <data name="o" format="tabular"/>
  </outputs>
  <help><![CDATA[help text]]></help>
  <tests>
    <test expect_num_outputs="1"><output name="o"/></test>
  </tests>
</tool>
""".strip(),
        encoding="utf-8",
    )


def test_evaluation_summary_includes_structural_metrics(tmp_path: Path) -> None:
    wrapper = tmp_path / "tool.xml"
    _write_tool_wrapper(wrapper)

    report_path = tmp_path / "summary.json"
    summary = evaluate_wrapper_paths([wrapper], output_report=report_path, run_planemo=False)
    assert summary.total_wrappers == 1
    assert summary.tool_root_count == 1
    assert summary.mean_structural_score > 0.0
    assert summary.wrapper_reports[0]["root_tag"] == "tool"
    assert summary.wrapper_reports[0]["root_is_tool"] is True
    assert summary.wrapper_reports[0]["structural"]["input_count"] == 1
    assert summary.wrapper_reports[0]["structural"]["output_count"] == 1
    assert summary.wrapper_reports[0]["structural"]["test_count"] == 1

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "mean_structural_score" in payload
    assert payload["tool_root_count"] == 1
    assert payload["planemo_test_status"] == "not_run"


def test_evaluation_non_tool_root_has_zero_structural_score(tmp_path: Path) -> None:
    wrapper = tmp_path / "macros.xml"
    wrapper.write_text("<macros><xml name='inputs'/></macros>", encoding="utf-8")

    summary = evaluate_wrapper_paths([wrapper], output_report=tmp_path / "summary.json")

    assert summary.total_wrappers == 1
    assert summary.xml_well_formed_count == 1
    assert summary.tool_root_count == 0
    assert summary.mean_structural_score == 0.0
    assert summary.wrapper_reports[0]["root_tag"] == "macros"
    assert summary.wrapper_reports[0]["root_is_tool"] is False
    assert summary.wrapper_reports[0]["structural"]["structural_score"] == 0.0
    assert "Expected root <tool>" in summary.wrapper_reports[0]["notes"][0]


def test_evaluation_supports_udt_yaml(tmp_path: Path) -> None:
    wrapper = tmp_path / "tool.yml"
    wrapper.write_text(
        """
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
""".strip(),
        encoding="utf-8",
    )

    summary = evaluate_wrapper_paths(
        [wrapper],
        output_report=tmp_path / "summary.json",
        artifact_format="udt-yaml",
        run_planemo=True,
    )

    assert summary.artifact_format == "udt_yaml"
    assert summary.total_wrappers == 1
    assert summary.artifact_valid_count == 1
    assert summary.yaml_well_formed_count == 1
    assert summary.udt_schema_valid_count == 1
    assert summary.user_tool_root_count == 1
    assert summary.planemo_status == "not_run"
    assert summary.wrapper_reports[0]["schema_valid"] is True
    assert "Planemo lint is XML-only" in " ".join(summary.wrapper_reports[0]["notes"])


def test_run_planemo_test_builds_command_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = tmp_path / "tool.xml"
    _write_tool_wrapper(wrapper)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(validation_mod, "resolve_planemo_executable", lambda: "/opt/planemo")
    monkeypatch.setattr(validation_mod.subprocess, "run", fake_run)

    status, message, artifacts = validation_mod.run_planemo_test(
        wrapper,
        enabled=True,
        options=PlanemoTestOptions(
            output_dir=tmp_path / "planemo-report",
            timeout_seconds=45,
            galaxy_root=tmp_path / "galaxy",
            install_galaxy=True,
            engine="galaxy",
            conda_prefix=tmp_path / "conda",
            test_data=tmp_path / "test-data",
            extra_tools=(tmp_path / "extra-tools",),
            no_dependency_resolution=True,
        ),
    )

    assert status == "passed"
    assert message == ""
    assert artifacts["output_json"].endswith("tool_test_output.json")
    command = commands[0]
    assert command[:3] == ["/opt/planemo", "test", str(wrapper)]
    assert "--test_output_json" in command
    assert "--test_timeout" in command
    assert "45" in command
    assert "--galaxy_root" in command
    assert "--install_galaxy" in command
    assert "--engine" in command
    assert "--conda_prefix" in command
    assert "--test_data" in command
    assert "--extra_tools" in command
    assert "--no_dependency_resolution" in command


def test_evaluation_records_planemo_test_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = tmp_path / "tool.xml"
    _write_tool_wrapper(wrapper)
    observed_output_dirs: list[Path] = []

    def fake_planemo_test(
        path: Path,
        *,
        enabled: bool,
        options: PlanemoTestOptions | None = None,
    ) -> tuple[str, str, dict[str, str]]:
        assert path == wrapper
        assert enabled is True
        assert options is not None
        assert options.output_dir is not None
        observed_output_dirs.append(options.output_dir)
        return "passed", "", {"output_json": str(options.output_dir / "tool_test_output.json")}

    monkeypatch.setattr(evaluation_mod, "run_planemo_test", fake_planemo_test)
    report_path = tmp_path / "reports" / "summary.json"

    summary = evaluate_wrapper_paths(
        [wrapper],
        output_report=report_path,
        run_planemo_tests=True,
        planemo_test_options=PlanemoTestOptions(timeout_seconds=30),
    )

    assert summary.planemo_test_status == "passed"
    assert summary.wrapper_reports[0]["planemo_test_status"] == "passed"
    assert summary.wrapper_reports[0]["planemo_test"]["output_json"].endswith(
        "tool_test_output.json"
    )
    assert observed_output_dirs[0].parent == report_path.parent / "planemo-tests"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["planemo_test_status"] == "passed"
    assert payload["wrapper_reports"][0]["planemo_test_status"] == "passed"
