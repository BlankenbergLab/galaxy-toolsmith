from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from xml.etree import ElementTree as ET

from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_XML,
    ARTIFACT_FORMAT_UDT_YAML,
    normalize_artifact_format,
)
from galaxy_toolsmith.inference.udt import udt_structural_report, validate_udt_yaml
from galaxy_toolsmith.inference.validation import (
    PlanemoTestOptions,
    run_planemo_lint,
    run_planemo_test,
    run_xsd_validation,
    validate_wrapper,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class EvaluationSummary:
    created_at: str
    total_wrappers: int
    xml_well_formed_count: int
    tool_root_count: int
    wrappers_with_unknown_datatypes: int
    artifact_format: str = ARTIFACT_FORMAT_XML
    artifact_valid_count: int = 0
    yaml_well_formed_count: int = 0
    udt_schema_valid_count: int = 0
    user_tool_root_count: int = 0
    xsd_status: str = "not_run"
    planemo_status: str = "not_run"
    planemo_test_status: str = "not_run"
    mean_structural_score: float = 0.0
    wrapper_reports: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _aggregate_status(values: list[str], default: str) -> str:
    if not values:
        return default
    if "failed" in values:
        return "failed"
    if "passed" in values:
        return "passed"
    if "not_available" in values:
        return "not_available"
    if "not_configured" in values:
        return "not_configured"
    return default


def _structural_report(xml_text: str) -> dict:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {
            "input_count": 0,
            "output_count": 0,
            "test_count": 0,
            "has_help": False,
            "has_command": False,
            "has_citations": False,
            "structural_score": 0.0,
        }
    if root.tag != "tool":
        return {
            "input_count": 0,
            "output_count": 0,
            "test_count": 0,
            "has_help": False,
            "has_command": False,
            "has_citations": False,
            "structural_score": 0.0,
        }

    input_count = len(root.findall(".//inputs//param"))
    output_count = len(root.findall(".//outputs//data")) + len(root.findall(".//outputs//collection"))
    test_count = len(root.findall(".//tests//test"))
    has_help = bool(root.findall(".//help"))
    has_command = bool(root.findall(".//command"))
    has_citations = bool(root.findall(".//citations"))

    score = 0.0
    score += 0.25 if input_count > 0 else 0.0
    score += 0.25 if output_count > 0 else 0.0
    score += 0.20 if test_count > 0 else 0.0
    score += 0.15 if has_help else 0.0
    score += 0.10 if has_command else 0.0
    score += 0.05 if has_citations else 0.0

    return {
        "input_count": input_count,
        "output_count": output_count,
        "test_count": test_count,
        "has_help": has_help,
        "has_command": has_command,
        "has_citations": has_citations,
        "structural_score": round(score, 4),
    }


def _safe_artifact_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return slug or "wrapper"


def _planemo_test_output_dir(root: Path, wrapper_path: Path) -> Path:
    digest = sha1(str(wrapper_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return root / f"{_safe_artifact_slug(wrapper_path.stem)}-{digest}"


def evaluate_wrapper_paths(
    wrapper_paths: list[Path],
    output_report: Path,
    xsd_path: Path | None = None,
    run_planemo: bool = False,
    run_planemo_tests: bool = False,
    planemo_test_options: PlanemoTestOptions | None = None,
    artifact_format: str = ARTIFACT_FORMAT_XML,
) -> EvaluationSummary:
    artifact_format = normalize_artifact_format(artifact_format)
    well_formed = 0
    tool_root_count = 0
    artifact_valid_count = 0
    yaml_well_formed_count = 0
    udt_schema_valid_count = 0
    user_tool_root_count = 0
    unknown_count = 0
    xsd_statuses: list[str] = []
    planemo_statuses: list[str] = []
    planemo_test_statuses: list[str] = []
    structural_scores: list[float] = []
    per_wrapper: list[dict] = []
    planemo_test_root = (
        planemo_test_options.output_dir
        if planemo_test_options and planemo_test_options.output_dir is not None
        else output_report.parent / "planemo-tests"
    )
    for path in wrapper_paths:
        artifact_text = path.read_text(encoding="utf-8")
        if artifact_format == ARTIFACT_FORMAT_UDT_YAML:
            udt_report = validate_udt_yaml(artifact_text, check_conversion=True)
            structural = udt_structural_report(artifact_text)
            xsd_status = "not_run"
            planemo_status = "not_run"
            planemo_test_status = "not_run"
            planemo_test_artifacts: dict[str, str] = {}
            wrapper_notes = list(udt_report.notes)
            if xsd_path is not None:
                wrapper_notes.append("xsd: XSD validation is XML-only and was skipped for UDT YAML.")
            if run_planemo:
                wrapper_notes.append("planemo: Planemo lint is XML-only in this evaluator and was skipped for UDT YAML.")
            if run_planemo_tests:
                wrapper_notes.append(
                    "planemo test: Planemo tests are XML-only in this evaluator and were skipped for UDT YAML."
                )
            xsd_statuses.append(xsd_status)
            planemo_statuses.append(planemo_status)
            planemo_test_statuses.append(planemo_test_status)
            structural_scores.append(structural["structural_score"])
            per_wrapper.append(
                {
                    "path": str(path),
                    "artifact_format": artifact_format,
                    "artifact_valid": udt_report.artifact_valid,
                    "yaml_well_formed": udt_report.yaml_well_formed,
                    "schema_valid": udt_report.schema_valid,
                    "root_class": udt_report.root_class,
                    "root_is_user_tool": udt_report.root_is_user_tool,
                    "missing_required": udt_report.missing_required,
                    "xsd_status": xsd_status,
                    "planemo_status": planemo_status,
                    "planemo_test_status": planemo_test_status,
                    "planemo_test": planemo_test_artifacts,
                    "structural": structural,
                    "notes": wrapper_notes,
                }
            )
            if udt_report.yaml_well_formed:
                yaml_well_formed_count += 1
            if udt_report.schema_valid:
                udt_schema_valid_count += 1
            if udt_report.root_is_user_tool:
                user_tool_root_count += 1
            if udt_report.artifact_valid:
                artifact_valid_count += 1
            continue

        xml_report = validate_wrapper(artifact_text)
        structural = _structural_report(artifact_text)
        xsd_status, xsd_message = run_xsd_validation(path, xsd_path)
        planemo_status, planemo_message = run_planemo_lint(path, enabled=run_planemo)
        per_wrapper_planemo_options = replace(
            planemo_test_options or PlanemoTestOptions(),
            output_dir=_planemo_test_output_dir(planemo_test_root, path),
        )
        planemo_test_status, planemo_test_message, planemo_test_artifacts = run_planemo_test(
            path,
            enabled=run_planemo_tests,
            options=per_wrapper_planemo_options,
        )

        xsd_statuses.append(xsd_status)
        planemo_statuses.append(planemo_status)
        planemo_test_statuses.append(planemo_test_status)
        structural_scores.append(structural["structural_score"])

        wrapper_notes = list(xml_report.notes)
        if xsd_message:
            wrapper_notes.append(f"xsd: {xsd_message}")
        if planemo_message:
            wrapper_notes.append(f"planemo: {planemo_message}")
        if planemo_test_message:
            wrapper_notes.append(f"planemo test: {planemo_test_message}")

        per_wrapper.append(
            {
                "path": str(path),
                "artifact_format": artifact_format,
                "artifact_valid": xml_report.xml_well_formed and xml_report.root_is_tool,
                "xml_well_formed": xml_report.xml_well_formed,
                "root_tag": xml_report.root_tag,
                "root_is_tool": xml_report.root_is_tool,
                "unknown_datatypes": xml_report.unknown_datatypes,
                "xsd_status": xsd_status,
                "planemo_status": planemo_status,
                "planemo_test_status": planemo_test_status,
                "planemo_test": planemo_test_artifacts,
                "structural": structural,
                "notes": wrapper_notes,
            }
        )

        if xml_report.xml_well_formed:
            well_formed += 1
        if xml_report.root_is_tool:
            tool_root_count += 1
            artifact_valid_count += 1
        if xml_report.unknown_datatypes:
            unknown_count += 1

    summary = EvaluationSummary(
        created_at=utc_now_iso(),
        total_wrappers=len(wrapper_paths),
        xml_well_formed_count=well_formed,
        tool_root_count=tool_root_count,
        wrappers_with_unknown_datatypes=unknown_count,
        artifact_format=artifact_format,
        artifact_valid_count=artifact_valid_count,
        yaml_well_formed_count=yaml_well_formed_count,
        udt_schema_valid_count=udt_schema_valid_count,
        user_tool_root_count=user_tool_root_count,
        xsd_status=_aggregate_status(xsd_statuses, default="not_run"),
        planemo_status=_aggregate_status(planemo_statuses, default="not_run"),
        planemo_test_status=_aggregate_status(planemo_test_statuses, default="not_run"),
        mean_structural_score=(
            round(sum(structural_scores) / len(structural_scores), 4) if structural_scores else 0.0
        ),
        wrapper_reports=per_wrapper,
        notes=[],
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary
