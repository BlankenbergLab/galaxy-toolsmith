from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from galaxy_toolsmith.inference.datatypes import known_galaxy_datatypes

KNOWN_GALAXY_DATATYPES = known_galaxy_datatypes()


@dataclass(frozen=True)
class ValidationReport:
    xml_well_formed: bool
    root_tag: str
    root_is_tool: bool
    unknown_datatypes: list[str]
    xsd_status: str
    planemo_status: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanemoTestOptions:
    output_dir: Path | None = None
    timeout_seconds: int = 0
    galaxy_root: Path | None = None
    install_galaxy: bool = False
    engine: str = ""
    conda_prefix: Path | None = None
    test_data: Path | None = None
    extra_tools: tuple[Path, ...] = ()
    no_dependency_resolution: bool = False


def validate_wrapper(xml_wrapper: str) -> ValidationReport:
    try:
        root = ET.fromstring(xml_wrapper)
    except ET.ParseError as error:
        return ValidationReport(
            xml_well_formed=False,
            root_tag="",
            root_is_tool=False,
            unknown_datatypes=[],
            xsd_status="not_run",
            planemo_status="not_run",
            notes=[f"XML parse error: {error}"],
        )

    root_tag = root.tag
    root_is_tool = root_tag == "tool"
    datatypes = set()
    for param in root.findall(".//inputs//param"):
        if param.attrib.get("type") in {"data", "data_collection"}:
            value = param.attrib.get("format") or param.attrib.get("ext")
            _add_datatypes(datatypes, value)
    for data in root.findall(".//outputs//data"):
        value = data.attrib.get("format") or data.attrib.get("ext")
        _add_datatypes(datatypes, value)
    for collection in root.findall(".//outputs//collection"):
        value = collection.attrib.get("format") or collection.attrib.get("type")
        _add_datatypes(datatypes, value)

    unknown = sorted(dtype for dtype in datatypes if dtype not in KNOWN_GALAXY_DATATYPES)
    notes = []
    if not root_is_tool:
        notes.append(f"Expected root <tool>; found <{root_tag}>.")
    if unknown:
        notes.append("Unknown datatypes detected; create Galaxy datatype scaffold/TODO entries.")

    return ValidationReport(
        xml_well_formed=True,
        root_tag=root_tag,
        root_is_tool=root_is_tool,
        unknown_datatypes=unknown,
        xsd_status="not_run",
        planemo_status="not_run",
        notes=notes,
    )


def _add_datatypes(datatypes: set[str], value: str | None) -> None:
    if not value:
        return
    datatypes.update(dtype.strip() for dtype in value.split(",") if dtype.strip())


def run_xsd_validation(wrapper_path: Path, xsd_path: Path | None) -> tuple[str, str]:
    if xsd_path is None:
        return "not_configured", "No XSD path provided."
    if not xsd_path.exists():
        return "not_configured", f"XSD path does not exist: {xsd_path}"
    if shutil.which("xmllint") is None:
        return "not_available", "xmllint not available on PATH."

    result = subprocess.run(
        ["xmllint", "--noout", "--schema", str(xsd_path), str(wrapper_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return "passed", ""
    message = (result.stderr or result.stdout).strip()
    return "failed", message


def resolve_planemo_executable() -> str | None:
    planemo = shutil.which("planemo")
    if planemo:
        return planemo
    bin_dir = Path(sys.executable).resolve().parent
    for executable in ("planemo", "planemo.exe"):
        candidate = bin_dir / executable
        if candidate.is_file():
            return str(candidate)
    return None


def run_planemo_lint(wrapper_path: Path, enabled: bool) -> tuple[str, str]:
    if not enabled:
        return "not_run", "Planemo lint disabled."
    planemo = resolve_planemo_executable()
    if planemo is None:
        return "not_available", "planemo not available on PATH or current Python environment."

    result = subprocess.run(
        [planemo, "lint", str(wrapper_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return "passed", ""
    message = (result.stderr or result.stdout).strip()
    return "failed", message


def run_planemo_test(
    wrapper_path: Path,
    *,
    enabled: bool,
    options: PlanemoTestOptions | None = None,
) -> tuple[str, str, dict[str, str]]:
    if not enabled:
        return "not_run", "Planemo test disabled.", {}
    planemo = resolve_planemo_executable()
    if planemo is None:
        return "not_available", "planemo not available on PATH or current Python environment.", {}

    selected_options = options or PlanemoTestOptions()
    output_dir = selected_options.output_dir or wrapper_path.parent / "planemo-test"
    job_output_dir = output_dir / "job_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "output_dir": str(output_dir),
        "output_html": str(output_dir / "tool_test_output.html"),
        "output_json": str(output_dir / "tool_test_output.json"),
        "output_text": str(output_dir / "tool_test_output.txt"),
        "job_output_files": str(job_output_dir),
    }

    command = [
        planemo,
        "test",
        str(wrapper_path),
        "--test_output",
        artifacts["output_html"],
        "--test_output_json",
        artifacts["output_json"],
        "--test_output_text",
        artifacts["output_text"],
        "--job_output_files",
        artifacts["job_output_files"],
    ]
    if selected_options.timeout_seconds > 0:
        command.extend(["--test_timeout", str(selected_options.timeout_seconds)])
    if selected_options.galaxy_root is not None:
        command.extend(["--galaxy_root", str(selected_options.galaxy_root)])
    if selected_options.install_galaxy:
        command.append("--install_galaxy")
    if selected_options.engine:
        command.extend(["--engine", selected_options.engine])
    if selected_options.conda_prefix is not None:
        command.extend(["--conda_prefix", str(selected_options.conda_prefix)])
    if selected_options.test_data is not None:
        command.extend(["--test_data", str(selected_options.test_data)])
    for extra_tool in selected_options.extra_tools:
        command.extend(["--extra_tools", str(extra_tool)])
    if selected_options.no_dependency_resolution:
        command.append("--no_dependency_resolution")

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return "passed", "", artifacts
    message = (result.stderr or result.stdout).strip()
    return "failed", message, artifacts
