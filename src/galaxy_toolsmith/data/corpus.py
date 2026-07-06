from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import warnings
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlparse
from xml.etree import ElementTree as ET

import requests
import yaml
from galaxy.tool_util.loader import load_tool_with_refereces
from galaxy.util import xml_to_string
from jinja2 import Environment, StrictUndefined, TemplateError, Undefined

from galaxy_toolsmith.http_client import (
    browser_fallback_user_agent,
    http_user_agent,
    should_retry_with_browser_user_agent,
    urlopen_with_user_agent_fallback,
    user_agent_header_attempts,
)
from galaxy_toolsmith.inference.datatypes import known_galaxy_datatypes
from galaxy_toolsmith.runtime.status import emit_status

DEFAULT_WRAPPER_SOURCE_MAX_BYTES = 256_000
DEFAULT_WRAPPER_CONFIGFILE_MAX_BYTES = 256_000


@dataclass(frozen=True)
class ExtractionSettings:
    max_workers: int = 4
    source_workers: int = 0
    container_prepare_workers: int = 2
    container_probe_workers: int = 4
    retries: int = 3
    retry_backoff_seconds: float = 0.5
    fetch_documentation: bool = True
    resolve_containers: bool = False
    expand_macros: bool = True
    execute_containers: bool = False
    container_runtime: str = "auto"
    container_cache_dir: Path | None = None
    singularity_depot_url: str = "https://depot.galaxyproject.org/singularity"
    docker_use_sudo: bool = False
    remove_images_after_use: bool = True
    container_sif_exec_mode: str = "auto"
    container_help_probe_mode: str = "exploratory"
    container_no_arg_timeout_seconds: int = 20
    container_run_timeout_seconds: int = 120
    container_pull_timeout_seconds: int = 300
    container_image_timeout_seconds: int = 300
    container_image_quarantine_seconds: int = 86_400
    container_image_quarantine_file: Path | None = None
    source_download_timeout_seconds: int = 60
    source_download_max_bytes: int = 0
    bioconda_checkout_sources: bool = False
    bioconda_ref: str = "master"
    synthesize_udt_yaml: bool = False
    wrapper_source_max_bytes: int = DEFAULT_WRAPPER_SOURCE_MAX_BYTES
    wrapper_configfile_max_bytes: int = DEFAULT_WRAPPER_CONFIGFILE_MAX_BYTES
    cache_root: Path | None = None
    restart: bool = False
    status_log_path: Path | None = None
    retry_manifest_path: Path | None = None
    extract_progress_interval_seconds: float = 30.0
    run_id: str = ""


@dataclass
class ToolRecord:
    # identity
    schema_version: str = "0.6.0"
    package_id: str = ""
    tool_name: str = ""
    tool_id: str = ""
    tool_dir: str = ""
    wrapper_path: str = ""
    xml_files: list[str] = field(default_factory=list)
    udt_yaml_path: str = ""
    udt_yaml_files: list[str] = field(default_factory=list)

    # suite/.shed metadata
    shed_name: str = ""
    shed_owner: str = ""
    shed_description: str = ""
    shed_homepage_url: str = ""
    shed_remote_repository_url: str = ""
    shed_categories: list[str] = field(default_factory=list)
    suite_id: str = ""
    suite_name: str = ""
    suite_members: list[str] = field(default_factory=list)
    is_suite_root: bool = False

    # content / derived
    help_text: str = ""
    original_help_text: str = ""
    container_help_text: str = ""
    container_usage_text: str = ""
    container_api_validation: list[dict] = field(default_factory=list)
    documentation: str = ""
    expanded_xml_path: str = ""
    macro_files: list[str] = field(default_factory=list)
    uses_macros: bool = False
    macro_expansion_status: str = "not_applicable"
    version_command_text: str = ""
    primary_command: str = ""
    subcommands: list[str] = field(default_factory=list)
    invocation_patterns: list[str] = field(default_factory=list)
    command_text: str = ""
    wrapper_helper_files: list[dict] = field(default_factory=list)
    wrapper_configfiles: list[dict] = field(default_factory=list)
    wrapper_sidecar_files: list[dict] = field(default_factory=list)
    wrapper_source_summary: dict = field(default_factory=dict)

    # datatype / tests
    input_parameter_types: list[str] = field(default_factory=list)
    input_datatypes: list[str] = field(default_factory=list)
    output_datatypes: list[str] = field(default_factory=list)
    datatype_report: dict = field(default_factory=dict)
    tests: list[dict] = field(default_factory=list)
    test_data_files: list[str] = field(default_factory=list)

    # requirements / containers
    requirement_packages: list[str] = field(default_factory=list)
    requirement_versions: dict[str, str] = field(default_factory=dict)
    container_candidates: list[str] = field(default_factory=list)
    container_candidate_details: list[dict] = field(default_factory=list)
    selected_container: str = ""
    selected_container_runtime: str = ""
    container_execution: list[dict] = field(default_factory=list)
    bioconda_sources: list[dict] = field(default_factory=list)
    version_consistency: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


KNOWN_GALAXY_DATATYPES = known_galaxy_datatypes()
_BIOCONTAINER_LOOKUP_CACHE: dict[str, list[str]] = {}
_QUAY_TAGS_CACHE: dict[tuple[str, str, str], list[str]] = {}
_CONDA_FORGE_FEEDSTOCK_CACHE: dict[tuple[str, str], tuple[Path | None, str, str]] = {}
_CONDA_FORGE_FEEDSTOCK_CACHE_LOCK = threading.Lock()
_SOURCE_COMMAND_HINT_CACHE: dict[tuple[str, str], list[str]] = {}
_SOURCE_COMMAND_HINT_CACHE_LOCK = threading.Lock()
_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)
_RECIPE_SET_RE = re.compile(r"{%\s*set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*%}")
_UNRESOLVED_TEMPLATE_RE = re.compile(r"({{.*?}}|{%.*?%}|{#.*?#})")
_CONDA_FORGE_FEEDSTOCK_ALIASES = {
    "libexpat": ("expat",),
    "libcurl": ("curl",),
    "libsqlite": ("sqlite",),
    "matplotlib-base": ("matplotlib",),
    "matplotlib_base": ("matplotlib",),
    "seaborn-base": ("seaborn",),
    "seaborn_base": ("seaborn",),
}
_XML_BRACED_REF_RE = re.compile(r"(?<![$\\])\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_XML_SIMPLE_REF_RE = re.compile(r"(?<![$\\])\$([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_.])")
_TOOL_DIRECTORY_RE = re.compile(r"\$?\{?__tool_directory__\}?")
_COMMAND_VARIABLE_RE = re.compile(r"\$+\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
_WRAPPER_SOURCE_MAX_BYTES = DEFAULT_WRAPPER_SOURCE_MAX_BYTES
_WRAPPER_CONFIGFILE_MAX_BYTES = DEFAULT_WRAPPER_CONFIGFILE_MAX_BYTES
_CONFIGFILE_TRUNCATION_MARKER = "\n[truncated configfile content]\n"
_HTTP_BROWSER_FALLBACK_MARKER = ".gtsm-http-browser-fallback.json"
_WRAPPER_SOURCE_EXTENSIONS = {
    ".awk",
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cwl",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".jl",
    ".js",
    ".json",
    ".lua",
    ".md",
    ".nf",
    ".pl",
    ".pm",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".wdl",
    ".yaml",
    ".yml",
}
_WRAPPER_SCRIPT_CONFIGFILE_EXTENSIONS = {
    ".awk",
    ".bash",
    ".jl",
    ".js",
    ".lua",
    ".pl",
    ".pm",
    ".py",
    ".r",
    ".rb",
    ".sh",
    ".ts",
}
_WRAPPER_CONFIG_TEMPLATE_EXTENSIONS = {
    ".cfg",
    ".ini",
    ".json",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_WRAPPER_BINARY_OR_DATA_EXTENSIONS = {
    ".7z",
    ".a",
    ".bam",
    ".bcf",
    ".bz2",
    ".cram",
    ".db",
    ".dll",
    ".dylib",
    ".fa",
    ".fasta",
    ".fastq",
    ".fq",
    ".gz",
    ".h5",
    ".jpg",
    ".jpeg",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".pyd",
    ".pkl",
    ".sam",
    ".so",
    ".sqlite",
    ".tar",
    ".tgz",
    ".tiff",
    ".vcf",
    ".xz",
    ".zip",
}
_CONFIGFILE_API_MODULE_ROOTS = {
    "alphagenome",
    "alphagenome_research",
    "easy_vitessce",
    "episcanpy",
    "liana",
    "scanpy",
}
_CONFIGFILE_API_CALL_LIMIT = 256
_CONFIGFILE_PARAMETER_DOC_LIMIT = 256
_CONFIGFILE_COMMAND_DOC_LINE_LIMIT = 120
_CONFIGFILE_HELP_CONTEXT_API_LIMIT = 24
_CONFIGFILE_API_DOC_LIMIT = 24
_CONFIGFILE_API_DOCSTRING_LIMIT = 240
_CONFIGFILE_HELP_CONTEXT_COMMAND_DOC_LIMIT = 4
_CONFIGFILE_HELP_CONTEXT_PARAMETER_LIMIT = 24
_CONFIGFILE_HELP_CONTEXT_PARAMETER_PER_FILE_LIMIT = 8


def _is_generic_container_image_key(image_key: str) -> bool:
    if not image_key:
        return True
    generic_keys = {_normalized_command_key(name) for name in _GENERIC_CONTAINER_NAMES}
    return image_key in generic_keys or image_key.startswith("mulledv2")


@dataclass(frozen=True)
class MulledTarget:
    package: str
    version: str = ""
    build: str = ""


@dataclass(frozen=True)
class ContainerRuntime:
    name: str
    executable: str


@dataclass(frozen=True)
class ContainerPreparation:
    ok: bool
    runtime: str
    image: str
    identifier: str = ""
    source: str = ""
    command: list[str] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    error_text: str = ""


@dataclass(frozen=True)
class BiocondaRecipeSnapshot:
    package: str
    recipe_path: str
    meta_text: str
    recipe_version: str = ""
    commit: str = ""
    commit_date: str = ""
    selection_reason: str = ""
    scanned_commits: int = 0
    error: str = ""


@dataclass(frozen=True)
class SemverishVersion:
    raw: str
    normalized: str
    numeric: tuple[int, ...]


@dataclass(frozen=True)
class ContainerCandidate:
    image: str
    source: str
    packages: tuple[str, ...] = ()
    priority: int = 0
    status: str = "ok"
    error_text: str = ""


@dataclass
class ContainerExecutionState:
    runtimes: list[ContainerRuntime] = field(default_factory=list)
    preparation_semaphore: threading.Semaphore | None = None
    planned_images: set[str] = field(default_factory=set)
    preparation_cache: dict[str, ContainerPreparation] = field(default_factory=dict)
    preparation_alias_cache: dict[str, ContainerPreparation] = field(default_factory=dict)
    failed_preparation_cache: dict[str, ContainerPreparation] = field(default_factory=dict)
    failed_preparation_at: dict[str, float] = field(default_factory=dict)
    image_quarantine_store: ContainerImageQuarantineStore | None = None
    preparation_locks: dict[str, threading.Lock] = field(default_factory=dict)
    preparation_locks_guard: threading.Lock = field(default_factory=threading.Lock)
    command_presence_lock: threading.Lock = field(default_factory=threading.Lock)
    command_presence_cache: dict[tuple[str, ...], subprocess.CompletedProcess] = field(
        default_factory=dict
    )
    missing_command_status_emitted: set[tuple[str, ...]] = field(default_factory=set)
    images_pulled: int = 0
    images_prepared: int = 0
    images_removed: int = 0
    commands_executed: int = 0
    commands_failed: int = 0
    help_ok: int = 0
    help_degraded: int = 0
    usage_degraded: int = 0
    api_validation_ok: int = 0
    api_validation_failed: int = 0
    missing_command: int = 0
    non_help_output: int = 0
    failed_probe: int = 0
    prepare_failed: int = 0
    runtime_error: int = 0
    timeout: int = 0
    runtime_fallbacks: int = 0


@dataclass
class ContainerImageQuarantineStore:
    path: Path | None = None
    entries: dict[str, dict[str, object]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()
_STATUS_ONCE_KEYS: set[tuple[str, ...]] = set()
_STATUS_ONCE_LOCK = threading.Lock()


def _emit_extract_status(settings: ExtractionSettings, payload: dict) -> None:
    if settings.run_id and "run_id" not in payload:
        payload = {"run_id": settings.run_id, **payload}
    emit_status(payload, status_log_path=settings.status_log_path)


def _emit_extract_status_once(
    settings: ExtractionSettings, key: tuple[str, ...], payload: dict
) -> None:
    run_key = (settings.run_id or "global", *key)
    with _STATUS_ONCE_LOCK:
        if run_key in _STATUS_ONCE_KEYS:
            return
        _STATUS_ONCE_KEYS.add(run_key)
    _emit_extract_status(settings, payload)


def _cache_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def _command_display(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def discover_tool_dirs(tools_root: Path) -> list[Path]:
    return sorted(path.parent for path in tools_root.glob("**/.shed.yml"))


def _strip_text(text: str | None) -> str:
    return (text or "").strip()


def _extract_datatypes(root: ET.Element) -> tuple[list[str], list[str], list[str]]:
    param_types: set[str] = set()
    inputs: set[str] = set()
    outputs: set[str] = set()

    for param in root.findall(".//inputs//param"):
        param_type = param.attrib.get("type")
        if param_type:
            param_types.add(param_type)
        if param_type in {"data", "data_collection"}:
            value = param.attrib.get("format") or param.attrib.get("ext")
            if value:
                inputs.add(value)
    for data in root.findall(".//outputs//data"):
        value = data.attrib.get("format") or data.attrib.get("ext")
        if value:
            outputs.add(value)
    for collection in root.findall(".//outputs//collection"):
        value = collection.attrib.get("format") or collection.attrib.get("type")
        if value:
            outputs.add(value)
    return sorted(param_types), sorted(inputs), sorted(outputs)


def _extract_tests(root: ET.Element) -> list[dict]:
    extracted: list[dict] = []
    for test in root.findall(".//tests//test"):
        output_assertions = []
        for output in test.findall("./output"):
            output_assertions.append(
                {
                    "name": output.attrib.get("name"),
                    "file": output.attrib.get("file"),
                    "ftype": output.attrib.get("ftype"),
                }
            )
        extracted.append(
            {
                "expect_num_outputs": test.attrib.get("expect_num_outputs"),
                "expect_exit_code": test.attrib.get("expect_exit_code"),
                "outputs": output_assertions,
            }
        )
    return extracted


def _extract_help(root: ET.Element) -> str:
    sections = []
    for help_node in root.findall(".//help"):
        text = _strip_text(help_node.text)
        if text:
            sections.append(text)
    return "\n\n".join(sections)


def _extract_requirements(root: ET.Element) -> tuple[list[str], dict[str, str], list[str]]:
    packages: set[str] = set()
    versions: dict[str, str] = {}
    containers: set[str] = set()
    for requirement in root.findall(".//requirements//requirement"):
        req_type = _strip_text(requirement.attrib.get("type")) or "package"
        if req_type != "package":
            continue
        text = _strip_text(requirement.text)
        if text:
            packages.add(text)
            version = _strip_text(requirement.attrib.get("version"))
            if version:
                versions[text] = version
    for container in root.findall(".//requirements//container"):
        text = _strip_text(container.text)
        if text:
            containers.add(text)
    return sorted(packages), versions, sorted(containers)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_within_directory(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _is_wrapper_test_data_path(relpath: Path) -> bool:
    return any(part in {"test", "test-data", "test_data", "tests"} for part in relpath.parts)


def _wrapper_source_path_skip_reason(path: Path, tool_dir: Path, max_bytes: int) -> str:
    try:
        resolved_tool_dir = tool_dir.resolve()
        resolved = path.resolve(strict=False)
    except OSError as error:
        return f"path_error:{error}"
    if not _is_within_directory(resolved, resolved_tool_dir):
        return "outside_tool_dir"
    try:
        relpath = resolved.relative_to(resolved_tool_dir)
    except ValueError:
        return "outside_tool_dir"
    if _is_wrapper_test_data_path(relpath):
        return "test_data"
    if path.is_symlink():
        return "symlink"
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "not_file"
    suffix = path.suffix.lower()
    if suffix in _WRAPPER_BINARY_OR_DATA_EXTENSIONS:
        return "binary_or_data"
    if suffix not in _WRAPPER_SOURCE_EXTENSIONS:
        return "unsupported_extension"
    try:
        stat = path.stat()
    except OSError as error:
        return f"stat_error:{error}"
    if stat.st_size > max(1, max_bytes):
        return "too_large"
    try:
        chunk = path.read_bytes()[:8192]
    except OSError as error:
        return f"read_error:{error}"
    if b"\x00" in chunk:
        return "binary_content"
    if chunk:
        control_count = sum(1 for byte in chunk if byte < 32 and byte not in b"\n\r\t\f\b")
        if control_count / len(chunk) >= 0.05:
            return "binary_content"
    return ""


def _helper_candidate_token_values(token: str) -> list[str]:
    cleaned = _clean_command_token(token).strip(",")
    if not cleaned:
        return []
    values = [cleaned]
    if "=" in cleaned:
        option_value = cleaned.split("=", 1)[1].strip()
        if option_value:
            values.append(option_value)
    return values


def _helper_candidate_path_from_token(token: str, tool_dir: Path) -> tuple[Path | None, str]:
    cleaned = _clean_command_token(token).strip(",")
    if not cleaned:
        return None, "empty"
    uses_tool_directory = "__tool_directory__" in cleaned
    normalized = _TOOL_DIRECTORY_RE.sub(".", cleaned)
    if "$" in normalized or "{" in normalized or "}" in normalized:
        return None, "dynamic"
    if normalized.startswith("-"):
        return None, "option"
    if not uses_tool_directory and "/" not in normalized and "\\" not in normalized:
        suffix = Path(normalized).suffix.lower()
        if suffix not in _WRAPPER_SOURCE_EXTENSIONS:
            return None, "not_path"
    candidate = Path(normalized).expanduser()
    path = candidate if candidate.is_absolute() else tool_dir / candidate
    suffix = path.suffix.lower()
    if suffix not in _WRAPPER_SOURCE_EXTENSIONS:
        return path, "unsupported_extension"
    return path, ""


def _helper_candidate_paths_from_command(command_text: str, tool_dir: Path) -> tuple[list[Path], dict[str, int]]:
    candidates: list[Path] = []
    skipped: dict[str, int] = {}
    seen: set[str] = set()
    for raw_line in command_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for segment in _command_segments(line):
            for token in _command_tokens(segment):
                for value in _helper_candidate_token_values(token):
                    path, skip_reason = _helper_candidate_path_from_token(value, tool_dir)
                    if skip_reason and skip_reason not in {"not_path", "option", "dynamic"}:
                        skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
                    if path is None:
                        continue
                    key = str(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(path)
    return candidates, skipped


def _wrapper_helper_record(path: Path, tool_dir: Path) -> dict:
    text_bytes = path.read_bytes()
    text = text_bytes.decode("utf-8", errors="ignore")
    relpath = path.resolve().relative_to(tool_dir.resolve()).as_posix()
    extension = path.suffix.lower()
    language = _configfile_language_hint(extension)
    return {
        "path": str(path),
        "relative_path": relpath,
        "extension": extension,
        "byte_count": len(text_bytes),
        "sha256": _sha256_bytes(text_bytes),
        "role_hint": "command_reference",
        "language": language,
        "api_calls": _extract_configfile_api_calls(text, language),
        "command_docs": _extract_configfile_command_docs(text),
        "parameter_docs": _extract_configfile_parameter_docs(text),
    }


def _extract_wrapper_helper_files(
    tool_dir: Path,
    command_text: str,
    *,
    max_bytes: int,
) -> tuple[list[dict], dict[str, int]]:
    candidates, skipped = _helper_candidate_paths_from_command(command_text, tool_dir)
    helpers: list[dict] = []
    seen: set[str] = set()
    for path in candidates:
        skip_reason = _wrapper_source_path_skip_reason(path, tool_dir, max_bytes)
        if skip_reason:
            skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        helpers.append(_wrapper_helper_record(path, tool_dir))
    return helpers, skipped


def _configfile_content(configfile: ET.Element) -> str:
    parts: list[str] = []
    if configfile.text:
        parts.append(configfile.text)
    for child in list(configfile):
        parts.append(ET.tostring(child, encoding="unicode"))
    return "".join(parts).strip("\n")


def _truncate_text_to_utf8_bytes(text: str, max_bytes: int, marker: str) -> tuple[str, bool]:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text, False
    marker_bytes = marker.encode("utf-8")
    if len(marker_bytes) >= max_bytes:
        return marker_bytes[:max_bytes].decode("utf-8", errors="ignore"), True
    prefix = raw[: max_bytes - len(marker_bytes)].decode("utf-8", errors="ignore")
    return prefix.rstrip("\n") + marker, True


def _bounded_configfile_content(
    content: str,
    *,
    max_bytes: int,
) -> tuple[str, int, int, str, bool]:
    full_bytes = content.encode("utf-8", errors="replace")
    stored_content, truncated = _truncate_text_to_utf8_bytes(
        content,
        max(1, max_bytes),
        _CONFIGFILE_TRUNCATION_MARKER,
    )
    stored_byte_count = len(stored_content.encode("utf-8", errors="replace"))
    return stored_content, len(full_bytes), stored_byte_count, _sha256_bytes(full_bytes), truncated


def _configfile_extension(name: str, filename: str) -> str:
    suffix = Path(filename or name).suffix.lower()
    return suffix


def _configfile_language_hint(extension: str) -> str:
    return {
        ".awk": "awk",
        ".bash": "bash",
        ".cfg": "config",
        ".ini": "ini",
        ".jl": "julia",
        ".js": "javascript",
        ".json": "json",
        ".lua": "lua",
        ".pl": "perl",
        ".pm": "perl",
        ".py": "python",
        ".r": "r",
        ".rb": "ruby",
        ".sh": "shell",
        ".toml": "toml",
        ".ts": "typescript",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(extension, "")


def _command_references_configfile_script(command_text: str, name: str, filename: str) -> bool:
    if not command_text.strip():
        return False
    references = [value for value in (name, filename) if value]
    if not references:
        return False
    lowered_command = command_text.lower()
    if not any(value in command_text or f"${value}" in command_text for value in references):
        return False
    return bool(
        re.search(
            r"\b(?:python(?:3(?:\.\d+)?)?|rscript|r\s+--vanilla|bash|sh|perl|julia|node)\b",
            lowered_command,
        )
    )


def _looks_like_script_configfile_content(content: str) -> bool:
    stripped = content.lstrip()
    if not stripped:
        return False
    first_line = stripped.splitlines()[0]
    if first_line.startswith("#!"):
        return True
    return bool(
        re.search(r"^\s*(?:import|from)\s+[A-Za-z_][A-Za-z0-9_.]*", content, flags=re.M)
        or re.search(r"\b(?:library|require|suppressPackageStartupMessages)\s*\(", content)
        or re.search(r"\b(?:argparse|OptionParser|optparse)\b", content)
    )


def _configfile_template_kind(
    extension: str,
    content: str,
    *,
    name: str = "",
    filename: str = "",
    command_text: str = "",
) -> str:
    first_line = content.lstrip().splitlines()[0] if content.strip() else ""
    if (
        extension in _WRAPPER_SCRIPT_CONFIGFILE_EXTENSIONS
        or first_line.startswith("#!")
        or "argparse" in content
        or "suppressPackageStartupMessages" in content
        or (
            not extension
            and _command_references_configfile_script(command_text, name, filename)
            and _looks_like_script_configfile_content(content)
        )
    ):
        return "script_template"
    if extension in _WRAPPER_CONFIG_TEMPLATE_EXTENSIONS or content.strip():
        return "config_template"
    return "other_template"


def _command_reference_names(command_text: str) -> set[str]:
    names: set[str] = set()
    for match in _COMMAND_VARIABLE_RE.finditer(command_text):
        name = match.group(1).strip()
        if name:
            names.add(name)
    return names


def _python_api_aliases(content: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    import_as_re = re.compile(
        r"^\s*import\s+(?P<module>[A-Za-z_][A-Za-z0-9_.]*)\s+as\s+"
        r"(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*$"
    )
    import_re = re.compile(r"^\s*import\s+(?P<module>[A-Za-z_][A-Za-z0-9_.]*)\s*$")
    from_import_re = re.compile(
        r"^\s*from\s+(?P<module>[A-Za-z_][A-Za-z0-9_.]*)\s+import\s+"
        r"(?P<names>[A-Za-z0-9_,\s]+)\s*$"
    )
    for line in content.splitlines():
        match = import_as_re.match(line)
        if match:
            module = match.group("module")
            root = module.split(".", 1)[0]
            if root in _CONFIGFILE_API_MODULE_ROOTS:
                aliases[match.group("alias")] = module
            continue
        match = import_re.match(line)
        if match:
            module = match.group("module")
            root = module.split(".", 1)[0]
            if root in _CONFIGFILE_API_MODULE_ROOTS:
                aliases[root] = module
            continue
        match = from_import_re.match(line)
        if not match:
            continue
        module = match.group("module")
        root = module.split(".", 1)[0]
        if root not in _CONFIGFILE_API_MODULE_ROOTS:
            continue
        for name in re.split(r"\s*,\s*", match.group("names").strip()):
            if name:
                aliases[name] = f"{module}.{name}"
    return aliases


def _source_line_for_offset(content: str, offset: int) -> tuple[int, str]:
    line_number = content.count("\n", 0, max(0, offset)) + 1
    lines = content.splitlines()
    line_text = lines[line_number - 1].strip() if 0 < line_number <= len(lines) else ""
    return line_number, _tail_text(line_text, limit=240)


def _extract_python_configfile_api_calls(content: str) -> list[dict]:
    aliases = _python_api_aliases(content)
    if not aliases:
        return []
    calls: list[dict] = []
    seen: set[str] = set()
    call_re = re.compile(
        r"\b(?P<alias>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?P<chain>(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\("
    )
    for match in call_re.finditer(content):
        alias = match.group("alias")
        module = aliases.get(alias)
        if not module:
            continue
        chain = match.group("chain")
        call = f"{alias}{chain}"
        qualified_call = f"{module}{chain}"
        if qualified_call in seen:
            continue
        seen.add(qualified_call)
        line_number, line_text = _source_line_for_offset(content, match.start())
        calls.append(
            {
                "language": "python",
                "module": module.split(".", 1)[0],
                "alias": alias,
                "call": call,
                "qualified_call": qualified_call,
                "line": line_number,
                "line_text": line_text,
            }
        )
        if len(calls) >= _CONFIGFILE_API_CALL_LIMIT:
            break
    return calls


def _extract_r_configfile_api_calls(content: str) -> list[dict]:
    calls: list[dict] = []
    seen: set[str] = set()
    call_re = re.compile(
        r"\b(?P<module>[A-Za-z_][A-Za-z0-9_.]*)::(?P<name>[A-Za-z_][A-Za-z0-9_.]*)\s*\("
    )
    for match in call_re.finditer(content):
        qualified_call = f"{match.group('module')}::{match.group('name')}"
        if qualified_call in seen:
            continue
        seen.add(qualified_call)
        line_number, line_text = _source_line_for_offset(content, match.start())
        calls.append(
            {
                "language": "r",
                "module": match.group("module"),
                "alias": "",
                "call": qualified_call,
                "qualified_call": qualified_call,
                "line": line_number,
                "line_text": line_text,
            }
        )
        if len(calls) >= _CONFIGFILE_API_CALL_LIMIT:
            break
    return calls


def _extract_configfile_api_calls(content: str, language: str) -> list[dict]:
    language = language.lower()
    if language == "python" or any(
        f"import {module}" in content for module in _CONFIGFILE_API_MODULE_ROOTS
    ):
        return _extract_python_configfile_api_calls(content)
    if language == "r" or "::" in content:
        return _extract_r_configfile_api_calls(content)
    return []


def _extract_configfile_parameter_docs(content: str) -> list[dict]:
    docs: list[dict] = []
    define_re = re.compile(r"^\s*#define\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<body>.*)$")
    for line in content.splitlines():
        match = define_re.match(line)
        if not match:
            continue
        body = match.group("body").strip()
        value_template = body.split("//", 1)[0].strip()
        comments = [part.strip() for part in body.split("//")[1:] if part.strip()]
        default = ""
        value_type = ""
        description_parts: list[str] = []
        for comment in comments:
            default_match = re.search(r"\bdefault\s*:\s*([^,\s/]+)", comment, flags=re.I)
            if default_match:
                default = default_match.group(1).strip()
                continue
            type_match = re.match(r"\(([^)]+)\)\s*(.*)", comment)
            if type_match:
                value_type = type_match.group(1).strip()
                if type_match.group(2).strip():
                    description_parts.append(type_match.group(2).strip())
                continue
            description_parts.append(comment)
        docs.append(
            {
                "name": match.group("name"),
                "value_template": value_template,
                "default": default,
                "type": value_type,
                "description": " ".join(description_parts).strip(),
            }
        )
        if len(docs) >= _CONFIGFILE_PARAMETER_DOC_LIMIT:
            break
    return docs


def _extract_configfile_command_docs(content: str) -> list[dict]:
    lines = content.splitlines()
    docs: list[dict] = []
    for index, line in enumerate(lines):
        if "command line options" not in line.lower():
            continue
        collected = [line.strip()]
        for next_line in lines[index + 1 :]:
            stripped = next_line.strip()
            if not stripped and len(collected) > 1:
                break
            if not stripped:
                continue
            collected.append(stripped)
            if len(collected) >= _CONFIGFILE_COMMAND_DOC_LINE_LIMIT:
                break
        docs.append(
            {
                "kind": "command_line_options",
                "line": index + 1,
                "text": "\n".join(collected).strip(),
            }
        )
    return docs


def _extract_wrapper_configfiles(
    root: ET.Element,
    command_text: str,
    *,
    max_bytes: int,
) -> list[dict]:
    reference_names = _command_reference_names(command_text)
    configfiles: list[dict] = []
    for index, configfile in enumerate(root.findall(".//configfiles//configfile"), start=1):
        attributes = {str(key): str(value) for key, value in sorted(configfile.attrib.items())}
        name = _strip_text(configfile.attrib.get("name"))
        filename = _strip_text(configfile.attrib.get("filename"))
        raw_content = _configfile_content(configfile)
        content, byte_count, stored_byte_count, sha256, content_truncated = (
            _bounded_configfile_content(raw_content, max_bytes=max_bytes)
        )
        extension = _configfile_extension(name, filename)
        language = _configfile_language_hint(extension)
        referenced = bool(
            (name and name in reference_names)
            or (filename and filename in command_text)
            or (name and f"${name}" in command_text)
        )
        template_kind = _configfile_template_kind(
            extension,
            raw_content,
            name=name,
            filename=filename,
            command_text=command_text,
        )
        api_calls = _extract_configfile_api_calls(raw_content, language)
        command_docs = _extract_configfile_command_docs(raw_content)
        parameter_docs = _extract_configfile_parameter_docs(raw_content)
        configfiles.append(
            {
                "name": name or f"configfile_{index}",
                "filename": filename,
                "attributes": attributes,
                "extension": extension,
                "language": language,
                "byte_count": byte_count,
                "stored_byte_count": stored_byte_count,
                "sha256": sha256,
                "content_truncated": content_truncated,
                "template_kind": template_kind,
                "role_hint": template_kind,
                "referenced_by_command": referenced,
                "api_calls": api_calls,
                "command_docs": command_docs,
                "parameter_docs": parameter_docs,
                "content": content,
            }
        )
    return configfiles


def _wrapper_source_summary(
    helper_files: list[dict],
    configfiles: list[dict],
    sidecar_files: list[dict],
    skipped: dict[str, int],
) -> dict:
    return {
        "helper_file_count": len(helper_files),
        "configfile_count": len(configfiles),
        "sidecar_file_count": len(sidecar_files),
        "macro_sidecar_count": sum(1 for item in sidecar_files if item.get("role") == "macros"),
        "tool_data_sidecar_count": sum(
            1 for item in sidecar_files if str(item.get("role", "")).startswith("tool_data")
        ),
        "truncated_configfile_count": sum(
            1 for item in configfiles if item.get("content_truncated")
        ),
        "helper_api_call_count": sum(
            len(item.get("api_calls", []) or []) for item in helper_files
        ),
        "helper_command_doc_count": sum(
            len(item.get("command_docs", []) or []) for item in helper_files
        ),
        "helper_parameter_doc_count": sum(
            len(item.get("parameter_docs", []) or []) for item in helper_files
        ),
        "configfile_api_call_count": sum(
            len(item.get("api_calls", []) or []) for item in configfiles
        ),
        "configfile_command_doc_count": sum(
            len(item.get("command_docs", []) or []) for item in configfiles
        ),
        "configfile_parameter_doc_count": sum(
            len(item.get("parameter_docs", []) or []) for item in configfiles
        ),
        "api_backed_wrapper": any(item.get("api_calls") for item in configfiles)
        or any(item.get("api_calls") for item in helper_files),
        "skipped_file_count": sum(skipped.values()),
        "skip_reasons": dict(sorted(skipped.items())),
    }


def _xml_file_root_tag(path: Path) -> str:
    try:
        return str(ET.parse(path).getroot().tag)
    except (ET.ParseError, OSError):
        return ""


def _sidecar_role(path: Path, root_tag: str = "") -> str:
    name = path.name.lower()
    rel = path.as_posix().lower()
    if root_tag == "macros" or "macro" in name:
        return "macros"
    if root_tag == "tables" or name.startswith("tool_data_table_conf"):
        return "tool_data_table_conf"
    if name.endswith(".loc.sample") or rel.endswith(".loc.sample"):
        return "tool_data_loc_sample"
    return ""


def _sidecar_candidate_paths(tool_dir: Path, macro_files_rel: list[str]) -> list[Path]:
    candidates: list[Path] = [tool_dir / rel for rel in macro_files_rel]
    candidates.extend(sorted(tool_dir.glob("tool_data_table_conf*.xml")))
    candidates.extend(sorted(tool_dir.glob("*.loc.sample")))
    candidates.extend(sorted((tool_dir / "tool-data").glob("*.loc.sample")))
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in candidates:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _extract_wrapper_sidecar_files(
    tool_dir: Path,
    *,
    primary_xml: Path,
    macro_files_rel: list[str],
    max_bytes: int,
) -> list[dict]:
    sidecars: list[dict] = []
    for path in _sidecar_candidate_paths(tool_dir, macro_files_rel):
        if not path.exists() or not path.is_file() or path.resolve() == primary_xml.resolve():
            continue
        if not _is_within_directory(path, tool_dir):
            continue
        try:
            raw_bytes = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw_bytes[:8192]:
            continue
        text = raw_bytes.decode("utf-8", errors="replace")
        root_tag = _xml_file_root_tag(path) if path.suffix.lower() == ".xml" else ""
        role = _sidecar_role(path.relative_to(tool_dir), root_tag=root_tag)
        if not role:
            continue
        content, byte_count, stored_byte_count, sha256, content_truncated = (
            _bounded_configfile_content(text, max_bytes=max_bytes)
        )
        sidecars.append(
            {
                "path": str(path),
                "relative_path": path.relative_to(tool_dir).as_posix(),
                "role": role,
                "root_tag": root_tag,
                "byte_count": byte_count,
                "stored_byte_count": stored_byte_count,
                "sha256": sha256,
                "content_truncated": content_truncated,
                "content": content,
            }
        )
    return sidecars


def _configfile_display_name(configfile: dict) -> str:
    return str(
        configfile.get("filename")
        or configfile.get("name")
        or configfile.get("relative_path")
        or "configfile"
    )


def _wrapper_source_display_name(source: dict) -> str:
    return str(
        source.get("filename")
        or source.get("name")
        or source.get("relative_path")
        or Path(str(source.get("path", "") or "")).name
        or "source"
    )


def _single_line_text(value: str, *, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + " [truncated]"


def _format_configfile_parameter_doc(doc: dict) -> str:
    name = str(doc.get("name", "") or "").strip()
    if not name:
        return ""
    details = []
    default = str(doc.get("default", "") or "").strip()
    value_type = str(doc.get("type", "") or "").strip()
    description = _single_line_text(str(doc.get("description", "") or ""), limit=120)
    if default:
        details.append(f"default={default}")
    if value_type:
        details.append(f"type={value_type}")
    if description:
        details.append(description)
    if details:
        return f"{name} ({'; '.join(details)})"
    return name


def _wrapper_configfile_help_context(configfiles: list[dict]) -> str:
    sections: list[str] = []
    api_lines: list[str] = []
    for configfile in configfiles:
        calls = configfile.get("api_calls", []) or []
        names = [
            str(call.get("qualified_call") or call.get("call") or "").strip()
            for call in calls
            if isinstance(call, dict)
            and str(call.get("qualified_call") or call.get("call") or "").strip()
        ]
        if not names:
            continue
        shown = names[:_CONFIGFILE_HELP_CONTEXT_API_LIMIT]
        suffix = ""
        if len(names) > len(shown):
            suffix = f" (+{len(names) - len(shown)} more)"
        api_lines.append(f"- {_configfile_display_name(configfile)}: {', '.join(shown)}{suffix}")
    if api_lines:
        sections.append("API calls used by generated wrapper scripts:\n" + "\n".join(api_lines))

    command_doc_lines: list[str] = []
    for configfile in configfiles:
        for doc in (configfile.get("command_docs", []) or []):
            if not isinstance(doc, dict):
                continue
            text = _single_line_text(str(doc.get("text", "") or ""), limit=700)
            if text:
                command_doc_lines.append(f"- {_configfile_display_name(configfile)}: {text}")
            if len(command_doc_lines) >= _CONFIGFILE_HELP_CONTEXT_COMMAND_DOC_LIMIT:
                break
        if len(command_doc_lines) >= _CONFIGFILE_HELP_CONTEXT_COMMAND_DOC_LIMIT:
            break
    if command_doc_lines:
        sections.append(
            "Command-line documentation embedded in wrapper configfiles:\n"
            + "\n".join(command_doc_lines)
        )

    parameter_lines: list[str] = []
    for configfile in configfiles:
        docs = [
            _format_configfile_parameter_doc(doc)
            for doc in (configfile.get("parameter_docs", []) or [])
            if isinstance(doc, dict)
        ]
        docs = [doc for doc in docs if doc]
        if not docs:
            continue
        shown = docs[:_CONFIGFILE_HELP_CONTEXT_PARAMETER_PER_FILE_LIMIT]
        suffix = ""
        if len(docs) > len(shown):
            suffix = f" (+{len(docs) - len(shown)} more)"
        parameter_lines.append(
            f"- {_configfile_display_name(configfile)}: " + "; ".join(shown) + suffix
        )
        if len(parameter_lines) >= _CONFIGFILE_HELP_CONTEXT_PARAMETER_LIMIT:
            break
    if parameter_lines:
        sections.append(
            "Parameter documentation embedded in wrapper configfiles:\n"
            + "\n".join(parameter_lines)
        )

    if not sections:
        return ""
    return "Wrapper configfile context:\n\n" + "\n\n".join(sections)


def _wrapper_helper_help_context(helper_files: list[dict]) -> str:
    sections: list[str] = []
    api_lines: list[str] = []
    for helper in helper_files:
        calls = helper.get("api_calls", []) or []
        names = [
            str(call.get("qualified_call") or call.get("call") or "").strip()
            for call in calls
            if isinstance(call, dict)
            and str(call.get("qualified_call") or call.get("call") or "").strip()
        ]
        if not names:
            continue
        shown = names[:_CONFIGFILE_HELP_CONTEXT_API_LIMIT]
        suffix = ""
        if len(names) > len(shown):
            suffix = f" (+{len(names) - len(shown)} more)"
        api_lines.append(f"- {_wrapper_source_display_name(helper)}: {', '.join(shown)}{suffix}")
    if api_lines:
        sections.append("API calls used by wrapper helper scripts:\n" + "\n".join(api_lines))

    command_doc_lines: list[str] = []
    for helper in helper_files:
        for doc in (helper.get("command_docs", []) or []):
            if not isinstance(doc, dict):
                continue
            text = _single_line_text(str(doc.get("text", "") or ""), limit=700)
            if text:
                command_doc_lines.append(f"- {_wrapper_source_display_name(helper)}: {text}")
            if len(command_doc_lines) >= _CONFIGFILE_HELP_CONTEXT_COMMAND_DOC_LIMIT:
                break
        if len(command_doc_lines) >= _CONFIGFILE_HELP_CONTEXT_COMMAND_DOC_LIMIT:
            break
    if command_doc_lines:
        sections.append(
            "Command-line documentation embedded in wrapper helper scripts:\n"
            + "\n".join(command_doc_lines)
        )

    parameter_lines: list[str] = []
    for helper in helper_files:
        docs = [
            _format_configfile_parameter_doc(doc)
            for doc in (helper.get("parameter_docs", []) or [])
            if isinstance(doc, dict)
        ]
        docs = [doc for doc in docs if doc]
        if not docs:
            continue
        shown = docs[:_CONFIGFILE_HELP_CONTEXT_PARAMETER_PER_FILE_LIMIT]
        suffix = ""
        if len(docs) > len(shown):
            suffix = f" (+{len(docs) - len(shown)} more)"
        parameter_lines.append(
            f"- {_wrapper_source_display_name(helper)}: " + "; ".join(shown) + suffix
        )
        if len(parameter_lines) >= _CONFIGFILE_HELP_CONTEXT_PARAMETER_LIMIT:
            break
    if parameter_lines:
        sections.append(
            "Parameter documentation embedded in wrapper helper scripts:\n"
            + "\n".join(parameter_lines)
        )

    if not sections:
        return ""
    return "Wrapper helper context:\n\n" + "\n\n".join(sections)


def _source_command_help_context(bioconda_sources: list[dict]) -> str:
    lines: list[str] = []
    for source in bioconda_sources:
        if not isinstance(source, dict):
            continue
        package = str(source.get("package", "") or "").strip()
        for doc in source.get("source_command_docs", []) or []:
            if not isinstance(doc, dict):
                continue
            text = _single_line_text(str(doc.get("text", "") or ""), limit=900)
            if not text:
                continue
            path = str(doc.get("path", "") or "").strip()
            label = package
            if path:
                label = f"{label} {path}" if label else path
            lines.append(f"- {label}: {text}")
            if len(lines) >= _CONFIGFILE_HELP_CONTEXT_COMMAND_DOC_LIMIT:
                break
        if len(lines) >= _CONFIGFILE_HELP_CONTEXT_COMMAND_DOC_LIMIT:
            break
    if not lines:
        return ""
    return "Underlying software source documentation:\n" + "\n".join(lines)


def _bool_attr(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in {"true", "yes", "1"}:
        return True
    if cleaned in {"false", "no", "0"}:
        return False
    return default


def _parse_int_attr(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _parse_float_attr(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def _udt_identifier(value: str, fallback: str) -> str:
    cleaned = _safe_slug(value or fallback).lower().replace(".", "_")
    if len(cleaned) < 3:
        cleaned = f"tool_{cleaned or 'generated'}"
    return cleaned[:255]


def _udt_name(value: str | None, fallback: str) -> str:
    cleaned = _strip_text(value)
    if cleaned:
        return cleaned
    return _udt_identifier(fallback, "parameter")


def _udt_extensions(value: str | None) -> list[str]:
    parts = [part.strip() for part in (value or "").replace(";", ",").split(",")]
    extensions = [part for part in parts if part]
    return extensions or ["data"]


def _dedupe_named_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: dict[str, int] = {}
    deduped: list[dict[str, object]] = []
    for item in items:
        raw_name = str(item.get("name", "") or "parameter")
        name = _udt_identifier(raw_name, "parameter")
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"
        item = dict(item)
        item["name"] = name
        deduped.append(item)
    return deduped


def _udt_ref_source_name(element: ET.Element) -> str:
    return _strip_text(element.attrib.get("name") or element.attrib.get("argument"))


def _synthesize_udt_input(param: ET.Element) -> dict[str, object]:
    param_type = _strip_text(param.attrib.get("type")) or "text"
    name = _udt_name(param.attrib.get("name") or param.attrib.get("argument"), "parameter")
    supported_type = {
        "boolean",
        "color",
        "data",
        "data_collection",
        "float",
        "integer",
        "select",
        "text",
    }
    item: dict[str, object] = {
        "name": name,
        "type": param_type if param_type in supported_type else "text",
    }
    label = _strip_text(param.attrib.get("label"))
    help_text = _strip_text(param.attrib.get("help"))
    argument = _strip_text(param.attrib.get("argument"))
    if label:
        item["label"] = label
    if help_text:
        item["help"] = help_text
    if argument:
        item["argument"] = argument
    item["optional"] = _bool_attr(param.attrib.get("optional"), default=False)

    input_type = str(item["type"])
    if input_type == "data":
        item["extensions"] = _udt_extensions(
            param.attrib.get("format") or param.attrib.get("ext")
        )
        item["multiple"] = _bool_attr(param.attrib.get("multiple"), default=False)
    elif input_type == "data_collection":
        item["extensions"] = _udt_extensions(
            param.attrib.get("format") or param.attrib.get("ext")
        )
        item["collection_type"] = _strip_text(param.attrib.get("collection_type")) or "list"
        item["value"] = {}
    elif input_type == "boolean":
        item["value"] = _bool_attr(
            param.attrib.get("checked") or param.attrib.get("value"), default=False
        )
        true_value = _strip_text(param.attrib.get("truevalue"))
        false_value = _strip_text(param.attrib.get("falsevalue"))
        if true_value:
            item["truevalue"] = true_value
        if false_value:
            item["falsevalue"] = false_value
    elif input_type == "integer":
        for key in ("value", "min", "max"):
            parsed = _parse_int_attr(param.attrib.get(key))
            if parsed is not None:
                item[key] = parsed
    elif input_type == "float":
        for key in ("value", "min", "max"):
            parsed = _parse_float_attr(param.attrib.get(key))
            if parsed is not None:
                item[key] = parsed
    elif input_type == "select":
        options = []
        for option in param.findall(".//option"):
            value = _strip_text(option.attrib.get("value")) or _strip_text(option.text)
            option_label = _strip_text(option.text) or value
            if value:
                options.append({"label": option_label, "value": value})
        if options:
            item["options"] = options
        item["multiple"] = _bool_attr(param.attrib.get("multiple"), default=False)
    elif input_type == "color":
        value = _strip_text(param.attrib.get("value"))
        if value:
            item["value"] = value
    elif input_type == "text":
        value = _strip_text(param.attrib.get("value"))
        if value:
            item["value"] = value
        item["area"] = _bool_attr(param.attrib.get("area"), default=False)
    return item


def _udt_ref_expression(kind: str, name: str, value_type: str) -> str:
    path_suffix = ".path" if kind == "outputs" or value_type in {"data", "data_collection"} else ""
    return f"$({kind}.{name}{path_suffix})"


def _synthesize_udt_inputs_and_refs(
    root: ET.Element,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    params = root.findall(".//inputs//param")
    source_names = [_udt_ref_source_name(param) for param in params]
    inputs = _dedupe_named_items([_synthesize_udt_input(param) for param in params])
    refs: dict[str, str] = {}
    for source_name, item in zip(source_names, inputs, strict=False):
        name = str(item.get("name", "") or "")
        if not name:
            continue
        expression = _udt_ref_expression("inputs", name, str(item.get("type", "") or ""))
        refs[name] = expression
        if source_name:
            refs[source_name] = expression
    return inputs, refs


def _synthesize_udt_inputs(root: ET.Element) -> list[dict[str, object]]:
    inputs, _refs = _synthesize_udt_inputs_and_refs(root)
    return inputs


def _synthesize_udt_outputs_and_refs(
    root: ET.Element,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    outputs: list[dict[str, object]] = []
    source_names: list[str] = []
    for data in root.findall(".//outputs//data"):
        item: dict[str, object] = {
            "name": _udt_name(data.attrib.get("name"), "output"),
            "type": "data",
        }
        label = _strip_text(data.attrib.get("label"))
        if label:
            item["label"] = label
        fmt = _strip_text(data.attrib.get("format") or data.attrib.get("ext"))
        if fmt:
            item["format"] = fmt
        from_work_dir = _strip_text(data.attrib.get("from_work_dir"))
        if from_work_dir:
            item["from_work_dir"] = from_work_dir
        item["hidden"] = _bool_attr(data.attrib.get("hidden"), default=False)
        outputs.append(item)
        source_names.append(_udt_ref_source_name(data))
    for collection in root.findall(".//outputs//collection"):
        collection_type = _strip_text(collection.attrib.get("type")) or _strip_text(
            collection.attrib.get("collection_type")
        )
        outputs.append(
            {
                "name": _udt_name(collection.attrib.get("name"), "collection"),
                "type": "collection",
                "hidden": _bool_attr(collection.attrib.get("hidden"), default=False),
                "structure": {"collection_type": collection_type or "list"},
            }
        )
        source_names.append(_udt_ref_source_name(collection))
    outputs = _dedupe_named_items(outputs)
    refs: dict[str, str] = {}
    for source_name, item in zip(source_names, outputs, strict=False):
        name = str(item.get("name", "") or "")
        if not name:
            continue
        expression = _udt_ref_expression("outputs", name, str(item.get("type", "") or ""))
        refs[name] = expression
        if source_name:
            refs[source_name] = expression
    return outputs, refs


def _synthesize_udt_outputs(root: ET.Element) -> list[dict[str, object]]:
    outputs, _refs = _synthesize_udt_outputs_and_refs(root)
    return outputs


def _xml_command_to_udt(
    command_text: str,
    *,
    input_refs: dict[str, str],
    output_refs: dict[str, str],
) -> str:
    refs = dict(input_refs)
    refs.update(output_refs)

    def replace_braced(match: re.Match[str]) -> str:
        return refs.get(match.group(1), match.group(0))

    def replace_simple(match: re.Match[str]) -> str:
        return refs.get(match.group(1), match.group(0))

    converted = _XML_BRACED_REF_RE.sub(replace_braced, command_text)
    converted = _XML_SIMPLE_REF_RE.sub(replace_simple, converted)
    return converted.strip() or "true"


def _synthesize_udt_configfiles(
    root: ET.Element,
    *,
    max_bytes: int,
) -> list[dict[str, object]]:
    configfiles: list[dict[str, object]] = []
    for index, configfile in enumerate(root.findall(".//configfiles//configfile"), start=1):
        content = _configfile_content(configfile)
        if not content.strip():
            continue
        if len(content.encode("utf-8", errors="replace")) > max(1, max_bytes):
            continue
        name = _strip_text(configfile.attrib.get("name"))
        filename = _strip_text(configfile.attrib.get("filename"))
        item: dict[str, object] = {"content": content}
        if name:
            item["name"] = name
        elif not filename:
            item["name"] = f"config_{index}"
        if filename:
            item["filename"] = filename
        eval_engine = _strip_text(configfile.attrib.get("eval_engine"))
        if eval_engine == "ecmascript":
            item["eval_engine"] = eval_engine
        configfiles.append(item)
    return configfiles


def _synthesize_udt_yaml(
    *,
    root: ET.Element,
    tool_id: str,
    tool_name: str,
    command_text: str,
    help_text: str,
    selected_container: str,
    container_refs: list[str],
    output_path: Path,
    wrapper_configfile_max_bytes: int,
) -> Path:
    container = selected_container or (container_refs[0] if container_refs else "")
    version = _strip_text(root.attrib.get("version")) or "0.1.0"
    if "@" in version or "$" in version:
        version = "0.1.0"
    description = _strip_text(root.attrib.get("description"))
    inputs, input_refs = _synthesize_udt_inputs_and_refs(root)
    outputs, output_refs = _synthesize_udt_outputs_and_refs(root)
    payload: dict[str, object] = {
        "class": "GalaxyUserTool",
        "id": _udt_identifier(tool_id, output_path.stem),
        "version": version,
        "name": tool_name or _udt_identifier(tool_id, output_path.stem),
        "container": container,
        "shell_command": _xml_command_to_udt(
            command_text,
            input_refs=input_refs,
            output_refs=output_refs,
        ),
        "inputs": inputs,
        "outputs": outputs,
    }
    if description:
        payload["description"] = description
    configfiles = _synthesize_udt_configfiles(
        root,
        max_bytes=wrapper_configfile_max_bytes,
    )
    if configfiles:
        payload["configfiles"] = configfiles
    if help_text.strip():
        payload["help"] = {"format": "markdown", "content": help_text.strip()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return output_path


def _merge_requirement_sets(
    *requirement_sets: tuple[list[str], dict[str, str], list[str]],
) -> tuple[list[str], dict[str, str], list[str]]:
    packages: set[str] = set()
    versions: dict[str, str] = {}
    containers: set[str] = set()
    for package_list, version_map, container_list in requirement_sets:
        packages.update(package_list)
        containers.update(container_list)
        for package, version in version_map.items():
            if package and version and package not in versions:
                versions[package] = version
    return sorted(packages), versions, sorted(containers)


def _build_datatype_report(input_datatypes: list[str], output_datatypes: list[str]) -> dict:
    all_types = sorted(set(input_datatypes) | set(output_datatypes))
    known = sorted(dtype for dtype in all_types if dtype in KNOWN_GALAXY_DATATYPES)
    unknown = sorted(dtype for dtype in all_types if dtype not in KNOWN_GALAXY_DATATYPES)
    confidence = "high" if not unknown else "low"
    suggestions = [
        {
            "datatype": dtype,
            "action": "create-datatype-scaffold",
            "reason": "Datatype not found in known list; requires Galaxy datatype registration review.",
        }
        for dtype in unknown
    ]
    return {
        "known_datatypes": known,
        "unknown_datatypes": unknown,
        "confidence": confidence,
        "suggestions": suggestions,
    }


def _load_shed_metadata(tool_dir: Path) -> dict:
    shed_path = tool_dir / ".shed.yml"
    if not shed_path.exists():
        return {}
    try:
        return yaml.safe_load(shed_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _shed_suite_fields(shed: dict) -> tuple[str, str, list[str], bool]:
    name = str(shed.get("name", "") or "").strip()
    owner = str(shed.get("owner", "") or "").strip()
    suite_type = str(shed.get("type", "") or "").strip().lower()
    is_suite_root = suite_type == "suite_repository" or name.startswith("suite_")
    suite_id = f"{owner}/{name}" if is_suite_root and owner and name else ""
    suite_name = name if is_suite_root else ""

    members: list[str] = []
    repositories = shed.get("repositories")
    if isinstance(repositories, list):
        for item in repositories:
            if isinstance(item, dict):
                member_name = str(item.get("name", "") or "").strip()
                if member_name:
                    members.append(member_name)
            elif isinstance(item, str):
                value = item.strip()
                if value:
                    members.append(value)
    return suite_id, suite_name, sorted(set(members)), is_suite_root


def _normalize_github_repo_url(homepage_url: str) -> str | None:
    if not homepage_url:
        return None
    url = homepage_url.strip().rstrip("/")
    if not url.startswith("https://github.com/"):
        return None
    parts = url[len("https://github.com/") :].split("/")
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    return f"https://github.com/{owner}/{repo}"


def _requests_get_with_user_agent_fallback(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", None)
    attempts = user_agent_header_attempts(headers)
    last_error: requests.RequestException | None = None
    for index, attempt_headers in enumerate(attempts):
        try:
            response = requests.get(url, headers=attempt_headers, **kwargs)
        except requests.RequestException as error:
            last_error = error
            if index + 1 < len(attempts):
                continue
            raise
        response.gtsm_user_agent_attempt_index = index
        response.gtsm_user_agent = attempt_headers.get("User-Agent", "")
        response.gtsm_user_agent_fallback = index > 0
        if index + 1 < len(attempts) and should_retry_with_browser_user_agent(
            response.status_code
        ):
            response.close()
            continue
        return response
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No HTTP request attempts were generated for {url}")


def _fetch_github_readme(homepage_url: str) -> str:
    repo_url = _normalize_github_repo_url(homepage_url)
    if not repo_url:
        return ""

    suffix = repo_url.removeprefix("https://github.com/")
    owner_repo = suffix.split("/")
    if len(owner_repo) != 2:
        return ""
    owner, repo = owner_repo
    names = ("README.md", "README.rst", "README.txt", "README")
    branches = ("main", "master")

    for branch in branches:
        for name in names:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}"
            try:
                response = _requests_get_with_user_agent_fallback(
                    raw_url,
                    timeout=10,
                )
                if response.status_code == 200 and response.text.strip():
                    return response.text
            except requests.RequestException:
                continue
    return ""


def _lookup_biocontainer_images(package_name: str, limit: int = 5) -> list[str]:
    key = package_name.strip().lower()
    if not key:
        return []
    if key in _BIOCONTAINER_LOOKUP_CACHE:
        return _BIOCONTAINER_LOOKUP_CACHE[key]

    url = f"https://api.biocontainers.pro/ga4gh/trs/v2/tools/{key}/versions"
    try:
        response = _requests_get_with_user_agent_fallback(url, timeout=12)
        if response.status_code != 200:
            _BIOCONTAINER_LOOKUP_CACHE[key] = []
            return []
        payload = response.json()
    except (requests.RequestException, ValueError):
        _BIOCONTAINER_LOOKUP_CACHE[key] = []
        return []

    images: list[tuple[int, str, str]] = []
    if isinstance(payload, list):
        for version in payload:
            for image in version.get("images", []):
                image_name = image.get("image_name")
                if not image_name:
                    continue
                downloads = int(image.get("downloads", 0) or 0)
                updated = str(image.get("updated", "") or "")
                images.append((downloads, updated, image_name))

    images.sort(key=lambda item: (item[0], item[1]), reverse=True)
    deduped: list[str] = []
    seen: set[str] = set()
    for _, _, image_name in images:
        if image_name in seen:
            continue
        seen.add(image_name)
        deduped.append(image_name)
        if len(deduped) >= limit:
            break

    _BIOCONTAINER_LOOKUP_CACHE[key] = deduped
    return deduped


def _loose_sort_key(value: str) -> tuple:
    parts: list[object] = []
    for token in re.split(r"([0-9]+)", value):
        if not token:
            continue
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token.lower())
    return tuple(parts)


def _lookup_quay_tags(
    namespace: str, repository: str, tag_prefix: str = "", limit: int = 5
) -> list[str]:
    namespace = namespace.strip()
    repository = repository.strip()
    tag_prefix = tag_prefix.strip()
    if not namespace or not repository:
        return []
    cache_key = (namespace, repository, tag_prefix)
    if cache_key in _QUAY_TAGS_CACHE:
        return _QUAY_TAGS_CACHE[cache_key]

    url = f"https://quay.io/api/v1/repository/{namespace}/{repository}"
    try:
        response = _requests_get_with_user_agent_fallback(url, timeout=12)
        if response.status_code != 200:
            _QUAY_TAGS_CACHE[cache_key] = []
            return []
        payload = response.json()
    except (requests.RequestException, ValueError):
        _QUAY_TAGS_CACHE[cache_key] = []
        return []

    tags_payload = payload.get("tags", {})
    if not isinstance(tags_payload, dict):
        _QUAY_TAGS_CACHE[cache_key] = []
        return []
    tags = [
        str(tag)
        for tag in tags_payload
        if tag != "latest" and (not tag_prefix or str(tag).startswith(tag_prefix))
    ]
    tags = sorted(tags, key=_loose_sort_key, reverse=True)[:limit]
    _QUAY_TAGS_CACHE[cache_key] = tags
    return tags


def _build_mulled_targets(
    package_names: list[str], requirement_versions: dict[str, str]
) -> list[MulledTarget]:
    targets: list[MulledTarget] = []
    versions = {
        package.strip().lower(): version for package, version in requirement_versions.items()
    }
    for package in sorted({name.strip().lower() for name in package_names if name.strip()}):
        targets.append(MulledTarget(package=package, version=versions.get(package, "").strip()))
    return targets


def _conda_build_target_str(target: MulledTarget) -> str:
    value = target.package
    if target.version:
        value += f"={target.version}"
        if target.build:
            value += f"={target.build}"
    return value


def _simple_mulled_image_name(target: MulledTarget, image_build: str = "") -> str:
    suffix = ""
    if target.version:
        build = target.build
        if not build and image_build and image_build != "0":
            build = image_build
        suffix += f":{target.version}"
        if build:
            suffix += f"--{build}"
    return f"{target.package}{suffix}"


def _mulled_v2_image_name(targets: list[MulledTarget], image_build: str = "") -> str:
    if not targets:
        return ""
    if len(targets) == 1:
        return _simple_mulled_image_name(targets[0], image_build=image_build)

    targets_order = sorted(targets, key=lambda target: target.package)
    package_name_buffer = "\n".join(target.package for target in targets_order)
    package_hash = hashlib.sha1(package_name_buffer.encode()).hexdigest()

    version_hash_str = ""
    if any(target.version for target in targets_order):
        version_name_buffer = "\n".join(target.version or "null" for target in targets_order)
        version_hash_str = hashlib.sha1(version_name_buffer.encode()).hexdigest()

    if not image_build:
        build_suffix = ""
    elif version_hash_str:
        build_suffix = f"-{image_build}"
    else:
        build_suffix = image_build

    suffix = ""
    if version_hash_str or build_suffix:
        suffix = f":{version_hash_str}{build_suffix}"
    return f"mulled-v2-{package_hash}{suffix}"


def _mulled_biocontainer_images(
    package_names: list[str],
    requirement_versions: dict[str, str],
    limit: int = 5,
) -> list[str]:
    return [
        candidate.image
        for candidate in _mulled_biocontainer_candidates(package_names, requirement_versions, limit)
    ]


def _mulled_biocontainer_candidates(
    package_names: list[str],
    requirement_versions: dict[str, str],
    limit: int = 5,
) -> list[ContainerCandidate]:
    targets = _build_mulled_targets(package_names, requirement_versions)
    if not targets:
        return []

    base_name = _mulled_v2_image_name(targets)
    if not base_name:
        return []
    if ":" in base_name:
        repository, tag_prefix = base_name.split(":", 1)
    else:
        repository, tag_prefix = base_name, ""

    tags = _lookup_quay_tags("biocontainers", repository, tag_prefix=tag_prefix, limit=limit)
    source = "mulled-single" if len(targets) == 1 else "mulled-v2"
    priority = 260 if source == "mulled-single" else 250
    packages = tuple(target.package for target in targets)
    return [
        ContainerCandidate(
            image=f"quay.io/biocontainers/{repository}:{tag}",
            source=source,
            packages=packages,
            priority=priority,
        )
        for tag in tags
    ][:limit]


def _version_from_image_ref(image_ref: str) -> str:
    ref = image_ref.strip()
    if not ref:
        return ""
    if "@" in ref:
        return ref.split("@", 1)[1]
    if ":" in ref:
        return ref.rsplit(":", 1)[1]
    return ""


def _image_matches_requirement_versions(
    image_ref: str, requirement_versions: dict[str, str]
) -> bool:
    if not requirement_versions:
        return True
    image_version = _version_from_image_ref(image_ref).lower()
    if not image_version:
        return False
    for version in requirement_versions.values():
        value = str(version).strip().lower()
        if not value:
            continue
        if value in image_version:
            return True
    return False


def _choose_container_image(candidates: list[str], requirement_versions: dict[str, str]) -> str:
    normalized = _normalize_container_candidates(candidates)
    if not normalized:
        return ""
    for image in normalized:
        if _image_matches_requirement_versions(image, requirement_versions):
            return image
    return normalized[0]


def _candidate_packages_for_image(image: str, fallback_packages: list[str]) -> tuple[str, ...]:
    base_name = _container_image_base_name(image)
    if base_name.startswith("mulled-v2-"):
        return tuple(
            sorted({package.strip().lower() for package in fallback_packages if package.strip()})
        )
    return (base_name.strip().lower(),) if base_name.strip() else ()


def _container_candidate_payload(candidate: ContainerCandidate) -> dict:
    return {
        "image": candidate.image,
        "source": candidate.source,
        "packages": list(candidate.packages),
        "priority": candidate.priority,
        "status": candidate.status,
        "error_text": candidate.error_text,
    }


def _dedupe_container_candidates(candidates: list[ContainerCandidate]) -> list[ContainerCandidate]:
    by_image: dict[str, ContainerCandidate] = {}
    for candidate in candidates:
        key = candidate.image
        existing = by_image.get(key)
        if existing is None or candidate.priority > existing.priority:
            by_image[key] = candidate
    return sorted(by_image.values(), key=lambda candidate: (-candidate.priority, candidate.image))


def _build_container_candidate_details(
    *,
    container_refs: list[str],
    package_names: list[str],
    requirement_versions: dict[str, str],
    settings: ExtractionSettings,
) -> list[dict]:
    candidates: list[ContainerCandidate] = []
    for raw_ref in container_refs:
        normalized = _normalize_container_candidate(raw_ref)
        if not normalized:
            candidates.append(
                ContainerCandidate(
                    image=raw_ref,
                    source="explicit",
                    priority=300,
                    status="skipped",
                    error_text=_container_ref_error(_normalize_container_ref(raw_ref)),
                )
            )
            continue
        candidates.append(
            ContainerCandidate(
                image=normalized,
                source="explicit",
                packages=_candidate_packages_for_image(normalized, package_names),
                priority=300,
            )
        )

    if settings.resolve_containers or settings.execute_containers:
        candidates.extend(_mulled_biocontainer_candidates(package_names, requirement_versions))
    if settings.resolve_containers:
        for package in package_names:
            for image in _lookup_biocontainer_images(package):
                normalized = _normalize_container_candidate(image)
                if normalized:
                    candidates.append(
                        ContainerCandidate(
                            image=normalized,
                            source="biocontainers-api",
                            packages=(package.strip().lower(),),
                            priority=140,
                        )
                    )

    payloads = [
        _container_candidate_payload(candidate)
        for candidate in _dedupe_container_candidates(candidates)
    ]
    return payloads


def _candidate_sort_key(candidate: dict) -> tuple[int, str]:
    return (-int(candidate.get("priority", 0) or 0), str(candidate.get("image", "")))


def _candidate_package_keys(candidate: dict) -> set[str]:
    keys = {
        key
        for key in (
            _normalized_command_key(str(package))
            for package in (candidate.get("packages", []) or [])
        )
        if key
    }
    image_key = _normalized_command_key(
        _container_image_base_name(str(candidate.get("image", "") or ""))
    )
    if not _is_generic_container_image_key(image_key):
        keys.add(image_key)
    return keys


def _requirement_package_keys(package_names: list[str] | tuple[str, ...] | None) -> set[str]:
    return {
        key
        for key in (_normalized_command_key(str(package)) for package in (package_names or []))
        if key
    }


def _candidate_source_rank(candidate: dict, *, full_coverage: bool) -> int:
    source = str(candidate.get("source", "") or "").lower()
    if source == "explicit" and full_coverage:
        return 70
    if source == "mulled-v2":
        return 65
    if source == "mulled-single":
        return 55
    if source == "explicit":
        return 45
    if source == "biocontainers-api":
        return 35
    if source == "legacy":
        return 25
    return 0


def _container_candidate_selection_key(
    candidate: dict,
    *,
    requirement_packages: list[str] | tuple[str, ...] | None = None,
    requirement_versions: dict[str, str] | None = None,
) -> tuple[int, int, int, int, int, str]:
    requirement_keys = _requirement_package_keys(requirement_packages)
    package_keys = _candidate_package_keys(candidate)
    coverage_count = len(requirement_keys & package_keys)
    full_coverage = bool(requirement_keys) and requirement_keys <= package_keys
    version_match = _image_matches_requirement_versions(
        str(candidate.get("image", "") or ""), requirement_versions or {}
    )
    full_score = 1 if full_coverage else 0
    source_rank = _candidate_source_rank(candidate, full_coverage=full_coverage)
    priority = int(candidate.get("priority", 0) or 0)
    return (
        -full_score,
        -coverage_count,
        -source_rank,
        -int(version_match),
        -priority,
        str(candidate.get("image", "")),
    )


def _sort_container_candidates_for_selection(
    candidates: list[dict],
    *,
    requirement_packages: list[str] | tuple[str, ...] | None = None,
    requirement_versions: dict[str, str] | None = None,
) -> list[dict]:
    return sorted(
        candidates,
        key=lambda candidate: _container_candidate_selection_key(
            candidate,
            requirement_packages=requirement_packages,
            requirement_versions=requirement_versions,
        ),
    )


def _choose_container_candidate(
    candidates: list[dict],
    requirement_versions: dict[str, str],
    requirement_packages: list[str] | tuple[str, ...] | None = None,
) -> dict:
    usable = _sort_container_candidates_for_selection(
        (
            candidate
            for candidate in candidates
            if candidate.get("status", "ok") == "ok" and candidate.get("image")
        ),
        requirement_packages=requirement_packages,
        requirement_versions=requirement_versions,
    )
    if not usable:
        return {}
    return usable[0]


def _legacy_container_candidate_details(record: ToolRecord) -> list[dict]:
    if record.container_candidate_details:
        return record.container_candidate_details
    refs = []
    if record.selected_container:
        refs.append(record.selected_container)
    refs.extend(record.container_candidates)
    candidates = [
        _container_candidate_payload(
            ContainerCandidate(
                image=ref,
                source="legacy",
                packages=_candidate_packages_for_image(ref, record.requirement_packages),
                priority=100,
            )
        )
        for ref in _normalize_container_candidates(refs)
    ]
    return candidates


def _docker_cmd(*parts: str, use_sudo: bool) -> list[str]:
    cmd: list[str] = []
    if use_sudo:
        cmd.append("sudo")
    cmd.append("docker")
    cmd.extend(parts)
    return cmd


def _tail_text(value: str, limit: int = 2000) -> str:
    value = value or ""
    return value[-limit:]


def _strip_container_runtime_noise(text: str) -> str:
    noisy_prefixes = (
        "WARNING: Skipping mount /etc/resolv.conf",
        "INFO:    Detected Singularity user configuration directory",
        "INFO:    squashfuse not found",
        "INFO:    gocryptfs not found",
        "INFO:    Converting SIF file to temporary sandbox",
        "INFO:    Cleaning up image",
    )
    lines = []
    for line in (text or "").splitlines():
        if any(line.startswith(prefix) for prefix in noisy_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _completed_error_text(result: subprocess.CompletedProcess) -> str:
    text = result.stderr or result.stdout or ""
    return _tail_text(_strip_container_runtime_noise(text))


def _is_url_ref(image_ref: str) -> bool:
    parsed = urlparse(image_ref.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _url_cache_image_name(image_ref: str) -> str:
    parsed = urlparse(image_ref.strip())
    name = Path(parsed.path).name
    if name:
        return name
    return f"{_safe_slug(image_ref)}.sif"


def _is_depot_url(image_ref: str, depot_url: str) -> bool:
    ref = image_ref.strip().rstrip("/")
    depot = depot_url.strip().rstrip("/")
    return bool(depot) and ref.startswith(f"{depot}/")


def _normalize_biocontainers_equals_ref(image_ref: str) -> str:
    ref = image_ref.strip()
    if "==" not in ref or _is_url_ref(ref) or "@" in ref:
        return ref
    prefix = ""
    name = ref
    if "/" in ref:
        prefix, name = ref.rsplit("/", 1)
    if "==" in name and ":" not in name:
        name = name.replace("==", ":", 1)
        return f"{prefix}/{name}" if prefix else name
    return ref


def _normalize_container_ref(image_ref: str) -> str:
    ref = image_ref.strip()
    if ref.startswith("docker://"):
        ref = ref[len("docker://") :]
    return _normalize_biocontainers_equals_ref(ref)


def _container_ref_error(image_ref: str) -> str:
    ref = image_ref.strip()
    if not ref:
        return "empty container reference"
    if (
        "__tool_directory__" in ref
        or re.search(r"@[A-Za-z_][A-Za-z0-9_]*@", ref)
        or re.search(r"\{[^}]+\}", ref)
    ):
        return "container reference contains unresolved placeholder"
    if _is_url_ref(ref):
        return ""
    if "://" in ref:
        return "unsupported container URL scheme"
    if re.search(r"\s", ref):
        return "container reference contains whitespace"
    if re.search(r"""["'<>|;&$`\\]""", ref):
        return "container reference contains shell metacharacters"
    if ref.startswith("-"):
        return "container reference starts with an option marker"
    if "==" in ref:
        return "container reference uses conda-style == instead of image tag syntax"
    return ""


def _normalize_container_candidate(image_ref: str) -> str:
    ref = _normalize_container_ref(image_ref)
    if _container_ref_error(ref):
        return ""
    return ref


def _normalize_container_candidates(candidates: list[str] | set[str]) -> list[str]:
    normalized: set[str] = set()
    for candidate in candidates:
        ref = _normalize_container_candidate(candidate)
        if ref:
            normalized.add(ref)
    return sorted(normalized)


def _docker_ref_for_image(image_ref: str) -> str:
    ref = _normalize_container_candidate(image_ref)
    if not ref:
        return ""
    if _is_url_ref(ref):
        return ""
    if "/" not in ref:
        return f"quay.io/biocontainers/{ref}"
    return ref


def _singularity_source_ref(image_ref: str) -> str:
    ref = _normalize_container_candidate(image_ref)
    if not ref:
        return ""
    if ref.startswith("docker://") or ref.startswith("http://") or ref.startswith("https://"):
        return ref
    docker_ref = _docker_ref_for_image(ref)
    return f"docker://{docker_ref}" if docker_ref else ""


def _depot_image_name(image_ref: str) -> str:
    ref = _docker_ref_for_image(image_ref)
    prefix = "quay.io/biocontainers/"
    if not ref.startswith(prefix):
        return ""
    return ref[len(prefix) :]


def _singularity_depot_image_url(image_ref: str, depot_url: str) -> str:
    if _is_url_ref(image_ref):
        return ""
    image_name = _depot_image_name(image_ref)
    if not image_name:
        return ""
    return f"{depot_url.rstrip('/')}/{quote(image_name, safe=':@._-+')}"


def _container_cache_root(settings: ExtractionSettings) -> Path:
    if settings.container_cache_dir is not None:
        return settings.container_cache_dir
    cache_root = settings.cache_root or Path(".gtsm-cache")
    return cache_root / "containers"


def _container_image_quarantine_path(settings: ExtractionSettings) -> Path | None:
    if settings.container_image_quarantine_file is not None:
        return settings.container_image_quarantine_file
    if settings.container_cache_dir is None:
        return None
    return _container_cache_root(settings) / "image-quarantine.json"


def _load_container_image_quarantine_store(
    settings: ExtractionSettings,
) -> ContainerImageQuarantineStore:
    path = _container_image_quarantine_path(settings)
    entries: dict[str, dict[str, object]] = {}
    if path is not None and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        raw_entries = payload.get("entries", payload) if isinstance(payload, dict) else {}
        if isinstance(raw_entries, dict):
            entries = {
                str(key): value
                for key, value in raw_entries.items()
                if isinstance(value, dict)
            }
    return ContainerImageQuarantineStore(path=path, entries=entries)


def _save_container_image_quarantine_store(store: ContainerImageQuarantineStore) -> None:
    if store.path is None:
        return
    store.path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = store.path.with_name(f"{store.path.name}.tmp")
    payload = {
        "schema_version": "0.1.0",
        "entries": store.entries,
    }
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(store.path)


def _container_cache_satisfies_preparation(
    image: str,
    *,
    runtimes: list[ContainerRuntime],
    settings: ExtractionSettings,
) -> bool:
    ref = _normalize_container_ref(image)
    if not ref:
        return False
    if Path(ref).exists():
        return True
    for runtime in runtimes:
        if runtime.name in {"singularity", "apptainer"} and _singularity_cache_path(
            ref, settings=settings
        ).exists():
            return True
    return False


def _container_quarantine_failure(entry: dict[str, object], image: str) -> ContainerPreparation:
    return ContainerPreparation(
        ok=False,
        runtime=str(entry.get("runtime", "") or ""),
        image=str(entry.get("image", "") or image),
        identifier=str(entry.get("identifier", "") or ""),
        source=str(entry.get("source", "") or "quarantine"),
        returncode=int(entry.get("returncode", 0) or 0),
        stdout=str(entry.get("stdout", "") or ""),
        stderr=str(entry.get("stderr", "") or ""),
        error_text=str(entry.get("error_text", "") or "container image is quarantined"),
    )


def _container_image_quarantine_get(
    store: ContainerImageQuarantineStore | None,
    key: str,
    settings: ExtractionSettings,
) -> dict[str, object] | None:
    if store is None or not key:
        return None
    now = time.time()
    with store.lock:
        entry = store.entries.get(key)
        if not isinstance(entry, dict):
            return None
        expires_at = float(entry.get("expires_at", 0) or 0)
        if expires_at and now >= expires_at:
            store.entries.pop(key, None)
            _save_container_image_quarantine_store(store)
            return None
        quarantine_seconds = max(0, int(settings.container_image_quarantine_seconds or 0))
        if not expires_at and quarantine_seconds:
            created_at = float(entry.get("created_at", 0) or 0)
            if created_at and now - created_at >= quarantine_seconds:
                store.entries.pop(key, None)
                _save_container_image_quarantine_store(store)
                return None
        return dict(entry)


def _container_image_quarantine_put(
    store: ContainerImageQuarantineStore | None,
    key: str,
    preparation: ContainerPreparation,
    settings: ExtractionSettings,
    *,
    reason: str,
) -> None:
    if store is None or not key:
        return
    quarantine_seconds = max(0, int(settings.container_image_quarantine_seconds or 0))
    now = time.time()
    with store.lock:
        store.entries[key] = {
            "created_at": now,
            "expires_at": now + quarantine_seconds if quarantine_seconds else 0,
            "quarantine_seconds": quarantine_seconds,
            "reason": reason,
            "runtime": preparation.runtime,
            "image": preparation.image,
            "identifier": preparation.identifier,
            "source": preparation.source,
            "returncode": preparation.returncode,
            "stdout": preparation.stdout,
            "stderr": preparation.stderr,
            "error_text": preparation.error_text,
        }
        _save_container_image_quarantine_store(store)


def _container_image_timeout(settings: ExtractionSettings) -> int:
    return max(
        1,
        int(settings.container_image_timeout_seconds or settings.container_pull_timeout_seconds),
    )


def _container_preparation_timed_out(preparation: ContainerPreparation) -> bool:
    if preparation.returncode == 124:
        return True
    text = "\n".join(
        part
        for part in (preparation.error_text, preparation.stderr, preparation.stdout)
        if part
    ).lower()
    return "timed out" in text or "timeout" in text


def _container_preparation_cache_key(
    image_ref: str,
    *,
    runtimes: list[ContainerRuntime],
    settings: ExtractionSettings,
) -> str:
    ref = _normalize_container_ref(image_ref)
    if not ref:
        return ""
    for runtime in runtimes:
        if runtime.name in {"singularity", "apptainer"}:
            return f"{runtime.name}:{_singularity_cache_path(ref, settings=settings)}"
        if runtime.name == "docker":
            return f"docker:{_docker_ref_for_image(ref)}"
    return ref


def _singularity_cache_path(image_ref: str, settings: ExtractionSettings) -> Path:
    ref = _normalize_container_ref(image_ref)
    image_name = _url_cache_image_name(ref) if _is_url_ref(ref) else _depot_image_name(ref)
    if not image_name:
        image_name = f"{_safe_slug(ref)}.sif"
    return _container_cache_root(settings) / "singularity" / image_name


def _container_sif_exec_mode(settings: ExtractionSettings) -> str:
    mode = (settings.container_sif_exec_mode or "auto").strip().lower()
    return mode if mode in {"auto", "sif", "sandbox"} else "auto"


def _container_runtime_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable

    for env_path in _active_python_env_path_entries():
        candidate = env_path / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _active_python_env_path_entries() -> list[Path]:
    env_bin = Path(sys.executable).resolve().parent
    paths = [env_bin]
    env_prefix = env_bin.parent
    env_sbin = env_prefix / "sbin"
    if env_bin.name == "bin" and env_sbin.is_dir() and (env_prefix / "conda-meta").is_dir():
        paths.append(env_sbin)
    return paths


def _direct_sif_mount_supported() -> bool:
    if not Path("/dev/fuse").exists():
        return False
    has_squashfuse = bool(
        _container_runtime_executable("squashfuse")
        or _container_runtime_executable("squashfuse_ll")
    )
    has_fusermount = bool(
        _container_runtime_executable("fusermount3")
        or _container_runtime_executable("fusermount")
    )
    return has_squashfuse and has_fusermount


def _sif_sandbox_cache_paths(
    *,
    image: str,
    runtime: ContainerRuntime,
    sif_path: Path,
    settings: ExtractionSettings,
) -> tuple[Path, Path]:
    base_name = _safe_slug(sif_path.name) or _safe_slug(image) or "image"
    digest = hashlib.sha256(
        f"{runtime.name}\n{_normalize_container_ref(image)}\n{sif_path.resolve()}".encode()
    ).hexdigest()[:16]
    sandbox_path = _container_cache_root(settings) / "sandboxes" / f"{base_name[:80]}--{digest}"
    metadata_path = sandbox_path.with_name(f"{sandbox_path.name}.gtsm.json")
    return sandbox_path, metadata_path


def _sif_sandbox_metadata(
    *,
    image: str,
    runtime: ContainerRuntime,
    sif_path: Path,
) -> dict[str, object]:
    stat = sif_path.stat()
    return {
        "schema_version": "0.1.0",
        "runtime": runtime.name,
        "image": _normalize_container_ref(image),
        "source_sif": str(sif_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "validated_exec": True,
    }


def _sif_sandbox_cache_valid(
    *,
    sandbox_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, object],
) -> bool:
    if not sandbox_path.is_dir() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(metadata.get(key) == value for key, value in expected_metadata.items())


def _sif_sandbox_failure(
    *,
    metadata_path: Path,
    expected_metadata: dict[str, object],
) -> dict[str, object] | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    comparable = {
        key: value for key, value in expected_metadata.items() if key != "validated_exec"
    }
    if not all(metadata.get(key) == value for key, value in comparable.items()):
        return None
    if metadata.get("validated_exec") is False:
        return metadata
    return None


def _write_sif_sandbox_failure(
    *,
    metadata_path: Path,
    expected_metadata: dict[str, object],
    result: subprocess.CompletedProcess,
) -> None:
    payload = {
        **expected_metadata,
        "validated_exec": False,
        "failed_at": datetime.now(UTC).isoformat(),
        "command": list(result.args) if isinstance(result.args, list) else [],
        "returncode": result.returncode,
        "stdout": _tail_text(result.stdout or ""),
        "stderr": _tail_text(result.stderr or ""),
        "error_text": _completed_error_text(result),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sif_squashfs_object_id(runtime: ContainerRuntime, sif_path: Path) -> str:
    result = _run_command(
        [runtime.executable, "sif", "list", str(sif_path)],
        timeout_seconds=60,
    )
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "FS (Squashfs" not in stripped:
            continue
        fields = stripped.split("|", 1)
        if fields and fields[0].strip().isdigit():
            return fields[0].strip()
    return ""


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _prepare_sif_sandbox_with_unsquashfs(
    *,
    runtime: ContainerRuntime,
    sif_path: Path,
    tmp_path: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess:
    object_id = _sif_squashfs_object_id(runtime, sif_path)
    if not object_id:
        return subprocess.CompletedProcess(
            [runtime.executable, "sif", "list", str(sif_path)],
            1,
            stdout="",
            stderr="No squashfs partition found in SIF image",
        )

    work_path = tmp_path.with_name(f"{tmp_path.name}.work")
    _remove_path(work_path)
    work_path.mkdir(parents=True, exist_ok=True)
    squashfs_path = work_path / "rootfs.sqfs"
    rootfs_path = work_path / "rootfs"
    dump_command = [runtime.executable, "sif", "dump", object_id, str(sif_path)]
    dump_result = _run_command_to_file(
        dump_command,
        squashfs_path,
        timeout_seconds=timeout_seconds,
    )
    if dump_result.returncode != 0:
        _remove_path(work_path)
        return dump_result

    unsquashfs = _container_runtime_executable("unsquashfs") or "unsquashfs"
    extract_command = [
        unsquashfs,
        "-quiet",
        "-no-progress",
        "-no-xattrs",
        "-ignore-errors",
        "-d",
        str(rootfs_path),
        str(squashfs_path),
    ]
    extract_result = _run_command(extract_command, timeout_seconds=timeout_seconds)
    if extract_result.returncode == 0 and rootfs_path.is_dir():
        _remove_path(tmp_path)
        rootfs_path.rename(tmp_path)
    _remove_path(work_path)
    return extract_result


def _prepare_sif_sandbox(
    *,
    preparation: ContainerPreparation,
    image: str,
    runtime: ContainerRuntime,
    settings: ExtractionSettings,
) -> ContainerPreparation:
    mode = _container_sif_exec_mode(settings)
    if mode == "sif" or preparation.runtime not in {"singularity", "apptainer"}:
        return preparation
    if mode == "auto" and _direct_sif_mount_supported():
        return preparation

    sif_path = Path(preparation.identifier)
    if not preparation.identifier or not sif_path.is_file():
        return preparation

    sandbox_path, metadata_path = _sif_sandbox_cache_paths(
        image=image, runtime=runtime, sif_path=sif_path, settings=settings
    )
    expected_metadata = _sif_sandbox_metadata(image=image, runtime=runtime, sif_path=sif_path)
    if _sif_sandbox_cache_valid(
        sandbox_path=sandbox_path,
        metadata_path=metadata_path,
        expected_metadata=expected_metadata,
    ):
        return replace(
            preparation,
            identifier=str(sandbox_path),
            source="sif-sandbox-cache",
        )
    prior_failure = _sif_sandbox_failure(
        metadata_path=metadata_path,
        expected_metadata=expected_metadata,
    )
    if prior_failure is not None:
        failed = ContainerPreparation(
            ok=False,
            runtime=runtime.name,
            image=preparation.image,
            identifier=str(sandbox_path),
            source="sif-sandbox-cache",
            command=[
                str(value)
                for value in prior_failure.get("command", [])
                if isinstance(value, str)
            ],
            returncode=int(prior_failure.get("returncode", 1) or 1),
            stdout=str(prior_failure.get("stdout", "") or ""),
            stderr=str(prior_failure.get("stderr", "") or ""),
            error_text=str(prior_failure.get("error_text", "") or "cached sandbox failure"),
        )
        return failed if mode == "sandbox" else preparation

    sandbox_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = sandbox_path.with_name(
        f".{sandbox_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    _remove_path(tmp_path)
    command = [runtime.executable, "build", "--sandbox", str(tmp_path), str(sif_path)]
    result = _run_command(command, timeout_seconds=_container_image_timeout(settings))
    if result.returncode != 0:
        _remove_path(tmp_path)
        command = [
            _container_runtime_executable("unsquashfs") or "unsquashfs",
            "-quiet",
            "-no-progress",
            "-no-xattrs",
            "-ignore-errors",
            "-d",
            str(tmp_path),
            "<sif-squashfs-partition>",
        ]
        result = _prepare_sif_sandbox_with_unsquashfs(
            runtime=runtime,
            sif_path=sif_path,
            tmp_path=tmp_path,
            timeout_seconds=_container_image_timeout(settings),
        )
        if isinstance(result.args, list):
            command = result.args
    if result.returncode == 0 and tmp_path.is_dir():
        validation_command = [
            runtime.executable,
            "exec",
            "--cleanenv",
            str(tmp_path),
            "sh",
            "-lc",
            "true",
        ]
        validation_result = _run_command(
            validation_command,
            timeout_seconds=min(30, _container_image_timeout(settings)),
        )
        if validation_result.returncode != 0:
            _remove_path(tmp_path)
            _write_sif_sandbox_failure(
                metadata_path=metadata_path,
                expected_metadata=expected_metadata,
                result=validation_result,
            )
            failed = ContainerPreparation(
                ok=False,
                runtime=runtime.name,
                image=preparation.image,
                identifier=str(sandbox_path),
                source="sif-sandbox-cache",
                command=validation_command,
                returncode=validation_result.returncode,
                stdout=_tail_text(validation_result.stdout),
                stderr=_tail_text(validation_result.stderr),
                error_text=_completed_error_text(validation_result),
            )
            return failed if mode == "sandbox" else preparation
        _remove_path(sandbox_path)
        tmp_path.rename(sandbox_path)
        payload = {
            **expected_metadata,
            "created_at": datetime.now(UTC).isoformat(),
            "source": preparation.source,
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return replace(
            preparation,
            identifier=str(sandbox_path),
            source="sif-sandbox-cache",
            command=command,
            stdout=_tail_text(result.stdout),
            stderr=_tail_text(result.stderr),
        )

    _remove_path(tmp_path)
    failed = ContainerPreparation(
        ok=False,
        runtime=runtime.name,
        image=preparation.image,
        identifier=str(sandbox_path),
        source="sif-sandbox-cache",
        command=command,
        returncode=result.returncode,
        stdout=_tail_text(result.stdout),
        stderr=_tail_text(result.stderr),
        error_text=_completed_error_text(result),
    )
    return failed if mode == "sandbox" else preparation


def _finalize_singularity_preparation(
    preparation: ContainerPreparation,
    *,
    image: str,
    runtime: ContainerRuntime,
    settings: ExtractionSettings,
) -> ContainerPreparation:
    if not preparation.ok:
        return preparation
    return _prepare_sif_sandbox(
        preparation=preparation,
        image=image,
        runtime=runtime,
        settings=settings,
    )


def _available_container_runtimes(settings: ExtractionSettings) -> list[ContainerRuntime]:
    requested = settings.container_runtime.strip().lower() or "auto"
    if requested not in {"auto", "singularity", "apptainer", "docker"}:
        raise ValueError(f"Unsupported container runtime: {settings.container_runtime}")

    runtimes: list[ContainerRuntime] = []
    if requested in {"auto", "singularity"}:
        executable = _container_runtime_executable("singularity")
        if executable:
            runtimes.append(ContainerRuntime(name="singularity", executable=executable))
        elif requested == "singularity":
            runtimes.append(ContainerRuntime(name="singularity", executable="singularity"))
    if requested in {"auto", "apptainer"}:
        executable = _container_runtime_executable("apptainer")
        if executable:
            runtimes.append(ContainerRuntime(name="apptainer", executable=executable))
        elif requested == "apptainer":
            runtimes.append(ContainerRuntime(name="apptainer", executable="apptainer"))
    if requested in {"auto", "docker"}:
        executable = _container_runtime_executable("docker")
        if executable:
            runtimes.append(ContainerRuntime(name="docker", executable=executable))
        elif requested == "docker":
            runtimes.append(ContainerRuntime(name="docker", executable="docker"))
    return runtimes


def _download_depot_image(
    url: str, destination: Path, timeout_seconds: int
) -> ContainerPreparation:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f"{destination.name}.tmp")
    try:
        with _requests_get_with_user_agent_fallback(
            url,
            timeout=timeout_seconds,
            stream=True,
        ) as response:
            if response.status_code != 200:
                return ContainerPreparation(
                    ok=False,
                    runtime="singularity",
                    image=url,
                    identifier=str(destination),
                    source="galaxy-depot",
                    returncode=response.status_code,
                    error_text=f"depot returned HTTP {response.status_code}",
                )
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp_path.replace(destination)
        return ContainerPreparation(
            ok=True,
            runtime="singularity",
            image=url,
            identifier=str(destination),
            source="galaxy-depot",
        )
    except requests.RequestException as error:
        if tmp_path.exists():
            tmp_path.unlink()
        return ContainerPreparation(
            ok=False,
            runtime="singularity",
            image=url,
            identifier=str(destination),
            source="galaxy-depot",
            error_text=str(error),
        )


def _prepare_singularity_container(
    image: str,
    runtime: ContainerRuntime,
    settings: ExtractionSettings,
) -> ContainerPreparation:
    image = _normalize_container_ref(image)
    ref_error = _container_ref_error(image)
    if ref_error:
        return ContainerPreparation(
            ok=False,
            runtime=runtime.name,
            image=image,
            source="invalid-ref",
            returncode=1,
            error_text=ref_error,
        )

    image_path = Path(image)
    if image_path.exists():
        return _finalize_singularity_preparation(
            ContainerPreparation(
                ok=True,
                runtime=runtime.name,
                image=image,
                identifier=str(image_path.resolve()),
                source="explicit-file",
            ),
            image=image,
            runtime=runtime,
            settings=settings,
        )

    cache_path = _singularity_cache_path(image, settings=settings)
    if cache_path.exists():
        return _finalize_singularity_preparation(
            ContainerPreparation(
                ok=True,
                runtime=runtime.name,
                image=image,
                identifier=str(cache_path),
                source="cache",
            ),
            image=image,
            runtime=runtime,
            settings=settings,
        )

    if _is_url_ref(image):
        source = (
            "galaxy-depot" if _is_depot_url(image, settings.singularity_depot_url) else "remote-url"
        )
        download_result = _download_depot_image(
            image,
            cache_path,
            timeout_seconds=_container_image_timeout(settings),
        )
        if download_result.ok:
            return _finalize_singularity_preparation(
                ContainerPreparation(
                    ok=True,
                    runtime=runtime.name,
                    image=image,
                    identifier=str(cache_path),
                    source=source,
                ),
                image=image,
                runtime=runtime,
                settings=settings,
            )
        return ContainerPreparation(
            ok=False,
            runtime=runtime.name,
            image=image,
            identifier=str(cache_path),
            source=source,
            returncode=download_result.returncode,
            stdout=download_result.stdout,
            stderr=download_result.stderr,
            error_text=download_result.error_text,
        )

    depot_url = _singularity_depot_image_url(image, settings.singularity_depot_url)
    if depot_url:
        depot_result = _download_depot_image(
            depot_url,
            cache_path,
            timeout_seconds=_container_image_timeout(settings),
        )
        if depot_result.ok:
            return _finalize_singularity_preparation(
                ContainerPreparation(
                    ok=True,
                    runtime=runtime.name,
                    image=image,
                    identifier=str(cache_path),
                    source="galaxy-depot",
                ),
                image=image,
                runtime=runtime,
                settings=settings,
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_ref = _singularity_source_ref(image)
    if not source_ref:
        return ContainerPreparation(
            ok=False,
            runtime=runtime.name,
            image=image,
            identifier=str(cache_path),
            source="invalid-ref",
            returncode=1,
            error_text="container reference cannot be converted to a Singularity source",
        )
    command = [runtime.executable, "build", str(cache_path), source_ref]
    result = _run_command(command, timeout_seconds=_container_image_timeout(settings))
    if result.returncode == 0:
        return _finalize_singularity_preparation(
            ContainerPreparation(
                ok=True,
                runtime=runtime.name,
                image=image,
                identifier=str(cache_path),
                source="docker-build",
                command=command,
                stdout=_tail_text(result.stdout),
                stderr=_tail_text(result.stderr),
            ),
            image=image,
            runtime=runtime,
            settings=settings,
        )
    return ContainerPreparation(
        ok=False,
        runtime=runtime.name,
        image=image,
        identifier=str(cache_path),
        source="docker-build",
        command=command,
        returncode=result.returncode,
        stdout=_tail_text(result.stdout),
        stderr=_tail_text(result.stderr),
        error_text=_completed_error_text(result),
    )


def _prepare_docker_container(
    image: str,
    runtime: ContainerRuntime,
    settings: ExtractionSettings,
) -> ContainerPreparation:
    docker_image = _docker_ref_for_image(image)
    if not docker_image:
        ref = _normalize_container_ref(image)
        reason = (
            "Docker cannot pull remote Singularity/Apptainer image URLs"
            if _is_url_ref(ref)
            else (
                _container_ref_error(ref)
                or "container reference cannot be converted to a Docker image"
            )
        )
        return ContainerPreparation(
            ok=False,
            runtime=runtime.name,
            image=ref,
            identifier=ref,
            source="unsupported-docker-source",
            returncode=1,
            error_text=reason,
        )
    pull = _run_command(
        _docker_cmd("pull", docker_image, use_sudo=settings.docker_use_sudo),
        timeout_seconds=_container_image_timeout(settings),
    )
    return ContainerPreparation(
        ok=pull.returncode == 0,
        runtime=runtime.name,
        image=docker_image,
        identifier=docker_image,
        source="docker-pull",
        command=_docker_cmd("pull", docker_image, use_sudo=settings.docker_use_sudo),
        returncode=pull.returncode,
        stdout=_tail_text(pull.stdout),
        stderr=_tail_text(pull.stderr),
        error_text="" if pull.returncode == 0 else _completed_error_text(pull),
    )


def _prepare_container(
    image: str,
    runtime: ContainerRuntime,
    settings: ExtractionSettings,
) -> ContainerPreparation:
    if runtime.name in {"singularity", "apptainer"}:
        return _prepare_singularity_container(image=image, runtime=runtime, settings=settings)
    if runtime.name == "docker":
        return _prepare_docker_container(image=image, runtime=runtime, settings=settings)
    return ContainerPreparation(
        ok=False,
        runtime=runtime.name,
        image=image,
        source="unsupported-runtime",
        error_text=f"Unsupported container runtime: {runtime.name}",
    )


def _container_shell_command(
    preparation: ContainerPreparation,
    shell_command: str,
    settings: ExtractionSettings,
    shell: str,
) -> list[str]:
    if preparation.runtime == "docker":
        return _docker_cmd(
            "run",
            "--rm",
            "--network",
            "none",
            "--workdir",
            "/tmp",
            "--env",
            "CI=1",
            "--env",
            "TERM=dumb",
            "--env",
            "NO_COLOR=1",
            preparation.identifier,
            shell,
            "-lc",
            shell_command,
            use_sudo=settings.docker_use_sudo,
        )
    executable = _container_runtime_executable(preparation.runtime) or preparation.runtime
    return [
        executable,
        "exec",
        "--cleanenv",
        preparation.identifier,
        shell,
        "-lc",
        shell_command,
    ]


def _container_run_command(
    preparation: ContainerPreparation,
    command: str,
    settings: ExtractionSettings,
    shell: str = "bash",
) -> list[str]:
    return _container_shell_command(
        preparation=preparation,
        shell_command=_probe_shell_command(command),
        settings=settings,
        shell=shell,
    )


def _probe_shell_command(command: str) -> str:
    return (
        "export CI=1 TERM=dumb NO_COLOR=1 PAGER=cat "
        "TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8 "
        "GALAXY_MEMORY_MB=1024; "
        "tmpdir=$(mktemp -d /tmp/gtsm-help.XXXXXX 2>/dev/null || mktemp -d); "
        "trap 'rm -rf \"$tmpdir\"' EXIT; "
        'cd "$tmpdir" || exit 125; '
        f"{command}"
    )


def _is_no_arg_probe(command: str) -> bool:
    parts = command.strip().split()
    if not parts:
        return False
    return not any(part.strip("'\"") in {"--help", "-h", "help", "--usage", "-?"} for part in parts[1:])


def _probe_command_base(command: str, primary: str = "") -> str:
    base_tokens: list[str] = []
    for token in _command_tokens(command):
        cleaned = token.strip("'\"")
        if _is_assignment_token(token):
            continue
        if cleaned in {"--help", "-h", "help", "--usage", "-?"}:
            continue
        base_tokens.append(token)
        if len(base_tokens) >= 4:
            break
    if base_tokens:
        return " ".join(base_tokens)
    return primary.strip() or _command_primary(command)


def _container_probe_timeout(command: str, settings: ExtractionSettings) -> int:
    if _is_no_arg_probe(command):
        return min(
            settings.container_run_timeout_seconds, settings.container_no_arg_timeout_seconds
        )
    return settings.container_run_timeout_seconds


def _shell_missing(result: subprocess.CompletedProcess, shell: str) -> bool:
    text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return result.returncode == 127 and shell in text and "not found" in text


def _run_container_probe(
    preparation: ContainerPreparation,
    command: str,
    settings: ExtractionSettings,
) -> subprocess.CompletedProcess:
    timeout_seconds = _container_probe_timeout(command, settings)
    result = _run_command(
        _container_run_command(
            preparation=preparation, command=command, settings=settings, shell="bash"
        ),
        timeout_seconds=timeout_seconds,
    )
    if _shell_missing(result, "bash"):
        return _run_command(
            _container_run_command(
                preparation=preparation, command=command, settings=settings, shell="sh"
            ),
            timeout_seconds=timeout_seconds,
        )
    return result


def _command_primary(command: str) -> str:
    parts = _command_tokens(command)
    for index, part in enumerate(parts):
        if _is_assignment_token(part):
            continue
        if _is_placeholder_token(part) or part.lower() in {"none", "null"}:
            continue
        if (
            index + 2 < len(parts)
            and part.lower() in {"python", "python2", "python3"}
            and parts[index + 1] == "-m"
        ):
            return " ".join(parts[index : index + 3])
        with suppress(OSError, RuntimeError):
            if Path(part).resolve() == Path(sys.executable).resolve():
                if index + 2 < len(parts) and parts[index + 1] == "-m":
                    return " ".join(["python", *parts[index + 1 : index + 3]])
                return "python"
        return part
    return ""


def _command_presence_executable(primary: str) -> str:
    tokens = _command_tokens(primary)
    for token in tokens:
        if _is_assignment_token(token):
            continue
        return token
    return primary.strip()


def _run_container_command_exists(
    preparation: ContainerPreparation,
    primary: str,
    settings: ExtractionSettings,
) -> subprocess.CompletedProcess:
    executable = _command_presence_executable(primary)
    shell_command = _probe_shell_command(f"command -v {shlex.quote(executable)}")
    timeout_seconds = min(settings.container_run_timeout_seconds, 30)
    result = _run_command(
        _container_shell_command(
            preparation=preparation,
            shell_command=shell_command,
            settings=settings,
            shell="bash",
        ),
        timeout_seconds=timeout_seconds,
    )
    if _shell_missing(result, "bash"):
        return _run_command(
            _container_shell_command(
                preparation=preparation,
                shell_command=shell_command,
                settings=settings,
                shell="sh",
            ),
            timeout_seconds=timeout_seconds,
        )
    return result


def _container_command_presence_key(
    image: str,
    primary: str,
    preparation: ContainerPreparation,
) -> tuple[str, ...]:
    identifier = preparation.identifier or preparation.image or image
    return (preparation.runtime, identifier, primary)


def _cached_container_command_exists(
    state: ContainerExecutionState,
    image: str,
    primary: str,
    preparation: ContainerPreparation,
    settings: ExtractionSettings,
) -> subprocess.CompletedProcess:
    presence_key = _container_command_presence_key(image, primary, preparation)
    with state.command_presence_lock:
        cached = state.command_presence_cache.get(presence_key)
    if cached is not None:
        return cached
    result = _run_container_command_exists(
        preparation=preparation,
        primary=primary,
        settings=settings,
    )
    with state.command_presence_lock:
        return state.command_presence_cache.setdefault(presence_key, result)


def _run_container_api_validation_probe(
    preparation: ContainerPreparation,
    command: str,
    settings: ExtractionSettings,
) -> subprocess.CompletedProcess:
    tokens = _command_tokens(command)
    with suppress(OSError, RuntimeError):
        if tokens and Path(tokens[0]).resolve() == Path(sys.executable).resolve():
            tokens[0] = "python"
            command = shlex.join(tokens)
    timeout_seconds = settings.container_run_timeout_seconds
    result = _run_command(
        _container_run_command(
            preparation=preparation, command=command, settings=settings, shell="bash"
        ),
        timeout_seconds=timeout_seconds,
    )
    if _shell_missing(result, "bash"):
        return _run_command(
            _container_run_command(
                preparation=preparation, command=command, settings=settings, shell="sh"
            ),
            timeout_seconds=timeout_seconds,
        )
    return result


def _record_api_validation_checks(record: ToolRecord) -> tuple[list[str], list[tuple[str, str]]]:
    python_checks: list[str] = []
    r_checks: list[tuple[str, str]] = []
    seen_python: set[str] = set()
    seen_r: set[tuple[str, str]] = set()
    script_sources = [*record.wrapper_configfiles, *record.wrapper_helper_files]
    for source in script_sources:
        for call in source.get("api_calls", []) or []:
            if not isinstance(call, dict):
                continue
            language = str(call.get("language", "") or "").lower()
            qualified_call = str(call.get("qualified_call", "") or "").strip()
            if language == "python" and "." in qualified_call:
                if qualified_call not in seen_python:
                    seen_python.add(qualified_call)
                    python_checks.append(qualified_call)
            elif language == "r" and "::" in qualified_call:
                package, name = qualified_call.split("::", 1)
                key = (package.strip(), name.strip())
                if key[0] and key[1] and key not in seen_r:
                    seen_r.add(key)
                    r_checks.append(key)
    return python_checks[:64], r_checks[:64]


def _python_api_validation_command(checks: list[str]) -> str:
    checks_json = json.dumps(checks)
    code = "\n".join(
        [
            "import importlib, inspect, json",
            f"checks = json.loads({checks_json!r})",
            "docs = []",
            "errors = []",
            "for qualified in checks:",
            "    try:",
            "        parts = qualified.split('.')",
            "        obj = None",
            "        remainder = []",
            "        last_error = None",
            "        for split_at in range(len(parts), 0, -1):",
            "            module_name = '.'.join(parts[:split_at])",
            "            try:",
            "                obj = importlib.import_module(module_name)",
            "                remainder = parts[split_at:]",
            "                break",
            "            except Exception as exc:",
            "                last_error = exc",
            "        if obj is None:",
            "            raise last_error or ImportError(qualified)",
            "        for part in remainder:",
            "            obj = getattr(obj, part)",
            "    except Exception as exc:",
            "        errors.append({'qualified_call': qualified, 'error': repr(exc)[:240]})",
            "        continue",
            f"    if len(docs) < {_CONFIGFILE_API_DOC_LIMIT}:",
            "        try:",
            "            signature = str(inspect.signature(obj))",
            "        except Exception:",
            "            signature = ''",
            "        doc = (inspect.getdoc(obj) or '').splitlines()",
            "        docs.append({'qualified_call': qualified, 'signature': signature[:240], "
            f"'doc': (doc[0] if doc else '')[:{_CONFIGFILE_API_DOCSTRING_LIMIT}]}})",
            "if docs:",
            "    print('GTSM_API_VALIDATION_OK python %d' % len(docs))",
            "else:",
            "    print('GTSM_API_VALIDATION_FAILED python %d' % len(checks))",
            "print('GTSM_API_DOCS ' + json.dumps(docs, separators=(',', ':')))",
            "print('GTSM_API_ERRORS ' + json.dumps(errors[:24], separators=(',', ':')))",
            "raise SystemExit(0 if docs else 1)",
        ]
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _r_string_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _r_api_validation_command(checks: list[tuple[str, str]]) -> str:
    checks_literal = ", ".join(
        f"c({_r_string_literal(package)}, {_r_string_literal(name)})"
        for package, name in checks
    )
    code = (
        f"checks <- list({checks_literal}); "
        "for (check in checks) { "
        "pkg <- check[[1]]; name <- check[[2]]; "
        "if (!requireNamespace(pkg, quietly=TRUE)) stop(pkg); "
        "getExportedValue(pkg, name); "
        "}; "
        "cat('GTSM_API_VALIDATION_OK r ', length(checks), '\\n', sep='')"
    )
    return f"Rscript -e {shlex.quote(code)}"


def _record_api_validation_commands(record: ToolRecord) -> list[dict]:
    python_checks, r_checks = _record_api_validation_checks(record)
    commands: list[dict] = []
    if python_checks:
        commands.append(
            {
                "language": "python",
                "command": _python_api_validation_command(python_checks),
                "check_count": len(python_checks),
                "checks": python_checks,
            }
        )
    if r_checks:
        commands.append(
            {
                "language": "r",
                "command": _r_api_validation_command(r_checks),
                "check_count": len(r_checks),
                "checks": [f"{package}::{name}" for package, name in r_checks],
            }
        )
    return commands


def _container_api_validation_status(result: subprocess.CompletedProcess) -> str:
    text = _combined_container_text(result)
    if result.returncode == 0 and "GTSM_API_VALIDATION_OK" in text:
        return "container-api-validation-ok"
    return "container-api-validation-failed"


def _container_api_docs(result: subprocess.CompletedProcess) -> list[dict[str, str]]:
    text = _combined_container_text(result)
    for line in text.splitlines():
        if not line.startswith("GTSM_API_DOCS "):
            continue
        try:
            payload = json.loads(line[len("GTSM_API_DOCS ") :])
        except ValueError:
            return []
        if not isinstance(payload, list):
            return []
        docs: list[dict[str, str]] = []
        for item in payload[:_CONFIGFILE_API_DOC_LIMIT]:
            if not isinstance(item, dict):
                continue
            qualified_call = str(item.get("qualified_call", "") or "").strip()
            if not qualified_call:
                continue
            docs.append(
                {
                    "qualified_call": qualified_call,
                    "signature": _single_line_text(str(item.get("signature", "") or ""), limit=240),
                    "doc": _single_line_text(
                        str(item.get("doc", "") or ""),
                        limit=_CONFIGFILE_API_DOCSTRING_LIMIT,
                    ),
                }
            )
        return docs
    return []


def _container_api_errors(result: subprocess.CompletedProcess) -> list[dict[str, str]]:
    text = _combined_container_text(result)
    for line in text.splitlines():
        if not line.startswith("GTSM_API_ERRORS "):
            continue
        try:
            payload = json.loads(line[len("GTSM_API_ERRORS ") :])
        except ValueError:
            return []
        if not isinstance(payload, list):
            return []
        errors: list[dict[str, str]] = []
        for item in payload[:24]:
            if not isinstance(item, dict):
                continue
            qualified_call = str(item.get("qualified_call", "") or "").strip()
            error = str(item.get("error", "") or "").strip()
            if qualified_call and error:
                errors.append({"qualified_call": qualified_call, "error": error[:240]})
        return errors
    return []


def _remove_prepared_container(
    preparation: ContainerPreparation, settings: ExtractionSettings
) -> subprocess.CompletedProcess:
    if preparation.runtime == "docker":
        return _run_command(
            _docker_cmd(
                "image", "rm", "-f", preparation.identifier, use_sudo=settings.docker_use_sudo
            ),
            timeout_seconds=120,
        )
    if preparation.source != "cache" and preparation.identifier:
        path = Path(preparation.identifier)
        if path.exists():
            path.unlink()
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _normalize_help_probe_mode(mode: str) -> str:
    cleaned = (mode or "exploratory").strip().lower()
    return cleaned if cleaned in _HELP_PROBE_MODES else "exploratory"


def _help_probe_commands(
    base: str,
    probe_mode: str,
    help_flags: tuple[str, ...] | list[str] | None = None,
    allow_positional_help: bool = True,
) -> list[str]:
    help_flags = help_flags or _HELP_FLAGS
    commands = [f"{base} {flag}" for flag in help_flags]
    if _normalize_help_probe_mode(probe_mode) == "exploratory":
        commands.append(f"{base} --usage")
        commands.append(f"{base} {shlex.quote('-?')}")
        if allow_positional_help:
            commands.append(f"{base} help")
        commands.append(base)
    return commands


def _allows_positional_help_probe(primary: str) -> bool:
    return _normalized_command_key(primary) not in {"bwa", "bwamem2"}


def _extract_help_commands(
    primary: str,
    subcommands: list[str],
    probe_mode: str = "exploratory",
    help_flags: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    commands: list[str] = []
    if primary:
        if _normalized_command_key(primary) == "bcftools" and subcommands:
            help_flags = help_flags or _HELP_FLAGS
            for subcommand in subcommands:
                cleaned = subcommand.strip()
                if cleaned:
                    for flag in help_flags:
                        commands.append(f"{primary} {cleaned} {flag}")
                    if _normalize_help_probe_mode(probe_mode) == "exploratory":
                        commands.append(f"{primary} help {cleaned}")
            return commands
        allow_positional_help = _allows_positional_help_probe(primary)
        subcommand_bases = [f"{primary} {sub.strip()}" for sub in subcommands if sub.strip()]
        if allow_positional_help:
            bases = [*subcommand_bases, primary]
        else:
            bases = [primary, *subcommand_bases]
        for base in bases:
            commands.extend(
                _help_probe_commands(
                    base,
                    probe_mode,
                    help_flags=help_flags,
                    allow_positional_help=allow_positional_help,
                )
            )
        if _normalize_help_probe_mode(probe_mode) == "exploratory" and allow_positional_help:
            for subcommand in subcommands:
                cleaned = subcommand.strip()
                if not cleaned:
                    continue
                commands.append(f"{primary} help {cleaned}")
                commands.append(f"{primary} {cleaned} help")
    deduped: list[str] = []
    seen: set[str] = set()
    for command in commands:
        if command in seen:
            continue
        seen.add(command)
        deduped.append(command)
    return deduped


def _record_help_flags(record: ToolRecord, primary: str) -> tuple[str, ...]:
    version_command = str(record.version_command_text or "")
    if not version_command or not primary:
        return _HELP_FLAGS
    primary_re = re.escape(primary)
    if re.search(rf"(?<!\S){primary_re}\s+-h(?:\s|$|[|;&])", version_command):
        return ("-h", "--help")
    if re.search(rf"(?<!\S){primary_re}\s+--help(?:\s|$|[|;&])", version_command):
        return ("--help", "-h")
    return _HELP_FLAGS


def _unquote_shell_value(value: str) -> str:
    try:
        parts = shlex.split(value, posix=True)
    except ValueError:
        parts = []
    if parts:
        return parts[0]
    return value.strip().strip("\"'")


def _safe_probe_environment_assignment(name: str, value: str) -> str:
    name = name.strip()
    value = _unquote_shell_value(value).strip()
    if not name or not value or "\n" in value or "\x00" in value:
        return ""
    if _is_placeholder_token(value) or "@" in value or re.search(r"\{[^}]+\}", value):
        return ""
    normalized_name = name.upper()
    if re.search(
        r"(?:^|_)(?:CONFIG|CONFIGFILE|CONFIG_FILE)(?:_|$)",
        normalized_name,
    ) and not Path(value).is_absolute():
        return ""
    return f"{name}={shlex.quote(value)}"


def _record_probe_environment_prefix(record: ToolRecord, primary: str) -> str:
    if not primary:
        return ""
    command_text = str(record.command_text or "")
    export_re = re.compile(
        r"\bexport\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>'[^']*'|\"[^\"]*\"|[^\s&;]+)\s*&&",
        re.MULTILINE,
    )
    primary_re = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(primary)}(?![A-Za-z0-9_.-])")
    for match in export_re.finditer(command_text):
        following = command_text[match.end() : match.end() + 800]
        if not primary_re.search(following):
            continue
        assignment = _safe_probe_environment_assignment(match.group("name"), match.group("value"))
        if assignment:
            return assignment
    for segment in _command_segments(command_text):
        tokens = _command_tokens(segment)
        assignments: list[str] = []
        for token in tokens:
            if _is_assignment_token(token):
                name, value = token.split("=", 1)
                assignment = _safe_probe_environment_assignment(name, value)
                if assignment:
                    assignments.append(assignment)
                continue
            if token == primary and assignments:
                return " ".join(assignments[:4])
            if token == primary:
                break
    return ""


def _with_environment_prefix(command: str, environment_prefix: str) -> str:
    environment_prefix = environment_prefix.strip()
    if not environment_prefix:
        return command
    return f"{environment_prefix} {command}"


def _looks_like_help_text(text: str) -> bool:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in ("usage:", "options:", "optional arguments", "--help", "commands:")
    ):
        return True
    if "help follows" in lowered and re.search(r"(?im)^\s*command\s+description\b", text):
        return True
    if re.search(r"(?im)^\s*(?:usage|options?|commands?)\s*:", text):
        return True
    if re.search(r"(?im)^\s*description\s*:", text) and re.search(
        r"(?im)^\s*usage\s*:", text
    ):
        return True
    if re.search(r"(?im)^\s*synopsis\s*$", text) and re.search(
        r"(?im)^\s*(?:options?|arguments?|parameters?)\s*$", text
    ):
        return True
    if "tool:" in lowered and "summary:" in lowered:
        return True
    if any(
        marker in lowered
        for marker in (
            "available analysis command line options",
            "available command line options",
            "for more information about a specific command",
            "commandname(help)",
        )
    ):
        return True
    option_lines = 0
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:#|//|\*)+\s*", "", line)
        if re.match(
            r"\s*(?:\[?\(?-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*|\[?\(?-[A-Za-z](?:,|\)|\||\s))", line
        ):
            option_lines += 1
    return option_lines >= 3


_DEGRADED_HELP_MARKERS = (
    "command not recognized",
    "illegal option",
    "need -g",
    "unrecognized parameter",
    "unrecognized option",
    "unknown option",
)
_WEAK_FATAL_HELP_MARKERS = (
    "cannot change locale",
    "warning:",
)
_FATAL_PROBE_MARKERS = (
    "can't open file",
    "cannot open file",
    "command not found",
    "could not find or load main class",
    "invalid choice",
    "no such file or directory",
    "not found",
    "syntax error",
    "you ran:",
)


def _looks_like_failed_probe_text(text: str) -> bool:
    lowered = text.lower()
    if _looks_like_help_text(text):
        strong_fatal_markers = tuple(
            marker
            for marker in _FATAL_PROBE_MARKERS
            if marker not in {"no such file or directory", "not found"}
        )
        if not any(marker in lowered for marker in strong_fatal_markers):
            return False
        if any(marker in lowered for marker in _WEAK_FATAL_HELP_MARKERS) and not any(
            marker in lowered
            for marker in ("command not found", "could not find or load main class", "you ran:")
        ):
            return False
    return any(marker in lowered for marker in (*_FATAL_PROBE_MARKERS, *_DEGRADED_HELP_MARKERS))


def _looks_like_missing_argument_traceback(text: str) -> bool:
    lowered = text.lower()
    if _looks_like_help_text(text):
        return False
    return (
        "traceback" in lowered
        and "indexerror: list index out of range" in lowered
        and "sys.argv[" in lowered
    )


def _looks_like_runtime_import_traceback(text: str) -> bool:
    lowered = text.lower()
    if _looks_like_help_text(text) or "traceback" not in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "attributeerror:",
            "importerror:",
            "modulenotfounderror:",
            "nameerror:",
            "cannot open shared object file",
            "global name 'exc'",
            "pkg_resources",
        )
    )


def _combined_container_text(result: subprocess.CompletedProcess) -> str:
    return "\n".join(
        part
        for part in (
            _strip_container_runtime_noise(result.stdout or ""),
            _strip_container_runtime_noise(result.stderr or ""),
        )
        if part
    ).strip()


def _looks_like_degraded_help_text(text: str) -> bool:
    lowered = text.lower()
    if not _looks_like_help_text(text):
        return False
    if not any(marker in lowered for marker in _DEGRADED_HELP_MARKERS):
        return False
    strong_fatal_markers = tuple(
        marker
        for marker in _FATAL_PROBE_MARKERS
        if marker not in {"no such file or directory", "not found"}
    )
    if not any(marker in lowered for marker in strong_fatal_markers):
        return True
    return any(marker in lowered for marker in _WEAK_FATAL_HELP_MARKERS) and not any(
        marker in lowered
        for marker in ("command not found", "could not find or load main class", "you ran:")
    )


def _strip_degraded_help_boilerplate(text: str) -> str:
    lines = text.splitlines()
    keep_from = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if set(stripped) <= {"*"}:
            continue
        if (
            "error:" in lowered
            or any(marker in lowered for marker in _DEGRADED_HELP_MARKERS)
            or any(marker in lowered for marker in _WEAK_FATAL_HELP_MARKERS)
        ):
            continue
        keep_from = index
        break
    return "\n".join(lines[keep_from:]).strip()


def _container_help_fragment(command: str, result: subprocess.CompletedProcess) -> str:
    primary = _command_primary(command)
    if _is_shell_command_denylisted_token(primary):
        return ""
    text = _combined_container_text(result)
    if not text:
        return ""
    if _looks_like_degraded_help_text(text):
        text = _strip_degraded_help_boilerplate(text)
        if _looks_like_help_text(text):
            return f"$ {command}\n{text}"
    if _looks_like_help_text(text) and not _looks_like_failed_probe_text(text):
        return f"$ {command}\n{text}"
    return ""


def _looks_like_degraded_usage_text(text: str) -> bool:
    lowered = text.lower()
    if _looks_like_help_text(text):
        return False
    if "type 'help' or '?' for help" in lowered:
        return True
    if "please provide" in lowered and (
        "no further parameters" in lowered or "parameters are necessary" in lowered
    ):
        return True
    return bool(
        re.search(r"\b(?:expected|requires?)\s+\d+\s+(?:arguments?|parameters?)\b", lowered)
    )


def _container_usage_fragment(command: str, result: subprocess.CompletedProcess) -> str:
    primary = _command_primary(command)
    if _is_shell_command_denylisted_token(primary):
        return ""
    text = _combined_container_text(result)
    if not text or not _looks_like_degraded_usage_text(text):
        return ""
    return f"$ {command}\n{text}"


def _container_probe_status(
    result: subprocess.CompletedProcess, fragment: str, usage_fragment: str = ""
) -> str:
    text = _combined_container_text(result)
    if fragment:
        if _looks_like_degraded_help_text(text):
            return "container-command-help-degraded"
        return "container-command-help"
    if usage_fragment:
        return "container-command-usage-degraded"
    if (
        _looks_like_missing_argument_traceback(text)
        or _looks_like_runtime_import_traceback(text)
        or _looks_like_failed_probe_text(text)
        or result.returncode == 127
    ):
        return "container-command-failed-probe"
    return "container-command-nonhelp"


def _is_missing_command_probe(result: subprocess.CompletedProcess) -> bool:
    text = _combined_container_text(result).lower()
    return (
        result.returncode == 127
        or "command not found" in text
    )


def _api_validation_help_context(container_api_validation: list[dict] | None) -> str:
    events = container_api_validation or []
    ok_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("status") == "container-api-validation-ok"
    ]
    if not ok_events:
        return ""
    lines = ["Runtime API validation from container execution:"]
    for event in ok_events[:4]:
        language = str(event.get("language", "") or "api").strip()
        checks = [
            str(check).strip()
            for check in (event.get("checks", []) or [])[:_CONFIGFILE_HELP_CONTEXT_API_LIMIT]
            if str(check).strip()
        ]
        suffix = ""
        check_count = int(event.get("check_count", 0) or 0)
        if check_count > len(checks):
            suffix = f" (+{check_count - len(checks)} more)"
        if checks:
            lines.append(f"- {language}: {', '.join(checks)}{suffix}")
        for doc in (event.get("api_docs", []) or [])[:8]:
            if not isinstance(doc, dict):
                continue
            call = str(doc.get("qualified_call", "") or "").strip()
            signature = str(doc.get("signature", "") or "").strip()
            text = str(doc.get("doc", "") or "").strip()
            detail = signature or text
            if call and detail:
                lines.append(f"  - {call}: {_single_line_text(detail, limit=300)}")
    return "\n".join(lines)


def _combine_help_text(
    original_help_text: str,
    container_help_text: str,
    container_usage_text: str = "",
    container_api_validation: list[dict] | None = None,
) -> str:
    sections = []
    if original_help_text.strip():
        sections.append(original_help_text.strip())
    if container_help_text.strip():
        sections.append(
            "Command-line help collected from container execution:\n\n"
            + container_help_text.strip()
        )
    if container_usage_text.strip():
        sections.append(
            "Runtime usage text collected from container execution:\n\n"
            + container_usage_text.strip()
        )
    api_context = _api_validation_help_context(container_api_validation)
    if api_context.strip():
        sections.append(api_context)
    return "\n\n".join(sections)


def _subprocess_env_with_active_paths() -> dict[str, str]:
    env = os.environ.copy()
    path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    env_path_entries = [
        str(path) for path in _active_python_env_path_entries() if str(path) not in path_parts
    ]
    if env_path_entries:
        env["PATH"] = os.pathsep.join(env_path_entries + path_parts)
    return env


def _run_command(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    env = _subprocess_env_with_active_paths()
    try:
        return subprocess.run(
            command,
            input="",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        detail = stderr or stdout or f"Command timed out after {timeout_seconds} seconds"
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=detail)
    except FileNotFoundError as error:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(error))


def _run_command_to_file(
    command: list[str],
    output_path: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess:
    env = _subprocess_env_with_active_paths()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output_handle:
            result = subprocess.run(
                command,
                input=b"",
                stdout=output_handle,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
                env=env,
            )
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        return subprocess.CompletedProcess(command, result.returncode, stdout="", stderr=stderr)
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else ""
        detail = stderr or f"Command timed out after {timeout_seconds} seconds"
        return subprocess.CompletedProcess(command, 124, stdout="", stderr=detail)
    except FileNotFoundError as error:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(error))


def _emit_command_status(
    settings: ExtractionSettings,
    *,
    status: str,
    step: str,
    command: list[str],
    result: subprocess.CompletedProcess,
    **extra: object,
) -> None:
    _emit_extract_status(
        settings,
        {
            "status": status,
            "step": step,
            "command": _command_display(command),
            "returncode": result.returncode,
            "error_text": "" if result.returncode == 0 else _completed_error_text(result),
            **extra,
        },
    )


def _ensure_conda_recipe_repo(
    *,
    cache_root: Path,
    repo_name: str,
    clone_url: str,
    ref: str,
    status_prefix: str,
    settings: ExtractionSettings,
) -> tuple[Path | None, str]:
    repo_root = cache_root / repo_name
    with _cache_lock(repo_root):
        _emit_extract_status(
            settings,
            {
                "status": f"{status_prefix}-repo-prepare",
                "repo": str(repo_root),
                "ref": ref,
                "source": "cache" if repo_root.exists() else "clone",
            },
        )
        if not repo_root.exists():
            repo_root.parent.mkdir(parents=True, exist_ok=True)
            command = ["git", "clone", clone_url, str(repo_root)]
            result = _run_command(command, timeout_seconds=300)
            _emit_command_status(
                settings,
                status=f"{status_prefix}-repo-command",
                step="clone",
                command=command,
                result=result,
                repo=str(repo_root),
                ref=ref,
            )
            if result.returncode != 0:
                error_text = _completed_error_text(result)
                _emit_extract_status(
                    settings,
                    {
                        "status": f"{status_prefix}-repo-unavailable",
                        "repo": str(repo_root),
                        "ref": ref,
                        "returncode": result.returncode,
                        "error_text": error_text,
                    },
                )
                return None, error_text

        if not (repo_root / ".git").exists():
            error_text = f"{repo_root} exists but is not a git checkout"
            _emit_extract_status(
                settings,
                {
                    "status": f"{status_prefix}-repo-unavailable",
                    "repo": str(repo_root),
                    "ref": ref,
                    "returncode": 1,
                    "error_text": error_text,
                },
            )
            return None, error_text

        fetch_command = ["git", "-C", str(repo_root), "fetch", "--all", "--tags"]
        fetch_result = _run_command(fetch_command, timeout_seconds=300)
        _emit_command_status(
            settings,
            status=f"{status_prefix}-repo-command",
            step="fetch",
            command=fetch_command,
            result=fetch_result,
            repo=str(repo_root),
            ref=ref,
        )

        checkout_result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        if ref and ref != "HEAD":
            checkout_command = ["git", "-C", str(repo_root), "checkout", ref]
            checkout_result = _run_command(checkout_command, timeout_seconds=120)
            _emit_command_status(
                settings,
                status=f"{status_prefix}-repo-command",
                step="checkout",
                command=checkout_command,
                result=checkout_result,
                repo=str(repo_root),
                ref=ref,
            )
        repo_warning = ""
        if fetch_result.returncode != 0:
            repo_warning = _completed_error_text(fetch_result)
        if checkout_result.returncode != 0:
            repo_warning = _completed_error_text(checkout_result)

        _emit_extract_status(
            settings,
            {
                "status": f"{status_prefix}-repo-ready",
                "repo": str(repo_root),
                "ref": ref,
                "returncode": checkout_result.returncode
                if checkout_result.returncode != 0
                else fetch_result.returncode,
                "error_text": repo_warning,
            },
        )
        return repo_root, repo_warning


def _ensure_bioconda_repo(
    cache_root: Path, ref: str, settings: ExtractionSettings
) -> tuple[Path | None, str]:
    return _ensure_conda_recipe_repo(
        cache_root=cache_root,
        repo_name="bioconda-recipes",
        clone_url="https://github.com/bioconda/bioconda-recipes.git",
        ref=ref,
        status_prefix="bioconda",
        settings=settings,
    )


def _conda_forge_feedstock_variants(package: str) -> list[str]:
    variants = _recipe_package_variants(package)
    alias_variants: list[str] = []
    for variant in variants:
        alias_variants.extend(_CONDA_FORGE_FEEDSTOCK_ALIASES.get(variant.lower(), ()))
    return _dedupe_strings([*alias_variants, *variants])


def _ensure_conda_forge_feedstock_repo(
    cache_root: Path,
    package: str,
    settings: ExtractionSettings,
) -> tuple[Path | None, str, str]:
    feedstock_root = cache_root / "conda-forge-feedstocks"
    cache_namespace = str(feedstock_root.resolve())
    last_error = ""
    for recipe_package in _conda_forge_feedstock_variants(package):
        cache_key = (cache_namespace, recipe_package.lower())
        with _CONDA_FORGE_FEEDSTOCK_CACHE_LOCK:
            cached = _CONDA_FORGE_FEEDSTOCK_CACHE.get(cache_key)
        if cached is not None:
            if cached[0] is not None:
                return cached
            last_error = cached[1]
            continue

        lookup_lock = _cache_lock(
            feedstock_root / ".gtsm-feedstock-lookup-locks" / _safe_slug(recipe_package)
        )
        with lookup_lock:
            with _CONDA_FORGE_FEEDSTOCK_CACHE_LOCK:
                cached = _CONDA_FORGE_FEEDSTOCK_CACHE.get(cache_key)
            if cached is not None:
                if cached[0] is not None:
                    return cached
                last_error = cached[1]
                continue

            repo_name = f"{recipe_package}-feedstock"
            repo, error = _ensure_conda_recipe_repo(
                cache_root=feedstock_root,
                repo_name=repo_name,
                clone_url=f"https://github.com/conda-forge/{repo_name}.git",
                ref="HEAD",
                status_prefix="conda-forge",
                settings=settings,
            )
            result = (repo, error, recipe_package)
            with _CONDA_FORGE_FEEDSTOCK_CACHE_LOCK:
                _CONDA_FORGE_FEEDSTOCK_CACHE.setdefault(cache_key, result)
                result = _CONDA_FORGE_FEEDSTOCK_CACHE[cache_key]
        if repo is not None:
            return result
        last_error = error
    return None, last_error or "conda-forge feedstock not found", package


def _parse_recipe_set_value(raw_value: str) -> object:
    value = _clean_recipe_scalar(raw_value.strip().rstrip(";").strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _render_recipe_set_expression(raw_value: str, context: dict[str, object]) -> tuple[object, str]:
    expression = raw_value.strip().rstrip(";").strip()
    if not expression:
        return "", ""
    if len(expression) >= 2 and expression[0] == expression[-1] and expression[0] in {"'", '"'}:
        return _parse_recipe_set_value(expression), ""
    lowered = expression.lower()
    if lowered in {"true", "false"} or re.fullmatch(r"-?\d+(?:\.\d+)?", expression):
        return _parse_recipe_set_value(expression), ""
    try:
        value = _JINJA_ENV.compile_expression(expression)(**context)
    except TemplateError as error:
        return _parse_recipe_set_value(expression), str(error)
    if isinstance(value, Undefined):
        return _parse_recipe_set_value(expression), f"undefined template expression: {expression}"
    if isinstance(value, str):
        return value.strip(), ""
    if value is None:
        return "", ""
    return value, ""


def _extract_recipe_set_variables(meta_text: str) -> dict[str, object]:
    variables: dict[str, object] = {}
    for match in _RECIPE_SET_RE.finditer(meta_text):
        name = match.group(1)
        value, _error = _render_recipe_set_expression(match.group(2), variables)
        variables[name] = value
    return variables


def _extract_recipe_version(meta_text: str) -> str:
    variables = _extract_recipe_set_variables(meta_text)
    if variables.get("version") is not None:
        return str(variables["version"]).strip()
    for line in meta_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version:"):
            return _clean_recipe_scalar(stripped.split(":", 1)[1])
    return ""


def _recipe_render_context(
    *,
    meta_text: str,
    package: str,
    requirement_version: str,
    recipe_version: str,
) -> dict[str, object]:
    context = _extract_recipe_set_variables(meta_text)
    context["package"] = package
    context["required_version"] = requirement_version
    context["recipe_version"] = recipe_version
    context.setdefault("cran_mirror", "https://cran.r-project.org")
    if not str(context.get("name", "") or "").strip():
        context["name"] = package
    if not str(context.get("version", "") or "").strip():
        context["version"] = recipe_version or requirement_version
    return context


def _has_unresolved_recipe_template(value: str) -> bool:
    return bool(_UNRESOLVED_TEMPLATE_RE.search(value))


def _render_recipe_template(value: str, context: dict[str, object]) -> tuple[str, str]:
    if not value:
        return "", ""
    try:
        rendered = _JINJA_ENV.from_string(value).render(context).strip()
    except TemplateError as error:
        return value, str(error)
    return rendered, ""


def _render_bioconda_source_fields(
    *,
    meta_text: str,
    package: str,
    requirement_version: str,
    recipe_version: str,
    source_url: str,
    source_ref: str,
) -> tuple[str, str, str]:
    context = _recipe_render_context(
        meta_text=meta_text,
        package=package,
        requirement_version=requirement_version,
        recipe_version=recipe_version,
    )
    rendered_url, url_error = _render_recipe_template(source_url, context)
    rendered_ref, ref_error = _render_recipe_template(source_ref, context)
    errors = [error for error in (url_error, ref_error) if error]
    unresolved = [
        value for value in (rendered_url, rendered_ref) if _has_unresolved_recipe_template(value)
    ]
    if unresolved:
        errors.append("unresolved template variables remain after rendering")
    return rendered_url, rendered_ref, "; ".join(errors)


def _render_bioconda_source_url_candidates(
    *,
    meta_text: str,
    package: str,
    requirement_version: str,
    recipe_version: str,
    source_urls: list[str],
    source_ref: str,
) -> tuple[list[str], str, str]:
    rendered: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for source_url in source_urls:
        rendered_url, rendered_ref, error = _render_bioconda_source_fields(
            meta_text=meta_text,
            package=package,
            requirement_version=requirement_version,
            recipe_version=recipe_version,
            source_url=source_url,
            source_ref=source_ref,
        )
        if error:
            errors.append(error)
            continue
        if rendered_url and rendered_url not in seen:
            seen.add(rendered_url)
            rendered.append(rendered_url)
        if rendered_ref:
            source_ref = rendered_ref
    return rendered, source_ref, "; ".join(sorted(set(errors)))


def _clean_recipe_scalar(value: str) -> str:
    cleaned = value.strip()
    quote_char = ""
    for index, char in enumerate(cleaned):
        if quote_char:
            if char == quote_char:
                quote_char = ""
            continue
        if char in {"'", '"'}:
            quote_char = char
            continue
        if char == "#" and (index == 0 or cleaned[index - 1].isspace()):
            cleaned = cleaned[:index].rstrip()
            break
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _extract_source_url_candidates(meta_text: str) -> tuple[list[str], str]:
    source_urls: list[str] = []
    source_ref = ""
    in_url_list = False
    url_list_indent = 0
    for line in meta_text.splitlines():
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if in_url_list:
            if indent > url_list_indent and stripped.startswith("- "):
                value = _clean_recipe_scalar(stripped[2:])
                if value:
                    source_urls.append(value)
                continue
            if indent <= url_list_indent:
                in_url_list = False
        if stripped.startswith("- url:"):
            value = _clean_recipe_scalar(stripped.split(":", 1)[1])
            if value:
                source_urls.append(value)
            else:
                in_url_list = True
                url_list_indent = indent
            continue
        if stripped.startswith("url:"):
            value = _clean_recipe_scalar(stripped.split(":", 1)[1])
            if value:
                source_urls.append(value)
            else:
                in_url_list = True
                url_list_indent = indent
            continue
        if stripped.startswith("- git_url:"):
            value = _clean_recipe_scalar(stripped.split(":", 1)[1])
            if value:
                source_urls.append(value)
            continue
        if stripped.startswith("git_url:"):
            value = _clean_recipe_scalar(stripped.split(":", 1)[1])
            if value:
                source_urls.append(value)
            continue
        if stripped.startswith("- git_rev:"):
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("git_rev:"):
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("- tag:") and not source_ref:
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("tag:") and not source_ref:
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
    return source_urls, source_ref


def _extract_source_fields(meta_text: str) -> tuple[str, str]:
    source_urls, source_ref = _extract_source_url_candidates(meta_text)
    return (source_urls[0] if source_urls else ""), source_ref


_SOURCE_PROVIDER_DENYLIST = {
    "bzip2",
    "fonts-conda-ecosystem",
    "libgcc",
    "libgcc-ng",
    "libstdcxx",
    "libstdcxx-ng",
    "openjdk",
    "perl",
    "pip",
    "python",
    "r-base",
    "setuptools",
    "xz",
    "zlib",
}
_SOURCE_PROVIDER_DENYLIST_KEYS = {
    re.sub(r"[^a-z0-9]+", "", package.lower()) for package in _SOURCE_PROVIDER_DENYLIST
}
_SOURCELESS_PACKAGE_KEYS = {*_SOURCE_PROVIDER_DENYLIST_KEYS, "fontscondaecosystem"}
_SOURCE_DOC_EXTENSIONS = {".md", ".rd", ".rst", ".txt"}
_SOURCE_DOC_FILENAME_PREFIXES = (
    "readme",
    "usage",
    "manual",
    "tutorial",
    "help",
    "cli",
    "command",
    "commands",
    "example",
    "examples",
)
_SOURCE_DOC_DIR_NAMES = {
    "doc",
    "docs",
    "help",
    "man",
    "manual",
    "usage",
    "example",
    "examples",
    "vignette",
    "vignettes",
}
_SOURCE_DOC_EXCLUDED_FILENAMES = {
    "cmakelists.txt",
    "makefile",
    "makefile.in",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "configure",
    "configure.ac",
    "configure.in",
}
_SOURCE_DOC_MAX_FILES = 160
_SOURCE_DOC_SCAN_FILE_LIMIT = 500
_SOURCE_DOC_MAX_BYTES = 96_000
_SOURCE_DOC_MAX_DOCS = 8
_SOURCE_DOC_CONTEXT_LINES = 7


def _recipe_requirement_name(spec: str) -> str:
    cleaned = _clean_recipe_scalar(str(spec or ""))
    if not cleaned or _has_unresolved_recipe_template(cleaned):
        return ""
    cleaned = cleaned.split("#", 1)[0].strip()
    if not cleaned:
        return ""
    return re.split(r"\s+|[<>=!~]+", cleaned, maxsplit=1)[0].strip()


def _extract_recipe_run_dependency_names(meta_text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    in_requirements = False
    in_run = False
    requirements_indent = 0
    run_indent = 0
    for line in meta_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "requirements:":
            in_requirements = True
            in_run = False
            requirements_indent = indent
            continue
        if in_requirements and indent <= requirements_indent and not stripped.startswith("- "):
            in_requirements = False
            in_run = False
        if not in_requirements:
            continue
        if stripped == "run:":
            in_run = True
            run_indent = indent
            continue
        if in_run and indent <= run_indent and not stripped.startswith("- "):
            in_run = False
        if not in_run or not stripped.startswith("- "):
            continue
        name = _recipe_requirement_name(stripped[2:])
        key = _normalized_command_key(name)
        if not name or key in seen or key in _SOURCE_PROVIDER_DENYLIST_KEYS:
            continue
        seen.add(key)
        names.append(name)
    return names


def _is_source_denylisted_package(package: str) -> bool:
    return _normalized_command_key(package) in _SOURCE_PROVIDER_DENYLIST_KEYS


def _recipe_relpath(package: str, *, feedstock: bool = False, version: str = "") -> str:
    if feedstock:
        return "recipe/meta.yaml"
    if version:
        return f"recipes/{package}/{version}/meta.yaml"
    return f"recipes/{package}/meta.yaml"


def _recipe_version_dir_variants(version: str) -> list[str]:
    cleaned = _clean_recipe_scalar(str(version or "")).strip()
    normalized = _normalize_recipe_version_text(cleaned)
    variants = [
        cleaned,
        normalized,
        cleaned.removeprefix("v"),
        normalized.removeprefix("v"),
    ]
    return _dedupe_strings([variant for variant in variants if variant])


def _recipe_candidate_relpaths(
    package: str,
    *,
    required_version: str = "",
    feedstock: bool = False,
) -> list[str]:
    relpaths = [_recipe_relpath(package, feedstock=feedstock)]
    if not feedstock:
        for version in _recipe_version_dir_variants(required_version):
            relpaths.append(_recipe_relpath(package, version=version))
    return _dedupe_strings(relpaths)


def _recipe_versioned_worktree_relpaths(recipes_repo: Path, package: str) -> list[str]:
    recipe_dir = recipes_repo / "recipes" / package
    if not recipe_dir.is_dir():
        return []
    relpaths: list[str] = []
    for child in sorted(recipe_dir.iterdir()):
        meta_path = child / "meta.yaml"
        if child.is_dir() and meta_path.is_file():
            relpaths.append(str(meta_path.relative_to(recipes_repo)))
    return relpaths


def _recipe_versioned_git_relpaths(
    recipes_repo: Path,
    ref: str,
    package: str,
) -> tuple[list[str], str]:
    prefix = f"recipes/{package}/"
    result = _run_command(
        ["git", "-C", str(recipes_repo), "ls-tree", "-r", "--name-only", ref, "--", prefix],
        timeout_seconds=60,
    )
    if result.returncode != 0:
        return [], _completed_error_text(result)
    relpaths: list[str] = []
    for line in result.stdout.splitlines():
        relpath = line.strip()
        if not relpath.endswith("/meta.yaml"):
            continue
        parts = relpath.split("/")
        if len(parts) == 4 and parts[:2] == ["recipes", package]:
            relpaths.append(relpath)
    return _dedupe_strings(relpaths), ""


def _recipe_package_variants(package: str) -> list[str]:
    cleaned = str(package or "").strip()
    variants = [
        cleaned,
        cleaned.lower(),
        cleaned.replace("_", "-"),
        cleaned.replace("-", "_"),
        cleaned.lower().replace("_", "-"),
        cleaned.lower().replace("-", "_"),
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return deduped


def _normalize_recipe_version_text(value: str) -> str:
    cleaned = _clean_recipe_scalar(str(value or "")).lower()
    for prefix in ("==", ">=", "<=", "~=", "=", ">", "<"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    if cleaned.startswith("v") and len(cleaned) > 1 and cleaned[1].isdigit():
        cleaned = cleaned[1:]
    return cleaned


def _parse_semverish_version(value: str) -> SemverishVersion | None:
    normalized = _normalize_recipe_version_text(value)
    numeric = tuple(int(token) for token in re.findall(r"\d+", normalized))
    if not normalized or not numeric:
        return None
    return SemverishVersion(raw=value, normalized=normalized, numeric=numeric)


def _version_equivalence_tokens(value: str) -> tuple[tuple[str, object], ...]:
    normalized = _normalize_recipe_version_text(value)
    tokens: list[tuple[str, object]] = []
    for token in re.findall(r"\d+|[a-z]+", normalized):
        if token.isdigit():
            tokens.append(("number", int(token)))
        else:
            tokens.append(("text", token))
    while tokens and tokens[-1] == ("number", 0):
        tokens.pop()
    return tuple(tokens)


def _numeric_versions_equivalent(left: SemverishVersion, right: SemverishVersion) -> bool:
    if _compare_numeric_versions(left.numeric, right.numeric) != 0:
        return False
    return _version_equivalence_tokens(left.raw) == _version_equivalence_tokens(right.raw)


def _compare_numeric_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    left_padded = left + (0,) * (width - len(left))
    right_padded = right + (0,) * (width - len(right))
    if left_padded == right_padded:
        return 0
    return 1 if left_padded > right_padded else -1


def _numeric_version_distance(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    width = max(len(left), len(right))
    left_padded = left + (0,) * (width - len(left))
    right_padded = right + (0,) * (width - len(right))
    return tuple(abs(a - b) for a, b in zip(left_padded, right_padded, strict=False))


def _recipe_selection_rank(
    snapshot: BiocondaRecipeSnapshot,
    required: SemverishVersion,
) -> tuple[tuple[object, ...], str]:
    candidate = _parse_semverish_version(snapshot.recipe_version)
    if candidate is None:
        return ((6, snapshot.recipe_version), "low_confidence_recipe_version")
    if candidate.normalized == required.normalized:
        return ((0,), "exact")
    comparison = _compare_numeric_versions(candidate.numeric, required.numeric)
    if comparison == 0 and _numeric_versions_equivalent(candidate, required):
        return ((1, candidate.normalized), "numeric_equivalent")
    if comparison == 0:
        return ((2, candidate.normalized), "same_numeric_variant")
    candidate_major = candidate.numeric[0]
    required_major = required.numeric[0]
    if candidate_major == required_major and comparison > 0:
        return (
            (3, _numeric_version_distance(candidate.numeric[1:], required.numeric[1:])),
            "same_major_newer",
        )
    if candidate_major == required_major:
        return (
            (4, _numeric_version_distance(candidate.numeric[1:], required.numeric[1:])),
            "same_major_older",
        )
    older_major_preference = 0 if candidate_major < required_major else 1
    return (
        (
            5,
            abs(candidate_major - required_major),
            older_major_preference,
            _numeric_version_distance(candidate.numeric, required.numeric),
        ),
        "closest_major",
    )


def _select_recipe_snapshot_from_candidates(
    *,
    package: str,
    required_version: str,
    candidates: list[BiocondaRecipeSnapshot],
    scanned_commits: int,
) -> BiocondaRecipeSnapshot:
    if not candidates:
        return BiocondaRecipeSnapshot(
            package=package,
            recipe_path=_recipe_relpath(package),
            meta_text="",
            selection_reason="recipe_not_found",
            scanned_commits=scanned_commits,
            error="recipe_not_found",
        )
    required = _parse_semverish_version(required_version)
    if required is None:
        reason = "latest" if not required_version else "low_confidence_required_version"
        return replace(candidates[0], selection_reason=reason, scanned_commits=scanned_commits)

    ranked = []
    for index, snapshot in enumerate(candidates):
        rank, reason = _recipe_selection_rank(snapshot, required)
        ranked.append((rank, index, reason, snapshot))
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, _, reason, selected = ranked[0]
    return replace(selected, selection_reason=reason, scanned_commits=scanned_commits)


def _git_recipe_commit_for_ref(
    recipes_repo: Path,
    ref: str,
    package: str,
    *,
    feedstock: bool = False,
    relpath: str = "",
) -> tuple[str, str]:
    relpath = relpath or _recipe_relpath(package, feedstock=feedstock)
    result = _run_command(
        ["git", "-C", str(recipes_repo), "log", "-1", "--format=%H%x09%cI", ref, "--", relpath],
        timeout_seconds=60,
    )
    if result.returncode != 0:
        return "", ""
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if "\t" not in line:
        return line, ""
    commit, commit_date = line.split("\t", 1)
    return commit, commit_date


def _worktree_recipe_snapshot(
    recipes_repo: Path,
    package: str,
    *,
    feedstock: bool = False,
    relpath: str = "",
) -> BiocondaRecipeSnapshot:
    relpath = relpath or _recipe_relpath(package, feedstock=feedstock)
    meta_path = recipes_repo / relpath
    if not meta_path.exists():
        return BiocondaRecipeSnapshot(
            package=package,
            recipe_path=str(meta_path),
            meta_text="",
            selection_reason="recipe_not_found",
            error="recipe_not_found",
        )
    meta_text = meta_path.read_text(encoding="utf-8")
    return BiocondaRecipeSnapshot(
        package=package,
        recipe_path=str(meta_path),
        meta_text=meta_text,
        recipe_version=_extract_recipe_version(meta_text),
        selection_reason="worktree",
    )


def _recipe_snapshot_at_ref(
    recipes_repo: Path,
    ref: str,
    package: str,
    *,
    feedstock: bool = False,
    relpath: str = "",
) -> BiocondaRecipeSnapshot:
    relpath = relpath or _recipe_relpath(package, feedstock=feedstock)
    result = _run_command(
        ["git", "-C", str(recipes_repo), "show", f"{ref}:{relpath}"],
        timeout_seconds=60,
    )
    if result.returncode != 0:
        return BiocondaRecipeSnapshot(
            package=package,
            recipe_path=str(recipes_repo / relpath),
            meta_text="",
            selection_reason="recipe_not_found",
            error=_completed_error_text(result) or "recipe_not_found",
        )
    commit, commit_date = _git_recipe_commit_for_ref(
        recipes_repo, ref, package, feedstock=feedstock, relpath=relpath
    )
    meta_text = result.stdout
    return BiocondaRecipeSnapshot(
        package=package,
        recipe_path=f"{ref}:{relpath}",
        meta_text=meta_text,
        recipe_version=_extract_recipe_version(meta_text),
        commit=commit,
        commit_date=commit_date,
        selection_reason="current_ref",
        scanned_commits=1,
    )


def _recipe_history_candidates(
    recipes_repo: Path,
    ref: str,
    package: str,
    *,
    feedstock: bool = False,
    relpath: str = "",
) -> tuple[list[BiocondaRecipeSnapshot], int, str]:
    relpath = relpath or _recipe_relpath(package, feedstock=feedstock)
    log_result = _run_command(
        ["git", "-C", str(recipes_repo), "log", "--format=%H%x09%cI", ref, "--", relpath],
        timeout_seconds=300,
    )
    if log_result.returncode != 0:
        return [], 0, _completed_error_text(log_result)

    candidates: list[BiocondaRecipeSnapshot] = []
    seen_versions: set[str] = set()
    scanned_commits = 0
    for line in log_result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        scanned_commits += 1
        if "\t" in stripped:
            commit, commit_date = stripped.split("\t", 1)
        else:
            commit, commit_date = stripped, ""
        show_result = _run_command(
            ["git", "-C", str(recipes_repo), "show", f"{commit}:{relpath}"],
            timeout_seconds=60,
        )
        if show_result.returncode != 0:
            continue
        meta_text = show_result.stdout
        recipe_version = _extract_recipe_version(meta_text)
        version_key = _normalize_recipe_version_text(recipe_version) or f"commit:{commit}"
        if version_key in seen_versions:
            continue
        seen_versions.add(version_key)
        candidates.append(
            BiocondaRecipeSnapshot(
                package=package,
                recipe_path=f"{commit}:{relpath}",
                meta_text=meta_text,
                recipe_version=recipe_version,
                commit=commit,
                commit_date=commit_date,
                scanned_commits=scanned_commits,
            )
        )
    return candidates, scanned_commits, ""


def _recipe_selection_cache_key(
    recipes_repo: Path,
    ref: str,
    package: str,
    required_version: str,
) -> tuple[str, str, str, str]:
    return (str(recipes_repo.resolve()), ref, package.lower(), required_version)


def _recipe_selection_cache_get(
    recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None,
    recipe_selection_cache_lock: threading.Lock | None,
    cache_key: tuple[str, str, str, str],
) -> BiocondaRecipeSnapshot | None:
    if recipe_selection_cache is None:
        return None
    if recipe_selection_cache_lock is None:
        return recipe_selection_cache.get(cache_key)
    with recipe_selection_cache_lock:
        return recipe_selection_cache.get(cache_key)


def _recipe_selection_cache_put(
    recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None,
    recipe_selection_cache_lock: threading.Lock | None,
    cache_key: tuple[str, str, str, str],
    snapshot: BiocondaRecipeSnapshot,
) -> BiocondaRecipeSnapshot:
    if recipe_selection_cache is None:
        return snapshot
    if recipe_selection_cache_lock is None:
        recipe_selection_cache.setdefault(cache_key, snapshot)
        return recipe_selection_cache[cache_key]
    with recipe_selection_cache_lock:
        recipe_selection_cache.setdefault(cache_key, snapshot)
        return recipe_selection_cache[cache_key]


def _select_bioconda_recipe_snapshot_uncached(
    *,
    recipes_repo: Path,
    ref: str,
    package: str,
    required_version: str,
    feedstock: bool = False,
) -> BiocondaRecipeSnapshot:
    first_missing: BiocondaRecipeSnapshot | None = None
    recipe_packages = [package] if feedstock else _recipe_package_variants(package)
    for recipe_package in recipe_packages:
        candidate_relpaths = _recipe_candidate_relpaths(
            recipe_package,
            required_version=required_version,
            feedstock=feedstock,
        )
        if not (recipes_repo / ".git").exists():
            if not feedstock:
                candidate_relpaths.extend(
                    relpath
                    for relpath in _recipe_versioned_worktree_relpaths(
                        recipes_repo, recipe_package
                    )
                    if relpath not in candidate_relpaths
                )
            candidates: list[BiocondaRecipeSnapshot] = []
            for relpath in candidate_relpaths:
                snapshot = _worktree_recipe_snapshot(
                    recipes_repo,
                    recipe_package,
                    feedstock=feedstock,
                    relpath=relpath,
                )
                if not snapshot.meta_text:
                    first_missing = first_missing or snapshot
                    continue
                candidates.append(snapshot)
            if not candidates:
                continue
            if not required_version:
                return candidates[0]
            return _select_recipe_snapshot_from_candidates(
                package=recipe_package,
                required_version=required_version,
                candidates=candidates,
                scanned_commits=0,
            )

        discovered_relpaths: list[str] = []
        discovered_error = ""
        if not feedstock:
            discovered_relpaths, discovered_error = _recipe_versioned_git_relpaths(
                recipes_repo, ref, recipe_package
            )
            candidate_relpaths.extend(
                relpath for relpath in discovered_relpaths if relpath not in candidate_relpaths
            )

        current_snapshots: list[BiocondaRecipeSnapshot] = []
        history_candidates: list[BiocondaRecipeSnapshot] = []
        scanned_commits = 0
        history_error = discovered_error
        for relpath in candidate_relpaths:
            current_snapshot = _recipe_snapshot_at_ref(
                recipes_repo,
                ref,
                recipe_package,
                feedstock=feedstock,
                relpath=relpath,
            )
            if current_snapshot.meta_text:
                required = _normalize_recipe_version_text(required_version)
                recipe = _normalize_recipe_version_text(current_snapshot.recipe_version)
                if not required_version or (required and required == recipe):
                    return replace(
                        current_snapshot,
                        selection_reason="exact" if required_version else "current_ref",
                    )
                current_snapshots.append(current_snapshot)

            candidates, path_scanned_commits, path_history_error = _recipe_history_candidates(
                recipes_repo,
                ref,
                recipe_package,
                feedstock=feedstock,
                relpath=relpath,
            )
            scanned_commits += path_scanned_commits
            if candidates:
                history_candidates.extend(candidates)
            if path_history_error:
                history_error = path_history_error
            if not current_snapshot.meta_text:
                first_missing = first_missing or replace(
                    current_snapshot,
                    package=package,
                    error=current_snapshot.error or history_error or "recipe_not_found",
                )
        if history_candidates:
            return _select_recipe_snapshot_from_candidates(
                package=recipe_package,
                required_version=required_version,
                candidates=history_candidates,
                scanned_commits=scanned_commits,
            )
        if current_snapshots:
            selected = _select_recipe_snapshot_from_candidates(
                package=recipe_package,
                required_version=required_version,
                candidates=current_snapshots,
                scanned_commits=scanned_commits,
            )
            reason = (
                "current_ref_fallback"
                if selected.selection_reason in {"latest", "low_confidence_required_version"}
                and not history_error
                else selected.selection_reason
            )
            return replace(
                selected,
                selection_reason=reason,
                error=history_error,
                scanned_commits=scanned_commits or selected.scanned_commits,
            )
    return first_missing or BiocondaRecipeSnapshot(
        package=package,
        recipe_path=_recipe_relpath(package, feedstock=feedstock),
        meta_text="",
        selection_reason="recipe_not_found",
        error="recipe_not_found",
    )


def _select_bioconda_recipe_snapshot(
    *,
    recipes_repo: Path,
    ref: str,
    package: str,
    required_version: str,
    feedstock: bool = False,
    recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None = None,
    recipe_selection_cache_lock: threading.Lock | None = None,
) -> BiocondaRecipeSnapshot:
    cache_key = _recipe_selection_cache_key(recipes_repo, ref, package, required_version)
    cached = _recipe_selection_cache_get(
        recipe_selection_cache, recipe_selection_cache_lock, cache_key
    )
    if cached is not None:
        return cached
    lock_name = _safe_slug(f"{ref}-{package}-{required_version or 'latest'}")
    with _cache_lock(recipes_repo / ".gtsm-recipe-history-locks" / lock_name):
        cached = _recipe_selection_cache_get(
            recipe_selection_cache, recipe_selection_cache_lock, cache_key
        )
        if cached is not None:
            return cached
        snapshot = _select_bioconda_recipe_snapshot_uncached(
            recipes_repo=recipes_repo,
            ref=ref,
            package=package,
            required_version=required_version,
            feedstock=feedstock,
        )
        return _recipe_selection_cache_put(
            recipe_selection_cache,
            recipe_selection_cache_lock,
            cache_key,
            snapshot,
        )


def _source_checkout_version_hint(required_version: str, recipe_version: str) -> str:
    required = _normalize_recipe_version_text(required_version)
    recipe = _normalize_recipe_version_text(recipe_version)
    if required_version and recipe_version and required != recipe:
        return f"required-{required_version}--recipe-{recipe_version}"
    return required_version or recipe_version or "latest"


def _emit_bioconda_recipe_selected(
    *,
    settings: ExtractionSettings,
    snapshot: BiocondaRecipeSnapshot,
    required_version: str,
    channel: str = "bioconda",
) -> None:
    _emit_extract_status_once(
        settings,
        (
            f"{channel}-recipe-selected",
            snapshot.package.lower(),
            required_version,
            snapshot.recipe_version,
            snapshot.commit,
            snapshot.selection_reason,
            snapshot.error,
        ),
        {
            "status": f"{channel}-recipe-selected",
            "source_channel": channel,
            "package": snapshot.package,
            "required_version": required_version,
            "recipe_version": snapshot.recipe_version,
            "recipe_path": snapshot.recipe_path,
            "recipe_commit": snapshot.commit,
            "recipe_commit_date": snapshot.commit_date,
            "selection_reason": snapshot.selection_reason,
            "scanned_commits": snapshot.scanned_commits,
            "returncode": 0 if snapshot.meta_text else 1,
            "error_text": snapshot.error,
        },
    )


def _http_browser_fallback_info_from_response(response: object) -> dict[str, str]:
    if not bool(getattr(response, "gtsm_user_agent_fallback", False)):
        return {}
    return {
        "http_user_agent_fallback": "browser",
        "http_user_agent": str(getattr(response, "gtsm_user_agent", "") or ""),
    }


def _write_http_browser_fallback_marker(checkout_dir: Path, info: dict[str, str]) -> None:
    marker = checkout_dir / _HTTP_BROWSER_FALLBACK_MARKER
    if info:
        marker.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()


def _source_http_browser_fallback_info(source_checkout: str) -> dict[str, str]:
    if not source_checkout:
        return {}
    checkout_path = Path(source_checkout)
    root = checkout_path.parent if checkout_path.is_file() else checkout_path
    marker = root / _HTTP_BROWSER_FALLBACK_MARKER
    if not marker.exists():
        return {}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if key in {"http_user_agent_fallback", "http_user_agent"} and value
    }


def _download_size_error(source_url: str, max_bytes: int) -> RuntimeError:
    return RuntimeError(
        f"source download exceeds configured maximum of {max_bytes} bytes: {source_url}"
    )


def _download_and_extract_archive(
    source_url: str,
    checkout_dir: Path,
    *,
    timeout_seconds: int = 60,
    max_bytes: int = 0,
) -> str:
    parsed = urlparse(source_url)
    archive_name = os.path.basename(parsed.path) or "source.tar.gz"
    archive_path = checkout_dir / archive_name
    tmp_path = checkout_dir / f"{archive_name}.tmp"
    complete_marker = checkout_dir / ".gtsm-source-complete"
    checkout_dir.mkdir(parents=True, exist_ok=True)
    if complete_marker.exists() and archive_path.exists():
        return str(archive_path)
    ignored_names = {archive_name, complete_marker.name, tmp_path.name}
    if archive_path.exists() and any(
        path.name not in ignored_names for path in checkout_dir.iterdir()
    ):
        complete_marker.write_text("ok\n", encoding="utf-8")
        return str(archive_path)

    if tmp_path.exists():
        tmp_path.unlink()
    timeout_seconds = max(1, int(timeout_seconds or 60))
    max_bytes = max(0, int(max_bytes or 0))
    http_browser_fallback_info: dict[str, str] = {}
    downloaded_bytes = 0
    try:
        if parsed.scheme.lower() == "ftp":
            with tmp_path.open("wb") as handle, urlopen_with_user_agent_fallback(
                source_url, timeout=timeout_seconds
            ) as response:
                http_browser_fallback_info = _http_browser_fallback_info_from_response(response)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded_bytes += len(chunk)
                    if max_bytes and downloaded_bytes > max_bytes:
                        raise _download_size_error(source_url, max_bytes)
                    handle.write(chunk)
        else:
            try:
                response_context = _requests_get_with_user_agent_fallback(
                    source_url,
                    timeout=timeout_seconds,
                    stream=True,
                )
            except requests.exceptions.SSLError:
                # Some legacy bioinformatics source hosts redirect HTTP archives to
                # HTTPS endpoints with stale certificates. Retry only for source
                # collection; the extracted URL remains recorded in diagnostics.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    response_context = _requests_get_with_user_agent_fallback(
                        source_url,
                        timeout=timeout_seconds,
                        stream=True,
                        verify=False,
                    )
            with response_context as response:
                http_browser_fallback_info = _http_browser_fallback_info_from_response(response)
                response.raise_for_status()
                response_headers = getattr(response, "headers", {}) or {}
                content_length = int(response_headers.get("content-length") or 0)
                if max_bytes and content_length > max_bytes:
                    raise _download_size_error(source_url, max_bytes)
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded_bytes += len(chunk)
                        if max_bytes and downloaded_bytes > max_bytes:
                            raise _download_size_error(source_url, max_bytes)
                        handle.write(chunk)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    tmp_path.replace(archive_path)
    _write_http_browser_fallback_marker(checkout_dir, http_browser_fallback_info)
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=checkout_dir)
    except tarfile.TarError:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(path=checkout_dir)
        except zipfile.BadZipFile:
            pass
    complete_marker.write_text("ok\n", encoding="utf-8")
    return str(archive_path)


def _source_result_cache_key(
    package: str,
    version_hint: str,
    source_url: str,
    source_ref: str,
) -> tuple[str, str, str, str]:
    return (package, version_hint, source_url, source_ref)


_BIOCONDA_SOURCE_RECIPE_ALIASES = {
    "metawrap-binning": "metawrap-mg",
    "metawrap-refinement": "metawrap-mg",
}


def _source_recipe_package_alias(package: str) -> str:
    return _BIOCONDA_SOURCE_RECIPE_ALIASES.get(package.strip().lower(), "")


def _source_result_cache_get(
    source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None,
    source_result_cache_lock: threading.Lock | None,
    cache_key: tuple[str, str, str, str],
) -> tuple[str, str] | None:
    if source_result_cache is None:
        return None
    if source_result_cache_lock is None:
        return source_result_cache.get(cache_key)
    with source_result_cache_lock:
        return source_result_cache.get(cache_key)


def _source_result_cache_put(
    source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None,
    source_result_cache_lock: threading.Lock | None,
    cache_key: tuple[str, str, str, str],
    result: tuple[str, str],
) -> tuple[str, str]:
    if source_result_cache is None:
        return result
    if source_result_cache_lock is None:
        source_result_cache.setdefault(cache_key, result)
        return source_result_cache[cache_key]
    with source_result_cache_lock:
        source_result_cache.setdefault(cache_key, result)
        return source_result_cache[cache_key]


def _emit_bioconda_source_cache_hit(
    *,
    package: str,
    source_url: str,
    source_ref: str,
    source_checkout: str,
    source_error: str,
    settings: ExtractionSettings,
    channel: str = "bioconda",
) -> None:
    _emit_extract_status_once(
        settings,
        (
            f"{channel}-source-cache-hit",
            package.lower(),
            source_url,
            source_ref,
            source_checkout,
            source_error,
        ),
        {
            "status": f"{channel}-source-cache-hit",
            "source_channel": channel,
            "package": package,
            "source_url": source_url,
            "source_ref": source_ref,
            "source_checkout": source_checkout,
            "returncode": 0 if not source_error else 1,
            "error_text": source_error,
        },
    )


def _checkout_bioconda_source(
    *,
    package: str,
    source_url: str,
    source_ref: str,
    checkout_dir: Path,
    settings: ExtractionSettings,
    channel: str = "bioconda",
    source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None = None,
    source_result_cache_lock: threading.Lock | None = None,
    source_result_cache_key: tuple[str, str, str, str] | None = None,
) -> tuple[str, str]:
    with _cache_lock(checkout_dir):
        if source_result_cache_key is not None:
            cached_result = _source_result_cache_get(
                source_result_cache,
                source_result_cache_lock,
                source_result_cache_key,
            )
            if cached_result is not None:
                _emit_bioconda_source_cache_hit(
                    package=package,
                    source_url=source_url,
                    source_ref=source_ref,
                    source_checkout=cached_result[0],
                    source_error=cached_result[1],
                    settings=settings,
                    channel=channel,
                )
                return cached_result

        def remember(source_checkout: str, source_error: str) -> tuple[str, str]:
            if source_result_cache_key is None:
                return source_checkout, source_error
            return _source_result_cache_put(
                source_result_cache,
                source_result_cache_lock,
                source_result_cache_key,
                (source_checkout, source_error),
            )

        cached = checkout_dir.exists() and (
            (checkout_dir / ".gtsm-source-complete").exists()
            or (checkout_dir / ".git").exists()
            or any(checkout_dir.iterdir())
        )
        _emit_extract_status(
            settings,
            {
                "status": f"{channel}-source-prepare",
                "source_channel": channel,
                "package": package,
                "source_url": source_url,
                "source_ref": source_ref,
                "checkout_dir": str(checkout_dir),
                "source": "cache" if cached else "download",
            },
        )
        if source_url.endswith(".git"):
            source_error = ""
            if checkout_dir.exists() and (checkout_dir / ".git").exists():
                fetch_command = ["git", "-C", str(checkout_dir), "fetch", "--all", "--tags"]
                fetch_result = _run_command(fetch_command, timeout_seconds=180)
                _emit_command_status(
                    settings,
                    status=f"{channel}-source-command",
                    step="fetch",
                    command=fetch_command,
                    result=fetch_result,
                    package=package,
                    checkout_dir=str(checkout_dir),
                )
                if fetch_result.returncode != 0:
                    source_error = _completed_error_text(fetch_result)
            else:
                checkout_dir.parent.mkdir(parents=True, exist_ok=True)
                clone_command = ["git", "clone"]
                if not source_ref:
                    clone_command.extend(["--depth", "1"])
                clone_command.extend([source_url, str(checkout_dir)])
                clone_result = _run_command(clone_command, timeout_seconds=300)
                _emit_command_status(
                    settings,
                    status=f"{channel}-source-command",
                    step="clone",
                    command=clone_command,
                    result=clone_result,
                    package=package,
                    checkout_dir=str(checkout_dir),
                )
                if clone_result.returncode != 0:
                    source_error = _completed_error_text(clone_result)
                    _emit_extract_status(
                        settings,
                        {
                            "status": f"{channel}-source-failed",
                            "source_channel": channel,
                            "package": package,
                            "source_url": source_url,
                            "checkout_dir": str(checkout_dir),
                            "returncode": clone_result.returncode,
                            "error_text": source_error,
                        },
                    )
                    return remember("", source_error)
            if source_ref:
                checkout_command = ["git", "-C", str(checkout_dir), "checkout", source_ref]
                checkout_result = _run_command(checkout_command, timeout_seconds=120)
                _emit_command_status(
                    settings,
                    status=f"{channel}-source-command",
                    step="checkout",
                    command=checkout_command,
                    result=checkout_result,
                    package=package,
                    checkout_dir=str(checkout_dir),
                )
                if checkout_result.returncode != 0:
                    source_error = _completed_error_text(checkout_result)
                    _emit_extract_status(
                        settings,
                        {
                            "status": f"{channel}-source-failed",
                            "source_channel": channel,
                            "package": package,
                            "source_url": source_url,
                            "source_ref": source_ref,
                            "checkout_dir": str(checkout_dir),
                            "returncode": checkout_result.returncode,
                            "error_text": source_error,
                        },
                    )
                    return remember("", source_error)
            (checkout_dir / ".gtsm-source-complete").write_text("ok\n", encoding="utf-8")
            ready_payload = {
                "status": f"{channel}-source-ready",
                "source_channel": channel,
                "package": package,
                "source_url": source_url,
                "source_checkout": str(checkout_dir),
                "returncode": 0 if not source_error else 1,
                "error_text": source_error,
            }
            ready_payload.update(_source_http_browser_fallback_info(str(checkout_dir)))
            _emit_extract_status(settings, ready_payload)
            return remember(str(checkout_dir), source_error)

        try:
            source_checkout = _download_and_extract_archive(
                source_url,
                checkout_dir,
                timeout_seconds=settings.source_download_timeout_seconds,
                max_bytes=settings.source_download_max_bytes,
            )
        except Exception as error:
            source_error = str(error)
            _emit_extract_status(
                settings,
                {
                    "status": f"{channel}-source-failed",
                    "source_channel": channel,
                    "package": package,
                    "source_url": source_url,
                    "checkout_dir": str(checkout_dir),
                    "returncode": 1,
                    "error_text": source_error,
                },
            )
            return remember("", source_error)

        ready_payload = {
            "status": f"{channel}-source-ready",
            "source_channel": channel,
            "package": package,
            "source_url": source_url,
            "source_checkout": source_checkout,
            "returncode": 0,
            "error_text": "",
        }
        ready_payload.update(_source_http_browser_fallback_info(source_checkout))
        _emit_extract_status(settings, ready_payload)
        return remember(source_checkout, "")


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _source_checkout_is_binary_artifact(source_checkout: str) -> bool:
    path = Path(str(source_checkout or ""))
    name = path.name.lower()
    if name.endswith((".jar", ".whl", ".gem", ".exe", ".bin")):
        return True
    if name.endswith(
        (
            ".tar",
            ".tar.gz",
            ".tgz",
            ".tar.bz2",
            ".tbz2",
            ".tar.xz",
            ".txz",
            ".zip",
            ".gz",
            ".bz2",
            ".xz",
        )
    ):
        return False
    if path.exists() and path.is_file():
        try:
            sample = path.read_bytes()[:1024]
        except OSError:
            return False
        if b"\0" in sample:
            return True
        if sample:
            non_text = sum(
                1
                for byte in sample
                if byte < 9 or (13 < byte < 32) or byte > 126
            )
            return non_text / len(sample) > 0.30
    return False


def _source_version_match_status(required_version: str, recipe_version: str) -> str:
    required = _normalize_recipe_version_text(required_version)
    recipe = _normalize_recipe_version_text(recipe_version)
    if not required:
        return "not_required"
    if not recipe:
        return "unknown"
    if required == recipe:
        return "exact"
    required_parsed = _parse_semverish_version(required_version)
    recipe_parsed = _parse_semverish_version(recipe_version)
    if (
        required_parsed is not None
        and recipe_parsed is not None
        and _numeric_versions_equivalent(required_parsed, recipe_parsed)
    ):
        return "numeric_equivalent"
    return "mismatch"


def _source_confidence_from_recipe_selection(
    selection_reason: str,
    *,
    required_version: str,
    recipe_version: str,
) -> str:
    version_match = _source_version_match_status(required_version, recipe_version)
    if version_match in {"exact", "numeric_equivalent", "not_required"}:
        return "exact"
    if selection_reason in {"same_numeric_variant", "same_major_newer", "same_major_older"}:
        return "near"
    return "weak"


def _github_source_fallback_candidates(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    if parsed.netloc.lower() != "github.com":
        return []
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1]
    candidates: list[str] = []
    if len(parts) >= 5 and parts[2] == "releases" and parts[3] == "download":
        tag = parts[4]
        candidates.append(f"https://github.com/{owner}/{repo}/archive/refs/tags/{tag}.tar.gz")
        candidates.append(f"https://github.com/{owner}/{repo}/archive/{tag}.tar.gz")
        if tag and not tag.lower().startswith("v"):
            candidates.append(
                f"https://github.com/{owner}/{repo}/archive/refs/tags/v{tag}.tar.gz"
            )
            candidates.append(f"https://github.com/{owner}/{repo}/archive/v{tag}.tar.gz")
    if len(parts) >= 4 and parts[2] in {"raw", "blob"}:
        ref = parts[3]
        candidates.append(f"https://github.com/{owner}/{repo}/archive/refs/heads/{ref}.tar.gz")
    return _dedupe_strings(candidates)


def _pypi_sdist_fallback_candidates(source_url: str) -> list[str]:
    filename = Path(urlparse(source_url).path).name
    if not filename.endswith(".whl"):
        return []
    wheel_parts = filename[: -len(".whl")].split("-")
    if len(wheel_parts) < 5:
        return []
    name = ""
    version = ""
    for index, part in enumerate(wheel_parts[:-3]):
        if index == 0:
            continue
        if re.match(r"^[0-9][A-Za-z0-9_.!+]*$", part):
            name = "-".join(wheel_parts[:index]).replace("_", "-")
            version = part
            break
    if not name or not version:
        return []
    first = name[0].lower()
    return [f"https://pypi.org/packages/source/{first}/{name}/{name}-{version}.tar.gz"]


def _bioconductor_archive_fallback_candidates(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    if "bioconductor.org" not in parsed.netloc.lower():
        return []
    filename = Path(parsed.path).name
    match = re.match(
        r"(?P<name>[A-Za-z][A-Za-z0-9_.]*)_(?P<version>[0-9][A-Za-z0-9_.-]*)\.tar\.gz$",
        filename,
    )
    if not match:
        return []
    name = match.group("name")
    version = match.group("version")
    parts = [part for part in parsed.path.split("/") if part]
    package_roots: list[str] = []
    if "src" in parts and "contrib" in parts:
        src_index = parts.index("src")
        package_root = "/".join(parts[:src_index])
        if package_root:
            package_roots.append(package_root)
    if len(parts) >= 3 and parts[0] == "packages":
        release = parts[1]
        package_area = parts[2]
        package_roots.append(f"packages/{release}/{package_area}")
        package_roots.append(f"packages/release/{package_area}")
    return _dedupe_strings(
        [
            f"https://bioconductor.org/{root}/src/contrib/Archive/{name}/{name}_{version}.tar.gz"
            for root in package_roots
        ]
    )


def _cran_archive_fallback_candidates(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[-3:-1] != ["src", "contrib"]:
        return []
    filename = parts[-1]
    match = re.match(
        r"(?P<name>[A-Za-z][A-Za-z0-9_.]*)_(?P<version>[0-9][A-Za-z0-9_.-]*)\.tar\.gz$",
        filename,
    )
    if not match:
        return []
    name = match.group("name")
    if "Archive" in parts:
        return []
    candidates = [
        f"https://cran.r-project.org/src/contrib/Archive/{name}/{filename}",
    ]
    if parsed.netloc and parsed.netloc.lower() != "cran.r-project.org":
        candidates.append(
            f"{parsed.scheme or 'https'}://{parsed.netloc}/src/contrib/Archive/{name}/{filename}"
        )
    return _dedupe_strings(candidates)


def _gnu_mirror_fallback_candidates(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    netloc = parsed.netloc.lower()
    parts = [part for part in parsed.path.split("/") if part]
    package = ""
    rest = ""
    if netloc == "ftpmirror.gnu.org" and len(parts) >= 2:
        package = parts[0]
        rest = "/".join(parts[1:])
    elif netloc in {"ftp.gnu.org", "www.gnu.org"} and len(parts) >= 3 and parts[0] == "gnu":
        package = parts[1]
        rest = "/".join(parts[2:])
    if not package or not rest:
        return []
    return _dedupe_strings(
        [
            f"https://ftp.gnu.org/gnu/{package}/{rest}",
            f"https://mirrors.kernel.org/gnu/{package}/{rest}",
            f"https://ftpmirror.gnu.org/{package}/{rest}",
        ]
    )


def _sourceforge_fallback_candidates(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    netloc = parsed.netloc.lower()
    parts = [part for part in parsed.path.split("/") if part]
    rest: list[str] = []
    if netloc == "sourceforge.net" and len(parts) >= 3 and parts[0] == "projects":
        project = parts[1]
        rest = [project, *parts[3:]] if parts[2] == "files" else [project, *parts[2:]]
    elif (
        netloc == "downloads.sourceforge.net" or netloc.endswith(".dl.sourceforge.net")
    ) and len(parts) >= 2:
        rest = parts[1:] if parts[0] == "project" else parts
    if rest and rest[-1] == "download":
        rest = rest[:-1]
    if not rest:
        return []
    return [f"https://downloads.sourceforge.net/project/{'/'.join(rest)}"]


def _url_encoded_source_fallback_candidates(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    if not parsed.path or parsed.path == quote(parsed.path, safe="/%:@!$&'()*+,;="):
        return []
    encoded_path = quote(parsed.path, safe="/%:@!$&'()*+,;=")
    return [parsed._replace(path=encoded_path).geturl()]


def _domain_mirror_fallback_candidates(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    netloc = parsed.netloc.lower()
    candidates: list[str] = []
    if netloc == "software-ab.informatik.uni-tuebingen.de":
        candidates.append(parsed._replace(netloc="software-ab.cs.uni-tuebingen.de").geturl())
    return candidates


def _binary_artifact_source_fallback_candidates(source_url: str) -> list[str]:
    return _dedupe_strings(
        [
            *_github_source_fallback_candidates(source_url),
            *_pypi_sdist_fallback_candidates(source_url),
        ]
    )


def _failed_source_fallback_candidates(source_url: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for fallback_url in _binary_artifact_source_fallback_candidates(source_url):
        candidates.append((fallback_url, "binary_artifact_source_fallback"))
    for fallback_url in _bioconductor_archive_fallback_candidates(source_url):
        candidates.append((fallback_url, "bioconductor_archive_fallback"))
    for fallback_url in _cran_archive_fallback_candidates(source_url):
        candidates.append((fallback_url, "cran_archive_fallback"))
    for fallback_url in _gnu_mirror_fallback_candidates(source_url):
        candidates.append((fallback_url, "gnu_mirror_fallback"))
    for fallback_url in _sourceforge_fallback_candidates(source_url):
        candidates.append((fallback_url, "sourceforge_mirror_fallback"))
    for fallback_url in _url_encoded_source_fallback_candidates(source_url):
        candidates.append((fallback_url, "url_encoded_source_fallback"))
    for fallback_url in _domain_mirror_fallback_candidates(source_url):
        candidates.append((fallback_url, "domain_mirror_fallback"))
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fallback_url, reason in candidates:
        if fallback_url in seen:
            continue
        seen.add(fallback_url)
        deduped.append((fallback_url, reason))
    return deduped


def _normalized_command_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _command_hint_from_name(value: str) -> str:
    name = Path(value.strip()).name
    if not name:
        return ""
    if name.endswith((".py", ".pl", ".R", ".r", ".sh")):
        name = Path(name).stem
    if _is_executable_candidate(name):
        return name
    return ""


def _extract_recipe_command_hints(meta_text: str) -> list[str]:
    hints: set[str] = set()
    for line in meta_text.splitlines():
        stripped = line.strip().strip("'\"")
        if not stripped or stripped.startswith("#"):
            continue
        entry_match = re.match(r"-\s*([A-Za-z][A-Za-z0-9_.+-]*)\s*=", stripped)
        if entry_match:
            hint = _command_hint_from_name(entry_match.group(1))
            if hint:
                hints.add(hint)
            continue
        script_match = re.match(r"(?:script|entry_points):\s*(.+)$", stripped)
        if script_match:
            for token in re.split(r"[\s,]+", script_match.group(1)):
                hint = _command_hint_from_name(token)
                if hint:
                    hints.add(hint)
    return sorted(hints)


def _source_hint_allowed(hint: str) -> bool:
    cleaned = _clean_command_token(hint)
    key = _normalized_command_key(cleaned)
    return bool(cleaned and key and key not in _SOURCE_HINT_DENYLIST_KEYS)


def _filter_source_command_hints(hints: list[str] | set[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        cleaned = _clean_command_token(str(hint))
        if not cleaned or not _source_hint_allowed(cleaned):
            continue
        if not _is_executable_candidate(cleaned):
            continue
        key = _normalized_command_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return sorted(deduped)


def _source_hint_matches_package(hint: str, package_key: str) -> bool:
    hint_key = _normalized_command_key(hint)
    if not hint_key or not package_key:
        return False
    return hint_key == package_key or package_key in hint_key or hint_key in package_key


def _source_command_hints_from_name(value: str) -> list[str]:
    name = Path(value.strip()).name
    lowered = name.lower()
    if not name or lowered in _SOURCE_METADATA_FILENAMES:
        return []
    if lowered.startswith(_SOURCE_NON_COMMAND_FILENAME_PREFIXES):
        return []
    hints: list[str] = []
    if _is_executable_script_token(name):
        hints.append(name)
        stem = Path(name).stem
        if _is_executable_candidate(stem):
            hints.append(stem)
    else:
        hint = _command_hint_from_name(name)
        if hint:
            hints.append(hint)
    return _filter_source_command_hints(hints)


def _source_scan_roots(root: Path) -> list[Path]:
    roots = [root]
    try:
        children = sorted(root.iterdir())[:50]
    except OSError:
        return roots
    for child in children:
        if (
            child.is_dir()
            and not child.name.startswith(".")
            and child.name.lower() not in _SOURCE_HINT_IGNORED_DIRS
        ):
            roots.append(child)
    return roots


def _extract_directory_source_command_hints(root: Path, package: str) -> list[str]:
    hints: set[str] = set()
    package_key = _normalized_command_key(package)
    for scan_root in _source_scan_roots(root):
        candidate_dirs = [scan_root / name for name in sorted(_SOURCE_HINT_EXECUTABLE_DIRS)]
        for directory in candidate_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            for path in sorted(directory.iterdir())[:100]:
                if path.is_file():
                    hints.update(_source_command_hints_from_name(path.name))

        for metadata_name in ("setup.py", "setup.cfg", "pyproject.toml"):
            path = scan_root / metadata_name
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hints.update(_extract_recipe_command_hints(text))

        try:
            files = sorted(scan_root.iterdir())[:200]
        except OSError:
            files = []
        for path in files:
            if not path.is_file():
                continue
            executable_file = os.access(path, os.X_OK)
            for hint in _source_command_hints_from_name(path.name):
                if _source_hint_matches_package(hint, package_key) or (
                    executable_file and not Path(hint).suffix
                ):
                    hints.add(hint)
    return _filter_source_command_hints(hints)


def _archive_member_is_executable_hint_path(parts: list[str]) -> bool:
    parents = {part.lower() for part in parts[:-1]}
    return bool(parents & _SOURCE_HINT_EXECUTABLE_DIRS)


def _archive_member_is_top_level_source_path(parts: list[str]) -> bool:
    # Archives commonly contain either files at root or a single root source directory.
    return 1 <= len(parts) <= 2


def _archive_member_is_package_module_path(parts: list[str], package_key: str) -> bool:
    if len(parts) < 2 or not package_key:
        return False
    parent_keys = {_normalized_command_key(part) for part in parts[:-1]}
    return any(_command_key_matches(parent_key, package_key) for parent_key in parent_keys)


def _extract_archive_source_command_hints(source_archive: Path, package: str) -> list[str]:
    hints: set[str] = set()
    package_key = _normalized_command_key(package)

    def add_name_hints(member_name: str, *, executable_file: bool = False) -> None:
        parts = [part for part in member_name.split("/") if part]
        if not parts:
            return
        if any(part.lower() in _SOURCE_HINT_IGNORED_DIRS for part in parts[:-1]):
            return
        name_hints = _source_command_hints_from_name(parts[-1])
        if not name_hints:
            return
        if _archive_member_is_executable_hint_path(parts):
            hints.update(name_hints)
            return
        if _archive_member_is_top_level_source_path(parts):
            hints.update(
                hint
                for hint in name_hints
                if _source_hint_matches_package(hint, package_key)
                or (executable_file and not Path(hint).suffix)
            )
            return
        if executable_file and _archive_member_is_package_module_path(parts, package_key):
            hints.update(name_hints)

    def add_metadata_hints(member_name: str, text: str) -> None:
        parts = [part for part in member_name.split("/") if part]
        if not parts or parts[-1].lower() not in _SOURCE_METADATA_FILENAMES:
            return
        if len(parts) > 3:
            return
        hints.update(_extract_recipe_command_hints(text[:_SOURCE_HINT_MAX_METADATA_BYTES]))

    try:
        if tarfile.is_tarfile(source_archive):
            with tarfile.open(source_archive, "r:*") as archive:
                members = archive.getmembers()[:_SOURCE_HINT_MAX_ARCHIVE_MEMBERS]
                for member in members:
                    if not member.isfile():
                        continue
                    add_name_hints(member.name, executable_file=bool(member.mode & 0o111))
                    if Path(member.name).name.lower() in _SOURCE_METADATA_FILENAMES:
                        handle = archive.extractfile(member)
                        if handle is None:
                            continue
                        text = handle.read(_SOURCE_HINT_MAX_METADATA_BYTES).decode(
                            "utf-8", errors="ignore"
                        )
                        add_metadata_hints(member.name, text)
            return _filter_source_command_hints(hints)
        if zipfile.is_zipfile(source_archive):
            with zipfile.ZipFile(source_archive) as archive:
                infos = archive.infolist()[:_SOURCE_HINT_MAX_ARCHIVE_MEMBERS]
                for info in infos:
                    name = info.filename
                    if name.endswith("/"):
                        continue
                    mode = info.external_attr >> 16
                    add_name_hints(name, executable_file=bool(mode & 0o111))
                    if Path(name).name.lower() in _SOURCE_METADATA_FILENAMES:
                        text = archive.read(name)[:_SOURCE_HINT_MAX_METADATA_BYTES].decode(
                            "utf-8", errors="ignore"
                        )
                        add_metadata_hints(name, text)
            return _filter_source_command_hints(hints)
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return []
    return []


def _extract_source_command_hints(source_checkout: str, package: str) -> list[str]:
    if not source_checkout:
        return []
    root = Path(source_checkout)
    archive_hints: list[str] = []
    if root.is_file():
        archive_hints = _extract_archive_source_command_hints(root, package)
        root = root.parent
    if not root.exists():
        return _filter_source_command_hints(set(archive_hints))

    hints = set(archive_hints)
    hints.update(_extract_directory_source_command_hints(root, package))
    return _filter_source_command_hints(hints)


def _cached_source_command_hints(source_checkout: str, package: str) -> list[str]:
    if not source_checkout:
        return []
    key = (str(source_checkout), str(package).lower())
    with _SOURCE_COMMAND_HINT_CACHE_LOCK:
        cached = _SOURCE_COMMAND_HINT_CACHE.get(key)
    if cached is not None:
        return cached
    hints = _extract_source_command_hints(source_checkout, package)
    with _SOURCE_COMMAND_HINT_CACHE_LOCK:
        _SOURCE_COMMAND_HINT_CACHE.setdefault(key, hints)
        return _SOURCE_COMMAND_HINT_CACHE[key]


def _source_doc_roots(source_checkout: str) -> list[Path]:
    root = Path(str(source_checkout or ""))
    if not root.exists():
        return []
    if root.is_file():
        root = root.parent
    roots = [root]
    try:
        children = sorted(root.iterdir())[:100]
    except OSError:
        return roots
    for child in children:
        if (
            child.is_dir()
            and not child.name.startswith(".")
            and child.name.lower() not in _SOURCE_HINT_IGNORED_DIRS
        ):
            roots.append(child)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in roots:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _source_doc_candidate_score(path: Path, terms: list[str]) -> tuple[int, int]:
    lowered = str(path).lower()
    score = 0
    if path.name.lower().startswith("readme"):
        score += 2
    for term in terms:
        if term.lower() in lowered:
            score += 5
    return score, -len(path.parts)


def _is_source_doc_candidate_file(path: Path) -> bool:
    lowered = path.name.lower()
    if lowered in _SOURCE_DOC_EXCLUDED_FILENAMES:
        return False
    suffix = path.suffix.lower()
    if not (
        suffix in _SOURCE_DOC_EXTENSIONS
        or lowered.startswith(_SOURCE_DOC_FILENAME_PREFIXES)
    ):
        return False
    if lowered.startswith(_SOURCE_DOC_FILENAME_PREFIXES):
        return True
    parent_names = {part.lower() for part in path.parts[:-1]}
    return bool(parent_names & _SOURCE_DOC_DIR_NAMES)


def _source_doc_candidate_files(source_checkout: str, terms: list[str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in _source_doc_roots(source_checkout):
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            dirs[:] = [
                name
                for name in sorted(dirs)
                if not name.startswith(".") and name.lower() not in _SOURCE_HINT_IGNORED_DIRS
            ][:25]
            for name in sorted(files):
                path = current_path / name
                if not _is_source_doc_candidate_file(path):
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(path)
                if len(candidates) >= _SOURCE_DOC_SCAN_FILE_LIMIT:
                    return sorted(
                        candidates,
                        key=lambda item: _source_doc_candidate_score(item, terms),
                        reverse=True,
                    )[:_SOURCE_DOC_MAX_FILES]
    return sorted(
        candidates,
        key=lambda item: _source_doc_candidate_score(item, terms),
        reverse=True,
    )[:_SOURCE_DOC_MAX_FILES]


def _source_doc_query_terms(package: str, command_hints: list[str] | set[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in [package]:
        cleaned = str(value or "").strip()
        key = _normalized_command_key(cleaned)
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            terms.append(cleaned)
    for value in command_hints:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        name = Path(cleaned.split()[0]).name
        for token in (cleaned, name, Path(name).stem):
            token = token.strip()
            key = _normalized_command_key(token)
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            terms.append(token)
    return terms


def _source_doc_command_terms(command_hints: list[str] | set[str]) -> list[str]:
    return _source_doc_query_terms("", command_hints)


def _source_doc_matches(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    for term in terms:
        if term.lower() in lowered:
            matches.append(term)
    return matches[:12]


def _source_usage_snippets_from_text(
    text: str, *, terms: list[str], relative_path: str
) -> list[dict[str, object]]:
    if not text.strip():
        return []
    lines = text.splitlines()
    snippets: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        stripped_line = line.strip()
        lowered = stripped_line.lower()
        is_usage = (
            "usage:" in lowered
            or lowered.strip().startswith("usage ")
            or stripped_line.startswith("\\usage")
        )
        line_matches = _source_doc_matches(line, terms)
        is_command_heading = stripped_line.startswith("#") and bool(line_matches)
        if not is_usage and not is_command_heading:
            continue
        start = max(0, index - 3)
        end = min(len(lines), index + _SOURCE_DOC_CONTEXT_LINES + 1)
        block = "\n".join(lines[start:end]).strip()
        if not block:
            continue
        block_matches = _source_doc_matches(block, terms)
        if not block_matches:
            continue
        normalized = re.sub(r"\s+", " ", block).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        snippets.append(
            {
                "path": relative_path,
                "line": index + 1,
                "command_matches": block_matches,
                "text": block[:1200],
            }
        )
        if len(snippets) >= _SOURCE_DOC_MAX_DOCS:
            break
    return snippets


def _extract_source_command_docs(
    source_checkout: str,
    package: str,
    command_hints: list[str] | set[str],
    preferred_command_hints: list[str] | set[str] | None = None,
) -> list[dict[str, object]]:
    preferred_terms = _source_doc_command_terms(preferred_command_hints or [])
    terms = _source_doc_query_terms(package, command_hints)
    if not source_checkout or not terms:
        return []
    checkout = Path(source_checkout)
    root = checkout.parent if checkout.is_file() else checkout

    def extract_with_terms(query_terms: list[str]) -> list[dict[str, object]]:
        docs: list[dict[str, object]] = []
        for path in _source_doc_candidate_files(source_checkout, query_terms):
            try:
                if path.stat().st_size > _SOURCE_DOC_MAX_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            try:
                relative_path = str(path.relative_to(root))
            except ValueError:
                relative_path = path.name
            docs.extend(
                _source_usage_snippets_from_text(
                    text,
                    terms=query_terms,
                    relative_path=relative_path,
                )
            )
            if len(docs) >= _SOURCE_DOC_MAX_DOCS:
                break
        return docs[:_SOURCE_DOC_MAX_DOCS]

    if preferred_terms:
        preferred_docs = extract_with_terms(preferred_terms)
        if preferred_docs:
            return preferred_docs

    return extract_with_terms(terms)


def _record_source_command_hints(
    *,
    primary_command: str,
    subcommands: list[str],
    command_text: str,
    wrapper_helper_files: list[dict],
    wrapper_configfiles: list[dict],
) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        cleaned = str(value or "").strip()
        if not cleaned:
            return
        first = cleaned.split()[0]
        name = Path(first).name
        for token in (cleaned, first, name, Path(name).stem):
            token = token.strip()
            key = _normalized_command_key(token)
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            hints.append(token)

    add(primary_command)
    for subcommand in subcommands:
        add(subcommand)
        if primary_command:
            add(f"{primary_command} {subcommand}")
    for primary, subcommand in _command_candidate_signatures(command_text):
        add(primary)
        if subcommand:
            add(subcommand)
            add(f"{primary} {subcommand}")
    for source in [*wrapper_helper_files, *wrapper_configfiles]:
        for value in (
            source.get("relative_path", ""),
            source.get("path", ""),
            source.get("filename", ""),
            source.get("name", ""),
        ):
            add(value)
        for signature in source.get("command_signatures", []) or []:
            if not isinstance(signature, dict):
                continue
            primary = str(signature.get("primary", "") or "")
            subcommand = str(signature.get("subcommand", "") or "")
            add(primary)
            if subcommand:
                add(subcommand)
                add(f"{primary} {subcommand}")
        for doc in source.get("command_docs", []) or []:
            if not isinstance(doc, dict):
                continue
            for command in doc.get("commands", []) or []:
                add(command)
    return hints

def _resolve_recipe_source_mapping(
    *,
    package: str,
    required_version: str,
    channel: str,
    settings: ExtractionSettings,
    recipes_repo: Path | None = None,
    recipes_repo_error: str = "",
    ref: str = "HEAD",
    feedstock: bool = False,
    recipe_package: str = "",
    source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None = None,
    source_result_cache_lock: threading.Lock | None = None,
    recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None = None,
    recipe_selection_cache_lock: threading.Lock | None = None,
    allow_source_provider: bool = True,
    source_provider_stack: tuple[str, ...] = (),
    preferred_command_hints: list[str] | set[str] | None = None,
) -> dict:
    cache_root = settings.cache_root or Path(".gtsm-cache") / "source-cache"
    if recipes_repo is None:
        return {
            "package": package,
            "required_version": required_version,
            "source_channel": channel,
            "error": f"{channel}_repo_unavailable",
            "source_error": recipes_repo_error,
            "command_hints": [],
        }

    recipe_package = recipe_package or package
    snapshot = _select_bioconda_recipe_snapshot(
        recipes_repo=recipes_repo,
        ref=ref,
        package=recipe_package,
        required_version=required_version,
        feedstock=feedstock,
        recipe_selection_cache=recipe_selection_cache,
        recipe_selection_cache_lock=recipe_selection_cache_lock,
    )
    _emit_bioconda_recipe_selected(
        settings=settings,
        snapshot=snapshot,
        required_version=required_version,
        channel=channel,
    )
    if not snapshot.meta_text:
        return {
            "package": package,
            "required_version": required_version,
            "source_channel": channel,
            "error": "recipe_not_found",
            "source_error": snapshot.error,
            "recipe_selection_reason": snapshot.selection_reason,
            "recipe_scanned_commits": snapshot.scanned_commits,
            "command_hints": [],
        }

    meta_text = snapshot.meta_text
    recipe_version = _extract_recipe_version(meta_text)
    source_url_templates, source_ref_template = _extract_source_url_candidates(meta_text)
    source_url_template = source_url_templates[0] if source_url_templates else ""
    source_url_candidates, source_ref, template_error = _render_bioconda_source_url_candidates(
        meta_text=meta_text,
        package=recipe_package,
        requirement_version=required_version,
        recipe_version=recipe_version,
        source_urls=source_url_templates,
        source_ref=source_ref_template,
    )
    source_url = source_url_candidates[0] if source_url_candidates else ""
    command_hints = set(_extract_recipe_command_hints(meta_text))
    source_checkout = ""
    source_error = ""
    source_attempts: list[dict[str, str]] = []
    source_artifact_url = ""
    source_artifact_checkout = ""
    source_fallback_reason = ""
    source_provider: dict[str, str] = {}
    if source_url_candidates:
        sources_root = cache_root / f"{channel}-sources"
        version_hint = _source_checkout_version_hint(required_version, recipe_version)

        def try_binary_artifact_source_fallbacks(
            candidate_url: str,
            *,
            reason: str = "binary_artifact_source_fallback",
            include_failed_source_fallbacks: bool = False,
        ) -> tuple[str, str, str, str]:
            fallback_error = ""
            if include_failed_source_fallbacks:
                fallback_candidates = _failed_source_fallback_candidates(candidate_url)
            else:
                fallback_candidates = [
                    (fallback_url, reason)
                    for fallback_url in _binary_artifact_source_fallback_candidates(candidate_url)
                ]
            for fallback_index, (fallback_url, fallback_reason) in enumerate(
                fallback_candidates, start=1
            ):
                fallback_checkout_dir = sources_root / (
                    f"{_safe_slug(package)}-{_safe_slug(version_hint)}--source-fallback-"
                    f"{fallback_index}"
                )
                fallback_cache_key = _source_result_cache_key(
                    f"{channel}:{package}:source-fallback",
                    version_hint,
                    fallback_url,
                    "",
                )
                fallback_checkout, fallback_error = _checkout_bioconda_source(
                    package=package,
                    source_url=fallback_url,
                    source_ref="",
                    checkout_dir=fallback_checkout_dir,
                    settings=settings,
                    channel=channel,
                    source_result_cache=source_result_cache,
                    source_result_cache_lock=source_result_cache_lock,
                    source_result_cache_key=fallback_cache_key,
                )
                fallback_attempt = {
                    "source_url": fallback_url,
                    "source_checkout": fallback_checkout,
                    "source_error": fallback_error,
                    "fallback_reason": fallback_reason,
                }
                fallback_attempt.update(_source_http_browser_fallback_info(fallback_checkout))
                source_attempts.append(fallback_attempt)
                if fallback_checkout:
                    return fallback_url, fallback_checkout, fallback_error, fallback_reason
            return "", "", fallback_error, ""

        for index, candidate_url in enumerate(source_url_candidates, start=1):
            checkout_dir = sources_root / f"{_safe_slug(package)}-{_safe_slug(version_hint)}"
            if index > 1:
                checkout_dir = checkout_dir.with_name(f"{checkout_dir.name}--source-{index}")
            cache_key = _source_result_cache_key(
                f"{channel}:{package}", version_hint, candidate_url, source_ref
            )
            candidate_checkout, candidate_error = _checkout_bioconda_source(
                package=package,
                source_url=candidate_url,
                source_ref=source_ref,
                checkout_dir=checkout_dir,
                settings=settings,
                channel=channel,
                source_result_cache=source_result_cache,
                source_result_cache_lock=source_result_cache_lock,
                source_result_cache_key=cache_key,
            )
            candidate_attempt = {
                "source_url": candidate_url,
                "source_checkout": candidate_checkout,
                "source_error": candidate_error,
            }
            candidate_attempt.update(_source_http_browser_fallback_info(candidate_checkout))
            source_attempts.append(candidate_attempt)
            if not candidate_checkout:
                source_error = candidate_error
                fallback_url, fallback_checkout, fallback_error, fallback_reason = (
                    try_binary_artifact_source_fallbacks(
                        candidate_url,
                        include_failed_source_fallbacks=True,
                    )
                )
                if fallback_checkout:
                    source_artifact_url = candidate_url
                    source_url = fallback_url
                    source_checkout = fallback_checkout
                    source_error = fallback_error
                    source_fallback_reason = fallback_reason or "source_url_fallback"
                    break
                if fallback_error:
                    source_error = fallback_error
                continue
            source_url = candidate_url
            source_checkout = candidate_checkout
            source_error = candidate_error
            if not _source_checkout_is_binary_artifact(source_checkout):
                break

            source_artifact_url = source_url
            source_artifact_checkout = source_checkout
            fallback_url, fallback_checkout, fallback_error, fallback_reason = (
                try_binary_artifact_source_fallbacks(source_url)
            )
            if fallback_checkout:
                source_url = fallback_url
                source_checkout = fallback_checkout
                source_error = fallback_error
                source_fallback_reason = fallback_reason or "binary_artifact_source_fallback"
            if source_fallback_reason:
                break
            source_error = fallback_error or "binary_artifact_no_source"
    elif source_url_templates and template_error:
        source_error = f"unresolved_template: {template_error}"
        _emit_extract_status(
            settings,
            {
                "status": f"{channel}-source-skipped-template",
                "source_channel": channel,
                "package": package,
                "source_url": source_url,
                "source_url_template": source_url_template,
                "source_ref": source_ref,
                "source_ref_template": source_ref_template,
                "returncode": 1,
                "error_text": source_error,
            },
        )
    elif allow_source_provider and not template_error:
        provider_stack_keys = {_normalized_command_key(item) for item in source_provider_stack}
        provider_stack_keys.add(_normalized_command_key(recipe_package or package))
        for provider_package in _extract_recipe_run_dependency_names(meta_text):
            provider_key = _normalized_command_key(provider_package)
            if not provider_key or provider_key in provider_stack_keys:
                continue
            provider_mapping = _resolve_recipe_source_mapping(
                package=provider_package,
                required_version="",
                channel=channel,
                settings=settings,
                recipes_repo=recipes_repo,
                recipes_repo_error=recipes_repo_error,
                ref=ref,
                feedstock=feedstock,
                recipe_package=provider_package,
                source_result_cache=source_result_cache,
                source_result_cache_lock=source_result_cache_lock,
                recipe_selection_cache=recipe_selection_cache,
                recipe_selection_cache_lock=recipe_selection_cache_lock,
                allow_source_provider=False,
                source_provider_stack=(*source_provider_stack, recipe_package or package),
                preferred_command_hints=preferred_command_hints,
            )
            if not provider_mapping.get("source_checkout"):
                continue
            source_checkout = str(provider_mapping.get("source_checkout", "") or "")
            source_error = str(provider_mapping.get("source_error", "") or "")
            command_hints.update(provider_mapping.get("command_hints", []) or [])
            source_provider = {
                "source_provider_package": str(provider_mapping.get("package", "") or ""),
                "source_provider_required_version": str(
                    provider_mapping.get("required_version", "") or ""
                ),
                "source_provider_channel": str(
                    provider_mapping.get("source_channel", "") or channel
                ),
                "source_provider_recipe_package": str(
                    provider_mapping.get("recipe_package", "") or ""
                ),
                "source_provider_recipe_path": str(
                    provider_mapping.get("recipe_path", "") or ""
                ),
                "source_provider_source_url": str(
                    provider_mapping.get("source_url", "") or ""
                ),
                "source_provider_source_ref": str(
                    provider_mapping.get("source_ref", "") or ""
                ),
                "source_provider_reason": "source_less_run_dependency",
            }
            _emit_extract_status_once(
                settings,
                (
                    f"{channel}-source-provider",
                    package.lower(),
                    provider_mapping.get("package", ""),
                    source_checkout,
                ),
                {
                    "status": f"{channel}-source-provider",
                    "source_channel": channel,
                    "package": package,
                    "provider_package": provider_mapping.get("package", ""),
                    "provider_source_checkout": source_checkout,
                    "reason": "source_less_run_dependency",
                },
            )
            break
    record_command_hints = set(preferred_command_hints or [])
    command_hints.update(record_command_hints)
    command_hints.update(_extract_source_command_hints(source_checkout, package))
    source_command_docs = _extract_source_command_docs(
        source_checkout,
        package,
        command_hints,
        preferred_command_hints=record_command_hints,
    )
    source_version_match = _source_version_match_status(required_version, recipe_version)
    source_confidence = _source_confidence_from_recipe_selection(
        snapshot.selection_reason,
        required_version=required_version,
        recipe_version=recipe_version,
    )
    mapping = {
        "package": package,
        "required_version": required_version,
        "source_channel": channel,
        "recipe_package": snapshot.package,
        "recipe_path": snapshot.recipe_path,
        "recipe_version": recipe_version,
        "recipe_commit": snapshot.commit,
        "recipe_commit_date": snapshot.commit_date,
        "recipe_selection_reason": snapshot.selection_reason,
        "recipe_scanned_commits": snapshot.scanned_commits,
        "source_confidence": source_confidence,
        "source_version_match": source_version_match,
        "source_url": source_url,
        "source_url_template": source_url_template,
        "source_ref": source_ref,
        "source_ref_template": source_ref_template,
        "source_checkout": source_checkout,
        "source_is_binary_artifact": _source_checkout_is_binary_artifact(source_checkout),
        "source_error": source_error,
        "command_hints": sorted(command_hints),
    }
    if source_command_docs:
        mapping["source_command_docs"] = source_command_docs
    if record_command_hints:
        mapping["record_command_hints"] = sorted(record_command_hints)
    if recipe_package != package:
        mapping["source_recipe_alias"] = recipe_package
    if source_url_candidates:
        mapping["source_url_candidates"] = source_url_candidates
    if source_attempts:
        mapping["source_attempts"] = source_attempts
    mapping.update(_source_http_browser_fallback_info(source_checkout))
    if source_artifact_url:
        mapping["source_artifact_url"] = source_artifact_url
        mapping["source_artifact_checkout"] = source_artifact_checkout
    if source_fallback_reason:
        mapping["source_fallback_reason"] = source_fallback_reason
    mapping.update(source_provider)
    return mapping


def _resolve_conda_forge_source_mapping(
    *,
    package: str,
    required_version: str,
    settings: ExtractionSettings,
    source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None = None,
    source_result_cache_lock: threading.Lock | None = None,
    recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None = None,
    recipe_selection_cache_lock: threading.Lock | None = None,
    preferred_command_hints: list[str] | set[str] | None = None,
) -> dict:
    cache_root = settings.cache_root or Path(".gtsm-cache") / "source-cache"
    feedstock_repo, feedstock_error, recipe_package = _ensure_conda_forge_feedstock_repo(
        cache_root, package, settings
    )
    return _resolve_recipe_source_mapping(
        package=package,
        required_version=required_version,
        channel="conda-forge",
        settings=settings,
        recipes_repo=feedstock_repo,
        recipes_repo_error=feedstock_error,
        ref="HEAD",
        feedstock=True,
        recipe_package=recipe_package,
        source_result_cache=source_result_cache,
        source_result_cache_lock=source_result_cache_lock,
        recipe_selection_cache=recipe_selection_cache,
        recipe_selection_cache_lock=recipe_selection_cache_lock,
        preferred_command_hints=preferred_command_hints,
    )


_SOURCE_CONFIDENCE_RANK = {"": 0, "low": 1, "weak": 1, "near": 2, "exact": 3}
_SOURCE_VERSION_MATCH_RANK = {
    "": 0,
    "unknown": 0,
    "mismatch": 1,
    "not_required": 3,
    "numeric_equivalent": 3,
    "exact": 4,
}


def _source_confidence_rank(mapping: dict) -> int:
    value = str(mapping.get("source_confidence", "") or "").strip().lower()
    return _SOURCE_CONFIDENCE_RANK.get(value, 0)


def _source_version_match_rank(mapping: dict) -> int:
    value = str(mapping.get("source_version_match", "") or "").strip().lower()
    return _SOURCE_VERSION_MATCH_RANK.get(value, 0)


def _should_consider_conda_forge_fallback(primary: dict) -> bool:
    primary_checkout = str(primary.get("source_checkout", "") or "")
    if not primary_checkout or bool(primary.get("source_is_binary_artifact")):
        return True
    return (
        _source_confidence_rank(primary) < _SOURCE_CONFIDENCE_RANK["exact"]
        or _source_version_match_rank(primary) < _SOURCE_VERSION_MATCH_RANK["numeric_equivalent"]
    )


def _should_use_conda_forge_fallback(primary: dict, fallback: dict) -> bool:
    primary_checkout = str(primary.get("source_checkout", "") or "")
    fallback_checkout = str(fallback.get("source_checkout", "") or "")
    if fallback_checkout and not bool(fallback.get("source_is_binary_artifact")):
        if not primary_checkout or bool(primary.get("source_is_binary_artifact")):
            return True
        if _source_version_match_rank(fallback) > _source_version_match_rank(primary):
            return True
        return _source_confidence_rank(fallback) > _source_confidence_rank(primary)
    if primary_checkout and not bool(primary.get("source_is_binary_artifact")):
        return False
    if fallback_checkout:
        return True
    if not primary.get("recipe_path") and fallback.get("recipe_path"):
        return True
    return bool(not primary.get("source_url") and fallback.get("source_url"))


def _annotate_conda_forge_fallback(fallback: dict, primary: dict) -> dict:
    annotated = dict(fallback)
    annotated["fallback_from_channel"] = primary.get("source_channel", "bioconda")
    annotated["fallback_from_error"] = primary.get("error", "")
    annotated["fallback_from_source_error"] = primary.get("source_error", "")
    annotated["fallback_from_recipe_selection_reason"] = primary.get(
        "recipe_selection_reason", ""
    )
    return annotated


def _resolve_bioconda_source_mappings(
    *,
    package_names: list[str],
    requirement_versions: dict[str, str],
    settings: ExtractionSettings,
    recipes_repo: Path | None = None,
    recipes_repo_error: str = "",
    source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None = None,
    source_result_cache_lock: threading.Lock | None = None,
    recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None = None,
    recipe_selection_cache_lock: threading.Lock | None = None,
    preferred_command_hints: list[str] | set[str] | None = None,
) -> list[dict]:
    if not settings.bioconda_checkout_sources:
        return []
    cache_root = settings.cache_root or Path(".gtsm-cache") / "source-cache"
    if recipes_repo is None and not recipes_repo_error:
        recipes_repo, recipes_repo_error = _ensure_bioconda_repo(
            cache_root, settings.bioconda_ref, settings
        )

    mappings: list[dict] = []
    for package in package_names:
        required_version = requirement_versions.get(package, "")
        if _is_source_denylisted_package(package):
            mappings.append(
                {
                    "package": package,
                    "required_version": required_version,
                    "source_channel": "runtime",
                    "recipe_package": "",
                    "recipe_path": "",
                    "recipe_version": "",
                    "recipe_commit": "",
                    "recipe_commit_date": "",
                    "recipe_selection_reason": "source_package_denylisted",
                    "recipe_scanned_commits": 0,
                    "source_confidence": "low",
                    "source_version_match": "unknown",
                    "source_url": "",
                    "source_url_template": "",
                    "source_ref": "",
                    "source_ref_template": "",
                    "source_checkout": "",
                    "source_is_binary_artifact": False,
                    "source_error": "",
                    "command_hints": [],
                }
            )
            continue
        primary = _resolve_recipe_source_mapping(
            package=package,
            required_version=required_version,
            channel="bioconda",
            settings=settings,
            recipes_repo=recipes_repo,
            recipes_repo_error=recipes_repo_error,
            ref=settings.bioconda_ref,
            recipe_package=_source_recipe_package_alias(package),
            source_result_cache=source_result_cache,
            source_result_cache_lock=source_result_cache_lock,
            recipe_selection_cache=recipe_selection_cache,
            recipe_selection_cache_lock=recipe_selection_cache_lock,
            preferred_command_hints=preferred_command_hints,
        )
        if not _should_consider_conda_forge_fallback(primary):
            mappings.append(primary)
            continue

        fallback = _resolve_conda_forge_source_mapping(
            package=package,
            required_version=required_version,
            settings=settings,
            source_result_cache=source_result_cache,
            source_result_cache_lock=source_result_cache_lock,
            recipe_selection_cache=recipe_selection_cache,
            recipe_selection_cache_lock=recipe_selection_cache_lock,
            preferred_command_hints=preferred_command_hints,
        )
        if _should_use_conda_forge_fallback(primary, fallback):
            mappings.append(_annotate_conda_forge_fallback(fallback, primary))
        else:
            mappings.append(primary)
    return mappings


def _version_consistency_report(
    *,
    requirement_versions: dict[str, str],
    selected_container: str,
    bioconda_mappings: list[dict],
) -> dict:
    issues: list[str] = []
    container_version = _version_from_image_ref(selected_container)
    for package, required in requirement_versions.items():
        if required and container_version and required not in container_version:
            issues.append(f"container_version_mismatch:{package}:{required}!={container_version}")
    for mapping in bioconda_mappings:
        required = str(mapping.get("required_version", "")).strip()
        recipe_version = str(mapping.get("recipe_version", "")).strip()
        required_normalized = _normalize_recipe_version_text(required)
        recipe_normalized = _normalize_recipe_version_text(recipe_version)
        if required and recipe_version and required_normalized != recipe_normalized:
            reason = str(mapping.get("recipe_selection_reason", "") or "unknown")
            issues.append(
                f"recipe_version_mismatch:{mapping.get('package', '')}:{required}!={recipe_version}:{reason}"
            )
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "selected_container_version": container_version,
    }


def _wrapper_uses_macros(xml_text: str) -> bool:
    lowered = xml_text.lower()
    return ("<expand " in lowered) or ("<macros>" in lowered)


def _extract_macro_tokens_from_text(xml_text: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return tokens
    for node in root.findall(".//token"):
        name = str(node.attrib.get("name", "") or "").strip()
        value = _strip_text(node.text)
        if name and value:
            tokens[name] = value
    return tokens


def _macro_tokens(wrapper_text: str, macro_files: list[Path]) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for macro_file in macro_files:
        try:
            tokens.update(_extract_macro_tokens_from_text(macro_file.read_text(encoding="utf-8")))
        except OSError:
            continue
    tokens.update(_extract_macro_tokens_from_text(wrapper_text))
    return tokens


def _apply_macro_tokens_to_text(xml_text: str, tokens: dict[str, str]) -> str:
    rendered = xml_text
    for name, value in sorted(tokens.items(), key=lambda item: len(item[0]), reverse=True):
        rendered = rendered.replace(name, value)
    return rendered


def _apply_macro_tokens_to_element(element: ET.Element, tokens: dict[str, str]) -> None:
    for key, value in list(element.attrib.items()):
        element.attrib[key] = _apply_macro_tokens_to_text(value, tokens)
    if element.text:
        element.text = _apply_macro_tokens_to_text(element.text, tokens)
    if element.tail:
        element.tail = _apply_macro_tokens_to_text(element.tail, tokens)
    for child in list(element):
        _apply_macro_tokens_to_element(child, tokens)


def _parse_macro_file_for_definitions(raw_text: str, tokens: dict[str, str]) -> ET.Element | None:
    try:
        return ET.fromstring(raw_text)
    except ET.ParseError:
        pass
    try:
        return ET.fromstring(_apply_macro_tokens_to_text(raw_text, tokens))
    except ET.ParseError:
        return None


def _macro_definitions(macro_files: list[Path], tokens: dict[str, str]) -> dict[str, ET.Element]:
    macro_map: dict[str, ET.Element] = {}
    for macro_file in macro_files:
        try:
            root = _parse_macro_file_for_definitions(
                macro_file.read_text(encoding="utf-8"),
                tokens,
            )
        except OSError:
            continue
        if root is None:
            continue
        for node in root.findall(".//xml"):
            name = str(node.attrib.get("name", "")).strip()
            if name:
                definition = copy.deepcopy(node)
                _apply_macro_tokens_to_element(definition, tokens)
                macro_map[name] = definition
    return macro_map


def _expand_wrapper_xml_with_galaxy(
    wrapper_xml: Path,
    expanded_root: Path,
) -> tuple[Path, str] | None:
    try:
        tree, _macro_paths = load_tool_with_refereces(str(wrapper_xml))
        root = tree.getroot()
        expanded_xml = xml_to_string(root)
    except Exception:
        return None

    if isinstance(expanded_xml, bytes):
        expanded_text = expanded_xml.decode("utf-8")
    else:
        expanded_text = str(expanded_xml)

    expanded_root.mkdir(parents=True, exist_ok=True)
    output_path = expanded_root / f"{wrapper_xml.stem}.expanded.xml"
    output_path.write_text(expanded_text, encoding="utf-8")
    status = "partial" if root.findall(".//expand") else "expanded"
    return output_path, status


def _replace_expand_nodes(parent: ET.Element, macro_map: dict[str, ET.Element]) -> tuple[int, int]:
    expanded = 0
    unresolved = 0
    children = list(parent)
    i = 0
    while i < len(children):
        child = children[i]
        if child.tag == "expand":
            macro_name = str(child.attrib.get("macro", "")).strip()
            if macro_name and macro_name in macro_map:
                definition = macro_map[macro_name]
                replacement_nodes = [copy.deepcopy(node) for node in list(definition)]
                parent.remove(child)
                for offset, repl in enumerate(replacement_nodes):
                    parent.insert(i + offset, repl)
                children = list(parent)
                expanded += 1
                i += len(replacement_nodes)
                continue
            unresolved += 1
        else:
            inner_expanded, inner_unresolved = _replace_expand_nodes(child, macro_map)
            expanded += inner_expanded
            unresolved += inner_unresolved
        i += 1
    return expanded, unresolved


def _expand_wrapper_xml(
    wrapper_xml: Path, macro_files: list[Path], expanded_root: Path
) -> tuple[Path, str]:
    raw_text = wrapper_xml.read_text(encoding="utf-8")
    tokens = _macro_tokens(raw_text, macro_files)
    rendered_text = _apply_macro_tokens_to_text(raw_text, tokens)
    uses_macros = _wrapper_uses_macros(raw_text)
    if not uses_macros and rendered_text == raw_text:
        return wrapper_xml, "not_applicable"

    if uses_macros:
        galaxy_expanded = _expand_wrapper_xml_with_galaxy(wrapper_xml, expanded_root)
        if galaxy_expanded is not None:
            return galaxy_expanded

    try:
        root = ET.fromstring(rendered_text)
    except ET.ParseError:
        return wrapper_xml, "parse_error"

    macro_map = _macro_definitions(macro_files, tokens)
    expanded_count, unresolved = _replace_expand_nodes(root, macro_map)
    if expanded_count == 0 and unresolved > 0 and rendered_text == raw_text:
        return wrapper_xml, "unresolved"

    expanded_root.mkdir(parents=True, exist_ok=True)
    output_path = expanded_root / f"{wrapper_xml.stem}.expanded.xml"
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    if unresolved > 0:
        return output_path, "partial"
    return output_path, "expanded"


_TOKEN_RE = re.compile(r"[A-Za-z0-9_.$:{}+/@=-]+")
_SHELL_COMMAND_DENYLIST = {
    ".",
    "awk",
    "basename",
    "cat",
    "case",
    "cd",
    "chmod",
    "chown",
    "cp",
    "do",
    "done",
    "elif",
    "else",
    "esac",
    "eval",
    "exec",
    "exit",
    "cut",
    "dirname",
    "echo",
    "env",
    "export",
    "fi",
    "find",
    "for",
    "grep",
    "gunzip",
    "gzip",
    "head",
    "if",
    "ln",
    "local",
    "ls",
    "mkdir",
    "mktemp",
    "mv",
    "print",
    "printf",
    "pwd",
    "read",
    "return",
    "rm",
    "rmdir",
    "sed",
    "set",
    "shift",
    "sort",
    "source",
    "tail",
    "tar",
    "tee",
    "then",
    "test",
    "touch",
    "trap",
    "tr",
    "ulimit",
    "uniq",
    "unset",
    "until",
    "wc",
    "while",
    "xargs",
}
_GENERIC_CONTAINER_NAMES = {
    "bioconductor-org.at.tair.db",
    "openjdk",
    "perl",
    "python",
    "r-base",
    "r-getopt",
}
_SETUP_HELPER_COMMAND_KEYS = {
    "bgzip",
    "hmmpress",
    "samtools",
    "tabix",
}
_FILELIKE_NAMES = {
    "file",
    "file1",
    "file2",
    "fasta_file",
    "index_dir",
    "input",
    "input_bam",
    "input_dir",
    "input_file",
    "input_path",
    "localbam",
    "out",
    "outdir",
    "outfile",
    "output",
    "output_dir",
    "output_file",
    "output_path",
    "path_to_html",
    "plots",
    "query",
    "result",
    "tabular",
}
_HELP_FLAGS = ("--help", "-h")
_HELP_PROBE_MODES = {"safe", "exploratory"}
_INTERPRETER_COMMANDS = {
    "bash",
    "java",
    "perl",
    "python",
    "python2",
    "python3",
    "r",
    "rscript",
    "sh",
}
_REDIRECT_TOKENS = {">", ">>", "<", "2>", "2>>", "&>", "|"}
_FILELIKE_SUFFIXES = {
    ".bam",
    ".bai",
    ".bed",
    ".bigsdb",
    ".bz2",
    ".csv",
    ".fa",
    ".fasta",
    ".fastq",
    ".fq",
    ".gz",
    ".jar",
    ".json",
    ".py",
    ".r",
    ".R",
    ".sam",
    ".sh",
    ".tabular",
    ".tar",
    ".tsv",
    ".txt",
    ".vcf",
    ".xml",
    ".zip",
}
_EXECUTABLE_SCRIPT_SUFFIXES = {".bash", ".pl", ".py", ".r", ".sh"}
_SOURCE_METADATA_FILENAMES = {
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}
_SOURCE_NON_COMMAND_FILENAME_PREFIXES = (
    "copying",
    "install",
    "license",
    "makefile",
    "manifest",
    "readme",
)
_SOURCE_HINT_DENYLIST_KEYS = {
    "build",
    "conda",
    "install",
    "mamba",
    "pip",
    "python",
    "python2",
    "python3",
    "setup",
    "setuptools",
    "wheel",
}
_SOURCE_HINT_EXECUTABLE_DIRS = {"bin", "script", "scripts"}
_SOURCE_HINT_IGNORED_DIRS = {"__pycache__", "test", "tests"}
_SOURCE_HINT_MAX_ARCHIVE_MEMBERS = 5000
_SOURCE_HINT_MAX_METADATA_BYTES = 512_000
_CHEETAH_CONTROL_RE = re.compile(
    r"^#(?:if|elif|else|end|for|set|from|import|def|while|try|except|slurp)\b"
)


def _command_tokens(segment: str) -> list[str]:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = _TOKEN_RE.findall(segment)
    return [token.strip() for token in tokens if token.strip()]


def _clean_command_token(token: str) -> str:
    return token.strip().strip("\"'").strip()


def _is_shell_command_denylisted_token(token: str) -> bool:
    cleaned = _clean_command_token(token)
    return cleaned == cleaned.lower() and cleaned in _SHELL_COMMAND_DENYLIST


def _is_assignment_token(token: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token))


def _is_placeholder_token(token: str) -> bool:
    return (
        "$" in token
        or "__tool_directory__" in token
        or token.startswith("{")
        or token.startswith("#")
    )


def _is_path_token(token: str) -> bool:
    return "/" in token or "\\" in token


def _is_python_module_token(token: str) -> bool:
    cleaned = _clean_command_token(token)
    if not cleaned or _is_placeholder_token(cleaned) or _is_path_token(cleaned):
        return False
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$", cleaned))


def _is_filelike_token(token: str) -> bool:
    lowered = token.lower()
    key = _normalized_command_key(token)
    if lowered in _FILELIKE_NAMES:
        return True
    if key in _FILELIKE_NAMES:
        return True
    if "_" in lowered and key.endswith(
        (
            "dir",
            "directory",
            "file",
            "filename",
            "input",
            "outdir",
            "outfile",
            "output",
            "path",
        )
    ):
        return True
    if any(lowered.endswith(suffix.lower()) for suffix in _FILELIKE_SUFFIXES):
        return True
    return "." in token


def _is_executable_script_token(token: str) -> bool:
    cleaned = _clean_command_token(token)
    if not cleaned or _is_path_token(cleaned) or _is_placeholder_token(cleaned):
        return False
    if cleaned.lower() in _SOURCE_METADATA_FILENAMES:
        return False
    suffix = Path(cleaned).suffix.lower()
    if suffix not in _EXECUTABLE_SCRIPT_SUFFIXES:
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_+.-]*$", cleaned))


def _is_executable_candidate(token: str) -> bool:
    cleaned = _clean_command_token(token)
    lowered = cleaned.lower()
    if not cleaned or cleaned in _REDIRECT_TOKENS or cleaned.startswith("-"):
        return False
    if lowered in {"none", "null"}:
        return False
    if _is_assignment_token(cleaned) or _is_placeholder_token(cleaned) or _is_path_token(cleaned):
        return False
    if _is_shell_command_denylisted_token(cleaned) or lowered in _INTERPRETER_COMMANDS:
        return False
    if _is_filelike_token(cleaned) and not _is_executable_script_token(cleaned):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_+.-]*$", cleaned))


def _is_subcommand_candidate(token: str) -> bool:
    cleaned = _clean_command_token(token)
    lowered = cleaned.lower()
    if not cleaned or cleaned in _REDIRECT_TOKENS or cleaned.startswith("-"):
        return False
    if lowered in {"none", "null"}:
        return False
    if _is_assignment_token(cleaned) or _is_placeholder_token(cleaned) or _is_path_token(cleaned):
        return False
    if _is_shell_command_denylisted_token(cleaned) or lowered in _INTERPRETER_COMMANDS:
        return False
    if _is_filelike_token(cleaned):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_+-]*$", cleaned))


def _is_input_like_subcommand_token(token: str) -> bool:
    key = _normalized_command_key(token)
    if not key:
        return True
    input_like_keys = {
        "arg",
        "args",
        "argument",
        "arguments",
        "db",
        "database",
        "file",
        "files",
        "input",
        "inputs",
        "output",
        "outputs",
        "path",
        "paths",
        "read",
        "reads",
        "sample",
        "samples",
    }
    if key in input_like_keys:
        return True
    return key.endswith(
        (
            "arg",
            "args",
            "argument",
            "arguments",
            "db",
            "database",
            "file",
            "files",
            "input",
            "inputs",
            "output",
            "outputs",
            "path",
            "paths",
            "read",
            "reads",
            "sample",
            "samples",
        )
    )


def _valid_subcommand_for_primary(primary: str, subcommand: str) -> bool:
    if not subcommand or not _is_subcommand_candidate(subcommand):
        return False
    if _is_input_like_subcommand_token(subcommand):
        return False
    subcommand_key = _normalized_command_key(subcommand)
    return bool(subcommand_key) and subcommand_key not in _command_scoring_keys(primary)


def _command_segments(line: str) -> list[str]:
    segments: list[str] = []
    quote = ""
    escaped = False
    start = 0
    index = 0
    while index < len(line):
        char = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        delimiter_width = 0
        if line.startswith(("&&", "||"), index):
            delimiter_width = 2
        elif char in {";", "|"}:
            delimiter_width = 1
        if delimiter_width:
            segment = line[start:index].strip()
            if segment:
                segments.append(segment)
            index += delimiter_width
            start = index
            continue
        index += 1
    segment = line[start:].strip()
    if segment:
        segments.append(segment)
    return segments


def _line_quote_state(line: str, quote: str = "") -> str:
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
    return quote


def _command_candidate_lines(command_text: str) -> list[str]:
    lines: list[str] = []
    in_cheetah_comment = False
    shell_quote = ""
    for raw in command_text.splitlines():
        line = raw.strip()
        while line:
            if in_cheetah_comment:
                end = line.find("*#")
                if end == -1:
                    line = ""
                    break
                line = line[end + 2 :].strip()
                in_cheetah_comment = False
                continue
            start = line.find("#*")
            if start == -1:
                break
            before = line[:start].strip()
            after = line[start + 2 :]
            end = after.find("*#")
            if before:
                line = before
                in_cheetah_comment = end == -1
                break
            if end == -1:
                in_cheetah_comment = True
                line = ""
                break
            line = after[end + 2 :].strip()

        if not line:
            continue
        if shell_quote:
            shell_quote = _line_quote_state(line, shell_quote)
            continue
        if line.startswith(("#", "<!--", "-->")):
            continue
        if _CHEETAH_CONTROL_RE.match(line):
            continue
        lines.append(line)
        shell_quote = _line_quote_state(line, shell_quote)
    return lines


def _signature_from_segment(segment: str) -> tuple[str, str]:
    if segment.lstrip().startswith(("'", '"')):
        return "", ""
    tokens = [_clean_command_token(token) for token in _command_tokens(segment)]
    tokens = [token for token in tokens if token and token not in _REDIRECT_TOKENS]
    skippable_prefixes = {"then", "do", "else"}
    while tokens and (_is_assignment_token(tokens[0]) or tokens[0].lower() in skippable_prefixes):
        tokens.pop(0)
    if not tokens:
        return "", ""
    primary = tokens[0]
    if (
        primary.lower() in {"python", "python2", "python3"}
        and len(tokens) >= 3
        and tokens[1] == "-m"
        and _is_python_module_token(tokens[2])
    ):
        module_primary = f"{primary} -m {tokens[2]}"
        for token in tokens[3:]:
            if token.startswith("-"):
                continue
            if _valid_subcommand_for_primary(module_primary, token):
                return module_primary, token
            break
        return module_primary, ""
    if _is_shell_command_denylisted_token(primary) or primary.lower() in _INTERPRETER_COMMANDS:
        return "", ""
    if not _is_executable_candidate(primary):
        return "", ""
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        if _valid_subcommand_for_primary(primary, token):
            return primary, token
        break
    return primary, ""


def _leading_subcommand_from_line(line: str) -> str:
    segments = _command_segments(line)
    if not segments:
        return ""
    tokens = [_clean_command_token(token) for token in _command_tokens(segments[0])]
    tokens = [token for token in tokens if token and token not in _REDIRECT_TOKENS]
    if not tokens:
        return ""
    token = tokens[0]
    return token if _is_subcommand_candidate(token) else ""


def _allows_next_line_subcommand(line: str, primary: str) -> bool:
    stripped = line.rstrip()
    if stripped.endswith(("&&", "||", "|", ";")):
        return False
    primary_key = _normalized_command_key(primary)
    return primary_key not in {
        _normalized_command_key(command) for command in _SETUP_HELPER_COMMAND_KEYS
    }


def _infer_command_signatures(command_text: str) -> tuple[str, list[str], list[str]]:
    lines = _command_candidate_lines(command_text)
    invocation_patterns = lines[:8]
    if not lines:
        return "", [], []

    primary = ""
    subcommands: list[str] = []
    for line_index, line in enumerate(lines):
        for segment in _command_segments(line):
            candidate, subcommand = _signature_from_segment(segment)
            if not candidate:
                continue
            if (
                not subcommand
                and line_index + 1 < len(lines)
                and _allows_next_line_subcommand(line, candidate)
            ):
                next_subcommand = _leading_subcommand_from_line(lines[line_index + 1])
                if _valid_subcommand_for_primary(candidate, next_subcommand):
                    subcommand = next_subcommand
            if not primary:
                primary = candidate
            if candidate == primary and subcommand and subcommand not in subcommands:
                subcommands.append(subcommand)
            if primary:
                break
        if primary:
            break
    return primary, subcommands, invocation_patterns


def _command_candidate_signatures(command_text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    lines = _command_candidate_lines(command_text)
    continuation_subcommand_lines: set[int] = set()
    for line_index, line in enumerate(lines):
        if line_index in continuation_subcommand_lines:
            continue
        for segment in _command_segments(line):
            candidate = _signature_from_segment(segment)
            if (
                candidate[0]
                and not candidate[1]
                and line_index + 1 < len(lines)
                and _allows_next_line_subcommand(line, candidate[0])
            ):
                subcommand = _leading_subcommand_from_line(lines[line_index + 1])
                if _valid_subcommand_for_primary(candidate[0], subcommand):
                    candidate = (candidate[0], subcommand)
                    continuation_subcommand_lines.add(line_index + 1)
            if not candidate[0] or candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _container_image_base_name(image_ref: str) -> str:
    ref = _normalize_container_ref(image_ref)
    name = _url_cache_image_name(ref) if _is_url_ref(ref) else ref.rsplit("/", 1)[-1]
    name = name.split("@", 1)[0].split(":", 1)[0]
    return name.strip()


def _identity_values_from_text(value: str) -> list[str]:
    values = []
    cleaned = value.strip()
    if cleaned:
        values.append(cleaned)
    values.extend(token for token in re.split(r"[^A-Za-z0-9_.+-]+", cleaned) if token)
    values.extend(token for token in re.split(r"[_\-.]+", cleaned) if token)
    return values


def _record_identity_keys(
    record: ToolRecord,
    image: str,
    candidate_packages: list[str] | tuple[str, ...] | None = None,
) -> tuple[set[str], set[str], set[str], set[str], str]:
    hint_keys: set[str] = set()
    core_identity_keys: set[str] = set()
    source_hint_keys: set[str] = set()
    package_keys: set[str] = set()
    owning_packages = list(candidate_packages or record.requirement_packages)
    filter_mappings_by_candidate = candidate_packages is not None

    core_values = [
        record.shed_name,
        record.tool_id,
        record.tool_name,
        Path(record.wrapper_path).stem if record.wrapper_path else "",
    ]
    for value in core_values:
        for token in _identity_values_from_text(str(value)):
            key = _normalized_command_key(token)
            if key:
                hint_keys.add(key)
                core_identity_keys.add(key)

    for value in owning_packages:
        for token in _identity_values_from_text(str(value)):
            key = _normalized_command_key(token)
            if key:
                hint_keys.add(key)

    for package in owning_packages:
        key = _normalized_command_key(package)
        if not _is_generic_container_image_key(key):
            package_keys.add(key)

    owning_package_keys = {_normalized_command_key(package) for package in owning_packages}
    for mapping in record.bioconda_sources:
        package = str(mapping.get("package", "") or "")
        package_key = _normalized_command_key(package)
        if (
            filter_mappings_by_candidate
            and owning_package_keys
            and package_key not in owning_package_keys
        ):
            continue
        for token in _identity_values_from_text(package):
            key = _normalized_command_key(token)
            if key:
                hint_keys.add(key)
        if package_key:
            package_keys.add(package_key)
        record_hint_keys = {
            _normalized_command_key(str(hint))
            for hint in (mapping.get("record_command_hints", []) or [])
        }
        command_hints = list(mapping.get("command_hints", []) or [])
        source_checkout = str(mapping.get("source_checkout", "") or "")
        source_checkout_hints: list[str] = []
        if source_checkout:
            source_checkout_hints.extend(_cached_source_command_hints(source_checkout, package))
            command_hints.extend(source_checkout_hints)
        source_checkout_hint_keys = {
            _normalized_command_key(str(hint)) for hint in source_checkout_hints
        }
        for hint in command_hints:
            key = _normalized_command_key(str(hint))
            if key:
                hint_keys.add(key)
                if key not in record_hint_keys or key in source_checkout_hint_keys:
                    source_hint_keys.add(key)

    image_key = _normalized_command_key(_container_image_base_name(image))
    return hint_keys, core_identity_keys, source_hint_keys, package_keys, image_key


def _command_key_matches(candidate_key: str, hint_key: str) -> bool:
    if not candidate_key or not hint_key:
        return False
    if candidate_key == hint_key:
        return True
    if len(candidate_key) >= 4 and len(hint_key) >= 4:
        return candidate_key in hint_key or hint_key in candidate_key
    return False


def _command_key_plural_pair(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left + "s" == right or right + "s" == left


def _command_scoring_keys(primary: str) -> set[str]:
    keys: set[str] = set()
    primary_key = _normalized_command_key(primary)
    if primary_key:
        keys.add(primary_key)
    tokens = _command_tokens(primary)
    if (
        len(tokens) >= 3
        and tokens[0].lower() in {"python", "python2", "python3"}
        and tokens[1] == "-m"
    ):
        module = tokens[2]
        module_key = _normalized_command_key(module)
        if module_key:
            keys.add(module_key)
        for part in module.split("."):
            part_key = _normalized_command_key(part)
            if part_key:
                keys.add(part_key)
    else:
        name = Path(tokens[0]).name if tokens else primary
        for value in (name, Path(name).stem):
            key = _normalized_command_key(value)
            if key:
                keys.add(key)
    return keys


def _score_command_candidate(
    primary: str,
    *,
    hint_keys: set[str],
    core_identity_keys: set[str],
    source_hint_keys: set[str],
    package_keys: set[str],
    image_key: str,
    strict_package_match: bool = False,
) -> int:
    primary_keys = _command_scoring_keys(primary)
    if not primary_keys:
        return 0

    score = 0
    if primary_keys & core_identity_keys:
        score += 180
    elif any(
        _command_key_matches(primary_key, identity_key)
        for primary_key in primary_keys
        for identity_key in core_identity_keys
    ):
        score += 120
    if primary_keys & source_hint_keys:
        score += 140
    if primary_keys & package_keys:
        score += 120
    elif any(
        _command_key_matches(primary_key, package_key)
        for primary_key in primary_keys
        for package_key in package_keys
    ):
        score += 70
    generic_image = _is_generic_container_image_key(image_key)
    if not generic_image and image_key and image_key in primary_keys:
        score += 110
    elif not generic_image and any(
        _command_key_matches(primary_key, image_key) for primary_key in primary_keys
    ):
        score += 80
    if primary_keys & hint_keys:
        score += 90
    elif any(
        _command_key_matches(primary_key, hint)
        for primary_key in primary_keys
        for hint in hint_keys
    ):
        score += 55

    if generic_image and not primary_keys & source_hint_keys and not primary_keys & package_keys:
        return 0
    if (
        strict_package_match
        and not primary_keys & source_hint_keys
        and not primary_keys & package_keys
        and not any(
            _command_key_matches(primary_key, package_key)
            for primary_key in primary_keys
            for package_key in package_keys
        )
        and (generic_image or image_key not in primary_keys)
        and (
            generic_image
            or not any(
                _command_key_matches(primary_key, image_key) for primary_key in primary_keys
            )
        )
    ):
        return 0
    return score


def _subcommand_matches_record_identity(
    subcommand: str,
    *,
    core_identity_keys: set[str],
    source_hint_keys: set[str],
    package_keys: set[str],
    image_key: str,
) -> bool:
    key = _normalized_command_key(subcommand)
    if not key:
        return False
    identity_keys = {*core_identity_keys, *source_hint_keys}
    if image_key:
        identity_keys.add(image_key)
    if key in identity_keys:
        return True
    if any(_command_key_matches(key, identity_key) for identity_key in identity_keys):
        return True
    return key in package_keys


def _filter_record_subcommands(
    primary: str,
    subcommands: list[str],
    *,
    core_identity_keys: set[str],
    source_hint_keys: set[str],
    package_keys: set[str],
    image_key: str,
) -> list[str]:
    valid = [
        subcommand
        for subcommand in subcommands
        if _valid_subcommand_for_primary(primary, subcommand)
    ]
    core_matched = [
        subcommand
        for subcommand in valid
        if _subcommand_matches_record_identity(
            subcommand,
            core_identity_keys=core_identity_keys,
            source_hint_keys=set(),
            package_keys=set(),
            image_key="",
        )
    ]
    if core_matched:
        matched_keys = {_normalized_command_key(subcommand) for subcommand in core_matched}
        expanded_core_matched = [
            subcommand
            for subcommand in valid
            if any(
                _normalized_command_key(subcommand) == matched_key
                or _command_key_plural_pair(_normalized_command_key(subcommand), matched_key)
                for matched_key in matched_keys
            )
        ]
        return expanded_core_matched
    matched = [
        subcommand
        for subcommand in valid
        if _subcommand_matches_record_identity(
            subcommand,
            core_identity_keys=core_identity_keys,
            source_hint_keys=source_hint_keys,
            package_keys=package_keys,
            image_key=image_key,
        )
    ]
    return matched or valid


def _command_probe_role(
    primary: str,
    *,
    core_identity_keys: set[str],
    source_hint_keys: set[str],
    package_keys: set[str],
    image_key: str,
) -> str:
    primary_keys = _command_scoring_keys(primary)
    if not primary_keys:
        return "auxiliary"
    if primary_keys & core_identity_keys or any(
        _command_key_matches(primary_key, identity_key)
        for primary_key in primary_keys
        for identity_key in core_identity_keys
    ):
        return "core"
    if primary_keys & source_hint_keys:
        return "core"
    if not _is_generic_container_image_key(image_key) and image_key and (
        image_key in primary_keys
        or any(_command_key_matches(primary_key, image_key) for primary_key in primary_keys)
    ):
        return "core"
    if primary_keys & {_normalized_command_key(command) for command in _SETUP_HELPER_COMMAND_KEYS}:
        return "auxiliary"
    if primary_keys & package_keys:
        return "core"
    return "auxiliary"


def _dedupe_command_candidates(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not candidate[0] or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _literal_command_signatures(command_text: str) -> list[tuple[str, str]]:
    return _command_candidate_signatures(command_text)


def _helper_literal_values(raw: str) -> list[str]:
    return re.findall(r"['\"]([^'\"]+)['\"]", raw)


def _helper_list_call_command_signatures(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    call_re = re.compile(
        r"(?<![A-Za-z0-9_.])(?:subprocess\.)?(?:run|call|check_call|check_output|Popen)\s*"
        r"\(\s*(?:args\s*=\s*)?\[(?P<items>[^\]]{0,500})\]",
        re.DOTALL,
    )
    assignment_re = re.compile(
        r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[(?P<items>[^\]]{0,500})\]"
    )
    for match in call_re.finditer(text):
        values = _helper_literal_values(match.group("items"))
        if values:
            candidates.extend(_literal_command_signatures(" ".join(values[:3])))
    for line in text.splitlines():
        match = assignment_re.match(line)
        if not match:
            continue
        if not _looks_like_command_assignment_name(match.group("name")):
            continue
        values = _helper_literal_values(match.group("items"))
        if values:
            candidates.extend(_literal_command_signatures(" ".join(values[:3])))
    return candidates


def _looks_like_command_assignment_name(name: str) -> bool:
    normalized = name.lower()
    return any(
        marker in normalized
        for marker in ("binary", "cli", "cmd", "command", "executable", "program", "tool")
    )


def _helper_string_command_signatures(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    patterns = [
        r"(?<![A-Za-z0-9_.])(?:os\.)?system\s*\(\s*['\"](?P<command>[^'\"]{1,500})['\"]",
        r"(?<![A-Za-z0-9_.])(?:subprocess\.)?(?:run|call|check_call|check_output|Popen)\s*"
        r"\(\s*['\"](?P<command>[^'\"]{1,500})['\"]",
        r"(?<![A-Za-z0-9_.])system2\s*\(\s*['\"](?P<command>[^'\"]{1,200})['\"]"
        r"(?:\s*,\s*(?:args\s*=\s*)?c?\((?P<args>[^)]{0,300})\))?",
        r"(?<![A-Za-z0-9_.])system\s*\(\s*['\"](?P<command>[^'\"]{1,500})['\"]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            command = match.group("command")
            args = match.groupdict().get("args") or ""
            values = [command, *_helper_literal_values(args)]
            candidates.extend(_literal_command_signatures(" ".join(values[:3])))

    assignment_re = re.compile(
        r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"](?P<command>[^'\"]{1,500})['\"]"
    )
    for line in text.splitlines():
        match = assignment_re.match(line)
        if match and _looks_like_command_assignment_name(match.group("name")):
            candidates.extend(_literal_command_signatures(match.group("command")))
    return candidates


def _is_shell_like_script_text(text: str, language: str = "", extension: str = "") -> bool:
    language = language.lower()
    extension = extension.lower()
    if language in {"awk", "bash", "shell"} or extension in {".awk", ".bash", ".sh"}:
        return True
    first_line = text.lstrip().splitlines()[0].lower() if text.strip() else ""
    return first_line.startswith("#!") and any(
        marker in first_line for marker in (" bash", "/bash", " sh", "/sh", " awk", "/awk")
    )


def _script_text_command_signatures(
    text: str, *, language: str = "", extension: str = ""
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    candidates.extend(_helper_list_call_command_signatures(text))
    candidates.extend(_helper_string_command_signatures(text))
    if _is_shell_like_script_text(text, language=language, extension=extension):
        candidates.extend(_command_candidate_signatures(text))
    return _dedupe_command_candidates(candidates)


def _wrapper_helper_identity_keys(record: ToolRecord) -> set[str]:
    keys: set[str] = set()
    for helper in record.wrapper_helper_files:
        for value in (helper.get("relative_path", ""), helper.get("path", "")):
            name = Path(str(value or "")).name
            for token in (name, Path(name).stem):
                key = _normalized_command_key(token)
                if key:
                    keys.add(key)
    for configfile in record.wrapper_configfiles:
        if configfile.get("template_kind") != "script_template":
            continue
        for value in (configfile.get("filename", ""), configfile.get("name", "")):
            name = Path(str(value or "")).name
            for token in (name, Path(name).stem):
                key = _normalized_command_key(token)
                if key:
                    keys.add(key)
    return keys


def _is_wrapper_helper_candidate(primary: str, helper_keys: set[str]) -> bool:
    if not helper_keys:
        return False
    return bool(_command_scoring_keys(primary) & helper_keys)


def _wrapper_script_command_candidates(record: ToolRecord) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for helper in record.wrapper_helper_files:
        path = Path(str(helper.get("path", "") or ""))
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        candidates.extend(
            _script_text_command_signatures(
                text,
                language=_configfile_language_hint(str(helper.get("extension", "") or "")),
                extension=str(helper.get("extension", "") or ""),
            )
        )
    for configfile in record.wrapper_configfiles:
        if configfile.get("template_kind") != "script_template":
            continue
        content = str(configfile.get("content", "") or "")
        if content:
            candidates.extend(
                _script_text_command_signatures(
                    content,
                    language=str(configfile.get("language", "") or ""),
                    extension=str(configfile.get("extension", "") or ""),
                )
            )
    return _dedupe_command_candidates(candidates)


def _record_help_command_plan(
    record: ToolRecord,
    image: str,
    probe_mode: str = "exploratory",
    candidate_packages: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    command_text = record.command_text or "\n".join(record.invocation_patterns)
    candidates = _command_candidate_signatures(command_text)
    candidates.extend(_wrapper_script_command_candidates(record))
    candidates = _dedupe_command_candidates(candidates)
    if not candidates and record.primary_command and not command_text.strip():
        candidates.append((record.primary_command, ""))
        for subcommand in record.subcommands:
            candidates.append((record.primary_command, subcommand))

    hint_keys, core_identity_keys, source_hint_keys, package_keys, image_key = _record_identity_keys(
        record,
        image,
        candidate_packages=candidate_packages,
    )
    strict_package_match = candidate_packages is not None and not _is_generic_container_image_key(
        image_key
    )
    scored: list[tuple[int, int, str, str]] = []
    helper_keys = _wrapper_helper_identity_keys(record)
    for index, (primary, subcommand) in enumerate(candidates):
        if _is_wrapper_helper_candidate(primary, helper_keys):
            continue
        score = _score_command_candidate(
            primary,
            hint_keys=hint_keys,
            core_identity_keys=core_identity_keys,
            source_hint_keys=source_hint_keys,
            package_keys=package_keys,
            image_key=image_key,
            strict_package_match=strict_package_match,
        )
        if score >= 100:
            scored.append((score, -index, primary, subcommand))
    if not scored:
        return []

    scored.sort(reverse=True)
    roles_by_primary = {
        primary: _command_probe_role(
            primary,
            core_identity_keys=core_identity_keys,
            source_hint_keys=source_hint_keys,
            package_keys=package_keys,
            image_key=image_key,
        )
        for _, _, primary, _ in scored
    }
    if (
        record.wrapper_source_summary.get("api_backed_wrapper")
        and roles_by_primary
        and all(role == "auxiliary" for role in roles_by_primary.values())
    ):
        return []

    commands: list[dict[str, str]] = []
    seen_commands: set[str] = set()
    seen_primaries: set[str] = set()
    for _, _, selected_primary, _ in scored:
        if selected_primary in seen_primaries:
            continue
        seen_primaries.add(selected_primary)
        subcommands: list[str] = []
        for score, _, primary, subcommand in scored:
            if score < 100 or primary != selected_primary or not subcommand:
                continue
            if subcommand not in subcommands:
                subcommands.append(subcommand)
        subcommands = _filter_record_subcommands(
            selected_primary,
            subcommands,
            core_identity_keys=core_identity_keys,
            source_hint_keys=source_hint_keys,
            package_keys=package_keys,
            image_key=image_key,
        )
        help_flags = _record_help_flags(record, selected_primary)
        environment_prefix = _record_probe_environment_prefix(record, selected_primary)
        for command in _extract_help_commands(
            selected_primary,
            subcommands,
            probe_mode=probe_mode,
            help_flags=help_flags,
        ):
            command = _with_environment_prefix(command, environment_prefix)
            if command in seen_commands:
                continue
            seen_commands.add(command)
            commands.append(
                {
                    "command": command,
                    "primary": selected_primary,
                    "probe_role": roles_by_primary.get(selected_primary, "core"),
                }
            )
        if len(seen_primaries) >= 4:
            break
    return commands


def _record_help_commands(
    record: ToolRecord,
    image: str,
    probe_mode: str = "exploratory",
    candidate_packages: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    return [
        str(item.get("command", ""))
        for item in _record_help_command_plan(
            record,
            image,
            probe_mode=probe_mode,
            candidate_packages=candidate_packages,
        )
        if str(item.get("command", "")).strip()
    ]


def _checkpoint_key(tools_root: Path, tool_dir: Path, xml_file: Path) -> str:
    return f"{tool_dir.relative_to(tools_root)}::{xml_file.name}"


def _user_defined_tool_yaml_files(tool_dir: Path) -> list[str]:
    matches: list[str] = []
    for path in sorted([*tool_dir.glob("*.yml"), *tool_dir.glob("*.yaml")]):
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(payload, dict) and payload.get("class") == "GalaxyUserTool":
            matches.append(str(path.relative_to(tool_dir)))
    return matches


def _primary_tool_xml_files(tool_dir: Path) -> list[str]:
    matches: list[str] = []
    for path in sorted(tool_dir.glob("*.xml")):
        if _xml_file_root_tag(path) != "tool":
            continue
        matches.append(path.relative_to(tool_dir).as_posix())
    return matches


def _extract_one_wrapper(
    tools_root: Path,
    tool_dir: Path,
    xml_file: Path,
    settings: ExtractionSettings,
    expanded_root: Path,
    bioconda_recipes_repo: Path | None,
    bioconda_recipes_repo_error: str,
    bioconda_source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None,
    bioconda_source_result_cache_lock: threading.Lock | None,
    bioconda_recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None,
    bioconda_recipe_selection_cache_lock: threading.Lock | None,
    source_semaphore: threading.Semaphore | None = None,
) -> ToolRecord:
    xml_rel = str(xml_file.relative_to(tool_dir))
    primary_xml_files = _primary_tool_xml_files(tool_dir)
    shed = _load_shed_metadata(tool_dir)
    package_name = str(shed.get("name", "") or "").strip() or tool_dir.name
    package_owner = str(shed.get("owner", "") or "").strip()
    package_id = f"{package_owner}/{package_name}" if package_owner else package_name
    suite_id, suite_name, suite_members, is_suite_root = _shed_suite_fields(shed)
    shed_categories = [
        str(item).strip() for item in (shed.get("categories") or []) if str(item).strip()
    ]

    xml_text = xml_file.read_text(encoding="utf-8")
    macro_files = sorted(tool_dir.glob("*macro*.xml")) + sorted(tool_dir.glob("macros*.xml"))
    macro_files = sorted({path.resolve() for path in macro_files})
    macro_files_rel = [
        str(path.relative_to(tool_dir))
        for path in macro_files
        if path.exists() and path.is_file() and path.is_relative_to(tool_dir)
    ]
    uses_macros = _wrapper_uses_macros(xml_text) or bool(
        _macro_tokens(xml_text, [tool_dir / rel for rel in macro_files_rel])
    )
    expanded_xml_path = xml_file
    macro_expansion_status = "not_applicable"
    if settings.expand_macros:
        expanded_xml_path, macro_expansion_status = _expand_wrapper_xml(
            wrapper_xml=xml_file,
            macro_files=[tool_dir / rel for rel in macro_files_rel],
            expanded_root=expanded_root / tool_dir.name,
        )
    metadata_root = (
        ET.parse(expanded_xml_path).getroot()
        if expanded_xml_path != xml_file
        else ET.fromstring(xml_text)
    )

    tool_id = str(metadata_root.attrib.get("id", "")).strip()
    tool_name = str(metadata_root.attrib.get("name", "")).strip() or tool_dir.name
    help_text = _extract_help(metadata_root)
    p_types, in_types, out_types = _extract_datatypes(metadata_root)
    tests = _extract_tests(metadata_root)
    package_names, requirement_versions, container_refs = _extract_requirements(metadata_root)
    macro_requirement_sets = []
    for macro_file in [tool_dir / rel for rel in macro_files_rel]:
        try:
            macro_root = ET.fromstring(macro_file.read_text(encoding="utf-8"))
        except (ET.ParseError, OSError):
            continue
        _apply_macro_tokens_to_element(macro_root, _macro_tokens(xml_text, [macro_file]))
        macro_requirement_sets.append(_extract_requirements(macro_root))
    if macro_requirement_sets:
        package_names, requirement_versions, container_refs = _merge_requirement_sets(
            (package_names, requirement_versions, container_refs),
            *macro_requirement_sets,
        )
    command_nodes = metadata_root.findall(".//command")
    command_text = "\n".join(
        _strip_text(node.text) for node in command_nodes if _strip_text(node.text)
    )
    version_command_nodes = metadata_root.findall(".//version_command")
    version_command_text = "\n".join(
        _strip_text(node.text) for node in version_command_nodes if _strip_text(node.text)
    )
    primary_command, subcommands, invocation_patterns = _infer_command_signatures(command_text)
    wrapper_helper_files, wrapper_helper_skipped = _extract_wrapper_helper_files(
        tool_dir,
        command_text,
        max_bytes=settings.wrapper_source_max_bytes,
    )
    wrapper_configfiles = _extract_wrapper_configfiles(
        metadata_root,
        command_text,
        max_bytes=settings.wrapper_configfile_max_bytes,
    )
    wrapper_sidecar_files = _extract_wrapper_sidecar_files(
        tool_dir,
        primary_xml=xml_file,
        macro_files_rel=macro_files_rel,
        max_bytes=settings.wrapper_configfile_max_bytes,
    )
    wrapper_source_summary = _wrapper_source_summary(
        wrapper_helper_files,
        wrapper_configfiles,
        wrapper_sidecar_files,
        wrapper_helper_skipped,
    )
    configfile_help_context = _wrapper_configfile_help_context(wrapper_configfiles)
    helper_help_context = _wrapper_helper_help_context(wrapper_helper_files)
    if configfile_help_context:
        help_text = "\n\n".join(
            part for part in (help_text.strip(), configfile_help_context) if part
        )
    if helper_help_context:
        help_text = "\n\n".join(
            part for part in (help_text.strip(), helper_help_context) if part
        )

    documentation = ""
    if settings.fetch_documentation:
        homepage = str(shed.get("homepage_url", "") or "").strip()
        if homepage:
            documentation = _fetch_github_readme(homepage)

    container_candidate_details = _build_container_candidate_details(
        container_refs=container_refs,
        package_names=package_names,
        requirement_versions=requirement_versions,
        settings=settings,
    )
    normalized_container_candidates = [
        str(candidate["image"])
        for candidate in container_candidate_details
        if candidate.get("status", "ok") == "ok" and candidate.get("image")
    ]
    selected_candidate = _choose_container_candidate(
        container_candidate_details,
        requirement_versions=requirement_versions,
        requirement_packages=package_names,
    )
    selected_container = str(selected_candidate.get("image", "") or "")
    record_source_command_hints = _record_source_command_hints(
        primary_command=primary_command,
        subcommands=subcommands,
        command_text=command_text,
        wrapper_helper_files=wrapper_helper_files,
        wrapper_configfiles=wrapper_configfiles,
    )
    if source_semaphore is not None:
        source_semaphore.acquire()
    try:
        bioconda_sources = _resolve_bioconda_source_mappings(
            package_names=package_names,
            requirement_versions=requirement_versions,
            settings=settings,
            recipes_repo=bioconda_recipes_repo,
            recipes_repo_error=bioconda_recipes_repo_error,
            source_result_cache=bioconda_source_result_cache,
            source_result_cache_lock=bioconda_source_result_cache_lock,
            recipe_selection_cache=bioconda_recipe_selection_cache,
            recipe_selection_cache_lock=bioconda_recipe_selection_cache_lock,
            preferred_command_hints=record_source_command_hints,
        )
    finally:
        if source_semaphore is not None:
            source_semaphore.release()
    source_help_context = _source_command_help_context(bioconda_sources)
    if source_help_context:
        help_text = "\n\n".join(
            part for part in (help_text.strip(), source_help_context) if part
        )
    version_consistency = _version_consistency_report(
        requirement_versions=requirement_versions,
        selected_container=selected_container,
        bioconda_mappings=bioconda_sources,
    )

    test_data_dir = tool_dir / "test-data"
    test_data_files: list[str] = []
    if test_data_dir.exists():
        test_data_files = sorted(
            str(path.relative_to(tool_dir)) for path in test_data_dir.glob("**/*") if path.is_file()
        )
    udt_yaml_files = _user_defined_tool_yaml_files(tool_dir)
    udt_yaml_path = str(tool_dir / udt_yaml_files[0]) if udt_yaml_files else ""
    if settings.synthesize_udt_yaml:
        rel_tool_dir = tool_dir.relative_to(tools_root)
        udt_output_path = (
            expanded_root.parent
            / "udt"
            / rel_tool_dir
            / f"{_safe_slug(xml_file.stem) or 'tool'}.udt.yml"
        )
        udt_yaml_path = str(
            _synthesize_udt_yaml(
                root=metadata_root,
                tool_id=tool_id,
                tool_name=tool_name,
                command_text=command_text,
                help_text=help_text,
                selected_container=selected_container,
                container_refs=container_refs,
                output_path=udt_output_path,
                wrapper_configfile_max_bytes=settings.wrapper_configfile_max_bytes,
            )
        )
        udt_yaml_files = [str(Path(udt_yaml_path).relative_to(expanded_root.parent))]

    return ToolRecord(
        package_id=package_id,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_dir=str(tool_dir),
        wrapper_path=str(xml_file),
        xml_files=primary_xml_files or [xml_rel],
        udt_yaml_path=udt_yaml_path,
        udt_yaml_files=udt_yaml_files,
        shed_name=package_name,
        shed_owner=package_owner,
        shed_description=str(shed.get("description", "") or "").strip(),
        shed_homepage_url=str(shed.get("homepage_url", "") or "").strip(),
        shed_remote_repository_url=str(shed.get("remote_repository_url", "") or "").strip(),
        shed_categories=shed_categories,
        suite_id=suite_id,
        suite_name=suite_name,
        suite_members=suite_members,
        is_suite_root=is_suite_root,
        help_text=help_text,
        original_help_text=help_text,
        documentation=documentation,
        expanded_xml_path=str(expanded_xml_path),
        macro_files=macro_files_rel,
        uses_macros=uses_macros,
        macro_expansion_status=macro_expansion_status,
        version_command_text=version_command_text,
        primary_command=primary_command,
        subcommands=subcommands,
        invocation_patterns=invocation_patterns,
        command_text=command_text,
        wrapper_helper_files=wrapper_helper_files,
        wrapper_configfiles=wrapper_configfiles,
        wrapper_sidecar_files=wrapper_sidecar_files,
        wrapper_source_summary=wrapper_source_summary,
        input_parameter_types=p_types,
        input_datatypes=in_types,
        output_datatypes=out_types,
        datatype_report=_build_datatype_report(in_types, out_types),
        tests=tests,
        test_data_files=test_data_files,
        requirement_packages=package_names,
        requirement_versions=requirement_versions,
        container_candidates=normalized_container_candidates,
        container_candidate_details=container_candidate_details,
        selected_container=selected_container,
        bioconda_sources=bioconda_sources,
        version_consistency=version_consistency,
    )


def _with_retries(
    tools_root: Path,
    tool_dir: Path,
    xml_file: Path,
    settings: ExtractionSettings,
    expanded_root: Path,
    bioconda_recipes_repo: Path | None,
    bioconda_recipes_repo_error: str,
    bioconda_source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None,
    bioconda_source_result_cache_lock: threading.Lock | None,
    bioconda_recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None,
    bioconda_recipe_selection_cache_lock: threading.Lock | None,
    source_semaphore: threading.Semaphore | None = None,
) -> ToolRecord:
    last_error: Exception | None = None
    for attempt in range(settings.retries):
        try:
            return _extract_one_wrapper(
                tools_root=tools_root,
                tool_dir=tool_dir,
                xml_file=xml_file,
                settings=settings,
                expanded_root=expanded_root,
                bioconda_recipes_repo=bioconda_recipes_repo,
                bioconda_recipes_repo_error=bioconda_recipes_repo_error,
                bioconda_source_result_cache=bioconda_source_result_cache,
                bioconda_source_result_cache_lock=bioconda_source_result_cache_lock,
                bioconda_recipe_selection_cache=bioconda_recipe_selection_cache,
                bioconda_recipe_selection_cache_lock=bioconda_recipe_selection_cache_lock,
                source_semaphore=source_semaphore,
            )
        except Exception as error:
            last_error = error
            time.sleep(settings.retry_backoff_seconds * (attempt + 1))
    return ToolRecord(
        package_id=tool_dir.name,
        tool_name=tool_dir.name,
        tool_dir=str(tool_dir),
        wrapper_path=str(xml_file),
        tests=[{"error": f"failed after retries: {last_error}"}],
    )


def _load_checkpoint(checkpoint_file: Path) -> set[str]:
    if not checkpoint_file.exists():
        return set()
    return {
        line.strip()
        for line in checkpoint_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _load_retry_manifest_wrapper_paths(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    wrappers = payload.get("wrappers", []) if isinstance(payload, dict) else []
    paths: set[str] = set()
    if not isinstance(wrappers, list):
        return paths
    for item in wrappers:
        if not isinstance(item, dict):
            continue
        wrapper_path = str(item.get("wrapper_path", "") or "").strip()
        if wrapper_path:
            paths.add(wrapper_path)
    return paths


def _utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"


def _unique_run_id(runs_root: Path, requested: str | None = None) -> str:
    base = _safe_slug(requested or _utc_run_id()) or _utc_run_id()
    run_id = base
    suffix = 1
    while (runs_root / run_id).exists():
        suffix += 1
        run_id = f"{base}-{suffix}"
    return run_id


def _run_artifact_targets(output_jsonl: Path, checkpoint_file: Path) -> list[Path]:
    return [
        output_jsonl,
        checkpoint_file,
        output_jsonl.with_suffix(".index.json"),
        output_jsonl.with_suffix(".execution.json"),
    ]


def _write_run_manifest(run_dir: Path, payload: dict[str, object]) -> Path:
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _archive_extract_outputs(
    output_jsonl: Path, checkpoint_file: Path, run_id: str
) -> tuple[list[str], str]:
    archived: list[str] = []
    runs_root = output_jsonl.parent / "runs"
    archive_id = _unique_run_id(runs_root, f"restart-archive-{run_id}")
    archive_dir = runs_root / archive_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    for path in _run_artifact_targets(output_jsonl, checkpoint_file):
        if path.exists():
            shutil.move(str(path), str(archive_dir / path.name))
            archived.append(str(path))

    expanded_root = output_jsonl.parent / "expanded"
    if expanded_root.exists():
        shutil.move(str(expanded_root), str(archive_dir / "expanded"))
        archived.append(str(expanded_root))
    udt_root = output_jsonl.parent / "udt"
    if udt_root.exists():
        shutil.move(str(udt_root), str(archive_dir / "udt"))
        archived.append(str(udt_root))

    if not archived:
        shutil.rmtree(archive_dir)
        return [], ""

    _write_run_manifest(
        archive_dir,
        {
            "schema_version": "0.1.0",
            "kind": "restart_archive",
            "run_id": archive_id,
            "archived_at": datetime.now(UTC).isoformat(),
            "source_output": str(output_jsonl),
            "source_checkpoint": str(checkpoint_file),
            "artifacts": archived,
        },
    )
    return archived, str(archive_dir)


def _snapshot_completed_run(
    *,
    output_jsonl: Path,
    checkpoint_file: Path,
    index_path: Path,
    execution_report_path: Path,
    expanded_root: Path,
    settings: ExtractionSettings,
    total_tools: int,
    total_wrappers: int,
    processed_now: int,
    execution_summary: dict[str, int],
) -> tuple[str, str]:
    runs_root = output_jsonl.parent / "runs"
    run_id = settings.run_id or _unique_run_id(runs_root)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for path in (output_jsonl, checkpoint_file, index_path, execution_report_path):
        if path.exists():
            shutil.copy2(path, run_dir / path.name)
            copied.append(path.name)
    if expanded_root.exists():
        run_expanded = run_dir / "expanded"
        if run_expanded.exists():
            shutil.rmtree(run_expanded)
        shutil.copytree(expanded_root, run_expanded)
        copied.append("expanded")
    udt_root = output_jsonl.parent / "udt"
    if udt_root.exists():
        run_udt = run_dir / "udt"
        if run_udt.exists():
            shutil.rmtree(run_udt)
        shutil.copytree(udt_root, run_udt)
        copied.append("udt")
    if settings.status_log_path and settings.status_log_path.exists():
        run_status_log = run_dir / settings.status_log_path.name
        if settings.status_log_path.resolve() != run_status_log.resolve():
            shutil.copy2(settings.status_log_path, run_status_log)
            copied.append(settings.status_log_path.name)

    _write_run_manifest(
        run_dir,
        {
            "schema_version": "0.1.0",
            "kind": "extract_corpus_run",
            "run_id": run_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "output": str(output_jsonl),
            "checkpoint": str(checkpoint_file),
            "index": str(index_path),
            "execution_report": str(execution_report_path),
            "total_tools": total_tools,
            "total_wrappers": total_wrappers,
            "processed_now": processed_now,
            "container_execution": execution_summary,
            "artifacts": copied,
        },
    )
    current_path = output_jsonl.parent / "current"
    current_path.write_text(run_id + "\n", encoding="utf-8")
    return str(run_dir), str(current_path)


def _write_dataset_index(output_jsonl: Path, records: list[ToolRecord]) -> Path:
    index_path = output_jsonl.with_suffix(".index.json")
    package_map: dict[str, dict] = {}
    suite_map: dict[str, dict] = {}
    wrapper_map: dict[str, dict] = {}

    for record in records:
        package_map.setdefault(record.package_id, {"wrappers": []})
        package_map[record.package_id]["wrappers"].append(record.wrapper_path)

        wrapper_key = record.tool_id or record.wrapper_path
        wrapper_map[wrapper_key] = {
            "package_id": record.package_id,
            "wrapper_path": record.wrapper_path,
            "expanded_xml_path": record.expanded_xml_path,
            "primary_command": record.primary_command,
            "subcommands": record.subcommands,
        }

        if record.suite_id:
            suite_map.setdefault(
                record.suite_id, {"suite_name": record.suite_name, "packages": set()}
            )
            suite_map[record.suite_id]["packages"].add(record.package_id)

    suite_payload = {
        suite_id: {"suite_name": value["suite_name"], "packages": sorted(value["packages"])}
        for suite_id, value in suite_map.items()
    }
    payload = {
        "schema_version": "0.4.0",
        "records": len(records),
        "packages": package_map,
        "suites": suite_payload,
        "wrappers": wrapper_map,
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return index_path


def _record_from_json(payload: dict) -> ToolRecord:
    record = ToolRecord()
    for field_info in dataclass_fields(ToolRecord):
        if field_info.name in payload:
            setattr(record, field_info.name, payload[field_info.name])
    return record


def _load_records_from_jsonl(output_jsonl: Path) -> list[ToolRecord]:
    if not output_jsonl.exists():
        return []
    records: list[ToolRecord] = []
    for line in output_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(_record_from_json(payload))
    return records


def _new_container_execution_state(settings: ExtractionSettings) -> ContainerExecutionState:
    if not settings.execute_containers:
        return ContainerExecutionState()
    prepare_workers = max(1, int(settings.container_prepare_workers or 1))
    return ContainerExecutionState(
        runtimes=_available_container_runtimes(settings),
        preparation_semaphore=threading.Semaphore(prepare_workers),
        image_quarantine_store=_load_container_image_quarantine_store(settings),
    )


def _container_execution_summary(state: ContainerExecutionState) -> dict[str, int]:
    return {
        "images_planned": len(state.planned_images),
        "images_prepared": state.images_prepared,
        "images_pulled": state.images_pulled,
        "images_removed": state.images_removed,
        "commands_executed": state.commands_executed,
        "commands_failed": state.commands_failed,
        "help_ok": state.help_ok,
        "help_degraded": state.help_degraded,
        "usage_degraded": state.usage_degraded,
        "api_validation_ok": state.api_validation_ok,
        "api_validation_failed": state.api_validation_failed,
        "missing_command": state.missing_command,
        "non_help_output": state.non_help_output,
        "failed_probe": state.failed_probe,
        "prepare_failed": state.prepare_failed,
        "runtime_error": state.runtime_error,
        "timeout": state.timeout,
        "runtime_fallbacks": state.runtime_fallbacks,
    }


def _record_probe_summary(state: ContainerExecutionState, status: str, returncode: int) -> None:
    if status == "container-command-help":
        state.help_ok += 1
    elif status == "container-command-help-degraded":
        state.help_degraded += 1
    elif status == "container-command-usage-degraded":
        state.usage_degraded += 1
    elif status == "container-api-validation-ok":
        state.api_validation_ok += 1
    elif status == "container-api-validation-failed":
        state.commands_failed += 1
        state.api_validation_failed += 1
    elif status == "container-command-nonhelp":
        state.commands_failed += 1
        state.non_help_output += 1
    elif status == "container-command-failed-probe":
        state.commands_failed += 1
        state.failed_probe += 1
    else:
        state.commands_failed += 1
    if returncode == 124:
        state.timeout += 1


def _container_execution_summary_from_records(records: list[ToolRecord]) -> dict[str, int]:
    state = ContainerExecutionState()
    prepared_images: set[str] = set()
    for record in records:
        for candidate in record.container_candidate_details:
            image = str(candidate.get("image", "") or "")
            if image and candidate.get("status", "ok") == "ok":
                state.planned_images.add(image)
        for event in record.container_execution:
            status = str(event.get("status", "") or "")
            image = str(event.get("image", "") or "")
            if image:
                prepared_images.add(image)
            returncode = int(event.get("returncode", 0) or 0)
            if event.get("phase") == "run":
                state.commands_executed += 1
                _record_probe_summary(state, status, returncode)
            elif status == "container-command-missing":
                state.commands_failed += 1
                state.missing_command += 1
                if returncode == 124:
                    state.timeout += 1
            elif status == "container-prepare-failed":
                state.commands_failed += 1
                state.prepare_failed += 1
                if returncode == 124:
                    state.timeout += 1
            elif event.get("phase") == "api_validation":
                state.commands_executed += 1
                _record_probe_summary(state, status, returncode)
            elif status:
                state.commands_failed += int(
                    status
                    not in {
                        "container-command-help",
                        "container-command-help-degraded",
                        "container-command-usage-degraded",
                        "container-api-validation-ok",
                    }
                )
                if returncode == 124:
                    state.timeout += 1
    state.images_prepared = len(prepared_images)
    summary = _container_execution_summary(state)
    summary["rebuilt_from_records"] = 1
    return summary


def rebuild_execution_report_from_jsonl(
    output_jsonl: Path, execution_report_path: Path | None = None
) -> Path:
    records = _load_records_from_jsonl(output_jsonl)
    report_path = execution_report_path or output_jsonl.with_suffix(".execution.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "rebuilt_from_records": True,
                "summary": _container_execution_summary_from_records(records),
                "records": [asdict(record) for record in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def _default_corpus_jsonl_for_execution_report(execution_report_path: Path) -> Path:
    name = execution_report_path.name
    if name.endswith(".execution.json"):
        return execution_report_path.with_name(name[: -len(".execution.json")] + ".jsonl")
    return execution_report_path.with_suffix(".jsonl")


def _default_checkpoint_for_execution_report(execution_report_path: Path) -> Path:
    name = execution_report_path.name
    if name.endswith(".execution.json"):
        return execution_report_path.with_name(name[: -len(".execution.json")] + ".checkpoint")
    return execution_report_path.with_suffix(".checkpoint")


def _jsonl_integrity(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "line_count": 0,
        "valid_json": False,
        "invalid_line_count": 0,
        "invalid_line_numbers": [],
    }
    if not path.exists():
        return payload
    invalid_lines: list[int] = []
    line_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                if len(invalid_lines) < 20:
                    invalid_lines.append(line_number)
    payload.update(
        {
            "line_count": line_count,
            "valid_json": not invalid_lines,
            "invalid_line_count": len(invalid_lines),
            "invalid_line_numbers": invalid_lines,
        }
    )
    return payload


def _line_count_integrity(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {"path": str(path), "exists": path.exists(), "line_count": 0}
    if not path.exists():
        return payload
    with path.open(encoding="utf-8") as handle:
        payload["line_count"] = sum(1 for line in handle if line.strip())
    return payload


def _source_mapping_status(source: dict) -> str:
    package_key = _normalized_command_key(str(source.get("package", "") or ""))
    source_channel = str(source.get("source_channel", "") or "").strip().lower()
    source_error = str(source.get("source_error", "") or "").strip()
    source_checkout = str(source.get("source_checkout", "") or "").strip()
    source_url = str(source.get("source_url", "") or "").strip()
    if (source_channel == "runtime" or package_key in _SOURCELESS_PACKAGE_KEYS) and not source_checkout:
        return "source_not_applicable"
    if source_checkout and source.get("source_confidence") == "weak":
        return "weak_source"
    if source.get("source_fallback_reason") and source_checkout:
        return str(source.get("source_fallback_reason"))
    if source.get("source_provider_package"):
        return "provider_source"
    if source_error:
        return "source_error"
    if source_checkout and _source_checkout_is_binary_artifact(source_checkout):
        return "binary_artifact"
    if source_checkout:
        return "usable_source"
    if not source_url:
        return "blank_source_url"
    return "missing_checkout"


def _source_status_is_usable(status: str) -> bool:
    if status in {"usable_source", "provider_source", "source_not_applicable"}:
        return True
    return status.endswith("_fallback")


def _source_diagnostics(records: list[dict]) -> tuple[dict[str, object], list[dict[str, object]]]:
    status_counts: dict[str, int] = {}
    source_channel_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    missing: list[dict[str, object]] = []
    records_with_usable_source = 0
    records_with_source = 0
    records_with_binary_artifact = 0
    records_with_provider_source = 0
    records_with_source_error = 0
    records_with_no_source_mapping = 0
    for record in records:
        sources = record.get("bioconda_sources", []) or []
        if not isinstance(sources, list):
            sources = []
        if sources:
            records_with_source += 1
        else:
            records_with_no_source_mapping += 1
            status = (
                "no_source_mapping"
                if record.get("requirement_packages")
                else "no_requirement_packages"
            )
            status_counts[status] = status_counts.get(status, 0) + 1
            missing.append(
                {
                    "tool_id": record.get("tool_id", ""),
                    "package_id": record.get("package_id", ""),
                    "requirement_packages": record.get("requirement_packages", []),
                    "status": status,
                }
            )
            continue
        record_statuses: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                continue
            status = _source_mapping_status(source)
            record_statuses.add(status)
            status_counts[status] = status_counts.get(status, 0) + 1
            channel = str(source.get("source_channel", "") or "").strip()
            if channel:
                source_channel_counts[channel] = source_channel_counts.get(channel, 0) + 1
            provider = str(source.get("source_provider_package", "") or "").strip()
            if provider:
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            if not _source_status_is_usable(status):
                missing.append(
                    {
                        "tool_id": record.get("tool_id", ""),
                        "package_id": record.get("package_id", ""),
                        "requirement_packages": record.get("requirement_packages", []),
                        "package": source.get("package", ""),
                        "required_version": source.get("required_version", ""),
                        "source_channel": source.get("source_channel", ""),
                        "source_url": source.get("source_url", ""),
                        "source_checkout": source.get("source_checkout", ""),
                        "source_error": source.get("source_error", ""),
                        "recipe_version": source.get("recipe_version", ""),
                        "recipe_selection_reason": source.get(
                            "recipe_selection_reason", ""
                        ),
                        "source_confidence": source.get("source_confidence", ""),
                        "source_version_match": source.get("source_version_match", ""),
                        "source_fallback_reason": source.get(
                            "source_fallback_reason", ""
                        ),
                        "fallback_from_channel": source.get("fallback_from_channel", ""),
                        "source_provider_package": source.get(
                            "source_provider_package", ""
                        ),
                        "status": status,
                    }
                )
        if any(_source_status_is_usable(status) for status in record_statuses):
            records_with_usable_source += 1
        if "binary_artifact" in record_statuses:
            records_with_binary_artifact += 1
        if "provider_source" in record_statuses:
            records_with_provider_source += 1
        if "source_error" in record_statuses:
            records_with_source_error += 1
    coverage = {
        "total_records": len(records),
        "records_with_source_mapping": records_with_source,
        "records_without_source_mapping": records_with_no_source_mapping,
        "records_with_usable_source": records_with_usable_source,
        "records_with_provider_source": records_with_provider_source,
        "records_with_binary_artifact": records_with_binary_artifact,
        "records_with_source_error": records_with_source_error,
        "source_status_counts": dict(
            sorted(status_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "source_channel_counts": dict(
            sorted(source_channel_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "source_provider_package_counts": dict(
            sorted(provider_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    return coverage, missing[:1000]


_CONTAINER_ISSUE_STATUSES = {
    "container-command-nonhelp",
    "container-command-help-degraded",
    "container-command-usage-degraded",
    "container-command-failed-probe",
    "container-command-missing",
    "container-api-validation-failed",
    "container-prepare-failed",
}
_HARD_CONTAINER_ISSUE_STATUSES = {
    "container-command-failed-probe",
    "container-command-missing",
    "container-api-validation-failed",
    "container-prepare-failed",
}


def _failure_text_blob(*values: object) -> str:
    return "\n".join(str(value or "") for value in values).lower()


def _looks_like_useful_nonhelp_banner(text: str) -> bool:
    lowered = text.lower()
    if not lowered.strip() or _looks_like_help_text(text):
        return False
    if "structure by pritchard" in lowered and 'can\'t open the file "mainparams"' in lowered:
        return True
    return bool(
        "version" in lowered
        and any(marker in lowered for marker in ("can't open the file", "cannot open file"))
        and any(marker in lowered for marker in ("exiting", "error(s)", "required"))
    )


def _container_failure_category(event: dict) -> str:
    status = str(event.get("status", "") or "")
    command = str(event.get("command", "") or "")
    failure_text = _failure_text_blob(
        event.get("error_text", ""),
        event.get("stderr", ""),
        event.get("stdout", ""),
    )
    text = _failure_text_blob(
        command,
        event.get("error_text", ""),
        event.get("stderr", ""),
        event.get("stdout", ""),
    )
    if status == "container-command-help-degraded" or status == "container-command-usage-degraded":
        return "usable_nonzero_help"
    if status == "container-api-validation-failed":
        return "api_import_failure"
    if status == "container-prepare-failed":
        if "timed out" in text or int(event.get("returncode", 0) or 0) == 124:
            return "container_prepare_timeout"
        return "container_prepare_failed"
    if "python -m " in command and status == "container-command-missing":
        return "missing_executable"
    if "$" in command or "__tool_directory__" in command or "{" in command:
        return "unresolved_template"
    if (
        "cannot open shared object file" in failure_text
        or "error while loading shared libraries" in failure_text
    ):
        return "runtime_dependency_missing"
    if _looks_like_missing_argument_traceback(failure_text):
        return "bad_probe_variant"
    if (
        "traceback" in failure_text
        or "modulenotfounderror" in failure_text
        or "importerror" in failure_text
    ):
        return "api_import_failure"
    if "can't open file" in text or "cannot open file" in text or "fail to open file 'help'" in text:
        return "bad_probe_variant"
    if status == "container-command-missing":
        return "missing_executable"
    if status == "container-command-nonhelp":
        if _looks_like_useful_nonhelp_banner(failure_text):
            return "usable_nonhelp_output"
        primary = str(event.get("primary_command", "") or _command_primary(command))
        if primary and Path(primary).suffix.lower() in _WRAPPER_SCRIPT_CONFIGFILE_EXTENSIONS:
            return "helper_script_probe"
        return "bad_probe_variant"
    if status == "container-command-failed-probe":
        return "bad_probe_variant"
    return "terminal_unknown"


def _source_failure_category(source: dict) -> str:
    status = _source_mapping_status(source)
    text = _failure_text_blob(source.get("source_error", ""), source.get("source_url", ""))
    source_error = str(source.get("source_error", "") or "").strip()
    if status == "binary_artifact":
        return "binary_artifact"
    if status == "weak_source":
        return "weak_source_version"
    if status == "no_source_mapping":
        return "no_source_mapping"
    if source_error == "binary_artifact_no_source":
        return "binary_artifact_no_source"
    if "source download exceeds configured maximum" in text:
        return "source_download_too_large"
    if (
        source.get("error") == "recipe_not_found"
        or source.get("recipe_selection_reason") == "recipe_not_found"
        or ("fatal: path 'recipes/" in text and "does not exist" in text)
    ):
        return "recipe_not_found"
    if "404" in text or "not found" in text:
        if _source_has_untried_fallbacks(source):
            return "source_404"
        return "source_404_exhausted"
    if "{{" in text or "{%" in text or "strictundefined" in text or "undefined" in text:
        return "source_template_unresolved"
    if status in {"source_error", "missing_checkout", "blank_source_url"}:
        return "source_unavailable"
    return status


def _source_has_untried_fallbacks(source: dict) -> bool:
    source_url = str(source.get("source_url", "") or "").strip()
    if not source_url:
        return False
    fallback_urls = [url for url, _ in _failed_source_fallback_candidates(source_url)]
    if not fallback_urls:
        return False
    attempts = source.get("source_attempts", []) or []
    if not isinstance(attempts, list) or not attempts:
        return True
    tried_urls = {
        str(attempt.get("source_url", "") or "").strip()
        for attempt in attempts
        if isinstance(attempt, dict)
    }
    tried_urls.add(source_url)
    return any(url not in tried_urls for url in fallback_urls)


def _container_issue_dedupe_signature(event: dict, category: str) -> tuple[str, ...]:
    status = str(event.get("status", "") or "")
    command = str(event.get("command", "") or "")
    primary = str(event.get("primary_command", "") or _command_primary(command))
    text = _single_line_text(
        str(event.get("error_text", "") or event.get("stderr", "") or event.get("stdout", "")),
        limit=500,
    )
    if category in {"api_import_failure", "runtime_dependency_missing"}:
        module_match = re.search(
            r"(ModuleNotFoundError|ImportError|IndexError|NameError|OSError|RuntimeError)"
            r"[^:\n]*(?::\s*([^;]+))?",
            text,
        )
        root = module_match.group(0) if module_match else text
        return (category, _normalized_command_key(primary), root)
    if status == "container-command-missing":
        return (status, _normalized_command_key(primary), text)
    if category == "usable_nonhelp_output":
        return (status, category, _normalized_command_key(primary), text)
    return (status, category, command, text)


def _record_has_container_version_mismatch(record: dict) -> bool:
    version_consistency = record.get("version_consistency", {})
    if not isinstance(version_consistency, dict):
        return False
    issues = version_consistency.get("issues", []) or []
    return any(str(issue).startswith("container_version_mismatch:") for issue in issues)


def _record_has_runtime_api_validation(record: dict) -> bool:
    validations = record.get("container_api_validation", []) or []
    executions = record.get("container_execution", []) or []
    return any(
        isinstance(event, dict)
        and str(event.get("status", "") or "") == "container-api-validation-ok"
        for event in (*validations, *executions)
    )


def _record_has_wrapper_or_source_context(record: dict) -> bool:
    if _record_has_runtime_api_validation(record):
        return True
    wrapper_summary = record.get("wrapper_source_summary", {})
    if isinstance(wrapper_summary, dict):
        context_keys = (
            "helper_api_call_count",
            "helper_command_doc_count",
            "helper_parameter_doc_count",
            "configfile_api_call_count",
            "configfile_command_doc_count",
            "configfile_parameter_doc_count",
        )
        if any(int(wrapper_summary.get(key, 0) or 0) > 0 for key in context_keys):
            return True
    for field_name in ("wrapper_helper_files", "wrapper_configfiles"):
        for item in record.get(field_name, []) or []:
            if not isinstance(item, dict):
                continue
            if (
                item.get("api_calls")
                or item.get("command_docs")
                or item.get("parameter_docs")
            ):
                return True
    for source in record.get("bioconda_sources", []) or []:
        if isinstance(source, dict) and source.get("source_command_docs"):
            return True
    return False


def _record_failure_items(record: dict) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    base = {
        "package_id": record.get("package_id", ""),
        "tool_id": record.get("tool_id", ""),
        "wrapper_path": record.get("wrapper_path", ""),
    }
    recovered_help_commands = {
        str(event.get("command", "") or "")
        for event in (record.get("container_execution", []) or [])
        if isinstance(event, dict)
        and str(event.get("status", "") or "") == "container-command-help"
        and str(event.get("command", "") or "")
    }
    recovered_runtime_bases = {
        _probe_command_base(
            str(event.get("command", "") or ""),
            str(event.get("primary_command", "") or ""),
        )
        for event in (record.get("container_execution", []) or [])
        if isinstance(event, dict)
        and str(event.get("status", "") or "")
        in {
            "container-command-help",
            "container-command-help-degraded",
            "container-command-usage-degraded",
        }
        and str(event.get("command", "") or "")
    }
    recovered_runtime_bases = {base for base in recovered_runtime_bases if base}
    record_has_runtime_context = bool(
        str(record.get("container_help_text", "") or "").strip()
        or str(record.get("container_usage_text", "") or "").strip()
        or any(
            isinstance(event, dict)
            and str(event.get("status", "") or "")
            in {
                "container-command-help",
                "container-command-help-degraded",
                "container-command-usage-degraded",
                "container-api-validation-ok",
            }
            for event in (record.get("container_execution", []) or [])
        )
    )
    record_has_wrapper_or_source_context = _record_has_wrapper_or_source_context(record)
    seen_container_issue_signatures: set[tuple[str, ...]] = set()
    for event in record.get("container_execution", []) or []:
        if not isinstance(event, dict):
            continue
        status = str(event.get("status", "") or "")
        if status not in _CONTAINER_ISSUE_STATUSES:
            continue
        command = str(event.get("command", "") or "")
        if command and command in recovered_help_commands:
            continue
        command_base = _probe_command_base(
            command,
            str(event.get("primary_command", "") or ""),
        )
        if (
            status
            not in {
                "container-command-help-degraded",
                "container-command-usage-degraded",
            }
            and command_base
            and command_base in recovered_runtime_bases
        ):
            continue
        category = _container_failure_category(event)
        if record_has_runtime_context and category in {"api_import_failure", "bad_probe_variant"}:
            continue
        signature = _container_issue_dedupe_signature(event, category)
        if signature in seen_container_issue_signatures:
            continue
        seen_container_issue_signatures.add(signature)
        severity = "hard" if status in _HARD_CONTAINER_ISSUE_STATUSES else "partial"
        if status == "container-command-missing" and _record_has_container_version_mismatch(record):
            category = "container_version_mismatch"
            severity = "partial"
        if category in {"bad_probe_variant", "helper_script_probe"}:
            severity = "partial"
        if category in {"bad_probe_variant", "helper_script_probe"} and (
            record_has_runtime_context or record_has_wrapper_or_source_context
        ):
            continue
        if category == "api_import_failure" and record_has_runtime_context:
            continue
        items.append(
            {
                **base,
                "kind": "container",
                "status": status,
                "category": category,
                "severity": severity,
                "command": command,
                "image": event.get("image", ""),
                "runtime": event.get("runtime", ""),
                "returncode": event.get("returncode", 0),
                "error_text": _single_line_text(
                    str(event.get("error_text", "") or ""), limit=500
                ),
            }
        )
    sources = record.get("bioconda_sources", []) or []
    if not isinstance(sources, list):
        sources = []
    if not sources and record.get("requirement_packages"):
        items.append(
            {
                **base,
                "kind": "source",
                "status": "no_source_mapping",
                "category": "no_source_mapping",
                "severity": "partial",
                "requirement_packages": record.get("requirement_packages", []),
            }
        )
    for source in sources:
        if not isinstance(source, dict):
            continue
        status = _source_mapping_status(source)
        if _source_status_is_usable(status):
            continue
        items.append(
            {
                **base,
                "kind": "source",
                "status": status,
                "category": _source_failure_category(source),
                "severity": "partial",
                "package": source.get("package", ""),
                "required_version": source.get("required_version", ""),
                "source_channel": source.get("source_channel", ""),
                "source_url": source.get("source_url", ""),
                "recipe_version": source.get("recipe_version", ""),
                "recipe_selection_reason": source.get("recipe_selection_reason", ""),
                "source_confidence": source.get("source_confidence", ""),
                "source_version_match": source.get("source_version_match", ""),
                "source_fallback_reason": source.get("source_fallback_reason", ""),
                "fallback_from_channel": source.get("fallback_from_channel", ""),
                "source_provider_package": source.get("source_provider_package", ""),
                "source_error": _single_line_text(
                    str(source.get("source_error", "") or ""), limit=500
                ),
            }
        )
    if (
        not str(record.get("container_help_text", "") or "").strip()
        and not str(record.get("container_usage_text", "") or "").strip()
        and record.get("container_execution")
        and not record_has_wrapper_or_source_context
    ):
        items.append(
            {
                **base,
                "kind": "coverage",
                "status": "no_container_help",
                "category": "no_container_help",
                "severity": "partial",
            }
        )
    return items


_RETRYABLE_FAILURE_CATEGORIES = {
    "api_import_failure",
    "bad_probe_variant",
    "container_prepare_failed",
    "container_prepare_timeout",
    "container_version_mismatch",
    "missing_executable",
    "runtime_dependency_missing",
    "source_404",
    "source_template_unresolved",
    "source_unavailable",
    "terminal_unknown",
    "unresolved_template",
}


def _failure_item_is_retryable(item: dict[str, object]) -> bool:
    category = str(item.get("category", "") or "")
    if category in _RETRYABLE_FAILURE_CATEGORIES:
        return True
    return str(item.get("severity", "") or "") == "hard" and category not in {
        "binary_artifact",
        "no_container_help",
        "usable_nonzero_help",
        "weak_source_version",
    }


def _failure_inventory_wrapper_key(record: dict) -> str:
    wrapper_path = str(record.get("wrapper_path", "") or "")
    return wrapper_path or f"{record.get('package_id', '')}:{record.get('tool_id', '')}"


def _update_failure_inventory_wrapper(
    wrappers: dict[str, dict[str, object]],
    record: dict,
    record_items: list[dict[str, object]],
) -> None:
    wrapper_path = str(record.get("wrapper_path", "") or "")
    wrapper = wrappers.setdefault(
        _failure_inventory_wrapper_key(record),
        {
            "package_id": record.get("package_id", ""),
            "tool_id": record.get("tool_id", ""),
            "wrapper_path": wrapper_path,
            "categories": [],
            "statuses": [],
            "severity": "partial",
        },
    )
    categories = set(wrapper["categories"])
    statuses = set(wrapper["statuses"])
    for item in record_items:
        categories.add(str(item.get("category", "") or "terminal_unknown"))
        statuses.add(str(item.get("status", "") or "unknown"))
        if item.get("severity") == "hard":
            wrapper["severity"] = "hard"
    wrapper["categories"] = sorted(categories)
    wrapper["statuses"] = sorted(statuses)


def _sorted_failure_wrappers(wrappers: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        wrappers.values(),
        key=lambda item: (
            str(item.get("severity", "")) != "hard",
            str(item.get("package_id", "")),
            str(item.get("tool_id", "")),
            str(item.get("wrapper_path", "")),
        ),
    )


def _build_failure_inventory(records: list[dict]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    wrappers: dict[str, dict[str, object]] = {}
    retry_wrappers_by_key: dict[str, dict[str, object]] = {}
    category_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    package_counts: dict[str, int] = {}
    samples_by_category: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        record_items = _record_failure_items(record)
        if not record_items:
            continue
        items.extend(record_items)
        _update_failure_inventory_wrapper(wrappers, record, record_items)
        retry_items = [item for item in record_items if _failure_item_is_retryable(item)]
        if retry_items:
            _update_failure_inventory_wrapper(retry_wrappers_by_key, record, retry_items)
        for item in record_items:
            category = str(item.get("category", "") or "terminal_unknown")
            status = str(item.get("status", "") or "unknown")
            package_id = str(item.get("package_id", "") or "")
            category_counts[category] = category_counts.get(category, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            if package_id:
                package_counts[package_id] = package_counts.get(package_id, 0) + 1
            samples = samples_by_category.setdefault(category, [])
            if len(samples) < 10:
                samples.append(item)
    issue_wrappers = _sorted_failure_wrappers(wrappers)
    retry_wrappers = _sorted_failure_wrappers(retry_wrappers_by_key)
    return {
        "schema_version": "0.1.0",
        "summary": {
            "total_records": len(records),
            "issue_events": len(items),
            "issue_wrappers": len(issue_wrappers),
            "hard_wrappers": sum(1 for item in issue_wrappers if item.get("severity") == "hard"),
            "partial_wrappers": sum(
                1 for item in issue_wrappers if item.get("severity") != "hard"
            ),
            "retry_wrappers": len(retry_wrappers),
            "retry_hard_wrappers": sum(
                1 for item in retry_wrappers if item.get("severity") == "hard"
            ),
            "retry_partial_wrappers": sum(
                1 for item in retry_wrappers if item.get("severity") != "hard"
            ),
            "category_counts": dict(
                sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "status_counts": dict(
                sorted(status_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "package_counts": dict(
                sorted(package_counts.items(), key=lambda item: (-item[1], item[0]))[:50]
            ),
        },
        "items": items,
        "wrappers": issue_wrappers,
        "samples_by_category": samples_by_category,
        "retry_manifest": {
            "schema_version": "0.1.0",
            "wrappers": retry_wrappers,
        },
    }


def _write_recovery_summary_markdown(path: Path, inventory: dict[str, object]) -> None:
    summary = inventory.get("summary", {}) if isinstance(inventory, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    category_counts = summary.get("category_counts", {})
    status_counts = summary.get("status_counts", {})
    package_counts = summary.get("package_counts", {})
    lines = [
        "# Corpus Recovery Summary",
        "",
        f"- Total records: {summary.get('total_records', 0)}",
        f"- Issue events: {summary.get('issue_events', 0)}",
        f"- Issue wrappers: {summary.get('issue_wrappers', 0)}",
        f"- Hard wrappers: {summary.get('hard_wrappers', 0)}",
        f"- Partial wrappers: {summary.get('partial_wrappers', 0)}",
        f"- Retry wrappers: {summary.get('retry_wrappers', 0)}",
        f"- Retry hard wrappers: {summary.get('retry_hard_wrappers', 0)}",
        f"- Retry partial wrappers: {summary.get('retry_partial_wrappers', 0)}",
        "",
        "## Root Cause Categories",
    ]
    if isinstance(category_counts, dict):
        lines.extend(f"- {key}: {value}" for key, value in category_counts.items())
    lines.append("")
    lines.append("## Status Counts")
    if isinstance(status_counts, dict):
        lines.extend(f"- {key}: {value}" for key, value in status_counts.items())
    lines.append("")
    lines.append("## Top Packages")
    if isinstance(package_counts, dict):
        lines.extend(f"- {key}: {value}" for key, value in package_counts.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_artifacts(
    *,
    records: list[ToolRecord] | list[dict],
    diagnostics_dir: Path | None = None,
    retry_manifest_path: Path | None = None,
) -> dict[str, str]:
    payload_records = [asdict(record) if isinstance(record, ToolRecord) else record for record in records]
    inventory = _build_failure_inventory(payload_records)
    paths: dict[str, str] = {}
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = diagnostics_dir / "failure-inventory.json"
        samples_path = diagnostics_dir / "failure-samples.json"
        retry_path = diagnostics_dir / "retry-manifest.json"
        summary_path = diagnostics_dir / "recovery-summary.md"
        inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
        samples_path.write_text(
            json.dumps(inventory.get("samples_by_category", {}), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        retry_path.write_text(
            json.dumps(inventory.get("retry_manifest", {}), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_recovery_summary_markdown(summary_path, inventory)
        paths.update(
            {
                "failure_inventory_path": str(inventory_path),
                "failure_samples_path": str(samples_path),
                "retry_manifest_path": str(retry_path),
                "recovery_summary_path": str(summary_path),
            }
        )
    if retry_manifest_path is not None:
        retry_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        retry_manifest_path.write_text(
            json.dumps(inventory.get("retry_manifest", {}), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["requested_retry_manifest_path"] = str(retry_manifest_path)
    return paths


def write_corpus_diagnostics(
    *,
    execution_report_path: Path,
    diagnostics_dir: Path,
    corpus_jsonl: Path | None = None,
    checkpoint_file: Path | None = None,
    current_run_path: Path | None = None,
    sample_limit: int = 100,
) -> dict[str, object]:
    execution_report_path = execution_report_path.resolve()
    corpus_jsonl = (
        corpus_jsonl or _default_corpus_jsonl_for_execution_report(execution_report_path)
    ).resolve()
    checkpoint_file = (
        checkpoint_file or _default_checkpoint_for_execution_report(execution_report_path)
    ).resolve()
    current_run_path = (current_run_path or execution_report_path.parent / "current").resolve()
    diagnostics_dir = diagnostics_dir.resolve()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(execution_report_path.read_text(encoding="utf-8"))
    records = report.get("records", []) if isinstance(report, dict) else []
    if not isinstance(records, list):
        records = []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    if not isinstance(summary, dict):
        summary = {}

    status_counts: dict[str, int] = {}
    nonhelp_samples: list[dict[str, object]] = []
    records_with_container_help = 0
    records_with_container_usage = 0
    records_with_api_validation = 0
    records_with_api_validation_ok = 0
    api_backed_records = 0
    configfile_doc_records = 0
    source_command_doc_records = 0
    records_with_source_provider = 0
    source_provider_package_counts: dict[str, int] = {}
    source_provider_reason_counts: dict[str, int] = {}
    records_with_source_errors = 0
    source_error_counts: dict[str, int] = {}
    records_with_container_execution = 0
    unresolved_records = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        wrapper_summary = record.get("wrapper_source_summary", {})
        if not isinstance(wrapper_summary, dict):
            wrapper_summary = {}
        if str(record.get("container_help_text", "") or "").strip():
            records_with_container_help += 1
        if str(record.get("container_usage_text", "") or "").strip():
            records_with_container_usage += 1
        if wrapper_summary.get("api_backed_wrapper"):
            api_backed_records += 1
        if int(wrapper_summary.get("configfile_command_doc_count", 0) or 0) > 0:
            configfile_doc_records += 1
        api_validation = record.get("container_api_validation", []) or []
        if isinstance(api_validation, list) and api_validation:
            records_with_api_validation += 1
            if any(
                isinstance(event, dict)
                and event.get("status") == "container-api-validation-ok"
                for event in api_validation
            ):
                records_with_api_validation_ok += 1
        sources = record.get("bioconda_sources", []) or []
        record_has_source_provider = False
        record_has_source_error = False
        record_has_source_command_docs = False
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict):
                    continue
                if source.get("source_command_docs"):
                    record_has_source_command_docs = True
                provider_package = str(source.get("source_provider_package", "") or "").strip()
                provider_reason = str(source.get("source_provider_reason", "") or "").strip()
                if provider_package:
                    record_has_source_provider = True
                    source_provider_package_counts[provider_package] = (
                        source_provider_package_counts.get(provider_package, 0) + 1
                    )
                if provider_reason:
                    source_provider_reason_counts[provider_reason] = (
                        source_provider_reason_counts.get(provider_reason, 0) + 1
                    )
                source_error = str(source.get("source_error", "") or "").strip()
                if source_error:
                    record_has_source_error = True
                    error_bucket = source_error[:200]
                    source_error_counts[error_bucket] = source_error_counts.get(error_bucket, 0) + 1
        if record_has_source_provider:
            records_with_source_provider += 1
        if record_has_source_command_docs:
            source_command_doc_records += 1
        if record_has_source_error:
            records_with_source_errors += 1
        events = record.get("container_execution", []) or []
        if isinstance(events, list) and events:
            records_with_container_execution += 1
        has_nonhelp = False
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                status = str(event.get("status", "") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "container-command-nonhelp":
                    has_nonhelp = True
        if has_nonhelp and len(nonhelp_samples) < sample_limit:
            nonhelp_samples.append(
                {
                    "tool_id": record.get("tool_id", ""),
                    "package_id": record.get("package_id", ""),
                    "wrapper_path": record.get("wrapper_path", ""),
                    "container_help_text": record.get("container_help_text", ""),
                    "container_usage_text": record.get("container_usage_text", ""),
                    "events": events,
                }
            )
        if not (
            str(record.get("container_help_text", "") or "").strip()
            or str(record.get("container_usage_text", "") or "").strip()
            or (
                isinstance(api_validation, list)
                and any(
                    isinstance(event, dict)
                    and event.get("status") == "container-api-validation-ok"
                    for event in api_validation
                )
            )
            or int(wrapper_summary.get("configfile_command_doc_count", 0) or 0) > 0
            or record_has_source_command_docs
        ):
            unresolved_records += 1

    execution_summary_path = diagnostics_dir / "execution-summary.json"
    execution_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    status_counts_path = diagnostics_dir / "container-status-counts.txt"
    status_counts_lines = [
        f"{count:8d} {status}"
        for status, count in sorted(status_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    status_counts_path.write_text(
        "\n".join(status_counts_lines) + ("\n" if status_counts_lines else ""), encoding="utf-8"
    )

    coverage = {
        "total_records": len(records),
        "records_with_container_help": records_with_container_help,
        "records_without_container_help": len(records) - records_with_container_help,
        "records_with_container_usage": records_with_container_usage,
        "records_with_api_validation": records_with_api_validation,
        "records_with_api_validation_ok": records_with_api_validation_ok,
        "api_backed_records": api_backed_records,
        "configfile_doc_records": configfile_doc_records,
        "source_command_doc_records": source_command_doc_records,
        "records_with_source_provider": records_with_source_provider,
        "source_provider_package_counts": dict(
            sorted(source_provider_package_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "source_provider_reason_counts": dict(
            sorted(source_provider_reason_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "records_with_source_errors": records_with_source_errors,
        "source_error_counts": dict(
            sorted(source_error_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "records_resolved_by_help_or_usage_or_api_or_docs": (
            len(records) - unresolved_records
        ),
        "records_unresolved_by_help_or_usage_or_api_or_docs": unresolved_records,
        "records_with_container_execution": records_with_container_execution,
        "records_without_container_execution": len(records) - records_with_container_execution,
    }
    coverage_path = diagnostics_dir / "container-help-coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")

    nonhelp_samples_path = diagnostics_dir / "nonhelp-samples.json"
    nonhelp_samples_path.write_text(
        json.dumps(nonhelp_samples, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    source_coverage, source_missing = _source_diagnostics(records)
    source_coverage_path = diagnostics_dir / "source-coverage.json"
    source_coverage_path.write_text(
        json.dumps(source_coverage, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_missing_path = diagnostics_dir / "source-missing.json"
    source_missing_path.write_text(
        json.dumps(source_missing, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    jsonl_integrity = _jsonl_integrity(corpus_jsonl)
    checkpoint_integrity = _line_count_integrity(checkpoint_file)
    current_run = ""
    if current_run_path.exists():
        current_run = current_run_path.read_text(encoding="utf-8").strip()
    report_count = len(records)
    jsonl_count = int(jsonl_integrity.get("line_count", 0) or 0)
    checkpoint_count = int(checkpoint_integrity.get("line_count", 0) or 0)
    integrity = {
        "execution_report": {
            "path": str(execution_report_path),
            "exists": execution_report_path.exists(),
            "record_count": report_count,
        },
        "corpus_jsonl": jsonl_integrity,
        "checkpoint": checkpoint_integrity,
        "current_run": {
            "path": str(current_run_path),
            "exists": current_run_path.exists(),
            "value": current_run,
        },
        "counts_consistent": (
            bool(jsonl_integrity.get("valid_json"))
            and report_count > 0
            and report_count == jsonl_count == checkpoint_count
        ),
    }
    integrity_path = diagnostics_dir / "integrity.json"
    integrity_path.write_text(json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8")
    failure_artifacts = _write_failure_artifacts(
        records=records,
        diagnostics_dir=diagnostics_dir,
    )

    return {
        "diagnostics_dir": str(diagnostics_dir),
        "execution_summary_path": str(execution_summary_path),
        "container_status_counts_path": str(status_counts_path),
        "container_help_coverage_path": str(coverage_path),
        "nonhelp_samples_path": str(nonhelp_samples_path),
        "source_coverage_path": str(source_coverage_path),
        "source_missing_path": str(source_missing_path),
        "integrity_path": str(integrity_path),
        **failure_artifacts,
        "summary": {
            "total_records": len(records),
            "records_with_container_help": records_with_container_help,
            "records_with_container_usage": records_with_container_usage,
            "records_with_api_validation_ok": records_with_api_validation_ok,
            "records_unresolved_by_help_or_usage_or_api_or_docs": unresolved_records,
            "nonhelp_sample_count": len(nonhelp_samples),
            "counts_consistent": integrity["counts_consistent"],
        },
    }


def _execute_container_help_for_record(
    record: ToolRecord,
    settings: ExtractionSettings,
    state: ContainerExecutionState,
) -> None:
    if not settings.execute_containers:
        record.help_text = _combine_help_text(
            record.original_help_text or record.help_text,
            record.container_help_text,
            record.container_usage_text,
            record.container_api_validation,
        )
        return
    if record.container_help_text.strip():
        record.help_text = _combine_help_text(
            record.original_help_text or record.help_text,
            record.container_help_text,
            record.container_usage_text,
            record.container_api_validation,
        )
        return

    record_status_context = {
        "package_id": record.package_id,
        "tool_id": record.tool_id,
        "wrapper_path": record.wrapper_path,
    }

    def prepare_candidate(
        image: str, candidate: dict, candidate_rank: int
    ) -> ContainerPreparation | None:
        cache_key = _container_preparation_cache_key(
            image,
            runtimes=state.runtimes,
            settings=settings,
        )
        lock_key = cache_key or image
        quarantine_key = cache_key or image
        with state.preparation_locks_guard:
            image_lock = state.preparation_locks.setdefault(lock_key, threading.Lock())
        with image_lock:
            if cache_key in state.preparation_alias_cache:
                return state.preparation_alias_cache[cache_key]
            if image in state.preparation_cache:
                cached = state.preparation_cache[image]
                if cache_key:
                    state.preparation_alias_cache[cache_key] = cached
                return cached
            if not _container_cache_satisfies_preparation(
                image,
                runtimes=state.runtimes,
                settings=settings,
            ):
                quarantined = _container_image_quarantine_get(
                    state.image_quarantine_store,
                    quarantine_key,
                    settings,
                )
                if quarantined is not None:
                    last_failure = _container_quarantine_failure(quarantined, image)
                    state.failed_preparation_cache[image] = last_failure
                    _emit_extract_status_once(
                        settings,
                        ("container-image-quarantined", quarantine_key),
                        {
                            "status": "container-image-quarantined",
                            **record_status_context,
                            "runtime": last_failure.runtime,
                            "image": image,
                            "candidate_source": candidate.get("source", ""),
                            "candidate_rank": candidate_rank,
                            "quarantine_seconds": int(
                                quarantined.get("quarantine_seconds", 0) or 0
                            ),
                            "expires_at": quarantined.get("expires_at", 0),
                            "reason": quarantined.get("reason", "container image quarantined"),
                            "returncode": last_failure.returncode,
                            "error_text": last_failure.error_text,
                        },
                    )
                    return None
            if image in state.failed_preparation_cache:
                failed_at = state.failed_preparation_at.get(image, 0.0)
                quarantine_seconds = max(0, int(settings.container_image_quarantine_seconds))
                if not quarantine_seconds or time.monotonic() - failed_at < quarantine_seconds:
                    _emit_extract_status_once(
                        settings,
                        ("container-image-quarantined", image),
                        {
                            "status": "container-image-quarantined",
                            **record_status_context,
                            "image": image,
                            "candidate_source": candidate.get("source", ""),
                            "candidate_rank": candidate_rank,
                            "quarantine_seconds": quarantine_seconds,
                            "error_text": state.failed_preparation_cache[image].error_text,
                        },
                    )
                    return None
                state.failed_preparation_cache.pop(image, None)
                state.failed_preparation_at.pop(image, None)

            failed_preparations: list[ContainerPreparation] = []
            for runtime in state.runtimes:
                if state.preparation_semaphore is not None:
                    state.preparation_semaphore.acquire()
                try:
                    attempted = _prepare_container(image=image, runtime=runtime, settings=settings)
                finally:
                    if state.preparation_semaphore is not None:
                        state.preparation_semaphore.release()
                _emit_extract_status(
                    settings,
                    {
                        "status": "container-prepare",
                        **record_status_context,
                        "runtime": attempted.runtime,
                        "image": image,
                        "source": attempted.source,
                        "candidate_source": candidate.get("source", ""),
                        "candidate_rank": candidate_rank,
                        "identifier": attempted.identifier,
                        "returncode": attempted.returncode,
                        "jobs": 1,
                        "error_text": attempted.error_text,
                    },
                )
                if attempted.ok:
                    state.preparation_cache[image] = attempted
                    if cache_key:
                        state.preparation_alias_cache[cache_key] = attempted
                    if failed_preparations:
                        state.runtime_fallbacks += 1
                    state.images_prepared += 1
                    if attempted.source == "docker-pull":
                        state.images_pulled += 1
                    return attempted
                failed_preparations.append(attempted)
                if _container_preparation_timed_out(attempted):
                    state.failed_preparation_cache[image] = attempted
                    state.failed_preparation_at[image] = time.monotonic()
                    _container_image_quarantine_put(
                        state.image_quarantine_store,
                        quarantine_key,
                        attempted,
                        settings,
                        reason="container preparation timed out",
                    )
                    _emit_extract_status_once(
                        settings,
                        ("container-image-quarantined", quarantine_key),
                        {
                            "status": "container-image-quarantined",
                            **record_status_context,
                            "runtime": attempted.runtime,
                            "image": image,
                            "candidate_source": candidate.get("source", ""),
                            "candidate_rank": candidate_rank,
                            "quarantine_seconds": max(
                                0, int(settings.container_image_quarantine_seconds)
                            ),
                            "reason": "container preparation timed out",
                            "returncode": attempted.returncode,
                            "error_text": attempted.error_text,
                        },
                    )
                    return None

            state.failed_preparation_cache[image] = (
                failed_preparations[-1]
                if failed_preparations
                else ContainerPreparation(
                    ok=False,
                    runtime=settings.container_runtime,
                    image=image,
                    error_text="container preparation failed",
                )
            )
            state.failed_preparation_at[image] = time.monotonic()
            return None

    def candidate_commands() -> list[tuple[str, list[dict], list[dict], dict, int]]:
        planned: list[tuple[str, list[dict], list[dict], dict, int]] = []
        for rank, candidate in enumerate(
            _sort_container_candidates_for_selection(
                _legacy_container_candidate_details(record),
                requirement_packages=record.requirement_packages,
                requirement_versions=record.requirement_versions,
            ),
            start=1,
        ):
            raw_image = str(candidate.get("image", "") or "")
            if candidate.get("status", "ok") != "ok":
                _emit_extract_status(
                    settings,
                    {
                        "status": "container-candidate-skipped",
                        **record_status_context,
                        "image": raw_image,
                        "source": candidate.get("source", ""),
                        "candidate_rank": rank,
                        "reason": candidate.get("error_text", "") or "candidate marked skipped",
                    },
                )
                continue
            image = _normalize_container_candidate(raw_image)
            if raw_image and not image:
                _emit_extract_status(
                    settings,
                    {
                        "status": "container-candidate-skipped",
                        **record_status_context,
                        "image": raw_image,
                        "source": candidate.get("source", ""),
                        "candidate_rank": rank,
                        "reason": _container_ref_error(_normalize_container_ref(raw_image)),
                    },
                )
                continue
            if not image:
                continue
            packages = candidate.get("packages", []) or []
            commands = _record_help_command_plan(
                record,
                image,
                probe_mode=settings.container_help_probe_mode,
                candidate_packages=packages,
            )
            api_commands = [] if commands else _record_api_validation_commands(record)
            if commands or api_commands:
                state.planned_images.add(image)
                planned.append((image, commands, api_commands, candidate, rank))
        return planned

    candidates = candidate_commands()
    if not candidates:
        record.help_text = _combine_help_text(
            record.original_help_text or record.help_text,
            record.container_help_text,
            record.container_usage_text,
            record.container_api_validation,
        )
        return

    if not state.runtimes:
        image, commands, api_commands, candidate, rank = candidates[0]
        first_command = (
            str(commands[0].get("command", ""))
            if commands
            else str(api_commands[0].get("command", ""))
        )
        record.container_execution.append(
            {
                "phase": "prepare",
                "runtime": settings.container_runtime,
                "image": image,
                "candidate_source": candidate.get("source", ""),
                "candidate_rank": rank,
                "command": first_command,
                "returncode": 127,
                "stdout": "",
                "stderr": "",
                "error_text": f"No available container runtime for {settings.container_runtime}",
            }
        )
        state.commands_failed += 1
        state.runtime_error += 1
        record.help_text = _combine_help_text(
            record.original_help_text or record.help_text,
            record.container_help_text,
            record.container_usage_text,
            record.container_api_validation,
        )
        return

    probed_preparation_keys: set[str] = set()
    for image, commands, api_commands, candidate, rank in candidates:
        if record.container_help_text.strip() or record.container_usage_text.strip():
            break

        preparation = prepare_candidate(image, candidate, rank)
        if preparation is None:
            last_failure = state.failed_preparation_cache.get(image) or ContainerPreparation(
                ok=False,
                runtime=settings.container_runtime,
                image=image,
                error_text="container preparation failed",
            )
            record.container_execution.append(
                {
                    "phase": "prepare",
                    "runtime": last_failure.runtime,
                    "image": image,
                    "candidate_source": candidate.get("source", ""),
                    "candidate_rank": rank,
                    "status": "container-prepare-failed",
                    "command": str(commands[0].get("command", ""))
                    if commands
                    else str(api_commands[0].get("command", "")),
                    "returncode": last_failure.returncode,
                    "stdout": last_failure.stdout,
                    "stderr": last_failure.stderr,
                    "error_text": last_failure.error_text,
                }
            )
            state.commands_failed += 1
            state.prepare_failed += 1
            continue

        prepared_probe_key = f"{preparation.runtime}:{preparation.identifier or preparation.image}"
        if prepared_probe_key in probed_preparation_keys:
            _emit_extract_status(
                settings,
                {
                    "status": "container-candidate-skipped",
                    **record_status_context,
                    "runtime": preparation.runtime,
                    "image": image,
                    "identifier": preparation.identifier,
                    "candidate_source": candidate.get("source", ""),
                    "candidate_rank": rank,
                    "reason": "duplicate prepared container image already probed",
                },
            )
            continue
        probed_preparation_keys.add(prepared_probe_key)

        if api_commands and not commands:
            for api_command in api_commands:
                command = str(api_command.get("command", "") or "")
                primary = _command_primary(command)
                if primary:
                    presence_key = _container_command_presence_key(
                        image, primary, preparation
                    )
                    presence_result = _cached_container_command_exists(
                        state=state,
                        image=image,
                        primary=primary,
                        preparation=preparation,
                        settings=settings,
                    )
                    if presence_result.returncode != 0:
                        state.commands_failed += 1
                        state.missing_command += 1
                        error_text = (
                            _completed_error_text(presence_result)
                            or f"{primary}: command not found in container"
                        )
                        record.container_execution.append(
                            {
                                "phase": "preflight",
                                "runtime": preparation.runtime,
                                "image": image,
                                "identifier": preparation.identifier,
                                "source": preparation.source,
                                "candidate_source": candidate.get("source", ""),
                                "candidate_rank": rank,
                                "fallback_reason": "missing-api-runtime-command",
                                "status": "container-command-missing",
                                "command": command,
                                "primary_command": primary,
                                "returncode": presence_result.returncode,
                                "stdout": _tail_text(
                                    _strip_container_runtime_noise(presence_result.stdout or "")
                                ),
                                "stderr": _tail_text(
                                    _strip_container_runtime_noise(presence_result.stderr or "")
                                ),
                                "error_text": error_text,
                            }
                        )
                        continue

                result = _run_container_api_validation_probe(
                    preparation=preparation,
                    command=command,
                    settings=settings,
                )
                state.commands_executed += 1
                status = _container_api_validation_status(result)
                _record_probe_summary(state, status, result.returncode)
                record.selected_container = image
                record.selected_container_runtime = preparation.runtime
                validation_event = {
                    "phase": "api_validation",
                    "runtime": preparation.runtime,
                    "image": image,
                    "identifier": preparation.identifier,
                    "source": preparation.source,
                    "candidate_source": candidate.get("source", ""),
                    "candidate_rank": rank,
                    "status": status,
                    "probe_role": "api_validation",
                    "language": api_command.get("language", ""),
                    "check_count": api_command.get("check_count", 0),
                    "checks": api_command.get("checks", []),
                    "api_docs": _container_api_docs(result),
                    "api_errors": _container_api_errors(result),
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": _tail_text(_strip_container_runtime_noise(result.stdout or "")),
                    "stderr": _tail_text(_strip_container_runtime_noise(result.stderr or "")),
                    "error_text": "" if result.returncode == 0 else _completed_error_text(result),
                }
                record.container_api_validation.append(validation_event)
                record.container_execution.append(validation_event)
                _emit_extract_status(
                    settings,
                    {
                        "status": status,
                        **record_status_context,
                        "runtime": preparation.runtime,
                        "image": image,
                        "candidate_source": candidate.get("source", ""),
                        "candidate_rank": rank,
                        "language": api_command.get("language", ""),
                        "check_count": api_command.get("check_count", 0),
                        "returncode": result.returncode,
                        "error_text": validation_event["error_text"],
                    },
                )
                if status == "container-api-validation-ok":
                    break
            if any(
                event.get("status") == "container-api-validation-ok"
                for event in record.container_api_validation
            ):
                break
            continue

        missing_primaries_for_candidate: set[str] = set()
        argument_error_primaries_for_candidate: set[str] = set()
        runtime_error_bases_for_candidate: set[str] = set()
        for command_plan in commands:
            if record.container_help_text.strip() or record.container_usage_text.strip():
                break
            command = str(command_plan.get("command", "") or "")
            probe_role = str(command_plan.get("probe_role", "") or "core")
            primary = str(command_plan.get("primary", "") or "") or _command_primary(command)
            runtime_error_base = _probe_command_base(command, primary)
            if runtime_error_base in runtime_error_bases_for_candidate:
                continue
            if primary:
                if (
                    primary in missing_primaries_for_candidate
                    or primary in argument_error_primaries_for_candidate
                ):
                    continue
                presence_key = _container_command_presence_key(image, primary, preparation)
                presence_result = _cached_container_command_exists(
                    state=state,
                    image=image,
                    primary=primary,
                    preparation=preparation,
                    settings=settings,
                )
                if presence_result.returncode != 0:
                    missing_primaries_for_candidate.add(primary)
                    state.commands_failed += 1
                    state.missing_command += 1
                    error_text = (
                        _completed_error_text(presence_result)
                        or f"{primary}: command not found in container"
                    )
                    if presence_key not in state.missing_command_status_emitted:
                        state.missing_command_status_emitted.add(presence_key)
                        _emit_extract_status(
                            settings,
                            {
                                "status": "container-command-missing",
                                **record_status_context,
                                "runtime": preparation.runtime,
                                "image": image,
                                "candidate_source": candidate.get("source", ""),
                                "candidate_rank": rank,
                                "fallback_reason": "missing-command",
                                "command": command,
                                "probe_role": probe_role,
                                "primary_command": primary,
                                "returncode": presence_result.returncode,
                                "error_text": error_text,
                            },
                        )
                    record.selected_container_runtime = preparation.runtime
                    record.container_execution.append(
                        {
                            "phase": "preflight",
                            "runtime": preparation.runtime,
                            "image": image,
                            "identifier": preparation.identifier,
                            "source": preparation.source,
                            "candidate_source": candidate.get("source", ""),
                            "candidate_rank": rank,
                            "fallback_reason": "missing-command",
                            "status": "container-command-missing",
                            "command": command,
                            "probe_role": probe_role,
                            "primary_command": primary,
                            "returncode": presence_result.returncode,
                            "stdout": _tail_text(
                                _strip_container_runtime_noise(presence_result.stdout or "")
                            ),
                            "stderr": _tail_text(
                                _strip_container_runtime_noise(presence_result.stderr or "")
                            ),
                            "error_text": error_text,
                        }
                    )
                    continue

            result = _run_container_probe(
                preparation=preparation, command=command, settings=settings
            )
            state.commands_executed += 1
            fragment = _container_help_fragment(command, result)
            usage_fragment = "" if fragment else _container_usage_fragment(command, result)
            probe_status = _container_probe_status(result, fragment, usage_fragment)
            record.selected_container = image
            _record_probe_summary(state, probe_status, result.returncode)
            if probe_status != "container-command-help":
                _emit_extract_status(
                    settings,
                    {
                        "status": probe_status,
                        **record_status_context,
                        "runtime": preparation.runtime,
                        "image": image,
                        "candidate_source": candidate.get("source", ""),
                        "candidate_rank": rank,
                        "command": command,
                        "probe_role": probe_role,
                        "returncode": result.returncode,
                        "error_text": _completed_error_text(result),
                    },
                )
            if fragment:
                record.container_help_text = "\n\n".join(
                    part for part in (record.container_help_text.strip(), fragment) if part
                )
                record.help_text = _combine_help_text(
                    record.original_help_text or record.help_text,
                    record.container_help_text,
                    record.container_usage_text,
                    record.container_api_validation,
                )
            elif usage_fragment:
                record.container_usage_text = "\n\n".join(
                    part
                    for part in (record.container_usage_text.strip(), usage_fragment)
                    if part
                )
                record.help_text = _combine_help_text(
                    record.original_help_text or record.help_text,
                    record.container_help_text,
                    record.container_usage_text,
                    record.container_api_validation,
                )
            elif _is_missing_command_probe(result):
                if primary:
                    missing_primaries_for_candidate.add(primary)
            elif _looks_like_missing_argument_traceback(_combined_container_text(result)):
                if primary:
                    argument_error_primaries_for_candidate.add(primary)
            elif _looks_like_runtime_import_traceback(_combined_container_text(result)):
                if runtime_error_base:
                    runtime_error_bases_for_candidate.add(runtime_error_base)
            record.selected_container_runtime = preparation.runtime
            record.container_execution.append(
                {
                    "phase": "run",
                    "runtime": preparation.runtime,
                    "image": image,
                    "identifier": preparation.identifier,
                    "source": preparation.source,
                    "candidate_source": candidate.get("source", ""),
                    "candidate_rank": rank,
                    "status": probe_status,
                    "command": command,
                    "probe_role": probe_role,
                    "returncode": result.returncode,
                    "stdout": _tail_text(_strip_container_runtime_noise(result.stdout or "")),
                    "stderr": _tail_text(_strip_container_runtime_noise(result.stderr or "")),
                    "error_text": "" if result.returncode == 0 else _completed_error_text(result),
                }
            )

        if record.container_help_text.strip() or record.container_usage_text.strip():
            break

    record.help_text = _combine_help_text(
        record.original_help_text or record.help_text,
        record.container_help_text,
        record.container_usage_text,
        record.container_api_validation,
    )


def _finalize_container_execution_state(
    state: ContainerExecutionState,
    settings: ExtractionSettings,
) -> dict[str, int]:
    if settings.execute_containers:
        for preparation in state.preparation_cache.values():
            if preparation.runtime == "docker" and settings.remove_images_after_use:
                rm = _remove_prepared_container(preparation, settings=settings)
                state.images_removed += 1
                _emit_extract_status(
                    settings,
                    {
                        "status": "container-rm-image",
                        "image": preparation.identifier,
                        "returncode": rm.returncode,
                        "error_text": "" if rm.returncode == 0 else _completed_error_text(rm),
                    },
                )
    return _container_execution_summary(state)


def _execute_container_help_batches(
    records: list[ToolRecord], settings: ExtractionSettings
) -> dict[str, int]:
    state = _new_container_execution_state(settings)
    for record in records:
        _execute_container_help_for_record(record, settings, state)
    return _finalize_container_execution_state(state, settings)


def _enrich_record_with_container_help(
    record: ToolRecord,
    settings: ExtractionSettings,
    state: ContainerExecutionState,
) -> ToolRecord:
    _execute_container_help_for_record(record, settings, state)
    return record


def extract_tools_corpus(
    tools_root: Path,
    output_jsonl: Path,
    checkpoint_file: Path,
    settings: ExtractionSettings,
) -> dict[str, object]:
    run_id = _unique_run_id(output_jsonl.parent / "runs", settings.run_id or None)
    settings = replace(settings, run_id=run_id)
    restart_removed: list[str] = []
    restart_archived: list[str] = []
    restart_archive_dir = ""
    if settings.restart:
        restart_archived, restart_archive_dir = _archive_extract_outputs(
            output_jsonl, checkpoint_file, settings.run_id
        )
    if settings.restart:
        _emit_extract_status(
            settings,
            {
                "status": "extract-restart",
                "output": str(output_jsonl),
                "checkpoint": str(checkpoint_file),
                "removed": restart_removed,
                "archived": restart_archived,
                "archive_dir": restart_archive_dir,
            },
        )
    tool_dirs = discover_tool_dirs(tools_root)
    wrappers: list[tuple[Path, Path]] = []
    for tool_dir in tool_dirs:
        for xml_file in sorted(tool_dir.glob("*.xml")):
            lowered_name = xml_file.name.lower()
            if "macro" in lowered_name:
                continue
            if _xml_file_root_tag(xml_file) != "tool":
                continue
            wrappers.append((tool_dir, xml_file))

    processed = _load_checkpoint(checkpoint_file)
    pending = [
        (tool_dir, xml_file)
        for tool_dir, xml_file in wrappers
        if _checkpoint_key(tools_root=tools_root, tool_dir=tool_dir, xml_file=xml_file)
        not in processed
    ]
    retry_wrapper_paths = _load_retry_manifest_wrapper_paths(settings.retry_manifest_path)
    if retry_wrapper_paths:
        pending = [
            (tool_dir, xml_file)
            for tool_dir, xml_file in pending
            if str(xml_file.resolve()) in retry_wrapper_paths or str(xml_file) in retry_wrapper_paths
        ]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    expanded_root = output_jsonl.parent / "expanded"
    expanded_root.mkdir(parents=True, exist_ok=True)

    _emit_extract_status(
        settings,
        {
            "status": "extract-plan",
            "tools_root": str(tools_root),
            "total_tools": len(tool_dirs),
            "total_wrappers": len(wrappers),
            "already_processed": len(processed),
            "pending_wrappers": len(pending),
            "retry_manifest_filter_wrappers": len(retry_wrapper_paths),
            "max_workers": settings.max_workers,
            "source_workers": settings.source_workers or min(8, max(1, settings.max_workers)),
            "container_prepare_workers": settings.container_prepare_workers,
            "container_probe_workers": settings.container_probe_workers,
            "container_sif_exec_mode": _container_sif_exec_mode(settings),
            "container_image_quarantine_file": str(
                _container_image_quarantine_path(settings)
                or ""
            ),
            "http_user_agent": http_user_agent(),
            "http_browser_fallback_user_agent": browser_fallback_user_agent(),
            "fetch_documentation": settings.fetch_documentation,
            "resolve_containers": settings.resolve_containers,
            "execute_containers": settings.execute_containers,
            "bioconda_checkout_sources": settings.bioconda_checkout_sources,
        },
    )

    bioconda_recipes_repo: Path | None = None
    bioconda_recipes_repo_error = ""
    if settings.bioconda_checkout_sources:
        cache_root = settings.cache_root or Path(".gtsm-cache") / "source-cache"
        bioconda_recipes_repo, bioconda_recipes_repo_error = _ensure_bioconda_repo(
            cache_root,
            settings.bioconda_ref,
            settings,
        )

    execution_state = _new_container_execution_state(settings)
    bioconda_source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None = (
        {} if settings.bioconda_checkout_sources else None
    )
    bioconda_source_result_cache_lock = (
        threading.Lock() if settings.bioconda_checkout_sources else None
    )
    bioconda_recipe_selection_cache: (
        dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None
    ) = {} if settings.bioconda_checkout_sources else None
    bioconda_recipe_selection_cache_lock = (
        threading.Lock() if settings.bioconda_checkout_sources else None
    )
    if settings.execute_containers:
        _emit_extract_status(
            settings,
            {
                "status": "container-execution-start",
                "runtime": settings.container_runtime,
                "available_runtimes": [runtime.name for runtime in execution_state.runtimes],
                "pending_wrappers": len(pending),
            },
        )

    emitted: list[ToolRecord] = []
    written = 0
    started_at = time.monotonic()
    progress_interval = max(float(settings.extract_progress_interval_seconds), 1.0)
    source_workers = max(
        1,
        int(settings.source_workers or min(8, max(1, settings.max_workers))),
    )
    source_semaphore = (
        threading.Semaphore(source_workers) if settings.bioconda_checkout_sources else None
    )
    extract_workers = max(1, int(settings.max_workers or 1))
    container_workers = max(1, int(settings.container_probe_workers or 1))
    with (
        ThreadPoolExecutor(max_workers=extract_workers) as extract_executor,
        ThreadPoolExecutor(max_workers=container_workers) as container_executor,
    ):
        extract_futures = {
            extract_executor.submit(
                _with_retries,
                tools_root,
                tool_dir,
                xml_file,
                settings,
                expanded_root,
                bioconda_recipes_repo,
                bioconda_recipes_repo_error,
                bioconda_source_result_cache,
                bioconda_source_result_cache_lock,
                bioconda_recipe_selection_cache,
                bioconda_recipe_selection_cache_lock,
                source_semaphore,
            ): (tool_dir, xml_file)
            for tool_dir, xml_file in pending
        }
        remaining_extract = set(extract_futures)
        container_futures: dict[object, tuple[Path, Path]] = {}
        remaining_container: set[object] = set()

        def write_record(
            record: ToolRecord,
            tool_dir: Path,
            xml_file: Path,
            out,
            ck,
        ) -> None:
            nonlocal written
            out.write(record.to_json() + "\n")
            key = _checkpoint_key(tools_root=tools_root, tool_dir=tool_dir, xml_file=xml_file)
            ck.write(key + "\n")
            out.flush()
            ck.flush()
            emitted.append(record)
            written += 1
            _emit_extract_status(
                settings,
                {
                    "status": "extract-record-completed",
                    "package_id": record.package_id,
                    "tool_id": record.tool_id,
                    "wrapper_path": record.wrapper_path,
                    "processed_now": written,
                    "pending_wrappers": len(pending),
                    "remaining_wrappers": len(remaining_extract) + len(remaining_container),
                    "remaining_extraction": len(remaining_extract),
                    "remaining_container_enrichment": len(remaining_container),
                    "elapsed_seconds": round(time.monotonic() - started_at, 1),
                },
            )

        with (
            output_jsonl.open("a", encoding="utf-8") as out,
            checkpoint_file.open("a", encoding="utf-8") as ck,
        ):
            while remaining_extract or remaining_container:
                waiting = set(remaining_extract) | set(remaining_container)
                done, _ = wait(
                    waiting, timeout=progress_interval, return_when=FIRST_COMPLETED
                )
                if not done:
                    _emit_extract_status(
                        settings,
                        {
                            "status": "extract-progress",
                            "phase": "wrapper-extraction-container-enrichment",
                            "processed_now": written,
                            "pending_wrappers": len(pending),
                            "remaining_wrappers": len(remaining_extract)
                            + len(remaining_container),
                            "remaining_extraction": len(remaining_extract),
                            "remaining_container_enrichment": len(remaining_container),
                            "elapsed_seconds": round(time.monotonic() - started_at, 1),
                        },
                    )
                    continue
                for future in done:
                    if future in extract_futures:
                        remaining_extract.discard(future)
                        tool_dir, xml_file = extract_futures[future]
                        record = future.result()
                        if settings.execute_containers:
                            container_future = container_executor.submit(
                                _enrich_record_with_container_help,
                                record,
                                settings,
                                execution_state,
                            )
                            container_futures[container_future] = (tool_dir, xml_file)
                            remaining_container.add(container_future)
                        else:
                            _execute_container_help_for_record(
                                record, settings=settings, state=execution_state
                            )
                            write_record(record, tool_dir, xml_file, out, ck)
                        continue
                    remaining_container.discard(future)
                    tool_dir, xml_file = container_futures[future]
                    record = future.result()
                    write_record(record, tool_dir, xml_file, out, ck)

    finalized_summary = _finalize_container_execution_state(
        execution_state, settings=settings
    )
    all_records = _load_records_from_jsonl(output_jsonl)
    records_for_reports = all_records or emitted
    execution_summary = (
        _container_execution_summary_from_records(records_for_reports)
        if settings.execute_containers
        else finalized_summary
    )
    if settings.execute_containers:
        execution_summary["images_removed"] = finalized_summary.get(
            "images_removed", execution_summary.get("images_removed", 0)
        )
    execution_report_path = output_jsonl.with_suffix(".execution.json")
    execution_report_path.write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "summary": execution_summary,
                "records": [asdict(record) for record in records_for_reports],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    diagnostics_dir = output_jsonl.parent / "diagnostics" / settings.run_id
    retry_manifest_path = settings.retry_manifest_path or output_jsonl.with_suffix(
        ".retry-manifest.json"
    )
    failure_artifacts = _write_failure_artifacts(
        records=records_for_reports,
        diagnostics_dir=diagnostics_dir,
        retry_manifest_path=retry_manifest_path,
    )

    index_path = _write_dataset_index(output_jsonl=output_jsonl, records=records_for_reports)
    run_dir, current_run_path = _snapshot_completed_run(
        output_jsonl=output_jsonl,
        checkpoint_file=checkpoint_file,
        index_path=index_path,
        execution_report_path=execution_report_path,
        expanded_root=expanded_root,
        settings=settings,
        total_tools=len(tool_dirs),
        total_wrappers=len(wrappers),
        processed_now=written,
        execution_summary=execution_summary,
    )
    run_diagnostics_dir = Path(run_dir) / "diagnostics"
    if diagnostics_dir.exists():
        if run_diagnostics_dir.exists():
            shutil.rmtree(run_diagnostics_dir)
        shutil.copytree(diagnostics_dir, run_diagnostics_dir)
    _emit_extract_status(
        settings,
        {
            "status": "extract-completed",
            "processed_now": written,
            "total_records": len(records_for_reports),
            "elapsed_seconds": round(time.monotonic() - started_at, 1),
            "container_execution": execution_summary,
            "run_dir": run_dir,
            "current_run_path": current_run_path,
            **failure_artifacts,
        },
    )
    return {
        "run_id": settings.run_id,
        "run_dir": run_dir,
        "current_run_path": current_run_path,
        "total_tools": len(tool_dirs),
        "total_wrappers": len(wrappers),
        "already_processed": len(processed),
        "processed_now": written,
        "restart": settings.restart,
        "restart_removed": restart_removed,
        "restart_archived": restart_archived,
        "restart_archive_dir": restart_archive_dir,
        "index_path": str(index_path),
        "container_execution": execution_summary,
        "execution_report_path": str(execution_report_path),
        **failure_artifacts,
    }
