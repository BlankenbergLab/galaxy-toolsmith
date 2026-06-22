from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from pathlib import Path
from urllib import request as urlrequest
from urllib.parse import quote, urlparse
from xml.etree import ElementTree as ET

import requests
import yaml
from jinja2 import Environment, StrictUndefined, TemplateError
from galaxy.tool_util.loader import load_tool_with_refereces
from galaxy.util import xml_to_string

from galaxy_toolsmith.inference.datatypes import known_galaxy_datatypes
from galaxy_toolsmith.runtime.status import emit_status


@dataclass(frozen=True)
class ExtractionSettings:
    max_workers: int = 4
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
    container_help_probe_mode: str = "exploratory"
    container_no_arg_timeout_seconds: int = 20
    container_run_timeout_seconds: int = 120
    container_pull_timeout_seconds: int = 300
    bioconda_checkout_sources: bool = False
    bioconda_ref: str = "master"
    cache_root: Path | None = None
    restart: bool = False
    status_log_path: Path | None = None
    extract_progress_interval_seconds: float = 30.0
    run_id: str = ""


@dataclass
class ToolRecord:
    # identity
    schema_version: str = "0.4.0"
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
    documentation: str = ""
    expanded_xml_path: str = ""
    macro_files: list[str] = field(default_factory=list)
    uses_macros: bool = False
    macro_expansion_status: str = "not_applicable"
    primary_command: str = ""
    subcommands: list[str] = field(default_factory=list)
    invocation_patterns: list[str] = field(default_factory=list)
    command_text: str = ""

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
_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)
_RECIPE_SET_RE = re.compile(r"{%\s*set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*%}")
_UNRESOLVED_TEMPLATE_RE = re.compile(r"({{.*?}}|{%.*?%}|{#.*?#})")


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
    planned_images: set[str] = field(default_factory=set)
    preparation_cache: dict[str, ContainerPreparation] = field(default_factory=dict)
    failed_preparation_cache: dict[str, ContainerPreparation] = field(default_factory=dict)
    command_presence_cache: dict[tuple[str, str], subprocess.CompletedProcess] = field(
        default_factory=dict
    )
    missing_command_status_emitted: set[tuple[str, str]] = field(default_factory=set)
    images_pulled: int = 0
    images_prepared: int = 0
    images_removed: int = 0
    commands_executed: int = 0
    commands_failed: int = 0
    help_ok: int = 0
    help_degraded: int = 0
    missing_command: int = 0
    non_help_output: int = 0
    failed_probe: int = 0
    prepare_failed: int = 0
    runtime_error: int = 0
    timeout: int = 0
    runtime_fallbacks: int = 0


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
                response = requests.get(raw_url, timeout=10)
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
        response = requests.get(url, timeout=12)
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
        response = requests.get(url, timeout=12)
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


def _choose_container_candidate(
    candidates: list[dict], requirement_versions: dict[str, str]
) -> dict:
    usable = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.get("status", "ok") == "ok" and candidate.get("image")
        ),
        key=_candidate_sort_key,
    )
    if not usable:
        return {}
    for candidate in usable:
        if _image_matches_requirement_versions(
            str(candidate.get("image", "")), requirement_versions
        ):
            return candidate
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


def _singularity_cache_path(image_ref: str, settings: ExtractionSettings) -> Path:
    ref = _normalize_container_ref(image_ref)
    image_name = _url_cache_image_name(ref) if _is_url_ref(ref) else _depot_image_name(ref)
    if not image_name:
        image_name = f"{_safe_slug(ref)}.sif"
    return _container_cache_root(settings) / "singularity" / image_name


def _available_container_runtimes(settings: ExtractionSettings) -> list[ContainerRuntime]:
    requested = settings.container_runtime.strip().lower() or "auto"
    if requested not in {"auto", "singularity", "apptainer", "docker"}:
        raise ValueError(f"Unsupported container runtime: {settings.container_runtime}")

    runtimes: list[ContainerRuntime] = []
    if requested in {"auto", "singularity"}:
        executable = shutil.which("singularity")
        if executable:
            runtimes.append(ContainerRuntime(name="singularity", executable=executable))
        elif requested == "singularity":
            runtimes.append(ContainerRuntime(name="singularity", executable="singularity"))
    if requested in {"auto", "apptainer"}:
        executable = shutil.which("apptainer")
        if executable:
            runtimes.append(ContainerRuntime(name="apptainer", executable=executable))
        elif requested == "apptainer":
            runtimes.append(ContainerRuntime(name="apptainer", executable="apptainer"))
    if requested in {"auto", "docker"}:
        executable = shutil.which("docker")
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
        with requests.get(url, timeout=timeout_seconds, stream=True) as response:
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
        return ContainerPreparation(
            ok=True,
            runtime=runtime.name,
            image=image,
            identifier=str(image_path.resolve()),
            source="explicit-file",
        )

    cache_path = _singularity_cache_path(image, settings=settings)
    if cache_path.exists():
        return ContainerPreparation(
            ok=True,
            runtime=runtime.name,
            image=image,
            identifier=str(cache_path),
            source="cache",
        )

    if _is_url_ref(image):
        source = (
            "galaxy-depot" if _is_depot_url(image, settings.singularity_depot_url) else "remote-url"
        )
        download_result = _download_depot_image(
            image,
            cache_path,
            timeout_seconds=settings.container_pull_timeout_seconds,
        )
        if download_result.ok:
            return ContainerPreparation(
                ok=True,
                runtime=runtime.name,
                image=image,
                identifier=str(cache_path),
                source=source,
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
            timeout_seconds=settings.container_pull_timeout_seconds,
        )
        if depot_result.ok:
            return ContainerPreparation(
                ok=True,
                runtime=runtime.name,
                image=image,
                identifier=str(cache_path),
                source="galaxy-depot",
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
    result = _run_command(command, timeout_seconds=settings.container_pull_timeout_seconds)
    if result.returncode == 0:
        return ContainerPreparation(
            ok=True,
            runtime=runtime.name,
            image=image,
            identifier=str(cache_path),
            source="docker-build",
            command=command,
            stdout=_tail_text(result.stdout),
            stderr=_tail_text(result.stderr),
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
        timeout_seconds=settings.container_pull_timeout_seconds,
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
    executable = shutil.which(preparation.runtime) or preparation.runtime
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
        "export CI=1 TERM=dumb NO_COLOR=1 PAGER=cat; "
        "tmpdir=$(mktemp -d /tmp/gtsm-help.XXXXXX 2>/dev/null || mktemp -d); "
        "trap 'rm -rf \"$tmpdir\"' EXIT; "
        'cd "$tmpdir" || exit 125; '
        f"{command}"
    )


def _is_no_arg_probe(command: str) -> bool:
    parts = command.strip().split()
    if not parts:
        return False
    return not any(part in {"--help", "-h", "help"} for part in parts[1:])


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
    parts = command.strip().split()
    return parts[0] if parts else ""


def _run_container_command_exists(
    preparation: ContainerPreparation,
    primary: str,
    settings: ExtractionSettings,
) -> subprocess.CompletedProcess:
    shell_command = _probe_shell_command(f"command -v {shlex.quote(primary)}")
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


def _help_probe_commands(base: str, probe_mode: str) -> list[str]:
    commands = [f"{base} {flag}" for flag in _HELP_FLAGS]
    if _normalize_help_probe_mode(probe_mode) == "exploratory":
        commands.append(f"{base} help")
        commands.append(base)
    return commands


def _extract_help_commands(
    primary: str, subcommands: list[str], probe_mode: str = "exploratory"
) -> list[str]:
    commands: list[str] = []
    if primary:
        bases = [f"{primary} {sub.strip()}" for sub in subcommands if sub.strip()]
        bases.append(primary)
        for base in bases:
            commands.extend(_help_probe_commands(base, probe_mode))
    deduped: list[str] = []
    seen: set[str] = set()
    for command in commands:
        if command in seen:
            continue
        seen.add(command)
        deduped.append(command)
    return deduped


def _looks_like_help_text(text: str) -> bool:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in ("usage:", "options:", "optional arguments", "--help", "commands:")
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
            marker for marker in _FATAL_PROBE_MARKERS if marker not in {"no such file or directory"}
        )
        if not any(marker in lowered for marker in strong_fatal_markers):
            return False
        if any(marker in lowered for marker in _WEAK_FATAL_HELP_MARKERS) and not any(
            marker in lowered
            for marker in ("command not found", "could not find or load main class", "you ran:")
        ):
            return False
    return any(marker in lowered for marker in (*_FATAL_PROBE_MARKERS, *_DEGRADED_HELP_MARKERS))


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
        marker for marker in _FATAL_PROBE_MARKERS if marker not in {"no such file or directory"}
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
    primary = _command_primary(command).lower()
    if primary in _SHELL_COMMAND_DENYLIST:
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


def _container_probe_status(result: subprocess.CompletedProcess, fragment: str) -> str:
    text = _combined_container_text(result)
    if fragment:
        if _looks_like_degraded_help_text(text):
            return "container-command-help-degraded"
        return "container-command-help"
    if _looks_like_failed_probe_text(text) or result.returncode == 127:
        return "container-command-failed-probe"
    return "container-command-nonhelp"


def _is_missing_command_probe(result: subprocess.CompletedProcess) -> bool:
    text = _combined_container_text(result).lower()
    return (
        result.returncode == 127
        or "command not found" in text
        or "no such file or directory" in text
    )


def _combine_help_text(original_help_text: str, container_help_text: str) -> str:
    sections = []
    if original_help_text.strip():
        sections.append(original_help_text.strip())
    if container_help_text.strip():
        sections.append(
            "Command-line help collected from container execution:\n\n"
            + container_help_text.strip()
        )
    return "\n\n".join(sections)


def _run_command(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, input="", capture_output=True, text=True, timeout=timeout_seconds, check=False
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        detail = stderr or stdout or f"Command timed out after {timeout_seconds} seconds"
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=detail)
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


def _ensure_conda_forge_feedstock_repo(
    cache_root: Path,
    package: str,
    settings: ExtractionSettings,
) -> tuple[Path | None, str, str]:
    feedstock_root = cache_root / "conda-forge-feedstocks"
    cache_namespace = str(feedstock_root.resolve())
    last_error = ""
    for recipe_package in _recipe_package_variants(package):
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
    value = raw_value.strip().rstrip(";").strip()
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


def _extract_recipe_set_variables(meta_text: str) -> dict[str, object]:
    return {
        match.group(1): _parse_recipe_set_value(match.group(2))
        for match in _RECIPE_SET_RE.finditer(meta_text)
    }


def _extract_recipe_version(meta_text: str) -> str:
    variables = _extract_recipe_set_variables(meta_text)
    if variables.get("version") is not None:
        return str(variables["version"]).strip()
    for line in meta_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version:"):
            return stripped.split(":", 1)[1].strip().strip("\"'")
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


def _extract_source_fields(meta_text: str) -> tuple[str, str]:
    source_url = ""
    source_ref = ""
    for line in meta_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- url:") and not source_url:
            source_url = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("url:") and not source_url:
            source_url = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("- git_url:") and not source_url:
            source_url = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("git_url:") and not source_url:
            source_url = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("- git_rev:"):
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("git_rev:"):
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("- tag:") and not source_ref:
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
        if stripped.startswith("tag:") and not source_ref:
            source_ref = _clean_recipe_scalar(stripped.split(":", 1)[1])
    return source_url, source_ref


def _recipe_relpath(package: str, *, feedstock: bool = False) -> str:
    if feedstock:
        return "recipe/meta.yaml"
    return f"recipes/{package}/meta.yaml"


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
    cleaned = str(value or "").strip().strip("\"'").lower()
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
        return ((5, snapshot.recipe_version), "low_confidence_recipe_version")
    if candidate.normalized == required.normalized:
        return ((0,), "exact")
    comparison = _compare_numeric_versions(candidate.numeric, required.numeric)
    if comparison == 0:
        return ((1, candidate.normalized), "numeric_equivalent")
    candidate_major = candidate.numeric[0]
    required_major = required.numeric[0]
    if candidate_major == required_major and comparison > 0:
        return (
            (2, _numeric_version_distance(candidate.numeric[1:], required.numeric[1:])),
            "same_major_newer",
        )
    if candidate_major == required_major:
        return (
            (3, _numeric_version_distance(candidate.numeric[1:], required.numeric[1:])),
            "same_major_older",
        )
    older_major_preference = 0 if candidate_major < required_major else 1
    return (
        (
            4,
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
    recipes_repo: Path, ref: str, package: str, *, feedstock: bool = False
) -> tuple[str, str]:
    relpath = _recipe_relpath(package, feedstock=feedstock)
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
    recipes_repo: Path, package: str, *, feedstock: bool = False
) -> BiocondaRecipeSnapshot:
    meta_path = recipes_repo / _recipe_relpath(package, feedstock=feedstock)
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
    recipes_repo: Path, ref: str, package: str, *, feedstock: bool = False
) -> BiocondaRecipeSnapshot:
    relpath = _recipe_relpath(package, feedstock=feedstock)
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
        recipes_repo, ref, package, feedstock=feedstock
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
) -> tuple[list[BiocondaRecipeSnapshot], int, str]:
    relpath = _recipe_relpath(package, feedstock=feedstock)
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
        if not (recipes_repo / ".git").exists():
            snapshot = _worktree_recipe_snapshot(
                recipes_repo, recipe_package, feedstock=feedstock
            )
            if not snapshot.meta_text:
                first_missing = first_missing or snapshot
                continue
            if required_version and snapshot.recipe_version:
                required = _normalize_recipe_version_text(required_version)
                recipe = _normalize_recipe_version_text(snapshot.recipe_version)
                if required != recipe:
                    return replace(snapshot, selection_reason="worktree_version_mismatch")
            return snapshot

        current_snapshot = _recipe_snapshot_at_ref(
            recipes_repo, ref, recipe_package, feedstock=feedstock
        )
        if current_snapshot.meta_text:
            required = _normalize_recipe_version_text(required_version)
            recipe = _normalize_recipe_version_text(current_snapshot.recipe_version)
            if not required_version or (required and required == recipe):
                return replace(
                    current_snapshot,
                    selection_reason="exact" if required_version else "current_ref",
                )

        candidates, scanned_commits, history_error = _recipe_history_candidates(
            recipes_repo, ref, recipe_package, feedstock=feedstock
        )
        if candidates:
            return _select_recipe_snapshot_from_candidates(
                package=recipe_package,
                required_version=required_version,
                candidates=candidates,
                scanned_commits=scanned_commits,
            )
        if current_snapshot.meta_text:
            reason = (
                "current_ref_fallback" if not history_error else "history_unavailable_current_ref"
            )
            return replace(
                current_snapshot,
                selection_reason=reason,
                error=history_error,
                scanned_commits=scanned_commits or current_snapshot.scanned_commits,
            )
        first_missing = first_missing or replace(
            current_snapshot,
            package=package,
            error=current_snapshot.error or history_error or "recipe_not_found",
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


def _download_and_extract_archive(source_url: str, checkout_dir: Path) -> str:
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
    if parsed.scheme.lower() == "ftp":
        with tmp_path.open("wb") as handle:
            with urlrequest.urlopen(source_url, timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    else:
        with requests.get(source_url, timeout=120, stream=True) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    tmp_path.replace(archive_path)
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=checkout_dir)
    except tarfile.TarError:
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
            _emit_extract_status(
                settings,
                {
                    "status": f"{channel}-source-ready",
                    "source_channel": channel,
                    "package": package,
                    "source_url": source_url,
                    "source_checkout": str(checkout_dir),
                    "returncode": 0 if not source_error else 1,
                    "error_text": source_error,
                },
            )
            return remember(str(checkout_dir), source_error)

        try:
            source_checkout = _download_and_extract_archive(source_url, checkout_dir)
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

        _emit_extract_status(
            settings,
            {
                "status": f"{channel}-source-ready",
                "source_channel": channel,
                "package": package,
                "source_url": source_url,
                "source_checkout": source_checkout,
                "returncode": 0,
                "error_text": "",
            },
        )
        return remember(source_checkout, "")


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


def _extract_source_command_hints(source_checkout: str, package: str) -> list[str]:
    if not source_checkout:
        return []
    root = Path(source_checkout)
    if root.is_file():
        root = root.parent
    if not root.exists():
        return []

    hints: set[str] = set()
    package_key = _normalized_command_key(package)
    candidate_dirs = [root / "bin", root / "scripts", root / "script"]
    for directory in candidate_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.iterdir())[:100]:
            if path.is_file():
                hint = _command_hint_from_name(path.name)
                if hint:
                    hints.add(hint)

    for metadata_name in ("setup.py", "setup.cfg", "pyproject.toml"):
        path = root / metadata_name
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hints.update(_extract_recipe_command_hints(text))

    for path in sorted(root.iterdir())[:200]:
        if not path.is_file():
            continue
        hint = _command_hint_from_name(path.name)
        if hint and (
            _normalized_command_key(hint) == package_key
            or package_key in _normalized_command_key(hint)
        ):
            hints.add(hint)
    return sorted(hints)


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
    source_url_template, source_ref_template = _extract_source_fields(meta_text)
    source_url, source_ref, template_error = _render_bioconda_source_fields(
        meta_text=meta_text,
        package=recipe_package,
        requirement_version=required_version,
        recipe_version=recipe_version,
        source_url=source_url_template,
        source_ref=source_ref_template,
    )
    command_hints = set(_extract_recipe_command_hints(meta_text))
    source_checkout = ""
    source_error = ""
    if source_url:
        sources_root = cache_root / f"{channel}-sources"
        version_hint = _source_checkout_version_hint(required_version, recipe_version)
        checkout_dir = sources_root / f"{_safe_slug(package)}-{_safe_slug(version_hint)}"
        cache_key = _source_result_cache_key(
            f"{channel}:{package}", version_hint, source_url, source_ref
        )
        cached_result = _source_result_cache_get(
            source_result_cache,
            source_result_cache_lock,
            cache_key,
        )
        if cached_result is not None:
            source_checkout, source_error = cached_result
            _emit_bioconda_source_cache_hit(
                package=package,
                source_url=source_url,
                source_ref=source_ref,
                source_checkout=source_checkout,
                source_error=source_error,
                settings=settings,
                channel=channel,
            )
        elif template_error:
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
                    "checkout_dir": str(checkout_dir),
                    "returncode": 1,
                    "error_text": source_error,
                },
            )
            _source_result_cache_put(
                source_result_cache,
                source_result_cache_lock,
                cache_key,
                ("", source_error),
            )
        else:
            source_checkout, source_error = _checkout_bioconda_source(
                package=package,
                source_url=source_url,
                source_ref=source_ref,
                checkout_dir=checkout_dir,
                settings=settings,
                channel=channel,
                source_result_cache=source_result_cache,
                source_result_cache_lock=source_result_cache_lock,
                source_result_cache_key=cache_key,
            )
    command_hints.update(_extract_source_command_hints(source_checkout, package))
    return {
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
        "source_url": source_url,
        "source_url_template": source_url_template,
        "source_ref": source_ref,
        "source_ref_template": source_ref_template,
        "source_checkout": source_checkout,
        "source_error": source_error,
        "command_hints": sorted(command_hints),
    }


def _resolve_conda_forge_source_mapping(
    *,
    package: str,
    required_version: str,
    settings: ExtractionSettings,
    source_result_cache: dict[tuple[str, str, str, str], tuple[str, str]] | None = None,
    source_result_cache_lock: threading.Lock | None = None,
    recipe_selection_cache: dict[tuple[str, str, str, str], BiocondaRecipeSnapshot] | None = None,
    recipe_selection_cache_lock: threading.Lock | None = None,
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
    )


def _should_use_conda_forge_fallback(primary: dict, fallback: dict) -> bool:
    if primary.get("source_checkout"):
        return False
    if fallback.get("source_checkout"):
        return True
    if not primary.get("recipe_path") and fallback.get("recipe_path"):
        return True
    if not primary.get("source_url") and fallback.get("source_url"):
        return True
    return False


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
        primary = _resolve_recipe_source_mapping(
            package=package,
            required_version=required_version,
            channel="bioconda",
            settings=settings,
            recipes_repo=recipes_repo,
            recipes_repo_error=recipes_repo_error,
            ref=settings.bioconda_ref,
            source_result_cache=source_result_cache,
            source_result_cache_lock=source_result_cache_lock,
            recipe_selection_cache=recipe_selection_cache,
            recipe_selection_cache_lock=recipe_selection_cache_lock,
        )
        if primary.get("source_checkout"):
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
_FILELIKE_NAMES = {
    "file",
    "file1",
    "file2",
    "fasta_file",
    "index_dir",
    "input",
    "input_bam",
    "input_file",
    "localbam",
    "out",
    "output",
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


def _command_tokens(segment: str) -> list[str]:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = _TOKEN_RE.findall(segment)
    return [token.strip() for token in tokens if token.strip()]


def _clean_command_token(token: str) -> str:
    return token.strip().strip("\"'").strip()


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


def _is_filelike_token(token: str) -> bool:
    lowered = token.lower()
    if lowered in _FILELIKE_NAMES:
        return True
    if any(lowered.endswith(suffix.lower()) for suffix in _FILELIKE_SUFFIXES):
        return True
    return "." in token


def _is_executable_candidate(token: str) -> bool:
    cleaned = _clean_command_token(token)
    lowered = cleaned.lower()
    if not cleaned or cleaned in _REDIRECT_TOKENS or cleaned.startswith("-"):
        return False
    if _is_assignment_token(cleaned) or _is_placeholder_token(cleaned) or _is_path_token(cleaned):
        return False
    if lowered in _SHELL_COMMAND_DENYLIST or lowered in _INTERPRETER_COMMANDS:
        return False
    if _is_filelike_token(cleaned):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_+.-]*$", cleaned))


def _is_subcommand_candidate(token: str) -> bool:
    cleaned = _clean_command_token(token)
    lowered = cleaned.lower()
    if not cleaned or cleaned in _REDIRECT_TOKENS or cleaned.startswith("-"):
        return False
    if _is_assignment_token(cleaned) or _is_placeholder_token(cleaned) or _is_path_token(cleaned):
        return False
    if lowered in _SHELL_COMMAND_DENYLIST or lowered in _INTERPRETER_COMMANDS:
        return False
    if _is_filelike_token(cleaned):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_+-]*$", cleaned))


def _command_segments(line: str) -> list[str]:
    return [
        segment.strip() for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", line) if segment.strip()
    ]


def _signature_from_segment(segment: str) -> tuple[str, str]:
    tokens = [_clean_command_token(token) for token in _command_tokens(segment)]
    tokens = [token for token in tokens if token and token not in _REDIRECT_TOKENS]
    skippable_prefixes = {"then", "do", "else"}
    while tokens and (_is_assignment_token(tokens[0]) or tokens[0].lower() in skippable_prefixes):
        tokens.pop(0)
    if not tokens:
        return "", ""
    primary = tokens[0]
    if primary.lower() in _SHELL_COMMAND_DENYLIST or primary.lower() in _INTERPRETER_COMMANDS:
        return "", ""
    if not _is_executable_candidate(primary):
        return "", ""
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        if _is_subcommand_candidate(token) and token != primary:
            return primary, token
        break
    return primary, ""


def _infer_command_signatures(command_text: str) -> tuple[str, list[str], list[str]]:
    lines = []
    for raw in command_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    invocation_patterns = lines[:8]
    if not lines:
        return "", [], []

    primary = ""
    subcommands: list[str] = []
    for line in lines:
        for segment in _command_segments(line):
            candidate, subcommand = _signature_from_segment(segment)
            if not candidate:
                continue
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
    for raw in command_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for segment in _command_segments(line):
            candidate = _signature_from_segment(segment)
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
) -> tuple[set[str], set[str], set[str], str]:
    hint_keys: set[str] = set()
    source_hint_keys: set[str] = set()
    package_keys: set[str] = set()
    owning_packages = list(candidate_packages or record.requirement_packages)
    filter_mappings_by_candidate = candidate_packages is not None

    values = [
        record.shed_name,
        record.tool_id,
        record.tool_name,
        Path(record.wrapper_path).stem if record.wrapper_path else "",
        _container_image_base_name(image),
    ]
    values.extend(owning_packages)
    for value in values:
        for token in _identity_values_from_text(str(value)):
            key = _normalized_command_key(token)
            if key:
                hint_keys.add(key)

    for package in owning_packages:
        key = _normalized_command_key(package)
        if key and key not in {_normalized_command_key(name) for name in _GENERIC_CONTAINER_NAMES}:
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
                package_keys.add(key)
        for hint in mapping.get("command_hints", []) or []:
            key = _normalized_command_key(str(hint))
            if key:
                hint_keys.add(key)
                source_hint_keys.add(key)

    image_key = _normalized_command_key(_container_image_base_name(image))
    return hint_keys, source_hint_keys, package_keys, image_key


def _command_key_matches(candidate_key: str, hint_key: str) -> bool:
    if not candidate_key or not hint_key:
        return False
    if candidate_key == hint_key:
        return True
    if len(candidate_key) >= 4 and len(hint_key) >= 4:
        return candidate_key in hint_key or hint_key in candidate_key
    return False


def _score_command_candidate(
    primary: str,
    *,
    hint_keys: set[str],
    source_hint_keys: set[str],
    package_keys: set[str],
    image_key: str,
    strict_package_match: bool = False,
) -> int:
    primary_key = _normalized_command_key(primary)
    if not primary_key:
        return 0

    score = 0
    if primary_key in source_hint_keys:
        score += 140
    if primary_key in package_keys:
        score += 120
    if primary_key == image_key:
        score += 110
    elif _command_key_matches(primary_key, image_key):
        score += 80
    if primary_key in hint_keys:
        score += 90
    elif any(_command_key_matches(primary_key, hint) for hint in hint_keys):
        score += 55

    generic_image = image_key in {
        _normalized_command_key(name) for name in _GENERIC_CONTAINER_NAMES
    }
    if generic_image and primary_key not in source_hint_keys and primary_key not in package_keys:
        return 0
    if (
        strict_package_match
        and primary_key not in source_hint_keys
        and primary_key not in package_keys
        and primary_key != image_key
        and not _command_key_matches(primary_key, image_key)
    ):
        return 0
    return score


def _record_help_commands(
    record: ToolRecord,
    image: str,
    probe_mode: str = "exploratory",
    candidate_packages: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    command_text = record.command_text or "\n".join(record.invocation_patterns)
    candidates = _command_candidate_signatures(command_text)
    if not candidates and record.primary_command:
        candidates.append((record.primary_command, ""))
        for subcommand in record.subcommands:
            candidates.append((record.primary_command, subcommand))

    hint_keys, source_hint_keys, package_keys, image_key = _record_identity_keys(
        record,
        image,
        candidate_packages=candidate_packages,
    )
    strict_package_match = candidate_packages is not None and image_key not in {
        _normalized_command_key(name) for name in _GENERIC_CONTAINER_NAMES
    }
    scored: list[tuple[int, int, str, str]] = []
    for index, (primary, subcommand) in enumerate(candidates):
        score = _score_command_candidate(
            primary,
            hint_keys=hint_keys,
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
    best_primary = scored[0][2]
    subcommands = []
    for score, _, primary, subcommand in scored:
        if score < 100 or primary != best_primary or not subcommand:
            continue
        if subcommand not in subcommands:
            subcommands.append(subcommand)
    return _extract_help_commands(best_primary, subcommands, probe_mode=probe_mode)


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
) -> ToolRecord:
    xml_rel = str(xml_file.relative_to(tool_dir))
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
    primary_command, subcommands, invocation_patterns = _infer_command_signatures(command_text)

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
    )
    selected_container = str(selected_candidate.get("image", "") or "")
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

    return ToolRecord(
        package_id=package_id,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_dir=str(tool_dir),
        wrapper_path=str(xml_file),
        xml_files=[xml_rel],
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
        primary_command=primary_command,
        subcommands=subcommands,
        invocation_patterns=invocation_patterns,
        command_text=command_text,
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
    return ContainerExecutionState(runtimes=_available_container_runtimes(settings))


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
            elif status:
                state.commands_failed += int(
                    status not in {"container-command-help", "container-command-help-degraded"}
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
    records_with_container_execution = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("container_help_text", "") or "").strip():
            records_with_container_help += 1
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
                    "events": events,
                }
            )

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

    return {
        "diagnostics_dir": str(diagnostics_dir),
        "execution_summary_path": str(execution_summary_path),
        "container_status_counts_path": str(status_counts_path),
        "container_help_coverage_path": str(coverage_path),
        "nonhelp_samples_path": str(nonhelp_samples_path),
        "integrity_path": str(integrity_path),
        "summary": {
            "total_records": len(records),
            "records_with_container_help": records_with_container_help,
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
            record.original_help_text or record.help_text, record.container_help_text
        )
        return
    if record.container_help_text.strip():
        record.help_text = _combine_help_text(
            record.original_help_text or record.help_text, record.container_help_text
        )
        return

    def prepare_candidate(
        image: str, candidate: dict, candidate_rank: int
    ) -> ContainerPreparation | None:
        if image in state.preparation_cache:
            return state.preparation_cache[image]
        if image in state.failed_preparation_cache:
            return None

        failed_preparations: list[ContainerPreparation] = []
        for runtime in state.runtimes:
            attempted = _prepare_container(image=image, runtime=runtime, settings=settings)
            _emit_extract_status(
                settings,
                {
                    "status": "container-prepare",
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
                if failed_preparations:
                    state.runtime_fallbacks += 1
                state.images_prepared += 1
                if attempted.source == "docker-pull":
                    state.images_pulled += 1
                return attempted
            failed_preparations.append(attempted)

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
        return None

    def candidate_commands() -> list[tuple[str, list[str], dict, int]]:
        planned: list[tuple[str, list[str], dict, int]] = []
        for rank, candidate in enumerate(
            sorted(_legacy_container_candidate_details(record), key=_candidate_sort_key), start=1
        ):
            raw_image = str(candidate.get("image", "") or "")
            if candidate.get("status", "ok") != "ok":
                _emit_extract_status(
                    settings,
                    {
                        "status": "container-candidate-skipped",
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
            commands = _record_help_commands(
                record,
                image,
                probe_mode=settings.container_help_probe_mode,
                candidate_packages=packages,
            )
            if commands:
                state.planned_images.add(image)
                planned.append((image, commands, candidate, rank))
        return planned

    candidates = candidate_commands()
    if not candidates:
        record.help_text = _combine_help_text(
            record.original_help_text or record.help_text, record.container_help_text
        )
        return

    if not state.runtimes:
        image, commands, candidate, rank = candidates[0]
        record.container_execution.append(
            {
                "phase": "prepare",
                "runtime": settings.container_runtime,
                "image": image,
                "candidate_source": candidate.get("source", ""),
                "candidate_rank": rank,
                "command": commands[0],
                "returncode": 127,
                "stdout": "",
                "stderr": "",
                "error_text": f"No available container runtime for {settings.container_runtime}",
            }
        )
        state.commands_failed += 1
        state.runtime_error += 1
        record.help_text = _combine_help_text(
            record.original_help_text or record.help_text, record.container_help_text
        )
        return

    for image, commands, candidate, rank in candidates:
        if record.container_help_text.strip():
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
                    "command": commands[0],
                    "returncode": last_failure.returncode,
                    "stdout": last_failure.stdout,
                    "stderr": last_failure.stderr,
                    "error_text": last_failure.error_text,
                }
            )
            state.commands_failed += 1
            state.prepare_failed += 1
            continue

        candidate_missing_command = False
        for command in commands:
            if record.container_help_text.strip():
                break
            primary = _command_primary(command)
            if primary:
                presence_key = (image, primary)
                if presence_key not in state.command_presence_cache:
                    state.command_presence_cache[presence_key] = _run_container_command_exists(
                        preparation=preparation,
                        primary=primary,
                        settings=settings,
                    )
                presence_result = state.command_presence_cache[presence_key]
                if presence_result.returncode != 0:
                    candidate_missing_command = True
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
                                "runtime": preparation.runtime,
                                "image": image,
                                "candidate_source": candidate.get("source", ""),
                                "candidate_rank": rank,
                                "fallback_reason": "missing-command",
                                "command": command,
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
                    break

            result = _run_container_probe(
                preparation=preparation, command=command, settings=settings
            )
            state.commands_executed += 1
            fragment = _container_help_fragment(command, result)
            probe_status = _container_probe_status(result, fragment)
            record.selected_container = image
            _record_probe_summary(state, probe_status, result.returncode)
            if probe_status != "container-command-help":
                _emit_extract_status(
                    settings,
                    {
                        "status": probe_status,
                        "runtime": preparation.runtime,
                        "image": image,
                        "candidate_source": candidate.get("source", ""),
                        "candidate_rank": rank,
                        "command": command,
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
                )
            elif _is_missing_command_probe(result):
                candidate_missing_command = True
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
                    "returncode": result.returncode,
                    "stdout": _tail_text(_strip_container_runtime_noise(result.stdout or "")),
                    "stderr": _tail_text(_strip_container_runtime_noise(result.stderr or "")),
                    "error_text": "" if result.returncode == 0 else _completed_error_text(result),
                }
            )
            if candidate_missing_command:
                break

        if record.container_help_text.strip():
            break

    record.help_text = _combine_help_text(
        record.original_help_text or record.help_text, record.container_help_text
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
            wrappers.append((tool_dir, xml_file))

    processed = _load_checkpoint(checkpoint_file)
    pending = [
        (tool_dir, xml_file)
        for tool_dir, xml_file in wrappers
        if _checkpoint_key(tools_root=tools_root, tool_dir=tool_dir, xml_file=xml_file)
        not in processed
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
            "max_workers": settings.max_workers,
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
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = {
            executor.submit(
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
            ): (tool_dir, xml_file)
            for tool_dir, xml_file in pending
        }
        remaining = set(futures)
        with (
            output_jsonl.open("a", encoding="utf-8") as out,
            checkpoint_file.open("a", encoding="utf-8") as ck,
        ):
            while remaining:
                done, remaining = wait(
                    remaining, timeout=progress_interval, return_when=FIRST_COMPLETED
                )
                if not done:
                    _emit_extract_status(
                        settings,
                        {
                            "status": "extract-progress",
                            "phase": "wrapper-extraction",
                            "processed_now": written,
                            "pending_wrappers": len(pending),
                            "remaining_wrappers": len(remaining),
                            "elapsed_seconds": round(time.monotonic() - started_at, 1),
                        },
                    )
                    continue
                for future in done:
                    tool_dir, xml_file = futures[future]
                    record = future.result()
                    _execute_container_help_for_record(
                        record, settings=settings, state=execution_state
                    )
                    out.write(record.to_json() + "\n")
                    key = _checkpoint_key(
                        tools_root=tools_root, tool_dir=tool_dir, xml_file=xml_file
                    )
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
                            "remaining_wrappers": len(remaining),
                            "elapsed_seconds": round(time.monotonic() - started_at, 1),
                        },
                    )

    execution_summary = _finalize_container_execution_state(execution_state, settings=settings)
    all_records = _load_records_from_jsonl(output_jsonl)
    records_for_reports = all_records or emitted
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
    }
