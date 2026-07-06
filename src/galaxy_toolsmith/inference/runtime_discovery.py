from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.data.corpus import (
    ContainerPreparation,
    ExtractionSettings,
    _available_container_runtimes,
    _build_container_candidate_details,
    _choose_container_candidate,
    _command_primary,
    _completed_error_text,
    _container_help_fragment,
    _container_probe_status,
    _container_usage_fragment,
    _extract_help_commands,
    _prepare_container,
    _resolve_bioconda_source_mappings,
    _run_container_command_exists,
    _run_container_probe,
    _sort_container_candidates_for_selection,
    _strip_container_runtime_noise,
    _tail_text,
)
from galaxy_toolsmith.inference.source_archives import resolve_source_archive
from galaxy_toolsmith.inference.suite import infer_suite_tool_plans

DISCOVERY_MODES = ("none", "conda", "biocontainer", "auto")
DEFAULT_DISCOVERY_CHANNELS = ("bioconda", "conda-forge")
ACCEPTED_HELP_STATUSES = {
    "container-command-help",
    "container-command-help-degraded",
    "container-command-usage-degraded",
}
_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*([=<>!~]{1,2})?\s*([^\s=<>!~]+)?")
_ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
)


@dataclass(frozen=True)
class RuntimeHelpProbe:
    command: str
    probe_role: str
    status: str
    returncode: int
    accepted: bool = False
    help_text: str = ""
    stdout: str = ""
    stderr: str = ""
    error_text: str = ""
    runtime: str = ""
    image: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeDiscoverySettings:
    mode: str = "none"
    package_specs: tuple[str, ...] = ()
    command: str = ""
    channels: tuple[str, ...] = DEFAULT_DISCOVERY_CHANNELS
    cache_dir: Path | None = None
    env_dir: Path | None = None
    conda_executable: str = ""
    conda_timeout_seconds: int = 900
    discover_subcommands: bool = True
    max_discovered_subcommands: int = 8
    container_runtime: str = "auto"
    container_cache_dir: Path | None = None
    docker_use_sudo: bool = False
    container_help_probe_mode: str = "exploratory"
    container_timeout_seconds: int = 120
    bioconda_ref: str = "master"
    source_download_max_bytes: int = 0
    source_download_timeout_seconds: int = 60


@dataclass(frozen=True)
class RuntimeDiscoveryResult:
    mode: str
    selected_runtime: str = ""
    command: str = ""
    package_specs: tuple[str, ...] = ()
    package_names: tuple[str, ...] = ()
    requirement_versions: Mapping[str, str] = field(default_factory=dict)
    env_dir: str = ""
    conda_executable: str = ""
    container_image: str = ""
    container_identifier: str = ""
    container_source: str = ""
    top_level_help: str = ""
    subcommand_help: Mapping[str, str] = field(default_factory=dict)
    combined_help_text: str = ""
    source_root: str = ""
    source_mappings: tuple[Mapping[str, Any], ...] = ()
    probes: tuple[RuntimeHelpProbe, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def has_help(self) -> bool:
        return bool(self.combined_help_text.strip())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probes"] = [probe.to_dict() for probe in self.probes]
        payload["source_mappings"] = [dict(mapping) for mapping in self.source_mappings]
        payload["requirement_versions"] = dict(self.requirement_versions)
        payload["subcommand_help"] = dict(self.subcommand_help)
        return payload


def normalize_discovery_mode(mode: str | None) -> str:
    cleaned = (mode or "none").strip().lower().replace("_", "-")
    if cleaned not in DISCOVERY_MODES:
        raise ValueError("Unsupported discovery mode. Use one of: " + ", ".join(DISCOVERY_MODES))
    return cleaned


def discover_runtime_context(
    *,
    paths: WorkspacePaths,
    settings: RuntimeDiscoverySettings,
) -> RuntimeDiscoveryResult:
    mode = normalize_discovery_mode(settings.mode)
    if mode == "none":
        return RuntimeDiscoveryResult(mode="none")
    normalized = _normalize_settings(paths=paths, settings=settings, mode=mode)
    if mode == "conda":
        return _discover_with_conda(paths=paths, settings=normalized)
    if mode == "biocontainer":
        return _discover_with_biocontainer(paths=paths, settings=normalized)

    conda_result = _discover_with_conda(paths=paths, settings=normalized)
    if conda_result.has_help:
        return conda_result
    container_result = _discover_with_biocontainer(paths=paths, settings=normalized)
    warnings = [
        *conda_result.warnings,
        "Conda discovery did not produce accepted help; tried Biocontainers fallback.",
        *container_result.warnings,
    ]
    errors = [*conda_result.errors, *container_result.errors]
    return RuntimeDiscoveryResult(
        mode="auto",
        selected_runtime=container_result.selected_runtime,
        command=container_result.command or conda_result.command,
        package_specs=container_result.package_specs or conda_result.package_specs,
        package_names=container_result.package_names or conda_result.package_names,
        requirement_versions=container_result.requirement_versions
        or conda_result.requirement_versions,
        env_dir=conda_result.env_dir,
        conda_executable=conda_result.conda_executable,
        container_image=container_result.container_image,
        container_identifier=container_result.container_identifier,
        container_source=container_result.container_source,
        top_level_help=container_result.top_level_help,
        subcommand_help=container_result.subcommand_help,
        combined_help_text=container_result.combined_help_text,
        source_root=container_result.source_root or conda_result.source_root,
        source_mappings=container_result.source_mappings or conda_result.source_mappings,
        probes=(*conda_result.probes, *container_result.probes),
        warnings=tuple(warnings),
        errors=tuple(error for error in errors if error),
    )


def _normalize_settings(
    *,
    paths: WorkspacePaths,
    settings: RuntimeDiscoverySettings,
    mode: str,
) -> RuntimeDiscoverySettings:
    package_specs = tuple(spec.strip() for spec in settings.package_specs if spec.strip())
    command = settings.command.strip()
    if not package_specs and command:
        package_specs = (_package_name_from_spec(command),)
    if package_specs and not command:
        command = _package_name_from_spec(package_specs[0])
    cache_dir = settings.cache_dir or paths.cache_root / "generation" / "runtime-discovery"
    container_cache_dir = settings.container_cache_dir or paths.cache_root / "containers"
    return RuntimeDiscoverySettings(
        mode=mode,
        package_specs=package_specs,
        command=command,
        channels=tuple(settings.channels or DEFAULT_DISCOVERY_CHANNELS),
        cache_dir=cache_dir,
        env_dir=settings.env_dir,
        conda_executable=settings.conda_executable,
        conda_timeout_seconds=max(1, int(settings.conda_timeout_seconds)),
        discover_subcommands=bool(settings.discover_subcommands),
        max_discovered_subcommands=max(0, int(settings.max_discovered_subcommands)),
        container_runtime=settings.container_runtime,
        container_cache_dir=container_cache_dir,
        docker_use_sudo=settings.docker_use_sudo,
        container_help_probe_mode=settings.container_help_probe_mode or "exploratory",
        container_timeout_seconds=max(1, int(settings.container_timeout_seconds)),
        bioconda_ref=settings.bioconda_ref or "master",
        source_download_max_bytes=max(0, int(settings.source_download_max_bytes)),
        source_download_timeout_seconds=max(1, int(settings.source_download_timeout_seconds)),
    )


def _package_name_from_spec(spec: str) -> str:
    match = _SPEC_RE.match(spec.strip())
    if match:
        return match.group(1)
    return spec.strip().split()[0] if spec.strip() else ""


def _package_versions_from_specs(package_specs: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    package_names: list[str] = []
    requirement_versions: dict[str, str] = {}
    for spec in package_specs:
        match = _SPEC_RE.match(spec)
        if not match:
            continue
        package = match.group(1).strip()
        if not package:
            continue
        package_names.append(package)
        operator = match.group(2) or ""
        version = (match.group(3) or "").strip()
        if version and operator in {"=", "=="}:
            requirement_versions[package] = version
    return package_names, requirement_versions


def _conda_executables(explicit: str = "") -> tuple[str, ...]:
    if explicit.strip():
        return (explicit.strip(),)
    candidates = [
        "mamba",
        "micromamba",
        "conda",
        str(Path.home() / "miniforge3" / "condabin" / "mamba"),
        str(Path.home() / "miniforge3" / "bin" / "mamba"),
        str(Path.home() / "miniforge3" / "condabin" / "conda"),
        str(Path.home() / "miniforge3" / "bin" / "conda"),
        str(Path.home() / "miniconda3" / "condabin" / "conda"),
        str(Path.home() / "miniconda3" / "bin" / "conda"),
    ]
    executables: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = shutil.which(candidate) if os.sep not in candidate else candidate
        if resolved and Path(resolved).exists() and resolved not in seen:
            seen.add(resolved)
            executables.append(resolved)
    return tuple(executables)


def _env_hash(settings: RuntimeDiscoverySettings) -> str:
    key = "\n".join([*settings.channels, "--", *settings.package_specs])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _env_dir(settings: RuntimeDiscoverySettings) -> Path:
    if settings.env_dir is not None:
        return settings.env_dir
    assert settings.cache_dir is not None
    return settings.cache_dir / "conda-envs" / _env_hash(settings)


def _create_or_reuse_conda_env(
    *,
    settings: RuntimeDiscoverySettings,
    env_dir: Path,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    executables = _conda_executables(settings.conda_executable)
    if not executables:
        return "", (), ("No mamba, micromamba, or conda executable was found.",)
    if not settings.package_specs:
        return executables[0], (), ("No discovery packages were provided.",)
    if (env_dir / "conda-meta").exists():
        return executables[0], (), ()
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for executable in executables:
        command = [executable, "create", "-y", "-p", str(env_dir)]
        for channel in settings.channels:
            command.extend(["-c", channel])
        command.extend(settings.package_specs)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=settings.conda_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{executable}: {error}")
            continue
        if result.returncode == 0:
            return executable, (), ()
        error_text = _tail_text(result.stderr or result.stdout or "", limit=4000)
        errors.append(f"{executable}: {error_text}")
    return executables[-1], (), (
        "Conda environment creation failed with all available executables: "
        + " | ".join(errors[-4:]),
    )


def _run_env_probe(
    *,
    env_dir: Path,
    command: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env.update(
        {
            "CI": "1",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "PAGER": "cat",
            "TZ": "UTC",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "GALAXY_MEMORY_MB": "1024",
        }
    )
    with tempfile.TemporaryDirectory(prefix="gtsm-help-") as tmpdir:
        return subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )


def _probe_from_result(
    *,
    command: str,
    probe_role: str,
    result: subprocess.CompletedProcess,
    runtime: str = "",
    image: str = "",
    source: str = "",
) -> RuntimeHelpProbe:
    fragment = _container_help_fragment(command, result)
    usage_fragment = "" if fragment else _container_usage_fragment(command, result)
    status = _container_probe_status(result, fragment, usage_fragment)
    help_text = fragment or usage_fragment
    return RuntimeHelpProbe(
        command=command,
        probe_role=probe_role,
        status=status,
        returncode=int(result.returncode),
        accepted=status in ACCEPTED_HELP_STATUSES,
        help_text=help_text,
        stdout=_tail_text(_strip_container_runtime_noise(result.stdout or "")),
        stderr=_tail_text(_strip_container_runtime_noise(result.stderr or "")),
        error_text="" if result.returncode == 0 else _completed_error_text(result),
        runtime=runtime,
        image=image,
        source=source,
    )


def _run_help_probe_sequence(
    *,
    runner: Any,
    command: str,
    probe_role: str,
    timeout_seconds: int,
    probe_mode: str,
    runtime: str = "",
    image: str = "",
    source: str = "",
) -> tuple[str, tuple[RuntimeHelpProbe, ...]]:
    probes: list[RuntimeHelpProbe] = []
    for probe_command in _extract_help_commands(command, [], probe_mode=probe_mode):
        try:
            result = runner(probe_command, timeout_seconds)
        except subprocess.TimeoutExpired as error:
            result = subprocess.CompletedProcess(
                args=probe_command,
                returncode=124,
                stdout=error.stdout or "",
                stderr=error.stderr or "probe timed out",
            )
        probe = _probe_from_result(
            command=probe_command,
            probe_role=probe_role,
            result=result,
            runtime=runtime,
            image=image,
            source=source,
        )
        probes.append(probe)
        if probe.accepted and probe.help_text.strip():
            return probe.help_text, tuple(probes)
    return "", tuple(probes)


def _subcommands_from_help(
    *,
    command: str,
    help_text: str,
    max_discovered_subcommands: int,
) -> list[str]:
    if not help_text.strip() or max_discovered_subcommands <= 0:
        return []
    plans = infer_suite_tool_plans(
        tool_name=command,
        help_text=help_text,
        max_suite_tools=max_discovered_subcommands,
        force_suite=True,
    )
    subcommands: list[str] = []
    command_prefix = command.strip()
    for plan in plans:
        focus = plan.command_focus.strip()
        subcommand = ""
        if command_prefix and focus.startswith(command_prefix):
            subcommand = focus[len(command_prefix) :].strip()
        elif focus and focus != command_prefix:
            subcommand = focus
        if subcommand and subcommand not in subcommands:
            subcommands.append(subcommand)
    return subcommands[:max_discovered_subcommands]


def _discover_subcommand_help(
    *,
    runner: Any,
    command: str,
    top_level_help: str,
    settings: RuntimeDiscoverySettings,
    runtime: str = "",
    image: str = "",
    source: str = "",
) -> tuple[dict[str, str], tuple[RuntimeHelpProbe, ...]]:
    if not settings.discover_subcommands:
        return {}, ()
    subcommand_help: dict[str, str] = {}
    probes: list[RuntimeHelpProbe] = []
    for subcommand in _subcommands_from_help(
        command=command,
        help_text=top_level_help,
        max_discovered_subcommands=settings.max_discovered_subcommands,
    ):
        full_command = f"{command} {subcommand}".strip()
        help_text, new_probes = _run_help_probe_sequence(
            runner=runner,
            command=full_command,
            probe_role="subcommand",
            timeout_seconds=settings.container_timeout_seconds,
            probe_mode=settings.container_help_probe_mode,
            runtime=runtime,
            image=image,
            source=source,
        )
        probes.extend(new_probes)
        if help_text.strip():
            subcommand_help[full_command] = help_text
    return subcommand_help, tuple(probes)


def _combined_help_text(top_level_help: str, subcommand_help: Mapping[str, str]) -> str:
    sections: list[str] = []
    if top_level_help.strip():
        sections.append("Runtime-discovered top-level command help:\n\n" + top_level_help.strip())
    for command, help_text in subcommand_help.items():
        if help_text.strip():
            sections.append(f"Runtime-discovered subcommand help for `{command}`:\n\n{help_text.strip()}")
    return "\n\n".join(sections)


def _resolve_sources(
    *,
    paths: WorkspacePaths,
    settings: RuntimeDiscoverySettings,
) -> tuple[tuple[Mapping[str, Any], ...], str, tuple[str, ...]]:
    package_names, requirement_versions = _package_versions_from_specs(settings.package_specs)
    if not package_names:
        return (), "", ()
    extraction_settings = ExtractionSettings(
        cache_root=paths.cache_root / "source-cache",
        bioconda_checkout_sources=True,
        bioconda_ref=settings.bioconda_ref,
        source_download_max_bytes=settings.source_download_max_bytes,
        source_download_timeout_seconds=settings.source_download_timeout_seconds,
    )
    try:
        mappings = _resolve_bioconda_source_mappings(
            package_names=package_names,
            requirement_versions=requirement_versions,
            settings=extraction_settings,
            preferred_command_hints=[settings.command] if settings.command else None,
        )
    except Exception as error:
        return (), "", (f"Source resolution failed: {error}",)
    source_root = ""
    warnings: list[str] = []
    for mapping in mappings:
        checkout = str(mapping.get("source_checkout", "") or "").strip()
        if not checkout:
            continue
        path = Path(checkout)
        if path.exists():
            source_root, warning = _source_root_from_checkout(path, settings=settings)
            if warning:
                warnings.append(warning)
            break
    return tuple(mappings), source_root, tuple(warnings)


def _source_root_from_checkout(
    path: Path,
    *,
    settings: RuntimeDiscoverySettings,
) -> tuple[str, str]:
    if path.is_dir():
        return str(path), ""
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES):
        assert settings.cache_dir is not None
        try:
            resolution = resolve_source_archive(
                str(path),
                cache_root=settings.cache_dir / "source-archives",
                max_bytes=settings.source_download_max_bytes,
                timeout_seconds=settings.source_download_timeout_seconds,
            )
            return str(Path(resolution.extracted_root).resolve()), ""
        except Exception as error:
            return str(path.parent), f"Source archive extraction failed for {path}: {error}"
    return str(path.parent), ""


def _discover_with_conda(
    *,
    paths: WorkspacePaths,
    settings: RuntimeDiscoverySettings,
) -> RuntimeDiscoveryResult:
    package_names, requirement_versions = _package_versions_from_specs(settings.package_specs)
    env_dir = _env_dir(settings)
    executable, _, env_errors = _create_or_reuse_conda_env(settings=settings, env_dir=env_dir)
    source_mappings, source_root, source_warnings = _resolve_sources(paths=paths, settings=settings)
    if env_errors:
        return RuntimeDiscoveryResult(
            mode="conda",
            selected_runtime="conda",
            command=settings.command,
            package_specs=settings.package_specs,
            package_names=tuple(package_names),
            requirement_versions=requirement_versions,
            env_dir=str(env_dir),
            conda_executable=executable,
            source_root=source_root,
            source_mappings=source_mappings,
            warnings=source_warnings,
            errors=env_errors,
        )

    def runner(command: str, timeout: int) -> subprocess.CompletedProcess:
        return _run_env_probe(
            env_dir=env_dir,
            command=command,
            timeout_seconds=timeout,
        )
    top_level_help, probes = _run_help_probe_sequence(
        runner=runner,
        command=settings.command,
        probe_role="top-level",
        timeout_seconds=settings.container_timeout_seconds,
        probe_mode=settings.container_help_probe_mode,
        runtime="conda",
        source="conda-env",
    )
    subcommand_help, subcommand_probes = _discover_subcommand_help(
        runner=runner,
        command=settings.command,
        top_level_help=top_level_help,
        settings=settings,
        runtime="conda",
        source="conda-env",
    )
    combined = _combined_help_text(top_level_help, subcommand_help)
    return RuntimeDiscoveryResult(
        mode="conda",
        selected_runtime="conda",
        command=settings.command,
        package_specs=settings.package_specs,
        package_names=tuple(package_names),
        requirement_versions=requirement_versions,
        env_dir=str(env_dir),
        conda_executable=executable,
        top_level_help=top_level_help,
        subcommand_help=subcommand_help,
        combined_help_text=combined,
        source_root=source_root,
        source_mappings=source_mappings,
        probes=(*probes, *subcommand_probes),
        warnings=source_warnings,
    )


def _container_settings(
    *,
    paths: WorkspacePaths,
    settings: RuntimeDiscoverySettings,
) -> ExtractionSettings:
    return ExtractionSettings(
        resolve_containers=True,
        execute_containers=True,
        container_runtime=settings.container_runtime,
        container_cache_dir=settings.container_cache_dir or paths.cache_root / "containers",
        docker_use_sudo=settings.docker_use_sudo,
        container_help_probe_mode=settings.container_help_probe_mode,
        container_run_timeout_seconds=settings.container_timeout_seconds,
        container_no_arg_timeout_seconds=min(settings.container_timeout_seconds, 30),
        container_image_timeout_seconds=max(settings.container_timeout_seconds, 900),
        source_download_max_bytes=settings.source_download_max_bytes,
        source_download_timeout_seconds=settings.source_download_timeout_seconds,
        cache_root=paths.cache_root / "source-cache",
        bioconda_checkout_sources=True,
        bioconda_ref=settings.bioconda_ref,
    )


def _discover_with_biocontainer(
    *,
    paths: WorkspacePaths,
    settings: RuntimeDiscoverySettings,
) -> RuntimeDiscoveryResult:
    package_names, requirement_versions = _package_versions_from_specs(settings.package_specs)
    source_mappings, source_root, source_warnings = _resolve_sources(paths=paths, settings=settings)
    extraction_settings = _container_settings(paths=paths, settings=settings)
    candidates = _build_container_candidate_details(
        container_refs=[],
        package_names=package_names,
        requirement_versions=requirement_versions,
        settings=extraction_settings,
    )
    candidate = _choose_container_candidate(
        candidates,
        requirement_versions,
        requirement_packages=package_names,
    )
    if not candidate:
        return RuntimeDiscoveryResult(
            mode="biocontainer",
            selected_runtime=settings.container_runtime,
            command=settings.command,
            package_specs=settings.package_specs,
            package_names=tuple(package_names),
            requirement_versions=requirement_versions,
            source_root=source_root,
            source_mappings=source_mappings,
            warnings=source_warnings,
            errors=("No usable Biocontainers candidate was found.",),
        )

    runtimes = _available_container_runtimes(extraction_settings)
    if not runtimes:
        return RuntimeDiscoveryResult(
            mode="biocontainer",
            selected_runtime=settings.container_runtime,
            command=settings.command,
            package_specs=settings.package_specs,
            package_names=tuple(package_names),
            requirement_versions=requirement_versions,
            container_image=str(candidate.get("image", "") or ""),
            source_root=source_root,
            source_mappings=source_mappings,
            warnings=source_warnings,
            errors=(f"No available container runtime for {settings.container_runtime}.",),
        )

    errors: list[str] = []
    preparation: ContainerPreparation | None = None
    selected_candidate = candidate
    for candidate_item in _sort_container_candidates_for_selection(
        candidates,
        requirement_packages=package_names,
        requirement_versions=requirement_versions,
    ):
        image = str(candidate_item.get("image", "") or "")
        if not image:
            continue
        for runtime in runtimes:
            prepared = _prepare_container(image=image, runtime=runtime, settings=extraction_settings)
            if prepared.ok:
                preparation = prepared
                selected_candidate = candidate_item
                break
            errors.append(f"{runtime.name}:{image}: {prepared.error_text or prepared.returncode}")
        if preparation is not None:
            break

    if preparation is None:
        return RuntimeDiscoveryResult(
            mode="biocontainer",
            selected_runtime=settings.container_runtime,
            command=settings.command,
            package_specs=settings.package_specs,
            package_names=tuple(package_names),
            requirement_versions=requirement_versions,
            container_image=str(candidate.get("image", "") or ""),
            source_root=source_root,
            source_mappings=source_mappings,
            warnings=source_warnings,
            errors=tuple(errors[-8:] or ["Container preparation failed."]),
        )

    def runner(command: str, timeout: int) -> subprocess.CompletedProcess:
        primary = _command_primary(command)
        if primary:
            presence = _run_container_command_exists(
                preparation=preparation,
                primary=primary,
                settings=extraction_settings,
            )
            if presence.returncode != 0:
                return presence
        return _run_container_probe(
            preparation=preparation,
            command=command,
            settings=extraction_settings,
        )

    top_level_help, probes = _run_help_probe_sequence(
        runner=runner,
        command=settings.command,
        probe_role="top-level",
        timeout_seconds=settings.container_timeout_seconds,
        probe_mode=settings.container_help_probe_mode,
        runtime=preparation.runtime,
        image=str(selected_candidate.get("image", "") or preparation.image),
        source=preparation.source,
    )
    subcommand_help, subcommand_probes = _discover_subcommand_help(
        runner=runner,
        command=settings.command,
        top_level_help=top_level_help,
        settings=settings,
        runtime=preparation.runtime,
        image=str(selected_candidate.get("image", "") or preparation.image),
        source=preparation.source,
    )
    combined = _combined_help_text(top_level_help, subcommand_help)
    return RuntimeDiscoveryResult(
        mode="biocontainer",
        selected_runtime=preparation.runtime,
        command=settings.command,
        package_specs=settings.package_specs,
        package_names=tuple(package_names),
        requirement_versions=requirement_versions,
        container_image=str(selected_candidate.get("image", "") or preparation.image),
        container_identifier=preparation.identifier,
        container_source=preparation.source,
        top_level_help=top_level_help,
        subcommand_help=subcommand_help,
        combined_help_text=combined,
        source_root=source_root,
        source_mappings=source_mappings,
        probes=(*probes, *subcommand_probes),
        warnings=source_warnings,
        errors=tuple(errors[-8:]),
    )
