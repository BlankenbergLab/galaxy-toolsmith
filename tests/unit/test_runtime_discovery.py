from __future__ import annotations

import subprocess
from pathlib import Path

import galaxy_toolsmith.inference.runtime_discovery as runtime_discovery
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.runtime_discovery import (
    RuntimeDiscoveryResult,
    RuntimeDiscoverySettings,
    discover_runtime_context,
)


def test_conda_discovery_collects_top_level_subcommand_help_and_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "alpha-src"
    source_root.mkdir()
    (source_root / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")

    def fake_sources(**kwargs):
        assert kwargs["package_names"] == ["alpha"]
        assert kwargs["requirement_versions"] == {"alpha": "1.0"}
        return [{"package": "alpha", "source_checkout": str(source_root)}]

    def fake_run(command, **kwargs):
        if isinstance(command, list):
            return subprocess.CompletedProcess(command, 0, stdout="created", stderr="")
        if command == "alpha --help":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="Usage: alpha {view,sort}\n\nCommands:\n  view  View reads\n  sort  Sort reads\n",
                stderr="",
            )
        if command == "alpha view --help":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="Usage: alpha view --input reads.bam\nOptions:\n  --input FILE\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="not help")

    monkeypatch.setattr(runtime_discovery, "_resolve_bioconda_source_mappings", fake_sources)
    monkeypatch.setattr(runtime_discovery.subprocess, "run", fake_run)

    paths = WorkspacePaths.from_repo_root(tmp_path)
    result = discover_runtime_context(
        paths=paths,
        settings=RuntimeDiscoverySettings(
            mode="conda",
            package_specs=("alpha=1.0",),
            command="alpha",
            cache_dir=tmp_path / "cache",
            conda_executable="/bin/true",
            max_discovered_subcommands=1,
        ),
    )

    assert result.selected_runtime == "conda"
    assert result.source_root == str(source_root)
    assert "Usage: alpha {view,sort}" in result.combined_help_text
    assert "Usage: alpha view --input" in result.combined_help_text
    assert result.subcommand_help["alpha view"].startswith("$ alpha view --help")
    assert [probe.command for probe in result.probes if probe.accepted] == [
        "alpha --help",
        "alpha view --help",
    ]


def test_auto_discovery_falls_back_to_biocontainer_when_conda_has_no_help(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)

    def fake_conda(*, paths, settings):
        return RuntimeDiscoveryResult(
            mode="conda",
            selected_runtime="conda",
            command=settings.command,
            errors=("no accepted conda help",),
        )

    def fake_container(*, paths, settings):
        return RuntimeDiscoveryResult(
            mode="biocontainer",
            selected_runtime="singularity",
            command=settings.command,
            combined_help_text="Runtime-discovered top-level command help:\n\nUsage: alpha",
        )

    monkeypatch.setattr(runtime_discovery, "_discover_with_conda", fake_conda)
    monkeypatch.setattr(runtime_discovery, "_discover_with_biocontainer", fake_container)

    result = discover_runtime_context(
        paths=paths,
        settings=RuntimeDiscoverySettings(
            mode="auto",
            package_specs=("alpha",),
            command="alpha",
        ),
    )

    assert result.mode == "auto"
    assert result.selected_runtime == "singularity"
    assert result.has_help is True
    assert "tried Biocontainers fallback" in " ".join(result.warnings)
