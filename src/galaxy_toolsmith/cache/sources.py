from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths

TOOLS_IUC_URL = "https://github.com/galaxyproject/tools-iuc.git"
GALAXY_SKILLS_URL = "https://github.com/galaxyproject/galaxy-skills.git"


@dataclass(frozen=True)
class SyncResult:
    name: str
    path: Path
    revision: str
    cloned: bool


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def sync_git_repo(name: str, url: str, ref: str, paths: WorkspacePaths) -> SyncResult:
    destination = paths.source_cache / name
    cloned = False

    if destination.exists():
        _run_git(["fetch", "--all", "--tags", "--prune"], cwd=destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", url, str(destination)])
        cloned = True

    _run_git(["checkout", ref], cwd=destination)
    revision = _run_git(["rev-parse", "HEAD"], cwd=destination)
    return SyncResult(name=name, path=destination, revision=revision, cloned=cloned)


def sync_tools_iuc(paths: WorkspacePaths, ref: str = "main") -> SyncResult:
    return sync_git_repo(name="tools-iuc", url=TOOLS_IUC_URL, ref=ref, paths=paths)


def sync_galaxy_skills(paths: WorkspacePaths, ref: str = "main") -> SyncResult:
    return sync_git_repo(name="galaxy-skills", url=GALAXY_SKILLS_URL, ref=ref, paths=paths)
