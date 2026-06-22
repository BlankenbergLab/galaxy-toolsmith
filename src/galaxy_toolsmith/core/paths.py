from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved repository-relative paths used by Galaxy Toolsmith."""

    repo_root: Path
    cache_root: Path
    source_cache: Path
    datasets_root: Path
    runs_root: Path
    models_root: Path
    xsd_root: Path
    configs_root: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> WorkspacePaths:
        repo_root = repo_root.resolve()
        cache_root = repo_root / ".gtsm-cache"
        return cls(
            repo_root=repo_root,
            cache_root=cache_root,
            source_cache=cache_root / "sources",
            datasets_root=cache_root / "datasets",
            runs_root=cache_root / "runs",
            models_root=cache_root / "models",
            xsd_root=cache_root / "xsd",
            configs_root=repo_root / "config",
        )

    def create_directories(self) -> None:
        for path in (
            self.cache_root,
            self.source_cache,
            self.datasets_root,
            self.runs_root,
            self.models_root,
            self.xsd_root,
            self.configs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
