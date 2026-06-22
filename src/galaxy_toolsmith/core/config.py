from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from galaxy_toolsmith.core.paths import WorkspacePaths


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "auto"
    max_workers: int = 4
    retry_limit: int = 5


@dataclass(frozen=True)
class ProviderConfig:
    local_enabled: bool = True
    external_enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig = RuntimeConfig()
    providers: ProviderConfig = ProviderConfig()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def config_path(paths: WorkspacePaths) -> Path:
    return paths.configs_root / "gtsm.config.json"


def write_default_config(paths: WorkspacePaths) -> Path:
    destination = config_path(paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(AppConfig().to_json(), encoding="utf-8")
    return destination
