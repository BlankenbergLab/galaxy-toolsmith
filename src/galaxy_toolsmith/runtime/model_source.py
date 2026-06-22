from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelSourcePolicy:
    registry: str
    revision: str
    cache_dir: str
    hub_cache_dir: str
    transformers_cache_dir: str
    cache_dir_source: str
    local_files_only: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_environment(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.registry:
            env["HF_ENDPOINT"] = self.registry
        if self.cache_dir:
            env["HF_HOME"] = self.cache_dir
        if self.hub_cache_dir:
            env["HUGGINGFACE_HUB_CACHE"] = self.hub_cache_dir
        if self.transformers_cache_dir:
            env["TRANSFORMERS_CACHE"] = self.transformers_cache_dir
        if self.local_files_only:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def load_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.revision:
            kwargs["revision"] = self.revision
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.local_files_only:
            kwargs["local_files_only"] = True
        return kwargs


def _path_from_env(name: str) -> str:
    return os.getenv(name, "").strip()


def _workspace_default_cache_dir(paths: Any | None) -> str:
    if paths is None:
        return ""
    models_root = getattr(paths, "models_root", None)
    if models_root is None:
        return ""
    return str(Path(models_root) / "hf-cache")


def resolve_model_source_policy(paths: Any | None = None) -> ModelSourcePolicy:
    registry = _path_from_env("GTSM_MODEL_SOURCE_REGISTRY") or _path_from_env("HF_ENDPOINT")
    revision = _path_from_env("GTSM_MODEL_REVISION")

    if _path_from_env("GTSM_MODEL_CACHE_ROOT"):
        cache_dir = _path_from_env("GTSM_MODEL_CACHE_ROOT")
        cache_dir_source = "GTSM_MODEL_CACHE_ROOT"
    elif _path_from_env("HF_HOME"):
        cache_dir = _path_from_env("HF_HOME")
        cache_dir_source = "HF_HOME"
    else:
        cache_dir = _workspace_default_cache_dir(paths)
        cache_dir_source = "workspace-default" if cache_dir else ""

    hub_cache_dir = _path_from_env("HUGGINGFACE_HUB_CACHE") or (
        str(Path(cache_dir) / "hub") if cache_dir else ""
    )
    transformers_cache_dir = _path_from_env("TRANSFORMERS_CACHE") or (
        str(Path(cache_dir) / "transformers") if cache_dir else ""
    )
    local_files_only = _env_bool("GTSM_MODEL_LOCAL_FILES_ONLY", default=False)
    return ModelSourcePolicy(
        registry=registry,
        revision=revision,
        cache_dir=cache_dir,
        hub_cache_dir=hub_cache_dir,
        transformers_cache_dir=transformers_cache_dir,
        cache_dir_source=cache_dir_source,
        local_files_only=local_files_only,
    )


def model_source_load_kwargs(source_policy: object) -> dict:
    if isinstance(source_policy, ModelSourcePolicy):
        return source_policy.load_kwargs()
    load_kwargs: dict = {}
    revision = str(getattr(source_policy, "revision", "") or "")
    cache_dir = str(getattr(source_policy, "cache_dir", "") or "")
    local_files_only = bool(getattr(source_policy, "local_files_only", False))
    if revision:
        load_kwargs["revision"] = revision
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        load_kwargs["local_files_only"] = True
    return load_kwargs


def model_source_environment(source_policy: object) -> dict[str, str]:
    if isinstance(source_policy, ModelSourcePolicy):
        return source_policy.to_environment()
    env: dict[str, str] = {}
    registry = str(getattr(source_policy, "registry", "") or "")
    cache_dir = str(getattr(source_policy, "cache_dir", "") or "")
    hub_cache_dir = str(getattr(source_policy, "hub_cache_dir", "") or "")
    transformers_cache_dir = str(getattr(source_policy, "transformers_cache_dir", "") or "")
    local_files_only = bool(getattr(source_policy, "local_files_only", False))
    if registry:
        env["HF_ENDPOINT"] = registry
    if cache_dir:
        env["HF_HOME"] = cache_dir
    if hub_cache_dir:
        env["HUGGINGFACE_HUB_CACHE"] = hub_cache_dir
    if transformers_cache_dir:
        env["TRANSFORMERS_CACHE"] = transformers_cache_dir
    if local_files_only:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def merged_model_source_environment(
    base_env: dict[str, str] | None,
    source_policy: object,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.update(model_source_environment(source_policy))
    return env


def apply_model_source_environment(source_policy: object) -> None:
    os.environ.update(model_source_environment(source_policy))


def model_cache_info(paths: Any) -> dict:
    policy = resolve_model_source_policy(paths)
    cache_path = Path(policy.cache_dir) if policy.cache_dir else None
    return {
        "model_source_policy": policy.to_dict(),
        "cache_root": str(cache_path) if cache_path else "",
        "cache_exists": cache_path.exists() if cache_path else False,
        "cache_size_bytes": _directory_size(cache_path) if cache_path and cache_path.exists() else 0,
        "environment": {
            "GTSM_MODEL_CACHE_ROOT": _path_from_env("GTSM_MODEL_CACHE_ROOT"),
            "HF_HOME": _path_from_env("HF_HOME"),
            "HUGGINGFACE_HUB_CACHE": _path_from_env("HUGGINGFACE_HUB_CACHE"),
            "TRANSFORMERS_CACHE": _path_from_env("TRANSFORMERS_CACHE"),
            "HF_TOKEN": "set" if _path_from_env("HF_TOKEN") else "",
        },
    }


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total
