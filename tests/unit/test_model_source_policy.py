from __future__ import annotations

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.runtime.model_source import (
    merged_model_source_environment,
    model_cache_info,
    resolve_model_source_policy,
)


def test_resolve_model_source_policy_defaults_to_workspace_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GTSM_MODEL_SOURCE_REGISTRY", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("GTSM_MODEL_REVISION", raising=False)
    monkeypatch.delenv("GTSM_MODEL_CACHE_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("GTSM_MODEL_LOCAL_FILES_ONLY", raising=False)
    paths = WorkspacePaths.from_repo_root(tmp_path)
    policy = resolve_model_source_policy(paths)
    assert policy.registry == ""
    assert policy.revision == ""
    assert policy.cache_dir == str(paths.models_root / "hf-cache")
    assert policy.hub_cache_dir == str(paths.models_root / "hf-cache" / "hub")
    assert policy.transformers_cache_dir == str(paths.models_root / "hf-cache" / "transformers")
    assert policy.cache_dir_source == "workspace-default"
    assert policy.local_files_only is False


def test_resolve_model_source_policy_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GTSM_MODEL_SOURCE_REGISTRY", "https://example.registry")
    monkeypatch.setenv("GTSM_MODEL_REVISION", "main")
    monkeypatch.setenv("GTSM_MODEL_CACHE_ROOT", "/tmp/model-cache")
    monkeypatch.setenv("GTSM_MODEL_LOCAL_FILES_ONLY", "true")
    policy = resolve_model_source_policy()
    assert policy.registry == "https://example.registry"
    assert policy.revision == "main"
    assert policy.cache_dir == "/tmp/model-cache"
    assert policy.hub_cache_dir == "/tmp/model-cache/hub"
    assert policy.transformers_cache_dir == "/tmp/model-cache/transformers"
    assert policy.cache_dir_source == "GTSM_MODEL_CACHE_ROOT"
    assert policy.local_files_only is True
    assert policy.load_kwargs() == {
        "revision": "main",
        "cache_dir": "/tmp/model-cache",
        "local_files_only": True,
    }
    assert policy.to_environment()["HF_HOME"] == "/tmp/model-cache"
    assert policy.to_environment()["HF_HUB_OFFLINE"] == "1"


def test_hf_home_used_when_gtsm_cache_root_unset(monkeypatch) -> None:
    monkeypatch.delenv("GTSM_MODEL_CACHE_ROOT", raising=False)
    monkeypatch.setenv("HF_HOME", "/tmp/hf-home")

    policy = resolve_model_source_policy()

    assert policy.cache_dir == "/tmp/hf-home"
    assert policy.cache_dir_source == "HF_HOME"


def test_model_cache_info_reports_size(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GTSM_MODEL_CACHE_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    paths = WorkspacePaths.from_repo_root(tmp_path)
    cache_root = paths.models_root / "hf-cache"
    cache_root.mkdir(parents=True)
    (cache_root / "weights.bin").write_bytes(b"abc")

    info = model_cache_info(paths)

    assert info["cache_root"] == str(cache_root)
    assert info["cache_exists"] is True
    assert info["cache_size_bytes"] == 3


def test_merged_model_source_environment_includes_cache_and_offline(monkeypatch) -> None:
    monkeypatch.setenv("GTSM_MODEL_CACHE_ROOT", "/tmp/model-cache")
    monkeypatch.setenv("GTSM_MODEL_LOCAL_FILES_ONLY", "true")
    policy = resolve_model_source_policy()

    env = merged_model_source_environment({"KEEP": "1"}, policy)

    assert env["KEEP"] == "1"
    assert env["HF_HOME"] == "/tmp/model-cache"
    assert env["HUGGINGFACE_HUB_CACHE"] == "/tmp/model-cache/hub"
    assert env["TRANSFORMERS_CACHE"] == "/tmp/model-cache/transformers"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
