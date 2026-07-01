from __future__ import annotations

import builtins
import json
import subprocess
from pathlib import Path

import pytest

from galaxy_toolsmith.core.manifests import ModelVariantManifest
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.orchestration import export as export_mod
from galaxy_toolsmith.orchestration.export import (
    ExportResult,
    export_model_artifacts,
    update_variant_ollama_metadata,
    write_ollama_modelfile,
)


def test_export_model_artifacts_merged(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()

    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model.bin").write_text("dummy-model", encoding="utf-8")

    variants_dir = paths.models_root / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    variant = ModelVariantManifest(
        variant_id="variant-a",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization="4bit",
        training_dataset_id="d1",
        provider="local",
        skills_profile="default",
        backend="axolotl",
        artifact_dir=str(artifact_dir),
    )
    (variants_dir / "variant-a.manifest.json").write_text(variant.to_json(), encoding="utf-8")

    result = export_model_artifacts(paths=paths, variant_id="variant-a", export_format="merged")
    assert result.status == "completed"
    assert result.quantizations == ["q4_k_m"]
    assert result.merged_path
    assert result.adapter_export_path == ""
    assert Path(result.merged_path).exists()
    assert (Path(result.merged_path) / "model.bin").exists()


def test_export_model_artifacts_merges_peft_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True)
    (paths.models_root / "variants").mkdir(parents=True)
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    (paths.models_root / "variants" / "variant-a.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-a",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            quantization="none",
            training_dataset_id="d1",
            provider="local",
            skills_profile="default",
            backend="axolotl",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_merge(
        variant: dict,
        artifact_dir: Path,
        merged_dir: Path,
        source_policy: object | None = None,
    ) -> None:
        assert source_policy is not None
        calls.append(artifact_dir)
        merged_dir.mkdir(parents=True)
        (merged_dir / "model.safetensors").write_text(
            str(variant["base_model"]), encoding="utf-8"
        )

    monkeypatch.setattr(export_mod, "_merge_peft_adapter", fake_merge)

    result = export_model_artifacts(paths=paths, variant_id="variant-a", export_format="merged")

    assert result.status == "completed"
    assert calls == [artifact_dir]
    assert result.merged_path
    assert result.adapter_export_path
    assert result.model_source_policy["cache_dir"] == str(paths.models_root / "hf-cache")
    assert (Path(result.merged_path) / "model.safetensors").exists()
    assert (Path(result.adapter_export_path) / "adapter_model.safetensors").exists()


def test_export_model_artifacts_skips_peft_merge_for_mlx_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-mlx"
    artifact_dir.mkdir(parents=True)
    (paths.models_root / "variants").mkdir(parents=True)
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "adapters.safetensors").write_text("adapter", encoding="utf-8")
    (paths.models_root / "variants" / "variant-mlx.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-mlx",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            backend="mlx-lm",
            artifact_kind="mlx_adapter",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )

    def fail_merge(*args, **kwargs) -> None:
        raise AssertionError("MLX artifacts must not use PEFT merge")

    monkeypatch.setattr(export_mod, "_merge_peft_adapter", fail_merge)

    result = export_model_artifacts(paths=paths, variant_id="variant-mlx", export_format="merged")

    assert result.status == "partial"
    assert result.merged_path == ""
    assert any("MLX artifacts" in note for note in result.notes)


def test_export_model_artifacts_reports_partial_when_gguf_backend_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True)
    (paths.models_root / "variants").mkdir(parents=True)
    (artifact_dir / "model.safetensors").write_text("model", encoding="utf-8")
    (paths.models_root / "variants" / "variant-a.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-a",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "unsloth":
            raise ImportError("no unsloth")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = export_model_artifacts(paths=paths, variant_id="variant-a", export_format="gguf")

    assert result.status == "partial"
    assert result.gguf_path == ""
    assert any("GGUF export skipped" in note for note in result.notes)


def test_export_model_artifacts_uses_user_space_llama_cpp_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True)
    (paths.models_root / "variants").mkdir(parents=True)
    (artifact_dir / "model.safetensors").write_text("model", encoding="utf-8")
    (paths.models_root / "variants" / "variant-a.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-a",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )

    llama_dir = tmp_path / "llama.cpp"
    quantizer = llama_dir / "build" / "bin" / "llama-quantize"
    quantizer.parent.mkdir(parents=True)
    (llama_dir / "convert_hf_to_gguf.py").write_text(
        """
from __future__ import annotations

import sys
from pathlib import Path

outfile = Path(sys.argv[sys.argv.index("--outfile") + 1])
outfile.parent.mkdir(parents=True, exist_ok=True)
outfile.write_text("base gguf", encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    quantizer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
method = sys.argv[3]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(source.read_text(encoding="utf-8") + "\\n" + method, encoding="utf-8")
""",
        encoding="utf-8",
    )
    quantizer.chmod(0o755)

    monkeypatch.setenv("GTSM_GGUF_BACKEND", "llama.cpp")
    monkeypatch.setenv("GTSM_LLAMA_CPP_DIR", str(llama_dir))

    result = export_model_artifacts(
        paths=paths,
        variant_id="variant-a",
        export_format="gguf",
        quantizations=["q4_k_m"],
    )

    assert result.status == "completed"
    assert result.gguf_path
    assert result.gguf_paths == {"q4_k_m": result.gguf_path}
    assert Path(result.gguf_path).exists()
    assert Path(result.gguf_path).read_text(encoding="utf-8").endswith("Q4_K_M")


def test_export_model_artifacts_reuses_existing_merged_model_for_adapter_gguf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    (paths.models_root / "variants").mkdir(parents=True)
    (paths.models_root / "variants" / "variant-a.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-a",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )
    merged_dir = paths.models_root / "exports" / "variant-a" / "merged"
    merged_dir.mkdir(parents=True)
    (merged_dir / "config.json").write_text("{}", encoding="utf-8")
    (merged_dir / "model.safetensors").write_text("merged", encoding="utf-8")

    def fail_merge(*args, **kwargs) -> None:
        raise AssertionError("existing merged model should be reused")

    def fake_gguf_export(
        *,
        model_dir: Path,
        gguf_dir: Path,
        variant_id: str,
        quant_methods: list[str],
    ) -> dict[str, str]:
        assert model_dir == merged_dir
        output = gguf_dir / quant_methods[0] / f"{variant_id}-{quant_methods[0]}.gguf"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("merged gguf", encoding="utf-8")
        return {quant_methods[0]: str(output)}

    monkeypatch.setattr(export_mod, "_merge_peft_adapter", fail_merge)
    monkeypatch.setattr(export_mod, "_export_gguf_with_llama_cpp", fake_gguf_export)
    monkeypatch.setenv("GTSM_GGUF_BACKEND", "llama.cpp")

    result = export_model_artifacts(
        paths=paths,
        variant_id="variant-a",
        export_format="gguf",
        quantizations=["q4_k_m"],
    )

    assert result.status == "completed"
    assert result.merged_path == str(merged_dir)
    assert result.gguf_path
    assert any("Reusing existing merged export" in note for note in result.notes)


def test_export_model_artifacts_does_not_convert_unmerged_adapter_to_gguf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    (paths.models_root / "variants").mkdir(parents=True)
    (paths.models_root / "variants" / "variant-a.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-a",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )
    gguf_calls: list[Path] = []

    def fail_merge(*args, **kwargs) -> None:
        raise RuntimeError("merge unavailable")

    def fake_gguf_export(
        *,
        model_dir: Path,
        gguf_dir: Path,
        variant_id: str,
        quant_methods: list[str],
    ) -> dict[str, str]:
        gguf_calls.append(model_dir)
        return {}

    monkeypatch.setattr(export_mod, "_merge_peft_adapter", fail_merge)
    monkeypatch.setattr(export_mod, "_export_gguf_with_llama_cpp", fake_gguf_export)
    monkeypatch.setenv("GTSM_GGUF_BACKEND", "llama.cpp")

    result = export_model_artifacts(
        paths=paths,
        variant_id="variant-a",
        export_format="gguf",
        quantizations=["q4_k_m"],
    )

    assert result.status == "partial"
    assert result.gguf_path == ""
    assert gguf_calls == []
    assert any("PEFT adapter export must be merged first" in note for note in result.notes)


def test_write_ollama_modelfile_uses_exported_gguf(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    export_root = paths.models_root / "exports" / "variant-a"
    gguf_file = export_root / "gguf" / "q4_k_m" / "variant-a.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_text("dummy-gguf", encoding="utf-8")
    export_result = {
        "variant_id": "variant-a",
        "format": "gguf",
        "quantizations": ["q4_k_m"],
        "merged_path": "",
        "gguf_path": str(gguf_file),
        "gguf_paths": {"q4_k_m": str(gguf_file)},
        "status": "completed",
        "notes": [],
    }
    (export_root / "export.result.json").write_text(json.dumps(export_result), encoding="utf-8")

    modelfile = write_ollama_modelfile(
        paths=paths,
        variant_id="variant-a",
        model_name="galaxy-toolsmith-test",
        from_quantization="q4_k_m",
    )
    assert modelfile.exists()
    content = modelfile.read_text(encoding="utf-8")
    assert "FROM " in content
    assert str(gguf_file) in content


def test_write_ollama_modelfile_preserves_non_ascii_metadata(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    variant_id = "väriant-a"
    export_root = paths.models_root / "exports" / variant_id
    gguf_file = export_root / "gguf" / "q4_k_m" / "mödèle.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_text("dummy-gguf", encoding="utf-8")
    export_result = {
        "variant_id": variant_id,
        "format": "gguf",
        "quantizations": ["q4_k_m"],
        "merged_path": "",
        "gguf_path": str(gguf_file),
        "gguf_paths": {"q4_k_m": str(gguf_file)},
        "status": "completed",
        "notes": [],
    }
    export_result_path = export_root / "export.result.json"
    export_result_path.write_text(json.dumps(export_result), encoding="utf-8")

    modelfile = write_ollama_modelfile(
        paths=paths,
        variant_id=variant_id,
        model_name="gtsm-mödèle",
        from_quantization="q4_k_m",
    )

    assert f"FROM {gguf_file}" in modelfile.read_text(encoding="utf-8")
    updated_json = export_result_path.read_text(encoding="utf-8")
    assert "gtsm-mödèle" in updated_json
    assert "\\u" not in updated_json


def test_write_ollama_modelfile_rejects_whitespace_gguf_paths(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    export_root = paths.models_root / "exports" / "variant-a"
    gguf_file = export_root / "gguf with space" / "variant-a.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_text("dummy-gguf", encoding="utf-8")
    export_result = {
        "variant_id": "variant-a",
        "format": "gguf",
        "quantizations": ["q4_k_m"],
        "merged_path": "",
        "gguf_path": str(gguf_file),
        "gguf_paths": {"q4_k_m": str(gguf_file)},
        "status": "completed",
        "notes": [],
    }
    (export_root / "export.result.json").write_text(json.dumps(export_result), encoding="utf-8")

    with pytest.raises(RuntimeError, match="whitespace"):
        write_ollama_modelfile(
            paths=paths,
            variant_id="variant-a",
            model_name="galaxy-toolsmith-test",
            from_quantization="q4_k_m",
        )


def test_export_result_json_preserves_non_ascii() -> None:
    result = ExportResult(
        variant_id="väriant-a",
        format="gguf",
        quantizations=["q4_k_m"],
        merged_path="",
        adapter_export_path="",
        gguf_path="/tmp/mödèle.gguf",
        gguf_paths={"q4_k_m": "/tmp/mödèle.gguf"},
        status="completed",
        notes=["créé"],
    )

    text = result.to_json()

    assert "väriant-a" in text
    assert "mödèle.gguf" in text
    assert "\\u" not in text


def test_update_variant_ollama_metadata_preserves_non_ascii(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    variant_id = "väriant-a"
    artifact_dir = paths.models_root / "artifacts" / variant_id
    artifact_dir.mkdir(parents=True)
    (paths.models_root / "variants").mkdir(parents=True)
    (paths.models_root / "variants" / f"{variant_id}.manifest.json").write_text(
        ModelVariantManifest(
            variant_id=variant_id,
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )

    variant_path = update_variant_ollama_metadata(
        paths=paths,
        variant_id=variant_id,
        ollama_model_name="gtsm-mödèle",
        ollama_modelfile_path="/tmp/Mödelfile",
        export_quantizations=["q4_k_m"],
    )

    text = variant_path.read_text(encoding="utf-8")
    assert "gtsm-mödèle" in text
    assert "/tmp/Mödelfile" in text
    assert "\\u" not in text


def test_llama_cpp_subprocesses_decode_utf8(monkeypatch, tmp_path: Path) -> None:
    llama_dir = tmp_path / "llama.cpp"
    converter = llama_dir / "convert_hf_to_gguf.py"
    quantizer = llama_dir / "build" / "bin" / "llama-quantize"
    converter.parent.mkdir(parents=True)
    quantizer.parent.mkdir(parents=True)
    converter.write_text("", encoding="utf-8")
    quantizer.write_text("", encoding="utf-8")
    monkeypatch.setenv("GTSM_LLAMA_CPP_DIR", str(llama_dir))
    observed_kwargs: list[dict] = []

    def fake_run(command, **kwargs):
        observed_kwargs.append(kwargs)
        if "--outfile" in command:
            output = Path(command[command.index("--outfile") + 1])
        else:
            output = Path(command[2])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("gguf", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="créé", stderr="")

    monkeypatch.setattr(export_mod.subprocess, "run", fake_run)

    result = export_mod._export_gguf_with_llama_cpp(
        model_dir=tmp_path / "model",
        gguf_dir=tmp_path / "gguf",
        variant_id="väriant-a",
        quant_methods=["q4_k_m"],
    )

    assert result == {"q4_k_m": str(tmp_path / "gguf" / "q4_k_m" / "väriant-a-q4_k_m.gguf")}
    assert observed_kwargs
    assert all(kwargs["encoding"] == "utf-8" for kwargs in observed_kwargs)
    assert all(kwargs["errors"] == "replace" for kwargs in observed_kwargs)


def test_create_ollama_model_decodes_utf8(monkeypatch, tmp_path: Path) -> None:
    observed_kwargs = {}

    def fake_run(command, **kwargs):
        observed_kwargs.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="créé\n", stderr="")

    monkeypatch.setattr(export_mod.subprocess, "run", fake_run)

    payload = export_mod.create_ollama_model(tmp_path / "Modelfile", "gtsm-mödèle")

    assert payload["stdout"] == "créé"
    assert observed_kwargs["encoding"] == "utf-8"
    assert observed_kwargs["errors"] == "replace"


def test_create_ollama_model_uses_configured_cli(monkeypatch, tmp_path: Path) -> None:
    observed_command = []

    def fake_run(command, **kwargs):
        observed_command.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("GTSM_OLLAMA_CLI", "/opt/ollama/bin/ollama")
    monkeypatch.setattr(export_mod.subprocess, "run", fake_run)

    export_mod.create_ollama_model(tmp_path / "Modelfile", "gtsm-test")

    assert observed_command[:2] == ["/opt/ollama/bin/ollama", "create"]
