from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.runtime.model_source import (
    apply_model_source_environment,
    model_source_load_kwargs,
    resolve_model_source_policy,
)


@dataclass(frozen=True)
class ExportResult:
    variant_id: str
    format: str
    quantizations: list[str]
    merged_path: str
    adapter_export_path: str
    gguf_path: str
    gguf_paths: dict[str, str]
    status: str
    notes: list[str]
    model_source_policy: dict = field(default_factory=dict)
    ollama_modelfile_path: str = ""
    ollama_model_name: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _variant_manifest_path(paths: WorkspacePaths, variant_id: str) -> Path:
    return paths.models_root / "variants" / f"{variant_id}.manifest.json"


def _load_variant(paths: WorkspacePaths, variant_id: str) -> dict:
    path = _variant_manifest_path(paths, variant_id)
    if not path.exists():
        raise FileNotFoundError(f"Variant manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_variant_ollama_metadata(
    paths: WorkspacePaths,
    variant_id: str,
    *,
    ollama_model_name: str,
    ollama_modelfile_path: str,
    export_quantizations: list[str] | None = None,
) -> Path:
    path = _variant_manifest_path(paths, variant_id)
    variant = _load_variant(paths, variant_id)
    variant["ollama_model_name"] = ollama_model_name
    variant["ollama_modelfile_path"] = ollama_modelfile_path
    if export_quantizations:
        variant["export_quantizations"] = list(export_quantizations)
    path.write_text(json.dumps(variant, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _is_peft_adapter(artifact_dir: Path) -> bool:
    return (artifact_dir / "adapter_config.json").exists()


def _is_hf_model_export(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    model_patterns = (
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
    )
    return any(any(model_dir.glob(pattern)) for pattern in model_patterns)


def _copy_adapter_export(artifact_dir: Path, export_root: Path) -> str:
    adapter_dir = export_root / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    shutil.copytree(artifact_dir, adapter_dir)
    return str(adapter_dir)


def _merge_peft_adapter(
    variant: dict,
    artifact_dir: Path,
    merged_dir: Path,
    source_policy: object | None = None,
) -> None:
    base_model = str(variant.get("base_model", "")).strip()
    if not base_model:
        raise RuntimeError("Variant manifest lacks base_model; cannot merge PEFT adapter.")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:  # pragma: no cover - optional runtime dependency path
        raise RuntimeError(
            "Merged export requires torch, transformers, and peft. Install training deps."
        ) from error

    if source_policy is not None:
        apply_model_source_environment(source_policy)
    source_kwargs = model_source_load_kwargs(source_policy) if source_policy is not None else {}
    load_kwargs: dict = {"trust_remote_code": True, **source_kwargs}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, **source_kwargs)
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model = PeftModel.from_pretrained(model, str(artifact_dir))
    model = model.merge_and_unload()
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))


def _path_from_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _llama_cpp_dir() -> Path | None:
    path = _path_from_env("GTSM_LLAMA_CPP_DIR")
    if path and path.exists():
        return path
    return None


def _llama_cpp_converter(llama_dir: Path | None) -> Path | None:
    configured = _path_from_env("GTSM_LLAMA_CPP_CONVERT")
    if configured and configured.is_file():
        return configured
    if llama_dir is None:
        return None
    candidate = llama_dir / "convert_hf_to_gguf.py"
    return candidate if candidate.is_file() else None


def _llama_cpp_quantizer(llama_dir: Path | None) -> Path | None:
    configured = _path_from_env("GTSM_LLAMA_CPP_QUANTIZE")
    if configured and configured.is_file():
        return configured
    if llama_dir is None:
        return None
    for relative in (
        "build/bin/llama-quantize",
        "build/bin/quantize",
        "bin/llama-quantize",
        "bin/quantize",
        "llama-quantize",
        "quantize",
    ):
        candidate = llama_dir / relative
        if candidate.is_file():
            return candidate
    return None


def _llama_cpp_env(llama_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if llama_dir is None:
        return env
    pythonpath = [str(llama_dir)]
    gguf_py = llama_dir / "gguf-py"
    if gguf_py.exists():
        pythonpath.append(str(gguf_py))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _llama_cpp_quantization_method(method: str) -> str:
    return method.strip().upper()


def _tail_process_error(error: subprocess.CalledProcessError) -> str:
    detail = ((error.stderr or "").strip() or (error.stdout or "").strip())[-1200:]
    if detail:
        return detail
    return f"exit code {error.returncode}"


def _ollama_from_path(gguf_path: str) -> str:
    path = str(gguf_path).strip()
    if not path:
        raise RuntimeError("No GGUF artifact available for Ollama Modelfile generation.")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise RuntimeError("Ollama Modelfile FROM path cannot contain control characters.")
    if any(character.isspace() for character in path):
        raise RuntimeError(
            "Ollama Modelfile FROM path cannot contain whitespace; move or symlink the GGUF "
            "artifact to a path without spaces before generating the Modelfile."
        )
    return path


def _export_gguf_with_llama_cpp(
    *,
    model_dir: Path,
    gguf_dir: Path,
    variant_id: str,
    quant_methods: list[str],
) -> dict[str, str]:
    llama_dir = _llama_cpp_dir()
    converter = _llama_cpp_converter(llama_dir)
    quantizer = _llama_cpp_quantizer(llama_dir)
    if converter is None or quantizer is None:
        raise RuntimeError(
            "llama.cpp GGUF backend is not configured. Set GTSM_LLAMA_CPP_DIR, or set "
            "GTSM_LLAMA_CPP_CONVERT and GTSM_LLAMA_CPP_QUANTIZE."
        )

    base_outtype = os.getenv("GTSM_LLAMA_CPP_OUTTYPE", "bf16").strip() or "bf16"
    base_gguf = gguf_dir / f"{variant_id}.{base_outtype}.gguf"
    converter_cmd = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(base_gguf),
        "--outtype",
        base_outtype,
    ]
    subprocess.run(
        converter_cmd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=str(llama_dir or converter.parent),
        env=_llama_cpp_env(llama_dir),
    )

    gguf_paths: dict[str, str] = {}
    for method in quant_methods:
        method_dir = gguf_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        output = method_dir / f"{variant_id}-{method}.gguf"
        quantize_cmd = [
            str(quantizer),
            str(base_gguf),
            str(output),
            _llama_cpp_quantization_method(method),
        ]
        subprocess.run(
            quantize_cmd,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(llama_dir or quantizer.parent),
            env=_llama_cpp_env(llama_dir),
        )
        if output.exists():
            gguf_paths[method] = str(output)
    return gguf_paths


def export_model_artifacts(
    paths: WorkspacePaths,
    variant_id: str,
    export_format: str = "all",
    quantizations: list[str] | None = None,
) -> ExportResult:
    variant = _load_variant(paths=paths, variant_id=variant_id)
    artifact_dir = Path(str(variant.get("artifact_dir", "")).strip())
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Variant artifact_dir does not exist: {artifact_dir}")

    export_root = paths.models_root / "exports" / variant_id
    export_root.mkdir(parents=True, exist_ok=True)

    merged_path = ""
    adapter_export_path = ""
    gguf_path = ""
    gguf_paths: dict[str, str] = {}
    notes: list[str] = []
    source_policy = resolve_model_source_policy(paths)
    quant_methods = quantizations or ["q4_k_m"]
    is_adapter = _is_peft_adapter(artifact_dir)

    if is_adapter:
        adapter_export_path = _copy_adapter_export(artifact_dir, export_root)

    if export_format in {"all", "merged"}:
        merged_dir = export_root / "merged"
        if is_adapter:
            try:
                _merge_peft_adapter(
                    variant=variant,
                    artifact_dir=artifact_dir,
                    merged_dir=merged_dir,
                    source_policy=source_policy,
                )
                merged_path = str(merged_dir)
            except Exception as error:
                notes.append(f"Merged export skipped: {error}")
        else:
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
            shutil.copytree(artifact_dir, merged_dir)
            merged_path = str(merged_dir)

    if export_format == "gguf" and is_adapter and not merged_path:
        merged_dir = export_root / "merged"
        if _is_hf_model_export(merged_dir):
            merged_path = str(merged_dir)
            notes.append("Reusing existing merged export for GGUF export.")
        else:
            try:
                _merge_peft_adapter(
                    variant=variant,
                    artifact_dir=artifact_dir,
                    merged_dir=merged_dir,
                    source_policy=source_policy,
                )
                merged_path = str(merged_dir)
            except Exception as error:
                notes.append(f"Merged export skipped before GGUF export: {error}")

    if export_format in {"all", "gguf"}:
        gguf_dir = export_root / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_file = gguf_dir / f"{variant_id}.gguf"
        if gguf_file.exists():
            gguf_file.unlink()

        gguf_backend = os.getenv("GTSM_GGUF_BACKEND", "auto").strip().lower() or "auto"
        if gguf_backend not in {"auto", "llama.cpp", "llamacpp", "unsloth"}:
            notes.append(f"Unknown GTSM_GGUF_BACKEND={gguf_backend}; using auto.")
            gguf_backend = "auto"

        gguf_source = Path(merged_path or str(artifact_dir))
        can_export_gguf = bool(merged_path) or not is_adapter
        if not can_export_gguf:
            notes.append("GGUF export skipped: PEFT adapter export must be merged first.")

        if can_export_gguf and gguf_backend in {"auto", "llama.cpp", "llamacpp"}:
            try:
                gguf_paths.update(
                    _export_gguf_with_llama_cpp(
                        model_dir=gguf_source,
                        gguf_dir=gguf_dir,
                        variant_id=variant_id,
                        quant_methods=quant_methods,
                    )
                )
            except subprocess.CalledProcessError as error:
                notes.append(f"llama.cpp GGUF export failed: {_tail_process_error(error)}")
            except Exception as error:
                notes.append(f"llama.cpp GGUF export unavailable: {error}")

        if can_export_gguf and not gguf_paths and gguf_backend in {"auto", "unsloth"}:
            try:
                from unsloth import FastLanguageModel  # pragma: no cover

                apply_model_source_environment(source_policy)
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(gguf_source),
                    max_seq_length=4096,
                    load_in_4bit=True,
                    **model_source_load_kwargs(source_policy),
                )
                for method in quant_methods:
                    method_dir = gguf_dir / method
                    method_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained_gguf(str(method_dir), tokenizer, quantization_method=method)
                    generated = list(method_dir.glob("*.gguf"))
                    if generated:
                        gguf_paths[method] = str(generated[0])
                    else:
                        notes.append(
                            f"GGUF export did not produce a *.gguf file for quantization={method}."
                        )
            except Exception as error:
                notes.append(f"Unsloth GGUF export failed: {error}")
        if gguf_paths:
            preferred = "q4_k_m" if "q4_k_m" in gguf_paths else sorted(gguf_paths)[0]
            gguf_path = gguf_paths[preferred]
        elif gguf_backend in {"llama.cpp", "llamacpp"}:
            notes.append("GGUF export skipped: llama.cpp backend did not produce output.")
        else:
            notes.append(
                "GGUF export skipped: configure user-space llama.cpp or install optional [unsloth] deps."
            )

    requested_results: list[bool] = []
    if export_format in {"all", "merged"}:
        requested_results.append(bool(merged_path))
    if export_format in {"all", "gguf"}:
        requested_results.append(bool(gguf_path))
    status = "completed" if requested_results and all(requested_results) else "partial"
    result = ExportResult(
        variant_id=variant_id,
        format=export_format,
        quantizations=quant_methods,
        merged_path=merged_path,
        adapter_export_path=adapter_export_path,
        gguf_path=gguf_path,
        gguf_paths=gguf_paths,
        status=status,
        notes=notes,
        model_source_policy=source_policy.to_dict(),
    )
    result_path = export_root / "export.result.json"
    result_path.write_text(result.to_json(), encoding="utf-8")
    return result


def write_ollama_modelfile(
    paths: WorkspacePaths,
    variant_id: str,
    *,
    model_name: str,
    from_quantization: str = "q4_k_m",
) -> Path:
    export_root = paths.models_root / "exports" / variant_id
    export_result_path = export_root / "export.result.json"
    if not export_result_path.exists():
        export_model_artifacts(
            paths=paths,
            variant_id=variant_id,
            export_format="gguf",
            quantizations=[from_quantization],
        )
    data = json.loads(export_result_path.read_text(encoding="utf-8"))
    gguf_paths = dict(data.get("gguf_paths", {}))
    gguf_path = _ollama_from_path(
        str(gguf_paths.get(from_quantization, "")).strip() or str(data.get("gguf_path", "")).strip()
    )
    modelfile_dir = export_root / "ollama"
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = modelfile_dir / "Modelfile"
    content = (
        f"FROM {gguf_path}\n"
        "TEMPLATE \"\"\"\n{{ .Prompt }}\n\"\"\"\n"
        "PARAMETER stop \"</tool>\"\n"
    )
    modelfile_path.write_text(content, encoding="utf-8")
    data["ollama_modelfile_path"] = str(modelfile_path)
    data["ollama_model_name"] = model_name
    export_result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return modelfile_path


def _ollama_cli() -> str:
    configured = os.getenv("GTSM_OLLAMA_CLI", "").strip()
    if configured:
        return str(Path(configured).expanduser())
    return shutil.which("ollama") or "ollama"


def create_ollama_model(modelfile_path: Path, model_name: str) -> dict:
    completed = subprocess.run(
        [_ollama_cli(), "create", model_name, "-f", str(modelfile_path)],
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return {
        "model_name": model_name,
        "modelfile_path": str(modelfile_path),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }
