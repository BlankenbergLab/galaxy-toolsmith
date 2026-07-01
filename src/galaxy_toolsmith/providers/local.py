from __future__ import annotations

import inspect
import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    normalize_artifact_format,
)
from galaxy_toolsmith.prompts import render_prompt_template
from galaxy_toolsmith.providers.base import (
    GenerationInput,
    GenerationOutput,
    extract_generated_artifact,
    generation_prompt_task,
)
from galaxy_toolsmith.runtime.model_source import (
    apply_model_source_environment,
    model_source_load_kwargs,
    resolve_model_source_policy,
)

TOOL_CLOSE_DECODE_WINDOW_TOKENS = 64
LOCAL_OFFLOAD_POLICIES = {"allow", "fail"}
MLX_LM_BACKEND_ALIASES = {"mlx-lm", "mlx", "mps"}


def _render_generation_prompt(request: GenerationInput) -> str:
    return render_prompt_template(
        task=generation_prompt_task(request),
        skills_profile=request.skills_profile,
        context={
            "tool_name": request.tool_name,
            "help_text": request.help_text,
            "source_code": request.source_code,
            "skills_profile": request.skills_profile,
            "repair_context": request.repair_context,
            "interface_hints": request.interface_hints,
        },
    )


def _decode_model_response(
    *,
    model: object,
    tokenizer: object,
    request: GenerationInput,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = _render_generation_prompt(request)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = None
    if device is not None:
        inputs = _move_inputs_to_device(inputs, device)
    generation_kwargs = _generation_input_kwargs(inputs)
    input_ids = generation_kwargs.get("input_ids")
    if input_ids is None:
        raise RuntimeError("Tokenizer chat template did not return input_ids for local generation.")
    input_length = _input_length(input_ids)
    model_generate_kwargs = {
        **generation_kwargs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if normalize_artifact_format(request.artifact_format) == ARTIFACT_FORMAT_XML:
        stopping_criteria = _tool_xml_stopping_criteria(tokenizer=tokenizer, input_length=input_length)
        if stopping_criteria is not None:
            model_generate_kwargs["stopping_criteria"] = stopping_criteria
    if temperature > 0:
        model_generate_kwargs["temperature"] = temperature
    outputs = model.generate(**model_generate_kwargs)
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _tool_xml_stopping_criteria(*, tokenizer: object, input_length: int) -> object | None:
    try:
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
    except Exception:  # pragma: no cover - optional runtime integration
        return None

    class StopOnToolClose(StoppingCriteria):
        def __init__(self) -> None:
            self._last_checked_length = -1

        def __call__(self, input_ids: object, scores: object, **kwargs: object) -> bool:
            shape = getattr(input_ids, "shape", None)
            if not shape:
                return False
            generated_length = int(shape[-1]) - input_length
            if generated_length <= 0:
                return False
            if generated_length == self._last_checked_length:
                return False
            if generated_length % 4 != 0:
                return False
            self._last_checked_length = generated_length
            return _generated_text_contains_tool_close(
                tokenizer=tokenizer,
                input_ids=input_ids,
                input_length=input_length,
            )

    return StoppingCriteriaList([StopOnToolClose()])


def _generated_text_contains_tool_close(
    *,
    tokenizer: object,
    input_ids: object,
    input_length: int,
) -> bool:
    try:
        shape = getattr(input_ids, "shape", None)
        if shape:
            suffix_start = max(input_length, int(shape[-1]) - TOOL_CLOSE_DECODE_WINDOW_TOKENS)
        else:
            suffix_start = input_length
        tokens = input_ids[0][suffix_start:]
        tolist = getattr(tokens, "tolist", None)
        if callable(tolist):
            tokens = tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception:
        return False
    return "</tool>" in text


def _move_inputs_to_device(inputs: object, device: object) -> object:
    to_method = getattr(inputs, "to", None)
    if callable(to_method):
        return to_method(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if callable(getattr(value, "to", None)) else value
            for key, value in inputs.items()
        }
    return inputs


def _generation_input_kwargs(inputs: object) -> dict:
    if isinstance(inputs, dict):
        return dict(inputs)
    keys = getattr(inputs, "keys", None)
    if callable(keys):
        return {key: inputs[key] for key in keys()}
    return {"input_ids": inputs}


def _input_length(input_ids: object) -> int:
    shape = getattr(input_ids, "shape", None)
    if not shape:
        raise RuntimeError("Tokenizer input_ids do not expose shape for local generation.")
    return int(shape[-1])


def _preview_text(text: str, limit: int = 300) -> str:
    value = text.replace("\r", "\\r").replace("\n", "\\n")
    return value[:limit]


def _cuda_max_memory(torch_module: object, reserve_gib: float) -> dict[int, str]:
    device_count = int(torch_module.cuda.device_count())
    max_memory: dict[int, str] = {}
    for index in range(device_count):
        props = torch_module.cuda.get_device_properties(index)
        total_gib = float(props.total_memory) / (1024**3)
        usable_gib = max(1, int(total_gib - max(0.0, reserve_gib)))
        max_memory[index] = f"{usable_gib}GiB"
    return max_memory


def _normalize_device_target(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("device(type="):
        if "'cuda'" in text or '"cuda"' in text:
            return "cuda"
        if "'cpu'" in text or '"cpu"' in text:
            return "cpu"
    return text


def _collect_hf_device_maps(model: object) -> dict[str, object]:
    collected: dict[str, object] = {}
    visited: set[int] = set()
    stack = [("", model)]
    while stack:
        prefix, item = stack.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        device_map = getattr(item, "hf_device_map", None)
        if isinstance(device_map, Mapping):
            for key, value in device_map.items():
                map_key = f"{prefix}.{key}" if prefix and key else str(key or prefix or "")
                collected[map_key] = value
        for attr in ("base_model", "model"):
            child = getattr(item, attr, None)
            if child is not None:
                child_prefix = f"{prefix}.{attr}" if prefix else attr
                stack.append((child_prefix, child))
    return collected


def _device_map_report(model: object) -> dict:
    device_map = _collect_hf_device_maps(model)
    normalized = {key: _normalize_device_target(value) for key, value in device_map.items()}
    offloaded = {
        key: value
        for key, value in normalized.items()
        if value in {"cpu", "disk", "meta"} or value.startswith("cpu:") or value.startswith("disk")
    }
    return {
        "device_map": {key: str(value) for key, value in device_map.items()},
        "offloaded_devices": offloaded,
        "has_offload": bool(offloaded),
    }


class LocalUnslothProvider:
    name = "local-unsloth"

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        source_policy: object | None = None,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.source_policy = source_policy or resolve_model_source_policy()
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> tuple[object, object]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            from unsloth import FastLanguageModel
        except Exception as error:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "unsloth is not installed. Install optional extras or configure GTSM_LOCAL_GENERATOR_CMD."
            ) from error

        apply_model_source_environment(self.source_policy)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=16384,
            load_in_4bit=True,
            **model_source_load_kwargs(self.source_policy),
        )
        if self.adapter_path:
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)
            except Exception as error:  # pragma: no cover - depends on optional runtime deps
                raise RuntimeError(f"Failed to load LoRA adapter from {self.adapter_path}: {error}") from error
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def ensure_loaded(self) -> dict:
        self._lazy_load()
        return {
            "backend": self.name,
            "model_name": self.model_name,
            "adapter_path": self.adapter_path or "",
        }

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        model, tokenizer = self._lazy_load()
        response_text = _decode_model_response(
            model=model,
            tokenizer=tokenizer,
            request=request,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        artifact = extract_generated_artifact(response_text, request.artifact_format)
        if not artifact:
            raise RuntimeError(
                "LocalUnslothProvider returned empty output after artifact stripping; "
                f"raw_response_length={len(response_text)} "
                f"raw_response_preview={_preview_text(response_text)!r}"
            )
        return GenerationOutput(
            artifact_text=artifact,
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
        )


class LocalPeftProvider:
    name = "local-peft"

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str | None,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        source_policy: object | None = None,
        offload_policy: str = "allow",
        gpu_memory_reserve_gib: float = 2.0,
    ):
        if offload_policy not in LOCAL_OFFLOAD_POLICIES:
            raise ValueError(f"Unsupported local offload policy: {offload_policy}")
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.source_policy = source_policy or resolve_model_source_policy()
        self.offload_policy = offload_policy
        self.gpu_memory_reserve_gib = gpu_memory_reserve_gib
        self._model = None
        self._tokenizer = None
        self._load_info: dict = {}

    def _lazy_load(self) -> tuple[object, object]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as error:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "Local HF inference requires torch and transformers. "
                "Install training deps or use a configured external/local generator."
            ) from error
        PeftModel = None
        if self.adapter_path:
            try:
                from peft import PeftModel
            except Exception as error:  # pragma: no cover - depends on optional runtime deps
                raise RuntimeError(
                    "Local PEFT inference requires peft. Install training deps or use a "
                    "configured external/local generator."
                ) from error

        apply_model_source_environment(self.source_policy)
        source_kwargs = model_source_load_kwargs(self.source_policy)
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True, **source_kwargs
        )
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(
            tokenizer, "eos_token", None
        ):
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict = {"trust_remote_code": True, **source_kwargs}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            max_memory = _cuda_max_memory(torch, self.gpu_memory_reserve_gib)
            if max_memory:
                load_kwargs["max_memory"] = max_memory
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.base_model, **load_kwargs)
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        device_report = _device_map_report(model)
        if self.offload_policy == "fail" and device_report["has_offload"]:
            raise RuntimeError(
                "Local PEFT model used CPU/disk/meta offload while local offload policy is fail: "
                f"{device_report['offloaded_devices']}"
            )
        self._load_info = {
            "offload_policy": self.offload_policy,
            "gpu_memory_reserve_gib": self.gpu_memory_reserve_gib,
            "max_memory": {str(key): value for key, value in load_kwargs.get("max_memory", {}).items()},
            **device_report,
        }
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def ensure_loaded(self) -> dict:
        self._lazy_load()
        return {
            "backend": self.name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path or "",
            **self._load_info,
        }

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        model, tokenizer = self._lazy_load()
        response_text = _decode_model_response(
            model=model,
            tokenizer=tokenizer,
            request=request,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        artifact = extract_generated_artifact(response_text, request.artifact_format)
        if not artifact:
            raise RuntimeError(
                "LocalPeftProvider returned empty output after artifact stripping; "
                f"base_model={self.base_model!r} adapter_path={self.adapter_path!r} "
                f"raw_response_length={len(response_text)} "
                f"raw_response_preview={_preview_text(response_text)!r}"
            )
        return GenerationOutput(
            artifact_text=artifact,
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
        )


class LocalMLXProvider:
    name = "local-mlx"

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        source_policy: object | None = None,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.source_policy = source_policy or resolve_model_source_policy()
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> tuple[object, object]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            from mlx_lm import load
        except Exception as error:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "Local MLX inference requires mlx-lm. Install the mps extra or use another local generator."
            ) from error

        apply_model_source_environment(self.source_policy)
        load_kwargs: dict = {
            "adapter_path": self.adapter_path,
            "tokenizer_config": {"trust_remote_code": True},
        }
        try:
            load_parameters = inspect.signature(load).parameters
        except (TypeError, ValueError):
            load_parameters = {}
        if "trust_remote_code" in load_parameters:
            load_kwargs["trust_remote_code"] = True
        revision = str(getattr(self.source_policy, "revision", "") or "")
        if revision:
            load_kwargs["revision"] = revision
        model, tokenizer = load(self.base_model, **load_kwargs)
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def ensure_loaded(self) -> dict:
        self._lazy_load()
        return {
            "backend": self.name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
        }

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
        except Exception as error:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "Local MLX generation requires mlx-lm. Install the mps extra or use another local generator."
            ) from error
        model, tokenizer = self._lazy_load()
        prompt = _mlx_prompt(tokenizer, _render_generation_prompt(request))
        response_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            sampler=make_sampler(self.temperature),
            verbose=False,
        )
        artifact = extract_generated_artifact(response_text, request.artifact_format)
        if not artifact:
            raise RuntimeError(
                "LocalMLXProvider returned empty output after artifact stripping; "
                f"base_model={self.base_model!r} adapter_path={self.adapter_path!r} "
                f"raw_response_length={len(response_text)} "
                f"raw_response_preview={_preview_text(response_text)!r}"
            )
        return GenerationOutput(
            artifact_text=artifact,
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
        )


def _mlx_prompt(tokenizer: object, prompt: str) -> object:
    if not getattr(tokenizer, "has_chat_template", False):
        return prompt
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return apply_chat_template(messages, add_generation_prompt=True)


class LocalProvider:
    name = "local"

    def __init__(
        self,
        command: str | None = None,
        *,
        paths: WorkspacePaths | None = None,
        model: str = "",
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        allow_stub: bool = False,
        local_offload_policy: str = "allow",
        local_gpu_memory_reserve_gib: float = 2.0,
    ):
        if local_offload_policy not in LOCAL_OFFLOAD_POLICIES:
            raise ValueError(f"Unsupported local offload policy: {local_offload_policy}")
        self.command = command if command is not None else os.getenv("GTSM_LOCAL_GENERATOR_CMD")
        self.paths = paths
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.allow_stub = allow_stub
        self.local_offload_policy = local_offload_policy
        self.local_gpu_memory_reserve_gib = local_gpu_memory_reserve_gib
        unsloth_model = os.getenv("GTSM_LOCAL_UNSLOTH_MODEL", "").strip()
        unsloth_adapter = os.getenv("GTSM_LOCAL_UNSLOTH_ADAPTER", "").strip() or None
        self.unsloth_provider: LocalUnslothProvider | None = None
        if unsloth_model:
            self.unsloth_provider = LocalUnslothProvider(
                model_name=unsloth_model,
                adapter_path=unsloth_adapter,
                source_policy=resolve_model_source_policy(paths),
            )
        self._peft_providers: dict[str, LocalPeftProvider] = {}
        self._mlx_providers: dict[str, LocalMLXProvider] = {}

    def _run_external_local_command(self, request: GenerationInput) -> str:
        payload = {
            "tool_name": request.tool_name,
            "help_text": request.help_text,
            "source_code": request.source_code,
            "model_variant": request.model_variant,
            "skills_profile": request.skills_profile,
            "repair_context": request.repair_context,
            "interface_hints": request.interface_hints,
            "artifact_format": request.artifact_format,
        }
        args = shlex.split(self.command) if self.command else []
        if not args:
            raise ValueError("GTSM_LOCAL_GENERATOR_CMD is configured but empty.")
        completed = subprocess.run(
            args,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Local generator command failed with exit code {completed.returncode}: {detail}")
        artifact = extract_generated_artifact(completed.stdout, request.artifact_format)
        if not artifact:
            raise RuntimeError("Local generator command returned empty output.")
        return artifact

    def _manifest_path(self, model_variant: str) -> Path | None:
        if self.paths is None:
            return None
        variant_id = model_variant.strip()
        if not variant_id:
            return None
        if variant_id.endswith(".manifest.json"):
            variant_id = variant_id[: -len(".manifest.json")]
        manifest_path = self.paths.models_root / "variants" / f"{variant_id}.manifest.json"
        return manifest_path if manifest_path.exists() else None

    def _peft_provider_for_variant(self, request: GenerationInput) -> LocalPeftProvider | None:
        manifest_path = self._manifest_path(request.model_variant)
        if manifest_path is None:
            return None
        cache_key = str(manifest_path)
        if cache_key in self._peft_providers:
            return self._peft_providers[cache_key]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_kind = str(manifest.get("artifact_kind", "") or "").strip().lower()
        if artifact_kind == "unknown":
            artifact_kind = ""
        backend = str(manifest.get("backend", "") or "").strip().lower()
        artifact_dir = Path(str(manifest.get("artifact_dir", "")).strip())
        if artifact_kind in {"mlx_adapter", "mlx_full_weights"} or backend in MLX_LM_BACKEND_ALIASES:
            return None
        if not artifact_dir.exists():
            raise RuntimeError(f"Model variant artifact_dir does not exist: {artifact_dir}")
        is_full_hf = artifact_kind == "hf_full_model"
        is_peft = artifact_kind == "peft_adapter" or (
            not artifact_kind and (artifact_dir / "adapter_config.json").exists()
        )
        if not is_full_hf and not is_peft:
            return None
        base_model = self.model.strip() or (
            str(artifact_dir)
            if is_full_hf
            else str(manifest.get("base_model", "")).strip()
        )
        if not base_model:
            raise RuntimeError(f"Model variant manifest lacks base_model: {manifest_path}")
        provider = LocalPeftProvider(
            base_model=base_model,
            adapter_path=None if is_full_hf else str(artifact_dir),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            source_policy=resolve_model_source_policy(self.paths),
            offload_policy=self.local_offload_policy,
            gpu_memory_reserve_gib=self.local_gpu_memory_reserve_gib,
        )
        self._peft_providers[cache_key] = provider
        return provider

    def _mlx_provider_for_variant(self, request: GenerationInput) -> LocalMLXProvider | None:
        manifest_path = self._manifest_path(request.model_variant)
        if manifest_path is None:
            return None
        cache_key = str(manifest_path)
        if cache_key in self._mlx_providers:
            return self._mlx_providers[cache_key]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        backend = str(manifest.get("backend", "")).strip().lower()
        artifact_kind = str(manifest.get("artifact_kind", "") or "").strip().lower()
        if artifact_kind == "unknown":
            artifact_kind = ""
        artifact_dir = Path(str(manifest.get("artifact_dir", "")).strip())
        if artifact_kind in {"peft_adapter", "hf_full_model"}:
            return None
        if backend not in MLX_LM_BACKEND_ALIASES and not (
            artifact_dir / "adapters.safetensors"
        ).exists():
            return None
        base_model = self.model.strip() or str(manifest.get("base_model", "")).strip()
        if not base_model:
            raise RuntimeError(f"Model variant manifest lacks base_model: {manifest_path}")
        if not artifact_dir.exists():
            raise RuntimeError(f"Model variant artifact_dir does not exist: {artifact_dir}")
        if not (artifact_dir / "adapters.safetensors").exists():
            raise RuntimeError(
                f"Model variant does not point to an MLX adapter artifact_dir: {manifest_path}"
            )
        provider = LocalMLXProvider(
            base_model=base_model,
            adapter_path=str(artifact_dir),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            source_policy=resolve_model_source_policy(self.paths),
        )
        self._mlx_providers[cache_key] = provider
        return provider

    def _unconfigured_error(self) -> RuntimeError:
        return RuntimeError(
            "No real local generator is configured. Set GTSM_LOCAL_GENERATOR_CMD, "
            "set GTSM_LOCAL_UNSLOTH_MODEL/GTSM_LOCAL_UNSLOTH_ADAPTER, or pass a "
            "model_variant with a local PEFT or MLX adapter manifest. Use --allow-stub-local "
            "only for smoke tests that intentionally use canned XML."
        )

    def ensure_ready(self, model_variant: str) -> None:
        if self.command or self.unsloth_provider is not None or self.allow_stub:
            return
        manifest_path = self._manifest_path(model_variant)
        if manifest_path is None:
            raise self._unconfigured_error()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        backend = str(manifest.get("backend", "")).strip().lower()
        artifact_kind = str(manifest.get("artifact_kind", "") or "").strip().lower()
        if artifact_kind == "unknown":
            artifact_kind = ""
        artifact_dir = Path(str(manifest.get("artifact_dir", "")).strip())
        mlx_adapter_exists = (artifact_dir / "adapters.safetensors").exists()
        if artifact_kind in {"mlx_adapter", "mlx_full_weights"} or (
            not artifact_kind and (backend in MLX_LM_BACKEND_ALIASES or mlx_adapter_exists)
        ):
            if artifact_dir.exists() and (artifact_dir / "adapters.safetensors").exists():
                return
            raise RuntimeError(
                f"Model variant does not point to an MLX adapter artifact_dir: {manifest_path}"
            )
        if artifact_kind == "hf_full_model":
            if artifact_dir.exists():
                return
            raise RuntimeError(f"Model variant artifact_dir does not exist: {artifact_dir}")
        if artifact_dir.exists() and (artifact_dir / "adapter_config.json").exists():
            return
        raise RuntimeError(
            f"Model variant does not point to a PEFT adapter artifact_dir: {manifest_path}"
        )

    def ensure_loaded(self, model_variant: str) -> dict:
        if self.command:
            return {"backend": "local-command", "model_loaded": False}
        if self.unsloth_provider is not None:
            loaded = self.unsloth_provider.ensure_loaded()
            return {"model_loaded": True, **loaded}

        request = GenerationInput(
            tool_name="_preload",
            help_text="",
            source_code="",
            model_variant=model_variant,
        )
        mlx_provider = self._mlx_provider_for_variant(request)
        if mlx_provider is not None:
            loaded = mlx_provider.ensure_loaded()
            return {"model_loaded": True, **loaded}

        peft_provider = self._peft_provider_for_variant(request)
        if peft_provider is not None:
            loaded = peft_provider.ensure_loaded()
            return {"model_loaded": True, **loaded}

        if self.allow_stub:
            return {"backend": "local-stub", "model_loaded": False}
        raise self._unconfigured_error()

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        if self.command:
            artifact = self._run_external_local_command(request)
            return GenerationOutput(
                artifact_text=artifact,
                artifact_format=request.artifact_format,
                provider=self.name,
                model_variant=request.model_variant,
            )
        if self.unsloth_provider is not None:
            return self.unsloth_provider.generate_wrapper(request)

        mlx_provider = self._mlx_provider_for_variant(request)
        if mlx_provider is not None:
            return mlx_provider.generate_wrapper(request)

        peft_provider = self._peft_provider_for_variant(request)
        if peft_provider is not None:
            return peft_provider.generate_wrapper(request)

        if not self.allow_stub:
            raise self._unconfigured_error()

        if normalize_artifact_format(request.artifact_format) == ARTIFACT_FORMAT_UDT_YAML:
            tool_name_yaml = json.dumps(request.tool_name)
            shell_command_yaml = json.dumps('echo "TODO: replace with command template" > output.txt')
            udt_yaml = f"""class: GalaxyUserTool
id: {tool_name_yaml}
version: "0.1.0"
name: {tool_name_yaml}
description: Generated by Galaxy Toolsmith local provider
container: busybox
shell_command: {shell_command_yaml}
inputs:
  - name: input1
    type: data
    format: txt
    label: Input file
outputs:
  - name: output1
    type: data
    format: txt
    from_work_dir: output.txt
help:
  format: markdown
  content: |
    Auto-generated starter user-defined tool.

    Source hint length: {len(request.source_code)}
    Help hint length: {len(request.help_text)}
"""
            return GenerationOutput(
                artifact_text=udt_yaml,
                artifact_format=request.artifact_format,
                provider="local-stub",
                model_variant=request.model_variant,
            )

        xml = f"""<tool id="{request.tool_name}" name="{request.tool_name}" version="0.1.0">
    <description>Generated by Galaxy Toolsmith local provider</description>
    <command detect_errors="exit_code"><![CDATA[
echo "TODO: replace with command template"
]]></command>
    <inputs>
        <param name="input1" type="data" format="txt" label="Input file"/>
    </inputs>
    <outputs>
        <data name="out_file" format="txt"/>
    </outputs>
    <help><![CDATA[
Auto-generated starter wrapper.

Source hint length: {len(request.source_code)}
Help hint length: {len(request.help_text)}
]]></help>
    <tests>
        <test expect_num_outputs="1">
            <output name="out_file"/>
        </test>
    </tests>
</tool>
"""
        return GenerationOutput(
            artifact_text=xml,
            artifact_format=request.artifact_format,
            provider="local-stub",
            model_variant=request.model_variant,
        )
