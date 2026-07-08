from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_MLX_TO_PEFT_ARCHITECTURES = {
    "qwen": "qwen2",
    "qwen2": "qwen2",
    "qwen2.5": "qwen2",
    "llama": "llama",
    "mistral": "mistral",
    "devstral": "mistral",
}

SUPPORTED_PROJECTION_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass(frozen=True)
class AdapterConversionResult:
    status: str
    from_format: str
    to_format: str
    base_model: str
    architecture: str
    adapter_dir: str
    output_dir: str
    converted_tensors: int
    target_modules: list[str]
    validation_status: str = "not_run"
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _architecture_for_base_model(base_model: str) -> str:
    lowered = base_model.lower()
    for marker, architecture in SUPPORTED_MLX_TO_PEFT_ARCHITECTURES.items():
        if marker in lowered:
            return architecture
    raise ValueError(
        "MLX LoRA -> PEFT conversion is only supported for known Qwen/Llama/Mistral "
        f"architectures; got base_model={base_model!r}."
    )


def _read_mlx_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"MLX adapter config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fine_tune_type = str(config.get("fine_tune_type", "lora")).strip().lower()
    if fine_tune_type == "full":
        raise ValueError("Cannot convert MLX fine_tune_type=full weights to a PEFT LoRA adapter.")
    if fine_tune_type != "lora":
        raise ValueError(f"Only MLX LoRA adapters can be converted to PEFT; got {fine_tune_type!r}.")
    return config


def _target_module_from_mlx_prefix(prefix: str) -> str:
    target = prefix.rsplit(".", 1)[-1]
    if target not in SUPPORTED_PROJECTION_MODULES:
        raise ValueError(
            f"Unsupported MLX LoRA target module {target!r} from key prefix {prefix!r}. "
            "Only standard decoder projection modules are supported."
        )
    return target


def _hf_module_path_from_mlx_prefix(prefix: str) -> str:
    if prefix.startswith("model."):
        return prefix
    if prefix.startswith("layers."):
        return f"model.{prefix}"
    raise ValueError(
        f"Unsupported MLX module path {prefix!r}; expected keys under layers.* or model.*."
    )


def _peft_key(prefix: str, matrix_name: str) -> str:
    return f"base_model.model.{_hf_module_path_from_mlx_prefix(prefix)}.{matrix_name}.weight"


def _peft_config(
    *,
    base_model: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> dict[str, Any]:
    return {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }


def convert_mlx_lora_to_peft(
    *,
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
) -> AdapterConversionResult:
    architecture = _architecture_for_base_model(base_model)
    config = _read_mlx_adapter_config(adapter_dir)
    weights_path = adapter_dir / "adapters.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"MLX adapter weights not found: {weights_path}")

    try:
        from safetensors.numpy import load_file, save_file
    except Exception as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "Adapter conversion requires the numpy and safetensors packages."
        ) from error

    tensors = load_file(str(weights_path))
    lora_params = config.get("lora_parameters") or {}
    configured_rank = int(lora_params.get("rank", 0) or 0)
    scale = float(lora_params.get("scale", 0.0) or 0.0)
    dropout = float(lora_params.get("dropout", 0.0) or 0.0)

    converted = {}
    target_modules: set[str] = set()
    observed_rank = configured_rank
    notes: list[str] = []

    for a_key in sorted(key for key in tensors if key.endswith(".lora_a")):
        prefix = a_key[: -len(".lora_a")]
        b_key = f"{prefix}.lora_b"
        if b_key not in tensors:
            raise ValueError(f"Missing matching MLX lora_b tensor for {a_key}.")
        a_tensor = tensors[a_key]
        b_tensor = tensors[b_key]
        if len(a_tensor.shape) != 2 or len(b_tensor.shape) != 2:
            raise ValueError(
                f"Unsupported non-2D MLX LoRA tensors for {prefix!r}; "
                f"got {a_tensor.shape} and {b_tensor.shape}."
            )
        if a_tensor.shape[1] != b_tensor.shape[0]:
            raise ValueError(
                f"Inconsistent LoRA rank for {prefix!r}: {a_tensor.shape} vs {b_tensor.shape}."
            )
        rank = int(a_tensor.shape[1])
        if observed_rank and observed_rank != rank:
            raise ValueError(
                f"MLX adapter rank metadata ({observed_rank}) does not match tensor rank ({rank})."
            )
        observed_rank = rank
        target_modules.add(_target_module_from_mlx_prefix(prefix))
        converted[_peft_key(prefix, "lora_A")] = a_tensor.T.copy()
        converted[_peft_key(prefix, "lora_B")] = b_tensor.T.copy()

    if not converted:
        raise ValueError("No MLX LoRA tensors ending in .lora_a/.lora_b were found.")
    if not observed_rank:
        raise ValueError("Could not determine LoRA rank from MLX adapter tensors.")
    if scale <= 0:
        notes.append("MLX adapter config did not define a positive scale; using scale=1.0.")
        scale = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_dir / "adapter_model.safetensors"))
    peft_config = _peft_config(
        base_model=base_model,
        rank=observed_rank,
        alpha=scale * observed_rank,
        dropout=dropout,
        target_modules=sorted(target_modules),
    )
    (output_dir / "adapter_config.json").write_text(
        json.dumps(peft_config, indent=2),
        encoding="utf-8",
    )
    result = AdapterConversionResult(
        status="completed",
        from_format="mlx",
        to_format="peft",
        base_model=base_model,
        architecture=architecture,
        adapter_dir=str(adapter_dir),
        output_dir=str(output_dir),
        converted_tensors=len(converted),
        target_modules=sorted(target_modules),
        notes=notes,
    )
    (output_dir / "conversion.result.json").write_text(result.to_json(), encoding="utf-8")
    return result
