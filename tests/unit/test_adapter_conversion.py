from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from galaxy_toolsmith.orchestration.adapter_conversion import convert_mlx_lora_to_peft


def _write_mlx_adapter(adapter_dir: Path, *, fine_tune_type: str = "lora") -> None:
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": fine_tune_type,
                "lora_parameters": {
                    "rank": 2,
                    "scale": 8,
                    "dropout": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    if fine_tune_type == "lora":
        save_file(
            {
                "layers.0.self_attn.q_proj.lora_a": np.arange(6, dtype=np.float32).reshape(3, 2),
                "layers.0.self_attn.q_proj.lora_b": np.arange(8, dtype=np.float32).reshape(2, 4),
            },
            str(adapter_dir / "adapters.safetensors"),
        )
    else:
        save_file(
            {"layers.0.self_attn.q_proj.weight": np.arange(4, dtype=np.float32)},
            str(adapter_dir / "adapters.safetensors"),
        )


def test_convert_mlx_lora_to_peft_transposes_weights(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "mlx"
    output_dir = tmp_path / "peft"
    _write_mlx_adapter(adapter_dir)

    result = convert_mlx_lora_to_peft(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_dir=adapter_dir,
        output_dir=output_dir,
    )

    assert result.status == "completed"
    assert result.architecture == "qwen2"
    assert result.converted_tensors == 2
    peft_config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert peft_config["peft_type"] == "LORA"
    assert peft_config["r"] == 2
    assert peft_config["lora_alpha"] == 16
    assert peft_config["lora_dropout"] == 0.05
    assert peft_config["target_modules"] == ["q_proj"]
    tensors = load_file(str(output_dir / "adapter_model.safetensors"))
    a_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    b_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    assert tensors[a_key].shape == (2, 3)
    assert tensors[b_key].shape == (4, 2)
    np.testing.assert_array_equal(
        tensors[a_key],
        np.arange(6, dtype=np.float32).reshape(3, 2).T,
    )
    np.testing.assert_array_equal(
        tensors[b_key],
        np.arange(8, dtype=np.float32).reshape(2, 4).T,
    )


def test_convert_mlx_lora_to_peft_rejects_full_weights(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "mlx"
    _write_mlx_adapter(adapter_dir, fine_tune_type="full")

    with pytest.raises(ValueError, match="fine_tune_type=full"):
        convert_mlx_lora_to_peft(
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_dir=adapter_dir,
            output_dir=tmp_path / "peft",
        )


def test_convert_mlx_lora_to_peft_rejects_unknown_architecture(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "mlx"
    _write_mlx_adapter(adapter_dir)

    with pytest.raises(ValueError, match="known Qwen/Llama/Mistral"):
        convert_mlx_lora_to_peft(
            base_model="example/unsupported-model",
            adapter_dir=adapter_dir,
            output_dir=tmp_path / "peft",
        )
