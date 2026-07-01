from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TRAINING_METHODS = {"lora", "qlora", "full"}


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    base_model: str
    provider: str
    quantization: str
    backend: str
    skills_profile: str
    default_command: list[str]
    max_seq_length: int = 4096
    pad_to_sequence_len: bool = True
    attn_implementation: str | None = None
    epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    distributed_strategy: str = "ddp"
    learning_rate: float = 2e-4
    training_method: str = "lora"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    seed: int = 3407


DEFAULT_TRAINING_PROFILES = [
    TrainingProfile(
        name="proto-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=2,
    ),
    TrainingProfile(
        name="proto-qwen25-7b-4bit",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="4bit",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=2,
    ),
    TrainingProfile(
        name="agentic-devstral-24b",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="agentic-devstral-24b-4bit",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="4bit",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="baseline-mistral-24b",
        base_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="baseline-mistral-24b-4bit",
        base_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        provider="local",
        quantization="4bit",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="deepseek-coder-v2-lite-instruct",
        base_model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="deepseek-r1-distill-qwen-14b",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="deepseek-r1-distill-qwen-32b",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
    ),
    TrainingProfile(
        name="mps-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
    ),
    TrainingProfile(
        name="mps-devstral-24b",
        base_model="mistralai/Devstral-Small-2505",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="mps-qwen25-14b",
        base_model="Qwen/Qwen2.5-Coder-14B-Instruct",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="mps-mistral31-24b",
        base_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
    ),
    TrainingProfile(
        name="mps-qwen25-32b",
        base_model="Qwen/Qwen2.5-Coder-32B-Instruct",
        provider="local",
        quantization="none",
        backend="mlx-lm",
        skills_profile="default",
        default_command=[],
        max_seq_length=4096,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
    ),
    TrainingProfile(
        name="full-qwen25-7b",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
        distributed_strategy="auto",
        learning_rate=2e-5,
        training_method="full",
    ),
    TrainingProfile(
        name="full-mistral-24b",
        base_model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        distributed_strategy="auto",
        learning_rate=2e-5,
        training_method="full",
    ),
    TrainingProfile(
        name="full-deepseek-r1-distill-qwen-14b",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=8192,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        distributed_strategy="auto",
        learning_rate=2e-5,
        training_method="full",
    ),
    TrainingProfile(
        name="full-deepseek-r1-distill-qwen-32b",
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        provider="local",
        quantization="none",
        backend="axolotl",
        skills_profile="default",
        default_command=[],
        max_seq_length=4096,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        distributed_strategy="auto",
        learning_rate=2e-5,
        training_method="full",
    ),
    TrainingProfile(
        name="external-api-reference",
        base_model="unset",
        provider="external",
        quantization="none",
        backend="cpu",
        skills_profile="default",
        default_command=[],
    ),
]


def write_default_training_profiles(path: Path) -> Path:
    payload = {"profiles": [asdict(profile) for profile in DEFAULT_TRAINING_PROFILES]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def normalize_training_method(value: str | None) -> str:
    normalized = str(value or "lora").strip().lower().replace("_", "-")
    if not normalized:
        normalized = "lora"
    if normalized not in TRAINING_METHODS:
        choices = ", ".join(sorted(TRAINING_METHODS))
        raise ValueError(f"Unsupported training_method '{value}'. Expected one of: {choices}.")
    return normalized


def load_training_profile(path: Path, profile_name: str) -> TrainingProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    for profile in data.get("profiles", []):
        if profile.get("name") == profile_name:
            return TrainingProfile(
                name=profile["name"],
                base_model=profile["base_model"],
                provider=profile["provider"],
                quantization=profile["quantization"],
                backend=profile["backend"],
                skills_profile=profile["skills_profile"],
                default_command=list(profile.get("default_command", [])),
                max_seq_length=int(profile.get("max_seq_length", 4096)),
                pad_to_sequence_len=bool(profile.get("pad_to_sequence_len", True)),
                attn_implementation=profile.get("attn_implementation"),
                epochs=int(profile.get("epochs", 1)),
                per_device_batch_size=int(profile.get("per_device_batch_size", 1)),
                gradient_accumulation_steps=int(profile.get("gradient_accumulation_steps", 1)),
                distributed_strategy=str(profile.get("distributed_strategy", "ddp")),
                learning_rate=float(profile.get("learning_rate", 2e-4)),
                training_method=normalize_training_method(profile.get("training_method", "lora")),
                lora_rank=int(profile.get("lora_rank", 16)),
                lora_alpha=int(profile.get("lora_alpha", 32)),
                lora_dropout=float(profile.get("lora_dropout", 0.0)),
                seed=int(profile.get("seed", 3407)),
            )
    raise ValueError(f"Training profile '{profile_name}' not found in {path}")
