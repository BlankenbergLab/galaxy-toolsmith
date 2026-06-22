from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ModelEstimate:
    profile: str
    model: str
    params: str
    target_hardware: str
    training_memory_tier: str
    inference_memory_tier: str
    relative_training_cost: str
    relative_inference_cost: str
    notes: str


ESTIMATES = [
    ModelEstimate(
        profile="mps-qwen25-7b",
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        params="7B",
        target_hardware="Apple Silicon (M4 Max high-memory and above)",
        training_memory_tier="low",
        inference_memory_tier="low",
        relative_training_cost="1x baseline",
        relative_inference_cost="1x baseline",
        notes="Best Apple Silicon starting point.",
    ),
    ModelEstimate(
        profile="mps-qwen25-14b",
        model="Qwen/Qwen2.5-Coder-14B-Instruct",
        params="14B",
        target_hardware="Apple Silicon (M4 Max high-memory)",
        training_memory_tier="medium",
        inference_memory_tier="medium",
        relative_training_cost="~2x vs 7B",
        relative_inference_cost="~2x vs 7B",
        notes="Recommended larger MPS tier.",
    ),
    ModelEstimate(
        profile="mps-mistral31-24b",
        model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        params="24B",
        target_hardware="Apple Silicon (M4 Max high-memory)",
        training_memory_tier="high",
        inference_memory_tier="high",
        relative_training_cost="~3-4x vs 7B",
        relative_inference_cost="~3x vs 7B",
        notes="High-quality local Apple Silicon tier.",
    ),
    ModelEstimate(
        profile="mps-qwen25-32b",
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        params="32B",
        target_hardware="Apple Silicon (highest-memory configurations)",
        training_memory_tier="very_high",
        inference_memory_tier="very_high",
        relative_training_cost="~5x+ vs 7B",
        relative_inference_cost="~4x+ vs 7B",
        notes="Stretch tier; tune sequence/batch aggressively.",
    ),
]


def model_estimates_json() -> str:
    payload = {
        "notes": [
            "Estimates are resource-tier and relative-cost guidance, not wall-clock duration guarantees.",
            "Use gtsm benchmark-generate and training metrics to calibrate local timing empirically.",
        ],
        "estimates": [asdict(item) for item in ESTIMATES],
    }
    return json.dumps(payload, indent=2)
