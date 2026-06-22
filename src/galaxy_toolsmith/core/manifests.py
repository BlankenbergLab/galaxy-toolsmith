from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

SCHEMA_VERSION = "0.1.0"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SourceRef:
    name: str
    url: str
    ref: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    sources: list[SourceRef] = field(default_factory=list)
    transforms: list[str] = field(default_factory=list)
    includes_tests: bool = True
    includes_datatype_report: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class ModelVariantManifest:
    variant_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    base_model: str = ""
    quantization: str = "none"
    training_dataset_id: str = ""
    provider: str = "local"
    skills_profile: str = "default"
    backend: str = "auto"
    artifact_dir: str = ""
    export_quantizations: list[str] = field(default_factory=list)
    ollama_model_name: str = ""
    ollama_modelfile_path: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class TrainingRunManifest:
    run_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    profile_name: str = "default"
    backend: str = "auto"
    provider: str = "local"
    base_model: str = ""
    quantization: str = "none"
    dataset_manifest_path: str = ""
    dataset_id: str = ""
    command: list[str] = field(default_factory=list)
    status: str = "pending"
    output_dir: str = ""
    checkpoints_dir: str = ""
    metrics_path: str = ""
    model_variant_path: str = ""
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
