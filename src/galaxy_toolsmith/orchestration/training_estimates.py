from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from galaxy_toolsmith.inference.artifacts import (
    ARTIFACT_FORMAT_UDT_YAML,
    ARTIFACT_FORMAT_XML,
    TRAINING_ARTIFACT_FORMAT_MIXED,
    normalize_training_artifact_format,
)
from galaxy_toolsmith.inference.source_context import (
    SOURCE_CONTEXT_MODES,
    SourceContextSettings,
    build_source_context_variants_from_record,
)
from galaxy_toolsmith.inference.source_context import (
    source_context_settings as make_source_context_settings,
)
from galaxy_toolsmith.models.training import TrainingProfile
from galaxy_toolsmith.orchestration.training import (
    _append_conversion_training_sample,
    _append_repair_training_sample,
    _append_training_sample,
    _record_training_sidecar_context,
    _resolve_training_udt_target,
    _resolve_training_xml_target,
    _training_data_diagnostics,
    _update_training_quality_diagnostics,
    _update_source_context_diagnostics,
)

DEFAULT_CHARS_PER_TOKEN = 3.7
DEFAULT_LONGEST_SAMPLE_COUNT = 25
DEFAULT_CONTEXT_LENGTHS = (12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072)
DEFAULT_SOURCE_BUDGET_LADDER = {
    2048: (3000, 16),
    4096: (6000, 32),
    8192: (12000, 64),
    12288: (24000, 96),
    16384: (36000, 128),
    24576: (64000, 192),
    32768: (96000, 256),
    49152: (144000, 384),
    65536: (192000, 512),
    98304: (288000, 768),
    131072: (384000, 1024),
}

_CONTEXT_LENGTH_RE = re.compile(r"^\s*(?P<size>\d+(?:\.\d+)?)\s*(?P<unit>k|m)?\s*$", re.I)
_DIAGNOSTIC_EXAMPLE_KEYS = (
    "missing_xml_target_examples",
    "empty_xml_target_examples",
    "skipped_non_tool_xml_target_examples",
    "missing_udt_target_examples",
    "empty_udt_target_examples",
    "help_only_command_examples",
    "optional_only_input_examples",
    "missing_effective_help_examples",
)


@dataclass(frozen=True)
class _EstimateCase:
    key: tuple[Any, ...]
    settings: SourceContextSettings
    max_seq_lengths: tuple[int, ...]


@dataclass
class _EstimateAccumulator:
    case: _EstimateCase
    diagnostics: dict[str, Any]
    token_lengths: list[int] = field(default_factory=list)
    char_lengths: list[int] = field(default_factory=list)
    by_task: Counter[str] = field(default_factory=Counter)
    longest_samples: list[dict[str, Any]] = field(default_factory=list)


def parse_context_lengths(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None or value == "":
        return DEFAULT_CONTEXT_LENGTHS
    if isinstance(value, str):
        raw_values: Iterable[str | int] = value.split(",")
    else:
        raw_values = value
    parsed: list[int] = []
    for raw in raw_values:
        if isinstance(raw, int):
            length = raw
        else:
            match = _CONTEXT_LENGTH_RE.match(raw)
            if not match:
                raise ValueError(f"Invalid context length: {raw!r}")
            multiplier = 1
            unit = (match.group("unit") or "").lower()
            if unit == "k":
                multiplier = 1024
            elif unit == "m":
                multiplier = 1024 * 1024
            length = int(float(match.group("size")) * multiplier)
        if length < 1:
            raise ValueError("Context lengths must be positive.")
        parsed.append(length)
    return tuple(dict.fromkeys(parsed))


def parse_source_context_modes(value: str | Sequence[str] | None, default: str) -> tuple[str, ...]:
    if value is None or value == "":
        raw_modes: Iterable[str] = (default,)
    elif isinstance(value, str):
        raw_modes = value.split(",")
    else:
        raw_modes = value
    modes: list[str] = []
    for raw in raw_modes:
        mode = raw.strip()
        if not mode:
            continue
        if mode not in SOURCE_CONTEXT_MODES:
            raise ValueError(f"Invalid source context mode: {mode!r}")
        modes.append(mode)
    return tuple(dict.fromkeys(modes or [default]))


def _training_text(record: dict[str, str]) -> str:
    return "\n\n".join(
        value
        for value in (
            str(record.get("instruction", "")),
            str(record.get("input", "")),
            str(record.get("output", "")),
        )
        if value
    )


def _classify_training_sample(record: dict[str, str]) -> str:
    instruction = str(record.get("instruction", "")).lstrip()
    output = str(record.get("output", "")).lstrip()
    if instruction.startswith("Repair the following generated Galaxy tool XML"):
        return "xml_repair"
    if instruction.startswith("Convert the following Galaxy User-Defined Tool YAML"):
        return "udt_to_xml"
    if output.startswith("<tool"):
        return "xml_generate"
    if output.startswith("class: GalaxyUserTool"):
        return "udt_yaml_generate"
    return "unknown"


def _load_exact_tokenizer(profile: TrainingProfile) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer
    except Exception as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Exact token estimation requires transformers to be installed."
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            profile.base_model,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as error:  # pragma: no cover - depends on local model cache
        raise RuntimeError(
            "Could not load the profile tokenizer from the local model cache; "
            "rerun without --exact-tokenizer or cache the model tokenizer first."
        ) from error

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True))

    return count_tokens


def _estimate_token_count(text: str, *, chars_per_token: float) -> int:
    return max(1, math.ceil(len(text) / chars_per_token))


def _percentile(sorted_values: Sequence[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return int(sorted_values[lower])
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return int(round(lower_value + (upper_value - lower_value) * (rank - lower)))


def _summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 0.50),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": round(sum(sorted_values) / len(sorted_values), 2),
    }


def _source_settings_for_case(
    base_settings: SourceContextSettings,
    *,
    mode: str,
    max_seq_length: int,
    source_context_budget_ladder: bool,
) -> SourceContextSettings:
    max_chars = base_settings.max_chars
    max_files = base_settings.max_files
    if source_context_budget_ladder:
        max_chars, max_files = DEFAULT_SOURCE_BUDGET_LADDER.get(
            max_seq_length,
            (max_chars, max_files),
        )
    return make_source_context_settings(
        mode=mode,
        max_chars=max_chars,
        max_files=max_files,
        source_root=base_settings.source_root,
        source_file=base_settings.source_file,
        test_context_mode=base_settings.test_context_mode,
        test_context_max_chars=base_settings.test_context_max_chars,
        test_context_max_files=base_settings.test_context_max_files,
        test_context_max_file_bytes=base_settings.test_context_max_file_bytes,
    )


def _threshold_summary(token_lengths: Sequence[int], max_seq_length: int) -> dict[str, Any]:
    over_count = sum(1 for length in token_lengths if length > max_seq_length)
    sample_count = len(token_lengths)
    return {
        "max_seq_length": max_seq_length,
        "over_max_seq_length": over_count,
        "fits_max_seq_length": sample_count - over_count,
        "overflow_fraction": round(over_count / sample_count, 6) if sample_count else 0.0,
    }


def _case_key(settings: SourceContextSettings) -> tuple[Any, ...]:
    return (
        settings.mode,
        settings.max_chars,
        settings.max_files,
        str(settings.source_root or ""),
        str(settings.source_file or ""),
        settings.test_context_mode,
        settings.test_context_max_chars,
        settings.test_context_max_files,
        settings.test_context_max_file_bytes,
    )


def _estimate_cases(
    *,
    base_settings: SourceContextSettings,
    modes: Sequence[str],
    lengths: Sequence[int],
    source_context_budget_ladder: bool,
) -> list[_EstimateCase]:
    grouped_lengths: dict[tuple[Any, ...], list[int]] = {}
    settings_by_key: dict[tuple[Any, ...], SourceContextSettings] = {}
    for mode in modes:
        for max_seq_length in lengths:
            settings = _source_settings_for_case(
                base_settings,
                mode=mode,
                max_seq_length=max_seq_length,
                source_context_budget_ladder=source_context_budget_ladder,
            )
            key = _case_key(settings)
            grouped_lengths.setdefault(key, []).append(max_seq_length)
            settings_by_key[key] = settings
    return [
        _EstimateCase(
            key=key,
            settings=settings_by_key[key],
            max_seq_lengths=tuple(group_lengths),
        )
        for key, group_lengths in grouped_lengths.items()
    ]


def _new_case_diagnostics(
    *,
    artifact_format: str,
    settings: SourceContextSettings,
) -> dict[str, Any]:
    diagnostics = _training_data_diagnostics()
    diagnostics["artifact_format"] = artifact_format
    diagnostics["source_context_mode"] = settings.mode
    diagnostics["test_context_mode"] = settings.test_context_mode
    return diagnostics


def _merge_diagnostics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in {"artifact_format", "source_context_mode", "test_context_mode"}:
            continue
        if key == "target_source_counts" and isinstance(value, Mapping):
            counts = target.setdefault("target_source_counts", {})
            if isinstance(counts, dict):
                for count_key, count_value in value.items():
                    counts[count_key] = int(counts.get(count_key, 0) or 0) + int(
                        count_value or 0
                    )
            continue
        if key in _DIAGNOSTIC_EXAMPLE_KEYS and isinstance(value, list):
            examples = target.setdefault(key, [])
            if isinstance(examples, list):
                for example in value:
                    if len(examples) >= 10:
                        break
                    examples.append(example)
            continue
        if isinstance(value, int):
            target[key] = int(target.get(key, 0) or 0) + value


def _increment_diagnostic(
    accumulators: dict[tuple[Any, ...], _EstimateAccumulator],
    key: str,
) -> None:
    for accumulator in accumulators.values():
        accumulator.diagnostics[key] = int(accumulator.diagnostics.get(key, 0) or 0) + 1


def _sample_metadata(
    record: Mapping[str, Any],
    *,
    training_record: Mapping[str, str],
    text: str,
    token_length: int,
    source_code: str,
    included_source_files: int,
    source_truncated: bool,
    included_test_files: int = 0,
    test_context_truncated: bool = False,
) -> dict[str, Any]:
    requirement_packages = record.get("requirement_packages", [])
    requirement_versions = record.get("requirement_versions", [])
    if not isinstance(requirement_packages, list):
        requirement_packages = []
    if not isinstance(requirement_versions, list):
        requirement_versions = []
    return {
        "token_length": token_length,
        "char_length": len(text),
        "task": _classify_training_sample(dict(training_record)),
        "tool_name": str(record.get("tool_name", "")),
        "tool_id": str(record.get("tool_id", "")),
        "package_id": str(record.get("package_id", "")),
        "primary_command": str(record.get("primary_command", "")),
        "wrapper_path": str(record.get("wrapper_path", "")),
        "expanded_xml_path": str(record.get("expanded_xml_path", "")),
        "udt_yaml_path": str(record.get("udt_yaml_path", "")),
        "requirements": [
            {
                "package": str(package),
                "version": str(requirement_versions[index])
                if index < len(requirement_versions)
                else "",
            }
            for index, package in enumerate(requirement_packages[:10])
        ],
        "source_context_chars": len(source_code),
        "source_context_files": included_source_files,
        "source_context_truncated": source_truncated,
        "test_context_files": included_test_files,
        "test_context_truncated": test_context_truncated,
    }


def _trim_longest_samples(samples: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        samples.clear()
        return
    samples.sort(
        key=lambda sample: (
            int(sample.get("token_length", 0) or 0),
            int(sample.get("char_length", 0) or 0),
            str(sample.get("tool_name", "")),
            str(sample.get("wrapper_path", "")),
        ),
        reverse=True,
    )
    del samples[limit:]


def _normalize_workers(workers: int | None) -> int:
    value = 0 if workers is None else int(workers)
    if value < 0:
        raise ValueError("workers must be greater than or equal to 0.")
    if value == 0:
        return max(1, min(8, os.cpu_count() or 1))
    return value


def _count_tokens_for_text(
    text: str,
    *,
    token_counter: Callable[[str], int] | None,
    token_counter_lock: threading.Lock | None,
    chars_per_token: float,
) -> int:
    if token_counter is None:
        return _estimate_token_count(text, chars_per_token=chars_per_token)
    if token_counter_lock is None:
        return token_counter(text)
    with token_counter_lock:
        return token_counter(text)


def _process_estimate_record(
    record: dict[str, Any],
    *,
    profile: TrainingProfile,
    repo_root: Path,
    artifact_format: str,
    cases: Sequence[_EstimateCase],
    token_counter: Callable[[str], int] | None,
    token_counter_lock: threading.Lock | None,
    chars_per_token: float,
    longest_sample_count: int,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    target_diagnostics = _training_data_diagnostics()
    xml_path: Path | None = None
    xml_source = ""
    xml_target = ""
    udt_path: Path | None = None
    udt_source = ""
    udt_target = ""
    if artifact_format in {ARTIFACT_FORMAT_XML, TRAINING_ARTIFACT_FORMAT_MIXED}:
        xml_path, xml_source, xml_target = _resolve_training_xml_target(
            record,
            repo_root=repo_root,
            diagnostics=target_diagnostics,
        )
    if artifact_format in {ARTIFACT_FORMAT_UDT_YAML, TRAINING_ARTIFACT_FORMAT_MIXED}:
        udt_path, udt_source, udt_target = _resolve_training_udt_target(
            record,
            repo_root=repo_root,
            diagnostics=target_diagnostics,
        )
    _update_training_quality_diagnostics(target_diagnostics, record, xml_target=xml_target)

    has_trainable_target = (
        artifact_format in {ARTIFACT_FORMAT_XML, TRAINING_ARTIFACT_FORMAT_MIXED}
        and xml_path is not None
        and bool(xml_target)
    ) or (
        artifact_format in {ARTIFACT_FORMAT_UDT_YAML, TRAINING_ARTIFACT_FORMAT_MIXED}
        and udt_path is not None
        and bool(udt_target)
    )
    source_context_results = (
        build_source_context_variants_from_record(record, [case.settings for case in cases])
        if has_trainable_target
        else ()
    )

    results: dict[tuple[Any, ...], dict[str, Any]] = {}
    tool_name = str(record.get("tool_name", "")).strip()
    for index, case in enumerate(cases):
        diagnostics = _new_case_diagnostics(
            artifact_format=artifact_format,
            settings=case.settings,
        )
        _merge_diagnostics(diagnostics, target_diagnostics)
        source_code = str(record.get("documentation", ""))
        if has_trainable_target:
            source_context = source_context_results[index]
            _update_source_context_diagnostics(diagnostics, source_context)
            if source_context.text.strip():
                source_code = source_context.text
            sidecar_context = _record_training_sidecar_context(record)
            if sidecar_context:
                source_code = "\n\n".join(part for part in (source_code, sidecar_context) if part)
            included_source_files = source_context.included_files
            source_truncated = source_context.truncated
            included_test_files = source_context.included_test_files
            test_context_truncated = source_context.test_context_truncated
        else:
            included_source_files = 0
            source_truncated = False
            included_test_files = 0
            test_context_truncated = False

        records: list[dict[str, str]] = []
        if (
            artifact_format in {ARTIFACT_FORMAT_XML, TRAINING_ARTIFACT_FORMAT_MIXED}
            and xml_path is not None
            and xml_target
        ):
            _append_training_sample(
                records,
                diagnostics=diagnostics,
                target_source=xml_source,
                task="xml_generate",
                profile=profile,
                record=record,
                tool_name=tool_name,
                output=xml_target,
                source_code=source_code,
            )
            _append_repair_training_sample(
                records,
                diagnostics=diagnostics,
                profile=profile,
                record=record,
                tool_name=tool_name,
                xml_target=xml_target,
                source_code=source_code,
            )
        if (
            artifact_format in {ARTIFACT_FORMAT_UDT_YAML, TRAINING_ARTIFACT_FORMAT_MIXED}
            and udt_path is not None
            and udt_target
        ):
            _append_training_sample(
                records,
                diagnostics=diagnostics,
                target_source=udt_source,
                task="udt_yaml_generate",
                profile=profile,
                record=record,
                tool_name=tool_name,
                output=udt_target,
                source_code=source_code,
            )
        if (
            artifact_format == TRAINING_ARTIFACT_FORMAT_MIXED
            and udt_path is not None
            and udt_target
            and xml_path is not None
            and xml_target
        ):
            _append_conversion_training_sample(
                records,
                diagnostics=diagnostics,
                profile=profile,
                tool_name=tool_name,
                udt_target=udt_target,
                xml_target=xml_target,
            )
        diagnostics["trainable_samples"] = len(records)
        training_texts = [_training_text(training_record) for training_record in records]
        token_lengths = [
            _count_tokens_for_text(
                text,
                token_counter=token_counter,
                token_counter_lock=token_counter_lock,
                chars_per_token=chars_per_token,
            )
            for text in training_texts
        ]
        longest_samples = [
            _sample_metadata(
                record,
                training_record=training_record,
                text=text,
                token_length=token_length,
                source_code=source_code,
                included_source_files=included_source_files,
                source_truncated=source_truncated,
                included_test_files=included_test_files,
                test_context_truncated=test_context_truncated,
            )
            for training_record, text, token_length in zip(
                records,
                training_texts,
                token_lengths,
                strict=True,
            )
        ]
        _trim_longest_samples(longest_samples, longest_sample_count)
        results[case.key] = {
            "diagnostics": diagnostics,
            "token_lengths": token_lengths,
            "char_lengths": [len(text) for text in training_texts],
            "by_task": Counter(_classify_training_sample(training_record) for training_record in records),
            "longest_samples": longest_samples,
        }
    return results


def _aggregate_record_result(
    accumulators: dict[tuple[Any, ...], _EstimateAccumulator],
    result: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    longest_sample_count: int,
) -> None:
    for key, payload in result.items():
        accumulator = accumulators[key]
        _merge_diagnostics(accumulator.diagnostics, payload.get("diagnostics", {}))
        accumulator.token_lengths.extend(int(value) for value in payload.get("token_lengths", []))
        accumulator.char_lengths.extend(int(value) for value in payload.get("char_lengths", []))
        by_task = payload.get("by_task", {})
        if isinstance(by_task, Counter):
            accumulator.by_task.update(by_task)
        elif isinstance(by_task, Mapping):
            accumulator.by_task.update({str(k): int(v) for k, v in by_task.items()})
        longest_samples = payload.get("longest_samples", [])
        if isinstance(longest_samples, list):
            accumulator.longest_samples.extend(
                sample for sample in longest_samples if isinstance(sample, dict)
            )
            _trim_longest_samples(accumulator.longest_samples, longest_sample_count)


def estimate_training_tokens(
    *,
    profile: TrainingProfile,
    corpus_jsonl_path: Path,
    repo_root: Path,
    artifact_format: str = ARTIFACT_FORMAT_XML,
    source_context_settings: SourceContextSettings | None = None,
    source_context_modes: Sequence[str] | None = None,
    max_seq_lengths: Sequence[int] | None = None,
    source_context_budget_ladder: bool = False,
    limit: int | None = None,
    exact_tokenizer: bool = False,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    progress_interval: int = 0,
    workers: int = 0,
    longest_sample_count: int = DEFAULT_LONGEST_SAMPLE_COUNT,
) -> dict[str, Any]:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided.")
    if longest_sample_count < 0:
        raise ValueError("longest_sample_count must be greater than or equal to 0.")
    if not corpus_jsonl_path.exists():
        raise FileNotFoundError(f"Corpus JSONL not found: {corpus_jsonl_path}")
    repo_root = repo_root.resolve()
    artifact_format = normalize_training_artifact_format(artifact_format)
    base_settings = (source_context_settings or SourceContextSettings()).normalized()
    modes = parse_source_context_modes(source_context_modes, default=base_settings.mode)
    lengths = parse_context_lengths(max_seq_lengths)
    cases = _estimate_cases(
        base_settings=base_settings,
        modes=modes,
        lengths=lengths,
        source_context_budget_ladder=source_context_budget_ladder,
    )
    token_counter = _load_exact_tokenizer(profile) if exact_tokenizer else None
    token_counter_lock = threading.Lock() if token_counter is not None else None
    worker_count = _normalize_workers(workers)
    accumulators = {
        case.key: _EstimateAccumulator(
            case=case,
            diagnostics=_new_case_diagnostics(
                artifact_format=artifact_format,
                settings=case.settings,
            ),
        )
        for case in cases
    }
    futures = []
    completed_records = 0
    last_reported = 0

    def report_progress(force: bool = False) -> None:
        nonlocal last_reported
        if progress_interval <= 0:
            return
        if not force and completed_records - last_reported < progress_interval:
            return
        last_reported = completed_records
        max_samples = max(
            (
                int(accumulator.diagnostics.get("trainable_samples", 0) or 0)
                for accumulator in accumulators.values()
            ),
            default=0,
        )
        print(
            "estimated "
            f"{completed_records} corpus records / {max_samples} samples "
            f"across {len(cases)} source-context cases with workers={worker_count}",
            file=sys.stderr,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        with corpus_jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                _increment_diagnostic(accumulators, "total_corpus_records")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    _increment_diagnostic(accumulators, "invalid_json_records")
                    completed_records += 1
                    report_progress()
                    if limit is not None and completed_records >= limit:
                        break
                    continue
                tool_name = str(record.get("tool_name", "")).strip()
                if not tool_name:
                    _increment_diagnostic(accumulators, "missing_tool_name_count")
                    completed_records += 1
                    report_progress()
                    if limit is not None and completed_records >= limit:
                        break
                    continue
                futures.append(
                    executor.submit(
                        _process_estimate_record,
                        record,
                        profile=profile,
                        repo_root=repo_root,
                        artifact_format=artifact_format,
                        cases=cases,
                        token_counter=token_counter,
                        token_counter_lock=token_counter_lock,
                        chars_per_token=chars_per_token,
                        longest_sample_count=longest_sample_count,
                    )
                )
                if limit is not None and int(
                    next(iter(accumulators.values())).diagnostics["total_corpus_records"]
                ) >= limit:
                    break

        for future in as_completed(futures):
            _aggregate_record_result(
                accumulators,
                future.result(),
                longest_sample_count=longest_sample_count,
            )
            completed_records += 1
            report_progress()

    report_progress(force=True)

    if not any(accumulator.token_lengths for accumulator in accumulators.values()):
        raise RuntimeError("No trainable samples found in corpus JSONL.")
    estimates: list[dict[str, Any]] = []
    for case in cases:
        accumulator = accumulators[case.key]
        estimates.append(
            {
                "source_context": case.settings.to_dict(),
                "artifact_format": artifact_format.replace("_", "-"),
                "samples": len(accumulator.token_lengths),
                "by_task": dict(sorted(accumulator.by_task.items())),
                "token_summary": _summary(accumulator.token_lengths),
                "char_summary": _summary(accumulator.char_lengths),
                "longest_samples": accumulator.longest_samples,
                "thresholds": [
                    _threshold_summary(accumulator.token_lengths, max_seq_length)
                    for max_seq_length in case.max_seq_lengths
                ],
                "training_data_diagnostics": accumulator.diagnostics,
            }
        )

    passing = [
        {
            "source_context": estimate["source_context"],
            **threshold,
        }
        for estimate in estimates
        for threshold in estimate["thresholds"]
        if int(threshold["over_max_seq_length"]) == 0
    ]
    recommendation = min(passing, key=lambda item: int(item["max_seq_length"])) if passing else {}
    return {
        "profile": profile.name,
        "base_model": profile.base_model,
        "artifact_format": artifact_format.replace("_", "-"),
        "context_lengths": list(lengths),
        "source_context_budget_ladder": source_context_budget_ladder,
        "limit": limit,
        "exact_tokenizer": exact_tokenizer,
        "chars_per_token": None if exact_tokenizer else chars_per_token,
        "workers": worker_count,
        "longest_sample_count": longest_sample_count,
        "estimates": estimates,
        "recommendation": recommendation,
    }
