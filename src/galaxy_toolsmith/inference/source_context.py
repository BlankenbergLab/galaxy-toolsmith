from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

SOURCE_CONTEXT_MODE_NONE = "none"
SOURCE_CONTEXT_MODE_METADATA = "metadata"
SOURCE_CONTEXT_MODE_SNIPPETS = "snippets"
SOURCE_CONTEXT_MODE_ALL_FILTERED = "all-filtered"
SOURCE_CONTEXT_MODE_ALL_RAW = "all-raw"
SOURCE_CONTEXT_MODES = (
    SOURCE_CONTEXT_MODE_NONE,
    SOURCE_CONTEXT_MODE_METADATA,
    SOURCE_CONTEXT_MODE_SNIPPETS,
    SOURCE_CONTEXT_MODE_ALL_FILTERED,
    SOURCE_CONTEXT_MODE_ALL_RAW,
)
TEST_CONTEXT_MODE_NONE = "none"
TEST_CONTEXT_MODE_METADATA = "metadata"
TEST_CONTEXT_MODE_SNIPPETS = "snippets"
TEST_CONTEXT_MODE_FIXTURES = "fixtures"
TEST_CONTEXT_MODES = (
    TEST_CONTEXT_MODE_NONE,
    TEST_CONTEXT_MODE_METADATA,
    TEST_CONTEXT_MODE_SNIPPETS,
    TEST_CONTEXT_MODE_FIXTURES,
)

DEFAULT_SOURCE_CONTEXT_MAX_CHARS = 8000
DEFAULT_SOURCE_CONTEXT_MAX_FILES = 12
DEFAULT_TEST_CONTEXT_MAX_CHARS = 4000
DEFAULT_TEST_CONTEXT_MAX_FILES = 4
DEFAULT_TEST_CONTEXT_MAX_FILE_BYTES = 64_000
MAX_SOURCE_FILE_BYTES = 256_000
FILE_SAMPLE_BYTES = 8192

SOURCE_EXTENSIONS = {
    ".awk",
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cwl",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".jl",
    ".js",
    ".json",
    ".lua",
    ".md",
    ".nf",
    ".pl",
    ".pm",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".wdl",
    ".yaml",
    ".yml",
}
SOURCE_BASENAMES = {
    "Snakefile",
    "Makefile",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "requirements.txt",
}
BINARY_OR_DATA_EXTENSIONS = {
    ".7z",
    ".a",
    ".bam",
    ".bcf",
    ".bz2",
    ".cram",
    ".db",
    ".dll",
    ".dylib",
    ".fa",
    ".fasta",
    ".fastq",
    ".fq",
    ".gz",
    ".h5",
    ".jpg",
    ".jpeg",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".pyd",
    ".pkl",
    ".sam",
    ".so",
    ".sqlite",
    ".tar",
    ".tgz",
    ".tiff",
    ".vcf",
    ".xz",
    ".zip",
}
EXCLUDED_DIRS = {
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs/_build",
    "node_modules",
    "site-packages",
    "target",
    "test",
    "test-data",
    "test_data",
    "tests",
    "venv",
    "vendor",
    "vendors",
}
TEST_CONTEXT_DIRS = {
    "api-test",
    "api-tests",
    "demo",
    "demos",
    "example",
    "examples",
    "fixture",
    "fixtures",
    "test",
    "test-data",
    "test_data",
    "tests",
}
TEST_CONTEXT_SCRIPT_EXTENSIONS = SOURCE_EXTENSIONS | {
    ".expected",
    ".out",
    ".stderr",
    ".stdout",
}
TEST_CONTEXT_FIXTURE_EXTENSIONS = {
    ".bed",
    ".csv",
    ".fa",
    ".fasta",
    ".fastq",
    ".fq",
    ".gff",
    ".gtf",
    ".json",
    ".paf",
    ".sam",
    ".tab",
    ".tabular",
    ".tsv",
    ".txt",
    ".vcf",
    ".yaml",
    ".yml",
}
CLI_PATTERNS = (
    "ArgumentParser",
    "OptionParser",
    "argparse",
    "click.",
    "docopt",
    "entry_points",
    "console_scripts",
    "typer.",
)


@dataclass(frozen=True)
class SourceContextSettings:
    mode: str = SOURCE_CONTEXT_MODE_NONE
    max_chars: int = DEFAULT_SOURCE_CONTEXT_MAX_CHARS
    max_files: int = DEFAULT_SOURCE_CONTEXT_MAX_FILES
    source_root: Path | None = None
    source_file: Path | None = None
    test_context_mode: str = TEST_CONTEXT_MODE_NONE
    test_context_max_chars: int = DEFAULT_TEST_CONTEXT_MAX_CHARS
    test_context_max_files: int = DEFAULT_TEST_CONTEXT_MAX_FILES
    test_context_max_file_bytes: int = DEFAULT_TEST_CONTEXT_MAX_FILE_BYTES

    def normalized(self) -> SourceContextSettings:
        mode = normalize_source_context_mode(self.mode)
        test_context_mode = normalize_test_context_mode(self.test_context_mode)
        max_chars = max(0, int(self.max_chars))
        max_files = max(0, int(self.max_files))
        test_context_max_chars = max(0, int(self.test_context_max_chars))
        test_context_max_files = max(0, int(self.test_context_max_files))
        test_context_max_file_bytes = max(0, int(self.test_context_max_file_bytes))
        return replace(
            self,
            mode=mode,
            max_chars=max_chars,
            max_files=max_files,
            test_context_mode=test_context_mode,
            test_context_max_chars=test_context_max_chars,
            test_context_max_files=test_context_max_files,
            test_context_max_file_bytes=test_context_max_file_bytes,
        )

    def with_paths(
        self,
        *,
        source_root: Path | None = None,
        source_file: Path | None = None,
    ) -> SourceContextSettings:
        return replace(self, source_root=source_root, source_file=source_file).normalized()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_root"] = str(self.source_root or "")
        payload["source_file"] = str(self.source_file or "")
        return payload


@dataclass(frozen=True)
class SourceContextResult:
    text: str
    mode: str
    max_chars: int
    max_files: int
    metadata_sources: int = 0
    scanned_files: int = 0
    included_files: int = 0
    included_chars: int = 0
    truncated: bool = False
    included_paths: tuple[str, ...] = ()
    test_context_mode: str = TEST_CONTEXT_MODE_NONE
    included_test_files: int = 0
    included_test_chars: int = 0
    test_context_truncated: bool = False
    included_test_paths: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SourceFile:
    path: Path | None
    root: Path | None
    relpath: str
    score: int
    label: str = "Source file"
    inline_text: str | None = None
    included_path: str = ""


@dataclass(frozen=True)
class _PreparedSourceContext:
    settings: SourceContextSettings
    metadata_mappings: tuple[Mapping[str, Any], ...]
    wrapper_metadata: str
    candidate_files: tuple[_SourceFile, ...]
    test_candidate_files: tuple[_SourceFile, ...] = ()
    scanned_files: int = 0
    scanned_test_files: int = 0
    errors: tuple[str, ...] = ()


def normalize_source_context_mode(mode: str | None) -> str:
    normalized = (mode or SOURCE_CONTEXT_MODE_NONE).strip().lower().replace("_", "-")
    if normalized not in SOURCE_CONTEXT_MODES:
        raise ValueError(
            "Unsupported source context mode. Use one of: " + ", ".join(SOURCE_CONTEXT_MODES)
        )
    return normalized


def normalize_test_context_mode(mode: str | None) -> str:
    normalized = (mode or TEST_CONTEXT_MODE_NONE).strip().lower().replace("_", "-")
    if normalized not in TEST_CONTEXT_MODES:
        raise ValueError(
            "Unsupported test context mode. Use one of: " + ", ".join(TEST_CONTEXT_MODES)
        )
    return normalized


def source_context_settings(
    *,
    mode: str | None = None,
    max_chars: int | None = None,
    max_files: int | None = None,
    source_root: Path | None = None,
    source_file: Path | None = None,
    test_context_mode: str | None = None,
    test_context_max_chars: int | None = None,
    test_context_max_files: int | None = None,
    test_context_max_file_bytes: int | None = None,
) -> SourceContextSettings:
    return SourceContextSettings(
        mode=normalize_source_context_mode(mode),
        max_chars=(DEFAULT_SOURCE_CONTEXT_MAX_CHARS if max_chars is None else int(max_chars)),
        max_files=DEFAULT_SOURCE_CONTEXT_MAX_FILES if max_files is None else int(max_files),
        source_root=source_root,
        source_file=source_file,
        test_context_mode=normalize_test_context_mode(test_context_mode),
        test_context_max_chars=(
            DEFAULT_TEST_CONTEXT_MAX_CHARS
            if test_context_max_chars is None
            else int(test_context_max_chars)
        ),
        test_context_max_files=(
            DEFAULT_TEST_CONTEXT_MAX_FILES
            if test_context_max_files is None
            else int(test_context_max_files)
        ),
        test_context_max_file_bytes=(
            DEFAULT_TEST_CONTEXT_MAX_FILE_BYTES
            if test_context_max_file_bytes is None
            else int(test_context_max_file_bytes)
        ),
    ).normalized()


def build_source_context_from_record(
    record: Mapping[str, Any],
    settings: SourceContextSettings | None,
) -> SourceContextResult:
    settings = (settings or SourceContextSettings()).normalized()
    if (
        settings.mode == SOURCE_CONTEXT_MODE_NONE
        and settings.test_context_mode == TEST_CONTEXT_MODE_NONE
    ):
        return _empty_result(settings)

    prepared = _prepare_source_context_from_record(record, settings)
    return _render_prepared_source_context(prepared, settings=settings)


def build_source_context_variants_from_record(
    record: Mapping[str, Any],
    settings_list: Sequence[SourceContextSettings | None],
) -> tuple[SourceContextResult, ...]:
    """Build several source-context budgets while scanning each semantic mode once."""

    normalized = [
        (settings or SourceContextSettings()).normalized()
        for settings in settings_list
    ]
    results: list[SourceContextResult | None] = [None] * len(normalized)
    grouped_indexes: dict[tuple[str, str, str, str, int], list[int]] = {}
    for index, settings in enumerate(normalized):
        if (
            settings.mode == SOURCE_CONTEXT_MODE_NONE
            and settings.test_context_mode == TEST_CONTEXT_MODE_NONE
        ):
            results[index] = _empty_result(settings)
            continue
        key = (
            settings.mode,
            str(settings.source_root or ""),
            str(settings.source_file or ""),
            settings.test_context_mode,
            settings.test_context_max_file_bytes,
        )
        grouped_indexes.setdefault(key, []).append(index)

    for indexes in grouped_indexes.values():
        prepared = _prepare_source_context_from_record(record, normalized[indexes[0]])
        block_cache: dict[str, tuple[str, str]] = {}
        for index in indexes:
            results[index] = _render_prepared_source_context(
                prepared,
                settings=normalized[index],
                block_cache=block_cache,
            )

    return tuple(result or _empty_result(settings) for result, settings in zip(results, normalized))


def _prepare_source_context_from_record(
    record: Mapping[str, Any],
    settings: SourceContextSettings,
) -> _PreparedSourceContext:
    settings = settings.normalized()
    mappings = _record_bioconda_sources(record)
    roots = _source_roots_from_record(
        mappings,
        include_weak=settings.mode == SOURCE_CONTEXT_MODE_ALL_RAW,
    )
    wrapper_sources = _wrapper_sources_from_record(record)
    wrapper_metadata = _format_wrapper_source_metadata(record)
    if settings.source_root is not None:
        roots.append(settings.source_root)
    files = [settings.source_file] if settings.source_file is not None else []
    return _prepare_source_context(
        settings=settings,
        metadata_mappings=mappings,
        wrapper_metadata=wrapper_metadata,
        wrapper_sources=wrapper_sources,
        roots=roots,
        files=files,
    )


def build_source_context_from_paths(
    *,
    settings: SourceContextSettings | None,
    source_root: Path | None = None,
    source_file: Path | None = None,
) -> SourceContextResult:
    settings = settings or SourceContextSettings()
    source_root = source_root if source_root is not None else settings.source_root
    source_file = source_file if source_file is not None else settings.source_file
    settings = settings.with_paths(
        source_root=source_root,
        source_file=source_file,
    )
    if settings.mode == SOURCE_CONTEXT_MODE_NONE:
        if settings.test_context_mode != TEST_CONTEXT_MODE_NONE:
            return _build_source_context(
                settings=settings,
                metadata_mappings=[],
                wrapper_metadata="",
                wrapper_sources=[],
                roots=[source_root] if source_root is not None else [],
                files=[source_file] if source_file is not None else [],
            )
        if source_file is None:
            return _empty_result(settings)
        try:
            text = source_file.read_text(encoding="utf-8")
        except OSError as error:
            return _empty_result(settings, errors=(f"{source_file}: {error}",))
        return SourceContextResult(
            text=text,
            mode=settings.mode,
            max_chars=settings.max_chars,
            max_files=settings.max_files,
            included_files=1,
            included_chars=len(text),
            included_paths=(str(source_file),),
        )
    return _build_source_context(
        settings=settings,
        metadata_mappings=[],
        wrapper_metadata="",
        wrapper_sources=[],
        roots=[source_root] if source_root is not None else [],
        files=[source_file] if source_file is not None else [],
    )


def _empty_result(
    settings: SourceContextSettings,
    *,
    errors: tuple[str, ...] = (),
) -> SourceContextResult:
    return SourceContextResult(
        text="",
        mode=settings.mode,
        max_chars=settings.max_chars,
        max_files=settings.max_files,
        errors=errors,
        test_context_mode=settings.test_context_mode,
    )


def _record_bioconda_sources(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_sources = record.get("bioconda_sources", [])
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        return []
    return [source for source in raw_sources if isinstance(source, Mapping)]


def _record_sequence(record: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    raw = record.get(key, [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _wrapper_sources_from_record(record: Mapping[str, Any]) -> list[_SourceFile]:
    sources: list[_SourceFile] = []
    for item in _record_sequence(record, "wrapper_helper_files"):
        raw_path = str(item.get("path", "") or "").strip()
        relpath = str(item.get("relative_path", "") or Path(raw_path).name).strip()
        if not raw_path or not relpath:
            continue
        path = Path(raw_path).expanduser()
        if not path.exists() or not path.is_file() or not _is_readable_text_source(path):
            continue
        sources.append(
            _SourceFile(
                path=path,
                root=path.parent,
                relpath=relpath,
                score=1000 + _score_path(relpath, keywords=()),
                label="Existing wrapper helper file",
                included_path=str(path),
            )
        )
    for item in _record_sequence(record, "wrapper_configfiles"):
        content = str(item.get("content", "") or "")
        if not content.strip():
            continue
        name = str(item.get("name", "") or "configfile").strip()
        filename = str(item.get("filename", "") or "").strip()
        relpath = filename or name
        template_kind = str(item.get("template_kind") or item.get("role_hint") or "")
        referenced = bool(item.get("referenced_by_command"))
        base_score = 980 if template_kind == "script_template" and referenced else 950
        if template_kind == "config_template" and not referenced:
            base_score = 930
        elif template_kind == "other_template":
            base_score = 900
        sources.append(
            _SourceFile(
                path=None,
                root=None,
                relpath=relpath,
                score=base_score + _score_path(relpath, keywords=()),
                label="Existing wrapper configfile",
                inline_text=content,
                included_path=f"configfile:{name}",
            )
        )
    for item in _record_sequence(record, "wrapper_sidecar_files"):
        content = str(item.get("content", "") or "")
        if not content.strip():
            continue
        relpath = str(item.get("relative_path") or item.get("path") or "sidecar").strip()
        role = str(item.get("role", "") or "sidecar").strip()
        root_tag = str(item.get("root_tag", "") or "").strip()
        score = 940 + _score_path(relpath, keywords=())
        if role == "macros":
            score += 20
        if role.startswith("tool_data"):
            score += 15
        sources.append(
            _SourceFile(
                path=None,
                root=None,
                relpath=relpath,
                score=score,
                label=f"Existing wrapper sidecar ({role}{', root=<' + root_tag + '>' if root_tag else ''})",
                inline_text=content,
                included_path=f"sidecar:{relpath}",
            )
        )
    return sources


def _format_wrapper_source_metadata(record: Mapping[str, Any]) -> str:
    helper_files = _record_sequence(record, "wrapper_helper_files")
    configfiles = _record_sequence(record, "wrapper_configfiles")
    sidecars = _record_sequence(record, "wrapper_sidecar_files")
    summary = record.get("wrapper_source_summary", {})
    if not helper_files and not configfiles and not sidecars and not summary:
        return ""
    blocks = ["Wrapper source metadata:\n"]
    if isinstance(summary, Mapping):
        for key in (
            "helper_file_count",
            "configfile_count",
            "sidecar_file_count",
            "macro_sidecar_count",
            "tool_data_sidecar_count",
            "truncated_configfile_count",
            "skipped_file_count",
        ):
            value = summary.get(key)
            if value not in (None, ""):
                blocks.append(f"- {key}: {value}\n")
        skip_reasons = summary.get("skip_reasons")
        if isinstance(skip_reasons, Mapping) and skip_reasons:
            reasons = ", ".join(f"{key}={value}" for key, value in sorted(skip_reasons.items()))
            blocks.append(f"- skip_reasons: {reasons}\n")
    for item in helper_files:
        relpath = str(item.get("relative_path", "") or "").strip()
        sha = str(item.get("sha256", "") or "").strip()
        if relpath:
            blocks.append(f"- helper: {relpath}")
            if sha:
                blocks.append(f" sha256={sha[:16]}")
            blocks.append("\n")
    for item in configfiles:
        name = str(item.get("name", "") or "").strip()
        filename = str(item.get("filename", "") or "").strip()
        template_kind = str(item.get("template_kind") or item.get("role_hint") or "").strip()
        language = str(item.get("language", "") or "").strip()
        referenced = item.get("referenced_by_command")
        truncated = item.get("content_truncated")
        stored_byte_count = item.get("stored_byte_count")
        sha = str(item.get("sha256", "") or "").strip()
        label = filename or name
        if label:
            blocks.append(f"- configfile: {label}")
            details = []
            if template_kind:
                details.append(f"kind={template_kind}")
            if language:
                details.append(f"language={language}")
            if referenced in {True, False}:
                details.append(f"referenced_by_command={str(referenced).lower()}")
            if truncated in {True, False}:
                details.append(f"content_truncated={str(truncated).lower()}")
            if truncated is True and stored_byte_count not in (None, ""):
                details.append(f"stored_bytes={stored_byte_count}")
            if sha:
                details.append(f"sha256={sha[:16]}")
            if details:
                blocks.append(f" {' '.join(details)}")
            blocks.append("\n")
    for item in sidecars:
        relpath = str(item.get("relative_path") or item.get("path") or "").strip()
        if not relpath:
            continue
        role = str(item.get("role", "") or "sidecar").strip()
        root_tag = str(item.get("root_tag", "") or "").strip()
        truncated = item.get("content_truncated")
        sha = str(item.get("sha256", "") or "").strip()
        details = []
        if role:
            details.append(f"role={role}")
        if root_tag:
            details.append(f"root=<{root_tag}>")
        if truncated in {True, False}:
            details.append(f"content_truncated={str(truncated).lower()}")
        if sha:
            details.append(f"sha256={sha[:16]}")
        blocks.append(f"- sidecar: {relpath}")
        if details:
            blocks.append(f" {' '.join(details)}")
        blocks.append("\n")
    return "".join(blocks).rstrip() + "\n"


def _source_mapping_allows_file_context(
    mapping: Mapping[str, Any],
    *,
    include_weak: bool = False,
) -> bool:
    checkout = str(mapping.get("source_checkout", "")).strip()
    if not checkout:
        return False
    if bool(mapping.get("source_is_binary_artifact", False)):
        return False
    if Path(checkout).name.lower().endswith((".jar", ".whl", ".gem")):
        return False
    confidence = str(mapping.get("source_confidence", "") or "exact").strip().lower()
    if confidence == "weak" and not include_weak:
        return False
    return True


def _source_roots_from_record(
    mappings: Sequence[Mapping[str, Any]],
    *,
    include_weak: bool = False,
) -> list[Path]:
    roots: list[Path] = []
    for mapping in mappings:
        if _source_mapping_allows_file_context(mapping, include_weak=include_weak):
            checkout = str(mapping.get("source_checkout", "")).strip()
            roots.append(Path(checkout).expanduser())
    return _dedupe_existing_paths(roots)


def _dedupe_existing_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _build_source_context(
    *,
    settings: SourceContextSettings,
    metadata_mappings: Sequence[Mapping[str, Any]],
    wrapper_metadata: str,
    wrapper_sources: Sequence[_SourceFile],
    roots: Sequence[Path | None],
    files: Sequence[Path | None],
) -> SourceContextResult:
    prepared = _prepare_source_context(
        settings=settings,
        metadata_mappings=metadata_mappings,
        wrapper_metadata=wrapper_metadata,
        wrapper_sources=wrapper_sources,
        roots=roots,
        files=files,
    )
    return _render_prepared_source_context(prepared, settings=settings)


def _prepare_source_context(
    *,
    settings: SourceContextSettings,
    metadata_mappings: Sequence[Mapping[str, Any]],
    wrapper_metadata: str,
    wrapper_sources: Sequence[_SourceFile],
    roots: Sequence[Path | None],
    files: Sequence[Path | None],
) -> _PreparedSourceContext:
    settings = settings.normalized()
    keywords = _keywords_from_mappings(metadata_mappings)

    errors: list[str] = []
    scanned_files = 0
    scanned_test_files = 0
    candidate_files: list[_SourceFile] = []
    test_candidate_files: list[_SourceFile] = []

    if settings.mode not in {SOURCE_CONTEXT_MODE_NONE, SOURCE_CONTEXT_MODE_METADATA}:
        candidate_files.extend(wrapper_sources)
        for file_path in files:
            if file_path is None:
                continue
            source_file = _manual_source_file(file_path, keywords=keywords)
            if source_file is None:
                errors.append(f"{file_path}: not a readable text source file")
                continue
            candidate_files.append(source_file)

        for root in roots:
            if root is None:
                continue
            scanned, candidates, root_errors = _scan_source_root(
                root,
                mode=settings.mode,
                keywords=keywords,
            )
            scanned_files += scanned
            candidate_files.extend(candidates)
            errors.extend(root_errors)

    if settings.test_context_mode != TEST_CONTEXT_MODE_NONE:
        for file_path in files:
            if file_path is None:
                continue
            source_file = _manual_test_context_file(
                file_path,
                mode=settings.test_context_mode,
                max_file_bytes=settings.test_context_max_file_bytes,
                keywords=keywords,
            )
            if source_file is not None:
                test_candidate_files.append(source_file)
        for root in roots:
            if root is None:
                continue
            scanned, candidates, root_errors = _scan_test_context_root(
                root,
                mode=settings.test_context_mode,
                max_file_bytes=settings.test_context_max_file_bytes,
                keywords=keywords,
            )
            scanned_test_files += scanned
            test_candidate_files.extend(candidates)
            errors.extend(root_errors)

    candidate_files = _dedupe_source_files(candidate_files)
    candidate_files.sort(key=lambda item: (-item.score, item.relpath.lower()))
    test_candidate_files = _dedupe_test_context_files(
        test_candidate_files,
        existing_files=candidate_files,
    )
    test_candidate_files.sort(key=lambda item: (-item.score, item.relpath.lower()))
    return _PreparedSourceContext(
        settings=settings,
        metadata_mappings=tuple(metadata_mappings),
        wrapper_metadata=wrapper_metadata,
        candidate_files=tuple(candidate_files),
        test_candidate_files=tuple(test_candidate_files),
        scanned_files=scanned_files,
        scanned_test_files=scanned_test_files,
        errors=tuple(errors),
    )


def _render_prepared_source_context(
    prepared: _PreparedSourceContext,
    *,
    settings: SourceContextSettings | None = None,
    block_cache: dict[str, tuple[str, str]] | None = None,
) -> SourceContextResult:
    settings = (settings or prepared.settings).normalized()
    errors: list[str] = list(prepared.errors)
    parts: list[str] = []
    used = 0
    truncated = False
    included_paths: list[str] = []
    included_files = 0

    metadata_text = _format_source_metadata(prepared.metadata_mappings)
    if metadata_text:
        used, truncated = _append_with_budget(
            parts,
            used,
            metadata_text,
            max_chars=settings.max_chars,
            truncated=truncated,
        )
    if prepared.wrapper_metadata:
        used, truncated = _append_with_budget(
            parts,
            used,
            prepared.wrapper_metadata,
            max_chars=settings.max_chars,
            truncated=truncated,
        )

    if settings.mode == SOURCE_CONTEXT_MODE_METADATA:
        test_text, test_files, test_chars, test_truncated, test_paths, test_errors = (
            _render_test_context(
                prepared.test_candidate_files,
                settings=settings,
                block_cache=block_cache,
            )
        )
        parts.append(test_text)
        errors.extend(test_errors)
        return SourceContextResult(
            text="".join(parts).strip(),
            mode=settings.mode,
            max_chars=settings.max_chars,
            max_files=settings.max_files,
            metadata_sources=len(prepared.metadata_mappings),
            included_chars=used + test_chars,
            truncated=truncated,
            test_context_mode=settings.test_context_mode,
            included_test_files=test_files,
            included_test_chars=test_chars,
            test_context_truncated=test_truncated,
            included_test_paths=test_paths,
            errors=tuple(errors),
        )

    if settings.max_files:
        selected_files = prepared.candidate_files[: settings.max_files]
        if len(prepared.candidate_files) > len(selected_files):
            truncated = True
    else:
        selected_files = []
        if prepared.candidate_files:
            truncated = True

    for source_file in selected_files:
        block, file_error = _source_file_block(source_file, block_cache=block_cache)
        if file_error:
            errors.append(file_error)
            continue
        previous_used = used
        used, truncated = _append_with_budget(
            parts,
            used,
            block,
            max_chars=settings.max_chars,
            truncated=truncated,
        )
        if used > previous_used:
            included_files += 1
            included_paths.append(source_file.included_path or str(source_file.path or ""))
        if used >= settings.max_chars:
            break

    test_text, test_files, test_chars, test_truncated, test_paths, test_errors = _render_test_context(
        prepared.test_candidate_files,
        settings=settings,
        block_cache=block_cache,
    )
    parts.append(test_text)
    errors.extend(test_errors)

    return SourceContextResult(
        text="".join(parts).strip(),
        mode=settings.mode,
        max_chars=settings.max_chars,
        max_files=settings.max_files,
        metadata_sources=len(prepared.metadata_mappings),
        scanned_files=prepared.scanned_files + prepared.scanned_test_files,
        included_files=included_files,
        included_chars=used + test_chars,
        truncated=truncated,
        included_paths=tuple(included_paths),
        test_context_mode=settings.test_context_mode,
        included_test_files=test_files,
        included_test_chars=test_chars,
        test_context_truncated=test_truncated,
        included_test_paths=test_paths,
        errors=tuple(errors[:10]),
    )


def _render_test_context(
    candidate_files: Sequence[_SourceFile],
    *,
    settings: SourceContextSettings,
    block_cache: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, int, int, bool, tuple[str, ...], tuple[str, ...]]:
    if (
        settings.test_context_mode == TEST_CONTEXT_MODE_NONE
        or not candidate_files
        or settings.test_context_max_chars <= 0
    ):
        return "", 0, 0, bool(candidate_files and settings.test_context_max_chars <= 0), (), ()

    parts: list[str] = []
    used = 0
    truncated = False
    errors: list[str] = []
    included_paths: list[str] = []
    included_files = 0
    header = (
        "\nSource test/example context:\n"
        "These files are optional behavioral sidecars for examples, expected outputs, "
        "and small fixtures; the primary generation target remains the tool wrapper.\n"
    )
    used, truncated = _append_with_budget(
        parts,
        used,
        header,
        max_chars=settings.test_context_max_chars,
        truncated=truncated,
    )

    if settings.test_context_max_files:
        selected_files = candidate_files[: settings.test_context_max_files]
        if len(candidate_files) > len(selected_files):
            truncated = True
    else:
        selected_files = []
        truncated = True

    for source_file in selected_files:
        if settings.test_context_mode == TEST_CONTEXT_MODE_METADATA:
            block, file_error = _test_context_metadata_block(source_file)
        else:
            block, file_error = _source_file_block(source_file, block_cache=block_cache)
        if file_error:
            errors.append(file_error)
            continue
        previous_used = used
        used, truncated = _append_with_budget(
            parts,
            used,
            block,
            max_chars=settings.test_context_max_chars,
            truncated=truncated,
        )
        if used > previous_used:
            included_files += 1
            included_paths.append(source_file.included_path or str(source_file.path or ""))
        if used >= settings.test_context_max_chars:
            break

    return "".join(parts), included_files, used, truncated, tuple(included_paths), tuple(errors)


def _source_file_cache_key(source_file: _SourceFile) -> str:
    if source_file.inline_text is not None:
        digest = hashlib.sha256(source_file.inline_text.encode("utf-8", errors="replace")).hexdigest()
        return f"inline:{source_file.label}:{source_file.relpath}:{digest}"
    if source_file.path is not None:
        try:
            return str(source_file.path.resolve())
        except OSError:
            return str(source_file.path)
    return f"unknown:{source_file.label}:{source_file.relpath}"


def _source_file_block(
    source_file: _SourceFile,
    *,
    block_cache: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str]:
    cache_key = _source_file_cache_key(source_file)
    if block_cache is not None and cache_key in block_cache:
        return block_cache[cache_key]
    if source_file.inline_text is not None:
        text = source_file.inline_text
        file_error = ""
    elif source_file.path is not None:
        text, file_error = _read_source_file(source_file.path)
    else:
        text, file_error = "", ""
    if file_error:
        result = ("", file_error)
    else:
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
        result = (
            (
                f"\n{source_file.label}: {source_file.relpath}\n"
                f"sha256: {digest}\n```\n{text.rstrip()}\n```\n"
            ),
            "",
        )
    if block_cache is not None:
        block_cache[cache_key] = result
    return result


def _test_context_metadata_block(source_file: _SourceFile) -> tuple[str, str]:
    if source_file.path is None:
        return "", ""
    try:
        stat = source_file.path.stat()
        digest = hashlib.sha256(source_file.path.read_bytes()).hexdigest()[:16]
    except OSError as error:
        return "", f"{source_file.path}: {error}"
    role = "fixture" if source_file.path.suffix.lower() in TEST_CONTEXT_FIXTURE_EXTENSIONS else "test"
    return (
        f"- {source_file.relpath} role={role} bytes={stat.st_size} sha256={digest}\n",
        "",
    )


def _append_with_budget(
    parts: list[str],
    used: int,
    text: str,
    *,
    max_chars: int,
    truncated: bool,
) -> tuple[int, bool]:
    if max_chars <= 0:
        return used, bool(text.strip()) or truncated
    if used + len(text) <= max_chars:
        parts.append(text)
        return used + len(text), truncated
    remaining = max_chars - used
    if remaining > 0:
        chunk = _truncate_source_context_chunk(text, remaining)
        parts.append(chunk)
        used += len(chunk)
    return used, True


def _truncate_source_context_chunk(text: str, max_chars: int) -> str:
    marker = "\n[truncated source context]\n"
    closing_fence = "\n```\n"
    if max_chars <= 0:
        return ""
    if max_chars <= len(marker):
        return text[:max_chars]

    chunk_budget = max_chars - len(marker)
    chunk = text[:chunk_budget].rstrip()
    if _has_open_markdown_fence(chunk):
        chunk_budget = max_chars - len(marker) - len(closing_fence)
        if chunk_budget > 0:
            chunk = text[:chunk_budget].rstrip()
            if _has_open_markdown_fence(chunk):
                return chunk + closing_fence + marker
            return chunk + marker
        return text[: max(0, max_chars - len(marker))].rstrip() + marker
    return chunk + marker


def _has_open_markdown_fence(text: str) -> bool:
    return text.count("```") % 2 == 1


def _format_source_metadata(mappings: Sequence[Mapping[str, Any]]) -> str:
    if not mappings:
        return ""
    blocks = ["Source metadata:\n"]
    keys = (
        "package",
        "required_version",
        "recipe_path",
        "recipe_version",
        "recipe_selection_reason",
        "source_confidence",
        "source_version_match",
        "recipe_commit",
        "source_url",
        "source_ref",
        "source_checkout",
        "source_is_binary_artifact",
        "source_artifact_url",
        "source_fallback_reason",
        "source_error",
        "fallback_from_channel",
        "fallback_from_recipe_selection_reason",
        "fallback_from_source_error",
        "source_provider_package",
        "source_provider_required_version",
        "source_provider_channel",
        "source_provider_recipe_package",
        "source_provider_recipe_path",
        "source_provider_source_url",
        "source_provider_source_ref",
        "source_provider_reason",
    )
    for index, mapping in enumerate(mappings, start=1):
        blocks.append(f"- source {index}:\n")
        for key in keys:
            value = str(mapping.get(key, "")).strip()
            if value:
                blocks.append(f"  {key}: {value}\n")
        hints = mapping.get("command_hints", [])
        if isinstance(hints, Sequence) and not isinstance(hints, (str, bytes)):
            hint_text = ", ".join(str(hint) for hint in hints if str(hint).strip())
            if hint_text:
                blocks.append(f"  command_hints: {hint_text}\n")
    return "".join(blocks).rstrip() + "\n"


def _keywords_from_mappings(mappings: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    keywords: set[str] = set()
    for mapping in mappings:
        for key in (
            "package",
            "required_version",
            "source_provider_package",
            "source_provider_recipe_package",
        ):
            keyword = _keyword(str(mapping.get(key, "")))
            if keyword:
                keywords.add(keyword)
        hints = mapping.get("command_hints", [])
        if isinstance(hints, Sequence) and not isinstance(hints, (str, bytes)):
            for hint in hints:
                keyword = _keyword(str(hint))
                if keyword:
                    keywords.add(keyword)
    return tuple(sorted(keywords))


def _keyword(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _manual_source_file(path: Path, *, keywords: Sequence[str]) -> _SourceFile | None:
    path = path.expanduser()
    if not path.exists() or not path.is_file() or not _is_readable_text_source(path):
        return None
    root = path.parent
    return _SourceFile(
        path=path,
        root=root,
        relpath=path.name,
        score=max(100, _score_path(path.name, keywords=keywords)),
    )


def _manual_test_context_file(
    path: Path,
    *,
    mode: str,
    max_file_bytes: int,
    keywords: Sequence[str],
) -> _SourceFile | None:
    path = path.expanduser()
    if not path.exists() or not path.is_file():
        return None
    if mode != TEST_CONTEXT_MODE_METADATA and not _is_readable_test_context_file(
        path,
        mode=mode,
        max_file_bytes=max_file_bytes,
    ):
        return None
    root = path.parent
    return _SourceFile(
        path=path,
        root=root,
        relpath=path.name,
        score=900 + _score_path(path.name, keywords=keywords) + _score_file_sample(path),
        label=_test_context_label(path.name),
        included_path=str(path),
    )


def _scan_source_root(
    root: Path,
    *,
    mode: str,
    keywords: Sequence[str],
) -> tuple[int, list[_SourceFile], list[str]]:
    root = root.expanduser()
    if root.is_file():
        root = root.parent
    if not root.exists() or not root.is_dir():
        return 0, [], [f"{root}: source root does not exist"]
    try:
        resolved_root = root.resolve()
    except OSError:
        resolved_root = root

    scanned = 0
    candidates: list[_SourceFile] = []
    errors: list[str] = []
    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(resolved_root)
        except ValueError:
            continue
        if _excluded_by_path(rel, mode=mode):
            continue
        scanned += 1
        if not _candidate_source_file(path, mode=mode):
            continue
        if not _is_readable_text_source(path):
            continue
        relpath = rel.as_posix()
        score = _score_path(relpath, keywords=keywords)
        score += _score_file_sample(path)
        candidates.append(_SourceFile(path=path, root=resolved_root, relpath=relpath, score=score))
    return scanned, candidates, errors


def _scan_test_context_root(
    root: Path,
    *,
    mode: str,
    max_file_bytes: int,
    keywords: Sequence[str],
) -> tuple[int, list[_SourceFile], list[str]]:
    root = root.expanduser()
    if root.is_file():
        root = root.parent
    if not root.exists() or not root.is_dir():
        return 0, [], [f"{root}: source root does not exist"]
    try:
        resolved_root = root.resolve()
    except OSError:
        resolved_root = root

    scanned = 0
    candidates: list[_SourceFile] = []
    errors: list[str] = []
    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            rel = path.relative_to(resolved_root)
        except ValueError:
            continue
        if _excluded_from_test_context(rel):
            continue
        if not _is_test_context_path(rel):
            continue
        scanned += 1
        if mode == TEST_CONTEXT_MODE_METADATA and not _is_candidate_test_context_file(
            path,
            mode=TEST_CONTEXT_MODE_FIXTURES,
        ):
            continue
        if mode != TEST_CONTEXT_MODE_METADATA and not _is_readable_test_context_file(
            path,
            mode=mode,
            max_file_bytes=max_file_bytes,
        ):
            continue
        relpath = rel.as_posix()
        score = 900 + _score_test_context_path(relpath, mode=mode, keywords=keywords)
        if mode != TEST_CONTEXT_MODE_METADATA:
            score += _score_file_sample(path)
        candidates.append(
            _SourceFile(
                path=path,
                root=resolved_root,
                relpath=relpath,
                score=score,
                label=_test_context_label(relpath),
            )
        )
    return scanned, candidates, errors


def _dedupe_source_files(files: Sequence[_SourceFile]) -> list[_SourceFile]:
    seen: set[str] = set()
    deduped: list[_SourceFile] = []
    for source_file in files:
        if source_file.inline_text is not None:
            digest = hashlib.sha256(
                source_file.inline_text.encode("utf-8", errors="replace")
            ).hexdigest()
            key = f"inline:{source_file.label}:{source_file.relpath}:{digest}"
        elif source_file.path is not None:
            try:
                key = str(source_file.path.resolve())
            except OSError:
                key = str(source_file.path)
        else:
            key = f"unknown:{source_file.label}:{source_file.relpath}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source_file)
    return deduped


def _dedupe_test_context_files(
    files: Sequence[_SourceFile],
    *,
    existing_files: Sequence[_SourceFile],
) -> list[_SourceFile]:
    existing_keys = {_source_file_cache_key(source_file) for source_file in existing_files}
    deduped: list[_SourceFile] = []
    seen: set[str] = set()
    for source_file in files:
        key = _source_file_cache_key(source_file)
        if key in existing_keys or key in seen:
            continue
        seen.add(key)
        deduped.append(source_file)
    return deduped


def _excluded_by_path(relpath: Path, *, mode: str) -> bool:
    parts = relpath.parts
    if any(part in {".git", ".hg", "__pycache__"} for part in parts):
        return True
    if mode == SOURCE_CONTEXT_MODE_ALL_RAW:
        return False
    rel_posix = relpath.as_posix()
    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    return any(rel_posix.startswith(excluded + "/") for excluded in EXCLUDED_DIRS)


def _excluded_from_test_context(relpath: Path) -> bool:
    parts = relpath.parts
    if any(part in {".git", ".hg", "__pycache__", ".pytest_cache", ".tox", ".venv"} for part in parts):
        return True
    return any(part in {"build", "dist", "node_modules", "site-packages", "venv"} for part in parts)


def _is_test_context_path(relpath: Path) -> bool:
    parts = {part.lower() for part in relpath.parts[:-1]}
    return bool(parts & TEST_CONTEXT_DIRS)


def _candidate_source_file(path: Path, *, mode: str) -> bool:
    suffix = path.suffix.lower()
    if suffix in BINARY_OR_DATA_EXTENSIONS:
        return False
    if mode == SOURCE_CONTEXT_MODE_ALL_RAW:
        return True
    return suffix in SOURCE_EXTENSIONS or path.name in SOURCE_BASENAMES


def _is_candidate_test_context_file(path: Path, *, mode: str) -> bool:
    suffix = path.suffix.lower()
    if suffix in TEST_CONTEXT_SCRIPT_EXTENSIONS or path.name in SOURCE_BASENAMES:
        return True
    return mode == TEST_CONTEXT_MODE_FIXTURES and suffix in TEST_CONTEXT_FIXTURE_EXTENSIONS


def _is_readable_test_context_file(path: Path, *, mode: str, max_file_bytes: int) -> bool:
    if not _is_candidate_test_context_file(path, mode=mode):
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    if max_file_bytes > 0 and stat.st_size > max_file_bytes:
        return False
    return _looks_like_text_file(path)


def _is_readable_text_source(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size > MAX_SOURCE_FILE_BYTES:
        return False
    return _looks_like_text_file(path)


def _looks_like_text_file(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:FILE_SAMPLE_BYTES]
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    if not chunk:
        return True
    control_count = sum(1 for byte in chunk if byte < 32 and byte not in b"\n\r\t\f\b")
    return control_count / len(chunk) < 0.05


def _read_source_file(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return "", f"{path}: {error}"
    if len(text) > MAX_SOURCE_FILE_BYTES:
        text = text[:MAX_SOURCE_FILE_BYTES].rstrip() + "\n[file truncated]\n"
    return text, ""


def _score_path(relpath: str, *, keywords: Sequence[str]) -> int:
    rel_lower = relpath.lower()
    basename = Path(relpath).name.lower()
    score = 0
    if rel_lower.startswith(".github/workflows/") or "/.github/workflows/" in f"/{rel_lower}":
        score -= 120
    if (
        "/bin/" in f"/{rel_lower}"
        or "/scripts/" in f"/{rel_lower}"
        or "/script/" in f"/{rel_lower}"
    ):
        score += 80
    if basename.startswith("readme"):
        score += 55
    if basename in {"setup.py", "setup.cfg", "pyproject.toml"}:
        score += 70
    if basename in {
        "cli.py",
        "commands.py",
        "command.py",
        "__main__.py",
        "main.py",
        "main.c",
        "main.cc",
        "main.cpp",
    } or basename.endswith(("-main.c", "-main.cc", "-main.cpp")):
        score += 60
    normalized_path = _keyword(relpath)
    for keyword in keywords:
        if keyword and keyword in normalized_path:
            score += 50
    return score


def _test_context_label(relpath: str) -> str:
    suffix = Path(relpath).suffix.lower()
    if suffix in TEST_CONTEXT_FIXTURE_EXTENSIONS:
        return "Source test/example fixture"
    return "Source test/example file"


def _score_test_context_path(relpath: str, *, mode: str, keywords: Sequence[str]) -> int:
    rel_lower = relpath.lower()
    basename = Path(relpath).name.lower()
    score = _score_path(relpath, keywords=keywords)
    if "/test" in f"/{rel_lower}" or "/example" in f"/{rel_lower}" or "/demo" in f"/{rel_lower}":
        score += 45
    if basename.startswith(("test_", "test-", "example", "expected")):
        score += 35
    suffix = Path(relpath).suffix.lower()
    if suffix in TEST_CONTEXT_FIXTURE_EXTENSIONS:
        score += 10 if mode == TEST_CONTEXT_MODE_FIXTURES else -25
    return score


def _score_file_sample(path: Path) -> int:
    try:
        sample = path.read_text(encoding="utf-8", errors="replace")[:FILE_SAMPLE_BYTES]
    except OSError:
        return 0
    score = 0
    for pattern in CLI_PATTERNS:
        if pattern in sample:
            score += 35
    lowered = sample.lower()
    if "usage:" in lowered or "getting started" in lowered:
        score += 30
    for marker in ("getopt", "ketopt", "argv", "argc", "subcommand"):
        if marker in lowered:
            score += 20
    return score
