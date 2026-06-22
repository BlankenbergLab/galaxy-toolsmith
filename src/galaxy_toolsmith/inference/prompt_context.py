from __future__ import annotations

import re
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict, dataclass

DEFAULT_MAX_PROMPT_HELP_CHARS = 12000
_MAX_LINE_COPIES = 2
_MAX_HINT_LINE_CHARS = 180
_MAX_USAGE_HINTS = 3
_MAX_OPTION_HINTS = 24
_MAX_REQUIRED_HINTS = 10
_MAX_OUTPUT_HINTS = 10
_MAX_DATATYPE_HINTS = 12

_COMMON_DATATYPE_TERMS = (
    "fastq.gz",
    "fasta.gz",
    "fasta.bz2",
    "genbank.gz",
    "fastq",
    "fasta",
    "genbank",
    "gbk",
    "embl",
    "bam",
    "sam",
    "vcf",
    "bed",
    "gff",
    "gtf",
    "tabular",
    "tsv",
    "csv",
    "txt",
    "json",
    "xml",
    "html",
    "directory",
)


@dataclass(frozen=True)
class PromptHelpText:
    text: str
    original_chars: int
    shaped_chars: int
    max_chars: int
    truncated: bool
    omitted_repeated_lines: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InterfaceHints:
    text: str
    usage_lines: list[str]
    metadata_lines: list[str]
    option_lines: list[str]
    required_lines: list[str]
    output_lines: list[str]
    datatype_terms: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def shape_help_text(
    help_text: str,
    *,
    max_chars: int = DEFAULT_MAX_PROMPT_HELP_CHARS,
) -> PromptHelpText:
    """Keep prompt help concise while preserving the most useful CLI facts."""
    normalized = str(help_text or "").replace("\r\n", "\n").replace("\r", "\n")
    original_chars = len(normalized)
    max_chars = max(0, int(max_chars))

    collapsed, omitted_repeated_lines = _collapse_repeated_lines(normalized)
    if len(collapsed) <= max_chars:
        return PromptHelpText(
            text=collapsed,
            original_chars=original_chars,
            shaped_chars=len(collapsed),
            max_chars=max_chars,
            truncated=False,
            omitted_repeated_lines=omitted_repeated_lines,
        )

    note = (
        "\n\n[Toolsmith note: help text was shortened for prompting; "
        "use the preserved usage/options/examples as source of truth.]"
    )
    budget = max(0, max_chars - len(note))
    shaped = _select_help_prefix_and_key_lines(collapsed.splitlines(), budget).strip()
    if note and max_chars >= len(note):
        shaped = f"{shaped}{note}" if shaped else note.strip()
    if len(shaped) > max_chars:
        shaped = shaped[:max_chars].rstrip()

    return PromptHelpText(
        text=shaped,
        original_chars=original_chars,
        shaped_chars=len(shaped),
        max_chars=max_chars,
        truncated=True,
        omitted_repeated_lines=omitted_repeated_lines,
    )


def extract_interface_hints(
    help_text: str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> InterfaceHints:
    """Extract a compact interface outline from raw CLI help for prompt grounding."""
    normalized = str(help_text or "").replace("\r\n", "\n").replace("\r", "\n")
    collapsed, _omitted = _collapse_repeated_lines(normalized)
    lines = [_truncate_line(line.strip()) for line in collapsed.splitlines() if line.strip()]

    metadata_lines = _metadata_hint_lines(metadata)
    usage_lines = _limited_unique(
        (line for line in lines if _is_usage_line(line)),
        _MAX_USAGE_HINTS,
    )
    option_lines = _limited_unique(
        (line for line in lines if _looks_like_option_line(line)),
        _MAX_OPTION_HINTS,
    )
    required_lines = _limited_unique(
        (line for line in lines if _is_required_or_input_line(line)),
        _MAX_REQUIRED_HINTS,
    )
    output_lines = _limited_unique(
        (line for line in lines if _is_output_line(line)),
        _MAX_OUTPUT_HINTS,
    )
    datatype_terms = _extract_datatype_terms(collapsed)

    text_parts: list[str] = []
    _append_hint_section(text_parts, "Metadata cues", metadata_lines)
    _append_hint_section(text_parts, "Usage", usage_lines)
    _append_hint_section(text_parts, "Core options", option_lines)
    _append_hint_section(text_parts, "Required/input cues", required_lines)
    _append_hint_section(text_parts, "Output cues", output_lines)
    if datatype_terms:
        text_parts.append("Datatype cues:")
        text_parts.append("- " + ", ".join(datatype_terms))

    return InterfaceHints(
        text="\n".join(text_parts).strip(),
        usage_lines=usage_lines,
        metadata_lines=metadata_lines,
        option_lines=option_lines,
        required_lines=required_lines,
        output_lines=output_lines,
        datatype_terms=datatype_terms,
    )


def _collapse_repeated_lines(text: str) -> tuple[str, int]:
    seen: dict[str, int] = {}
    kept: list[str] = []
    omitted = 0
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        key = line.strip()
        if key:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > _MAX_LINE_COPIES:
                omitted += 1
                continue
        kept.append(line)
    return "\n".join(kept).strip(), omitted


def _select_help_prefix_and_key_lines(lines: list[str], budget: int) -> str:
    if budget <= 0:
        return ""
    selected: list[str] = []
    used = 0

    def add(line: str) -> bool:
        nonlocal used
        needed = len(line) + (1 if selected else 0)
        if used + needed > budget:
            return False
        selected.append(line)
        used += needed
        return True

    prefix_budget = min(3000, max(600, budget // 3))
    for line in lines:
        if used + len(line) + (1 if selected else 0) > prefix_budget:
            break
        add(line)

    seen = set(selected)
    for line in lines:
        if line in seen:
            continue
        if _is_key_help_line(line):
            if not add(line):
                break
            seen.add(line)

    for line in lines:
        if line in seen:
            continue
        if not add(line):
            break
        seen.add(line)

    return "\n".join(selected)


def _truncate_line(line: str) -> str:
    value = line.strip()
    if len(value) <= _MAX_HINT_LINE_CHARS:
        return value
    return value[: _MAX_HINT_LINE_CHARS - 3].rstrip() + "..."


def _limited_unique(lines: Iterable[str], limit: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for line in lines:
        value = str(line).strip()
        key = value.lower()
        if not value or key in seen:
            continue
        selected.append(value)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def _append_hint_section(parts: list[str], title: str, lines: list[str]) -> None:
    if not lines:
        return
    parts.append(f"{title}:")
    parts.extend(f"- {line}" for line in lines)


def _metadata_hint_lines(metadata: Mapping[str, object] | None) -> list[str]:
    if not metadata:
        return []
    allowed = (
        ("package_id", "Package"),
        ("tool_id", "Tool id"),
        ("primary_command", "Primary command"),
    )
    lines: list[str] = []
    for key, label in allowed:
        value = str(metadata.get(key, "")).strip()
        if value:
            lines.append(f"{label}: {_truncate_line(value)}")
    return lines


def _is_usage_line(line: str) -> bool:
    value = line.strip().lower()
    return value.startswith(("usage", "synopsis"))


def _looks_like_option_line(line: str) -> bool:
    value = line.strip().lower()
    if not value:
        return False
    if _is_usage_line(value):
        return False
    if value.startswith(("-", "--")):
        return True
    if " --" in value or "\t--" in value:
        return True
    return bool(re.search(r"(^|\s)-[A-Za-z](,|\s|$)", line))


def _is_required_or_input_line(line: str) -> bool:
    value = line.strip().lower()
    if _is_usage_line(value) or _looks_like_option_line(value):
        return False
    return any(
        marker in value
        for marker in (
            "required",
            "mandatory",
            "input",
            "file",
            "read",
            "assembly",
            "contig",
            "sample",
        )
    )


def _is_output_line(line: str) -> bool:
    value = line.strip().lower()
    if _is_usage_line(value) or _looks_like_option_line(value):
        return False
    return any(
        marker in value
        for marker in (
            "output",
            "result",
            "write",
            "written",
            "prefix",
            "directory",
            "report",
        )
    )


def _extract_datatype_terms(text: str) -> list[str]:
    found: list[str] = []
    for term in _COMMON_DATATYPE_TERMS:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(term)
        if len(found) >= _MAX_DATATYPE_HINTS:
            break
    return found


def _is_key_help_line(line: str) -> bool:
    value = line.strip().lower()
    if not value:
        return False
    if value.startswith(
        (
            "usage",
            "options",
            "optional",
            "required",
            "arguments",
            "commands",
            "subcommands",
            "examples",
            "example",
        )
    ):
        return True
    if value.startswith(("-", "--")):
        return True
    return " --" in value or "\t--" in value
