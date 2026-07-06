from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from xml.etree import ElementTree as ET

_OPTION_TAG_RE = re.compile(r"<option\b", re.IGNORECASE)
_OPTION_VALUE_RE = re.compile(r"<option\b[^>]*\bvalue=(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)
_OPTION_TEXT_RE = re.compile(r"<option\b[^>]*>(.*?)</option>", re.IGNORECASE | re.DOTALL)
_TAG_END_RE = re.compile(r"<[^>]*\Z")
_TAG_RE = re.compile(r"<[^>]+>")
_OUTPUT_VARIABLE_RE = re.compile(r"(?:#set\s+)?\$?(out_[A-Za-z0-9_]+)")
_TEST_TAG_RE = re.compile(r"<test\b", re.IGNORECASE)

MAX_OPTION_COUNT = 60
MAX_OPTION_VALUE_CHARS = 96
MAX_OPTION_LABEL_CHARS = 140
REPEATED_PREFIX_MIN_COUNT = 5
REPEATED_PREFIX_CHARS = 24
REPEATED_XML_LINE_MIN_COUNT = 12
REPEATED_CHEETAH_FRAGMENT_MIN_COUNT = 8
MAX_GENERATED_TEST_COUNT = 1


@dataclass(frozen=True)
class GeneratedXmlDiagnostics:
    has_problems: bool
    problems: list[str] = field(default_factory=list)
    option_count: int = 0
    long_option_values: int = 0
    long_option_labels: int = 0
    repeated_option_prefixes: list[str] = field(default_factory=list)
    repeated_xml_lines: list[str] = field(default_factory=list)
    repeated_xml_line_count: int = 0
    repeated_cheetah_fragments: int = 0
    repeated_cheetah_fragment_details: list[dict[str, object]] = field(default_factory=list)
    test_count: int = 0
    too_many_tests: bool = False
    missing_closing_tool: bool = False
    ends_mid_tag: bool = False
    unclosed_cdata: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_generated_xml(xml_wrapper: str) -> GeneratedXmlDiagnostics:
    value = str(xml_wrapper or "").strip()
    problems: list[str] = []

    missing_closing_tool = "</tool>" not in value
    if missing_closing_tool:
        problems.append("Generated XML does not contain a closing </tool> tag.")

    ends_mid_tag = bool(_TAG_END_RE.search(value)) and not value.endswith(">")
    if ends_mid_tag:
        problems.append("Generated XML appears to end mid-tag, likely due to output truncation.")

    unclosed_cdata = value.count("<![CDATA[") > value.count("]]>")
    if unclosed_cdata:
        problems.append("Generated XML contains an unclosed CDATA section.")

    option_count = len(_OPTION_TAG_RE.findall(value))
    if option_count > MAX_OPTION_COUNT:
        problems.append(
            f"Generated XML contains {option_count} <option> tags, which suggests an overlong or hallucinated select list."
        )

    option_values = [match.group(2).strip() for match in _OPTION_VALUE_RE.finditer(value)]
    long_option_values = sum(1 for option_value in option_values if len(option_value) > MAX_OPTION_VALUE_CHARS)
    if long_option_values:
        problems.append(
            f"Generated XML contains {long_option_values} unusually long <option> value attributes."
        )

    option_labels = [_strip_tags(match.group(1)).strip() for match in _OPTION_TEXT_RE.finditer(value)]
    long_option_labels = sum(1 for label in option_labels if len(label) > MAX_OPTION_LABEL_CHARS)
    if long_option_labels:
        problems.append(f"Generated XML contains {long_option_labels} unusually long <option> labels.")

    repeated_prefixes = _repeated_option_prefixes(option_values)
    if repeated_prefixes:
        joined = ", ".join(repeated_prefixes[:3])
        problems.append(
            f"Generated XML contains repeated synthetic-looking <option> prefixes: {joined}."
        )

    repeated_xml_lines = _repeated_xml_lines(value)
    repeated_xml_line_count = len(repeated_xml_lines)
    if repeated_xml_lines:
        problems.append(
            f"Generated XML contains repeated identical XML lines: {repeated_xml_lines[0]!r}."
        )

    repeated_cheetah_fragments = _repeated_cheetah_fragment_count(value)
    repeated_cheetah_fragment_details = _repeated_cheetah_fragment_details(value)
    if repeated_cheetah_fragments >= REPEATED_CHEETAH_FRAGMENT_MIN_COUNT:
        problems.append(
            f"Generated XML contains {repeated_cheetah_fragments} repeated Cheetah output-variable fragments."
        )

    test_count = _direct_test_count(value)
    too_many_tests = test_count > MAX_GENERATED_TEST_COUNT
    if too_many_tests:
        problems.append(
            f"Generated XML contains {test_count} direct <test> elements; generated wrappers should include one compact test."
        )

    return GeneratedXmlDiagnostics(
        has_problems=bool(problems),
        problems=problems,
        option_count=option_count,
        long_option_values=long_option_values,
        long_option_labels=long_option_labels,
        repeated_option_prefixes=repeated_prefixes,
        repeated_xml_lines=repeated_xml_lines,
        repeated_xml_line_count=repeated_xml_line_count,
        repeated_cheetah_fragments=repeated_cheetah_fragments,
        repeated_cheetah_fragment_details=repeated_cheetah_fragment_details,
        test_count=test_count,
        too_many_tests=too_many_tests,
        missing_closing_tool=missing_closing_tool,
        ends_mid_tag=ends_mid_tag,
        unclosed_cdata=unclosed_cdata,
    )


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


def _repeated_option_prefixes(option_values: list[str]) -> list[str]:
    prefixes: list[str] = []
    for option_value in option_values:
        compact = re.sub(r"[^A-Za-z0-9]+", "", option_value).lower()
        if len(compact) >= REPEATED_PREFIX_CHARS:
            prefixes.append(compact[:REPEATED_PREFIX_CHARS])
    counts = Counter(prefixes)
    return sorted(
        prefix
        for prefix, count in counts.items()
        if count >= REPEATED_PREFIX_MIN_COUNT
    )


def _repeated_xml_lines(xml_wrapper: str) -> list[str]:
    counts = Counter(
        line.strip()
        for line in xml_wrapper.splitlines()
        if line.strip().startswith("<") and len(line.strip()) >= 12
    )
    return sorted(
        line
        for line, count in counts.items()
        if count >= REPEATED_XML_LINE_MIN_COUNT
    )


def _repeated_cheetah_fragment_details(xml_wrapper: str) -> list[dict[str, object]]:
    fragments = [_normalize_output_variable(match.group(1)) for match in _OUTPUT_VARIABLE_RE.finditer(xml_wrapper)]
    counts = Counter(fragments)
    return [
        {"fragment": fragment, "count": count}
        for fragment, count in sorted(counts.items())
        if count >= REPEATED_CHEETAH_FRAGMENT_MIN_COUNT
    ]


def _repeated_cheetah_fragment_count(xml_wrapper: str) -> int:
    return sum(int(item["count"]) for item in _repeated_cheetah_fragment_details(xml_wrapper))


def _normalize_output_variable(value: str) -> str:
    return re.sub(r"_\d+\Z", "", value.strip())


def _direct_test_count(xml_wrapper: str) -> int:
    try:
        root = ET.fromstring(xml_wrapper)
    except ET.ParseError:
        return len(_TEST_TAG_RE.findall(xml_wrapper))
    tests = root.find("tests")
    if tests is None:
        return 0
    return sum(1 for child in list(tests) if child.tag == "test")
