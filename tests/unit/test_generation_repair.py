from __future__ import annotations

from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference import generation as generation_mod
from galaxy_toolsmith.inference.generation import (
    generate_wrapper,
    generate_wrapper_from_content,
    generate_xml_from_content,
)
from galaxy_toolsmith.inference.source_context import (
    SOURCE_CONTEXT_MODE_ALL_RAW,
    SourceContextSettings,
)
from galaxy_toolsmith.providers.base import (
    GenerationInput,
    GenerationOutput,
    extract_complete_tool_xml_candidates,
)


class _RepairProvider:
    name = "repair-provider"

    def __init__(self) -> None:
        self.requests: list[GenerationInput] = []

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        self.requests.append(request)
        if len(self.requests) == 1:
            return GenerationOutput(
                artifact_text="<tables><table name='refs'/></tables>",
                artifact_format=request.artifact_format,
                provider=self.name,
                model_variant=request.model_variant,
                raw_response_text="<tables><table name='refs'/></tables>",
            )
        assert "Generate the wrapper <tool> first" in request.repair_context
        return GenerationOutput(
            artifact_text="<tool id='fixed' name='Fixed' version='0.1.0'><command>fixed</command></tool>",
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
            raw_response_text="<tool id='fixed' name='Fixed' version='0.1.0'><command>fixed</command></tool>",
        )


def test_generate_wrapper_repairs_sidecar_as_primary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    help_text = tmp_path / "help.txt"
    help_text.write_text("Usage: fixed --input reads.fq\n", encoding="utf-8")
    output = tmp_path / "fixed.xml"
    raw_log = tmp_path / "fixed.raw.log"
    provider = _RepairProvider()
    monkeypatch.setattr(generation_mod, "_get_provider", lambda *args, **kwargs: provider)

    record = generate_wrapper(
        paths=paths,
        tool_name="fixed",
        help_text_path=help_text,
        source_path=None,
        output_path=output,
        provider_name="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=128,
        raw_response_log_path=raw_log,
    )

    assert len(provider.requests) == 2
    assert record.repair_attempted is True
    assert record.attempt_count == 2
    assert record.initial_validation["root_tag"] == "tables"
    assert record.validation["root_is_tool"] is True
    assert record.selected_candidate_attempt == 2
    assert record.selected_candidate_index == 1
    assert len(record.candidate_artifacts) == 2
    assert record.candidate_artifacts[0]["attempt"] == 1
    assert record.candidate_artifacts[0]["selected"] is False
    assert record.candidate_artifacts[1]["attempt"] == 2
    assert record.candidate_artifacts[1]["selected"] is True
    assert output.read_text(encoding="utf-8").startswith("<tool")
    assert raw_log.read_text(encoding="utf-8").startswith("<tables")
    assert (tmp_path / "fixed.raw.attempt-2.log").read_text(encoding="utf-8").startswith("<tool")
    assert (tmp_path / ".gtsm" / "candidates" / "fixed" / "candidate-1.xml").exists()
    assert (
        tmp_path / ".gtsm" / "candidates" / "fixed" / "candidate-attempt-2-1.xml"
    ).exists()


def test_extract_complete_tool_xml_candidates_returns_all_tool_blocks() -> None:
    raw = "\n".join(
        [
            "```xml",
            "<tool id='one'><command>one</command></tool>",
            "extra text",
            "<tool id='two'><command>two</command></tool>",
            "```",
        ]
    )

    candidates = extract_complete_tool_xml_candidates(raw)

    assert len(candidates) == 2
    assert "id='one'" in candidates[0]
    assert "id='two'" in candidates[1]


class _MultiCandidateProvider:
    name = "multi-candidate-provider"

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        raw = "\n".join(
            [
                "<tool id='weak' name='Weak' version='0.1.0'><help>placeholder</help></tool>",
                (
                    "<tool id='strong' name='Strong' version='0.1.0'>"
                    "<requirements><requirement type='package' version='0.3'>minibwa</requirement></requirements>"
                    "<command><![CDATA[minibwa map $index $reads > $alignment]]></command>"
                    "<inputs><param name='index' type='data' format='mbw'/><param name='reads' type='data' format='fastq'/></inputs>"
                    "<outputs><data name='alignment' format='bam'/></outputs>"
                    "<tests><test><param name='reads' value='reads.fastq'/><output name='alignment'/></test></tests>"
                    "<help>Map reads.</help>"
                    "</tool>"
                ),
            ]
        )
        return GenerationOutput(
            artifact_text="<tool id='weak' name='Weak' version='0.1.0'><help>placeholder</help></tool>",
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
            raw_response_text=raw,
        )


class _RepetitionCandidateProvider:
    name = "repetition-candidate-provider"

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        repeated_assertions = "\n".join("<has_text text='100.00'/>" for _ in range(14))
        raw = "\n".join(
            [
                (
                    "<tool id='rich' name='Rich' version='0.1.0'>"
                    "<requirements><requirement type='package'>minibwa</requirement></requirements>"
                    "<command><![CDATA[minibwa map $reads > $alignment]]></command>"
                    "<inputs><param name='reads' type='data' format='fastq'/></inputs>"
                    "<outputs><data name='alignment' format='bam'><assert_contents>"
                    f"{repeated_assertions}"
                    "</assert_contents></data></outputs>"
                    "<tests><test><param name='reads' value='reads.fastq'/></test></tests>"
                    "<help>Map reads.</help>"
                    "</tool>"
                ),
                (
                    "<tool id='compact' name='Compact' version='0.1.0'>"
                    "<command><![CDATA[minibwa map $reads > $alignment]]></command>"
                    "<inputs><param name='reads' type='data' format='fastq'/></inputs>"
                    "<outputs><data name='alignment' format='bam'/></outputs>"
                    "</tool>"
                ),
            ]
        )
        return GenerationOutput(
            artifact_text="<tool id='rich' name='Rich'><command>bad</command></tool>",
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
            raw_response_text=raw,
        )


def test_generate_wrapper_scores_and_writes_alternate_tool_candidates(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    output = tmp_path / "minibwa_map.xml"

    record = generate_wrapper_from_content(
        paths=paths,
        tool_name="minibwa_map",
        help_text="Usage: minibwa map <index> <reads>",
        source_code="",
        output_path=output,
        provider_name="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=256,
        provider_instance=_MultiCandidateProvider(),
        tool_id="minibwa_map",
        tool_display_name="minibwa map",
        include_toolsmith_citation=False,
    )

    saved = output.read_text(encoding="utf-8")
    assert "minibwa map $index $reads" in saved
    assert record.selected_candidate_attempt == 1
    assert record.selected_candidate_index == 2
    assert len(record.candidate_artifacts) == 2
    assert record.candidate_artifacts[0]["selected"] is False
    assert record.candidate_artifacts[1]["selected"] is True
    assert record.candidate_artifacts[1]["score"] > record.candidate_artifacts[0]["score"]
    assert Path(record.candidate_artifacts[0]["path"]).exists()
    assert Path(record.candidate_artifacts[1]["path"]).exists()
    assert (
        tmp_path / ".gtsm" / "candidates" / "minibwa_map" / "manifest.json"
    ).exists()


def test_generate_wrapper_prefers_valid_candidate_over_repetitive_candidate(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    output = tmp_path / "minibwa_map.xml"

    record = generate_wrapper_from_content(
        paths=paths,
        tool_name="minibwa_map",
        help_text="Usage: minibwa map <reads>",
        source_code="",
        output_path=output,
        provider_name="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=256,
        provider_instance=_RepetitionCandidateProvider(),
        tool_id="minibwa_map",
        tool_display_name="minibwa map",
        include_toolsmith_citation=False,
    )

    assert "id=\"minibwa_map\"" in output.read_text(encoding="utf-8")
    assert "minibwa map $reads" in output.read_text(encoding="utf-8")
    assert record.selected_candidate_index == 2
    assert record.candidate_artifacts[0]["validation"]["generation_diagnostics"]["has_problems"] is True
    assert record.candidate_artifacts[1]["selected"] is True


def test_generate_xml_from_content_scores_and_returns_alternate_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    provider = _MultiCandidateProvider()
    monkeypatch.setattr(generation_mod, "_get_provider", lambda *args, **kwargs: provider)

    result = generate_xml_from_content(
        tool_name="minibwa_map",
        help_text="Usage: minibwa map <index> <reads>",
        source_code="",
        provider_name="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=256,
        paths=paths,
        tool_id="minibwa_map",
        tool_display_name="minibwa map",
        include_toolsmith_citation=False,
    )

    assert "minibwa map $index $reads" in result["artifact_text"]
    assert result["selected_candidate_attempt"] == 1
    assert result["selected_candidate_index"] == 2
    assert len(result["candidate_artifacts"]) == 2
    assert result["candidate_artifacts"][0]["selected"] is False
    assert result["candidate_artifacts"][1]["selected"] is True
    assert "artifact_text" in result["candidate_artifacts"][1]


class _RepetitionRepairProvider:
    name = "repetition-repair-provider"

    def __init__(self) -> None:
        self.requests: list[GenerationInput] = []

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        self.requests.append(request)
        if len(self.requests) == 1:
            assert request.source_code
            repeated_assertions = "\n".join("<has_text text='100.00'/>" for _ in range(14))
            raw = (
                "<tool id='map' name='Map' version='0.1.0'>"
                "<command><![CDATA[minibwa map $reads > $alignment]]></command>"
                "<inputs><param name='reads' type='data' format='fastq'/></inputs>"
                "<outputs><data name='alignment' format='bam'><assert_contents>"
                f"{repeated_assertions}"
                "</assert_contents></data></outputs>"
                "<tests><test><param name='reads' value='reads.fastq'/></test></tests>"
                "</tool>"
            )
            return GenerationOutput(
                artifact_text=raw,
                artifact_format=request.artifact_format,
                provider=self.name,
                model_variant=request.model_variant,
                raw_response_text=raw,
            )
        assert request.source_code == ""
        assert "Regenerate a compact wrapper" in request.repair_context
        assert "Include exactly one minimal <test> element." in request.repair_context
        raw = (
            "<tool id='map' name='Map' version='0.1.0'>"
            "<command><![CDATA[minibwa map $reads > $alignment]]></command>"
            "<inputs><param name='reads' type='data' format='fastq'/></inputs>"
            "<outputs><data name='alignment' format='bam'/></outputs>"
            "</tool>"
        )
        return GenerationOutput(
            artifact_text=raw,
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
            raw_response_text=raw,
        )


def test_generate_wrapper_repetition_repair_omits_source_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    help_text = tmp_path / "help.txt"
    help_text.write_text("Usage: minibwa map <reads>\n", encoding="utf-8")
    source = tmp_path / "minibwa.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    output = tmp_path / "minibwa_map.xml"
    provider = _RepetitionRepairProvider()
    monkeypatch.setattr(generation_mod, "_get_provider", lambda *args, **kwargs: provider)

    record = generate_wrapper(
        paths=paths,
        tool_name="minibwa_map",
        help_text_path=help_text,
        source_path=source,
        output_path=output,
        provider_name="local",
        model_variant="variant-a",
        model="",
        temperature=0.0,
        max_tokens=256,
        source_context_settings=SourceContextSettings(mode=SOURCE_CONTEXT_MODE_ALL_RAW),
        include_toolsmith_citation=False,
    )

    assert len(provider.requests) == 2
    assert record.repair_attempted is True
    assert record.repair_mode == "repetition_compaction"
    assert record.validation["root_is_tool"] is True
