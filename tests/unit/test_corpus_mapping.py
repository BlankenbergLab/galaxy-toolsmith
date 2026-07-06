from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from galaxy_toolsmith.data.corpus import (
    _CONFIGFILE_TRUNCATION_MARKER,
    ContainerPreparation,
    ContainerRuntime,
    ExtractionSettings,
    ToolRecord,
    extract_tools_corpus,
    rebuild_execution_report_from_jsonl,
    write_corpus_diagnostics,
)
from galaxy_toolsmith.inference.udt import validate_udt_yaml


def test_extract_corpus_includes_shed_suite_and_mapping_fields(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "samtools_suite"
    tool_dir.mkdir(parents=True, exist_ok=True)

    (tool_dir / ".shed.yml").write_text(
        """
name: suite_samtools
owner: iuc
description: Samtools suite
homepage_url: https://github.com/galaxyproject/tools-iuc
remote_repository_url: https://github.com/galaxyproject/tools-iuc
categories:
  - Sequence Analysis
type: suite_repository
repositories:
  - name: samtools_view
  - name: samtools_sort
""".strip(),
        encoding="utf-8",
    )
    (tool_dir / "macros.xml").write_text(
        """
<macros>
  <xml name="insert_help">
    <help><![CDATA[Macro help text]]></help>
  </xml>
</macros>
""".strip(),
        encoding="utf-8",
    )
    (tool_dir / "tool_data_table_conf.xml").write_text(
        """
<tables>
  <table name="samtools_ref" comment_char="#">
    <columns>value, label, path</columns>
    <file path="tool-data/samtools_ref.loc" />
  </table>
</tables>
""".strip(),
        encoding="utf-8",
    )
    (tool_dir / "tool-data").mkdir()
    (tool_dir / "tool-data" / "samtools_ref.loc.sample").write_text(
        "hg38\tHuman hg38\t/path/to/hg38.fa\n",
        encoding="utf-8",
    )
    (tool_dir / "samtools_view.xml").write_text(
        """
<tool id="samtools_view" name="samtools_view" version="1.0.0">
  <command><![CDATA[samtools view -h '$input' > '$output']]></command>
  <expand macro="insert_help"/>
  <inputs>
    <param name="input" type="data" format="bam"/>
  </inputs>
  <outputs>
    <data name="output" format="sam"/>
  </outputs>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    result = extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    assert result["total_tools"] == 1
    assert result["total_wrappers"] == 1
    assert result["processed_now"] == 1

    records = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    rec = records[0]
    assert rec["tool_id"] == "samtools_view"
    assert rec["suite_id"] == "iuc/suite_samtools"
    assert rec["is_suite_root"] is True
    assert "samtools_view" in rec["suite_members"]
    assert rec["primary_command"] == "samtools"
    assert "view" in rec["subcommands"]
    assert rec["macro_expansion_status"] in {"expanded", "partial", "not_applicable"}
    assert Path(rec["expanded_xml_path"]).exists()
    sidecars = {item["relative_path"]: item for item in rec["wrapper_sidecar_files"]}
    assert sidecars["macros.xml"]["role"] == "macros"
    assert sidecars["macros.xml"]["root_tag"] == "macros"
    assert sidecars["tool_data_table_conf.xml"]["role"] == "tool_data_table_conf"
    assert sidecars["tool_data_table_conf.xml"]["root_tag"] == "tables"
    assert sidecars["tool-data/samtools_ref.loc.sample"]["role"] == "tool_data_loc_sample"
    assert rec["wrapper_source_summary"]["sidecar_file_count"] == 3

    index_path = Path(result["index_path"])
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert "iuc/suite_samtools" in index_payload["suites"]


def test_extract_corpus_uses_galaxy_macro_expansion_for_requirements(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "ampvis2"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: ampvis2\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "macros.xml").write_text(
        """
<macros>
  <token name="@TOOL_VERSION@">2.8.11</token>
  <xml name="header">
    <requirements>
      <requirement type="package" version="@TOOL_VERSION@">r-ampvis2</requirement>
      <requirement type="package" version="2.1.5">r-readr</requirement>
    </requirements>
  </xml>
  <xml name="inputs">
    <inputs>
      <param name="data" type="data" format="ampvis2"/>
    </inputs>
  </xml>
</macros>
""".strip(),
        encoding="utf-8",
    )
    (tool_dir / "frequency.xml").write_text(
        """
<tool id="ampvis2_frequency" name="ampvis2 frequency plot" version="@TOOL_VERSION@">
  <macros>
    <import>macros.xml</import>
  </macros>
  <expand macro="header"/>
  <command><![CDATA[Rscript '$rscript']]></command>
  <expand macro="inputs"/>
  <outputs>
    <data name="plot" format="pdf"/>
  </outputs>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    result = extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    records = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rec = records[0]
    expanded_xml = Path(rec["expanded_xml_path"]).read_text(encoding="utf-8")

    assert result["processed_now"] == 1
    assert rec["macro_expansion_status"] == "expanded"
    assert rec["requirement_packages"] == ["r-ampvis2", "r-readr"]
    assert rec["requirement_versions"] == {"r-ampvis2": "2.8.11", "r-readr": "2.1.5"}
    assert "r-ampvis2" in expanded_xml
    assert 'macro="header"' not in expanded_xml


def test_extract_corpus_synthesizes_schema_valid_udt_yaml(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "samtools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: samtools\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "view.xml").write_text(
        """
<tool id="samtools_view" name="samtools view" version="1.20">
  <requirements>
    <container type="docker">quay.io/biocontainers/samtools:1.20--h50ea8bc_1</container>
  </requirements>
  <command><![CDATA[samtools view '$input' > '$output']]></command>
  <configfiles>
    <configfile name="settings" filename="settings.json"><![CDATA[
{"threads": "$threads"}
    ]]></configfile>
  </configfiles>
  <inputs>
    <param name="input" type="data" format="bam" label="Input BAM"/>
    <param name="threads" type="integer" value="1" min="1"/>
  </inputs>
  <outputs>
    <data name="output" format="sam"/>
  </outputs>
  <help><![CDATA[Convert BAM to SAM.]]></help>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(
            max_workers=1,
            retries=1,
            fetch_documentation=False,
            synthesize_udt_yaml=True,
        ),
    )

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    udt_path = Path(record["udt_yaml_path"])
    assert udt_path.exists()
    assert record["udt_yaml_files"] == ["udt/samtools/view.udt.yml"]
    udt_text = udt_path.read_text(encoding="utf-8")
    report = validate_udt_yaml(udt_text, check_conversion=True)
    assert report.artifact_valid is True, report.notes
    assert report.conversion_supported is True, report.notes
    assert "samtools view" in udt_text
    assert "quay.io/biocontainers/samtools:1.20--h50ea8bc_1" in udt_text
    assert "configfiles:" in udt_text
    assert "settings.json" in udt_text
    assert "threads" in udt_text
    assert "$(inputs.input.path)" in udt_text
    assert "$(outputs.output.path)" in udt_text
    assert "$input" not in udt_text


def test_extract_corpus_records_wrapper_helper_scripts_and_configfiles(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "helper_tool"
    (tool_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: helper_tool\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "helper.py").write_text("print('python helper')\n", encoding="utf-8")
    (tool_dir / "plot.R").write_text("print('r helper')\n", encoding="utf-8")
    (tool_dir / "scripts" / "run.sh").write_text("echo shell helper\n", encoding="utf-8")
    (tool_dir / "helper.xml").write_text(
        """
<tool id="helper_tool" name="Helper Tool" version="1.0">
  <command><![CDATA[
python '$__tool_directory__/helper.py' &&
Rscript plot.R &&
bash scripts/run.sh &&
python '$script'
  ]]></command>
  <configfiles>
    <configfile name="script" filename="generated.py" foo="bar"><![CDATA[
import scanpy as sc
adata = sc.read_h5ad("anndata.h5ad")
sc.pp.filter_cells(adata, min_counts=1)
    ]]></configfile>
    <configfile name="mainparams"><![CDATA[
KEY PARAMETERS FOR THE PROGRAM structure.

#define MAXPOPS    $main.MAXPOPS  // default:2      // (int) number of populations assumed

Command line options:
-m mainparams
-e extraparams
-K MAXPOPS
    ]]></configfile>
    <configfile name="settings" filename="settings.json"><![CDATA[
{"mode": "fast"}
    ]]></configfile>
    <configfile name="xml_template" filename="template.xml">
prefix <section attr="one"><item>$input</item></section> suffix
    </configfile>
  </configfiles>
  <outputs>
    <data name="output" format="txt"/>
  </outputs>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    helper_paths = {item["relative_path"] for item in record["wrapper_helper_files"]}
    assert helper_paths == {"helper.py", "plot.R", "scripts/run.sh"}
    assert record["wrapper_source_summary"]["helper_file_count"] == 3
    assert record["wrapper_source_summary"]["configfile_count"] == 4
    assert record["wrapper_source_summary"]["truncated_configfile_count"] == 0
    assert record["wrapper_source_summary"]["api_backed_wrapper"] is True
    assert record["wrapper_source_summary"]["configfile_api_call_count"] == 2
    assert record["wrapper_source_summary"]["configfile_command_doc_count"] == 1
    assert record["wrapper_source_summary"]["configfile_parameter_doc_count"] == 1
    assert "Wrapper configfile context:" in record["help_text"]
    assert "API calls used by generated wrapper scripts:" in record["help_text"]
    assert "scanpy.read_h5ad, scanpy.pp.filter_cells" in record["help_text"]
    assert "Command-line documentation embedded in wrapper configfiles:" in record["help_text"]
    assert "-m mainparams" in record["help_text"]
    assert "Parameter documentation embedded in wrapper configfiles:" in record["help_text"]
    assert "MAXPOPS (default=2; type=int; number of populations assumed)" in record["help_text"]
    configfiles = {item["name"]: item for item in record["wrapper_configfiles"]}
    script_config = configfiles["script"]
    assert script_config["filename"] == "generated.py"
    assert script_config["attributes"]["foo"] == "bar"
    assert script_config["template_kind"] == "script_template"
    assert script_config["language"] == "python"
    assert script_config["referenced_by_command"] is True
    assert "sc.pp.filter_cells" in script_config["content"]
    assert [call["qualified_call"] for call in script_config["api_calls"]] == [
        "scanpy.read_h5ad",
        "scanpy.pp.filter_cells",
    ]
    mainparams_config = configfiles["mainparams"]
    assert mainparams_config["command_docs"][0]["kind"] == "command_line_options"
    assert "-m mainparams" in mainparams_config["command_docs"][0]["text"]
    assert mainparams_config["parameter_docs"][0]["name"] == "MAXPOPS"
    assert mainparams_config["parameter_docs"][0]["default"] == "2"
    assert mainparams_config["parameter_docs"][0]["type"] == "int"
    settings_config = configfiles["settings"]
    assert settings_config["template_kind"] == "config_template"
    assert settings_config["language"] == "json"
    assert settings_config["referenced_by_command"] is False
    xml_config = configfiles["xml_template"]
    assert xml_config["template_kind"] == "config_template"
    assert "<section attr=\"one\"><item>$input</item></section>" in xml_config["content"]
    assert "prefix" in xml_config["content"]
    assert "suffix" in xml_config["content"]
    assert xml_config["byte_count"] == xml_config["stored_byte_count"]
    assert xml_config["content_truncated"] is False


def test_extract_corpus_records_api_calls_from_wrapper_helper_scripts(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "alphagenome_tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: alphagenome_tool\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "alphagenome_helper.py").write_text(
        """
import argparse
from alphagenome.models import dna_client

parser = argparse.ArgumentParser(
    description="Score genetic variants using AlphaGenome predict_variant()"
)
parser.add_argument("--output-types", default=["RNA_SEQ"], help="AlphaGenome output")
client = dna_client.create(api_key="token")
""".strip(),
        encoding="utf-8",
    )
    (tool_dir / "alphagenome.xml").write_text(
        """
<tool id="alphagenome_tool" name="AlphaGenome Tool" version="1.0">
  <requirements>
    <requirement type="package" version="0.6.1">alphagenome</requirement>
  </requirements>
  <command><![CDATA[
python '$__tool_directory__/alphagenome_helper.py' --input '$input' --output '$output'
  ]]></command>
  <inputs>
    <param name="input" type="data" format="vcf"/>
  </inputs>
  <outputs>
    <data name="output" format="vcf"/>
  </outputs>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    helper = record["wrapper_helper_files"][0]
    assert helper["relative_path"] == "alphagenome_helper.py"
    assert [call["qualified_call"] for call in helper["api_calls"]] == [
        "alphagenome.models.dna_client.create"
    ]
    assert record["wrapper_source_summary"]["helper_api_call_count"] == 1
    assert record["wrapper_source_summary"]["api_backed_wrapper"] is True
    assert "Wrapper helper context:" in record["help_text"]
    assert "alphagenome.models.dna_client.create" in record["help_text"]


def test_extract_corpus_treats_extensionless_referenced_python_configfile_as_script(
    tmp_path: Path,
) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "api_config_tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: api_config_tool\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "api.xml").write_text(
        """
<tool id="api_config_tool" name="API Config Tool" version="1.0">
  <requirements>
    <requirement type="package" version="0.3.2">episcanpy</requirement>
  </requirements>
  <command><![CDATA[
python '$script_file'
  ]]></command>
  <configfiles>
    <configfile name="script_file"><![CDATA[
import episcanpy as epi
adata = epi.read("input.h5ad")
epi.pp.binarize(adata)
    ]]></configfile>
  </configfiles>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    configfile = record["wrapper_configfiles"][0]
    assert configfile["name"] == "script_file"
    assert configfile["extension"] == ""
    assert configfile["template_kind"] == "script_template"
    assert configfile["referenced_by_command"] is True
    assert record["wrapper_source_summary"]["api_backed_wrapper"] is True
    assert [call["qualified_call"] for call in configfile["api_calls"]] == [
        "episcanpy.read",
        "episcanpy.pp.binarize",
    ]
    assert "Wrapper configfile context:" in record["help_text"]
    assert "API calls used by generated wrapper scripts:" in record["help_text"]
    assert "episcanpy.read, episcanpy.pp.binarize" in record["help_text"]


def test_extract_corpus_truncates_large_configfiles_and_omits_from_udt(
    tmp_path: Path,
) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "large_config_tool"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: large_config_tool\nowner: iuc\n", encoding="utf-8")
    configfile_max_bytes = 512
    large_content = "x" * (configfile_max_bytes + 1024)
    (tool_dir / "large.xml").write_text(
        f"""
<tool id="large_config_tool" name="Large Config Tool" version="1.0">
  <command><![CDATA[cat '$input' > '$output']]></command>
  <configfiles>
    <configfile name="large" filename="large.txt"><![CDATA[
{large_content}
    ]]></configfile>
  </configfiles>
  <inputs>
    <param name="input" type="data" format="txt"/>
  </inputs>
  <outputs>
    <data name="output" format="txt"/>
  </outputs>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(
            max_workers=1,
            retries=1,
            fetch_documentation=False,
            synthesize_udt_yaml=True,
            wrapper_configfile_max_bytes=configfile_max_bytes,
        ),
    )

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    configfile = record["wrapper_configfiles"][0]
    full_content = f"{large_content}\n    "
    full_bytes = full_content.encode("utf-8", errors="replace")
    assert configfile["byte_count"] == len(full_bytes)
    assert configfile["stored_byte_count"] <= configfile_max_bytes
    assert configfile["sha256"] == hashlib.sha256(full_bytes).hexdigest()
    assert configfile["content_truncated"] is True
    assert configfile["content"].endswith(_CONFIGFILE_TRUNCATION_MARKER)
    assert record["wrapper_source_summary"]["truncated_configfile_count"] == 1

    udt_text = Path(record["udt_yaml_path"]).read_text(encoding="utf-8")
    assert "configfiles:" not in udt_text
    assert "large.txt" not in udt_text


def test_extract_corpus_rejects_unsafe_or_nontext_wrapper_helpers(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "unsafe_tool"
    (tool_dir / "test-data").mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: unsafe_tool\nowner: iuc\n", encoding="utf-8")
    (tools_root / "outside.py").write_text("print('outside')\n", encoding="utf-8")
    (tool_dir / "test-data" / "skip.py").write_text("print('skip')\n", encoding="utf-8")
    (tool_dir / "large.py").write_text("x = 1\n" * 50_000, encoding="utf-8")
    (tool_dir / "binary.py").write_bytes(b"print('binary')\x00\n")
    (tool_dir / "unsafe.xml").write_text(
        """
<tool id="unsafe_tool" name="Unsafe Tool" version="1.0">
  <command><![CDATA[
python '$__tool_directory__/../outside.py' &&
python '$__tool_directory__/test-data/skip.py' &&
python '$__tool_directory__/large.py' &&
python '$__tool_directory__/binary.py'
  ]]></command>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert record["wrapper_helper_files"] == []
    skip_reasons = record["wrapper_source_summary"]["skip_reasons"]
    assert skip_reasons["outside_tool_dir"] == 1
    assert skip_reasons["test_data"] == 1
    assert skip_reasons["too_large"] == 1
    assert skip_reasons["binary_content"] == 1


def test_extract_corpus_writes_container_enriched_help_before_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "samtools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: samtools\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "samtools_view.xml").write_text(
        """
<tool id="samtools_view" name="samtools_view" version="1.0.0">
  <requirements>
    <container type="docker">quay.io/biocontainers/samtools:1.10--h2e538c0_3</container>
  </requirements>
  <command><![CDATA[samtools view -h '$input' > '$output']]></command>
  <inputs>
    <param name="input" type="data" format="bam"/>
  </inputs>
  <outputs>
    <data name="output" format="sam"/>
  </outputs>
  <help><![CDATA[Wrapper help text]]></help>
</tool>
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [ContainerRuntime("apptainer", "apptainer")],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._prepare_container",
        lambda image, runtime, settings: ContainerPreparation(
            ok=True,
            runtime="apptainer",
            image=image,
            identifier=str(tmp_path / "samtools.sif"),
            source="galaxy-depot",
        ),
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._run_command",
        lambda command, timeout_seconds: subprocess.CompletedProcess(
            command,
            0,
            stdout="Usage: samtools [options]\n",
            stderr="",
        ),
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(
            max_workers=1, retries=1, fetch_documentation=False, execute_containers=True
        ),
    )

    records = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["original_help_text"] == "Wrapper help text"
    assert "Usage: samtools" in records[0]["container_help_text"]
    assert "Usage: samtools" in records[0]["help_text"]
    assert records[0]["selected_container_runtime"] == "apptainer"
    assert checkpoint.read_text(encoding="utf-8").strip()


def test_extract_corpus_uses_token_expanded_xml_for_container_refs(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "intervene"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: intervene\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "macros.xml").write_text(
        """
<macros>
  <token name="@TOOL_VERSION@">0.5.9--py27r3.4.1_0</token>
</macros>
""".strip(),
        encoding="utf-8",
    )
    (tool_dir / "intervene.xml").write_text(
        """
<tool id="intervene" name="intervene" version="@TOOL_VERSION@">
  <requirements>
    <container type="docker">quay.io/biocontainers/intervene:@TOOL_VERSION@</container>
  </requirements>
  <command><![CDATA[intervene venn --input '$input' --output '$output']]></command>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    records = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    rec = records[0]
    assert rec["container_candidates"] == ["quay.io/biocontainers/intervene:0.5.9--py27r3.4.1_0"]
    assert rec["selected_container"] == "quay.io/biocontainers/intervene:0.5.9--py27r3.4.1_0"
    assert "@TOOL_VERSION@" not in Path(rec["expanded_xml_path"]).read_text(encoding="utf-8")


def test_extract_corpus_restart_removes_previous_outputs(tmp_path: Path) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "samtools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: samtools\nowner: iuc\n", encoding="utf-8")
    (tool_dir / "samtools_view.xml").write_text(
        """
<tool id="samtools_view" name="samtools_view" version="1.0.0">
  <command><![CDATA[samtools view -h '$input' > '$output']]></command>
  <inputs>
    <param name="input" type="data" format="bam"/>
  </inputs>
  <outputs>
    <data name="output" format="sam"/>
  </outputs>
</tool>
""".strip(),
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    expanded = out_jsonl.parent / "expanded"
    expanded.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text('{"old": true}\n', encoding="utf-8")
    checkpoint.write_text("samtools::old.xml\n", encoding="utf-8")
    out_jsonl.with_suffix(".index.json").write_text("{}", encoding="utf-8")
    out_jsonl.with_suffix(".execution.json").write_text("{}", encoding="utf-8")
    (expanded / "old.xml").write_text("<tool />", encoding="utf-8")

    result = extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(
            max_workers=1, retries=1, fetch_documentation=False, restart=True
        ),
    )

    assert result["restart"] is True
    assert result["restart_removed"] == []
    assert str(out_jsonl) in result["restart_archived"]
    assert str(checkpoint) in result["restart_archived"]
    assert str(expanded) in result["restart_archived"]
    archive_dir = Path(result["restart_archive_dir"])
    assert (archive_dir / out_jsonl.name).read_text(encoding="utf-8") == '{"old": true}\n'
    assert (archive_dir / checkpoint.name).read_text(encoding="utf-8") == "samtools::old.xml\n"
    assert (archive_dir / "manifest.json").exists()

    records = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert "old" not in records[0]
    assert result["already_processed"] == 0
    run_dir = Path(result["run_dir"])
    assert (run_dir / out_jsonl.name).exists()
    assert (run_dir / out_jsonl.with_suffix(".execution.json").name).exists()
    assert (out_jsonl.parent / "current").read_text(encoding="utf-8").strip() == result["run_id"]


def test_extract_corpus_emits_precontainer_status_and_checkpoints_incrementally(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "samtools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    xml_file = tool_dir / "samtools_view.xml"
    (tool_dir / ".shed.yml").write_text("name: samtools\nowner: iuc\n", encoding="utf-8")
    xml_file.write_text("<tool id='samtools_view' name='samtools_view' />", encoding="utf-8")

    events: list[dict] = []
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status",
        lambda payload, status_log_path=None: events.append(payload),
    )

    def fake_with_retries(*args, **kwargs):
        return ToolRecord(
            package_id="iuc/samtools",
            tool_id="samtools_view",
            tool_name="samtools_view",
            tool_dir=str(tool_dir),
            wrapper_path=str(xml_file),
            help_text="Wrapper help",
            original_help_text="Wrapper help",
        )

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._with_retries", fake_with_retries)

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    result = extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(max_workers=1, retries=1, fetch_documentation=False),
    )

    statuses = [event["status"] for event in events]
    assert "extract-plan" in statuses
    assert "extract-record-completed" in statuses
    assert "extract-completed" in statuses
    assert result["processed_now"] == 1
    assert out_jsonl.read_text(encoding="utf-8").strip()
    assert checkpoint.read_text(encoding="utf-8").strip() == "samtools::samtools_view.xml"


def test_rebuild_execution_report_from_jsonl(tmp_path: Path) -> None:
    corpus_jsonl = tmp_path / "corpus.jsonl"
    record = ToolRecord(
        package_id="iuc/samtools",
        tool_id="samtools_depth",
        wrapper_path="/tools/samtools/depth.xml",
        container_candidate_details=[
            {
                "image": "quay.io/biocontainers/samtools:1.23--h96c455f_0",
                "source": "mulled-single",
                "status": "ok",
            }
        ],
        container_execution=[
            {
                "phase": "run",
                "status": "container-command-help-degraded",
                "image": "quay.io/biocontainers/samtools:1.23--h96c455f_0",
                "returncode": 1,
            }
        ],
    )
    corpus_jsonl.write_text(record.to_json() + "\n", encoding="utf-8")

    report_path = rebuild_execution_report_from_jsonl(corpus_jsonl)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rebuilt_from_records"] is True
    assert report["summary"]["commands_executed"] == 1
    assert report["summary"]["commands_failed"] == 0
    assert report["summary"]["help_degraded"] == 1
    assert report["records"][0]["tool_id"] == "samtools_depth"


def test_write_corpus_diagnostics(tmp_path: Path) -> None:
    corpus_jsonl = tmp_path / "tools-iuc-corpus.jsonl"
    checkpoint = tmp_path / "tools-iuc-corpus.checkpoint"
    execution_report = tmp_path / "tools-iuc-corpus.execution.json"
    current_run = tmp_path / "current"
    diagnostics_dir = tmp_path / "diagnostics"
    records = [
        ToolRecord(
            package_id="iuc/samtools",
            tool_id="samtools_depth",
            wrapper_path="/tools/samtools/depth.xml",
            container_help_text="Usage: samtools depth",
            container_usage_text="$ samtools depth\nUsage: samtools depth",
            container_api_validation=[{"status": "container-api-validation-ok"}],
            wrapper_source_summary={
                "api_backed_wrapper": True,
                "configfile_command_doc_count": 1,
            },
            bioconda_sources=[
                {
                    "package": "tabix",
                    "source_provider_package": "htslib",
                    "source_provider_reason": "source_less_run_dependency",
                    "source_command_docs": [
                        {
                            "path": "README.md",
                            "line": 3,
                            "text": "Usage: tabix [options] file",
                        }
                    ],
                }
            ],
            container_execution=[
                {
                    "phase": "run",
                    "status": "container-command-help-degraded",
                    "image": "quay.io/biocontainers/samtools:1.23",
                    "returncode": 1,
                }
            ],
        ),
        ToolRecord(
            package_id="iuc/awk",
            tool_id="awk",
            wrapper_path="/tools/awk/awk.xml",
            bioconda_sources=[{"package": "awk", "source_error": "source unavailable"}],
            container_execution=[
                {
                    "phase": "run",
                    "status": "container-command-nonhelp",
                    "image": "quay.io/biocontainers/awk:1",
                    "returncode": 0,
                    "stdout": "awk banner",
                }
            ],
        ),
    ]
    corpus_jsonl.write_text(
        "\n".join(record.to_json() for record in records) + "\n", encoding="utf-8"
    )
    checkpoint.write_text("samtools::depth.xml\nawk::awk.xml\n", encoding="utf-8")
    current_run.write_text("run-1\n", encoding="utf-8")
    execution_report.write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "summary": {"commands_executed": 2, "commands_failed": 1},
                "records": [json.loads(record.to_json()) for record in records],
            }
        ),
        encoding="utf-8",
    )

    result = write_corpus_diagnostics(
        execution_report_path=execution_report,
        diagnostics_dir=diagnostics_dir,
        corpus_jsonl=corpus_jsonl,
        checkpoint_file=checkpoint,
        current_run_path=current_run,
    )

    assert result["summary"]["total_records"] == 2
    assert result["summary"]["records_with_container_help"] == 1
    assert result["summary"]["nonhelp_sample_count"] == 1
    assert result["summary"]["counts_consistent"] is True
    coverage = json.loads(
        (diagnostics_dir / "container-help-coverage.json").read_text(encoding="utf-8")
    )
    assert coverage["records_without_container_help"] == 1
    assert coverage["records_with_container_usage"] == 1
    assert coverage["records_with_api_validation_ok"] == 1
    assert coverage["api_backed_records"] == 1
    assert coverage["configfile_doc_records"] == 1
    assert coverage["source_command_doc_records"] == 1
    assert coverage["records_with_source_provider"] == 1
    assert coverage["source_provider_package_counts"] == {"htslib": 1}
    assert coverage["source_provider_reason_counts"] == {"source_less_run_dependency": 1}
    assert coverage["records_with_source_errors"] == 1
    assert coverage["source_error_counts"] == {"source unavailable": 1}
    counts = (diagnostics_dir / "container-status-counts.txt").read_text(encoding="utf-8")
    assert "container-command-nonhelp" in counts
    samples = json.loads((diagnostics_dir / "nonhelp-samples.json").read_text(encoding="utf-8"))
    assert samples[0]["tool_id"] == "awk"
    source_coverage = json.loads(
        (diagnostics_dir / "source-coverage.json").read_text(encoding="utf-8")
    )
    assert source_coverage["records_with_source_mapping"] == 2
    assert source_coverage["records_with_usable_source"] == 1
    assert source_coverage["records_with_provider_source"] == 1
    assert source_coverage["records_with_source_error"] == 1
    assert source_coverage["source_status_counts"] == {
        "provider_source": 1,
        "source_error": 1,
    }
    source_missing = json.loads(
        (diagnostics_dir / "source-missing.json").read_text(encoding="utf-8")
    )
    assert source_missing[0]["tool_id"] == "awk"
    assert source_missing[0]["status"] == "source_error"
    integrity = json.loads((diagnostics_dir / "integrity.json").read_text(encoding="utf-8"))
    assert integrity["current_run"]["value"] == "run-1"
    retry_manifest = json.loads(
        (diagnostics_dir / "retry-manifest.json").read_text(encoding="utf-8")
    )
    assert retry_manifest["wrappers"]
    assert any(item["tool_id"] == "awk" for item in retry_manifest["wrappers"])
    failure_inventory = json.loads(
        (diagnostics_dir / "failure-inventory.json").read_text(encoding="utf-8")
    )
    assert failure_inventory["summary"]["issue_wrappers"] >= 1
    assert "bad_probe_variant" in failure_inventory["summary"]["category_counts"]
    assert (diagnostics_dir / "failure-samples.json").exists()
    assert (diagnostics_dir / "recovery-summary.md").exists()


def test_extract_corpus_prepares_bioconda_repo_once_for_multiple_wrappers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tools_root = tmp_path / "tools"
    tool_dir = tools_root / "samtools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / ".shed.yml").write_text("name: samtools\nowner: iuc\n", encoding="utf-8")
    for name in ("samtools_view.xml", "samtools_sort.xml"):
        (tool_dir / name).write_text(
            """
<tool id="samtools" name="samtools">
  <requirements>
    <requirement type="package" version="1.10">samtools</requirement>
  </requirements>
  <command><![CDATA[samtools view '$input']]></command>
</tool>
""".strip(),
            encoding="utf-8",
        )

    fake_repo = tmp_path / "bioconda-recipes"
    recipe_dir = fake_repo / "recipes" / "samtools"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        "package:\n  name: samtools\nversion: 1.10\n", encoding="utf-8"
    )

    calls: list[tuple[Path, str]] = []

    def fake_ensure(cache_root: Path, ref: str, settings: ExtractionSettings):
        calls.append((cache_root, ref))
        return fake_repo, ""

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._ensure_bioconda_repo", fake_ensure)
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )

    out_jsonl = tmp_path / "datasets" / "corpus.jsonl"
    checkpoint = tmp_path / "datasets" / "corpus.checkpoint"
    extract_tools_corpus(
        tools_root=tools_root,
        output_jsonl=out_jsonl,
        checkpoint_file=checkpoint,
        settings=ExtractionSettings(
            max_workers=2,
            retries=1,
            fetch_documentation=False,
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
    )

    assert calls == [(tmp_path / "source-cache", "master")]
    records = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert all(record["bioconda_sources"][0]["package"] == "samtools" for record in records)
