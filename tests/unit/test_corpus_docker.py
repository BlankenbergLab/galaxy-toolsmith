from __future__ import annotations

import io
import json
import shlex
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError

import requests

from galaxy_toolsmith.http_client import (
    DEFAULT_BROWSER_FALLBACK_USER_AGENT,
    DEFAULT_HTTP_USER_AGENT,
    HTTP_USER_AGENT_ENV_VAR,
)
from galaxy_toolsmith.data.corpus import (
    BiocondaRecipeSnapshot,
    ContainerImageQuarantineStore,
    ContainerPreparation,
    ContainerRuntime,
    ExtractionSettings,
    MulledTarget,
    ToolRecord,
    _build_failure_inventory,
    _build_container_candidate_details,
    _choose_container_candidate,
    _choose_container_image,
    _command_primary,
    _conda_forge_feedstock_variants,
    _container_help_fragment,
    _available_container_runtimes,
    _container_shell_command,
    _container_usage_fragment,
    _container_probe_status,
    _container_probe_timeout,
    _container_run_command,
    _container_image_quarantine_get,
    _container_image_quarantine_put,
    _docker_ref_for_image,
    _download_and_extract_archive,
    _execute_container_help_batches,
    _ensure_conda_forge_feedstock_repo,
    _extract_help_commands,
    _extract_python_configfile_api_calls,
    _extract_recipe_run_dependency_names,
    _extract_source_fields,
    _extract_recipe_version,
    _extract_source_command_hints,
    _extract_source_command_docs,
    _failed_source_fallback_candidates,
    _image_matches_requirement_versions,
    _infer_command_signatures,
    _is_missing_command_probe,
    _is_source_denylisted_package,
    _load_container_image_quarantine_store,
    _looks_like_help_text,
    _mulled_biocontainer_images,
    _mulled_v2_image_name,
    _normalize_container_candidate,
    _prepare_docker_container,
    _prepare_singularity_container,
    _probe_command_base,
    _python_api_validation_command,
    _record_failure_items,
    _record_help_command_plan,
    _record_help_commands,
    _record_api_validation_commands,
    _render_bioconda_source_fields,
    _resolve_bioconda_source_mappings,
    _resolve_recipe_source_mapping,
    _run_command,
    _select_recipe_snapshot_from_candidates,
    _singularity_cache_path,
    _singularity_depot_image_url,
    _sif_sandbox_cache_paths,
    _script_text_command_signatures,
    _source_recipe_package_alias,
    _source_version_match_status,
)


def _git(repo, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_recipe_repo(repo) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "gtsm@example.org")
    _git(repo, "config", "user.name", "GTSM Tests")
    _git(repo, "checkout", "-B", "master")


def test_command_primary_preserves_python_module_command() -> None:
    assert _command_primary("python -m amas.AMAS replicate --help") == "python -m amas.AMAS"
    assert _command_primary("FUNANNOTATE_DB='$database.fields.path' funannotate database") == (
        "funannotate"
    )
    assert _command_primary("NULL command") == "command"


def test_container_probe_command_uses_utf8_locale() -> None:
    preparation = ContainerPreparation(
        ok=True,
        runtime="singularity",
        image="tool:1.0",
        identifier="/cache/tool.sif",
    )

    command = _container_run_command(preparation, "tool --help", ExtractionSettings())
    shell_command = command[-1]

    assert "LANG=C.UTF-8" in shell_command
    assert "LC_ALL=C.UTF-8" in shell_command
    assert "PYTHONIOENCODING=utf-8" in shell_command


def test_extract_help_commands_avoids_unsafe_bwa_positional_help() -> None:
    commands = _extract_help_commands("bwa-mem2", ["index"], probe_mode="exploratory")
    assert "bwa-mem2 index help" not in commands
    assert "bwa-mem2 help index" not in commands
    assert "bwa-mem2 index --help" in commands
    assert "bwa-mem2 --help" in commands
    assert commands.index("bwa-mem2 --help") < commands.index("bwa-mem2 index --help")


def test_extract_help_commands_uses_bcftools_subcommand_help_flags() -> None:
    assert _extract_help_commands("bcftools", ["mpileup"], probe_mode="exploratory") == [
        "bcftools mpileup --help",
        "bcftools mpileup -h",
        "bcftools help mpileup",
    ]


def _commit_recipe(repo, package: str, version: str) -> None:
    recipe_dir = repo / "recipes" / package
    recipe_dir.mkdir(parents=True, exist_ok=True)
    (recipe_dir / "meta.yaml").write_text(
        f"""
{{% set name = "{package}" %}}
{{% set version = "{version}" %}}
source:
  url: https://example.org/{{{{ name }}}}-{{{{ version }}}}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    _git(repo, "add", f"recipes/{package}/meta.yaml")
    _git(repo, "commit", "-m", f"{package} {version}")


def _commit_feedstock_recipe(repo, package: str, version: str, source_url: str | None = None) -> None:
    recipe_dir = repo / "recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    source_url = source_url or f"https://example.org/{package}-{version}.tar.gz"
    (recipe_dir / "meta.yaml").write_text(
        f"""
{{% set name = "{package}" %}}
{{% set version = "{version}" %}}
source:
  - url: {source_url}
""".strip(),
        encoding="utf-8",
    )
    _git(repo, "add", "recipe/meta.yaml")
    _git(repo, "commit", "-m", f"{package} feedstock {version}")


def test_choose_container_prefers_matching_version() -> None:
    candidates = [
        "quay.io/biocontainers/samtools:1.17--h00cdaf9_0",
        "quay.io/biocontainers/samtools:1.10--h2e538c0_3",
    ]
    chosen = _choose_container_image(candidates, {"samtools": "1.10"})
    assert "1.10" in chosen


def test_image_matches_requirement_versions() -> None:
    assert _image_matches_requirement_versions("quay.io/tool:2.1--build", {"tool": "2.1"}) is True
    assert _image_matches_requirement_versions("quay.io/tool:2.2--build", {"tool": "2.1"}) is False


def test_extract_help_commands_dedupes() -> None:
    commands = _extract_help_commands("samtools", ["view", "view", "sort"])
    assert commands[0] == "samtools view --help"
    assert "samtools view -h" in commands
    assert "samtools view help" in commands
    assert "samtools view" in commands
    assert "samtools view --help" in commands
    assert "samtools --help" in commands
    assert "samtools -h" in commands
    assert "samtools help" in commands
    assert "samtools" in commands
    assert "view --help" not in commands
    assert len(commands) == len(set(commands))


def test_extract_help_commands_safe_mode_uses_only_help_flags() -> None:
    commands = _extract_help_commands("samtools", ["view"], probe_mode="safe")
    assert commands == [
        "samtools view --help",
        "samtools view -h",
        "samtools --help",
        "samtools -h",
    ]


def test_help_classifier_accepts_command_description_tables() -> None:
    text = """
No command selected. Help follows:

COMMAND        DESCRIPTION
summary        summarize MAT content
extract        extract paths and samples
""".strip()

    assert _looks_like_help_text(text) is True


def test_mulled_v2_image_name_matches_galaxy_examples() -> None:
    targets = [
        MulledTarget("samtools", version="1.3.1"),
        MulledTarget("bwa", version="0.7.13"),
    ]
    assert (
        _mulled_v2_image_name(targets)
        == "mulled-v2-fe8faa35dbf6dc65a0f7f5d4ea12e31a79f73e40:4d0535c94ef45be8459f429561f0894c3fe0ebcf"
    )


def test_singularity_depot_url_maps_biocontainers_ref() -> None:
    url = _singularity_depot_image_url(
        "quay.io/biocontainers/samtools:1.10--h2e538c0_3",
        "https://depot.galaxyproject.org/singularity",
    )
    assert url == "https://depot.galaxyproject.org/singularity/samtools:1.10--h2e538c0_3"


def test_container_ref_normalizes_biocontainers_equals_syntax() -> None:
    assert (
        _normalize_container_candidate("astral-tree==5.7.8--hdfd78af_0")
        == "astral-tree:5.7.8--hdfd78af_0"
    )
    assert (
        _normalize_container_candidate("quay.io/biocontainers/bmtagger==3.101--h470a237_4")
        == "quay.io/biocontainers/bmtagger:3.101--h470a237_4"
    )
    assert _docker_ref_for_image("bellerophon==1.0--pyh5e36f6f_0") == (
        "quay.io/biocontainers/bellerophon:1.0--pyh5e36f6f_0"
    )


def test_container_ref_rejects_unresolved_placeholders() -> None:
    assert _normalize_container_candidate("quay.io/biocontainers/intervene:@TOOL_VERSION@") == ""
    assert _normalize_container_candidate("quay.io/biocontainers/tool:${VERSION}") == ""
    assert _normalize_container_candidate("quay.io/biocontainers/tool:{version}") == ""
    assert _normalize_container_candidate("__tool_directory__/tool.sif") == ""


def test_mulled_candidates_use_real_quay_tags_without_exact_fallback(monkeypatch) -> None:
    calls = []

    def fake_lookup(namespace, repository, tag_prefix="", limit=5):
        calls.append((namespace, repository, tag_prefix, limit))
        if repository == "samtools":
            return ["1.23--h96c455f_0", "1.23--h50ea8bc_1"]
        return []

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._lookup_quay_tags", fake_lookup)

    assert _mulled_biocontainer_images(["samtools"], {"samtools": "1.23"}) == [
        "quay.io/biocontainers/samtools:1.23--h96c455f_0",
        "quay.io/biocontainers/samtools:1.23--h50ea8bc_1",
    ]
    assert calls[0] == ("biocontainers", "samtools", "1.23", 5)
    assert _mulled_biocontainer_images(["ucsc-genepredtobed"], {"ucsc-genepredtobed": "357"}) == []


def test_mulled_v2_candidates_are_validated_against_quay(monkeypatch) -> None:
    def fake_lookup(namespace, repository, tag_prefix="", limit=5):
        assert namespace == "biocontainers"
        assert repository.startswith("mulled-v2-")
        assert tag_prefix
        return [f"{tag_prefix}-0"]

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._lookup_quay_tags", fake_lookup)

    images = _mulled_biocontainer_images(
        ["samtools", "bwa"], {"samtools": "1.3.1", "bwa": "0.7.13"}
    )

    assert len(images) == 1
    assert images[0].startswith("quay.io/biocontainers/mulled-v2-")


def test_build_container_candidates_records_invalid_explicit_refs(monkeypatch) -> None:
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._lookup_quay_tags", lambda *args, **kwargs: []
    )
    details = _build_container_candidate_details(
        container_refs=["quay.io/biocontainers/intervene:@TOOL_VERSION@"],
        package_names=["intervene"],
        requirement_versions={"intervene": "0.5.9"},
        settings=ExtractionSettings(resolve_containers=False, execute_containers=False),
    )

    assert details == [
        {
            "image": "quay.io/biocontainers/intervene:@TOOL_VERSION@",
            "source": "explicit",
            "packages": [],
            "priority": 300,
            "status": "skipped",
            "error_text": "container reference contains unresolved placeholder",
        }
    ]


def test_explicit_depot_url_downloads_for_singularity_without_build(monkeypatch, tmp_path) -> None:
    image = "https://depot.galaxyproject.org/singularity/argnorm:0.6.0--pyhdfd78af_0"
    settings = ExtractionSettings(container_cache_dir=tmp_path / "containers")
    calls = []

    def fake_download(url, destination, timeout_seconds):
        calls.append((url, destination, timeout_seconds))
        return ContainerPreparation(
            ok=True, runtime="singularity", image=url, identifier=str(destination)
        )

    def fail_run(command, timeout_seconds):
        raise AssertionError(f"unexpected runtime command: {command}")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_depot_image", fake_download)
    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fail_run)

    prepared = _prepare_singularity_container(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        settings=settings,
    )

    assert prepared.ok is True
    assert prepared.runtime == "apptainer"
    assert prepared.source == "galaxy-depot"
    assert prepared.identifier.endswith("argnorm:0.6.0--pyhdfd78af_0")
    assert calls[0][0] == image
    assert _singularity_cache_path(image, settings).name == "argnorm:0.6.0--pyhdfd78af_0"


def test_auto_sif_exec_mode_keeps_cached_sif_when_direct_mount_supported(
    monkeypatch, tmp_path
) -> None:
    image = "quay.io/biocontainers/samtools:1.10--h2e538c0_3"
    settings = ExtractionSettings(container_cache_dir=tmp_path / "containers")
    cache_path = _singularity_cache_path(image, settings)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"sif")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._direct_sif_mount_supported", lambda: True)
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._run_command",
        lambda command, timeout_seconds: (_ for _ in ()).throw(
            AssertionError(f"unexpected sandbox build: {command}")
        ),
    )

    prepared = _prepare_singularity_container(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        settings=settings,
    )

    assert prepared.ok is True
    assert prepared.source == "cache"
    assert prepared.identifier == str(cache_path)


def test_auto_sif_exec_mode_builds_and_reuses_persistent_sandbox(
    monkeypatch, tmp_path
) -> None:
    image = "quay.io/biocontainers/samtools:1.10--h2e538c0_3"
    settings = ExtractionSettings(container_cache_dir=tmp_path / "containers")
    cache_path = _singularity_cache_path(image, settings)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"sif")
    calls = []

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._direct_sif_mount_supported", lambda: False)

    def fake_run(command, timeout_seconds):
        calls.append(command)
        if command[1:3] == ["exec", "--cleanenv"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        sandbox_tmp = command[-2]
        assert command[:3] == ["apptainer", "build", "--sandbox"]
        assert command[-1] == str(cache_path)
        (tmp_path / "marker").write_text("called", encoding="utf-8")
        sandbox_tmp_path = tmp_path / "unused"
        sandbox_tmp_path = type(cache_path)(sandbox_tmp)
        sandbox_tmp_path.mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="sandbox ok", stderr="")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    prepared = _prepare_singularity_container(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        settings=settings,
    )
    reused = _prepare_singularity_container(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        settings=settings,
    )

    sandbox_path, metadata_path = _sif_sandbox_cache_paths(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        sif_path=cache_path,
        settings=settings,
    )
    assert prepared.ok is True
    assert prepared.source == "sif-sandbox-cache"
    assert prepared.identifier == str(sandbox_path)
    assert reused.identifier == str(sandbox_path)
    assert metadata_path.exists()
    assert len(calls) == 2
    assert calls[0][:3] == ["apptainer", "build", "--sandbox"]
    assert calls[1][1:3] == ["exec", "--cleanenv"]


def test_sif_sandbox_falls_back_to_unsquashfs_when_apptainer_build_fails(
    monkeypatch, tmp_path
) -> None:
    image = "quay.io/biocontainers/samtools:1.10--h2e538c0_3"
    settings = ExtractionSettings(
        container_cache_dir=tmp_path / "containers",
        container_sif_exec_mode="sandbox",
    )
    cache_path = _singularity_cache_path(image, settings)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"sif")
    commands = []

    def fake_run(command, timeout_seconds):
        commands.append(command)
        if command[1:3] == ["build", "--sandbox"]:
            return subprocess.CompletedProcess(command, 255, stdout="", stderr="build failed")
        if command[1:3] == ["sif", "list"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "ID |GROUP |LINK |SIF POSITION (start-end) |TYPE\n"
                    "3  |1     |NONE |40960-4276224            |FS (Squashfs/*System/amd64)\n"
                ),
                stderr="",
            )
        if command[0].endswith("unsquashfs") or command[0] == "unsquashfs":
            type(cache_path)(command[-2]).mkdir(parents=True)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[1:3] == ["exec", "--cleanenv"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

    def fake_run_to_file(command, output_path, timeout_seconds):
        output_path.write_bytes(b"squashfs")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)
    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command_to_file", fake_run_to_file)

    prepared = _prepare_singularity_container(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        settings=settings,
    )

    sandbox_path, _ = _sif_sandbox_cache_paths(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        sif_path=cache_path,
        settings=settings,
    )
    assert prepared.ok is True
    assert prepared.source == "sif-sandbox-cache"
    assert prepared.identifier == str(sandbox_path)
    assert any(command[1:3] == ["sif", "list"] for command in commands)
    assert any(command[0].endswith("unsquashfs") for command in commands)


def test_auto_sif_exec_mode_falls_back_to_sif_when_sandbox_build_fails(
    monkeypatch, tmp_path
) -> None:
    image = "quay.io/biocontainers/samtools:1.10--h2e538c0_3"
    settings = ExtractionSettings(container_cache_dir=tmp_path / "containers")
    cache_path = _singularity_cache_path(image, settings)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"sif")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._direct_sif_mount_supported", lambda: False)
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._run_command",
        lambda command, timeout_seconds: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="sandbox failed"
        ),
    )

    prepared = _prepare_singularity_container(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        settings=settings,
    )

    assert prepared.ok is True
    assert prepared.source == "cache"
    assert prepared.identifier == str(cache_path)


def test_forced_sandbox_sif_exec_mode_reports_sandbox_build_failure(
    monkeypatch, tmp_path
) -> None:
    image = "quay.io/biocontainers/samtools:1.10--h2e538c0_3"
    settings = ExtractionSettings(
        container_cache_dir=tmp_path / "containers",
        container_sif_exec_mode="sandbox",
    )
    cache_path = _singularity_cache_path(image, settings)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"sif")

    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._run_command",
        lambda command, timeout_seconds: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="sandbox failed"
        ),
    )

    prepared = _prepare_singularity_container(
        image=image,
        runtime=ContainerRuntime("apptainer", "apptainer"),
        settings=settings,
    )

    assert prepared.ok is False
    assert prepared.source == "sif-sandbox-cache"
    assert prepared.error_text


def test_docker_rejects_explicit_depot_url_without_pull(monkeypatch, tmp_path) -> None:
    image = "https://depot.galaxyproject.org/singularity/bedtools:2.31.1--h13024bc_3"

    def fail_run(command, timeout_seconds):
        raise AssertionError(f"unexpected docker pull: {command}")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fail_run)
    prepared = _prepare_docker_container(
        image=image,
        runtime=ContainerRuntime("docker", "docker"),
        settings=ExtractionSettings(container_cache_dir=tmp_path / "containers"),
    )

    assert prepared.ok is False
    assert prepared.source == "unsupported-docker-source"
    assert "Docker cannot pull" in prepared.error_text


def test_download_and_extract_archive_reuses_completed_checkout(monkeypatch, tmp_path) -> None:
    checkout = tmp_path / "source"
    checkout.mkdir()
    archive = checkout / "tool.tar.gz"
    archive.write_bytes(b"cached")
    (checkout / "extracted").mkdir()

    def fail_get(*args, **kwargs):
        raise AssertionError("cached source archive should not be downloaded again")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus.requests.get", fail_get)

    assert _download_and_extract_archive("https://example.org/tool.tar.gz", checkout) == str(
        archive
    )


def test_download_and_extract_archive_retries_browser_user_agent_after_403(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(HTTP_USER_AGENT_ENV_VAR, raising=False)
    checkout = tmp_path / "source"
    payload = b"browser fallback source archive"
    calls = []

    class FakeResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def close(self) -> None:
            self.closed = True

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def iter_content(self, chunk_size):
            yield payload

    def fake_get(url, **kwargs):
        calls.append(kwargs["headers"]["User-Agent"])
        return FakeResponse(403 if len(calls) == 1 else 200)

    monkeypatch.setattr("galaxy_toolsmith.data.corpus.requests.get", fake_get)

    archive = _download_and_extract_archive("https://blocked.example.org/tool.tar.gz", checkout)

    assert archive == str(checkout / "tool.tar.gz")
    assert (checkout / "tool.tar.gz").read_bytes() == payload
    assert calls == [
        DEFAULT_HTTP_USER_AGENT,
        DEFAULT_BROWSER_FALLBACK_USER_AGENT,
    ]
    marker = json.loads((checkout / ".gtsm-http-browser-fallback.json").read_text())
    assert marker == {
        "http_user_agent": DEFAULT_BROWSER_FALLBACK_USER_AGENT,
        "http_user_agent_fallback": "browser",
    }


def test_download_and_extract_archive_rejects_oversized_http_response(
    monkeypatch, tmp_path
) -> None:
    checkout = tmp_path / "source"

    class FakeResponse:
        status_code = 200
        headers = {"content-length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size):
            raise AssertionError("oversized content-length should skip streaming")

    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.requests.get",
        lambda url, **kwargs: FakeResponse(),
    )

    try:
        _download_and_extract_archive(
            "https://example.org/large.tar.gz",
            checkout,
            max_bytes=5,
        )
    except RuntimeError as error:
        assert "source download exceeds configured maximum" in str(error)
    else:
        raise AssertionError("oversized source download should fail")

    assert not (checkout / "large.tar.gz").exists()
    assert not (checkout / "large.tar.gz.tmp").exists()


def test_download_and_extract_archive_supports_ftp(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(HTTP_USER_AGENT_ENV_VAR, raising=False)
    checkout = tmp_path / "source"
    payload = b"ftp source archive"
    calls = []

    class FakeFtpResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, timeout):
        calls.append((url.full_url, timeout, url.get_header("User-agent")))
        return FakeFtpResponse(payload)

    monkeypatch.setattr("galaxy_toolsmith.http_client.urlrequest.urlopen", fake_urlopen)

    archive = _download_and_extract_archive("ftp://example.org/pub/tool.tar.gz", checkout)

    assert archive == str(checkout / "tool.tar.gz")
    assert (checkout / "tool.tar.gz").read_bytes() == payload
    assert calls == [
        (
            "ftp://example.org/pub/tool.tar.gz",
            60,
            DEFAULT_HTTP_USER_AGENT,
        )
    ]

    custom_checkout = tmp_path / "custom-source"
    archive = _download_and_extract_archive(
        "ftp://example.org/pub/custom.tar.gz", custom_checkout, timeout_seconds=17
    )
    assert archive == str(custom_checkout / "custom.tar.gz")
    assert calls[-1] == (
        "ftp://example.org/pub/custom.tar.gz",
        17,
        DEFAULT_HTTP_USER_AGENT,
    )


def test_download_and_extract_archive_retries_ftp_with_browser_user_agent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(HTTP_USER_AGENT_ENV_VAR, raising=False)
    checkout = tmp_path / "source"
    payload = b"ftp browser fallback"
    calls = []

    class FakeFtpResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, timeout):
        calls.append(url.get_header("User-agent"))
        if len(calls) == 1:
            raise URLError("temporary FTP failure")
        return FakeFtpResponse(payload)

    monkeypatch.setattr("galaxy_toolsmith.http_client.urlrequest.urlopen", fake_urlopen)

    archive = _download_and_extract_archive("ftp://example.org/pub/tool.tar.gz", checkout)

    assert archive == str(checkout / "tool.tar.gz")
    assert (checkout / "tool.tar.gz").read_bytes() == payload
    assert calls == [
        DEFAULT_HTTP_USER_AGENT,
        DEFAULT_BROWSER_FALLBACK_USER_AGENT,
    ]
    marker = json.loads((checkout / ".gtsm-http-browser-fallback.json").read_text())
    assert marker == {
        "http_user_agent": DEFAULT_BROWSER_FALLBACK_USER_AGENT,
        "http_user_agent_fallback": "browser",
    }


def test_container_image_quarantine_persists(tmp_path) -> None:
    settings = ExtractionSettings(
        container_cache_dir=tmp_path / "containers",
        container_image_quarantine_seconds=120,
    )
    store = _load_container_image_quarantine_store(settings)
    assert isinstance(store, ContainerImageQuarantineStore)

    preparation = ContainerPreparation(
        ok=False,
        runtime="singularity",
        image="docker.io/example/slow:1.0",
        identifier=str(tmp_path / "containers" / "singularity" / "slow.sif"),
        source="docker-build",
        returncode=124,
        error_text="Command timed out after 300 seconds",
    )
    key = f"singularity:{preparation.identifier}"

    _container_image_quarantine_put(
        store,
        key,
        preparation,
        settings,
        reason="container preparation timed out",
    )

    reloaded = _load_container_image_quarantine_store(settings)
    entry = _container_image_quarantine_get(reloaded, key, settings)

    assert entry is not None
    assert entry["runtime"] == "singularity"
    assert entry["image"] == "docker.io/example/slow:1.0"
    assert entry["returncode"] == 124
    assert entry["reason"] == "container preparation timed out"


def test_download_and_extract_archive_retries_stale_tls(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(HTTP_USER_AGENT_ENV_VAR, raising=False)
    checkout = tmp_path / "source"
    payload = b"legacy source archive"
    calls = []

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size):
            yield payload

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("verify", True), kwargs["headers"]["User-Agent"]))
        if kwargs.get("verify", True):
            raise requests.exceptions.SSLError("certificate expired")
        return FakeResponse()

    monkeypatch.setattr("galaxy_toolsmith.data.corpus.requests.get", fake_get)

    archive = _download_and_extract_archive("https://legacy.example.org/tool.tar.gz", checkout)

    assert archive == str(checkout / "tool.tar.gz")
    assert (checkout / "tool.tar.gz").read_bytes() == payload
    assert calls == [
        ("https://legacy.example.org/tool.tar.gz", True, DEFAULT_HTTP_USER_AGENT),
        (
            "https://legacy.example.org/tool.tar.gz",
            True,
            DEFAULT_BROWSER_FALLBACK_USER_AGENT,
        ),
        ("https://legacy.example.org/tool.tar.gz", False, DEFAULT_HTTP_USER_AGENT),
    ]


def test_extract_source_fields_strips_conda_selector_comments() -> None:
    source_url, source_ref = _extract_source_fields(
        """
source:
  url: "http://hgdownload.cse.ucsc.edu/admin/exe/userApps.v357.src.tgz"  # [linux]
  tag: "release-1.0"  # [linux]
""".strip()
    )

    assert source_url == "http://hgdownload.cse.ucsc.edu/admin/exe/userApps.v357.src.tgz"
    assert source_ref == "release-1.0"


def test_extract_source_fields_handles_conda_source_lists() -> None:
    source_url, source_ref = _extract_source_fields(
        """
source:
  - url: https://example.org/tool-1.0.tar.gz
  - git_rev: release-1.0
""".strip()
    )

    assert source_url == "https://example.org/tool-1.0.tar.gz"
    assert source_ref == "release-1.0"


def test_extract_source_fields_handles_multiline_url_lists() -> None:
    source_url, source_ref = _extract_source_fields(
        """
{% set version = "1.0" %}
source:
  url:
    - https://example.org/tool-{{ version }}.tar.gz
    - https://mirror.example.org/tool-{{ version }}.tar.gz
  tag: "v{{ version }}"
""".strip()
    )

    assert source_url == "https://example.org/tool-{{ version }}.tar.gz"
    assert source_ref == "v{{ version }}"


def test_source_less_recipe_uses_run_dependency_source_provider(monkeypatch, tmp_path) -> None:
    repo = tmp_path / "recipes-repo"
    (repo / "recipes" / "tabix").mkdir(parents=True)
    (repo / "recipes" / "tabix" / "meta.yaml").write_text(
        """
{% set version = "1.11" %}
package:
  name: tabix
  version: {{ version }}
build:
  noarch: generic
requirements:
  run:
    - htslib >=1.9
test:
  commands:
    - test -x "$PREFIX/bin/tabix"
""".strip(),
        encoding="utf-8",
    )
    (repo / "recipes" / "htslib").mkdir(parents=True)
    (repo / "recipes" / "htslib" / "meta.yaml").write_text(
        """
{% set version = "1.22.1" %}
package:
  name: htslib
  version: {{ version }}
source:
  url: https://example.org/htslib-{{ version }}.tar.bz2
""".strip(),
        encoding="utf-8",
    )

    def fake_checkout(**kwargs):
        checkout_dir = kwargs["checkout_dir"]
        checkout_dir.mkdir(parents=True, exist_ok=True)
        (checkout_dir / "tabix.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
        return str(checkout_dir), ""

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._checkout_bioconda_source", fake_checkout)

    mapping = _resolve_recipe_source_mapping(
        package="tabix",
        required_version="1.11",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=repo,
        ref="HEAD",
    )

    assert mapping["package"] == "tabix"
    assert mapping["source_url"] == ""
    assert mapping["source_provider_package"] == "htslib"
    assert mapping["source_provider_source_url"] == "https://example.org/htslib-1.22.1.tar.bz2"
    assert mapping["source_provider_reason"] == "source_less_run_dependency"
    assert mapping["source_checkout"].endswith("htslib-1.22.1")


def test_extract_recipe_run_dependency_names_skips_runtime_support_packages() -> None:
    names = _extract_recipe_run_dependency_names(
        """
requirements:
  run:
    - python >=3.10
    - r-base >=4.3
    - openjdk >=17
    - htslib >=1.22
    - zlib
    - samtools
""".strip()
    )

    assert names == ["htslib", "samtools"]
    assert _is_source_denylisted_package("python")
    assert _is_source_denylisted_package("r-base")


def test_resolve_bioconda_source_skips_direct_runtime_support_packages(
    monkeypatch, tmp_path
) -> None:
    def fail_resolve(*args, **kwargs):
        raise AssertionError("runtime support packages should not resolve source recipes")

    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._resolve_recipe_source_mapping", fail_resolve
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._resolve_conda_forge_source_mapping", fail_resolve
    )

    mappings = _resolve_bioconda_source_mappings(
        package_names=["python", "r-base", "openjdk"],
        requirement_versions={"python": "3.9.23", "r-base": "4.4.0", "openjdk": "17"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=tmp_path / "bioconda-recipes",
        source_result_cache={},
        source_result_cache_lock=None,
    )

    assert [mapping["package"] for mapping in mappings] == ["python", "r-base", "openjdk"]
    assert {mapping["source_channel"] for mapping in mappings} == {"runtime"}
    assert {mapping["recipe_selection_reason"] for mapping in mappings} == {
        "source_package_denylisted"
    }
    assert all(mapping["source_checkout"] == "" for mapping in mappings)
    assert all(mapping["source_error"] == "" for mapping in mappings)
    assert (
        _record_failure_items(
            {
                "package_id": "iuc/runtime",
                "tool_id": "runtime",
                "wrapper_path": "runtime.xml",
                "requirement_packages": ["python"],
                "bioconda_sources": [mappings[0]],
                "container_execution": [],
            }
        )
        == []
    )


def test_source_recipe_package_aliases_cover_metawrap_subcommands() -> None:
    assert _source_recipe_package_alias("metawrap-binning") == "metawrap-mg"
    assert _source_recipe_package_alias("metawrap-refinement") == "metawrap-mg"
    assert _source_recipe_package_alias("metawrap-mg") == ""


def test_render_bioconda_source_fields_handles_common_templates() -> None:
    github_meta = """
{% set name = "abricate" %}
{% set user = "tseemann" %}
{% set version = "1.4.0" %}
""".strip()
    source_url, source_ref, error = _render_bioconda_source_fields(
        meta_text=github_meta,
        package="abricate",
        requirement_version="1.4.0",
        recipe_version="1.4.0",
        source_url="https://github.com/{{ user }}/{{ name }}/archive/v{{ version }}.tar.gz",
        source_ref="",
    )
    assert source_url == "https://github.com/tseemann/abricate/archive/v1.4.0.tar.gz"
    assert source_ref == ""
    assert error == ""

    pypi_meta = """
{% set name = "cyvcf2" %}
{% set version = "0.31.4" %}
""".strip()
    source_url, _, error = _render_bioconda_source_fields(
        meta_text=pypi_meta,
        package="cyvcf2",
        requirement_version="0.31.4",
        recipe_version="0.31.4",
        source_url="https://pypi.org/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz",
        source_ref="",
    )
    assert source_url == "https://pypi.org/packages/source/c/cyvcf2/cyvcf2-0.31.4.tar.gz"
    assert error == ""

    source_url, _, error = _render_bioconda_source_fields(
        meta_text='{% set name = "qgraph" %}\n{% set version = "1.9.8" %}',
        package="r-qgraph",
        requirement_version="1.9.8",
        recipe_version="1.9.8",
        source_url="{{ cran_mirror }}/src/contrib/qgraph_{{ version }}.tar.gz",
        source_ref="",
    )
    assert source_url == "https://cran.r-project.org/src/contrib/qgraph_1.9.8.tar.gz"
    assert error == ""

    source_url, source_ref, error = _render_bioconda_source_fields(
        meta_text='{% set version = "3.12.8" %}',
        package="ncbi-amrfinderplus",
        requirement_version="3.12.8",
        recipe_version="3.12.8",
        source_url="https://github.com/ncbi/amr.git",
        source_ref="amrfinder_v{{ version }}",
    )
    assert source_url == "https://github.com/ncbi/amr.git"
    assert source_ref == "amrfinder_v3.12.8"
    assert error == ""


def test_render_bioconda_source_fields_resolves_set_expressions() -> None:
    source_url, _, error = _render_bioconda_source_fields(
        meta_text='{% set version = "2.7.3" %}\n{% set ver = version|replace(".", "_") %}',
        package="expat",
        requirement_version="2.7.3",
        recipe_version="2.7.3",
        source_url="https://github.com/libexpat/libexpat/releases/download/R_{{ ver }}/expat-{{ version }}.tar.xz",
        source_ref="",
    )
    assert source_url == "https://github.com/libexpat/libexpat/releases/download/R_2_7_3/expat-2.7.3.tar.xz"
    assert error == ""

    source_url, _, error = _render_bioconda_source_fields(
        meta_text=(
            '{% set version = "2.62.5" %}\n'
            "{% set version_majmin = version.rsplit('.', 1)[0] %}"
        ),
        package="librsvg",
        requirement_version="2.62.5",
        recipe_version="2.62.5",
        source_url="https://download.gnome.org/sources/librsvg/{{ version_majmin }}/librsvg-{{ version }}.tar.xz",
        source_ref="",
    )
    assert source_url == "https://download.gnome.org/sources/librsvg/2.62/librsvg-2.62.5.tar.xz"
    assert error == ""

    source_url, _, error = _render_bioconda_source_fields(
        meta_text=(
            '{% set version = "3.40.0" %}\n'
            '{% set year = "2022" %}\n'
            '{% set version_split = version.split(".") %}\n'
            "{% set major = version_split[0] %}\n"
            "{% set minor = version_split[1]|int * 10 %}\n"
            "{% set bugfix = version_split[2]|int * 100 %}\n"
            '{% set version_coded=(major ~ (("%03d" % minor)|string) ~ '
            '(("%03d" % bugfix)|string)) %}'
        ),
        package="sqlite",
        requirement_version="3.40.0",
        recipe_version="3.40.0",
        source_url="https://www.sqlite.org/{{ year }}/sqlite-autoconf-{{ version_coded }}.tar.gz",
        source_ref="",
    )
    assert source_url == "https://www.sqlite.org/2022/sqlite-autoconf-3400000.tar.gz"
    assert error == ""


def test_extract_recipe_version_strips_inline_comments() -> None:
    version = _extract_recipe_version(
        '{% set version = "0.53" # Keep without patch release decimal. %}'
    )

    assert version == "0.53"


def test_conda_forge_feedstock_variants_include_split_package_aliases() -> None:
    assert "matplotlib" in _conda_forge_feedstock_variants("matplotlib-base")
    assert "seaborn" in _conda_forge_feedstock_variants("seaborn-base")
    assert "sqlite" in _conda_forge_feedstock_variants("libsqlite")


def test_resolve_bioconda_source_tries_multiline_source_url_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "multisource"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "1.0" %}
source:
  url:
    - https://primary.example.org/multisource-{{ version }}.tar.gz
    - https://mirror.example.org/multisource-{{ version }}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fake_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        if source_url.startswith("https://primary.example.org/"):
            raise RuntimeError("primary failed")
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="multisource",
        required_version="1.0",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert calls == [
        "https://primary.example.org/multisource-1.0.tar.gz",
        "https://mirror.example.org/multisource-1.0.tar.gz",
    ]
    assert mapping["source_url"] == "https://mirror.example.org/multisource-1.0.tar.gz"
    assert mapping["source_checkout"].endswith("source.tar.gz")
    assert len(mapping["source_attempts"]) == 2


def test_resolve_bioconda_source_replaces_binary_artifact_with_source_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "jartool"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "1.0" %}
source:
  url: https://github.com/example/jartool/releases/download/{{ version }}/jartool.jar
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )

    def fake_download(source_url, checkout_dir, **kwargs):
        if source_url.endswith(".jar"):
            return str(checkout_dir / "jartool.jar")
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="jartool",
        required_version="1.0",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert mapping["source_artifact_url"].endswith("/jartool.jar")
    assert mapping["source_url"] == "https://github.com/example/jartool/archive/refs/tags/1.0.tar.gz"
    assert mapping["source_fallback_reason"] == "binary_artifact_source_fallback"
    assert mapping["source_checkout"].endswith("source.tar.gz")


def test_resolve_bioconda_source_replaces_extensionless_binary_artifact_with_source_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "vg"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "1.23.0" %}
source:
  url: https://github.com/vgteam/vg/releases/download/v{{ version }}/vg
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )

    def fake_download(source_url, checkout_dir, **kwargs):
        checkout_dir.mkdir(parents=True, exist_ok=True)
        if source_url.endswith("/vg"):
            path = checkout_dir / "vg"
            path.write_bytes(b"\x7fELF\x00binary")
            return str(path)
        path = checkout_dir / "source.tar.gz"
        path.write_bytes(b"source")
        return str(path)

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="vg",
        required_version="1.23.0",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert mapping["source_artifact_url"].endswith("/v1.23.0/vg")
    assert mapping["source_url"] == "https://github.com/vgteam/vg/archive/refs/tags/v1.23.0.tar.gz"
    assert mapping["source_fallback_reason"] == "binary_artifact_source_fallback"
    assert mapping["source_checkout"].endswith("source.tar.gz")


def test_resolve_bioconda_source_tries_github_source_fallback_after_release_404(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "bax2bam"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "0.0.11" %}
source:
  url: https://github.com/PacificBiosciences/bax2bam/releases/download/{{ version }}/bax2bam
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fake_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        if source_url.endswith("/bax2bam"):
            raise RuntimeError("404")
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="bax2bam",
        required_version="0.0.11",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert calls == [
        "https://github.com/PacificBiosciences/bax2bam/releases/download/0.0.11/bax2bam",
        "https://github.com/PacificBiosciences/bax2bam/archive/refs/tags/0.0.11.tar.gz",
    ]
    assert mapping["source_artifact_url"].endswith("/0.0.11/bax2bam")
    assert mapping["source_url"] == (
        "https://github.com/PacificBiosciences/bax2bam/archive/refs/tags/0.0.11.tar.gz"
    )
    assert mapping["source_fallback_reason"] == "binary_artifact_source_fallback"
    assert mapping["source_checkout"].endswith("source.tar.gz")


def test_resolve_bioconda_source_tries_bioconductor_archive_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "bioconductor-dropletutils"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "1.10.0" %}
source:
  url: https://bioconductor.org/packages/3.12/bioc/src/contrib/DropletUtils_{{ version }}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fake_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        if "/Archive/DropletUtils/" not in source_url:
            raise RuntimeError("404")
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="bioconductor-dropletutils",
        required_version="1.10.0",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert calls == [
        "https://bioconductor.org/packages/3.12/bioc/src/contrib/DropletUtils_1.10.0.tar.gz",
        "https://bioconductor.org/packages/3.12/bioc/src/contrib/Archive/DropletUtils/DropletUtils_1.10.0.tar.gz",
    ]
    assert mapping["source_url"].endswith(
        "/Archive/DropletUtils/DropletUtils_1.10.0.tar.gz"
    )
    assert mapping["source_fallback_reason"] == "bioconductor_archive_fallback"
    assert mapping["source_checkout"].endswith("source.tar.gz")


def test_resolve_bioconda_source_tries_gnu_mirror_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "datamash"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "1.9" %}
source:
  url: http://ftpmirror.gnu.org/datamash/datamash-{{ version }}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fake_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        if source_url.startswith("http://ftpmirror.gnu.org/"):
            raise RuntimeError("403")
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="datamash",
        required_version="1.9",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert calls == [
        "http://ftpmirror.gnu.org/datamash/datamash-1.9.tar.gz",
        "https://ftp.gnu.org/gnu/datamash/datamash-1.9.tar.gz",
    ]
    assert mapping["source_url"] == "https://ftp.gnu.org/gnu/datamash/datamash-1.9.tar.gz"
    assert mapping["source_fallback_reason"] == "gnu_mirror_fallback"
    assert mapping["source_checkout"].endswith("source.tar.gz")


def test_resolve_bioconda_source_uses_versioned_recipe_path(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(recipes_repo)
    recipe_dir = recipes_repo / "recipes" / "qiime" / "1.9.1"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "1.9.1" %}
source:
  url: https://example.org/qiime-{{ version }}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    _git(recipes_repo, "add", "recipes/qiime/1.9.1/meta.yaml")
    _git(recipes_repo, "commit", "-m", "qiime 1.9.1")
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fake_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="qiime",
        required_version="1.9.1",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert calls == ["https://example.org/qiime-1.9.1.tar.gz"]
    assert mapping["recipe_path"].endswith("recipes/qiime/1.9.1/meta.yaml")
    assert mapping["recipe_selection_reason"] == "exact"
    assert mapping["source_url"] == "https://example.org/qiime-1.9.1.tar.gz"


def test_resolve_bioconda_source_selects_versioned_blast_legacy_recipe(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(recipes_repo)
    for version in ("2.2.19", "2.2.26"):
        recipe_dir = recipes_repo / "recipes" / "blast-legacy" / version
        recipe_dir.mkdir(parents=True)
        (recipe_dir / "meta.yaml").write_text(
            f"""
{{% set version = "{version}" %}}
source:
  url: https://example.org/blast-legacy-{{{{ version }}}}.tar.gz
""".strip(),
            encoding="utf-8",
        )
    _git(recipes_repo, "add", "recipes/blast-legacy")
    _git(recipes_repo, "commit", "-m", "blast legacy versions")
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fake_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="blast-legacy",
        required_version="2.2.26",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert calls == ["https://example.org/blast-legacy-2.2.26.tar.gz"]
    assert mapping["recipe_path"].endswith("recipes/blast-legacy/2.2.26/meta.yaml")
    assert mapping["recipe_selection_reason"] == "exact"
    assert mapping["recipe_version"] == "2.2.26"


def test_conda_forge_feedstock_variants_maps_libcurl_to_curl() -> None:
    variants = _conda_forge_feedstock_variants("libcurl")

    assert variants[0] == "curl"
    assert "libcurl" in variants


def test_failed_source_fallback_candidates_cover_common_legacy_hosts() -> None:
    cran = _failed_source_fallback_candidates(
        "https://cran.r-project.org/src/contrib/ExomeDepth_1.1.15.tar.gz"
    )
    assert (
        "https://cran.r-project.org/src/contrib/Archive/ExomeDepth/ExomeDepth_1.1.15.tar.gz",
        "cran_archive_fallback",
    ) in cran

    sourceforge = _failed_source_fallback_candidates(
        "https://sourceforge.net/projects/mapsembler2/files/mapsembler2_2.2.4.zip/download"
    )
    assert (
        "https://downloads.sourceforge.net/project/mapsembler2/mapsembler2_2.2.4.zip",
        "sourceforge_mirror_fallback",
    ) in sourceforge

    encoded = _failed_source_fallback_candidates(
        "https://example.org/downloads/tool source 1.0.tar.gz"
    )
    assert (
        "https://example.org/downloads/tool%20source%201.0.tar.gz",
        "url_encoded_source_fallback",
    ) in encoded

    mirror = _failed_source_fallback_candidates(
        "https://software-ab.informatik.uni-tuebingen.de/download/megan6/tool.tar.gz"
    )
    assert (
        "https://software-ab.cs.uni-tuebingen.de/download/megan6/tool.tar.gz",
        "domain_mirror_fallback",
    ) in mirror

    pypi = _failed_source_fallback_candidates(
        "https://files.pythonhosted.org/packages/ab/cd/panta-1.0.1-py3-none-any.whl"
    )
    assert (
        "https://pypi.org/packages/source/p/panta/panta-1.0.1.tar.gz",
        "binary_artifact_source_fallback",
    ) in pypi
    assert all("panta-1.0.1-py3-none.tar.gz" not in url for url, _ in pypi)


def test_resolve_bioconda_source_tries_cran_archive_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "r-exomedepth"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set version = "1.1.15" %}
source:
  url: https://cran.r-project.org/src/contrib/ExomeDepth_{{ version }}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fake_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        if "/Archive/ExomeDepth/" not in source_url:
            raise RuntimeError("404")
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mapping = _resolve_recipe_source_mapping(
        package="r-exomedepth",
        required_version="1.1.15",
        channel="bioconda",
        settings=ExtractionSettings(cache_root=tmp_path / "cache", bioconda_checkout_sources=True),
        recipes_repo=recipes_repo,
        ref="HEAD",
        source_result_cache={},
    )

    assert calls == [
        "https://cran.r-project.org/src/contrib/ExomeDepth_1.1.15.tar.gz",
        "https://cran.r-project.org/src/contrib/Archive/ExomeDepth/ExomeDepth_1.1.15.tar.gz",
    ]
    assert mapping["source_url"].endswith(
        "/Archive/ExomeDepth/ExomeDepth_1.1.15.tar.gz"
    )
    assert mapping["source_fallback_reason"] == "cran_archive_fallback"
    assert mapping["source_checkout"].endswith("source.tar.gz")


def test_resolve_bioconda_source_skips_unresolved_templates(monkeypatch, tmp_path) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "templated"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set name = "templated" %}
{% set version = "1.0" %}
source:
  url: https://github.com/{{ user }}/{{ name }}/archive/v{{ version }}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    events = []
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status",
        lambda payload, status_log_path=None: events.append(payload),
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("unresolved source templates should not be downloaded")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fail_download)

    mappings = _resolve_bioconda_source_mappings(
        package_names=["templated"],
        requirement_versions={"templated": "1.0"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=recipes_repo,
        source_result_cache={},
        source_result_cache_lock=None,
    )

    assert mappings[0]["source_checkout"] == ""
    assert "unresolved_template" in mappings[0]["source_error"]
    assert any(event["status"] == "bioconda-source-skipped-template" for event in events)


def test_resolve_bioconda_source_reuses_failed_result(monkeypatch, tmp_path) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    recipe_dir = recipes_repo / "recipes" / "repeat"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "meta.yaml").write_text(
        """
{% set name = "repeat" %}
{% set version = "1.0" %}
source:
  url: https://example.org/{{ name }}-{{ version }}.tar.gz
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    calls = []

    def fail_download(source_url, checkout_dir, **kwargs):
        calls.append(source_url)
        raise RuntimeError("boom")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fail_download)
    source_result_cache = {}

    for _ in range(2):
        mappings = _resolve_bioconda_source_mappings(
            package_names=["repeat"],
            requirement_versions={"repeat": "1.0"},
            settings=ExtractionSettings(
                bioconda_checkout_sources=True,
                cache_root=tmp_path / "source-cache",
            ),
            recipes_repo=recipes_repo,
            source_result_cache=source_result_cache,
            source_result_cache_lock=None,
        )
        assert mappings[0]["source_checkout"] == ""
        assert mappings[0]["source_error"] == "boom"

    assert calls == ["https://example.org/repeat-1.0.tar.gz"]


def test_resolve_bioconda_source_selects_exact_historical_recipe(monkeypatch, tmp_path) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(recipes_repo)
    _commit_recipe(recipes_repo, "historytool", "1.0")
    _commit_recipe(recipes_repo, "historytool", "2.0")
    head_before = _git(recipes_repo, "rev-parse", "HEAD")

    events = []
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status",
        lambda payload, status_log_path=None: events.append(payload),
    )
    downloads = []

    def fake_download(source_url, checkout_dir, **kwargs):
        downloads.append((source_url, checkout_dir.name))
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mappings = _resolve_bioconda_source_mappings(
        package_names=["historytool"],
        requirement_versions={"historytool": "1.0"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=recipes_repo,
        source_result_cache={},
        source_result_cache_lock=None,
        recipe_selection_cache={},
        recipe_selection_cache_lock=None,
    )

    assert _git(recipes_repo, "rev-parse", "HEAD") == head_before
    assert mappings[0]["recipe_version"] == "1.0"
    assert mappings[0]["recipe_selection_reason"] == "exact"
    assert mappings[0]["source_url"] == "https://example.org/historytool-1.0.tar.gz"
    assert downloads == [("https://example.org/historytool-1.0.tar.gz", "historytool-1.0")]
    assert any(
        event["status"] == "bioconda-recipe-selected"
        and event["package"] == "historytool"
        and event["selection_reason"] == "exact"
        for event in events
    )


def test_resolve_bioconda_source_tries_lowercase_recipe_name(monkeypatch, tmp_path) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(recipes_repo)
    _commit_recipe(recipes_repo, "ampligone", "2.0.1")

    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._download_and_extract_archive",
        lambda source_url, checkout_dir, **kwargs: str(checkout_dir / "source.tar.gz"),
    )

    mappings = _resolve_bioconda_source_mappings(
        package_names=["AmpliGone"],
        requirement_versions={"AmpliGone": "2.0.1"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=recipes_repo,
        source_result_cache={},
        source_result_cache_lock=None,
        recipe_selection_cache={},
        recipe_selection_cache_lock=None,
    )

    assert mappings[0]["recipe_version"] == "2.0.1"
    assert mappings[0]["recipe_path"].endswith("recipes/ampligone/meta.yaml")
    assert mappings[0]["source_error"] == ""


def test_resolve_bioconda_source_falls_back_to_conda_forge_when_recipe_missing(
    monkeypatch,
    tmp_path,
) -> None:
    bioconda_repo = tmp_path / "bioconda-recipes"
    bioconda_repo.mkdir()
    feedstock_repo = tmp_path / "numpy-feedstock"
    _init_recipe_repo(feedstock_repo)
    _commit_feedstock_recipe(feedstock_repo, "numpy", "1.23.3")
    events = []
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status",
        lambda payload, status_log_path=None: events.append(payload),
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._ensure_conda_forge_feedstock_repo",
        lambda cache_root, package, settings: (feedstock_repo, "", "numpy"),
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._download_and_extract_archive",
        lambda source_url, checkout_dir, **kwargs: str(checkout_dir / "source.tar.gz"),
    )

    mappings = _resolve_bioconda_source_mappings(
        package_names=["numpy"],
        requirement_versions={"numpy": "1.23.3"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=bioconda_repo,
        source_result_cache={},
        source_result_cache_lock=None,
        recipe_selection_cache={},
        recipe_selection_cache_lock=None,
    )

    assert mappings[0]["source_channel"] == "conda-forge"
    assert mappings[0]["fallback_from_channel"] == "bioconda"
    assert mappings[0]["recipe_path"] == "HEAD:recipe/meta.yaml"
    assert mappings[0]["source_checkout"].endswith("source.tar.gz")
    assert any(event["status"] == "conda-forge-recipe-selected" for event in events)


def test_resolve_bioconda_source_falls_back_to_conda_forge_when_download_fails(
    monkeypatch,
    tmp_path,
) -> None:
    bioconda_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(bioconda_repo)
    _commit_recipe(bioconda_repo, "gawk", "5.0.1")
    feedstock_repo = tmp_path / "gawk-feedstock"
    _init_recipe_repo(feedstock_repo)
    _commit_feedstock_recipe(
        feedstock_repo,
        "gawk",
        "5.0.1",
        source_url="https://conda-forge.example.org/gawk-5.0.1.tar.gz",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._ensure_conda_forge_feedstock_repo",
        lambda cache_root, package, settings: (feedstock_repo, "", "gawk"),
    )

    def fake_download(source_url, checkout_dir, **kwargs):
        if source_url.startswith("https://example.org/"):
            raise RuntimeError("403")
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

    mappings = _resolve_bioconda_source_mappings(
        package_names=["gawk"],
        requirement_versions={"gawk": "5.0.1"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=bioconda_repo,
        source_result_cache={},
        source_result_cache_lock=None,
        recipe_selection_cache={},
        recipe_selection_cache_lock=None,
    )

    assert mappings[0]["source_channel"] == "conda-forge"
    assert mappings[0]["source_url"] == "https://conda-forge.example.org/gawk-5.0.1.tar.gz"
    assert mappings[0]["source_checkout"].endswith("source.tar.gz")
    assert mappings[0]["fallback_from_source_error"] == "403"


def test_resolve_bioconda_source_replaces_weak_bioconda_with_exact_conda_forge(
    monkeypatch,
    tmp_path,
) -> None:
    bioconda_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(bioconda_repo)
    _commit_recipe(bioconda_repo, "xgboost", "0.6a2")
    feedstock_repo = tmp_path / "xgboost-feedstock"
    _init_recipe_repo(feedstock_repo)
    _commit_feedstock_recipe(
        feedstock_repo,
        "xgboost",
        "3.0.4",
        source_url="https://conda-forge.example.org/xgboost-3.0.4.tar.gz",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._ensure_conda_forge_feedstock_repo",
        lambda cache_root, package, settings: (feedstock_repo, "", "xgboost"),
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._download_and_extract_archive",
        lambda source_url, checkout_dir, **kwargs: str(checkout_dir / "source.tar.gz"),
    )

    mappings = _resolve_bioconda_source_mappings(
        package_names=["xgboost"],
        requirement_versions={"xgboost": "3.0.4"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=bioconda_repo,
        source_result_cache={},
        source_result_cache_lock=None,
        recipe_selection_cache={},
        recipe_selection_cache_lock=None,
    )

    assert mappings[0]["source_channel"] == "conda-forge"
    assert mappings[0]["recipe_version"] == "3.0.4"
    assert mappings[0]["source_confidence"] == "exact"
    assert mappings[0]["source_version_match"] == "exact"
    assert mappings[0]["fallback_from_recipe_selection_reason"] == "closest_major"


def test_conda_forge_feedstock_missing_lookup_is_cached_concurrently(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    def fake_ensure_conda_recipe_repo(**kwargs):
        calls.append(kwargs["repo_name"])
        time.sleep(0.05)
        return None, "missing"

    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._ensure_conda_recipe_repo",
        fake_ensure_conda_recipe_repo,
    )

    def lookup():
        return _ensure_conda_forge_feedstock_repo(
            tmp_path / "source-cache",
            "missingpkg",
            ExtractionSettings(),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: lookup(), range(8)))

    assert results == [(None, "missing", "missingpkg")] * 8
    assert calls == ["missingpkg-feedstock"]


def test_select_recipe_snapshot_fallback_ranking() -> None:
    def snapshot(version: str) -> BiocondaRecipeSnapshot:
        return BiocondaRecipeSnapshot(
            package="ranked",
            recipe_path=f"{version}:recipes/ranked/meta.yaml",
            meta_text=f'{{% set version = "{version}" %}}',
            recipe_version=version,
        )

    selected = _select_recipe_snapshot_from_candidates(
        package="ranked",
        required_version="1.5",
        candidates=[snapshot("2.0"), snapshot("1.3"), snapshot("1.7")],
        scanned_commits=3,
    )
    assert selected.recipe_version == "1.7"
    assert selected.selection_reason == "same_major_newer"

    selected = _select_recipe_snapshot_from_candidates(
        package="ranked",
        required_version="1.5",
        candidates=[snapshot("2.0"), snapshot("1.3"), snapshot("1.4")],
        scanned_commits=3,
    )
    assert selected.recipe_version == "1.4"
    assert selected.selection_reason == "same_major_older"

    selected = _select_recipe_snapshot_from_candidates(
        package="ranked",
        required_version="2.0",
        candidates=[snapshot("3.0"), snapshot("1.9")],
        scanned_commits=2,
    )
    assert selected.recipe_version == "1.9"
    assert selected.selection_reason == "closest_major"


def test_version_matching_keeps_release_suffixes_distinct() -> None:
    def snapshot(version: str) -> BiocondaRecipeSnapshot:
        return BiocondaRecipeSnapshot(
            package="openssl",
            recipe_path=f"{version}:recipes/openssl/meta.yaml",
            meta_text=f'{{% set version = "{version}" %}}',
            recipe_version=version,
        )

    selected = _select_recipe_snapshot_from_candidates(
        package="openssl",
        required_version="1.1.1s",
        candidates=[snapshot("1.1.1a")],
        scanned_commits=1,
    )
    assert selected.selection_reason == "same_numeric_variant"
    assert _source_version_match_status("1.1.1s", "1.1.1a") == "mismatch"
    assert _source_version_match_status("1.8_4", "1.8-4") == "numeric_equivalent"
    assert _source_version_match_status("1.0.0", "1") == "numeric_equivalent"


def test_resolve_bioconda_source_uses_distinct_cache_dir_for_version_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(recipes_repo)
    _commit_recipe(recipes_repo, "fallbacktool", "1.3")
    _commit_recipe(recipes_repo, "fallbacktool", "1.7")
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    checkout_names = []

    def fake_download(source_url, checkout_dir, **kwargs):
        checkout_names.append(checkout_dir.name)
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._resolve_conda_forge_source_mapping",
        lambda **kwargs: {"source_checkout": "", "source_url": ""},
    )

    mappings = _resolve_bioconda_source_mappings(
        package_names=["fallbacktool"],
        requirement_versions={"fallbacktool": "1.5"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=recipes_repo,
        source_result_cache={},
        source_result_cache_lock=None,
        recipe_selection_cache={},
        recipe_selection_cache_lock=None,
    )

    assert mappings[0]["recipe_version"] == "1.7"
    assert mappings[0]["recipe_selection_reason"] == "same_major_newer"
    assert mappings[0]["source_confidence"] == "near"
    assert mappings[0]["source_version_match"] == "mismatch"
    assert checkout_names == ["fallbacktool-required-1.5--recipe-1.7"]


def test_resolve_bioconda_source_labels_cross_major_fallback_as_weak(
    monkeypatch,
    tmp_path,
) -> None:
    recipes_repo = tmp_path / "bioconda-recipes"
    _init_recipe_repo(recipes_repo)
    _commit_recipe(recipes_repo, "fallbacktool", "1.9")
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus.emit_status", lambda payload, status_log_path=None: None
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._download_and_extract_archive",
        lambda source_url, checkout_dir, **kwargs: str(checkout_dir / "source.tar.gz"),
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._resolve_conda_forge_source_mapping",
        lambda **kwargs: {"source_checkout": "", "source_url": ""},
    )

    mappings = _resolve_bioconda_source_mappings(
        package_names=["fallbacktool"],
        requirement_versions={"fallbacktool": "2.0"},
        settings=ExtractionSettings(
            bioconda_checkout_sources=True,
            cache_root=tmp_path / "source-cache",
        ),
        recipes_repo=recipes_repo,
        source_result_cache={},
        source_result_cache_lock=None,
        recipe_selection_cache={},
        recipe_selection_cache_lock=None,
    )

    assert mappings[0]["recipe_version"] == "1.9"
    assert mappings[0]["recipe_selection_reason"] == "closest_major"
    assert mappings[0]["source_confidence"] == "weak"
    assert mappings[0]["source_version_match"] == "mismatch"


def test_infer_command_signatures_filters_wrapper_setup_and_files() -> None:
    assert _infer_command_signatures("mkdir ./tabular\n./tabular --option")[0] == ""
    assert _infer_command_signatures("cp annotatemyids_script out_rscript")[0] == ""
    assert _infer_command_signatures("ln input_bam localbam.bam")[0] == ""
    assert _infer_command_signatures("mkdir -p 'output_dir/'\noutput_dir")[0] == ""
    assert _infer_command_signatures("awk -f helper.awk '$input' > '$output'")[0] == ""
    assert _script_text_command_signatures("awk '{print $1}' '$input'\n", language="bash") == []
    assert _infer_command_signatures("set -e\nbowtie2 '$input'")[0] == "bowtie2"
    assert (
        _infer_command_signatures("if [ -s '$input' ]; then augustus '$input'; fi")[0] == "augustus"
    )

    primary, subcommands, _ = _infer_command_signatures("autoBIGS database_origin.bigsdb")
    assert primary == "autoBIGS"
    assert subcommands == []

    assert (
        _infer_command_signatures("python __tool_directory__/fastaregexfinder.py '$input'")[0] == ""
    )
    assert _infer_command_signatures("Rscript __tool_directory__/script.R '$input'")[0] == ""
    assert _infer_command_signatures("java __tool_directory__/FourColorPlot.jar '$input'")[0] == ""

    primary, subcommands, _ = _infer_command_signatures("samtools view -h '$input' > '$output'")
    assert primary == "samtools"
    assert subcommands == ["view"]

    candidates = _script_text_command_signatures(
        "bgzip -c '$input_file' > input.vcf.gz &&\n"
        "bcftools index input.vcf.gz &&\n"
        "bcftools annotate input.vcf.gz > '$output'",
        language="bash",
    )
    assert ("bgzip", "bcftools") not in candidates
    assert ("bcftools", "annotate") in candidates

    primary, subcommands, _ = _infer_command_signatures(
        "python -m amas.AMAS\nsummary\n--in-files '$input'"
    )
    assert primary == "python -m amas.AMAS"
    assert subcommands == ["summary"]

    primary, subcommands, _ = _infer_command_signatures(
        "GAMMA.py '$input_fasta' '$input_db' gamma_out"
    )
    assert primary == "GAMMA.py"
    assert subcommands == []

    anndata_sed = (
        "cat '$input' | "
        "sed -r '1 s|AnnData object with (.+) = (.*)|\\1: \\2|g' > '$output'"
    )
    assert _infer_command_signatures(anndata_sed)[0] == ""

    recentrifuge_command = "#*\nMake a directory for reports.\n*#\nrecentrifuge '$input'"
    assert _infer_command_signatures(recentrifuge_command)[0] == "recentrifuge"


def test_infer_command_signatures_ignores_quoted_config_literals() -> None:
    command = (
        "printf '%s\\n' \\\n"
        "  'process {' \\\n"
        "  '  resourceLimits = [' \\\n"
        "  '  ]' > galaxy.conf\n"
        "&& nextflow run dessimozlab/FastOMA -r v1.0"
    )

    primary, subcommands, _ = _infer_command_signatures(command)

    assert primary == "nextflow"
    assert subcommands == ["run"]


def test_infer_command_signatures_ignores_multiline_awk_body() -> None:
    command = (
        "awk -F $'\\t' ' {\n"
        "f = \"Gene\";\n"
        "print f, i, o; } ' OFS=$'\\t' '$input' > '$output'\n"
    )

    assert _infer_command_signatures(command)[0] == ""


def test_extract_source_command_hints_reads_archives_and_nested_roots(tmp_path) -> None:
    source_root = tmp_path / "GAMMA-2.2"
    source_root.mkdir()
    (source_root / "GAMMA.py").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (source_root / "GAMMA-S.py").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (source_root / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='GAMMA')\n", encoding="utf-8"
    )

    archive_path = tmp_path / "v2.2.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_root, arcname="GAMMA-2.2")

    archive_hints = _extract_source_command_hints(str(archive_path), "GAMMA")
    assert "GAMMA.py" in archive_hints
    assert "GAMMA-S.py" in archive_hints
    assert "pip" not in archive_hints
    assert "install" not in archive_hints

    checkout_root = tmp_path / "checkout"
    nested_root = checkout_root / "GAMMA-2.2"
    nested_root.mkdir(parents=True)
    (nested_root / "GAMMA.py").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    directory_hints = _extract_source_command_hints(str(checkout_root), "GAMMA")
    assert "GAMMA.py" in directory_hints

    rcf_root = tmp_path / "recentrifuge-1.16.1"
    rcf_root.mkdir()
    rcf_path = rcf_root / "rcf"
    rcf_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    rcf_path.chmod(0o755)
    (rcf_root / "README.md").write_text("documentation\n", encoding="utf-8")
    (rcf_root / "Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")
    rcf_archive_path = tmp_path / "recentrifuge.tar.gz"
    with tarfile.open(rcf_archive_path, "w:gz") as archive:
        archive.add(rcf_root, arcname="recentrifuge-1.16.1")

    recentrifuge_hints = _extract_source_command_hints(str(rcf_archive_path), "recentrifuge")
    assert "rcf" in recentrifuge_hints
    assert "Makefile" not in recentrifuge_hints


def test_record_help_commands_augments_hints_from_source_checkout(tmp_path) -> None:
    rcf_root = tmp_path / "recentrifuge-1.16.1"
    rcf_root.mkdir()
    rcf_path = rcf_root / "rcf"
    rcf_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    rcf_path.chmod(0o755)
    rcf_archive_path = tmp_path / "recentrifuge.tar.gz"
    with tarfile.open(rcf_archive_path, "w:gz") as archive:
        archive.add(rcf_root, arcname="recentrifuge-1.16.1")

    record = ToolRecord(
        shed_name="recentrifuge",
        requirement_packages=["recentrifuge"],
        selected_container="quay.io/biocontainers/recentrifuge:1.16.1--pyhdfd78af_0",
        command_text="rcf -n '$database' -f '$input'",
        bioconda_sources=[
            {
                "package": "recentrifuge",
                "command_hints": ["pip"],
                "source_checkout": str(rcf_archive_path),
            }
        ],
    )

    commands = _record_help_commands(
        record,
        record.selected_container,
        candidate_packages=["recentrifuge"],
    )

    assert commands[0] == "rcf --help"


def test_record_help_commands_prefers_identity_over_helper_commands() -> None:
    record = ToolRecord(
        shed_name="bigscape",
        tool_id="bigscape",
        tool_name="BiG-SCAPE",
        selected_container="quay.io/biocontainers/bigscape:2.0.0--pyhdfd78af_0",
        requirement_packages=["bigscape"],
        command_text="hmmpress profiles.hmm && bigscape --input input --output result",
    )

    commands = _record_help_commands(record, record.selected_container)

    assert commands
    assert commands[0] == "bigscape --help"
    assert "bigscape help" in commands
    assert "bigscape" in commands
    assert not any(command.startswith("hmmpress ") for command in commands)


def test_record_help_commands_prefers_vcfanno_over_tabix_setup_helper() -> None:
    record = ToolRecord(
        shed_name="vcfanno",
        tool_id="vcfanno",
        tool_name="VCFanno",
        selected_container="quay.io/biocontainers/mulled-v2-samtools-tabix-vcfanno:hash-0",
        requirement_packages=["samtools", "tabix", "vcfanno"],
        command_text=(
            "bgzip -c '$annotation' > annotation.vcf.gz && "
            "tabix -p vcf annotation.vcf.gz && "
            "vcfanno config.toml '$input' > '$output'"
        ),
        bioconda_sources=[
            {"package": "tabix", "source_provider_package": "htslib", "command_hints": ["tabix"]},
            {"package": "vcfanno", "command_hints": ["vcfanno"]},
        ],
    )

    plan = _record_help_command_plan(record, record.selected_container)
    commands = [item["command"] for item in plan]

    assert commands[0] == "vcfanno --help"
    assert plan[0]["probe_role"] == "core"
    assert commands.index("vcfanno --help") < commands.index("tabix vcf --help")


def test_record_help_commands_filters_setup_subcommands_when_identity_subcommand_exists() -> None:
    record = ToolRecord(
        shed_name="bcftools",
        tool_id="bcftools_annotate",
        tool_name="bcftools annotate",
        selected_container="quay.io/biocontainers/bcftools:1.22--h3a4d415_1",
        requirement_packages=["bcftools"],
        command_text=(
            "bcftools index '$input' && "
            "bcftools annotate --annotations '$ann' '$input' > '$output'"
        ),
    )

    commands = _record_help_commands(record, record.selected_container)

    assert "bcftools annotate --help" in commands
    assert "bcftools annotate" not in commands
    assert "bcftools index --help" not in commands


def test_record_help_commands_ignores_input_like_subcommand_tokens() -> None:
    record = ToolRecord(
        shed_name="tesseract",
        tool_id="tesseract",
        tool_name="Tesseract OCR",
        selected_container="quay.io/biocontainers/tesseract:5.5.2--h7b50bb2_0",
        requirement_packages=["tesseract"],
        command_text="tesseract img_paths '$output' --psm 6",
    )

    commands = _record_help_commands(record, record.selected_container)

    assert commands[0] == "tesseract --help"
    assert all("img_paths" not in command for command in commands)


def test_record_help_commands_ignores_duplicate_multiline_subcommand_tokens() -> None:
    record = ToolRecord(
        shed_name="trinotate",
        tool_id="trinotate",
        tool_name="Trinotate",
        selected_container="quay.io/biocontainers/trinotate:3.2.2--pl5321hdfd78af_0",
        requirement_packages=["trinotate"],
        command_text="Trinotate\nTrinotate '$database' '$input'",
    )

    commands = _record_help_commands(record, record.selected_container)

    assert commands[0] == "Trinotate --help"
    assert all("Trinotate Trinotate" not in command for command in commands)


def test_record_help_commands_skips_auxiliary_only_candidates_for_api_backed_wrapper() -> None:
    record = ToolRecord(
        shed_name="episcanpy",
        tool_id="episcanpy_build_matrix",
        tool_name="Build AnnData matrix",
        selected_container="quay.io/biocontainers/mulled-v2-episcanpy-tabix:hash-0",
        requirement_packages=["episcanpy", "tabix"],
        command_text=(
            "bgzip -c '$fragments' > fragments.gz && "
            "tabix -p bed fragments.gz && "
            "python '$script_file'"
        ),
        wrapper_configfiles=[
            {
                "name": "script_file",
                "template_kind": "script_template",
                "api_calls": [
                    {"language": "python", "qualified_call": "episcanpy.pp.lazy"},
                ],
            }
        ],
        wrapper_source_summary={"api_backed_wrapper": True, "configfile_api_call_count": 1},
    )

    assert _record_help_commands(record, record.selected_container) == []


def test_record_help_commands_skips_shell_and_filelike_candidates() -> None:
    record = ToolRecord(
        shed_name="augustus",
        tool_id="augustus",
        tool_name="AUGUSTUS",
        selected_container="https://depot.galaxyproject.org/singularity/augustus:3.5.0--pl5321heb9362c_5",
        requirement_packages=["augustus"],
        command_text="if [ -s '$input' ]; then result='$output'; fi",
    )

    assert _record_help_commands(record, record.selected_container) == []


def test_record_help_commands_mines_real_commands_from_wrapper_scripts(tmp_path) -> None:
    helper = tmp_path / "bwa_wrapper.py"
    helper.write_text(
        "import subprocess\n"
        "subprocess.run(['bwa', 'mem', '-t', '4', 'ref.fa', 'reads.fq'])\n",
        encoding="utf-8",
    )
    record = ToolRecord(
        shed_name="bwa",
        tool_id="bwa_wrapper",
        tool_name="BWA wrapper",
        selected_container="quay.io/biocontainers/bwa:0.7.18--h577a1d6_0",
        requirement_packages=["bwa"],
        command_text="python '$__tool_directory__/bwa_wrapper.py' '$input'",
        wrapper_helper_files=[
            {
                "path": str(helper),
                "relative_path": helper.name,
                "extension": ".py",
                "byte_count": helper.stat().st_size,
                "sha256": "test",
                "role_hint": "command_reference",
            }
        ],
    )

    commands = _record_help_commands(record, record.selected_container)

    assert commands[:2] == ["bwa --help", "bwa -h"]
    assert commands.index("bwa --help") < commands.index("bwa mem --help")
    assert "bwa mem --usage" in commands
    assert "bwa mem '-?'" in commands
    assert "bwa mem help" not in commands
    assert "bwa mem" in commands
    assert not any("bwa_wrapper.py" in command for command in commands)


def test_record_help_commands_ignore_argparse_helper_text_for_api_wrappers(tmp_path) -> None:
    helper = tmp_path / "alphagenome_wrapper.py"
    helper.write_text(
        "import argparse\n"
        "from alphagenome.models import dna_client\n"
        "parser = argparse.ArgumentParser(\n"
        "    description='Score genetic variants using AlphaGenome predict_variant()'\n"
        ")\n"
        "parser.add_argument('--output-types', default=['RNA_SEQ'], help='AlphaGenome output')\n"
        "client = dna_client.create(api_key='token')\n",
        encoding="utf-8",
    )
    helper_fields = {
        "path": str(helper),
        "relative_path": helper.name,
        "extension": ".py",
        "byte_count": helper.stat().st_size,
        "sha256": "test",
        "role_hint": "command_reference",
        "api_calls": [
            {"language": "python", "qualified_call": "alphagenome.models.dna_client.create"}
        ],
    }
    record = ToolRecord(
        shed_name="alphagenome",
        tool_id="alphagenome_variant_effect",
        tool_name="AlphaGenome Variant Effect",
        selected_container="quay.io/biocontainers/alphagenome:0.6.1--pyhdfd78af_0",
        requirement_packages=["alphagenome"],
        command_text="python '$__tool_directory__/alphagenome_wrapper.py' '$input'",
        wrapper_helper_files=[helper_fields],
        wrapper_source_summary={"api_backed_wrapper": True, "helper_api_call_count": 1},
    )

    assert _script_text_command_signatures(
        helper.read_text(encoding="utf-8"), language="python", extension=".py"
    ) == []
    assert _record_help_commands(record, record.selected_container) == []
    api_commands = _record_api_validation_commands(record)
    assert api_commands[0]["language"] == "python"
    assert api_commands[0]["checks"] == ["alphagenome.models.dna_client.create"]


def test_python_api_calls_require_explicit_import_aliases() -> None:
    assert _extract_python_configfile_api_calls("li = []\nli.append('x')\n") == []

    calls = _extract_python_configfile_api_calls(
        "import liana as li\n"
        "import scanpy as sc\n"
        "adata = sc.read_h5ad('input.h5ad')\n"
        "li.method.rank_aggregate(adata=adata)\n"
    )

    assert [call["qualified_call"] for call in calls] == [
        "scanpy.read_h5ad",
        "liana.method.rank_aggregate",
    ]


def test_record_help_commands_mines_real_commands_from_script_configfiles() -> None:
    record = ToolRecord(
        shed_name="bedtools",
        tool_id="bedtools_configfile_wrapper",
        tool_name="Bedtools configfile wrapper",
        selected_container="quay.io/biocontainers/bedtools:2.31.1--hf5e1c6e_2",
        requirement_packages=["bedtools"],
        command_text="bash '$script'",
        wrapper_configfiles=[
            {
                "name": "script",
                "filename": "run.sh",
                "extension": ".sh",
                "template_kind": "script_template",
                "content": "bedtools intersect -a '$a' -b '$b' > '$out'\n",
            }
        ],
    )

    commands = _record_help_commands(record, record.selected_container)

    assert commands[:2] == ["bedtools intersect --help", "bedtools intersect -h"]
    assert "bedtools intersect --usage" in commands
    assert "bedtools intersect '-?'" in commands
    assert "bedtools intersect help" in commands
    assert "bedtools intersect" in commands


def test_record_help_commands_prefers_version_command_help_flag() -> None:
    record = ToolRecord(
        shed_name="arriba",
        tool_id="arriba",
        tool_name="Arriba",
        selected_container="quay.io/biocontainers/arriba:2.5.1--h87b9561_0",
        requirement_packages=["arriba"],
        command_text="arriba -x '$input' -o '$output'",
        version_command_text="arriba -h | grep Version | sed 's/^.* //'",
    )

    commands = _record_help_commands(record, record.selected_container)

    assert commands[:2] == ["arriba -h", "arriba --help"]


def test_record_help_commands_preserves_headless_environment_prefix() -> None:
    record = ToolRecord(
        shed_name="bandage",
        tool_id="bandage_image",
        tool_name="Bandage Image",
        selected_container="quay.io/biocontainers/bandage_ng:2022.09--h4ac6f70_2",
        requirement_packages=["bandage_ng"],
        command_text=(
            "export QT_QPA_PLATFORM='offscreen' &&\n"
            "Bandage\n"
            "    image\n"
            "    input.gfa\n"
            "    output.png\n"
        ),
    )

    plan = _record_help_command_plan(record, record.selected_container)

    assert plan[0]["primary"] == "Bandage"
    assert plan[0]["command"] == "QT_QPA_PLATFORM=offscreen Bandage image --help"
    assert all(item["primary"] != "image" for item in plan)


def test_record_help_commands_combines_multiline_cat_subcommands() -> None:
    record = ToolRecord(
        shed_name="cat_bins",
        tool_id="cat_bins",
        tool_name="CAT bins",
        selected_container="quay.io/biocontainers/cat:5.2.3--pyhdfd78af_0",
        requirement_packages=["cat"],
        command_text=(
            "CAT \\\n"
            "    bins -s '$summary' \\\n"
            "    --bin_suffix fa\n"
            "CAT \\\n"
            "    bin -b '$bin' \\\n"
            "    --output '$output'\n"
        ),
    )

    plan = _record_help_command_plan(record, record.selected_container)
    commands = [item["command"] for item in plan]

    assert commands[:2] == ["CAT bins --help", "CAT bins -h"]
    assert "CAT bin --help" in commands
    assert all(item["primary"] != "bins" for item in plan)
    assert all(item["primary"] != "bin" for item in plan)


def test_record_help_commands_skips_configfile_environment_prefix() -> None:
    record = ToolRecord(
        shed_name="varvamp",
        tool_id="varvamp",
        tool_name="varVAMP",
        selected_container="quay.io/biocontainers/varvamp:1.0.0--pyhdfd78af_0",
        requirement_packages=["varvamp"],
        command_text="VARVAMP_CONFIG=custom_config varvamp $mode.m_select",
        wrapper_configfiles=[
            {
                "name": "custom_config",
                "filename": "custom_config",
                "template_kind": "config_template",
                "content": "[general]\n",
            }
        ],
    )

    commands = _record_help_commands(record, record.selected_container)

    assert commands[0] == "varvamp --help"
    assert all(not command.startswith("VARVAMP_CONFIG=") for command in commands)


def test_failure_inventory_ignores_container_failures_recovered_by_later_help() -> None:
    record = {
        "package_id": "iuc/bandage",
        "tool_id": "bandage_image",
        "wrapper_path": "bandage_image.xml",
        "container_execution": [
            {
                "status": "container-command-failed-probe",
                "command": "Bandage image --help",
                "returncode": 127,
                "error_text": "libEGL.so.1: cannot open shared object file",
            },
            {
                "status": "container-command-help",
                "command": "Bandage image --help",
                "returncode": 0,
                "error_text": "",
            },
        ],
        "container_help_text": "$ Bandage image --help\nUsage: Bandage image\n",
        "bioconda_sources": [],
    }

    assert _record_failure_items(record) == []


def test_failure_inventory_dedupes_repeated_runtime_import_crashes() -> None:
    traceback = (
        "Traceback (most recent call last): File \"/usr/local/bin/medaka\", line 7, "
        "in <module> import libmedaka ModuleNotFoundError: No module named '_cffi_backend'"
    )
    record = {
        "package_id": "iuc/medaka",
        "tool_id": "medaka_consensus",
        "wrapper_path": "medaka.xml",
        "container_execution": [
            {
                "status": "container-command-failed-probe",
                "command": "medaka --help",
                "returncode": 1,
                "error_text": traceback,
            },
            {
                "status": "container-command-failed-probe",
                "command": "medaka -h",
                "returncode": 1,
                "error_text": traceback,
            },
        ],
        "container_help_text": "",
        "bioconda_sources": [],
    }

    items = _record_failure_items(record)

    assert [item["category"] for item in items] == ["api_import_failure", "no_container_help"]


def test_failure_inventory_treats_missing_argument_traceback_as_probe_variant() -> None:
    traceback = (
        "Traceback (most recent call last): File \"/usr/local/bin/trim_Ns_DNAnexus.py\", "
        "line 14, in main output_file = sys.argv[2] IndexError: list index out of range"
    )
    record = {
        "package_id": "iuc/trimns",
        "tool_id": "trimns",
        "wrapper_path": "TrimNs.xml",
        "container_execution": [
            {
                "status": "container-command-failed-probe",
                "command": "trim_Ns_DNAnexus.py --help",
                "returncode": 1,
                "error_text": traceback,
            }
        ],
        "container_help_text": "",
        "bioconda_sources": [],
    }

    items = _record_failure_items(record)

    assert items[0]["category"] == "bad_probe_variant"
    assert items[0]["severity"] == "partial"


def test_failure_inventory_suppresses_helper_probe_when_source_docs_exist() -> None:
    traceback = (
        "Traceback (most recent call last): File \"/usr/local/bin/trim_Ns_DNAnexus.py\", "
        "line 14, in main output_file = sys.argv[2] IndexError: list index out of range"
    )
    record = {
        "package_id": "iuc/trimns",
        "tool_id": "trimns",
        "wrapper_path": "TrimNs.xml",
        "container_execution": [
            {
                "status": "container-command-failed-probe",
                "command": "trim_Ns_DNAnexus.py --help",
                "returncode": 1,
                "error_text": traceback,
            }
        ],
        "container_help_text": "",
        "bioconda_sources": [
            {
                "package": "trimns_vgp",
                "source_checkout": "/tmp/trimns",
                "source_command_docs": [
                    {
                        "path": "README.md",
                        "line": 1,
                        "text": "Usage: python3 trim_Ns_DNAnexus.py <input.fa> <output.list>",
                    }
                ],
            }
        ],
    }

    assert _record_failure_items(record) == []


def test_failure_inventory_suppresses_failed_variants_after_recovered_help() -> None:
    record = {
        "package_id": "iuc/colibread",
        "tool_id": "mapsembler2",
        "wrapper_path": "mapsembler2.xml",
        "container_help_text": "$ run_mapsembler2_pipeline.sh -h\nUsage: run_mapsembler2_pipeline.sh",
        "container_execution": [
            {
                "status": "container-command-missing",
                "command": "run_mapsembler2_pipeline.sh --help",
                "primary_command": "run_mapsembler2_pipeline.sh",
                "returncode": 1,
                "error_text": "run_mapsembler2_pipeline.sh: command not found in container",
            },
            {
                "status": "container-command-failed-probe",
                "command": "run_mapsembler2_pipeline.sh --help",
                "primary_command": "run_mapsembler2_pipeline.sh",
                "returncode": 1,
                "error_text": "illegal option -- -",
            },
            {
                "status": "container-command-help",
                "command": "run_mapsembler2_pipeline.sh -h",
                "primary_command": "run_mapsembler2_pipeline.sh",
                "returncode": 0,
                "error_text": "",
            },
        ],
        "bioconda_sources": [],
    }

    assert _record_failure_items(record) == []


def test_failure_inventory_classifies_structure_banner_as_usable_nonhelp() -> None:
    record = {
        "package_id": "iuc/structure",
        "tool_id": "structure",
        "wrapper_path": "structure.xml",
        "container_help_text": "",
        "container_usage_text": "",
        "container_execution": [
            {
                "status": "container-command-nonhelp",
                "command": "structure --help",
                "primary_command": "structure",
                "returncode": 1,
                "stdout": (
                    "STRUCTURE by Pritchard, Stephens and Donnelly (2000)\n"
                    "Version 2.3.4 (Jul 2012)\n\n"
                    "Can't open the file \"mainparams\".\n\n"
                    "Exiting the program due to error(s) listed above.\n"
                ),
            },
            {
                "status": "container-command-nonhelp",
                "command": "structure -h",
                "primary_command": "structure",
                "returncode": 1,
                "stdout": (
                    "STRUCTURE by Pritchard, Stephens and Donnelly (2000)\n"
                    "Version 2.3.4 (Jul 2012)\n\n"
                    "Can't open the file \"mainparams\".\n\n"
                    "Exiting the program due to error(s) listed above.\n"
                ),
            },
        ],
        "bioconda_sources": [],
    }

    inventory = _build_failure_inventory([record])

    assert inventory["summary"]["category_counts"]["usable_nonhelp_output"] == 1
    assert inventory["summary"]["retry_wrappers"] == 0


def test_failure_inventory_does_not_report_no_help_when_usage_was_captured() -> None:
    record = {
        "package_id": "iuc/genehunter_modscore",
        "tool_id": "genehunter_modscore",
        "wrapper_path": "genehunter_modscore.xml",
        "container_usage_text": "$ ghm linkage2allegro --help\nType 'help' or '?' for help.",
        "container_execution": [
            {
                "status": "container-command-usage-degraded",
                "command": "ghm linkage2allegro --help",
                "primary_command": "ghm",
                "returncode": 0,
                "error_text": "",
            }
        ],
        "bioconda_sources": [],
    }

    items = _record_failure_items(record)

    assert [item["category"] for item in items] == ["usable_nonzero_help"]


def test_failure_inventory_suppresses_no_help_when_api_validation_succeeds() -> None:
    record = {
        "package_id": "iuc/liana",
        "tool_id": "liana_methods",
        "wrapper_path": "liana.xml",
        "container_help_text": "",
        "container_usage_text": "",
        "container_execution": [
            {
                "status": "container-api-validation-ok",
                "command": "python -c 'import liana'",
                "returncode": 0,
            }
        ],
        "container_api_validation": [
            {
                "status": "container-api-validation-ok",
                "checks": ["liana.method.rank_aggregate"],
            }
        ],
        "wrapper_source_summary": {"api_backed_wrapper": True, "configfile_api_call_count": 1},
        "bioconda_sources": [],
    }

    assert _record_failure_items(record) == []


def test_failure_inventory_reports_missing_command_with_version_mismatch_as_partial() -> None:
    record = {
        "package_id": "iuc/stacks",
        "tool_id": "stacks_rxstacks",
        "wrapper_path": "rxstacks.xml",
        "container_help_text": "",
        "container_usage_text": "",
        "container_execution": [
            {
                "status": "container-command-missing",
                "command": "rxstacks --help",
                "primary_command": "rxstacks",
                "returncode": 127,
                "error_text": "rxstacks: command not found in container",
            }
        ],
        "version_consistency": {
            "issues": ["container_version_mismatch:stacks:1.46!=2.65"]
        },
        "bioconda_sources": [],
    }

    items = _record_failure_items(record)

    assert items[0]["category"] == "container_version_mismatch"
    assert items[0]["severity"] == "partial"


def test_failure_inventory_retry_manifest_keeps_only_actionable_items() -> None:
    inventory = _build_failure_inventory(
        [
            {
                "package_id": "iuc/r-exomedepth",
                "tool_id": "exomedepth",
                "wrapper_path": "exomedepth.xml",
                "bioconda_sources": [
                    {
                        "package": "r-exomedepth",
                        "required_version": "1.1.15",
                        "source_url": "https://cran.r-project.org/src/contrib/ExomeDepth_1.1.15.tar.gz",
                        "source_error": "404 Client Error",
                        "source_attempts": [
                            {
                                "source_url": "https://cran.r-project.org/src/contrib/ExomeDepth_1.1.15.tar.gz",
                                "source_error": "404 Client Error",
                            },
                            {
                                "source_url": "https://cran.r-project.org/src/contrib/Archive/ExomeDepth/ExomeDepth_1.1.15.tar.gz",
                                "source_error": "404 Client Error",
                            },
                        ],
                    }
                ],
            },
            {
                "package_id": "iuc/actionable",
                "tool_id": "actionable",
                "wrapper_path": "actionable.xml",
                "bioconda_sources": [
                    {
                        "package": "actionable",
                        "required_version": "1.0",
                        "source_url": "https://github.com/example/actionable/releases/download/1.0/actionable-1.0.tar.gz",
                        "source_error": "404 Client Error",
                    }
                ],
            },
            {
                "package_id": "iuc/weak",
                "tool_id": "weak",
                "wrapper_path": "weak.xml",
                "bioconda_sources": [
                    {
                        "package": "weak",
                        "source_checkout": "/tmp/source",
                        "source_confidence": "weak",
                    }
                ],
            },
            {
                "package_id": "iuc/genehunter_modscore",
                "tool_id": "genehunter_modscore",
                "wrapper_path": "genehunter_modscore.xml",
                "container_usage_text": "$ ghm --help\nType 'help' or '?' for help.",
                "container_execution": [
                    {
                        "status": "container-command-usage-degraded",
                        "command": "ghm --help",
                        "returncode": 0,
                    }
                ],
                "bioconda_sources": [],
            },
        ]
    )

    assert inventory["summary"]["issue_wrappers"] == 4
    assert inventory["summary"]["retry_wrappers"] == 1
    assert inventory["summary"]["category_counts"]["source_404_exhausted"] == 1
    assert inventory["summary"]["category_counts"]["source_404"] == 1
    assert inventory["summary"]["category_counts"]["weak_source_version"] == 1
    assert inventory["summary"]["category_counts"]["usable_nonzero_help"] == 1
    assert inventory["retry_manifest"]["wrappers"][0]["wrapper_path"] == "actionable.xml"


def test_failure_inventory_keeps_recipe_and_binary_source_gaps_report_only() -> None:
    inventory = _build_failure_inventory(
        [
            {
                "package_id": "iuc/qiime",
                "tool_id": "qiime_align",
                "wrapper_path": "qiime.xml",
                "bioconda_sources": [
                    {
                        "package": "qiime",
                        "source_error": "fatal: path 'recipes/qiime/meta.yaml' does not exist in 'master'",
                        "error": "recipe_not_found",
                        "recipe_selection_reason": "recipe_not_found",
                    }
                ],
            },
            {
                "package_id": "iuc/beagle",
                "tool_id": "beagle",
                "wrapper_path": "beagle.xml",
                "bioconda_sources": [
                    {
                        "package": "beagle",
                        "source_error": "binary_artifact_no_source",
                        "source_url": "https://example.org/beagle.jar",
                    }
                ],
            },
        ]
    )

    assert inventory["summary"]["issue_wrappers"] == 2
    assert inventory["summary"]["retry_wrappers"] == 0
    assert inventory["summary"]["category_counts"]["recipe_not_found"] == 1
    assert inventory["summary"]["category_counts"]["binary_artifact_no_source"] == 1


def test_record_help_commands_does_not_probe_api_only_python_configfiles() -> None:
    record = ToolRecord(
        shed_name="liana",
        tool_id="liana_methods",
        tool_name="Liana methods",
        selected_container="quay.io/biocontainers/liana:1.7.1--pyhdfd78af_0",
        requirement_packages=["liana", "scanpy"],
        command_text="python '$script_file'",
        wrapper_configfiles=[
            {
                "name": "script_file",
                "filename": "",
                "extension": "",
                "language": "python",
                "template_kind": "script_template",
                "content": (
                    "import liana as li\n"
                    "import scanpy as sc\n"
                    "adata = sc.read_h5ad('anndata.h5ad')\n"
                    "li.method.rank_aggregate(adata=adata, groupby='cluster')\n"
                ),
            }
        ],
    )

    assert _record_help_commands(record, record.selected_container) == []


def test_record_help_commands_ignores_stale_primary_when_command_text_has_no_candidate() -> None:
    command_text = (
        "cat 'anndata_info.txt' | "
        "sed -r '1 s|AnnData object with (.+) = (.*)|\\1: \\2|g'"
    )
    record = ToolRecord(
        shed_name="episcanpy",
        tool_id="episcanpy_preprocess",
        tool_name="scATAC-seq Preprocessing",
        selected_container="quay.io/biocontainers/anndata:0.6.10--py_0",
        requirement_packages=["anndata"],
        command_text=command_text,
        primary_command="AnnData",
        subcommands=["object"],
    )

    assert _record_help_commands(record, record.selected_container) == []


def test_record_help_commands_requires_source_hint_for_generic_container() -> None:
    record = ToolRecord(
        shed_name="irma",
        tool_id="irma",
        tool_name="IRMA",
        selected_container="quay.io/biocontainers/python:3.12.12",
        requirement_packages=["python"],
        command_text="IRMA '$input' result",
    )
    assert _record_help_commands(record, record.selected_container) == []

    record.bioconda_sources = [{"package": "irma", "command_hints": ["IRMA"]}]
    assert _record_help_commands(record, record.selected_container)[0] == "IRMA --help"


def test_record_help_commands_respects_candidate_package_ownership() -> None:
    record = ToolRecord(
        shed_name="samtools_faidx",
        tool_id="samtools_faidx",
        tool_name="samtools faidx",
        selected_container="quay.io/biocontainers/coreutils:9.5",
        requirement_packages=["coreutils", "samtools"],
        command_text="samtools faidx '$input'",
    )

    assert (
        _record_help_commands(
            record,
            "quay.io/biocontainers/coreutils:9.5",
            candidate_packages=["coreutils"],
        )
        == []
    )
    assert (
        _record_help_commands(
            record,
            "quay.io/biocontainers/samtools:1.23--h96c455f_0",
            candidate_packages=["samtools"],
        )[0]
        == "samtools faidx --help"
    )


def test_record_help_commands_do_not_promote_record_hints_to_source_hints() -> None:
    record = ToolRecord(
        shed_name="genehunter_modscore",
        tool_id="genehunter_modscore",
        tool_name="Genehunter-Modscore",
        selected_container="quay.io/biocontainers/mulled-v2-ghm-linkage2allegro:hash-0",
        requirement_packages=["ghm", "linkage2allegro"],
        command_text=(
            "ghm < '$setup_file'\n"
            "&& linkage2allegro\n"
            "    '${inp_ped}'\n"
            "    '${inp_map}'\n"
            "    genehunter\n"
            "    -l gh.out\n"
        ),
        bioconda_sources=[
            {
                "package": "ghm",
                "command_hints": ["genehunter", "ghm", "linkage2allegro"],
                "record_command_hints": ["genehunter", "ghm", "linkage2allegro"],
                "source_checkout": "",
            }
        ],
    )

    commands = _record_help_commands(
        record,
        record.selected_container,
        candidate_packages=["ghm", "linkage2allegro"],
    )

    assert "ghm --help" in commands
    assert "ghm linkage2allegro --help" in commands
    assert all(not command.startswith("genehunter ") for command in commands)


def test_record_help_commands_rejects_unowned_stacks_summary_candidate() -> None:
    record = ToolRecord(
        shed_name="stacks",
        tool_id="stacks_rxstacks",
        tool_name="Stacks rxstacks",
        requirement_packages=["stacks", "stacks_summary"],
        command_text="rxstacks -P '$input' -o '$output'",
        bioconda_sources=[
            {"package": "stacks", "command_hints": ["rxstacks"]},
            {"package": "stacks_summary", "command_hints": ["stacks_summary.py"]},
        ],
    )

    assert (
        _record_help_commands(
            record,
            "quay.io/biocontainers/stacks_summary:1.1--pyhdfd78af_0",
            candidate_packages=["stacks_summary"],
        )
        == []
    )
    assert (
        _record_help_commands(
            record,
            "quay.io/biocontainers/stacks:2.68--h5efdd21_1",
            candidate_packages=["stacks"],
        )[0]
        == "rxstacks --help"
    )
    assert (
        _record_help_commands(
            record,
            "quay.io/biocontainers/mulled-v2-stacks-velvet:hash-0",
            candidate_packages=["stacks", "stacks_summary", "velvet"],
        )[0]
        == "rxstacks --help"
    )


def test_record_help_commands_skips_commands_for_unrelated_explicit_container() -> None:
    record = ToolRecord(
        shed_name="cite_seq_count",
        tool_id="cite_seq_count",
        tool_name="CITE-seq Count",
        selected_container="bzip2:1.0.8",
        requirement_packages=["bzip2", "cite-seq-count"],
        command_text="CITE-seq-Count --tags tags.csv",
    )

    assert _record_help_commands(record, "bzip2:1.0.8", candidate_packages=["bzip2"]) == []


def test_record_help_commands_safe_mode_excludes_exploratory_variants() -> None:
    record = ToolRecord(
        shed_name="samtools",
        selected_container="quay.io/biocontainers/samtools:1.10--h2e538c0_3",
        requirement_packages=["samtools"],
        command_text="samtools view '$input'",
    )
    commands = _record_help_commands(record, record.selected_container, probe_mode="safe")
    assert commands == [
        "samtools view --help",
        "samtools view -h",
        "samtools --help",
        "samtools -h",
    ]


def test_extract_help_commands_exploratory_includes_usage_and_help_subcommand_forms() -> None:
    commands = _extract_help_commands("samtools", ["view"], probe_mode="exploratory")

    assert commands[:4] == [
        "samtools view --help",
        "samtools view -h",
        "samtools view --usage",
        "samtools view '-?'",
    ]
    assert "samtools help view" in commands
    assert "samtools view help" in commands
    assert "samtools --usage" in commands
    assert "samtools '-?'" in commands


def test_choose_container_candidate_prefers_full_mulled_v2_over_single_version_match() -> None:
    candidates = [
        {
            "image": "quay.io/biocontainers/bzip2:1.0.8--h4bc722e_7",
            "source": "mulled-single",
            "packages": ["bzip2"],
            "priority": 260,
            "status": "ok",
        },
        {
            "image": "quay.io/biocontainers/mulled-v2-example:hash-0",
            "source": "mulled-v2",
            "packages": ["bzip2", "cite-seq-count"],
            "priority": 250,
            "status": "ok",
        },
    ]

    selected = _choose_container_candidate(
        candidates,
        requirement_versions={"bzip2": "1.0.8", "cite-seq-count": "1.5.0"},
        requirement_packages=["bzip2", "cite-seq-count"],
    )

    assert selected["source"] == "mulled-v2"


def test_container_run_command_isolates_probe_for_docker() -> None:
    command = _container_run_command(
        ContainerPreparation(
            ok=True, runtime="docker", image="tool", identifier="tool:1", source="docker-pull"
        ),
        "tool",
        ExtractionSettings(),
    )
    assert "--network" in command
    assert "none" in command
    assert "--workdir" in command
    assert "/tmp" in command
    assert command[-3:] == ["bash", "-lc", command[-1]]
    assert "mktemp -d /tmp/gtsm-help" in command[-1]
    assert "tool" in command[-1]


def test_available_container_runtimes_find_env_local_apptainer(
    monkeypatch, tmp_path
) -> None:
    env_prefix = tmp_path / "env"
    env_bin = env_prefix / "bin"
    env_bin.mkdir(parents=True)
    (env_prefix / "conda-meta").mkdir()
    apptainer = env_bin / "apptainer"
    singularity = env_bin / "singularity"
    python = env_bin / "python"
    apptainer.write_text("#!/bin/sh\n", encoding="utf-8")
    singularity.write_text("#!/bin/sh\n", encoding="utf-8")
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    apptainer.chmod(0o755)
    singularity.chmod(0o755)
    python.chmod(0o755)

    monkeypatch.setattr("galaxy_toolsmith.data.corpus.shutil.which", lambda name: None)
    monkeypatch.setattr("galaxy_toolsmith.data.corpus.sys.executable", str(python))

    runtimes = _available_container_runtimes(ExtractionSettings(container_runtime="auto"))

    assert [(runtime.name, runtime.executable) for runtime in runtimes] == [
        ("singularity", str(singularity)),
        ("apptainer", str(apptainer)),
    ]


def test_container_shell_command_uses_env_local_apptainer(monkeypatch, tmp_path) -> None:
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    apptainer = env_bin / "apptainer"
    python = env_bin / "python"
    apptainer.write_text("#!/bin/sh\n", encoding="utf-8")
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    apptainer.chmod(0o755)
    python.chmod(0o755)

    monkeypatch.setattr("galaxy_toolsmith.data.corpus.shutil.which", lambda name: None)
    monkeypatch.setattr("galaxy_toolsmith.data.corpus.sys.executable", str(python))

    command = _container_shell_command(
        ContainerPreparation(
            ok=True,
            runtime="apptainer",
            image="tool",
            identifier="/tmp/tool.sif",
            source="cache",
        ),
        "tool --help",
        ExtractionSettings(),
        "bash",
    )

    assert command[:4] == [str(apptainer), "exec", "--cleanenv", "/tmp/tool.sif"]
    assert command[-3:] == ["bash", "-lc", "tool --help"]


def test_run_command_prepends_active_env_bin_and_sbin(monkeypatch, tmp_path) -> None:
    env_prefix = tmp_path / "env"
    env_bin = env_prefix / "bin"
    env_sbin = env_prefix / "sbin"
    env_bin.mkdir(parents=True)
    env_sbin.mkdir()
    (env_prefix / "conda-meta").mkdir()
    python = env_bin / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    captured = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr("galaxy_toolsmith.data.corpus.sys.executable", str(python))
    monkeypatch.setattr("galaxy_toolsmith.data.corpus.subprocess.run", fake_run)

    _run_command(["apptainer", "--version"], timeout_seconds=5)

    assert captured["env"]["PATH"].split(":")[:3] == [
        str(env_bin),
        str(env_sbin),
        "/usr/bin",
    ]


def test_no_arg_probe_uses_short_timeout() -> None:
    settings = ExtractionSettings(
        container_run_timeout_seconds=120, container_no_arg_timeout_seconds=7
    )
    assert _container_probe_timeout("samtools", settings) == 7
    assert _container_probe_timeout("samtools --help", settings) == 120
    assert _container_probe_timeout("samtools help", settings) == 120
    assert _container_probe_timeout("samtools --usage", settings) == 120
    assert _container_probe_timeout("samtools '-?'", settings) == 120


def test_failed_probe_text_is_not_added_as_help() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr=(
            "WARNING: Skipping mount /etc/resolv.conf [files]: /etc/resolv.conf doesn't exist in container\n"
            "mkdir: unrecognized option '--help'\n"
            "Usage: mkdir [OPTIONS] DIRECTORY...\n"
        ),
    )
    assert _container_help_fragment("mkdir ./tabular --help", result) == ""


def test_nonhelp_banner_and_failed_usage_are_not_added_as_help() -> None:
    breseq = subprocess.CompletedProcess(
        ["singularity", "exec"],
        255,
        stdout="breseq 0.39.0\nCopyright (c) 2008-2022\nIf you use breseq in your research, please cite:\n",
        stderr="",
    )
    assert _container_help_fragment("breseq --help", breseq) == ""

    berokka = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr="You ran: /usr/local/bin/berokka --help\nPlease specify the output folder with --outdir",
    )
    assert _container_help_fragment("berokka --help", berokka) == ""


def test_mothur_command_list_is_accepted_as_help() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        0,
        stdout=(
            "mothur v.1.39.5\n"
            "Valid commands are: align.check, align.seqs, amova, biom.info, "
            "classify.otu, classify.seqs, cluster, count.seqs, make.contigs, "
            "summary.seqs, trim.seqs, unique.seqs.\n"
            "For more information about a specific command type 'commandName(help)' "
            "i.e. 'cluster(help)'\n"
            "For further assistance please refer to the Mothur manual on our wiki.\n"
        ),
        stderr="",
    )
    fragment = _container_help_fragment("mothur --help", result)
    assert "$ mothur --help" in fragment
    assert "commandName(help)" in fragment
    assert _container_probe_status(result, fragment) == "container-command-help"


def test_hyphy_option_section_is_accepted_as_help() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        0,
        stdout=(
            "HyPhy version 2.5.70\n\n"
            "Available analysis command line options\n"
            "  --alignment FILE       Input alignment\n"
            "  --tree FILE            Input tree\n"
            "  --branches VALUE       Branch set\n"
        ),
        stderr="",
    )
    fragment = _container_help_fragment("hyphy --help", result)
    assert "$ hyphy --help" in fragment
    assert "Available analysis command line options" in fragment
    assert _container_probe_status(result, fragment) == "container-command-help"


def test_synopsis_options_section_is_accepted_as_help() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        0,
        stdout=(
            "proteinortho_summary.pl        produces a summary on species level.\n\n"
            "SYNOPSIS\n\n"
            "proteinortho_summary.pl (options) GRAPH (GRAPH2)\n\n"
            "    GRAPH   Path to a Proteinortho graph file.\n"
            "    GRAPH2  Optional second graph file.\n\n"
            "OPTIONS\n\n"
            "    -format,-f  enables a specific output format.\n"
            "    -help,-h    prints this help text.\n"
        ),
        stderr="",
    )
    fragment = _container_help_fragment("proteinortho_summary.pl --help", result)
    assert "$ proteinortho_summary.pl --help" in fragment
    assert "SYNOPSIS" in fragment
    assert _container_probe_status(result, fragment) == "container-command-help"


def test_help_text_with_not_found_prose_is_not_failed_probe() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        0,
        stdout=(
            "positional arguments:\n"
            "  path                  path to the PhysiCell output directory\n\n"
            "options:\n"
            "  -h, --help            show this help message and exit\n"
            "  --settingxml SETTINGXML\n"
            "                        the settings.xml that is loaded if this\n"
            "                        information is not found in the output xml file.\n"
        ),
        stderr="",
    )
    fragment = _container_help_fragment("pcdl_make_cell_vtk --help", result)
    assert "$ pcdl_make_cell_vtk --help" in fragment
    assert "not found in the output xml file" in fragment
    assert _container_probe_status(result, fragment) == "container-command-help"


def test_missing_input_file_probe_is_not_missing_command() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr="[bwa_idx_build] fail to open file 'help' : No such file or directory",
    )
    assert _is_missing_command_probe(result) is False

    missing = subprocess.CompletedProcess(
        ["singularity", "exec"],
        127,
        stdout="",
        stderr="bash: bwa-mem2: command not found",
    )
    assert _is_missing_command_probe(missing) is True


def test_degraded_bedtools_help_is_cleaned_and_accepted() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr=(
            "*****ERROR: Unrecognized parameter: --help *****\n\n"
            "*****\n"
            "*****ERROR: Need -g (genome) file. \n"
            "*****\n\n"
            "Tool:    bedtools bedpetobam (aka bedpeToBam)\n"
            "Version: v2.31.1\n"
            "Summary: Converts feature records to BAM format.\n\n"
            "Usage:   bedpetobam [OPTIONS] -i <bed/gff/vcf> -g <genome>\n\n"
            "Options:\n\t-mapq\tSet the mapping quality for the BAM records.\n"
        ),
    )
    fragment = _container_help_fragment("bedtools bedpetobam --help", result)
    assert "$ bedtools bedpetobam --help" in fragment
    assert "Usage:   bedpetobam" in fragment
    assert "Unrecognized parameter" not in fragment
    assert "Need -g" not in fragment
    assert _container_probe_status(result, fragment) == "container-command-help-degraded"


def test_degraded_lordec_help_is_cleaned_and_accepted() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr=(
            "lordec-correct: unrecognized option '--help'\n"
            "LoRDEC v0.9\n"
            "using GATB v1.4.1\n\n"
            "Usage :\n\n"
            "lordec-correct\n\n"
            "-i|--long_reads <long read FASTA/Q file>\n"
            "-2|--short_reads <short read FASTA/Q file(s)>\n"
            "-k|--kmer_len <k-mer size>\n"
        ),
    )
    fragment = _container_help_fragment("lordec-correct --help", result)
    assert "$ lordec-correct --help" in fragment
    assert "LoRDEC v0.9" in fragment
    assert "Usage :" in fragment
    assert "unrecognized option" not in fragment
    assert _container_probe_status(result, fragment) == "container-command-help-degraded"


def test_varscan_usage_with_locale_warning_is_degraded_help() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        0,
        stdout="",
        stderr=(
            "/usr/local/bin/varscan: line 6: warning: setlocale: LC_ALL: cannot change locale "
            "(en_US.UTF-8): No such file or directory\n"
            "Command not recognized\n"
            "VarScan v2.4.3\n\n"
            "USAGE: java -jar VarScan.jar [COMMAND] [OPTIONS]\n\n"
            "COMMANDS:\n"
            "\tpileup2snp\t\tIdentify SNPs from a pileup file\n"
            "\tsomatic\t\t\tCall germline/somatic variants from tumor-normal pileups\n"
        ),
    )
    fragment = _container_help_fragment("varscan --help", result)
    assert "$ varscan --help" in fragment
    assert "VarScan v2.4.3" in fragment
    assert "USAGE: java -jar VarScan.jar" in fragment
    assert "cannot change locale" not in fragment
    assert "Command not recognized" not in fragment
    assert _container_probe_status(result, fragment) == "container-command-help-degraded"


def test_option_heavy_nonzero_help_is_accepted() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="\n  [--samplingrounds N]\n  [-w REPEAT]\n  [--polylimit N]\n  [-x]\n",
        stderr="",
    )
    fragment = _container_help_fragment("astral --help", result)
    assert "$ astral --help" in fragment


def test_commented_option_block_is_accepted_as_help() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        255,
        stdout="",
        stderr=(
            "file. See documentation for formatting.\n"
            "#  --gene_predictions <str>    gene predictions gff3 file\n"
            "#\n"
            "#  --segmentSize <str>          length of a single sequence\n"
            "#  --overlapSize  <str>         length of sequence overlap\n"
            "#\n"
            "# flags:\n"
            "#  --forwardStrandOnly          runs only on the forward strand\n"
            "#  --reverseStrandOnly          runs only on the reverse strand\n"
            "#  --version                    report version and exit\n"
        ),
    )
    fragment = _container_help_fragment("EVidenceModeler --help", result)
    assert "$ EVidenceModeler --help" in fragment
    assert "--gene_predictions" in fragment
    assert _container_probe_status(result, fragment) == "container-command-help"


def test_argument_count_runtime_text_is_captured_as_degraded_usage() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr=(
            "INFO:    squashfuse not found, will not be able to mount SIF or other squashfs files\n"
            "INFO:    gocryptfs not found, will not be able to use gocryptfs\n"
            "INFO:    Converting SIF file to temporary sandbox...\n"
            "Version: 0.7.1\n"
            "Please provide the (coordinate) sorted input SAM File, as well as the MT "
            "identifier. No further parameters are necessary!\n"
            "INFO:    Cleaning up image...\n"
        ),
    )

    fragment = _container_help_fragment("mtnucratio --help", result)
    usage = _container_usage_fragment("mtnucratio --help", result)

    assert fragment == ""
    assert "$ mtnucratio --help" in usage
    assert "No further parameters" in usage
    assert "squashfuse" not in usage
    assert "Converting SIF" not in usage
    assert "Cleaning up image" not in usage
    assert _container_probe_status(result, fragment, usage) == "container-command-usage-degraded"


def test_interactive_help_banner_is_captured_as_degraded_usage() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        0,
        stdout=(
            "GENEHUNTER-MODSCORE - A modified version of GENEHUNTER\n"
            "(version 3.0)\n\n"
            "Type 'help' or '?' for help.\n"
            "Can't find help file - detailed help information is not available.\n"
            "npl:1>\n"
        ),
        stderr="",
    )

    fragment = _container_help_fragment("ghm --help", result)
    usage = _container_usage_fragment("ghm --help", result)

    assert fragment == ""
    assert "$ ghm --help" in usage
    assert "GENEHUNTER-MODSCORE" in usage
    assert _container_probe_status(result, fragment, usage) == "container-command-usage-degraded"


def test_missing_argument_traceback_is_failed_probe() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "/usr/local/bin/trim_Ns_DNAnexus.py", line 137, in <module>\n'
            "    main()\n"
            '  File "/usr/local/bin/trim_Ns_DNAnexus.py", line 14, in main\n'
            "    output_file = sys.argv[2]\n"
            "IndexError: list index out of range\n"
        ),
    )
    fragment = _container_help_fragment("trim_Ns_DNAnexus.py --help", result)
    usage = _container_usage_fragment("trim_Ns_DNAnexus.py --help", result)

    assert fragment == ""
    assert usage == ""
    assert _container_probe_status(result, fragment, usage) == "container-command-failed-probe"


def test_runtime_import_traceback_is_failed_probe() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "/usr/local/bin/medaka", line 5, in <module>\n'
            "    from medaka.medaka import main\n"
            "ModuleNotFoundError: No module named '_cffi_backend'\n"
        ),
    )
    fragment = _container_help_fragment("medaka --help", result)
    usage = _container_usage_fragment("medaka --help", result)

    assert fragment == ""
    assert usage == ""
    assert _container_probe_status(result, fragment, usage) == "container-command-failed-probe"


def test_aligned_usage_help_labels_are_accepted() -> None:
    result = subprocess.CompletedProcess(
        ["singularity", "exec"],
        0,
        stdout=(
            "description: Collects the ancestor terms from a given term in the given OBO ontology.\n"
            "\tusage      : get_ancestor_terms.pl [options]\n"
            "\toptions    :\n"
            "\t\t-f  \t OBO input file\n"
            "\t\t-t \t term ID\n"
            "\texample:\n"
            "\t\tperl get_ancestor_terms.pl -f go.obo -t GO:0000234\n"
        ),
        stderr="",
    )

    fragment = _container_help_fragment("get_ancestor_terms.pl --help", result)

    assert "$ get_ancestor_terms.pl --help" in fragment
    assert "usage      :" in fragment
    assert _container_probe_status(result, fragment) == "container-command-help"


def test_execute_container_help_falls_back_to_docker_and_enriches_help(monkeypatch) -> None:
    record = ToolRecord(
        tool_name="samtools",
        help_text="Wrapper help",
        original_help_text="Wrapper help",
        selected_container="quay.io/biocontainers/samtools:1.10--h2e538c0_3",
        primary_command="samtools",
    )

    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [
            ContainerRuntime("singularity", "singularity"),
            ContainerRuntime("docker", "docker"),
        ],
    )

    def fake_prepare(image, runtime, settings):
        if runtime.name == "singularity":
            return ContainerPreparation(
                ok=False,
                runtime="singularity",
                image=image,
                source="docker-build",
                returncode=1,
                error_text="singularity build failed",
            )
        return ContainerPreparation(
            ok=True,
            runtime="docker",
            image=image,
            identifier=image,
            source="docker-pull",
        )

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._prepare_container", fake_prepare)

    def fake_run(command, timeout_seconds):
        if "run" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="Usage: samtools [options]\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, remove_images_after_use=True),
    )

    assert summary["runtime_fallbacks"] == 1
    assert summary["commands_executed"] == 1
    assert record.selected_container_runtime == "docker"
    assert "Wrapper help" in record.help_text
    assert "Usage: samtools" in record.container_help_text


def test_execute_container_help_quarantines_timed_out_image_without_runtime_fallback(
    monkeypatch,
) -> None:
    record = ToolRecord(
        tool_name="tiberius",
        selected_container="larsgabriel23/tiberius:2.0.3",
        primary_command="tiberius",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [
            ContainerRuntime("singularity", "singularity"),
            ContainerRuntime("apptainer", "apptainer"),
            ContainerRuntime("docker", "docker"),
        ],
    )
    attempted_runtimes = []

    def fake_prepare(image, runtime, settings):
        attempted_runtimes.append(runtime.name)
        return ContainerPreparation(
            ok=False,
            runtime=runtime.name,
            image=image,
            source="docker-build",
            returncode=124,
            error_text="Command timed out after 300 seconds",
        )

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._prepare_container", fake_prepare)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_runtime="auto"),
    )

    assert attempted_runtimes == ["singularity"]
    assert summary["prepare_failed"] == 1
    assert summary["runtime_fallbacks"] == 0
    assert record.container_execution[0]["status"] == "container-prepare-failed"
    assert record.container_execution[0]["runtime"] == "singularity"
    assert record.container_execution[0]["returncode"] == 124


def test_execute_container_validates_api_backed_wrapper_without_cli_help(monkeypatch) -> None:
    record = ToolRecord(
        tool_name="scATAC-seq Preprocessing",
        tool_id="episcanpy_preprocess",
        selected_container="quay.io/biocontainers/mulled-v2-episcanpy:hash-0",
        requirement_packages=["episcanpy", "scanpy"],
        command_text="python '$script_file'",
        wrapper_configfiles=[
            {
                "name": "script_file",
                "template_kind": "script_template",
                "api_calls": [
                    {
                        "language": "python",
                        "qualified_call": "episcanpy.pp.binarize",
                    },
                    {
                        "language": "python",
                        "qualified_call": "episcanpy.pp.filter_cells",
                    },
                ],
            }
        ],
        wrapper_source_summary={"api_backed_wrapper": True, "configfile_api_call_count": 2},
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
            identifier=image,
            source="cache",
        ),
    )

    seen_probes = []

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        seen_probes.append(probe)
        if "command -v python" in probe:
            return subprocess.CompletedProcess(command, 0, stdout="/usr/bin/python\n", stderr="")
        if "python -c" in probe:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "GTSM_API_VALIDATION_OK python 2\n"
                    'GTSM_API_DOCS [{"qualified_call":"episcanpy.pp.binarize",'
                    '"signature":"(adata)","doc":"Binarize a matrix."}]\n'
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert summary["commands_executed"] == 1
    assert summary["api_validation_ok"] == 1
    assert summary["commands_failed"] == 0
    assert record.container_help_text == ""
    assert record.container_api_validation[0]["status"] == "container-api-validation-ok"
    assert record.container_api_validation[0]["probe_role"] == "api_validation"
    assert record.container_api_validation[0]["api_docs"][0]["signature"] == "(adata)"
    assert "Runtime API validation from container execution" in record.help_text
    assert "episcanpy.pp.binarize" in record.help_text
    assert record.container_execution[0]["phase"] == "api_validation"
    assert any("python -c" in probe for probe in seen_probes)


def test_execute_container_accepts_partial_python_api_validation(monkeypatch) -> None:
    record = ToolRecord(
        selected_container="quay.io/biocontainers/alphagenome:0.6.1--pyhdfd78af_0",
        requirement_packages=["alphagenome"],
        command_text="python '$__tool_directory__/alphagenome_sequence_predictor.py'",
        wrapper_helper_files=[
            {
                "relative_path": "alphagenome_sequence_predictor.py",
                "api_calls": [
                    {
                        "language": "python",
                        "qualified_call": "missing_optional.module.create",
                    },
                    {
                        "language": "python",
                        "qualified_call": "alphagenome.models.dna_client.create",
                    },
                ],
            }
        ],
        wrapper_source_summary={"api_backed_wrapper": True, "helper_api_call_count": 2},
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
            identifier=image,
            source="cache",
        ),
    )

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        if "command -v python" in probe:
            return subprocess.CompletedProcess(command, 0, stdout="/usr/bin/python\n", stderr="")
        if "python -c" in probe:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "GTSM_API_VALIDATION_OK python 1\n"
                    'GTSM_API_DOCS [{"qualified_call":"alphagenome.models.dna_client.create",'
                    '"signature":"(api_key)","doc":"Create a DNA model client."}]\n'
                    'GTSM_API_ERRORS [{"qualified_call":"missing_optional.module.create",'
                    '"error":"ModuleNotFoundError()"}]\n'
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert summary["api_validation_ok"] == 1
    assert record.container_api_validation[0]["status"] == "container-api-validation-ok"
    assert record.container_api_validation[0]["api_docs"][0]["qualified_call"] == (
        "alphagenome.models.dna_client.create"
    )
    assert record.container_api_validation[0]["api_errors"][0]["qualified_call"] == (
        "missing_optional.module.create"
    )


def test_python_api_validation_imports_longest_module_prefix() -> None:
    command = _python_api_validation_command(
        ["email.parser.Parser.parse", "missing_optional.module.create"]
    )

    result = subprocess.run(
        shlex.split(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "GTSM_API_VALIDATION_OK python 1" in result.stdout
    assert '"qualified_call":"email.parser.Parser.parse"' in result.stdout
    assert '"qualified_call":"missing_optional.module.create"' in result.stdout


def test_execute_container_captures_degraded_usage_text(monkeypatch) -> None:
    record = ToolRecord(
        tool_name="Mt/Nuc Ratio Calculator",
        selected_container="quay.io/biocontainers/mtnucratio:0.7.1--hdfd78af_0",
        requirement_packages=["mtnucratio"],
        primary_command="mtnucratio",
        original_help_text="Wrapper help",
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
            identifier=image,
            source="cache",
        ),
    )

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        if "command -v mtnucratio" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/mtnucratio\n", stderr=""
            )
        if "mtnucratio --help" in probe:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=(
                    "Version: 0.7.1\n"
                    "Please provide the (coordinate) sorted input SAM File, as well as the MT "
                    "identifier. No further parameters are necessary!\n"
                ),
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert summary["commands_executed"] == 1
    assert summary["usage_degraded"] == 1
    assert summary["commands_failed"] == 0
    assert record.container_help_text == ""
    assert "$ mtnucratio --help" in record.container_usage_text
    assert "Runtime usage text collected" in record.help_text
    assert record.container_execution[-1]["status"] == "container-command-usage-degraded"


def test_extract_source_command_docs_reads_usage_from_underlying_source(tmp_path) -> None:
    source_root = tmp_path / "trimns_vgp"
    docs = source_root / "pipeline" / "bionano" / "trimNs"
    docs.mkdir(parents=True)
    (docs / "README.md").write_text(
        """# TrimNs

### remove_fake_cut_sites_DNAnexus.py

Usage: python3 remove_fake_cut_sites_DNAnexus.py <input.fa> <output.fa> <output.log>

### trim_Ns_DNAnexus.py

Usage: python3 trim_Ns_DNAnexus.py <input.fa> <output.list>

### clip_regions_DNAnexus.py

python3 clip_regions_DNAnexus.py <input.fa> <input.list> <output.fa>
""",
        encoding="utf-8",
    )

    extracted = _extract_source_command_docs(
        str(source_root),
        "trimns_vgp",
        {
            "remove_fake_cut_sites_DNAnexus.py",
            "trim_Ns_DNAnexus.py",
            "clip_regions_DNAnexus.py",
        },
    )

    combined = "\n".join(str(item["text"]) for item in extracted)
    assert "remove_fake_cut_sites_DNAnexus.py <input.fa>" in combined
    assert "trim_Ns_DNAnexus.py <input.fa>" in combined


def test_extract_source_command_docs_prefers_wrapper_commands(tmp_path) -> None:
    source_root = tmp_path / "trimns_vgp"
    broad_docs = source_root / "dx_applets" / "meryl_genomescope"
    broad_docs.mkdir(parents=True)
    (broad_docs / "Readme.md").write_text(
        """# Meryl and Genomescope

Usage: meryl count k=31 input.fa output merylDb
""",
        encoding="utf-8",
    )
    focused_docs = source_root / "pipeline" / "bionano" / "trimNs"
    focused_docs.mkdir(parents=True)
    (focused_docs / "README.md").write_text(
        """# TrimNs

### trim_Ns_DNAnexus.py

Usage: python3 trim_Ns_DNAnexus.py <input.fa> <output.list>
""",
        encoding="utf-8",
    )

    extracted = _extract_source_command_docs(
        str(source_root),
        "trimns_vgp",
        {"meryl", "genomescope", "trim_Ns_DNAnexus.py"},
        preferred_command_hints={"trim_Ns_DNAnexus.py"},
    )

    combined = "\n".join(str(item["text"]) for item in extracted)
    assert "trim_Ns_DNAnexus.py <input.fa>" in combined
    assert "meryl count" not in combined


def test_extract_source_command_docs_ignores_build_files(tmp_path) -> None:
    source_root = tmp_path / "diamond"
    source_root.mkdir()
    (source_root / "CMakeLists.txt").write_text(
        """# diamond build

Usage: cmake -DDIAMOND_BUILD=ON .
""",
        encoding="utf-8",
    )
    docs = source_root / "doc"
    docs.mkdir()
    (docs / "manual.txt").write_text(
        """# DIAMOND manual

Usage: diamond makedb --in proteins.fasta -d proteins
""",
        encoding="utf-8",
    )

    extracted = _extract_source_command_docs(
        str(source_root),
        "diamond",
        {"diamond", "makedb"},
    )

    combined = "\n".join(str(item["text"]) for item in extracted)
    assert "diamond makedb --in" in combined
    assert "cmake -DDIAMOND_BUILD" not in combined


def test_execute_container_help_falls_back_between_probe_variants(monkeypatch) -> None:
    record = ToolRecord(
        selected_container="quay.io/biocontainers/breseq:0.39.0--h077b44d_3",
        requirement_packages=["breseq"],
        command_text="breseq '$input'",
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
            identifier=image,
            source="cache",
        ),
    )

    seen_commands = []

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        seen_commands.append(probe)
        if "command -v breseq" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/breseq\n", stderr=""
            )
        if "breseq --help" in probe:
            return subprocess.CompletedProcess(
                command, 255, stdout="breseq 0.39.0\nplease cite this tool\n", stderr=""
            )
        if "breseq -h" in probe:
            return subprocess.CompletedProcess(
                command, 1, stdout="Usage: breseq [OPTIONS]\n  -r REF\n  -o OUT\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="exploratory"),
    )

    assert summary["commands_executed"] == 2
    assert "command -v breseq" in seen_commands[0]
    assert "breseq --help" in seen_commands[1]
    assert "breseq -h" in seen_commands[2]
    assert "$ breseq -h" in record.container_help_text
    assert "please cite this tool" not in record.container_help_text


def test_execute_container_help_skips_unrelated_candidate_before_preflight(monkeypatch) -> None:
    record = ToolRecord(
        selected_container="https://depot.galaxyproject.org/singularity/augustus:3.5.0--pl5321heb9362c_5",
        requirement_packages=["busco"],
        command_text="busco '$input'",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [ContainerRuntime("singularity", "singularity")],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._prepare_container",
        lambda image, runtime, settings: ContainerPreparation(
            ok=True,
            runtime="singularity",
            image=image,
            identifier=image,
            source="cache",
        ),
    )

    seen_commands = []

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        seen_commands.append(probe)
        if "command -v busco" in probe:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        raise AssertionError(f"unexpected probe after missing preflight: {probe}")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="exploratory"),
    )

    assert summary["commands_executed"] == 0
    assert summary["commands_failed"] == 0
    assert seen_commands == []
    assert record.container_execution == []


def test_execute_container_help_falls_back_to_sh_when_bash_is_missing(monkeypatch) -> None:
    record = ToolRecord(
        selected_container="quay.io/biocontainers/samtools:1.10--h2e538c0_3",
        requirement_packages=["samtools"],
        primary_command="samtools",
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
            identifier=image,
            source="cache",
        ),
    )

    seen_shells = []

    def fake_run(command, timeout_seconds):
        shell = command[-3]
        probe = command[-1]
        seen_shells.append(shell)
        if shell == "bash":
            return subprocess.CompletedProcess(command, 127, stdout="", stderr="bash: not found")
        if "command -v samtools" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/samtools\n", stderr=""
            )
        if "samtools --help" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="Usage: samtools [options]\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert summary["commands_executed"] == 1
    assert summary["commands_failed"] == 0
    assert seen_shells == ["bash", "sh", "bash", "sh"]
    assert "$ samtools --help" in record.container_help_text


def test_execute_container_help_tries_later_candidate_after_missing_command(monkeypatch) -> None:
    record = ToolRecord(
        requirement_packages=["samtools"],
        command_text="samtools faidx '$input'",
        container_candidate_details=[
            {
                "image": "quay.io/biocontainers/samtools:broken",
                "source": "mulled-single",
                "packages": ["samtools"],
                "priority": 300,
                "status": "ok",
                "error_text": "",
            },
            {
                "image": "quay.io/biocontainers/samtools:1.23--h96c455f_0",
                "source": "mulled-single",
                "packages": ["samtools"],
                "priority": 200,
                "status": "ok",
                "error_text": "",
            },
        ],
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
            identifier=image,
            source="cache",
        ),
    )

    seen_identifiers = []

    def fake_run(command, timeout_seconds):
        identifier = command[3]
        probe = command[-1]
        seen_identifiers.append(identifier)
        if "command -v samtools" in probe and identifier.endswith(":broken"):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        if "command -v samtools" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/samtools\n", stderr=""
            )
        if "samtools faidx --help" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="Usage: samtools faidx [options]\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert summary["commands_executed"] == 1
    assert "$ samtools faidx --help" in record.container_help_text
    assert record.selected_container == "quay.io/biocontainers/samtools:1.23--h96c455f_0"
    assert seen_identifiers[0].endswith(":broken")
    assert seen_identifiers[-1].endswith(":1.23--h96c455f_0")
    assert record.container_execution[0]["status"] == "container-command-missing"
    assert record.container_execution[-1]["status"] == "container-command-help"


def test_execute_container_help_tries_later_primary_in_same_candidate(monkeypatch) -> None:
    record = ToolRecord(
        requirement_packages=["gamma"],
        command_text="badcmd '$input'\nGAMMA.py '$input'",
        container_candidate_details=[
            {
                "image": "quay.io/biocontainers/gamma:2.2--pyhdfd78af_0",
                "source": "mulled-single",
                "packages": ["gamma"],
                "priority": 300,
                "status": "ok",
                "error_text": "",
            }
        ],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._record_help_command_plan",
        lambda *args, **kwargs: [
            {"command": "badcmd --help", "primary": "badcmd", "probe_role": "core"},
            {"command": "GAMMA.py --help", "primary": "GAMMA.py", "probe_role": "core"},
        ],
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
            identifier=image,
            source="cache",
        ),
    )

    seen_probes = []

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        seen_probes.append(probe)
        if "command -v badcmd" in probe:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        if "command -v GAMMA.py" in probe:
            return subprocess.CompletedProcess(command, 0, stdout="/usr/bin/GAMMA.py\n", stderr="")
        if "GAMMA.py --help" in probe:
            return subprocess.CompletedProcess(command, 0, stdout="Usage: GAMMA.py [options]\n", stderr="")
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert summary["commands_executed"] == 1
    assert summary["missing_command"] == 1
    assert "$ GAMMA.py --help" in record.container_help_text
    assert not any("badcmd --help" in probe for probe in seen_probes)
    assert record.container_execution[0]["status"] == "container-command-missing"
    assert record.container_execution[-1]["status"] == "container-command-help"


def test_execute_container_help_keeps_probing_after_missing_input_file(monkeypatch) -> None:
    record = ToolRecord(
        requirement_packages=["bwa-mem2"],
        command_text="bwa-mem2 index -p reference '$input'",
        container_candidate_details=[
            {
                "image": "quay.io/biocontainers/bwa-mem2:2.2.1--he70b90d_8",
                "source": "mulled-single",
                "packages": ["bwa-mem2"],
                "priority": 300,
                "status": "ok",
                "error_text": "",
            }
        ],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._record_help_command_plan",
        lambda *args, **kwargs: [
            {
                "command": "bwa-mem2 index help",
                "primary": "bwa-mem2",
                "probe_role": "core",
            },
            {"command": "bwa-mem2 --help", "primary": "bwa-mem2", "probe_role": "core"},
        ],
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
            identifier=image,
            source="cache",
        ),
    )

    seen_probes = []

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        seen_probes.append(probe)
        if "command -v bwa-mem2" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/bwa-mem2\n", stderr=""
            )
        if "bwa-mem2 index help" in probe:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="[bwa_idx_build] fail to open file 'help' : No such file or directory",
            )
        if "bwa-mem2 --help" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="Usage: bwa-mem2 <command> [options]\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="exploratory"),
    )

    assert summary["commands_executed"] >= 1
    assert summary["missing_command"] == 0
    assert "$ bwa-mem2 --help" in record.container_help_text
    assert any("bwa-mem2 index help" in probe for probe in seen_probes)
    assert any("bwa-mem2 --help" in probe for probe in seen_probes)


def test_execute_container_help_skips_repeated_runtime_import_failures(monkeypatch) -> None:
    record = ToolRecord(
        requirement_packages=["medaka"],
        selected_container="quay.io/biocontainers/medaka:2.1.0--pyhdfd78af_0",
        command_text="medaka tools '$input'",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._record_help_command_plan",
        lambda *args, **kwargs: [
            {"command": "medaka tools --help", "primary": "medaka", "probe_role": "core"},
            {"command": "medaka tools -h", "primary": "medaka", "probe_role": "core"},
            {"command": "medaka --help", "primary": "medaka", "probe_role": "core"},
            {"command": "medaka -h", "primary": "medaka", "probe_role": "core"},
        ],
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
            identifier=image,
            source="cache",
        ),
    )

    seen_probes = []
    traceback = (
        "Traceback (most recent call last):\n"
        "  File \"/usr/local/bin/medaka\", line 7, in <module>\n"
        "ModuleNotFoundError: No module named '_cffi_backend'\n"
    )

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        seen_probes.append(probe)
        if "command -v medaka" in probe:
            return subprocess.CompletedProcess(command, 0, stdout="/usr/bin/medaka\n", stderr="")
        if "medaka tools" in probe or "medaka " in probe:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=traceback)
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="exploratory"),
    )

    run_probes = [
        probe.split("; ")[-1] for probe in seen_probes if "command -v" not in probe
    ]
    assert _probe_command_base("medaka help tools", "medaka") == "medaka tools"
    assert run_probes == [
        "medaka tools --help",
        "medaka --help",
    ]
    assert summary["commands_executed"] == 2
    assert summary["failed_probe"] == 2


def test_execute_container_help_stops_preparing_candidates_after_success(monkeypatch) -> None:
    record = ToolRecord(
        requirement_packages=["ampligone"],
        command_text="ampligone '$input'",
        container_candidate_details=[
            {
                "image": "quay.io/biocontainers/ampligone:2.0.2--pyhdfd78af_0",
                "source": "mulled-single",
                "packages": ["ampligone"],
                "priority": 300,
                "status": "ok",
                "error_text": "",
            },
            {
                "image": "quay.io/biocontainers/ampligone:2.0.1--pyhdfd78af_0",
                "source": "mulled-single",
                "packages": ["ampligone"],
                "priority": 200,
                "status": "ok",
                "error_text": "",
            },
            {
                "image": "quay.io/biocontainers/ampligone:1.3.1--pyhdfd78af_0",
                "source": "mulled-single",
                "packages": ["ampligone"],
                "priority": 100,
                "status": "ok",
                "error_text": "",
            },
        ],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [ContainerRuntime("singularity", "singularity")],
    )
    prepared_images = []

    def fake_prepare(image, runtime, settings):
        prepared_images.append(image)
        return ContainerPreparation(
            ok=True,
            runtime="singularity",
            image=image,
            identifier=image,
            source="cache",
        )

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._prepare_container", fake_prepare)

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        if "command -v ampligone" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/ampligone\n", stderr=""
            )
        if "ampligone --help" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="Usage: ampligone [options]\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert prepared_images == ["quay.io/biocontainers/ampligone:2.0.2--pyhdfd78af_0"]
    assert summary["images_prepared"] == 1
    assert summary["commands_executed"] == 1
    assert "$ ampligone --help" in record.container_help_text


def test_execute_container_help_reuses_prepared_image_for_multiple_records(monkeypatch) -> None:
    records = [
        ToolRecord(
            requirement_packages=["samtools"],
            command_text="samtools view '$input'",
            container_candidate_details=[
                {
                    "image": "quay.io/biocontainers/samtools:1.23--h96c455f_0",
                    "source": "mulled-single",
                    "packages": ["samtools"],
                    "priority": 300,
                    "status": "ok",
                    "error_text": "",
                }
            ],
        ),
        ToolRecord(
            requirement_packages=["samtools"],
            command_text="samtools faidx '$input'",
            container_candidate_details=[
                {
                    "image": "quay.io/biocontainers/samtools:1.23--h96c455f_0",
                    "source": "mulled-single",
                    "packages": ["samtools"],
                    "priority": 300,
                    "status": "ok",
                    "error_text": "",
                }
            ],
        ),
    ]
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [ContainerRuntime("singularity", "singularity")],
    )
    prepared_images = []

    def fake_prepare(image, runtime, settings):
        prepared_images.append(image)
        return ContainerPreparation(
            ok=True,
            runtime="singularity",
            image=image,
            identifier=image,
            source="cache",
        )

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._prepare_container", fake_prepare)

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        if "command -v samtools" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/samtools\n", stderr=""
            )
        if "samtools view --help" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="Usage: samtools view [options]\n", stderr=""
            )
        if "samtools faidx --help" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="Usage: samtools faidx [options]\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        records,
        ExtractionSettings(execute_containers=True, container_help_probe_mode="safe"),
    )

    assert prepared_images == ["quay.io/biocontainers/samtools:1.23--h96c455f_0"]
    assert summary["images_prepared"] == 1
    assert summary["commands_executed"] == 2
    assert "Usage: samtools view" in records[0].container_help_text
    assert "Usage: samtools faidx" in records[1].container_help_text


def test_execute_container_help_skips_duplicate_prepared_aliases(monkeypatch) -> None:
    record = ToolRecord(
        requirement_packages=["trimns_vgp"],
        command_text="trim_Ns_DNAnexus.py '$input' out.fa\nremove_fake_cut_sites_DNAnexus.py out.fa clean.fa",
        container_candidate_details=[
            {
                "image": "quay.io/biocontainers/trimns_vgp:1.0--py_0",
                "source": "mulled-single",
                "packages": ["trimns_vgp"],
                "priority": 300,
                "status": "ok",
                "error_text": "",
            },
            {
                "image": "https://depot.galaxyproject.org/singularity/trimns_vgp:1.0--py_0",
                "source": "biocontainers-api",
                "packages": ["trimns_vgp"],
                "priority": 200,
                "status": "ok",
                "error_text": "",
            },
            {
                "image": "trimns_vgp:1.0--py_0",
                "source": "biocontainers-api",
                "packages": ["trimns_vgp"],
                "priority": 100,
                "status": "ok",
                "error_text": "",
            },
        ],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._record_help_command_plan",
        lambda *args, **kwargs: [
            {
                "command": "trim_Ns_DNAnexus.py --help",
                "primary": "trim_Ns_DNAnexus.py",
                "probe_role": "core",
            },
            {
                "command": "trim_Ns_DNAnexus.py -h",
                "primary": "trim_Ns_DNAnexus.py",
                "probe_role": "core",
            },
            {
                "command": "remove_fake_cut_sites_DNAnexus.py --help",
                "primary": "remove_fake_cut_sites_DNAnexus.py",
                "probe_role": "core",
            },
            {
                "command": "remove_fake_cut_sites_DNAnexus.py -h",
                "primary": "remove_fake_cut_sites_DNAnexus.py",
                "probe_role": "core",
            },
        ],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [ContainerRuntime("singularity", "singularity")],
    )
    prepared_images = []

    def fake_prepare(image, runtime, settings):
        prepared_images.append(image)
        return ContainerPreparation(
            ok=True,
            runtime="singularity",
            image=image,
            identifier=str(_singularity_cache_path(image, settings=settings)),
            source="cache",
        )

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._prepare_container", fake_prepare)

    seen_probes = []

    def fake_run(command, timeout_seconds):
        probe = command[-1]
        seen_probes.append(probe)
        if "command -v trim_Ns_DNAnexus.py" in probe:
            return subprocess.CompletedProcess(
                command, 0, stdout="/usr/local/bin/trim_Ns_DNAnexus.py\n", stderr=""
            )
        if "command -v remove_fake_cut_sites_DNAnexus.py" in probe:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="/usr/local/bin/remove_fake_cut_sites_DNAnexus.py\n",
                stderr="",
            )
        if "trim_Ns_DNAnexus.py" in probe:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=(
                    "Traceback (most recent call last):\n"
                    '  File "/usr/local/bin/trim_Ns_DNAnexus.py", line 14, in main\n'
                    "    output_file = sys.argv[2]\n"
                    "IndexError: list index out of range\n"
                ),
            )
        if "remove_fake_cut_sites_DNAnexus.py" in probe:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=(
                    "Traceback (most recent call last):\n"
                    '  File "/usr/local/bin/remove_fake_cut_sites_DNAnexus.py", line 14, in main\n'
                    "    output_file = sys.argv[2]\n"
                    "IndexError: list index out of range\n"
                ),
            )
        return subprocess.CompletedProcess(command, 127, stdout="", stderr="unexpected probe")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._run_command", fake_run)

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, container_help_probe_mode="exploratory"),
    )

    assert prepared_images == ["quay.io/biocontainers/trimns_vgp:1.0--py_0"]
    assert summary["commands_executed"] == 2
    assert summary["failed_probe"] == 2
    assert "trim_Ns_DNAnexus.py -h" not in seen_probes
    assert "remove_fake_cut_sites_DNAnexus.py -h" not in seen_probes
    assert not record.container_help_text


def test_execute_container_help_records_nonzero_error_text(monkeypatch) -> None:
    record = ToolRecord(
        selected_container="quay.io/biocontainers/missing:1.0--0",
        primary_command="missing",
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._available_container_runtimes",
        lambda settings: [ContainerRuntime("docker", "docker")],
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._prepare_container",
        lambda image, runtime, settings: ContainerPreparation(
            ok=True,
            runtime="docker",
            image=image,
            identifier=image,
            source="docker-pull",
        ),
    )
    monkeypatch.setattr(
        "galaxy_toolsmith.data.corpus._run_command",
        lambda command, timeout_seconds: subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr="missing: command not found",
        ),
    )

    summary = _execute_container_help_batches(
        [record],
        ExtractionSettings(execute_containers=True, remove_images_after_use=False),
    )

    assert summary["commands_failed"] == 1
    assert record.container_execution[0]["returncode"] == 127
    assert record.container_execution[0]["error_text"] == "missing: command not found"
    assert record.container_help_text == ""
