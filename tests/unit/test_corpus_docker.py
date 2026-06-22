from __future__ import annotations

import io
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from galaxy_toolsmith.data.corpus import (
    BiocondaRecipeSnapshot,
    ContainerPreparation,
    ContainerRuntime,
    ExtractionSettings,
    MulledTarget,
    ToolRecord,
    _build_container_candidate_details,
    _choose_container_image,
    _container_help_fragment,
    _container_probe_status,
    _container_probe_timeout,
    _container_run_command,
    _docker_ref_for_image,
    _download_and_extract_archive,
    _execute_container_help_batches,
    _ensure_conda_forge_feedstock_repo,
    _extract_help_commands,
    _extract_source_fields,
    _image_matches_requirement_versions,
    _infer_command_signatures,
    _mulled_biocontainer_images,
    _mulled_v2_image_name,
    _normalize_container_candidate,
    _prepare_docker_container,
    _prepare_singularity_container,
    _record_help_commands,
    _render_bioconda_source_fields,
    _resolve_bioconda_source_mappings,
    _select_recipe_snapshot_from_candidates,
    _singularity_cache_path,
    _singularity_depot_image_url,
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


def test_download_and_extract_archive_supports_ftp(monkeypatch, tmp_path) -> None:
    checkout = tmp_path / "source"
    payload = b"ftp source archive"
    calls = []

    class FakeFtpResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return FakeFtpResponse(payload)

    monkeypatch.setattr("galaxy_toolsmith.data.corpus.urlrequest.urlopen", fake_urlopen)

    archive = _download_and_extract_archive("ftp://example.org/pub/tool.tar.gz", checkout)

    assert archive == str(checkout / "tool.tar.gz")
    assert (checkout / "tool.tar.gz").read_bytes() == payload
    assert calls == [("ftp://example.org/pub/tool.tar.gz", 120)]


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

    def fail_download(source_url, checkout_dir):
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

    def fake_download(source_url, checkout_dir):
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
        lambda source_url, checkout_dir: str(checkout_dir / "source.tar.gz"),
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
        lambda source_url, checkout_dir: str(checkout_dir / "source.tar.gz"),
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

    def fake_download(source_url, checkout_dir):
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

    def fake_download(source_url, checkout_dir):
        checkout_names.append(checkout_dir.name)
        return str(checkout_dir / "source.tar.gz")

    monkeypatch.setattr("galaxy_toolsmith.data.corpus._download_and_extract_archive", fake_download)

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
    assert checkout_names == ["fallbacktool-required-1.5--recipe-1.7"]


def test_infer_command_signatures_filters_wrapper_setup_and_files() -> None:
    assert _infer_command_signatures("mkdir ./tabular\n./tabular --option")[0] == ""
    assert _infer_command_signatures("cp annotatemyids_script out_rscript")[0] == ""
    assert _infer_command_signatures("ln input_bam localbam.bam")[0] == ""
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


def test_no_arg_probe_uses_short_timeout() -> None:
    settings = ExtractionSettings(
        container_run_timeout_seconds=120, container_no_arg_timeout_seconds=7
    )
    assert _container_probe_timeout("samtools", settings) == 7
    assert _container_probe_timeout("samtools --help", settings) == 120
    assert _container_probe_timeout("samtools help", settings) == 120


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
