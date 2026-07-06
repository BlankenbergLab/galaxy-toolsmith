from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from galaxy_toolsmith.inference.source_archives import resolve_source_archive
from galaxy_toolsmith.inference.source_context import (
    build_source_context_from_paths,
    source_context_settings,
)


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None):
        super().__init__(payload)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _write_tar_gz(path: Path, root_name: str = "tool-1.0") -> None:
    source_root = path.parent / root_name
    (source_root / "src").mkdir(parents=True)
    (source_root / "src" / "cli.py").write_text(
        "from argparse import ArgumentParser\n",
        encoding="utf-8",
    )
    with tarfile.open(path, "w:gz") as archive:
        archive.add(source_root, arcname=root_name)


def test_resolve_source_archive_extracts_local_tar_single_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "tool.tar.gz"
    _write_tar_gz(archive_path)

    result = resolve_source_archive(
        str(archive_path),
        cache_root=tmp_path / "cache",
        max_bytes=1_000_000,
    )

    extracted_root = Path(result.extracted_root)
    assert extracted_root.name == "tool-1.0"
    assert (extracted_root / "src" / "cli.py").read_text(encoding="utf-8").startswith(
        "from argparse"
    )
    assert Path(result.metadata_path).exists()


def test_resolved_source_archive_can_feed_source_context(tmp_path: Path) -> None:
    archive_path = tmp_path / "tool.tar.gz"
    _write_tar_gz(archive_path)
    result = resolve_source_archive(
        str(archive_path),
        cache_root=tmp_path / "cache",
        max_bytes=1_000_000,
    )

    context = build_source_context_from_paths(
        settings=source_context_settings(
            mode="snippets",
            source_root=Path(result.extracted_root),
            max_chars=5000,
            max_files=4,
        )
    )

    assert "Source file: src/cli.py" in context.text
    assert "ArgumentParser" in context.text


def test_resolve_source_archive_extracts_zip_multi_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "tool.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cli.py", "print('cli')\n")
        archive.writestr("pkg/core.py", "print('core')\n")

    result = resolve_source_archive(
        str(archive_path),
        cache_root=tmp_path / "cache",
        max_bytes=1_000_000,
    )

    extracted_root = Path(result.extracted_root)
    assert extracted_root.name == "extracted"
    assert (extracted_root / "cli.py").exists()
    assert (extracted_root / "pkg" / "core.py").exists()


def test_resolve_source_archive_downloads_url_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "source.tar.gz"
    _write_tar_gz(archive_path)
    payload = archive_path.read_bytes()
    calls: list[tuple[str, int]] = []

    def fake_urlopen(url: str, *, timeout: int):
        calls.append((url, timeout))
        return _FakeResponse(payload, headers={"content-length": str(len(payload))})

    monkeypatch.setattr(
        "galaxy_toolsmith.inference.source_archives.urlopen_with_user_agent_fallback",
        fake_urlopen,
    )

    result = resolve_source_archive(
        "https://example.org/source.tar.gz",
        cache_root=tmp_path / "cache",
        max_bytes=1_000_000,
        timeout_seconds=17,
    )
    assert Path(result.extracted_root, "src", "cli.py").exists()
    assert calls == [("https://example.org/source.tar.gz", 17)]

    def fail_urlopen(url: str, *, timeout: int):  # pragma: no cover - should not run
        raise AssertionError("cached URL archive should not be downloaded again")

    monkeypatch.setattr(
        "galaxy_toolsmith.inference.source_archives.urlopen_with_user_agent_fallback",
        fail_urlopen,
    )
    cached = resolve_source_archive(
        "https://example.org/source.tar.gz",
        cache_root=tmp_path / "cache",
        max_bytes=1_000_000,
    )
    assert cached.extracted_root == result.extracted_root


def test_resolve_source_archive_rejects_oversized_content_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_urlopen(url: str, *, timeout: int):
        return _FakeResponse(b"", headers={"content-length": "100"})

    monkeypatch.setattr(
        "galaxy_toolsmith.inference.source_archives.urlopen_with_user_agent_fallback",
        fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="source archive exceeds configured maximum"):
        resolve_source_archive(
            "https://example.org/large.tar.gz",
            cache_root=tmp_path / "cache",
            max_bytes=5,
        )

    assert not list((tmp_path / "cache").glob("**/*.tmp"))


def test_resolve_source_archive_rejects_streaming_over_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_urlopen(url: str, *, timeout: int):
        return _FakeResponse(b"0123456789", headers={})

    monkeypatch.setattr(
        "galaxy_toolsmith.inference.source_archives.urlopen_with_user_agent_fallback",
        fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="source archive exceeds configured maximum"):
        resolve_source_archive(
            "https://example.org/large.tar.gz",
            cache_root=tmp_path / "cache",
            max_bytes=5,
        )

    assert not list((tmp_path / "cache").glob("**/*.tmp"))


def test_resolve_source_archive_rejects_tar_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar"
    payload = b"evil"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("../evil.py")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe relative path"):
        resolve_source_archive(str(archive_path), cache_root=tmp_path / "cache")


def test_resolve_source_archive_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../evil.py", "evil\n")

    with pytest.raises(ValueError, match="unsafe relative path"):
        resolve_source_archive(str(archive_path), cache_root=tmp_path / "cache")


def test_resolve_source_archive_rejects_tar_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad-links.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("link.py")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/target.py"
        archive.addfile(info)

    with pytest.raises(ValueError, match="unsupported link entry"):
        resolve_source_archive(str(archive_path), cache_root=tmp_path / "cache")
