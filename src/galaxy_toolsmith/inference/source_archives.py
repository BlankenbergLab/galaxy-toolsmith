from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from galaxy_toolsmith.http_client import urlopen_with_user_agent_fallback

DEFAULT_SOURCE_ARCHIVE_MAX_BYTES = 1_000_000_000
DEFAULT_SOURCE_ARCHIVE_TIMEOUT_SECONDS = 120
_CHUNK_SIZE = 1024 * 1024
_COMPLETE_MARKER = ".gtsm-source-archive-complete"
_METADATA_NAME = "source-archive.json"


@dataclass(frozen=True)
class SourceArchiveResolution:
    source: str
    source_type: str
    archive_path: str
    extracted_root: str
    cache_dir: str
    metadata_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_source_archive(
    source: str,
    *,
    cache_root: Path,
    max_bytes: int = DEFAULT_SOURCE_ARCHIVE_MAX_BYTES,
    timeout_seconds: int = DEFAULT_SOURCE_ARCHIVE_TIMEOUT_SECONDS,
) -> SourceArchiveResolution:
    source = str(source or "").strip()
    if not source:
        raise ValueError("source archive path or URL is empty")
    max_bytes = max(0, int(max_bytes or 0))
    timeout_seconds = max(1, int(timeout_seconds or DEFAULT_SOURCE_ARCHIVE_TIMEOUT_SECONDS))
    parsed = urlparse(source)
    if parsed.scheme.lower() in {"http", "https", "ftp"}:
        return _resolve_url_archive(
            source,
            cache_root=cache_root,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
        )
    return _resolve_local_archive(source, cache_root=cache_root, max_bytes=max_bytes)


def _resolve_url_archive(
    source_url: str,
    *,
    cache_root: Path,
    max_bytes: int,
    timeout_seconds: int,
) -> SourceArchiveResolution:
    key = hashlib.sha256(f"url:{source_url}".encode()).hexdigest()
    cache_dir = cache_root / key[:32]
    archive_name = _archive_name_from_url(source_url)
    archive_path = cache_dir / archive_name
    metadata_path = cache_dir / _METADATA_NAME
    complete_marker = cache_dir / _COMPLETE_MARKER
    if complete_marker.exists() and metadata_path.exists() and archive_path.exists():
        return _load_resolution(metadata_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f".{archive_name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    sha256 = hashlib.sha256()
    size_bytes = 0
    try:
        with tmp_path.open("wb") as handle, urlopen_with_user_agent_fallback(
            source_url,
            timeout=timeout_seconds,
        ) as response:
            headers = getattr(response, "headers", {}) or {}
            content_length = int(headers.get("content-length") or 0)
            if max_bytes and content_length > max_bytes:
                raise _source_archive_size_error(source_url, max_bytes)
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if max_bytes and size_bytes > max_bytes:
                    raise _source_archive_size_error(source_url, max_bytes)
                sha256.update(chunk)
                handle.write(chunk)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.replace(archive_path)
    return _extract_and_record(
        source=source_url,
        source_type="url",
        archive_path=archive_path,
        cache_dir=cache_dir,
        size_bytes=size_bytes,
        sha256=sha256.hexdigest(),
    )


def _resolve_local_archive(
    source_path: str,
    *,
    cache_root: Path,
    max_bytes: int,
) -> SourceArchiveResolution:
    original_path = Path(source_path).expanduser().resolve()
    if not original_path.is_file():
        raise FileNotFoundError(f"source archive does not exist or is not a file: {original_path}")

    sha256, size_bytes = _hash_file_with_cap(original_path, max_bytes=max_bytes)
    cache_dir = cache_root / sha256[:32]
    archive_path = cache_dir / original_path.name
    metadata_path = cache_dir / _METADATA_NAME
    complete_marker = cache_dir / _COMPLETE_MARKER
    if complete_marker.exists() and metadata_path.exists() and archive_path.exists():
        return _load_resolution(metadata_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f".{original_path.name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()
    shutil.copy2(original_path, tmp_path)
    tmp_path.replace(archive_path)
    return _extract_and_record(
        source=str(original_path),
        source_type="path",
        archive_path=archive_path,
        cache_dir=cache_dir,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _extract_and_record(
    *,
    source: str,
    source_type: str,
    archive_path: Path,
    cache_dir: Path,
    size_bytes: int,
    sha256: str,
) -> SourceArchiveResolution:
    extract_dir = cache_dir / "extracted"
    metadata_path = cache_dir / _METADATA_NAME
    complete_marker = cache_dir / _COMPLETE_MARKER
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        _safe_extract_archive(archive_path, extract_dir)
    except Exception:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        raise
    extracted_root = _single_top_level_dir(extract_dir) or extract_dir
    resolution = SourceArchiveResolution(
        source=source,
        source_type=source_type,
        archive_path=str(archive_path),
        extracted_root=str(extracted_root),
        cache_dir=str(cache_dir),
        metadata_path=str(metadata_path),
        size_bytes=size_bytes,
        sha256=sha256,
    )
    payload = {
        **resolution.to_dict(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    complete_marker.write_text("ok\n", encoding="utf-8")
    return resolution


def _safe_extract_archive(archive_path: Path, extract_dir: Path) -> None:
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            _safe_extract_tar(archive, extract_dir)
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract_zip(archive, extract_dir)
        return
    raise ValueError(f"unsupported source archive format: {archive_path}")


def _safe_extract_tar(archive: tarfile.TarFile, extract_dir: Path) -> None:
    for member in archive.getmembers():
        _validate_archive_member_name(member.name, extract_dir)
        if member.issym() or member.islnk():
            raise ValueError(f"source archive contains unsupported link entry: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"source archive contains unsupported special entry: {member.name}")
    for member in archive.getmembers():
        archive.extract(member, path=extract_dir)


def _safe_extract_zip(archive: zipfile.ZipFile, extract_dir: Path) -> None:
    for info in archive.infolist():
        _validate_archive_member_name(info.filename, extract_dir)
        if _zip_info_is_symlink(info):
            raise ValueError(f"source archive contains unsupported symlink entry: {info.filename}")
    for info in archive.infolist():
        target = _safe_member_target(info.filename, extract_dir)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source_handle, target.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)


def _validate_archive_member_name(name: str, extract_dir: Path) -> None:
    _safe_member_target(name, extract_dir)


def _safe_member_target(name: str, extract_dir: Path) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts:
        raise ValueError(f"source archive contains unsafe absolute path: {name}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"source archive contains unsafe relative path: {name}")
    target = extract_dir.joinpath(*pure.parts)
    try:
        common = os.path.commonpath([extract_dir.resolve(), target.resolve(strict=False)])
    except OSError as error:
        raise ValueError(f"source archive path cannot be resolved: {name}") from error
    if common != str(extract_dir.resolve()):
        raise ValueError(f"source archive member escapes extraction directory: {name}")
    return target


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _single_top_level_dir(extract_dir: Path) -> Path | None:
    children = list(extract_dir.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return None


def _hash_file_with_cap(path: Path, *, max_bytes: int) -> tuple[str, int]:
    sha256 = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            size_bytes += len(chunk)
            if max_bytes and size_bytes > max_bytes:
                raise _source_archive_size_error(str(path), max_bytes)
            sha256.update(chunk)
    return sha256.hexdigest(), size_bytes


def _source_archive_size_error(source: str, max_bytes: int) -> RuntimeError:
    return RuntimeError(
        f"source archive exceeds configured maximum of {max_bytes} bytes: {source}"
    )


def _archive_name_from_url(source_url: str) -> str:
    parsed = urlparse(source_url)
    name = unquote(Path(parsed.path).name or "source-archive")
    if not name or name in {".", ".."}:
        return "source-archive"
    return name


def _load_resolution(metadata_path: Path) -> SourceArchiveResolution:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return SourceArchiveResolution(
        source=str(payload.get("source", "")),
        source_type=str(payload.get("source_type", "")),
        archive_path=str(payload.get("archive_path", "")),
        extracted_root=str(payload.get("extracted_root", "")),
        cache_dir=str(payload.get("cache_dir", "")),
        metadata_path=str(payload.get("metadata_path", metadata_path)),
        size_bytes=int(payload.get("size_bytes") or 0),
        sha256=str(payload.get("sha256", "")),
    )
