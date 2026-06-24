from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.http_client import urlopen_with_user_agent_fallback


@dataclass(frozen=True)
class XSDSyncResult:
    url: str
    path: Path
    bytes_written: int


def sync_galaxy_xsd(paths: WorkspacePaths, ref: str = "dev") -> XSDSyncResult:
    """
    Download and cache the Galaxy tool schema used for wrapper validation.
    """
    url = f"https://raw.githubusercontent.com/galaxyproject/galaxy/{ref}/lib/galaxy/tool_util/xsd/galaxy.xsd"
    destination = paths.xsd_root / "galaxy.xsd"
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen_with_user_agent_fallback(url, timeout=60) as response:
        content = response.read()

    destination.write_bytes(content)
    return XSDSyncResult(url=url, path=destination, bytes_written=len(content))
