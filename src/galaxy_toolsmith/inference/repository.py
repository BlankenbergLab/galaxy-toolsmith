from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ToolShedMetadata:
    name: str = ""
    owner: str = ""
    description: str = ""
    homepage_url: str = ""
    remote_repository_url: str = ""
    categories: tuple[str, ...] = field(default_factory=tuple)
    suite: bool = False
    repositories: tuple[str, ...] = field(default_factory=tuple)

    def to_shed_yml_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.name:
            payload["name"] = self.name
        if self.owner:
            payload["owner"] = self.owner
        if self.description:
            payload["description"] = self.description
        if self.homepage_url:
            payload["homepage_url"] = self.homepage_url
        if self.remote_repository_url:
            payload["remote_repository_url"] = self.remote_repository_url
        categories = [category for category in self.categories if category]
        if categories:
            payload["categories"] = categories
        if self.suite:
            payload["type"] = "suite_repository"
            payload["repositories"] = [
                {"name": repository} for repository in self.repositories if repository
            ]
        return payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_repository_name(value: str, *, default: str = "generated_tool") -> str:
    text = _SAFE_NAME_RE.sub("_", str(value or "").strip().lower()).strip("_.-")
    if not text:
        text = default
    if not re.match(r"^[A-Za-z]", text):
        text = f"tool_{text}"
    return text


def safe_tool_id(value: str, *, default: str = "generated_tool") -> str:
    return safe_repository_name(value, default=default)


def build_tool_shed_metadata(
    *,
    name: str,
    owner: str = "",
    description: str = "",
    homepage_url: str = "",
    remote_repository_url: str = "",
    categories: list[str] | tuple[str, ...] | None = None,
    suite: bool = False,
    repositories: list[str] | tuple[str, ...] | None = None,
) -> ToolShedMetadata:
    repo_names = tuple(safe_repository_name(item) for item in (repositories or ()) if str(item).strip())
    metadata_name = safe_repository_name(name, default="suite_generated_tools" if suite else "generated_tool")
    if suite and not metadata_name.startswith("suite_"):
        metadata_name = f"suite_{metadata_name}"
    return ToolShedMetadata(
        name=metadata_name,
        owner=str(owner or "").strip(),
        description=str(description or "").strip(),
        homepage_url=str(homepage_url or "").strip(),
        remote_repository_url=str(remote_repository_url or "").strip(),
        categories=tuple(str(category).strip() for category in (categories or ()) if str(category).strip()),
        suite=suite,
        repositories=repo_names,
    )


def write_shed_yml(path: Path, metadata: ToolShedMetadata) -> Path:
    payload = metadata.to_shed_yml_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_gtsm_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
