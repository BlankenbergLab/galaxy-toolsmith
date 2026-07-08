from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template

_PROMPTS_ROOT = Path(__file__).resolve().parent / "templates"


def _template_path(skills_profile: str, task: str) -> Path:
    return _PROMPTS_ROOT / skills_profile / f"{task}.txt"


def _default_template_path(task: str) -> Path:
    return _PROMPTS_ROOT / "default" / f"{task}.txt"


def render_prompt_template(task: str, context: dict[str, Any], skills_profile: str = "default") -> str:
    candidate = _template_path(skills_profile, task)
    if candidate.exists():
        raw = candidate.read_text(encoding="utf-8")
    else:
        fallback = _default_template_path(task)
        if not fallback.exists():
            raise FileNotFoundError(
                f"Prompt template not found for task={task!r}, skills_profile={skills_profile!r}"
            )
        raw = fallback.read_text(encoding="utf-8")
    template = Template(raw, undefined=StrictUndefined)
    return str(template.render(**context)).strip()
