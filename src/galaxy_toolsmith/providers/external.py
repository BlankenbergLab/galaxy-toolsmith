from __future__ import annotations

import json
import os
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from galaxy_toolsmith.http_client import with_user_agent_headers
from galaxy_toolsmith.prompts import render_prompt_template
from galaxy_toolsmith.providers.base import (
    GenerationInput,
    GenerationOutput,
    extract_generated_artifact,
    generation_prompt_task,
)


def _prompt(request: GenerationInput) -> str:
    return render_prompt_template(
        task=generation_prompt_task(request),
        skills_profile=request.skills_profile,
        context={
            "tool_name": request.tool_name,
            "help_text": request.help_text,
            "source_code": request.source_code,
            "skills_profile": request.skills_profile,
            "repair_context": request.repair_context,
            "interface_hints": request.interface_hints,
        },
    )


def _post_json(url: str, headers: dict[str, str], payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = with_user_agent_headers(headers)
    req = urlrequest.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Request to {url} failed: {error.reason}") from error


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.api_key = os.getenv("GTSM_OPENAI_API_KEY")
        self.url = os.getenv("GTSM_OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.model = model or os.getenv("GTSM_OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        if not self.api_key:
            raise ValueError("Missing GTSM_OPENAI_API_KEY for provider=openai.")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a Galaxy tool artifact generator."},
                {"role": "user", "content": _prompt(request)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = _post_json(self.url, headers, payload)
        content = response["choices"][0]["message"]["content"]
        artifact = extract_generated_artifact(content, request.artifact_format)
        return GenerationOutput(
            artifact_text=artifact,
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.api_key = os.getenv("GTSM_ANTHROPIC_API_KEY")
        self.url = os.getenv("GTSM_ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
        self.model = model or os.getenv("GTSM_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        if not self.api_key:
            raise ValueError("Missing GTSM_ANTHROPIC_API_KEY for provider=anthropic.")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": _prompt(request)}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        response = _post_json(self.url, headers, payload)
        blocks = response.get("content", [])
        text_chunks = [item.get("text", "") for item in blocks if item.get("type") == "text"]
        content = "\n".join(text_chunks).strip()
        artifact = extract_generated_artifact(content, request.artifact_format)
        return GenerationOutput(
            artifact_text=artifact,
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
        )


class CopilotProvider:
    name = "copilot"

    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.api_key = os.getenv("GTSM_COPILOT_API_KEY")
        self.url = os.getenv("GTSM_COPILOT_BASE_URL", "https://api.githubcopilot.com/chat/completions")
        self.model = model or os.getenv("GTSM_COPILOT_MODEL", "gpt-4.1")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        if not self.api_key:
            raise ValueError("Missing GTSM_COPILOT_API_KEY for provider=copilot.")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a Galaxy tool artifact generator."},
                {"role": "user", "content": _prompt(request)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = _post_json(self.url, headers, payload)
        content = response["choices"][0]["message"]["content"]
        artifact = extract_generated_artifact(content, request.artifact_format)
        return GenerationOutput(
            artifact_text=artifact,
            artifact_format=request.artifact_format,
            provider=self.name,
            model_variant=request.model_variant,
        )
