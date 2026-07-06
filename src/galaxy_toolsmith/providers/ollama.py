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
    generation_output_from_response,
    generation_prompt_task,
)


def _positive_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = int(text)
    return parsed if parsed > 0 else None


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        context_tokens: int | None = None,
    ):
        base = os.getenv("GTSM_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.base_url = base
        self.url = f"{base}/api/generate"
        self.model = model or os.getenv("GTSM_OLLAMA_MODEL", "llama3")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_tokens = _positive_int_or_none(
            context_tokens
            if context_tokens is not None
            else os.getenv("GTSM_OLLAMA_CONTEXT_TOKENS")
        )
        self.timeout_seconds = int(os.getenv("GTSM_OLLAMA_TIMEOUT_SECONDS", "120"))
        self.auth_header = os.getenv("GTSM_OLLAMA_AUTH_HEADER", "").strip()

    def _prompt(self, request: GenerationInput) -> str:
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
                "generate_sidecars": request.generate_sidecars,
            },
        )

    def generate_wrapper(self, request: GenerationInput) -> GenerationOutput:
        payload = {
            "model": self.model,
            "prompt": self._prompt(request),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if self.context_tokens is not None:
            payload["options"]["num_ctx"] = self.context_tokens
        headers = with_user_agent_headers({"Content-Type": "application/json"})
        if self.auth_header:
            if ":" in self.auth_header:
                key, value = self.auth_header.split(":", 1)
                headers[key.strip()] = value.strip()
            else:
                headers["Authorization"] = self.auth_header
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url=self.url, data=body, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(
                "Ollama request failed for "
                f"{self.url}: {error.reason}. "
                "Ensure Ollama is running and reachable, or set GTSM_OLLAMA_BASE_URL "
                "to the correct server."
            ) from error

        content = str(data.get("response", "")).strip()
        if not content:
            raise RuntimeError("Ollama returned empty response.")
        return generation_output_from_response(
            response_text=content,
            request=request,
            provider=self.name,
        )
