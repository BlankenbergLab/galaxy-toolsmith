from __future__ import annotations

from typing import Any
from urllib.error import URLError

import pytest

from galaxy_toolsmith.providers.base import GenerationInput
from galaxy_toolsmith.providers.ollama import OllamaProvider


def _request() -> GenerationInput:
    return GenerationInput(
        tool_name="echo_tool",
        help_text="Usage: echo_tool --input TEXT",
        source_code="",
        model_variant="ollama-variant",
    )


def test_ollama_provider_reports_connection_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GTSM_OLLAMA_BASE_URL", "http://ollama.example:11434/")

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr(
        "galaxy_toolsmith.providers.ollama.urlrequest.urlopen",
        fake_urlopen,
    )
    provider = OllamaProvider(model="tool-model", temperature=0, max_tokens=64)

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate_wrapper(_request())

    message = str(exc_info.value)
    assert "http://ollama.example:11434/api/generate" in message
    assert "GTSM_OLLAMA_BASE_URL" in message
    assert "Connection refused" in message


def test_ollama_provider_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"response": "<tool id=\\"echo_tool\\" name=\\"Echo Tool\\"></tool>"}'

    def fake_urlopen(*_args: Any, **kwargs: Any) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setenv("GTSM_OLLAMA_TIMEOUT_SECONDS", "900")
    monkeypatch.setattr(
        "galaxy_toolsmith.providers.ollama.urlrequest.urlopen",
        fake_urlopen,
    )
    provider = OllamaProvider(model="tool-model", temperature=0, max_tokens=64)

    output = provider.generate_wrapper(_request())

    assert captured["timeout"] == 900
    assert output.artifact_text == '<tool id="echo_tool" name="Echo Tool"></tool>'
