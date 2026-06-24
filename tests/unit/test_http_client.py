from __future__ import annotations

from galaxy_toolsmith.http_client import (
    DEFAULT_BROWSER_FALLBACK_USER_AGENT,
    DEFAULT_HTTP_USER_AGENT,
    HTTP_BROWSER_FALLBACK_USER_AGENT_ENV_VAR,
    HTTP_USER_AGENT_ENV_VAR,
    browser_fallback_user_agent,
    http_user_agent,
    user_agent_header_attempts,
    with_user_agent_headers,
)


def test_http_user_agent_default(monkeypatch) -> None:
    monkeypatch.delenv(HTTP_USER_AGENT_ENV_VAR, raising=False)
    monkeypatch.delenv(HTTP_BROWSER_FALLBACK_USER_AGENT_ENV_VAR, raising=False)

    assert http_user_agent() == DEFAULT_HTTP_USER_AGENT
    assert browser_fallback_user_agent() == DEFAULT_BROWSER_FALLBACK_USER_AGENT
    assert with_user_agent_headers({"Accept": "application/json"}) == {
        "Accept": "application/json",
        "User-Agent": DEFAULT_HTTP_USER_AGENT,
    }


def test_http_user_agent_env_override(monkeypatch) -> None:
    monkeypatch.setenv(HTTP_USER_AGENT_ENV_VAR, "Galaxy-Toolsmith-Test/1.0")
    monkeypatch.setenv(HTTP_BROWSER_FALLBACK_USER_AGENT_ENV_VAR, "Mozilla/Test")

    assert http_user_agent() == "Galaxy-Toolsmith-Test/1.0"
    assert browser_fallback_user_agent() == "Mozilla/Test"
    assert with_user_agent_headers()["User-Agent"] == "Galaxy-Toolsmith-Test/1.0"


def test_user_agent_header_attempts_preserve_headers(monkeypatch) -> None:
    monkeypatch.delenv(HTTP_USER_AGENT_ENV_VAR, raising=False)
    monkeypatch.delenv(HTTP_BROWSER_FALLBACK_USER_AGENT_ENV_VAR, raising=False)

    attempts = user_agent_header_attempts({"Accept": "application/json"})

    assert attempts == [
        {
            "Accept": "application/json",
            "User-Agent": DEFAULT_HTTP_USER_AGENT,
        },
        {
            "Accept": "application/json",
            "User-Agent": DEFAULT_BROWSER_FALLBACK_USER_AGENT,
        },
    ]
