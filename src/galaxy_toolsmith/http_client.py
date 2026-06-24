from __future__ import annotations

import os
from collections.abc import Mapping
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from galaxy_toolsmith import __version__

HTTP_USER_AGENT_ENV_VAR = "GTSM_HTTP_USER_AGENT"
HTTP_BROWSER_FALLBACK_USER_AGENT_ENV_VAR = "GTSM_HTTP_BROWSER_FALLBACK_USER_AGENT"
DEFAULT_HTTP_USER_AGENT = (
    f"Galaxy-Toolsmith/{__version__} "
    "(+https://github.com/BlankenbergLab/galaxy-toolsmith)"
)
DEFAULT_BROWSER_FALLBACK_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def http_user_agent() -> str:
    return os.getenv(HTTP_USER_AGENT_ENV_VAR, "").strip() or DEFAULT_HTTP_USER_AGENT


def browser_fallback_user_agent() -> str:
    return (
        os.getenv(HTTP_BROWSER_FALLBACK_USER_AGENT_ENV_VAR, "").strip()
        or DEFAULT_BROWSER_FALLBACK_USER_AGENT
    )


def with_user_agent_headers(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = dict(headers or {})
    merged.setdefault("User-Agent", http_user_agent())
    return merged


def user_agent_header_attempts(headers: Mapping[str, str] | None = None) -> list[dict[str, str]]:
    primary = with_user_agent_headers(headers)
    fallback = dict(headers or {})
    fallback["User-Agent"] = browser_fallback_user_agent()
    if fallback["User-Agent"] == primary.get("User-Agent"):
        return [primary]
    return [primary, fallback]


def should_retry_with_browser_user_agent(status_code: int) -> bool:
    return status_code in {403, 406, 429} or 500 <= status_code < 600


def url_request_with_user_agent(
    url: str,
    *,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    method: str | None = None,
) -> urlrequest.Request:
    return urlrequest.Request(
        url=url,
        data=data,
        headers=with_user_agent_headers(headers),
        method=method,
    )


def url_request_user_agent_attempts(
    url: str,
    *,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    method: str | None = None,
) -> list[urlrequest.Request]:
    return [
        urlrequest.Request(url=url, data=data, headers=attempt_headers, method=method)
        for attempt_headers in user_agent_header_attempts(headers)
    ]


def urlopen_with_user_agent_fallback(
    url: str,
    *,
    timeout: float,
    data: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    method: str | None = None,
):
    requests = url_request_user_agent_attempts(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    last_error: HTTPError | URLError | None = None
    for index, request in enumerate(requests):
        try:
            response = urlrequest.urlopen(request, timeout=timeout)
            setattr(response, "gtsm_user_agent_attempt_index", index)
            setattr(response, "gtsm_user_agent", request.get_header("User-agent") or "")
            setattr(response, "gtsm_user_agent_fallback", index > 0)
            return response
        except HTTPError as error:
            last_error = error
            if index + 1 < len(requests) and should_retry_with_browser_user_agent(error.code):
                continue
            raise
        except URLError as error:
            last_error = error
            if index + 1 < len(requests):
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No HTTP request attempts were generated for {url}")
