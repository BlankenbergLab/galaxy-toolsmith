from __future__ import annotations

import json
from pathlib import Path
import time
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor


def request_remote_generation(
    server_url: str,
    auth_token: str | None,
    payload: dict,
    retries: int = 2,
) -> dict:
    url = server_url.rstrip("/") + "/generate"
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        import httpx

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as error:
                last_error = error
                if attempt >= retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Remote generation request failed after retries: {last_error}")
    except ImportError:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote generation failed HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Remote generation request failed: {error.reason}") from error


def request_remote_json(
    *,
    server_url: str,
    endpoint: str,
    method: str,
    auth_token: str | None,
    payload: dict | None = None,
    timeout: float = 120.0,
) -> dict:
    url = server_url.rstrip("/") + endpoint
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, headers=headers, json=payload)
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()
    except ImportError:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urlrequest.Request(url=url, data=data, headers=headers, method=method.upper())
        try:
            with urlrequest.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote request failed HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Remote request failed: {error.reason}") from error


def download_remote_file(
    *,
    server_url: str,
    endpoint: str,
    auth_token: str | None,
    destination: Path,
    timeout: float = 300.0,
) -> Path:
    url = server_url.rstrip("/") + endpoint
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination
    except ImportError:
        req = urlrequest.Request(url=url, headers=headers, method="GET")
        try:
            with urlrequest.urlopen(req, timeout=timeout) as response:
                destination.write_bytes(response.read())
            return destination
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote download failed HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Remote download failed: {error.reason}") from error


def fetch_training_artifacts_parallel(
    *,
    server_url: str,
    job_id: str,
    output_dir: Path,
    auth_token: str | None,
    max_workers: int,
) -> dict:
    listing = request_remote_json(
        server_url=server_url,
        endpoint=f"/train/artifacts/{urlparse.quote(job_id)}",
        method="GET",
        auth_token=auth_token,
    )
    artifacts = list(listing.get("artifacts", []))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not artifacts:
        return {"downloaded": [], "count": 0}

    def _download(item: dict) -> dict:
        artifact_id = str(item.get("artifact_id", ""))
        name = str(item.get("name", f"{artifact_id}.bin"))
        target = output_dir / name
        endpoint = f"/train/artifacts/{urlparse.quote(job_id)}/download/{urlparse.quote(artifact_id)}"
        path = download_remote_file(
            server_url=server_url,
            endpoint=endpoint,
            auth_token=auth_token,
            destination=target,
        )
        return {"artifact_id": artifact_id, "path": str(path), "size_bytes": path.stat().st_size}

    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        downloaded = list(pool.map(_download, artifacts))
    return {"downloaded": downloaded, "count": len(downloaded)}
