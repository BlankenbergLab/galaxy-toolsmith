from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from galaxy_toolsmith import __version__
from galaxy_toolsmith.runtime.capabilities import detect_runtime_capabilities

KEY_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "axolotl",
    "unsloth",
    "fastapi",
)

KEY_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "AXOLOTL_DO_NOT_TRACK",
)

NVIDIA_SMI_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class NvidiaSmiResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    error: str = ""


def _package_versions() -> dict[str, str]:
    versions = {"galaxy_toolsmith": __version__}
    for package in KEY_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = ""
    return versions


def _output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _preview_output(*values: str, limit: int = 240) -> str:
    text = "\n".join(value.strip() for value in values if value.strip()).strip()
    if not text:
        return ""
    first_line = text.splitlines()[0]
    return first_line[:limit]


def _run_nvidia_smi(args: list[str]) -> NvidiaSmiResult:
    command = ["nvidia-smi", *args]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        stdout = _output_text(getattr(error, "stdout", None) or getattr(error, "output", None))
        stderr = _output_text(getattr(error, "stderr", None))
        return NvidiaSmiResult(
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            error=(
                f"{' '.join(command)} timed out after "
                f"{NVIDIA_SMI_TIMEOUT_SECONDS:g}s"
            ),
        )
    except OSError as error:
        return NvidiaSmiResult(error=f"{type(error).__name__}: {error}")

    stdout = _output_text(result.stdout)
    stderr = _output_text(result.stderr)
    error_message = ""
    if result.returncode != 0:
        detail = _preview_output(stderr, stdout)
        error_message = f"{' '.join(command)} exited with code {result.returncode}"
        if detail:
            error_message = f"{error_message}: {detail}"
    return NvidiaSmiResult(
        stdout=stdout,
        stderr=stderr,
        returncode=result.returncode,
        error=error_message,
    )


def _gpu_summary() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "cuda_version": "",
            "gpus": [],
            "nvidia_smi_timeout_seconds": NVIDIA_SMI_TIMEOUT_SECONDS,
            "nvidia_smi_timed_out": False,
            "nvidia_smi_errors": [],
        }

    query_result = _run_nvidia_smi(
        [
            "--query-gpu=index,name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, str]] = []
    for line in query_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mib": parts[2],
                "memory_used_mib": parts[3],
                "driver_version": parts[4],
            }
        )

    cuda_version = ""
    version_result = _run_nvidia_smi(["--version"])
    match = re.search(r"CUDA Version\s*:\s*([0-9.]+)", version_result.stdout)
    if match:
        cuda_version = match.group(1)
    errors = [result.error for result in (query_result, version_result) if result.error]

    return {
        "available": True,
        "cuda_version": cuda_version,
        "gpus": gpus,
        "nvidia_smi_timeout_seconds": NVIDIA_SMI_TIMEOUT_SECONDS,
        "nvidia_smi_timed_out": query_result.timed_out or version_result.timed_out,
        "nvidia_smi_errors": errors,
    }


def collect_environment_snapshot(*, cwd: Path | None = None) -> dict[str, Any]:
    return {
        "gtsm_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "hostname": socket.gethostname(),
        "cwd": str(cwd or Path.cwd()),
        "environment_variables": {name: os.getenv(name, "") for name in KEY_ENV_VARS},
        "runtime_capabilities": detect_runtime_capabilities().to_dict(),
        "package_versions": _package_versions(),
        "gpu_summary": _gpu_summary(),
    }
