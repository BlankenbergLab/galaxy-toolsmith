from __future__ import annotations

import platform
import shutil
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    platform: str
    machine: str
    cpu_available: bool
    cuda_available: bool
    rocm_available: bool
    mps_available: bool
    recommended_backend: str

    def to_dict(self) -> dict:
        return asdict(self)


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _cuda_available() -> bool:
    return bool(_has_command("nvidia-smi"))


def _rocm_available() -> bool:
    return bool(_has_command("rocminfo"))


def _mps_available() -> bool:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "darwin":
        return False
    return machine in {"arm64", "aarch64"}


def detect_runtime_capabilities() -> RuntimeCapabilities:
    cuda = _cuda_available()
    rocm = _rocm_available()
    mps = _mps_available()
    if cuda:
        backend = "cuda"
    elif rocm:
        backend = "rocm"
    elif mps:
        backend = "mps"
    else:
        backend = "cpu"

    return RuntimeCapabilities(
        platform=platform.system(),
        machine=platform.machine(),
        cpu_available=True,
        cuda_available=cuda,
        rocm_available=rocm,
        mps_available=mps,
        recommended_backend=backend,
    )
