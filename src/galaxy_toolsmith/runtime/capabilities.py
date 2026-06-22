from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import shutil
import subprocess


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
    if _has_command("nvidia-smi"):
        return True
    return False


def _rocm_available() -> bool:
    if _has_command("rocminfo"):
        return True
    return False


def _mps_available() -> bool:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "darwin":
        return False
    if machine not in {"arm64", "aarch64"}:
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return True


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
