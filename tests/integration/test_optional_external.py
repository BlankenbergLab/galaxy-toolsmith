from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.inference.validation import (
    PlanemoTestOptions,
    resolve_planemo_executable,
    run_planemo_test,
)
from galaxy_toolsmith.orchestration.export import create_ollama_model, write_ollama_modelfile


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


@pytest.mark.optional_external
@pytest.mark.planemo_live
def test_live_planemo_tool_test_smoke(tmp_path: Path) -> None:
    if not _truthy_env("GTSM_TEST_LIVE_PLANEMO"):
        pytest.skip("Set GTSM_TEST_LIVE_PLANEMO=1 to run live Planemo tests.")
    if resolve_planemo_executable() is None:
        pytest.skip("planemo is not available on PATH or next to the active Python executable.")

    galaxy_root = _optional_path_env("GTSM_TEST_PLANEMO_GALAXY_ROOT")
    install_galaxy = _truthy_env("GTSM_TEST_PLANEMO_INSTALL_GALAXY")
    if galaxy_root is None and not install_galaxy:
        pytest.skip(
            "Set GTSM_TEST_PLANEMO_GALAXY_ROOT or GTSM_TEST_PLANEMO_INSTALL_GALAXY=1."
        )

    test_data = tmp_path / "test-data"
    test_data.mkdir()
    (test_data / "input.txt").write_text("hello planemo\n", encoding="utf-8")
    (test_data / "expected.txt").write_text("hello planemo\n", encoding="utf-8")
    wrapper = tmp_path / "gtsm_optional_echo.xml"
    wrapper.write_text(
        """<tool id="gtsm_optional_echo" name="GTSM Optional Echo" version="0.1.0">
  <command><![CDATA[cat '$input' > '$output']]></command>
  <inputs>
    <param name="input" type="data" format="txt" />
  </inputs>
  <outputs>
    <data name="output" format="txt" />
  </outputs>
  <tests>
    <test>
      <param name="input" value="input.txt" />
      <output name="output" file="expected.txt" />
    </test>
  </tests>
</tool>
""",
        encoding="utf-8",
    )

    status, message, artifacts = run_planemo_test(
        wrapper,
        enabled=True,
        options=PlanemoTestOptions(
            output_dir=tmp_path / "planemo-report",
            timeout_seconds=int(os.getenv("GTSM_TEST_PLANEMO_TIMEOUT", "120")),
            galaxy_root=galaxy_root,
            install_galaxy=install_galaxy,
            engine=os.getenv("GTSM_TEST_PLANEMO_ENGINE", "").strip(),
            test_data=test_data,
            no_dependency_resolution=True,
        ),
    )

    assert status == "passed", f"planemo test {status}: {message}\nartifacts={artifacts}"


@pytest.mark.optional_external
@pytest.mark.ollama_live
def test_live_ollama_create_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not _truthy_env("GTSM_TEST_LIVE_OLLAMA"):
        pytest.skip("Set GTSM_TEST_LIVE_OLLAMA=1 to run live Ollama tests.")

    gguf_path = _optional_path_env("GTSM_TEST_OLLAMA_GGUF")
    if gguf_path is None:
        pytest.skip("Set GTSM_TEST_OLLAMA_GGUF to a real GGUF artifact.")
    if not gguf_path.is_file():
        pytest.skip(f"GTSM_TEST_OLLAMA_GGUF does not exist: {gguf_path}")
    if any(character.isspace() for character in str(gguf_path)):
        pytest.skip("GTSM_TEST_OLLAMA_GGUF cannot contain whitespace for current Modelfile output.")

    configured_ollama = _optional_path_env("GTSM_TEST_OLLAMA_BIN")
    ollama = str(configured_ollama) if configured_ollama is not None else shutil.which("ollama")
    if not ollama:
        pytest.skip("ollama executable is not available; set GTSM_TEST_OLLAMA_BIN.")
    if configured_ollama is not None:
        if not configured_ollama.is_file():
            pytest.skip(f"GTSM_TEST_OLLAMA_BIN does not exist: {configured_ollama}")
        monkeypatch.setenv(
            "PATH",
            f"{configured_ollama.parent}{os.pathsep}{os.getenv('PATH', '')}",
        )

    list_result = subprocess.run(
        [ollama, "list"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=15,
    )
    if list_result.returncode != 0:
        pytest.skip(f"Ollama server is not reachable: {list_result.stderr or list_result.stdout}")

    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    variant_id = "optional-ollama-live"
    export_root = paths.models_root / "exports" / variant_id
    export_root.mkdir(parents=True)
    (export_root / "export.result.json").write_text(
        json.dumps(
            {
                "variant_id": variant_id,
                "format": "gguf",
                "quantizations": ["q4_k_m"],
                "merged_path": "",
                "gguf_path": str(gguf_path),
                "gguf_paths": {"q4_k_m": str(gguf_path)},
                "status": "completed",
                "notes": [],
            }
        ),
        encoding="utf-8",
    )

    model_name = f"gtsm-optional-test-{uuid.uuid4().hex[:12]}"
    modelfile = write_ollama_modelfile(
        paths=paths,
        variant_id=variant_id,
        model_name=model_name,
        from_quantization="q4_k_m",
    )

    try:
        result = create_ollama_model(modelfile, model_name)
        assert result["returncode"] == 0
        show_result = subprocess.run(
            [ollama, "show", model_name],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert show_result.returncode == 0, show_result.stderr or show_result.stdout
    finally:
        subprocess.run(
            [ollama, "rm", model_name],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=30,
        )
