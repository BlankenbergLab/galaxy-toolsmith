from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gtsm_context_ladder_train.sh"


def _script_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CONTEXT_LADDER_ROOT": str(tmp_path / "context-ladder"),
            "RUN_TAG": "unit-test",
        }
    )
    return env


def test_context_ladder_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=REPO_ROOT, check=True)


def test_context_ladder_classifies_stale_gpu_oom_as_contaminated(tmp_path: Path) -> None:
    log_path = tmp_path / "status.jsonl"
    log_path.write_text(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 640.00 MiB. "
        "GPU 1 has a total capacity of 39.49 GiB of which 536.94 MiB is free. "
        "Process 511822 has 20.46 GiB memory in use.\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT), "classify-contamination", str(log_path)],
        cwd=REPO_ROOT,
        env=_script_env(tmp_path),
        check=False,
    )

    assert completed.returncode == 0


def test_context_ladder_does_not_classify_ordinary_oom_as_contaminated(tmp_path: Path) -> None:
    log_path = tmp_path / "status.jsonl"
    log_path.write_text(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.45 GiB. "
        "GPU 0 has a total capacity of 39.49 GiB of which 3.33 GiB is free. "
        "Including non-PyTorch memory, this process has 36.15 GiB memory in use.\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT), "classify-contamination", str(log_path)],
        cwd=REPO_ROOT,
        env=_script_env(tmp_path),
        check=False,
    )

    assert completed.returncode == 1


def test_context_ladder_cleanup_matches_axolotl_module_workers() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert r"-m\s+axolotl\.cli\.train" in script_text
    assert "pt_data_worker" in script_text


def test_context_ladder_supports_external_gguf_export_env() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert "POST_EXPORT_ENV_DIR" in script_text
    assert "POST_EXPORT_LLAMA_CPP_DIR" in script_text
    assert "run_external_post_export" in script_text
    assert "scripts/gtsm_llama_cpp_gguf.sh" in script_text
    assert "post-export-failed" in script_text


def test_context_ladder_forwards_test_context_to_estimation_and_training() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert ": \"${TEST_CONTEXT_MODE:=fixtures}\"" in script_text
    assert ": \"${TEST_CONTEXT_MAX_CHARS:=4000}\"" in script_text
    assert ": \"${TEST_CONTEXT_MAX_FILES:=6}\"" in script_text
    assert ": \"${TEST_CONTEXT_MAX_FILE_BYTES:=64KB}\"" in script_text
    assert script_text.count("--test-context-mode") >= 3
    assert script_text.count("--test-context-max-chars") >= 3
    assert script_text.count("--test-context-max-files") >= 3
    assert script_text.count("--test-context-max-file-bytes") >= 3
    assert "\"TEST_CONTEXT_MODE=$TEST_CONTEXT_MODE\"" in script_text
    assert "printf 'TEST_CONTEXT_MODE=%q\\n' \"$TEST_CONTEXT_MODE\"" in script_text
