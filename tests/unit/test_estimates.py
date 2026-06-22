from __future__ import annotations

import json

from galaxy_toolsmith.runtime.estimates import model_estimates_json


def test_model_estimates_json_includes_expected_mps_profiles() -> None:
    payload = json.loads(model_estimates_json())
    profiles = {entry["profile"] for entry in payload["estimates"]}
    assert "mps-qwen25-14b" in profiles
    assert "mps-mistral31-24b" in profiles
    assert "mps-qwen25-32b" in profiles
