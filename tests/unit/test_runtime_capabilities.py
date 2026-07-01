from __future__ import annotations

import galaxy_toolsmith.runtime.capabilities as capabilities_mod


def test_mps_available_on_darwin_arm64(monkeypatch) -> None:
    monkeypatch.setattr(capabilities_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(capabilities_mod.platform, "machine", lambda: "arm64")

    assert capabilities_mod._mps_available() is True
