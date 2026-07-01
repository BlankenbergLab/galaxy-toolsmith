from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def _config_path_from_args(args: Sequence[str]) -> Path | None:
    for index, arg in enumerate(args):
        if arg == "--config" and index + 1 < len(args):
            return Path(args[index + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return None


def _tokenizer_config_from_yaml(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        return {}
    tokenizer_config = data.get("tokenizer_config") or {}
    if not isinstance(tokenizer_config, Mapping):
        return {}
    return dict(tokenizer_config)


def _merge_tokenizer_config(
    existing: Mapping[str, Any] | None,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(dict(extra))
    return merged


def main() -> None:
    tokenizer_config_extra = _tokenizer_config_from_yaml(_config_path_from_args(sys.argv[1:]))

    import mlx_lm.lora as lora

    if tokenizer_config_extra:
        original_load = lora.load

        def _load_with_config(*args: Any, **kwargs: Any) -> Any:
            kwargs["tokenizer_config"] = _merge_tokenizer_config(
                kwargs.get("tokenizer_config"),
                tokenizer_config_extra,
            )
            return original_load(*args, **kwargs)

        lora.load = _load_with_config

    lora.main()


if __name__ == "__main__":
    main()
