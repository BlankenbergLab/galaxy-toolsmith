from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from galaxy_toolsmith.core.manifests import ModelVariantManifest
from galaxy_toolsmith.core.paths import WorkspacePaths
from galaxy_toolsmith.providers import local as local_mod
from galaxy_toolsmith.providers.base import GenerationInput, GenerationOutput


class _FakeTensor:
    def __init__(self, values: list[int]):
        self.values = values
        self.shape = (1, len(values))
        self.device = None

    def to(self, device: str) -> _FakeTensor:
        self.device = device
        return self


class _FakeBatch(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = None

    def to(self, device: str) -> _FakeBatch:
        self.device = device
        for value in self.values():
            if hasattr(value, "to"):
                value.to(device)
        return self


class _FakeModel:
    device = "cuda:0"

    def __init__(self):
        self.generate_kwargs = {}

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[10, 11, 12, 13, 14]]


class _FakeTokenizer:
    eos_token_id = 99

    def __init__(self, template_output):
        self.template_output = template_output
        self.decoded_tokens = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors):
        assert messages
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        return self.template_output

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        self.decoded_tokens = list(tokens)
        return "decoded-output"


def _request(model_variant: str = "variant-a") -> GenerationInput:
    return GenerationInput(
        tool_name="echo_tool",
        help_text="Usage: echo_tool --input FILE",
        source_code="",
        model_variant=model_variant,
    )


def test_local_provider_requires_real_backend_by_default(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    provider = local_mod.LocalProvider(paths=paths)

    with pytest.raises(RuntimeError, match="No real local generator"):
        provider.generate_wrapper(_request("missing-variant"))


def test_local_provider_stub_requires_opt_in(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    provider = local_mod.LocalProvider(paths=paths, allow_stub=True)

    output = provider.generate_wrapper(_request("missing-variant"))

    assert output.provider == "local-stub"
    assert "<tool" in output.xml_wrapper


def test_local_provider_stub_can_generate_udt_yaml(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    provider = local_mod.LocalProvider(paths=paths, allow_stub=True)
    request = GenerationInput(
        tool_name="echo_tool",
        help_text="Usage: echo_tool --input FILE",
        source_code="",
        model_variant="missing-variant",
        artifact_format="udt_yaml",
    )

    output = provider.generate_wrapper(request)

    assert output.provider == "local-stub"
    assert output.artifact_format == "udt_yaml"
    assert output.xml_wrapper == ""
    assert output.artifact_text.startswith("class: GalaxyUserTool")
    assert "shell_command:" in output.artifact_text


def test_local_provider_uses_peft_variant_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True)
    (paths.models_root / "variants").mkdir(parents=True)
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (paths.models_root / "variants" / "variant-a.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-a",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            provider="local",
            backend="axolotl",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )

    def fake_generate(
        self: local_mod.LocalPeftProvider,
        request: GenerationInput,
    ) -> GenerationOutput:
        assert self.source_policy.cache_dir == str(paths.models_root / "hf-cache")
        return GenerationOutput(
            xml_wrapper="<tool id='echo_tool' name='echo_tool' version='0.1.0'/>",
            provider=self.name,
            model_variant=request.model_variant,
        )

    monkeypatch.setattr(local_mod.LocalPeftProvider, "generate_wrapper", fake_generate)
    provider = local_mod.LocalProvider(paths=paths)

    output = provider.generate_wrapper(_request("variant-a"))

    assert output.provider == "local-peft"
    assert output.model_variant == "variant-a"


def test_local_provider_preloads_peft_variant_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WorkspacePaths.from_repo_root(tmp_path)
    paths.create_directories()
    artifact_dir = paths.models_root / "artifacts" / "variant-a"
    artifact_dir.mkdir(parents=True)
    (paths.models_root / "variants").mkdir(parents=True)
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (paths.models_root / "variants" / "variant-a.manifest.json").write_text(
        ModelVariantManifest(
            variant_id="variant-a",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            provider="local",
            backend="axolotl",
            artifact_dir=str(artifact_dir),
        ).to_json(),
        encoding="utf-8",
    )
    loaded = []

    def fake_ensure_loaded(self: local_mod.LocalPeftProvider) -> dict:
        loaded.append(self.adapter_path)
        return {
            "backend": self.name,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
        }

    monkeypatch.setattr(local_mod.LocalPeftProvider, "ensure_loaded", fake_ensure_loaded)
    provider = local_mod.LocalProvider(paths=paths)

    info = provider.ensure_loaded("variant-a")

    assert loaded == [str(artifact_dir)]
    assert info["model_loaded"] is True
    assert info["backend"] == "local-peft"
    assert info["adapter_path"] == str(artifact_dir)


def _install_fake_peft_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_map: dict[str, str],
) -> dict:
    captured: dict = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            return SimpleNamespace(total_memory=40 * 1024**3)

    fake_torch = SimpleNamespace(
        cuda=FakeCuda,
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
    )

    class FakeTokenizer:
        pad_token_id = None
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["tokenizer_kwargs"] = kwargs
            return cls()

    class FakeBaseModel:
        hf_device_map = {"": "cuda:0"}

    class FakeModel:
        def __init__(self, base_model):
            self.base_model = base_model
            self.hf_device_map = device_map

        def eval(self):
            captured["eval_called"] = True

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["model_kwargs"] = kwargs
            return FakeBaseModel()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, base_model, adapter_path):
            captured["adapter_path"] = adapter_path
            return FakeModel(base_model)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeAutoModel, AutoTokenizer=FakeTokenizer),
    )
    return captured


def test_local_peft_provider_allows_cpu_offload_when_policy_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fake_peft_runtime(
        monkeypatch,
        device_map={"": "cuda:0", "layer.1": "cpu"},
    )
    provider = local_mod.LocalPeftProvider(
        base_model="base-model",
        adapter_path="/tmp/adapter",
        offload_policy="allow",
        gpu_memory_reserve_gib=3.0,
    )

    info = provider.ensure_loaded()

    assert info["has_offload"] is True
    assert info["offloaded_devices"] == {"layer.1": "cpu"}
    assert info["max_memory"] == {"0": "37GiB"}
    assert captured["model_kwargs"]["device_map"] == "auto"
    assert captured["model_kwargs"]["max_memory"] == {0: "37GiB"}


def test_local_peft_provider_fails_cpu_offload_when_policy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_peft_runtime(
        monkeypatch,
        device_map={"": "cuda:0", "layer.1": "cpu"},
    )
    provider = local_mod.LocalPeftProvider(
        base_model="base-model",
        adapter_path="/tmp/adapter",
        offload_policy="fail",
    )

    with pytest.raises(RuntimeError, match="offload policy is fail"):
        provider.ensure_loaded()


def test_local_peft_provider_strict_policy_accepts_all_cuda_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_peft_runtime(
        monkeypatch,
        device_map={"": "cuda:0", "layer.1": "cuda:0"},
    )
    provider = local_mod.LocalPeftProvider(
        base_model="base-model",
        adapter_path="/tmp/adapter",
        offload_policy="fail",
    )

    info = provider.ensure_loaded()

    assert info["has_offload"] is False
    assert info["offloaded_devices"] == {}


def test_local_peft_provider_empty_output_error_includes_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = local_mod.LocalPeftProvider(base_model="base-model", adapter_path="/tmp/adapter")
    monkeypatch.setattr(provider, "_lazy_load", lambda: (object(), object()))
    monkeypatch.setattr(local_mod, "_decode_model_response", lambda **kwargs: "\n```  \n```\n")

    with pytest.raises(RuntimeError) as exc:
        provider.generate_wrapper(_request("variant-a"))

    message = str(exc.value)
    assert "LocalPeftProvider returned empty output" in message
    assert "base-model" in message
    assert "/tmp/adapter" in message
    assert "raw_response_length=" in message
    assert "raw_response_preview=" in message


def test_local_peft_provider_extracts_first_complete_tool_xml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = local_mod.LocalPeftProvider(base_model="base-model", adapter_path="/tmp/adapter")
    monkeypatch.setattr(provider, "_lazy_load", lambda: (object(), object()))
    monkeypatch.setattr(
        local_mod,
        "_decode_model_response",
        lambda **kwargs: (
            "Here is the wrapper:\n"
            "<tool id='echo_tool' name='Echo Tool' version='0.1.0'>\n"
            "  <command>echo test</command>\n"
            "</tool>\n"
            "Trailing explanation that should not be written."
        ),
    )

    output = provider.generate_wrapper(_request("variant-a"))

    assert output.xml_wrapper == (
        "<tool id='echo_tool' name='Echo Tool' version='0.1.0'>\n"
        "  <command>echo test</command>\n"
        "</tool>"
    )


def test_decode_model_response_handles_tensor_chat_template_output() -> None:
    input_ids = _FakeTensor([1, 2])
    model = _FakeModel()
    tokenizer = _FakeTokenizer(input_ids)

    response = local_mod._decode_model_response(
        model=model,
        tokenizer=tokenizer,
        request=_request(),
        max_new_tokens=7,
        temperature=0.0,
    )

    assert response == "decoded-output"
    assert input_ids.device == "cuda:0"
    assert model.generate_kwargs["input_ids"] is input_ids
    assert model.generate_kwargs["max_new_tokens"] == 7
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs
    assert model.generate_kwargs["pad_token_id"] == 99
    assert tokenizer.decoded_tokens == [12, 13, 14]


def test_decode_model_response_adds_tool_close_stopping_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = _FakeTensor([1, 2])
    model = _FakeModel()
    tokenizer = _FakeTokenizer(input_ids)
    sentinel = object()
    calls = []

    def fake_stopping_criteria(*, tokenizer, input_length):
        calls.append((tokenizer, input_length))
        return sentinel

    monkeypatch.setattr(local_mod, "_tool_xml_stopping_criteria", fake_stopping_criteria)

    local_mod._decode_model_response(
        model=model,
        tokenizer=tokenizer,
        request=_request(),
        max_new_tokens=7,
        temperature=0.0,
    )

    assert calls == [(tokenizer, 2)]
    assert model.generate_kwargs["stopping_criteria"] is sentinel


def test_generated_text_contains_tool_close_detects_decoded_suffix() -> None:
    class DecodeTokenizer:
        decoded_tokens = []

        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens is True
            self.decoded_tokens = list(tokens)
            return "<tool id='echo'></tool>"

    input_ids = [[1, 2, 3, 4, 5]]
    tokenizer = DecodeTokenizer()

    assert local_mod._generated_text_contains_tool_close(
        tokenizer=tokenizer,
        input_ids=input_ids,
        input_length=2,
    )
    assert tokenizer.decoded_tokens == [3, 4, 5]


def test_decode_model_response_handles_batch_chat_template_output() -> None:
    input_ids = _FakeTensor([1, 2, 3])
    attention_mask = _FakeTensor([1, 1, 1])
    batch = _FakeBatch(input_ids=input_ids, attention_mask=attention_mask)
    model = _FakeModel()
    tokenizer = _FakeTokenizer(batch)

    response = local_mod._decode_model_response(
        model=model,
        tokenizer=tokenizer,
        request=_request(),
        max_new_tokens=5,
        temperature=0.2,
    )

    assert response == "decoded-output"
    assert batch.device == "cuda:0"
    assert input_ids.device == "cuda:0"
    assert attention_mask.device == "cuda:0"
    assert model.generate_kwargs["input_ids"] is input_ids
    assert model.generate_kwargs["attention_mask"] is attention_mask
    assert model.generate_kwargs["temperature"] == 0.2
    assert model.generate_kwargs["do_sample"] is True
    assert tokenizer.decoded_tokens == [13, 14]
