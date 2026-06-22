# Galaxy Toolsmith

Galaxy Toolsmith is a library-first and CLI-driven toolkit for building datasets, training/evaluating model variants, and generating Galaxy tool wrappers.

## Package and CLI
- Python package: `galaxy_toolsmith`
- CLI: `gtsm`

## Quick start
```bash
gtsm doctor
gtsm init-workspace
gtsm sync-tools-iuc --ref main
gtsm sync-galaxy-skills --ref main
gtsm sync-galaxy-xsd --ref dev
gtsm extract-corpus --max-workers 8
# optional: disable README fetching from tool homepage URLs
# gtsm extract-corpus --max-workers 8 --no-fetch-docs
# optional: enrich records by running resolved containers for CLI help
# gtsm extract-corpus --max-workers 8 --resolve-containers --execute-containers
gtsm list-train-profiles
gtsm train --profile agentic-devstral-24b --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl
gtsm list-model-variants
gtsm export-model --variant-id bootstrap-dataset-agentic-devstral-24b --format all --quantizations q8_0,q6_k,q5_k_m,q4_k_m
gtsm estimate-model-resources
gtsm generate-wrapper --tool-name my_tool --help-text-file help.txt --output my_tool.xml
gtsm generate-wrapper --tool-name my_tool --help-text-file help.txt --artifact-format udt-yaml --output my_tool.yml
gtsm convert-udt --input my_tool.yml --output my_tool.xml --report my_tool.udt-conversion.json
gtsm evaluate-wrappers --wrappers my_tool.xml
gtsm evaluate-wrappers --artifact-format udt-yaml --wrappers my_tool.yml
gtsm evaluate-wrappers --wrappers my_tool.xml --run-planemo --run-planemo-tests --planemo-install-galaxy
gtsm benchmark-generate --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl --limit 50
gtsm promote-candidate --candidate-summary .gtsm-cache/runs/benchmark/benchmark.summary.json
gtsm list-promotion-policies
```

## Test suites
```bash
make test
make lint

# Optional live Planemo/Galaxy smoke test. Use an existing Galaxy checkout or
# opt in to Planemo's Galaxy install behavior.
make test-optional-planemo PLANEMO_TEST_GALAXY_ROOT=/path/to/galaxy
make test-optional-planemo PLANEMO_TEST_INSTALL_GALAXY=1

# Optional live Ollama smoke test. Requires a running Ollama server and a real
# GGUF artifact path without whitespace.
make test-optional-ollama OLLAMA_TEST_GGUF=/absolute/path/model.gguf
```

The optional tests can also be run directly with pytest. They are skipped unless
enabled with `GTSM_TEST_LIVE_PLANEMO=1` or `GTSM_TEST_LIVE_OLLAMA=1`. The Ollama
test accepts `GTSM_TEST_OLLAMA_BIN` when `ollama` is not on `PATH`.

Suggested gate before switching any default to DeepSeek:
1. Run `benchmark-generate` for current primary default profile output.
2. Run `benchmark-generate` for candidate DeepSeek profile output on the same corpus slice.
3. Run `promote-candidate` with baseline comparison and enforce your policy thresholds.

`extract-corpus` emits wrapper-level records (one per wrapper XML), includes normalized `.shed.yml` metadata (including suite context), writes a companion index file (`*.index.json`), and can enrich training records with command-line help collected from Singularity/Apptainer or Docker containers.

On Linux nodes, use the Makefile for the container-enabled path:

```bash
make install
make sync
make extract-corpus CONTAINER_RUNTIME=auto MAX_WORKERS=32
make train PROFILE=agentic-devstral-24b
```

If you already have a compatible conda environment active, use `make install-active` instead of creating `.conda/gtsm`.

`export-model` now supports multi-quant GGUF exports in one run with `--quantizations`.

For no-sudo GGUF export on shared Linux nodes, build and use a user-space
`llama.cpp` quantizer instead of Unsloth's fallback installer:

```bash
make prepare-llama-cpp-export GGUF_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export
make export-gguf-llama-cpp \
  GGUF_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
  GGUF_VARIANT_ID=tools-iuc-qwen25-7b-full-20260617T221121Z-safe \
  GGUF_QUANTIZATIONS=q4_k_m
make finalize-gguf-export \
  GGUF_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
  GGUF_VARIANT_ID=tools-iuc-qwen25-7b-full-20260617T221121Z-safe \
  GGUF_RUN_TAG=20260617T221121Z
```

The helper configures `llama.cpp` with OpenMP disabled to avoid conda
`libgomp` linker issues, then runs `export-model` with
`GTSM_GGUF_BACKEND=llama.cpp`. The finalize target verifies the produced
GGUF, refreshes the overnight export JSON, and writes an Ollama Modelfile
without running `ollama create` unless `GGUF_OLLAMA_CREATE=1` is set.

Training defaults are accessibility-first and now make quantization state explicit in `list-train-profiles`. Recommended flow is to fine-tune non-quantized profiles first, then export quantized variants; optional pre-quantized profiles (`*-4bit`) remain available.

DeepSeek profiles are included as **opt-in evaluation defaults** (coding/distilled variants). Keep primary defaults on the existing Devstral/Mistral set unless DeepSeek variants pass your benchmark and promotion gates.

Model download/source behavior can be controlled with:
- `GTSM_MODEL_SOURCE_REGISTRY` (or `HF_ENDPOINT`) for registry endpoint
- `GTSM_MODEL_REVISION` for pinned model revision
- `GTSM_MODEL_CACHE_ROOT` (or `HF_HOME`) for cache location; defaults to `.gtsm-cache/models/hf-cache`
- `GTSM_MODEL_LOCAL_FILES_ONLY=true` for offline/local-only loading

Inspect resolved cache settings with:

```bash
gtsm model-cache-info
```

Set `HF_TOKEN` when available to avoid unauthenticated Hugging Face Hub rate limits; cached local weights are reused either way after the first successful download.

## Optional server/client mode
```bash
gtsm serve --host 127.0.0.1 --port 8765 --provider local
gtsm generate-wrapper-remote --server-url http://127.0.0.1:8765 --tool-name my_tool --help-text-file help.txt --output my_tool.xml
# inspect available model variants served by FastAPI:
curl http://127.0.0.1:8765/variants
# open browser monitoring dashboard:
# http://127.0.0.1:8765/monitor
```

Set `GTSM_SERVER_AUTH_TOKEN` to require/send bearer auth for remote generation.

### Distributed training server/worker mode
```bash
# start server (set one or more tokens to lock down training/artifact endpoints)
gtsm serve --host 127.0.0.1 --port 8765 --auth-token "$GTSM_SERVER_AUTH_TOKEN"

# submit training job to coordinator
gtsm train-remote-submit --server-url http://127.0.0.1:8765 --profile agentic-devstral-24b --dataset-manifest config/dataset.manifest.json --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl

# run one or more workers
gtsm train-worker --server-url http://127.0.0.1:8765

# fetch produced artifacts in parallel
gtsm train-artifacts-fetch --server-url http://127.0.0.1:8765 --job-id <job-id> --output-dir .gtsm-cache/models/remote-artifacts --max-workers 4
```

## External generation providers
- `--provider openai`: uses `GTSM_OPENAI_API_KEY` and optional `GTSM_OPENAI_BASE_URL`
- `--provider anthropic`: uses `GTSM_ANTHROPIC_API_KEY` and optional `GTSM_ANTHROPIC_BASE_URL`
- `--provider copilot`: uses `GTSM_COPILOT_API_KEY` and optional `GTSM_COPILOT_BASE_URL`
- `--provider ollama`: uses `GTSM_OLLAMA_BASE_URL` and optional `GTSM_OLLAMA_MODEL`

## Ollama export/deployment helpers
- Generate Modelfile from exported GGUF:
  - `gtsm export-ollama-model --variant-id <variant-id> --model-name <name>`
- Optionally register model in local Ollama runtime:
  - `gtsm export-ollama-model --variant-id <variant-id> --model-name <name> --create`
- Optional post-training hook:
  - `gtsm train ... --post-export-quantizations q8_0,q6_k,q5_k_m,q4_k_m --post-ollama-model-name <name> [--post-ollama-create]`

## Optional local unsloth provider
- Set `GTSM_LOCAL_UNSLOTH_MODEL` to enable local unsloth inference in `--provider local`
- Optionally set `GTSM_LOCAL_UNSLOTH_ADAPTER` to load a LoRA adapter path
- `GTSM_LOCAL_GENERATOR_CMD` still takes precedence when set (external command mode)

## Documentation
- Build docs locally:
  - `python -m pip install -r docs/requirements.txt`
  - `python -m mkdocs build --strict`
  - `python -m mkdocs serve -a 127.0.0.1:8000`
- Production docs deployment uses Cloudflare Workers static assets configured in `wrangler.jsonc`.
- Full 4xA100 pipeline runbook: `docs/example.md`
