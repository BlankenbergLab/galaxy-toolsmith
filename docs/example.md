# Example: Running Galaxy Toolsmith on a 4xA100 (40GB) host

This runbook is for a machine with:

- 4x NVIDIA A100 GPUs (40GB each)
- 224 CPU cores
- 6TiB system memory
- local checkout of this repository

## 1. Create or populate the Python environment

This assumes Miniforge is already installed at `~/miniforge3` and this repository is checked out locally.

### From scratch

From the repository root, create a project-local environment at `.conda/gtsm`:

```bash
make install
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$PWD/.conda/gtsm"
```

The expanded equivalent is:

```bash
~/miniforge3/bin/conda create -y -p .conda/gtsm -c conda-forge -c bioconda python=3.11 apptainer squashfuse libfuse3
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$PWD/.conda/gtsm"
python -m pip install --upgrade pip
python -m pip install -e ".[training,server,eval]"
```

Python 3.11 is correct for this project (`requires-python >=3.11`). `apptainer` provides the Singularity-compatible runtime used by the corpus extraction command on linux-64. `squashfuse` lets Apptainer mount cached SIF images directly when user FUSE is available, and `libfuse3` provides the env-local `sbin/fusermount3` helper. Without host FUSE support (`/dev/fuse`), Galaxy Toolsmith's default SIF execution mode uses a persistent sandbox cache to avoid repeated temporary extraction where possible.

If the `mamba` wrapper in the Miniforge base environment fails with an import error, do not repair the base environment for this workflow. Use the default `conda`-based `make install` path above. On machines with a working alternative solver, override environment creation explicitly:

```bash
make install ENV_CREATE="~/miniforge3/bin/mamba create"
```

### Already have an active environment

If a compatible conda environment is already active, populate it instead of creating `.conda/gtsm`:

```bash
conda activate galaxy-toolsmith
make install-active
make doctor-active
```

## 2. Initialize workspace and pull sources

```bash
gtsm init-workspace
gtsm sync-tools-iuc --ref main
gtsm sync-galaxy-skills --ref main
gtsm sync-galaxy-xsd --ref dev
gtsm extract-corpus \
  --max-workers 8 \
  --source-workers 8 \
  --no-fetch-docs \
  --resolve-containers \
  --execute-containers \
  --container-runtime auto \
  --container-prepare-workers 2 \
  --container-probe-workers 4 \
  --container-help-probe-mode exploratory \
  --container-cache-dir .gtsm-cache/containers \
  --container-sif-exec-mode auto \
  --source-download-timeout-seconds 60 \
  --status-log .gtsm-cache/logs/extract-corpus.status.jsonl \
  --retry-manifest .gtsm-cache/datasets/tools-iuc-corpus.retry-manifest.json \
  --bioconda-checkout-sources \
  --bioconda-ref master
```

Equivalent Makefile flow:

```bash
make sync
make extract-corpus MAX_WORKERS=8 CONTAINER_RUNTIME=auto SOURCE_DOWNLOAD_TIMEOUT_SECONDS=60 STATUS_LOG=.gtsm-cache/logs/extract-corpus.status.jsonl
```

Extraction execution mode expands Galaxy XML macros before deriving requirements, commands, and container refs. It resolves explicit `<container>` refs and Galaxy-compatible mulled/BioContainers names from package requirements, validates generated mulled tags against Quay, and records all plausible candidates. Wrapper parsing, upstream source checkout, container image preparation, and container help/API probing have separate concurrency controls. During execution, candidates are prepared lazily per wrapper: the first candidate that provides useful help stops further image preparation for that wrapper. Prepared images are shared across wrappers, downloaded depot images are stored in the Singularity/Apptainer cache, and Docker is only used as a fallback for Docker-compatible image refs.

`extract-corpus` emits JSON status events before and during container execution, including restart cleanup, wrapper counts, Bioconda recipe/source preparation, container runtime discovery, per-wrapper completion, and final summary. Add `--status-log .gtsm-cache/logs/extract-corpus.status.jsonl` to keep the same stream in a file. `--no-fetch-docs` is recommended for large exploratory container/source runs because GitHub README fetching adds many quiet network requests that are not needed for command-line help extraction.

Completed runs also write recovery diagnostics and a retry manifest. If the
manifest path already exists, `--retry-manifest` limits extraction to the listed
wrappers and then writes an updated manifest. This is useful for spot-checking
known bad tools before committing to a full corpus rerun.

When `--bioconda-checkout-sources` is enabled, the Bioconda recipes checkout is prepared once before wrapper workers start, then upstream source checkouts are cached per package/version and reused across restarts. If the current recipe does not match the wrapper requirement version, extraction searches the local Bioconda git history for the newest exact matching recipe and otherwise falls back to the closest semver-style recipe. If Bioconda has no usable recipe/source for a requirement, Galaxy Toolsmith can check the matching conda-forge feedstock and record that fallback as `source_channel: conda-forge`. Source checkout resolves the real upstream software source from the recipe, including git sources and HTTP(S)/FTP source archives; the extracted source tree is what later source-context training uses. Recipe/source data is also used to identify likely installed CLI entry points. Low-confidence wrapper shell fragments, helper commands, local script paths, and generic output names are skipped instead of being probed.

Source retrieval also handles common historical-package edge cases. If a recipe
points at a binary release artifact, Galaxy Toolsmith can try upstream tag
source archives. Older Bioconductor and CRAN packages are recovered from archive
URLs when the direct `src/contrib` URL has moved. Legacy source hosts that
redirect to stale HTTPS certificates are retried for source collection, with
the original URL and fallback reason kept in diagnostics. Source docs are
mined from README/manual/help/example material and documentation directories,
not generic build files, so command-line context is useful for training without
encouraging generated wrappers to create helper scripts.

The default `--container-help-probe-mode exploratory` preflights each candidate binary with `command -v`, then tries `--help`, `-h`, `help`, and finally a no-argument probe in an isolated temporary working directory. Probes run under `bash` first because many Bioconda images rely on bash-compatible activation snippets; if bash is unavailable, the probe falls back to `sh`.

For Apptainer/Singularity runs, keep `squashfuse` and `libfuse3` available in the same conda environment as `gtsm`. Galaxy Toolsmith prepends that environment's `bin` and `sbin` directories to subprocess `PATH`, allowing Apptainer to find helper binaries such as `squashfuse`, `mount.fuse3`, and `fusermount3` even when `gtsm` is invoked by absolute path. Direct SIF mounting still depends on host policy: `/dev/fuse` must be present for user FUSE mounts. When it is not present, the default `--container-sif-exec-mode auto` materializes cached SIF images into reusable sandboxes under `.gtsm-cache/containers/sandboxes/`. This is compatible with extraction and avoids repeated per-probe temporary conversion. Use `--container-sif-exec-mode sif` to preserve raw runtime behavior, or `--container-sif-exec-mode sandbox` to force persistent sandbox use during spot checks.

Only output classified as real help is appended to the corpus. Some tools return nonzero for useful help, for example `unrecognized option --help` followed by `Usage:` and option text; these are kept as `container-command-help-degraded` after stripping the leading error boilerplate. Missing commands, wrapper-local script paths, shell setup fragments, unresolved container placeholders, and non-help banners remain in `container_execution` or candidate skip metadata and are not appended. If one attempted candidate image does not contain the selected command, extraction can try the next plausible candidate while preserving cache state.

To restart a failed corpus extraction without manually removing the output JSONL/checkpoint files:

```bash
gtsm extract-corpus --restart --max-workers 8 --no-fetch-docs --resolve-containers --execute-containers --container-runtime auto --container-help-probe-mode exploratory --status-log .gtsm-cache/logs/extract-corpus.status.jsonl --bioconda-checkout-sources
make extract-corpus RESTART=1 MAX_WORKERS=8 CONTAINER_RUNTIME=auto CONTAINER_HELP_PROBE_MODE=exploratory STATUS_LOG=.gtsm-cache/logs/extract-corpus.status.jsonl
```

`--restart` clears the selected corpus output, checkpoint, index, execution report, and expanded XML cache. It keeps source, BioConda, and container caches so successful downloads/checkouts can be reused while wrappers are reparsed and probes retried; use `make clean-containers` when a cached image itself is bad. Without `--restart`, completed wrapper records in the checkpoint are skipped and the final index/report are rebuilt from the complete JSONL corpus.

Optional Docker fallback with sudo:

```bash
gtsm extract-corpus --max-workers 8 --no-fetch-docs --resolve-containers --execute-containers --container-runtime auto --docker-use-sudo --bioconda-checkout-sources
make extract-corpus MAX_WORKERS=8 CONTAINER_RUNTIME=auto DOCKER_USE_SUDO=1
```

## 3. Configure model/source cache and token guards

Set explicit model source/cache controls:

```bash
export GTSM_MODEL_SOURCE_REGISTRY="https://huggingface.co"
export GTSM_MODEL_CACHE_ROOT="$PWD/.gtsm-cache/models/hf-cache"
export GTSM_MODEL_REVISION=""
export GTSM_MODEL_LOCAL_FILES_ONLY=false
```

`GTSM_MODEL_CACHE_ROOT` defaults to `$PWD/.gtsm-cache/models/hf-cache` for `gtsm` commands when neither `GTSM_MODEL_CACHE_ROOT` nor `HF_HOME` is set. Use `gtsm model-cache-info` to inspect the resolved cache. Set `HF_TOKEN` when available for higher Hugging Face Hub rate limits; the cache still works without it.

Set token guard(s) for server endpoints:

```bash
export GTSM_SERVER_AUTH_TOKEN="$(openssl rand -hex 32)"
printf "%s\n" "$GTSM_SERVER_AUTH_TOKEN" > .gtsm-cache/server.tokens
chmod 600 .gtsm-cache/server.tokens
```

## 4. Start the coordinator server (token-protected)

```bash
gtsm serve \
  --host 0.0.0.0 \
  --port 8765 \
  --provider local \
  --auth-tokens-file .gtsm-cache/server.tokens \
  --require-generate-auth \
  --status-log .gtsm-cache/logs/server.status.jsonl \
  --detach \
  --detach-log .gtsm-cache/logs/server.detach.log
```

`--status-log` is optional and disabled by default. Without it, status updates go to console only.
`--detach` is optional and disabled by default. When enabled, the process continues after disconnect.

## 5. Submit training job and start worker pool

### Submit job

Use non-quantized profile first (Devstral default):

```bash
gtsm train-remote-submit \
  --server-url http://127.0.0.1:8765 \
  --profile agentic-devstral-24b \
  --dataset-manifest config/dataset.manifest.json \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl
```

### Start workers

Open separate shells (or managed services) and run:

```bash
export GTSM_SERVER_AUTH_TOKEN="$(cat .gtsm-cache/server.tokens)"
CUDA_VISIBLE_DEVICES=0 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu0.status.jsonl --detach --detach-log .gtsm-cache/logs/worker-gpu0.detach.log
CUDA_VISIBLE_DEVICES=1 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu1.status.jsonl --detach --detach-log .gtsm-cache/logs/worker-gpu1.detach.log
CUDA_VISIBLE_DEVICES=2 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu2.status.jsonl --detach --detach-log .gtsm-cache/logs/worker-gpu2.detach.log
CUDA_VISIBLE_DEVICES=3 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu3.status.jsonl --detach --detach-log .gtsm-cache/logs/worker-gpu3.detach.log
```

Check job status:

```bash
gtsm train-remote-status --server-url http://127.0.0.1:8765 --job-id <job-id> --status-log .gtsm-cache/logs/train-status.jsonl
```

Watch progress in browser (auto-refresh dashboard):

```text
http://127.0.0.1:8765/monitor
```

If auth is enabled, provide bearer token in the dashboard token field, or open with query token:

```text
http://127.0.0.1:8765/monitor?token=<token>
```

Browser monitoring and JSONL status logs can be used together.

Fetch artifacts in parallel:

```bash
gtsm train-artifacts-fetch \
  --server-url http://127.0.0.1:8765 \
  --job-id <job-id> \
  --output-dir .gtsm-cache/models/remote-artifacts \
  --max-workers 8
```

## 6. Non-quantized-first tuning policy

Recommended flow:

1. Train with non-quantized profile (`quantization: none`)
2. Export quantized variants for deployment/runtime targets

Example quantized export:

```bash
gtsm export-model \
  --variant-id <variant-id> \
  --format all \
  --quantizations q8_0,q6_k,q5_k_m,q4_k_m
```

Optional Ollama packaging:

```bash
gtsm export-ollama-model --variant-id <variant-id> --model-name gtsm-wrapper-model
# optional:
gtsm export-ollama-model --variant-id <variant-id> --model-name gtsm-wrapper-model --create
```

## 7. Which model is strongest here?

### Current practical guidance

- **Default non-quantized starting point (recommended):**
  - `agentic-devstral-24b`
- **Alternative fast-iteration options:**
  - `proto-qwen25-7b`
  - `deepseek-coder-v2-lite-instruct`
- **Stronger but heavier (advanced/experimental):**
  - `deepseek-r1-distill-qwen-14b`
  - `deepseek-r1-distill-qwen-32b`
  - `baseline-mistral-24b`

Because tuning stability and throughput depend on sequence length, batch sizing, and runtime path, treat 32B as an advanced tier and benchmark against 14B/24B before adopting as your default.

## 8. What changes if only 2 GPUs are used?

Use the same flow, but:

- run 2 workers instead of 4,
- prefer 14B/24B tiers first,
- lower concurrency and expect slower throughput,
- keep non-quantized-first policy, then export quantized outputs.

## 9. Can the server run as a daemon?

Yes.

### Option A (recommended): systemd

Example unit (`/etc/systemd/system/gtsm-server.service`):

```ini
[Unit]
Description=Galaxy Toolsmith Coordinator Server
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=<repo-root>
Environment=PYTHONPATH=<repo-root>/src
Environment=GTSM_SERVER_AUTH_TOKEN_FILE=<repo-root>/.gtsm-cache/server.tokens
ExecStart=/bin/bash -lc 'TOKENS_FILE=.gtsm-cache/server.tokens gtsm serve --host 0.0.0.0 --port 8765 --provider local --auth-tokens-file .gtsm-cache/server.tokens --require-generate-auth'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gtsm-server
sudo systemctl status gtsm-server
```

### Option B: nohup/tmux

```bash
nohup gtsm serve --host 0.0.0.0 --port 8765 --provider local --auth-tokens-file .gtsm-cache/server.tokens --require-generate-auth > .gtsm-cache/server.log 2>&1 &
```

## 10. Is there an interactive setup process?

Currently: no dedicated wizard command.

Practical options now:

- use this runbook as a copy/paste setup flow,
- create a local setup script that writes tokens/env and starts server/workers.

Future enhancement candidate:

- `gtsm setup-wizard` to interactively generate env/config/token and startup commands.

## 11. Promotion gate before changing primary default model

Use benchmark + promotion checks before switching default profile:

```bash
gtsm benchmark-generate --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl --limit 50 --model-variant baseline
gtsm benchmark-generate --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl --limit 50 --model-variant candidate --status-log .gtsm-cache/logs/benchmark.status.jsonl
gtsm promote-candidate --candidate-summary .gtsm-cache/runs/benchmark/benchmark.summary.json --baseline-summary <baseline-summary.json> --policy staging
```
