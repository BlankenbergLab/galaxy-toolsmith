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

## 4. Run the observed local context ladder

The strongest documented run on this host is
`devstral-sidecars-fixtures-20260707`. It used the local, single-node path: four
GPU processes over the same training job, with model state sharded across the
GPUs by FSDP/ZeRO-style distributed training. This is different from running
four independent full models, one per GPU; sharding lets a larger model and
longer context fit than any single A100 40GB card could handle alone.

```bash
RUN_TAG=devstral-sidecars-fixtures-20260707 \
PROFILE=agentic-devstral-24b \
ARTIFACT_FORMAT=mixed \
SOURCE_MODES=all-raw,all-filtered \
SOURCE_MODE_PREFERENCE=all-raw,all-filtered \
TEST_CONTEXT_MODE=fixtures \
TEST_CONTEXT_MAX_CHARS=4000 \
TEST_CONTEXT_MAX_FILES=6 \
TEST_CONTEXT_MAX_FILE_BYTES=64KB \
CONTEXT_LADDER=32k,24k,16k,12k,8k,4k,2k \
DISTRIBUTED_STRATEGIES=fsdp,deepspeed-zero3 \
GPU_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
POST_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
POST_EXPORT_QUANTIZATIONS=q4_k_m \
POST_OLLAMA_CREATE=1 \
scripts/gtsm_context_ladder_train.sh launch
```

Monitor the detached run:

```bash
RUN_TAG=devstral-sidecars-fixtures-20260707 scripts/gtsm_context_ladder_train.sh status
RUN_TAG=devstral-sidecars-fixtures-20260707 scripts/gtsm_context_ladder_train.sh tail
```

The run writes audit files under:

```text
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/
```

Important files are `summary.md`, `probes.tsv`, `full-training.tsv`,
`selection.env`, and `logs/`.

## 5. What the 20260707 run selected

The ladder tried 32k, 24k, 16k, 12k, 8k, 4k, and 2k. It tested both raw and
filtered source-context modes, and both FSDP and DeepSpeed ZeRO-3. The higher
contexts failed during short GPU probes, so the selected setting was:

| Setting | Value |
| --- | --- |
| Context length | `12288` |
| Source context | `all-raw` |
| Source budget | `24000` chars, `96` files |
| Test/example sidecars | `fixtures`, capped at `4000` chars and `6` files |
| Distributed strategy | `fsdp` |
| GPU processes | `4` |
| Variant id | `tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707` |

The 12k setting is not a model limit; it is the highest setting this run proved
stable for this corpus, sidecar budget, batch shape, and 4xA100 40GB machine.
Increasing context raises memory roughly with attention and activation cost,
while increasing source/test sidecar budgets raises prompt length pressure. The
runner keeps falling back so an unattended run can still produce a model when
larger contexts OOM.

## 6. What context is used for fine-tuning

The training samples combine the primary target artifact with sidecar context:

| Context | How it is used |
| --- | --- |
| Galaxy wrapper XML | Primary supervised target for `xml` examples. |
| Expanded XML | Macro-expanded wrapper view produced through Galaxy's tool loader. |
| UDT YAML | Additional target form for `mixed` examples when convertible/synthesized UDT exists. |
| Command metadata/help | Requirement-derived command names plus help collected from containers. |
| Upstream source | Real source downloaded from Bioconda or conda-forge recipe sources, including archives and git sources. |
| Helper/configfile code | Existing wrapper-local scripts and configfile templates, used as context for understanding legacy wrappers. |
| Tests/examples/fixtures | Optional sidecar context controlled by `TEST_CONTEXT_*`, not part of the primary output. |
| Shed metadata | Repository/suite metadata, used for Tool Shed packaging context. |

During generation, macros, tool data tables, datatype scaffold files, and
`.shed.yml` are sidecars. The primary model output remains a Galaxy `<tool>`
unless suite generation is explicitly requested.

## 7. Export and Ollama packaging

The run used an external export environment because GGUF/Unsloth/llama.cpp
dependencies can conflict with the main training environment:

```bash
POST_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export
```

The full model export produced:

```text
.gtsm-cache/models/exports/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707/gguf/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707.bf16.gguf
.gtsm-cache/models/exports/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707/gguf/q4_k_m/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707-q4_k_m.gguf
```

The first post-export status recorded `post-export-failed` because `ollama
create` was not on `PATH` inside the export environment. The GGUF files were
valid; rerunning registration with the Ollama binary available created:

```text
gtsm-tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-3f8e387314
```

Ollama packaging writes a Modelfile that points at a GGUF artifact and registers
that file under a local Ollama model name. It does not retrain the model.

## 8. Minibwa generation comparison

After training and q4 export, the run compared four minibwa suite-generation
paths under:

```text
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/minibwa-tests/
```

| Scenario | Exit | Records | XML tools | Repairs | Unknown datatypes | Params | Outputs | Tests | Citations | Help chars | Command chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full local PEFT, Bioconda discovery | 0 | 11 | 11 | 0 | 8 | 87 | 13 | 11 | 11 | 14485 | 1517 |
| Full local PEFT, static help/source | 0 | 11 | 11 | 0 | 8 | 61 | 11 | 11 | 11 | 7656 | 1223 |
| Ollama q4, Bioconda discovery | 0 | 11 | 11 | 0 | 7 | 129 | 13 | 11 | 9 | 608 | 4684 |
| Ollama q4, static help/source | 0 | 11 | 11 | 0 | 7 | 129 | 13 | 11 | 9 | 608 | 4721 |

The full local PEFT model with Bioconda discovery was the best qualitative
result: it had richer discovered help/source context, stronger command shapes,
and complete Toolsmith citations. The q4 Ollama model generated structurally
valid XML, but it was more verbose and speculative, had shorter final help
sections, and produced fewer citations. These checks were structural generation
comparisons, not Planemo execution.

## 9. Client/server architecture mode

The local context ladder is the recommended path for single-host long-context
training. Galaxy Toolsmith also supports a client/server mode when jobs,
workers, and monitoring need to be separated.

Start a token-protected coordinator:

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

Submit a training job:

```bash
gtsm train-remote-submit \
  --server-url http://127.0.0.1:8765 \
  --profile agentic-devstral-24b \
  --dataset-manifest config/dataset.manifest.json \
  --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl
```

Run workers from separate shells or service units:

```bash
export GTSM_SERVER_AUTH_TOKEN="$(cat .gtsm-cache/server.tokens)"
CUDA_VISIBLE_DEVICES=0 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu0.status.jsonl
CUDA_VISIBLE_DEVICES=1 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu1.status.jsonl
CUDA_VISIBLE_DEVICES=2 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu2.status.jsonl
CUDA_VISIBLE_DEVICES=3 gtsm train-worker --server-url http://127.0.0.1:8765 --status-log .gtsm-cache/logs/worker-gpu3.status.jsonl
```

The server exposes a browser monitor at:

```text
http://127.0.0.1:8765/monitor
```

## 10. Model guidance on this host class

Use `agentic-devstral-24b` when the goal is a high-capability local wrapper
model and the host has 4xA100 40GB available. Use smaller 7B/14B profiles for
fast iteration or when fewer GPUs are available. Keep the non-quantized-first
policy: train the full/PEFT model first, then export GGUF quantizations for
deployment runtimes such as Ollama.

With only 2 A100 40GB GPUs, use the same tooling but expect a lower selected
context, slower probes, and more value from 7B/14B iteration runs. The ladder is
designed to record the failures and fall back rather than silently changing the
training shape.

## 11. Promotion gate before changing defaults

Use benchmark and promotion checks before changing a default model:

```bash
gtsm benchmark-generate --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl --limit 50 --model-variant baseline
gtsm benchmark-generate --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl --limit 50 --model-variant candidate --status-log .gtsm-cache/logs/benchmark.status.jsonl
gtsm promote-candidate --candidate-summary .gtsm-cache/runs/benchmark/benchmark.summary.json --baseline-summary <baseline-summary.json> --policy staging
```
