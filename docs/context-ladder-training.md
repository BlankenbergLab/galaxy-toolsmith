# Context Ladder Training

Galaxy Toolsmith includes a reusable unattended runner for long-context training
on multi-GPU CUDA hosts:

```bash
scripts/gtsm_context_ladder_train.sh launch
scripts/gtsm_context_ladder_train.sh status
scripts/gtsm_context_ladder_train.sh tail
```

The runner estimates sample sizes, probes high-context settings with short
training runs, then launches full training at the highest GPU-only setting that
passes.

`launch` uses `tmux` automatically when available, which is preferred for
overnight runs because the training process survives shell/session cleanup. Set
`LAUNCH_BACKEND=nohup` only on hosts where `tmux` is unavailable or unwanted.

## Default Policy

The default ladder is:

```text
128k,96k,64k,48k,32k,24k,16k,12k,8k,4k,2k
```

The source-context budget is reduced with the context length. For example, 12k
uses a larger source budget than 8k, and 4k/2k are fallback settings for memory
or fit failures.

Source modes are tried in this order:

```text
all-raw,all-filtered
```

The default distributed strategies are GPU-only:

```text
deepspeed-zero3,fsdp
```

`deepspeed-zero3-offload` is intentionally not used by default. It offloads
model or optimizer state to CPU memory, which can fit larger runs but is often
much slower. Use it only as a manual salvage option when throughput is less
important than completing a run.

Candidate selection does not require every sample to fit the proposed sequence
length. The estimator records overflow counts and the longest sample examples,
then the ladder starts with the highest context and falls back after failed GPU
probes. This keeps pathological wrappers from blocking all training attempts
while still making the outliers visible in `estimate.approx.json` and
`estimate.exact.json`.

## Why Estimate Before Aggressive Probing?

The estimator is still useful even when the runner is allowed to probe high
settings aggressively:

- it records overflow counts and the worst outlier samples for each rung
- it chooses between `all-raw` and `all-filtered` source context before spending
  GPU time
- it reduces the number of CUDA OOM cycles and cleanup events

By default, the script runs an approximate estimate first, then an exact-tokenizer
estimate for the top candidate and the next lower context. If exact-tokenizer
loading fails, the script records the failure and continues with the approximate
candidates unless `EXACT_TOKENIZER_REQUIRED=1` is set.

## Common Runs

Run the full unattended pipeline:

```bash
scripts/gtsm_context_ladder_train.sh launch
```

Use explicit GPUs and skip the idle preflight when `nvidia-smi` query mode is
unreliable but the device list is already known:

```bash
GPU_DEVICES=0,1,2,3 NUM_PROCESSES=4 GPU_PREFLIGHT=0 scripts/gtsm_context_ladder_train.sh launch
```

Run estimation only:

```bash
ESTIMATE_ONLY=1 scripts/gtsm_context_ladder_train.sh run
```

Run probes only after estimating:

```bash
PROBE_ONLY=1 scripts/gtsm_context_ladder_train.sh run
```

Use a smaller ladder:

```bash
CONTEXT_LADDER=32k,24k,16k,12k,8k scripts/gtsm_context_ladder_train.sh launch
```

Use only filtered source:

```bash
SOURCE_MODES=all-filtered SOURCE_MODE_PREFERENCE=all-filtered scripts/gtsm_context_ladder_train.sh launch
```

Use CPU offload manually:

```bash
DISTRIBUTED_STRATEGIES=deepspeed-zero3-offload,fsdp scripts/gtsm_context_ladder_train.sh launch
```

Use a separate GGUF export environment after successful full training:

```bash
POST_EXPORT_ENV_DIR=.conda/gtsm-unsloth-export \
POST_EXPORT_LLAMA_CPP_DIR=.gtsm-cache/llama.cpp \
scripts/gtsm_context_ladder_train.sh launch
```

This mode avoids importing GGUF or Unsloth packaging dependencies into the main
training environment. The runner skips inline `gtsm train` post-export flags,
then calls `scripts/gtsm_llama_cpp_gguf.sh export` and `finalize` from the
external environment after full training succeeds.

## Observed 4xA100 Sidecar Run

The current full-length example is
`devstral-sidecars-fixtures-20260707`, run on 4 NVIDIA A100 40GB GPUs with
1,985 IUC corpus records. It trained Devstral 24B on mixed XML/UDT targets with
expanded XML, raw source context, command help, and upstream test/example
fixtures as separate sidecar context.

The launch shape was:

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

The ladder probes failed at 32k, 24k, and 16k for both `all-raw` and
`all-filtered` source modes under both FSDP and DeepSpeed ZeRO-3. The selected
candidate was:

| Setting | Value |
| --- | --- |
| Context length | `12288` |
| Source mode | `all-raw` |
| Source budget | `24000` chars, `96` files |
| Test sidecar mode | `fixtures` |
| Distributed strategy | `fsdp` |
| Processes | `4` |
| Variant id | `tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707` |

The full training run reached `training-finished` and recorded `completed` in
the training status stream. External export produced a bf16 GGUF and a
`q4_k_m` quantization:

```text
.gtsm-cache/models/exports/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707/gguf/
.gtsm-cache/models/exports/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707/gguf/q4_k_m/
```

The first post-export run recorded `post-export-failed` because `ollama create`
could not find `ollama` on `PATH` inside the separate export environment. The
GGUF export and quantization were present; registration was rerun after pointing
at the Ollama binary, producing the normalized model name
`gtsm-tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-3f8e387314`.

Use these run-local files when auditing or resuming:

```text
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/summary.md
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/probes.tsv
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/full-training.tsv
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/selection.env
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/minibwa-tests/comparison.md
```

## Outputs

Run outputs are written under:

```text
.gtsm-cache/runs/context-ladder/<run-tag>/
```

Important files include:

- `estimate.approx.json`
- `estimate.exact.json`
- `candidates.tsv`
- `probes.tsv`
- `full-training.tsv`
- `selection.env`
- `summary.md`
- `logs/`
- `status/`

The selected variant id encodes the artifact format, source mode, context
length, distributed strategy, and run tag.

## Key Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `PROFILE` | `agentic-devstral-24b` | Training profile. |
| `ARTIFACT_FORMAT` | `mixed` | Training target format. |
| `CORPUS_JSONL` | `.gtsm-cache/datasets/tools-iuc-corpus.jsonl` | Corpus JSONL. |
| `SOURCE_MODES` | `all-raw,all-filtered` | Source context modes to estimate. |
| `TEST_CONTEXT_MODE` | `fixtures` | Optional upstream test/example sidecar context mode forwarded to token estimation and training. Use `none`, `metadata`, `snippets`, or `fixtures`. |
| `TEST_CONTEXT_MAX_CHARS` | `4000` | Maximum test/example sidecar characters per prompt. |
| `TEST_CONTEXT_MAX_FILES` | `6` | Maximum test/example sidecar files per prompt. |
| `TEST_CONTEXT_MAX_FILE_BYTES` | `64KB` | Maximum bytes per test/example sidecar file. Accepts human sizes; use `0` for no per-file cap. |
| `DISTRIBUTED_STRATEGIES` | `deepspeed-zero3,fsdp` | Probe/training strategy order. |
| `CONTEXT_LADDER` | `128k,96k,64k,48k,32k,24k,16k,12k,8k,4k,2k` | Context candidates. |
| `PROBE_MAX_STEPS` | `5` | Short training probe steps. |
| `ESTIMATE_WORKERS` | `0` | Worker threads for token/source-context estimation. `0` chooses a bounded automatic value. |
| `ESTIMATE_LONGEST_SAMPLE_COUNT` | `50` | Longest sample examples retained per source-context estimate. |
| `CANDIDATE_MAX_OVERFLOW_FRACTION` | `1.0` | Maximum allowed overflow fraction for candidate rows. The default keeps all contexts available as OOM fallbacks. |
| `CANDIDATE_MAX_OVERFLOW_SAMPLES` | `1000000000` | Maximum allowed overflow sample count for candidate rows. Lower this to enforce stricter fit-only training. |
| `NUM_PROCESSES` | `auto` | Number of GPU processes. |
| `GPU_DEVICES` | `auto` | CUDA visible device list. |
| `GPU_PREFLIGHT` | `1` | Check GPU idle state before launch and after failed probes. Set `0` when explicit GPU ids are supplied and `nvidia-smi` query mode is unreliable. |
| `LAUNCH_BACKEND` | `auto` | Detached launch mode. `auto` prefers `tmux`, otherwise `nohup`. |
| `POST_EXPORT_QUANTIZATIONS` | `q4_k_m` | Post-training GGUF quantizations. |
| `POST_EXPORT_ENV_DIR` | empty | External environment for GGUF export and Ollama Modelfile generation. If unset, `gtsm train` uses inline post-export hooks. Falls back to `GGUF_EXPORT_ENV_DIR` when set. |
| `POST_EXPORT_LLAMA_CPP_DIR` | `.gtsm-cache/llama.cpp` | llama.cpp checkout used by the external export helper. |
| `POST_EXPORT_GGUF_OUTTYPE` | `bf16` | Base GGUF outtype before quantization. |
| `POST_EXPORT_PREPARE` | `0` | Run the helper's `prepare` step before external export. |
| `POST_EXPORT_SYNC_OVERNIGHT` | `0` | Whether the helper should sync an overnight export JSON during finalization. |
| `POST_OLLAMA_FROM_QUANTIZATION` | `q4_k_m` | Quantization referenced by the generated Ollama Modelfile. |
| `POST_OLLAMA_CREATE` | `0` | Run `ollama create` during external or inline post-export finalization. |

`status` reports the detached PID, elapsed wall time, latest backend status
event, progress, and ETA when the backend status payload includes enough
progress information.

## Cleanup

Failed OOM probes can leave child processes behind. The runner checks for
training-like processes before it starts and cleans leftover training children
after failed attempts. To run cleanup manually:

```bash
scripts/gtsm_context_ladder_train.sh clean
```
