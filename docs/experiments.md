# Experiments and Current Results

This page records the current evidence base for Galaxy Toolsmith model and
pipeline decisions. The current primary observed run is:

- Run tag: `devstral-sidecars-fixtures-20260707`
- Run directory: `.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/`
- Corpus: `.gtsm-cache/datasets/tools-iuc-corpus.jsonl`
- Corpus records: 1,985
- Hardware: 4 NVIDIA A100 GPUs, 40GB each
- Profile: `agentic-devstral-24b`
- Artifact format: `mixed`
- Selected context: `12288`
- Selected source mode: `all-raw`
- Selected distributed strategy: `fsdp`

The numbers below are useful for orientation and planning. They are not a final
production model ranking unless the row explicitly reports Planemo execution.
The 20260707 minibwa comparison is a structural generation comparison, not a
Planemo test run.

## Current 20260707 Sidecar Run

The run trained Devstral 24B with mixed XML/UDT targets and enriched sidecar
context:

| Context | Included |
| --- | --- |
| Wrapper XML and expanded XML | Yes |
| UDT YAML examples | Yes, through `mixed` artifact format where available or synthesized |
| Container command help | Yes, when extracted from resolved containers |
| Upstream source code | Yes, with `all-raw` selected for the final run |
| Wrapper helper/configfile code | Yes, as context for understanding existing wrappers |
| Upstream tests/examples/fixtures | Yes, through `TEST_CONTEXT_MODE=fixtures` |
| Shed metadata | Yes |

The context ladder tried 32k, 24k, 16k, 12k, 8k, 4k, and 2k. Both `all-raw`
and `all-filtered` source modes were evaluated, and both FSDP and DeepSpeed
ZeRO-3 were probed. The selected candidate was the first GPU-stable rung:

| Setting | Value |
| --- | --- |
| Context length | `12288` |
| Source context | `all-raw` |
| Source budget | `24000` chars, `96` files |
| Test sidecar mode | `fixtures` |
| Test sidecar budget | `4000` chars, `6` files, `64KB` per file |
| Distributed strategy | `fsdp` |
| GPU processes | `4` |
| Variant id | `tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707` |

High-context probe summary:

| Context | Source mode | Strategy | Result | Approx p99 tokens |
| --- | --- | --- | --- | ---: |
| 32k | `all-raw` / `all-filtered` | FSDP and ZeRO-3 | Failed probe | 55,604 to 55,656 |
| 24k | `all-raw` / `all-filtered` | FSDP and ZeRO-3 | Failed probe | 48,368 to 48,838 |
| 16k | `all-raw` / `all-filtered` | FSDP and ZeRO-3 | Failed probe | 44,290 to 44,831 |
| 12k | `all-raw` | FSDP | Passed probe and selected | 42,104 |

The full training run reached `training-finished` and completed. The first
post-export status recorded `post-export-failed`, but the failure was limited
to the `ollama create` registration step because the separate export
environment did not have `ollama` on `PATH`. The bf16 GGUF and `q4_k_m`
quantized GGUF were produced.

## Current Export Artifacts

| Artifact | Path or value |
| --- | --- |
| bf16 GGUF | `.gtsm-cache/models/exports/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707/gguf/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707.bf16.gguf` |
| q4_k_m GGUF | `.gtsm-cache/models/exports/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707/gguf/q4_k_m/tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-devstral-sidecars-fixtures-20260707-q4_k_m.gguf` |
| Ollama model | `gtsm-tools-iuc-devstral-24b-mixed-all-raw-12288-fsdp-3f8e387314` |

The Ollama name is normalized and shortened so it remains valid for local model
registration while preserving the distinguishing variant hash.

## Minibwa Full vs Q4 Comparison

The minibwa comparison tested suite generation from the full local PEFT path
and from the q4 Ollama export, both with Bioconda discovery and with static
help/source context. Results were written under:

```text
.gtsm-cache/runs/context-ladder/devstral-sidecars-fixtures-20260707/minibwa-tests/
```

| Scenario | Exit | Records | Tool XML | Repairs | Unknown datatypes | Params | Required params | Outputs | Tests | Requirements | Citations | Help chars | Command chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full local PEFT, Bioconda discovery | 0 | 11 | 11 | 0 | 8 | 87 | 87 | 13 | 11 | 11 | 11 | 14485 | 1517 |
| Full local PEFT, static help/source | 0 | 11 | 11 | 0 | 8 | 61 | 61 | 11 | 11 | 11 | 11 | 7656 | 1223 |
| Ollama q4, Bioconda discovery | 0 | 11 | 11 | 0 | 7 | 129 | 124 | 13 | 11 | 11 | 9 | 608 | 4684 |
| Ollama q4, static help/source | 0 | 11 | 11 | 0 | 7 | 129 | 124 | 13 | 11 | 11 | 9 | 608 | 4721 |

Qualitative interpretation:

- Full local PEFT with Bioconda discovery was the best current result.
- Full local PEFT with static help/source remained structurally valid, but the
  reduced discovery context made some command shapes weaker.
- Ollama q4 generated valid XML but was more verbose and speculative, with much
  shorter help text and fewer Toolsmith citations.
- Discovery mode provided more useful runtime/source context than a static
  `help.txt` for this package.

Known review items from this comparison:

- Unknown datatypes still need curation or package-specific mapping.
- These outputs were not Planemo-executed, so runtime correctness is not proven.
- q4 quality is useful for deployment testing, but full local PEFT remains the
  stronger authoring path for this slice.

## Historical 20260617 Benchmark Run

The older `20260617T221121Z` run remains useful as a baseline for early model
selection and benchmark behavior:

- Corpus: `.gtsm-cache/datasets/tools-iuc-corpus.jsonl`
- Hardware: 4 NVIDIA A100 GPUs, 40GB each
- Primary run summary: `.gtsm-cache/runs/overnight/20260617T221121Z/summary.md`

### Corpus and training data

| Metric | Value |
| --- | ---: |
| Corpus records | 1,985 |
| Trainable samples | 1,985 |
| Missing XML targets | 0 |
| Empty XML targets | 0 |
| Expanded XML targets | 197 |
| Wrapper XML targets | 1,788 |
| Container-help records | 643 |

### Training runs

| Run | Model/profile | Outcome | Key settings | Runtime |
| --- | --- | --- | --- | --- |
| `train-20260617T221121Z-qwen7b` | Qwen 2.5 Coder 7B | Failed | 8,192 context, batch size 2 | Failed with CUDA OOM. |
| `train-20260617T221121Z-qwen7b-safe` | Qwen 2.5 Coder 7B safe | Completed | 4,096 context, batch size 1, grad accumulation 2 | About 1,253 seconds. |
| `train-20260617T221121Z-devstral24b` | Devstral 24B | Completed | 8,192 context, batch size 1, grad accumulation 2, FSDP | About 10,846 seconds. |
| `train-d41f1819f749` | Qwen bootstrap subset | Completed | 197 samples, 4,096 context | About 485 seconds. |

The Qwen OOM followed by a successful safe run is an operational reminder that
context length and batch sizing matter as much as nominal model size on 40GB
A100s.

### Benchmark summaries

| Benchmark | Variant | Attempts | Success | XML well formed | Tool root | Repair attempt | Repair success | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen smoke | Qwen 7B safe | 5 | 5 | 1.00 | 1.00 | 0.40 | 1.00 | 0.501 wrappers/min |
| Qwen baseline | repaired corpus Qwen 7B baseline | 100 | 98 | 0.98 | 0.98 | 0.27 | 0.926 | 2.412 wrappers/min |
| Qwen candidate | Qwen 7B safe | 100 | 100 | 1.00 | 1.00 | 0.49 | 1.00 | 1.340 wrappers/min |
| Devstral smoke | Devstral 24B | 5 | 5 | 1.00 | 1.00 | 0.20 | 1.00 | 0.013 wrappers/min |
| Devstral candidate | Devstral 24B | 100 | 96 | 0.97 | 0.97 | 0.60 | 0.933 | 0.216 wrappers/min |

### Fidelity and structure

| Benchmark | Effective structural score | Avg input count error | Avg output count error | Input datatype Jaccard | Output datatype Jaccard | Requirement Jaccard | Command presence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen baseline | 0.9225 | 6.694 | 1.847 | 0.280 | 0.239 | 0.196 | 0.759 |
| Qwen safe candidate | 0.9675 | 7.990 | 2.260 | 0.241 | 0.263 | 0.458 | 0.915 |
| Devstral candidate | 0.8820 | 7.208 | 2.021 | 0.332 | 0.328 | 0.356 | 0.857 |

### Historical export artifacts

| Variant | Export status | Quantizations | Ollama name |
| --- | --- | --- | --- |
| `tools-iuc-qwen25-7b-full-20260617T221121Z-safe` | Completed | `q4_k_m` | `gtsm-tools-iuc-qwen25-7b-q4` |
| `tools-iuc-devstral-24b-full-20260617T221121Z` | Completed | `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m` | `gtsm-tools-iuc-devstral-24b-q4` |

## Current Pros and Cons

| Choice | Pros | Cons |
| --- | --- | --- |
| Full local Devstral 24B PEFT authoring | Best current minibwa qualitative result, can use rich discovery/source/test sidecars. | Slow, GPU-heavy, requires careful context ladder selection. |
| Ollama q4 deployment | Small enough for practical local serving, validates the export path. | Lower semantic quality in the observed minibwa slice; context and output can truncate without careful runtime settings. |
| Bioconda/BioContainers discovery | Better command and source context than static help for minibwa. | Depends on package availability, container behavior, and source recipe quality. |
| Optional Planemo tests | Converts structural XML checks into executable Galaxy confidence. | Expensive, requires Galaxy/Planemo runtime, unsuitable for default unit tests. |
| Test/example sidecars | Adds useful evidence without turning fixtures into primary generated artifacts. | Raises token pressure and may force lower context/source budgets. |

## Next Experiments

- Run Planemo lint/tests for selected minibwa outputs and a broader benchmark
  slice.
- Compare q4 Ollama generation with larger runtime context settings against
  full local PEFT for the same suite plans.
- Add datatype curation for unknown minibwa-style index and alignment formats.
- Continue testing sidecar budgets so tests/examples improve command fidelity
  without forcing the ladder below 12k on 4xA100.
- Compare Devstral 24B against smaller Qwen/DeepSeek profiles on the same
  source/test-sidecar corpus.
