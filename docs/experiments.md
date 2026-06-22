# Experiments and Current Results

This page records the current evidence base for Galaxy Toolsmith model and
pipeline decisions. The main observed run is:

- Run tag: `20260617T221121Z`
- Corpus: `.gtsm-cache/datasets/tools-iuc-corpus.jsonl`
- Hardware: 4 NVIDIA A100 GPUs, 40GB each
- Primary run summary: `.gtsm-cache/runs/overnight/20260617T221121Z/summary.md`

The numbers below are useful for orientation and planning. They are not a final
production model ranking because the benchmark summaries did not run XSD
validation or Planemo lint/tests.

## Corpus and training data

| Metric | Value |
| --- | ---: |
| Corpus records | 1,985 |
| Trainable samples | 1,985 |
| Missing XML targets | 0 |
| Empty XML targets | 0 |
| Expanded XML targets | 197 |
| Wrapper XML targets | 1,788 |
| Container-help records | 643 |

The corpus combines wrapper XML, expanded macro output, shed metadata, command
metadata, and optional command help extracted from resolved containers. That mix
is important because wrapper generation needs both Galaxy XML patterns and
source-of-truth command-line interfaces.

## Training runs

| Run | Model/profile | Outcome | Key settings | Runtime |
| --- | --- | --- | --- | --- |
| `train-20260617T221121Z-qwen7b` | Qwen 2.5 Coder 7B | Failed | 8,192 context, batch size 2 | Failed with CUDA OOM. |
| `train-20260617T221121Z-qwen7b-safe` | Qwen 2.5 Coder 7B safe | Completed | 4,096 context, batch size 1, grad accumulation 2 | About 1,253 seconds. |
| `train-20260617T221121Z-devstral24b` | Devstral 24B | Completed | 8,192 context, batch size 1, grad accumulation 2, FSDP | About 10,846 seconds. |
| `train-d41f1819f749` | Qwen bootstrap subset | Completed | 197 samples, 4,096 context | About 485 seconds. |

The Qwen OOM followed by a successful safe run is the clearest operational
lesson: context length and batch sizing matter more than nominal model size when
trying to keep the pipeline reliable on 40GB A100s.

## Benchmark summaries

| Benchmark | Variant | Attempts | Success | XML well formed | Tool root | Repair attempt | Repair success | Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen smoke | Qwen 7B safe | 5 | 5 | 1.00 | 1.00 | 0.40 | 1.00 | 0.501 wrappers/min |
| Qwen baseline | repaired corpus Qwen 7B baseline | 100 | 98 | 0.98 | 0.98 | 0.27 | 0.926 | 2.412 wrappers/min |
| Qwen candidate | Qwen 7B safe | 100 | 100 | 1.00 | 1.00 | 0.49 | 1.00 | 1.340 wrappers/min |
| Devstral smoke | Devstral 24B | 5 | 5 | 1.00 | 1.00 | 0.20 | 1.00 | 0.013 wrappers/min |
| Devstral candidate | Devstral 24B | 100 | 96 | 0.97 | 0.97 | 0.60 | 0.933 | 0.216 wrappers/min |

The most presentation-friendly takeaway is that Qwen 7B safe produced the best
observed reliability and practical throughput, while Devstral 24B was much more
expensive to run and did not win the overall operational result.

## Fidelity and structure

| Benchmark | Effective structural score | Avg input count error | Avg output count error | Input datatype Jaccard | Output datatype Jaccard | Requirement Jaccard | Command presence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen baseline | 0.9225 | 6.694 | 1.847 | 0.280 | 0.239 | 0.196 | 0.759 |
| Qwen safe candidate | 0.9675 | 7.990 | 2.260 | 0.241 | 0.263 | 0.458 | 0.915 |
| Devstral candidate | 0.8820 | 7.208 | 2.021 | 0.332 | 0.328 | 0.356 | 0.857 |

These results show the main nuance. Qwen safe won reliability, structural score,
requirement-package similarity, and command preservation. Devstral had better
datatype overlap in this run, but it was slower and had more generation
failures. The next quality push should focus on interface fidelity, not only XML
validity.

## Export and Ollama artifacts

| Variant | Export status | Quantizations | Ollama name |
| --- | --- | --- | --- |
| `tools-iuc-qwen25-7b-full-20260617T221121Z-safe` | Completed | `q4_k_m` | `gtsm-tools-iuc-qwen25-7b-q4` |
| `tools-iuc-devstral-24b-full-20260617T221121Z` | Completed | `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m` | `gtsm-tools-iuc-devstral-24b-q4` |

The Devstral export includes an operational note: automated output capture hit a
UTF-8 decode issue while wrapping `llama.cpp`; the requested quantizations were
completed manually with `llama-quantize`. The codebase now has more explicit
UTF-8 handling around export subprocess capture and stricter Ollama Modelfile
path validation.

## Current pros and cons

| Choice | Pros | Cons |
| --- | --- | --- |
| Qwen 7B safe as iteration target | Reliable in the observed benchmark, fast enough for repeated slices, deployable as `q4_k_m` GGUF/Ollama. | Safe settings reduce context length; datatype fidelity still needs work. |
| Devstral 24B as high-capability candidate | Larger coding model, completed full training, exported multiple quantizations. | Slow benchmark throughput, lower success rate, more environment pressure. |
| Optional Planemo tests | Converts XML generation into executable Galaxy confidence. | Expensive, requires Galaxy/Planemo runtime, unsuitable for default unit tests. |
| Non-quantized first training | Keeps tuning quality separate from runtime compression. | Requires extra export step before local deployment. |
| Local run registry and status logs | Makes detached and overnight runs auditable. | Documentation must summarize cache artifacts carefully because paths are local. |

## Next experiments

- Run the same benchmark slice with XSD validation and `--run-planemo-tests`.
- TODO: Run deferred Planemo tool tests for the `20260617T221121Z` Qwen and
  Devstral benchmark outputs, using only paths from each `generation.records.json`
  so stale wrapper attempts are excluded. Reuse the cache-local Galaxy source at
  `.gtsm-cache/planemo/galaxy`, Galaxy virtualenv workspace at
  `.gtsm-cache/planemo/workspace`, and conda prefix at
  `.gtsm-cache/planemo/conda`; do not reinstall Galaxy for each model. Record
  reports under `.gtsm-cache/runs/planemo/20260617T221121Z/`, one directory per
  model.
- User-Defined Tool (UDT) support is now a first-class optional artifact path:
  `--artifact-format udt-yaml` generates and benchmarks Galaxy UDT YAML,
  validates against the vendored Galaxy `UserToolSource` JSON schema, and
  `convert-udt` converts supported UDT YAML into standard Galaxy tool XML.
  Training can use `--artifact-format xml`, `udt-yaml`, or `mixed`; mixed mode
  emits XML generation, UDT YAML generation, and paired UDT-to-XML examples when
  real targets exist in the corpus. Future work: add live Galaxy/Planemo runtime
  checks for UDTs once Galaxy exposes a stable non-UI validation path.
- Compare Qwen 7B safe against DeepSeek Coder V2 Lite and DeepSeek R1 Distill
  Qwen 14B.
- Add a harder benchmark slice with wrappers known to have complex conditionals,
  collections, multiple outputs, and datatype edge cases.
- Evaluate quantized Ollama runtime quality versus PEFT local generation for the
  same prompt set.
- Add targeted repair prompts for input/output count and datatype mismatch
  rather than only malformed XML.
