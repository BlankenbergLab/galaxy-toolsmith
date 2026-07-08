# Model Selection Guide

This page summarizes practical model choices for Galaxy Toolsmith training and
generation workflows. The central policy is evidence-first: choose models by
same-slice benchmark results, deployment constraints, and operational stability,
not by parameter count alone.

## Selection policy

1. Compare model families across the same corpus slice.
2. Fine-tune LoRA profiles first for iteration; use opt-in full profiles only
   when the dataset, hardware, and learning-rate schedule justify it.
3. Promote a primary default only after benchmark and operability gates pass.
4. Treat DeepSeek, larger Qwen, and larger Mistral/Devstral variants as opt-in
   candidates until they beat the current baseline under the same gate.

## Recommended model tiers by hardware

| Hardware profile | Best starting tier | Higher-capability tier | Notes |
| --- | --- | --- | --- |
| 4xA100 40GB | `proto-qwen25-7b`, `deepseek-r1-distill-qwen-14b`, `agentic-devstral-24b` | `baseline-mistral-24b`, `deepseek-r1-distill-qwen-32b` | Use 7B/14B for iteration; benchmark 24B/32B before default promotion. |
| 2xA100 40GB | `proto-qwen25-7b`, `deepseek-coder-v2-lite-instruct`, `deepseek-r1-distill-qwen-14b` | `baseline-mistral-24b` with careful tuning | Prioritize memory envelope and throughput. |
| Apple Silicon MPS | `mps-qwen25-7b` | `mps-qwen25-14b`, `mps-mistral31-24b`, `mps-qwen25-32b` | Uses `mlx-lm`; CUDA-focused profiles are not the best fit on MPS. |

Existing profiles default to LoRA. Explicit `training_method` values are:

| Method | Best use | Notes |
| --- | --- | --- |
| `lora` | Default adapter tuning and local iteration. | 4-bit LoRA profiles run as effective QLoRA on HF/Axolotl. |
| `qlora` | Explicit 4-bit PEFT adapter tuning. | CUDA/HF/Axolotl path only; not supported by MLX-LM. |
| `full` | Full-parameter specialization on strong CUDA hosts. | Requires non-quantized profiles and much lower learning rates than LoRA. |

Opt-in full profiles include `full-qwen25-7b`, `full-mistral-24b`,
`full-deepseek-r1-distill-qwen-14b`, and
`full-deepseek-r1-distill-qwen-32b`. Treat 64B/70B full-parameter runs as
custom multi-GPU jobs rather than defaults; confirm the exact base model and
memory strategy before adding a profile.

## Choose-by-goal playbook

| Goal | Good first choice | Reason |
| --- | --- | --- |
| Fastest iteration | `proto-qwen25-7b` or `deepseek-coder-v2-lite-instruct` | Lower resource burden and faster experiment loops. |
| Best historical A100 benchmark operability | Qwen 2.5 Coder 7B safe profile | Completed training and reached 100/100 generation success in the 20260617 benchmark slice. |
| Current rich-context 4xA100 authoring example | `agentic-devstral-24b` through the context ladder | Completed the 20260707 sidecar run at 12k with raw source and fixture context. |
| Stronger coding or agentic exploration | `agentic-devstral-24b`, `baseline-mistral-24b`, DeepSeek distilled profiles | Potentially higher ceiling, but requires benchmark evidence to justify cost. |
| Simplest deployment path | Non-quantized tuning profile plus `export-model` and `export-ollama-model` | Separates training quality from quantized runtime packaging. |

## Current observed sidecar run

The current full pipeline example is `devstral-sidecars-fixtures-20260707` on
4xA100 40GB. The context ladder selected Devstral 24B at 12k context, `all-raw`
source context, fixture sidecars, and FSDP. It exported bf16 and `q4_k_m` GGUF
artifacts, then compared minibwa suite generation through full local PEFT and
q4 Ollama. Full local PEFT with Bioconda discovery was the strongest qualitative
path in that comparison. See [Experiments and Current Results](experiments.md)
and [A100 Pipeline Example](example.md).

## Observed results from 20260617T221121Z

These results come from one 4xA100 run and should be treated as exploratory.
XSD validation was not configured and Planemo lint/tests were not run in these
benchmark summaries.

### Training outcomes

| Model variant | Training setup | Outcome | Notes |
| --- | --- | --- | --- |
| Qwen 2.5 Coder 7B initial | 8,192 sequence length, per-device batch size 2 | Failed | CUDA OOM before useful training completion. |
| Qwen 2.5 Coder 7B safe | 4,096 sequence length, per-device batch size 1, gradient accumulation 2 | Completed | 1,985 samples, about 1,253 seconds elapsed. |
| Devstral 24B | 8,192 sequence length, per-device batch size 1, gradient accumulation 2, FSDP | Completed | 1,985 samples, about 10,846 seconds elapsed. |

### Benchmark outcomes

| Variant | Attempts | Success | XML/tool root | Throughput | Structural | Command presence | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen 7B baseline | 100 | 98 | 0.98 | 2.41 wrappers/min | 0.922 effective | 0.759 | Faster baseline, but two generation failures. |
| Qwen 7B safe candidate | 100 | 100 | 1.00 | 1.34 wrappers/min | 0.968 effective | 0.915 | Best observed reliability and command preservation. |
| Devstral 24B candidate | 100 | 96 | 0.97 | 0.216 wrappers/min | 0.882 effective | 0.857 | Slower and less reliable, with somewhat stronger datatype signals. |

Reference-fidelity details show that wrapper generation is not solved by XML
validity alone. Qwen 7B safe had strong generation reliability and requirement
package similarity, while Devstral 24B had higher input/output datatype Jaccard
scores in this run. Both still showed meaningful input/output count error.

## Portfolio comparison

| Model/profile | Pros | Cons | Current interpretation |
| --- | --- | --- | --- |
| `proto-qwen25-7b` / Qwen 2.5 Coder 7B | Fast, practical memory footprint, completed safe training, strong observed generation reliability. | Lower capability ceiling than 24B/32B models; safe settings reduce context length. | Best current iteration and deployment candidate based on observed data. |
| `agentic-devstral-24b` | Larger coding/agentic model, completed the 20260707 rich-context sidecar run, exported bf16 and `q4_k_m` GGUF artifacts. | Much slower than 7B iteration paths, heavier export/runtime path, lower 20260617 benchmark throughput. | Current best documented high-context authoring example; still needs broader Planemo-backed promotion evidence. |
| `baseline-mistral-24b` | Strong general instruction-following family and mature baseline target. | Similar operational cost class to Devstral; needs same-slice evidence. | Good comparison baseline for future promotion gates. |
| `deepseek-coder-v2-lite-instruct` | Coding-specialized and likely faster than larger reasoning models. | May underperform deeper reasoning tasks. | Good next fast-loop candidate. |
| `deepseek-r1-distill-qwen-14b` | Strong reasoning/coding balance with manageable A100 footprint. | Still needs tuning and benchmark evidence in this repo. | High-priority candidate for next model comparison. |
| `deepseek-r1-distill-qwen-32b` | Higher reasoning ceiling. | Slower and heavier; likely more difficult to iterate. | Advanced candidate only if quality gains justify cost. |
| Full `DeepSeek-R1` / `R1-Zero` | Very high reasoning potential. | Very large MoE footprint and more complex serving/training operations. | Not a practical default for the current local tuning path. |
| `mps-*` profiles | Fit Apple Silicon development through `mlx-lm`, with 7B as the safest starting tier. | Not representative of CUDA A100 deployment. | Useful for local experimentation, not the main benchmark tier. |

MLX and PEFT adapters use different formats even when filenames look similar.
Direct MLX LoRA to PEFT conversion is available only for known
Qwen/Llama/Mistral-style projection modules. For broader interchange, merge a
PEFT adapter into a full HF model and then convert/load the full model in the
target runtime.

## Export and deployment choices

The current export strategy keeps deployment flexible:

- Qwen 7B safe exported to `q4_k_m` GGUF and Ollama model
  `gtsm-tools-iuc-qwen25-7b-q4`.
- Devstral 24B exported to `q8_0`, `q6_k`, `q5_k_m`, and `q4_k_m` GGUF, with
  Ollama model `gtsm-tools-iuc-devstral-24b-q4`.

Pros:

- GGUF and Ollama make local runtime testing easier.
- Multiple Devstral quantizations allow quality/runtime tradeoff testing.
- Qwen 7B `q4_k_m` is a practical smoke/deployment target.

Cons:

- Quantized export does not prove wrapper quality.
- Ollama smoke tests need a running server and valid GGUF path.
- Large-model export can expose environment issues such as UTF-8 capture and
  shared-library conflicts.

## Promotion gate reminder

Before changing primary defaults:

1. Run `benchmark-generate` for the current baseline.
2. Run `benchmark-generate` for the candidate on the same corpus slice.
3. Run `promote-candidate` with baseline comparison.
4. For release-grade candidates, include XSD validation and
   `--run-planemo-tests`, then require Planemo pass status in the promotion
   policy.
