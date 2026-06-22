# Architecture and Decisions

Galaxy Toolsmith is organized as a library-first toolkit with a thin CLI layer.
The main objective is to make wrapper generation experiments reproducible:
corpus extraction, model tuning, generation, validation, benchmarking, export,
and promotion all leave structured artifacts under the workspace cache.

## Component map

| Area | Responsibility |
| --- | --- |
| `data` | Build the wrapper corpus from synced Galaxy sources, expanded XML, metadata, and optional container help output. |
| `prompts` | Load generation and critique templates, then shape command help and metadata into prompt context. |
| `providers` | Isolate generation backends: local PEFT, OpenAI-compatible APIs, Anthropic, Copilot-compatible APIs, and Ollama. |
| `inference` | Generate wrappers, validate XML, diagnose malformed output, evaluate wrapper structure, and run optional Planemo checks. |
| `orchestration` | Coordinate training, benchmark generation, promotion decisions, distributed training, and model export. |
| `runtime` | Track environment capabilities, run records, progress, status logs, and model source/cache policy. |
| `server` and `client` | Provide optional FastAPI remote generation and training coordination endpoints. |
| `cli` | Map commands to library calls while keeping behavior testable from Python. |

The CLI entrypoint is `gtsm`, but the implementation is intentionally split so
the same behavior can be used from tests, remote workers, and future workflow
systems without shelling out.

## Pipeline flows

### 1. Workspace and source sync

The setup flow initializes a local workspace, pulls `tools-iuc`, Galaxy skills,
and Galaxy XSD sources, and stores derived data under `.gtsm-cache/`.

Key commands:

```bash
gtsm init-workspace
gtsm sync-tools-iuc --ref main
gtsm sync-galaxy-skills --ref main
gtsm sync-galaxy-xsd --ref dev
```

### 2. Corpus extraction and enrichment

`extract-corpus` parses wrappers into wrapper-level JSONL records. It expands
wrapper macros with Galaxy's `galaxy.tool_util.loader` path first, preserving
the same macro semantics used by Planemo and Galaxy. It can preserve shed
metadata, resolve containers, probe command help inside Apptainer/Singularity or
Docker images, and use Bioconda or conda-forge source recipes to find likely
command entry points and upstream source checkouts.

The enriched corpus is the foundation for both supervised tuning and benchmark
sampling. In the observed A100 run, the corpus contained 1,985 records, all
trainable, with 643 records enriched by container help output. XML targets came
from 197 expanded wrapper records and 1,788 original wrapper targets.

### 3. Training

Training is driven by named profiles and Axolotl-compatible generated configs.
The current policy is to fine-tune non-quantized base models first, then export
quantized deployment artifacts. This keeps training quality and deployment
constraints separate.

Observed examples from the 20260617T221121Z run:

| Variant | Profile behavior | Outcome |
| --- | --- | --- |
| Qwen 2.5 Coder 7B, initial | 8,192 sequence length, per-device batch size 2 | Failed with CUDA OOM. |
| Qwen 2.5 Coder 7B, safe | 4,096 sequence length, per-device batch size 1, gradient accumulation 2 | Completed in about 1,253 seconds. |
| Devstral 24B | 8,192 sequence length, per-device batch size 1, gradient accumulation 2, FSDP | Completed in about 10,846 seconds. |

The training output is a PEFT adapter plus tokenizer/config metadata. Metrics
and logs remain in the run directory for later benchmarking and export.

### 4. Generation

Generation accepts command help text, optional source code, model/provider
settings, and prompt shaping limits. The provider returns Galaxy XML, which is
written to disk and validated immediately.

Generation records include:

- provider and model variant,
- output path,
- validation result,
- shaped prompt help statistics,
- interface hints extracted from usage/options,
- repair status and attempt count.

Benchmark generation uses the same path but samples from corpus records and can
shard work across local GPUs. Local PEFT backends can run one process per GPU or
model-parallel style depending on the requested topology.

### 5. Validation, repair, and evaluation

Validation starts with fast local checks: XML well-formedness, root tag, known
datatypes, and output diagnostics for common failure modes such as truncation or
unclosed CDATA. Benchmark repair can retry malformed output. Severe truncation
is reported as a benchmark failure by default; a compact placeholder fallback is
available only when explicitly enabled for smoke tests.

Evaluation can add:

- Galaxy XSD validation when an XSD path is configured,
- `planemo lint` with `--run-planemo`,
- `planemo test` with `--run-planemo-tests`,
- structural scores for command, inputs, outputs, tests, help, and citations,
- reference-fidelity comparison against corpus wrappers.

Planemo checks are deliberately optional because they require a heavier external
runtime. Promotion can require Planemo test pass status when live testing is
part of the gate.

### 6. Benchmarking and promotion

`benchmark-generate` produces generated artifacts, generation records,
evaluation summaries, compact benchmark summaries, and optional status logs. XML
wrappers remain the default artifact format, while `--artifact-format udt-yaml`
uses the UDT prompt and validates generated YAML against Galaxy's vendored
`UserToolSource` schema. The promotion layer compares candidate and baseline
summaries with named policies.

Benchmark metrics currently cover:

- generation success/failure counts,
- XML/tool-root validity or UDT YAML/schema validity,
- repair attempt and repair success rates,
- throughput,
- structural score,
- input/output count error,
- datatype and requirement-package Jaccard similarity,
- primary command presence.

UDT support is intentionally conservative. Schema-valid UDT YAML can be
converted to standard Galaxy tool XML with `convert-udt` when `shell_command`
uses simple `$(inputs.name)` or `$(inputs.name.path)` references. More complex
JavaScript expressions are reported instead of silently producing misleading XML.

These metrics are intentionally operational as well as structural. A model that
is slightly more faithful but dramatically slower or less reliable may still be
a poor default.

### 7. Export and deployment

`export-model` can merge adapters and produce GGUF quantizations. The Ollama
helpers write a Modelfile and can optionally register the model with the local
Ollama runtime.

Observed exports:

| Variant | Quantizations | Ollama name |
| --- | --- | --- |
| `tools-iuc-qwen25-7b-full-20260617T221121Z-safe` | `q4_k_m` | `gtsm-tools-iuc-qwen25-7b-q4` |
| `tools-iuc-devstral-24b-full-20260617T221121Z` | `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m` | `gtsm-tools-iuc-devstral-24b-q4` |

The Devstral export notes record that an automated wrapper hit UTF-8 decode
handling while capturing `llama.cpp` output; the requested GGUF quantizations
were completed manually with `llama-quantize`. That incident motivated explicit
UTF-8 handling and stricter Ollama/GGUF path validation.

### 8. Server and distributed operation

The optional server exposes remote generation, training job submission, worker
claim/heartbeat/complete endpoints, artifact listing/download, and a browser
monitoring dashboard. This is useful for long-running GPU experiments because
workers can run separately from the client shell and status can be inspected
through JSON endpoints or the dashboard.

The local run registry stores monitor records for generation, inference,
training, export, and server activity. This keeps operational state inspectable
even when a long run is detached.

## Design decisions

### Library-first, CLI-driven

The CLI is the main user interface, but library boundaries keep behavior
testable. This also makes future notebook, web, or workflow integrations easier.

### Non-quantized training before quantized deployment

Fine-tuning non-quantized models avoids mixing training instability with runtime
compression choices. Quantized GGUF and Ollama artifacts are produced after
training.

### Evidence-first model promotion

Model defaults should change only after same-slice benchmark comparison and a
promotion gate. The current benchmark data is useful for direction, but XSD and
Planemo were not configured in the observed summaries, so those runs should not
be treated as final production promotion evidence.

### Provider abstraction

Providers keep prompt construction and XML validation independent from model
runtime details. This allows local PEFT, hosted APIs, and Ollama to share the
same generation and evaluation path.

### Optional heavy dependencies

Planemo, server mode, training dependencies, local Unsloth/PEFT inference, and
Ollama are optional because not every development or CI environment has the same
external tools.

## Current risks and possibilities

| Area | Current risk | Possibility |
| --- | --- | --- |
| Wrapper fidelity | Generated wrappers can be valid XML but still miss detailed inputs, outputs, datatypes, or command semantics. | Add richer reference-aware scoring, Planemo test fixtures, and targeted repair loops for interface fidelity. |
| Throughput | Larger models can be too slow for broad benchmark slices. | Use smaller coding models for iteration, then reserve larger models for selected hard cases. |
| Validation depth | XML well-formedness is necessary but insufficient. | Make optional Planemo tests part of promotion policies for release candidates. |
| Corpus quality | Container help is valuable but incomplete and can depend on image behavior. | Improve command discovery, classify degraded help more precisely, and add source-derived examples. |
| Model portfolio | One observed run is not enough to declare a permanent default. | Compare Qwen, Devstral/Mistral, DeepSeek Coder, and DeepSeek distilled models on fixed benchmark slices. |
