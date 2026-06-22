#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-run}"
case "$MODE" in
  run|resume|status|stop-stretch-candidate) ;;
  *)
    echo "Usage: $0 {run|resume|status|stop-stretch-candidate}" >&2
    exit 2
    ;;
esac

OVERNIGHT_ROOT="${OVERNIGHT_ROOT:-.gtsm-cache/runs/overnight}"
CURRENT_FILE="$OVERNIGHT_ROOT/current"

if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ "$MODE" == "run" ]]; then
    RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
  elif [[ -f "$CURRENT_FILE" ]]; then
    RUN_TAG="$(cat "$CURRENT_FILE")"
  else
    echo "RUN_TAG is not set and no current overnight run exists." >&2
    exit 2
  fi
fi

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must contain only letters, numbers, '.', '_', and '-'." >&2
  exit 2
fi

RUN_ROOT="$OVERNIGHT_ROOT/$RUN_TAG"
STATE_ENV="$RUN_ROOT/state.env"
if [[ "$MODE" != "run" && ! -f "$STATE_ENV" ]]; then
  echo "No state found for RUN_TAG=$RUN_TAG at $STATE_ENV." >&2
  exit 2
fi

BENCHMARK_GPU_TOPOLOGY_OVERRIDE_SET=0
BENCHMARK_OFFLOAD_POLICY_OVERRIDE_SET=0
BENCHMARK_RESUME_EXISTING_OVERRIDE_SET=0
BENCHMARK_RECORD_TIMEOUT_SECONDS_OVERRIDE_SET=0
BENCHMARK_GPU_MEMORY_RESERVE_GIB_OVERRIDE_SET=0
BENCHMARK_PREFLIGHT_OVERRIDE_SET=0
if [[ ${BENCHMARK_GPU_TOPOLOGY+x} ]]; then
  BENCHMARK_GPU_TOPOLOGY_OVERRIDE_SET=1
  BENCHMARK_GPU_TOPOLOGY_OVERRIDE="$BENCHMARK_GPU_TOPOLOGY"
fi
if [[ ${BENCHMARK_OFFLOAD_POLICY+x} ]]; then
  BENCHMARK_OFFLOAD_POLICY_OVERRIDE_SET=1
  BENCHMARK_OFFLOAD_POLICY_OVERRIDE="$BENCHMARK_OFFLOAD_POLICY"
fi
if [[ ${BENCHMARK_RESUME_EXISTING+x} ]]; then
  BENCHMARK_RESUME_EXISTING_OVERRIDE_SET=1
  BENCHMARK_RESUME_EXISTING_OVERRIDE="$BENCHMARK_RESUME_EXISTING"
fi
if [[ ${BENCHMARK_RECORD_TIMEOUT_SECONDS+x} ]]; then
  BENCHMARK_RECORD_TIMEOUT_SECONDS_OVERRIDE_SET=1
  BENCHMARK_RECORD_TIMEOUT_SECONDS_OVERRIDE="$BENCHMARK_RECORD_TIMEOUT_SECONDS"
fi
if [[ ${BENCHMARK_GPU_MEMORY_RESERVE_GIB+x} ]]; then
  BENCHMARK_GPU_MEMORY_RESERVE_GIB_OVERRIDE_SET=1
  BENCHMARK_GPU_MEMORY_RESERVE_GIB_OVERRIDE="$BENCHMARK_GPU_MEMORY_RESERVE_GIB"
fi
if [[ ${BENCHMARK_PREFLIGHT+x} ]]; then
  BENCHMARK_PREFLIGHT_OVERRIDE_SET=1
  BENCHMARK_PREFLIGHT_OVERRIDE="$BENCHMARK_PREFLIGHT"
fi

mkdir -p "$RUN_ROOT"
if [[ -f "$STATE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_ENV"
fi
if [[ "$BENCHMARK_GPU_TOPOLOGY_OVERRIDE_SET" == "1" ]]; then
  BENCHMARK_GPU_TOPOLOGY="$BENCHMARK_GPU_TOPOLOGY_OVERRIDE"
fi
if [[ "$BENCHMARK_OFFLOAD_POLICY_OVERRIDE_SET" == "1" ]]; then
  BENCHMARK_OFFLOAD_POLICY="$BENCHMARK_OFFLOAD_POLICY_OVERRIDE"
fi
if [[ "$BENCHMARK_RESUME_EXISTING_OVERRIDE_SET" == "1" ]]; then
  BENCHMARK_RESUME_EXISTING="$BENCHMARK_RESUME_EXISTING_OVERRIDE"
fi
if [[ "$BENCHMARK_RECORD_TIMEOUT_SECONDS_OVERRIDE_SET" == "1" ]]; then
  BENCHMARK_RECORD_TIMEOUT_SECONDS="$BENCHMARK_RECORD_TIMEOUT_SECONDS_OVERRIDE"
fi
if [[ "$BENCHMARK_GPU_MEMORY_RESERVE_GIB_OVERRIDE_SET" == "1" ]]; then
  BENCHMARK_GPU_MEMORY_RESERVE_GIB="$BENCHMARK_GPU_MEMORY_RESERVE_GIB_OVERRIDE"
fi
if [[ "$BENCHMARK_PREFLIGHT_OVERRIDE_SET" == "1" ]]; then
  BENCHMARK_PREFLIGHT="$BENCHMARK_PREFLIGHT_OVERRIDE"
fi

: "${DRY_RUN:=0}"
: "${GTSM:=gtsm}"
: "${CORPUS_JSONL:=.gtsm-cache/datasets/tools-iuc-corpus.jsonl}"
: "${GPU_DEVICES:=0,1,2,3}"
: "${NUM_PROCESSES:=4}"
: "${SMOKE_LIMIT:=5}"
: "${SMOKE_MIN_SUCCEEDED:=4}"
: "${CANDIDATE_LIMIT:=100}"
: "${MIN_TRAIN_SAMPLES:=1500}"
: "${REBUILD_CORPUS_ON_LOW_SAMPLES:=1}"
: "${EXTRACT_MAX_WORKERS:=16}"
: "${CONTAINER_RUNTIME:=apptainer}"
: "${CONTAINER_HELP_PROBE_MODE:=exploratory}"
: "${CONTAINER_CACHE_DIR:=.gtsm-cache/containers}"
: "${CORPUS_CHECKPOINT:=.gtsm-cache/datasets/tools-iuc-corpus.checkpoint}"
: "${BIOCONDA_REF:=master}"
: "${DOCKER_USE_SUDO:=0}"
: "${NO_FETCH_DOCS:=1}"
: "${BASELINE_VARIANT:=repaired-corpus-qwen25-7b-nonquant-4gpu-smoke}"
: "${QWEN_PROFILE:=proto-qwen25-7b}"
: "${STRETCH_PROFILE:=agentic-devstral-24b}"
: "${STRETCH_4BIT_PROFILE:=agentic-devstral-24b-4bit}"
: "${STRETCH_ON_SUCCESS:=1}"
: "${QWEN_DISTRIBUTED_STRATEGY:=ddp}"
: "${STRETCH_DISTRIBUTED_STRATEGY:=fsdp}"
: "${STRETCH_DEEPSPEED_FALLBACK:=1}"
: "${STRETCH_DEEPSPEED_OFFLOAD_FALLBACK:=1}"
: "${STRETCH_4BIT_FALLBACK:=0}"
: "${EXPORT_QUANTIZATIONS:=q8_0,q6_k,q5_k_m,q4_k_m}"
: "${EXPORT_ON_BENCHMARK_FAILURE:=1}"
: "${MAX_TOKENS:=4096}"
: "${MAX_PROMPT_HELP_CHARS:=4000}"
: "${DATASET_MANIFEST:=config/dataset.manifest.json}"
: "${STATUS_INTERVAL_SECONDS:=30}"
: "${LOG_TAIL_LINES:=40}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
: "${BENCHMARK_GPU_TOPOLOGY:=per-process}"
: "${BENCHMARK_OFFLOAD_POLICY:=allow}"
: "${BENCHMARK_RESUME_EXISTING:=1}"
: "${BENCHMARK_RECORD_TIMEOUT_SECONDS:=0}"
: "${BENCHMARK_GPU_MEMORY_RESERVE_GIB:=2.0}"
: "${BENCHMARK_PREFLIGHT:=1}"
: "${QWEN_VARIANT:=tools-iuc-qwen25-7b-full-$RUN_TAG}"
: "${QWEN_SAFE_VARIANT:=tools-iuc-qwen25-7b-full-$RUN_TAG-safe}"
: "${QWEN_DRYRUN_VARIANT:=tools-iuc-qwen25-7b-full-$RUN_TAG-dryrun}"
: "${STRETCH_VARIANT:=tools-iuc-devstral-24b-full-$RUN_TAG}"
: "${STRETCH_SAFE_VARIANT:=tools-iuc-devstral-24b-full-$RUN_TAG-safe}"
: "${STRETCH_DEEPSPEED_VARIANT:=tools-iuc-devstral-24b-full-$RUN_TAG-zero3}"
: "${STRETCH_DEEPSPEED_OFFLOAD_VARIANT:=tools-iuc-devstral-24b-full-$RUN_TAG-zero3-offload}"
: "${STRETCH_4BIT_VARIANT:=tools-iuc-devstral-24b-4bit-$RUN_TAG}"
: "${ACTIVE_QWEN_VARIANT:=}"
: "${ACTIVE_STRETCH_VARIANT:=}"

LOG_DIR="$RUN_ROOT/logs"
MARKER_DIR="$RUN_ROOT/markers"
BENCHMARK_DIR="$RUN_ROOT/benchmarks"
EXPORT_DIR="$RUN_ROOT/exports"
RUN_LOG="$RUN_ROOT/run.log"
SUMMARY_MD="$RUN_ROOT/summary.md"
TRAIN_ROOT=".gtsm-cache/runs/training"
MODEL_VARIANTS_DIR=".gtsm-cache/models/variants"

mkdir -p "$LOG_DIR" "$MARKER_DIR" "$BENCHMARK_DIR" "$EXPORT_DIR"

STEPS=(
  preflight
  qwen_dry_run
  qwen_train
  qwen_smoke
  qwen_candidate
  qwen_baseline
  qwen_promotion
  qwen_export
  stretch_train
  stretch_smoke
  stretch_candidate
  stretch_promotion
  stretch_export
)

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_dry_run() {
  is_true "$DRY_RUN"
}

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  local message="$*"
  printf '[%s] %s\n' "$(timestamp)" "$message" | tee -a "$RUN_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

save_state() {
  {
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'GTSM=%q\n' "$GTSM"
    printf 'CORPUS_JSONL=%q\n' "$CORPUS_JSONL"
    printf 'GPU_DEVICES=%q\n' "$GPU_DEVICES"
    printf 'NUM_PROCESSES=%q\n' "$NUM_PROCESSES"
    printf 'SMOKE_LIMIT=%q\n' "$SMOKE_LIMIT"
    printf 'SMOKE_MIN_SUCCEEDED=%q\n' "$SMOKE_MIN_SUCCEEDED"
    printf 'CANDIDATE_LIMIT=%q\n' "$CANDIDATE_LIMIT"
    printf 'MIN_TRAIN_SAMPLES=%q\n' "$MIN_TRAIN_SAMPLES"
    printf 'REBUILD_CORPUS_ON_LOW_SAMPLES=%q\n' "$REBUILD_CORPUS_ON_LOW_SAMPLES"
    printf 'EXTRACT_MAX_WORKERS=%q\n' "$EXTRACT_MAX_WORKERS"
    printf 'CONTAINER_RUNTIME=%q\n' "$CONTAINER_RUNTIME"
    printf 'CONTAINER_HELP_PROBE_MODE=%q\n' "$CONTAINER_HELP_PROBE_MODE"
    printf 'CONTAINER_CACHE_DIR=%q\n' "$CONTAINER_CACHE_DIR"
    printf 'CORPUS_CHECKPOINT=%q\n' "$CORPUS_CHECKPOINT"
    printf 'BIOCONDA_REF=%q\n' "$BIOCONDA_REF"
    printf 'DOCKER_USE_SUDO=%q\n' "$DOCKER_USE_SUDO"
    printf 'NO_FETCH_DOCS=%q\n' "$NO_FETCH_DOCS"
    printf 'BASELINE_VARIANT=%q\n' "$BASELINE_VARIANT"
    printf 'QWEN_PROFILE=%q\n' "$QWEN_PROFILE"
    printf 'STRETCH_PROFILE=%q\n' "$STRETCH_PROFILE"
    printf 'STRETCH_4BIT_PROFILE=%q\n' "$STRETCH_4BIT_PROFILE"
    printf 'STRETCH_ON_SUCCESS=%q\n' "$STRETCH_ON_SUCCESS"
    printf 'QWEN_DISTRIBUTED_STRATEGY=%q\n' "$QWEN_DISTRIBUTED_STRATEGY"
    printf 'STRETCH_DISTRIBUTED_STRATEGY=%q\n' "$STRETCH_DISTRIBUTED_STRATEGY"
    printf 'STRETCH_DEEPSPEED_FALLBACK=%q\n' "$STRETCH_DEEPSPEED_FALLBACK"
    printf 'STRETCH_DEEPSPEED_OFFLOAD_FALLBACK=%q\n' "$STRETCH_DEEPSPEED_OFFLOAD_FALLBACK"
    printf 'STRETCH_4BIT_FALLBACK=%q\n' "$STRETCH_4BIT_FALLBACK"
    printf 'EXPORT_QUANTIZATIONS=%q\n' "$EXPORT_QUANTIZATIONS"
    printf 'EXPORT_ON_BENCHMARK_FAILURE=%q\n' "$EXPORT_ON_BENCHMARK_FAILURE"
    printf 'MAX_TOKENS=%q\n' "$MAX_TOKENS"
    printf 'MAX_PROMPT_HELP_CHARS=%q\n' "$MAX_PROMPT_HELP_CHARS"
    printf 'DATASET_MANIFEST=%q\n' "$DATASET_MANIFEST"
    printf 'STATUS_INTERVAL_SECONDS=%q\n' "$STATUS_INTERVAL_SECONDS"
    printf 'LOG_TAIL_LINES=%q\n' "$LOG_TAIL_LINES"
    printf 'PYTORCH_CUDA_ALLOC_CONF=%q\n' "$PYTORCH_CUDA_ALLOC_CONF"
    printf 'BENCHMARK_GPU_TOPOLOGY=%q\n' "$BENCHMARK_GPU_TOPOLOGY"
    printf 'BENCHMARK_OFFLOAD_POLICY=%q\n' "$BENCHMARK_OFFLOAD_POLICY"
    printf 'BENCHMARK_RESUME_EXISTING=%q\n' "$BENCHMARK_RESUME_EXISTING"
    printf 'BENCHMARK_RECORD_TIMEOUT_SECONDS=%q\n' "$BENCHMARK_RECORD_TIMEOUT_SECONDS"
    printf 'BENCHMARK_GPU_MEMORY_RESERVE_GIB=%q\n' "$BENCHMARK_GPU_MEMORY_RESERVE_GIB"
    printf 'BENCHMARK_PREFLIGHT=%q\n' "$BENCHMARK_PREFLIGHT"
    printf 'QWEN_VARIANT=%q\n' "$QWEN_VARIANT"
    printf 'QWEN_SAFE_VARIANT=%q\n' "$QWEN_SAFE_VARIANT"
    printf 'QWEN_DRYRUN_VARIANT=%q\n' "$QWEN_DRYRUN_VARIANT"
    printf 'STRETCH_VARIANT=%q\n' "$STRETCH_VARIANT"
    printf 'STRETCH_SAFE_VARIANT=%q\n' "$STRETCH_SAFE_VARIANT"
    printf 'STRETCH_DEEPSPEED_VARIANT=%q\n' "$STRETCH_DEEPSPEED_VARIANT"
    printf 'STRETCH_DEEPSPEED_OFFLOAD_VARIANT=%q\n' "$STRETCH_DEEPSPEED_OFFLOAD_VARIANT"
    printf 'STRETCH_4BIT_VARIANT=%q\n' "$STRETCH_4BIT_VARIANT"
    printf 'ACTIVE_QWEN_VARIANT=%q\n' "$ACTIVE_QWEN_VARIANT"
    printf 'ACTIVE_STRETCH_VARIANT=%q\n' "$ACTIVE_STRETCH_VARIANT"
  } > "$STATE_ENV"
  printf '%s\n' "$RUN_TAG" > "$CURRENT_FILE"
}

command_string() {
  local rendered=""
  local arg
  for arg in "$@"; do
    printf -v rendered '%s%q ' "$rendered" "$arg"
  done
  printf '%s' "${rendered% }"
}

run_cmd_allow_failure() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  log "+ $(command_string "$@")"
  if is_dry_run; then
    return 0
  fi
  set +e
  "$@" 2>&1 | tee -a "$RUN_LOG" "$log_file"
  local rc="${PIPESTATUS[0]}"
  set -e
  return "$rc"
}

run_cmd() {
  local log_file="$1"
  shift
  run_cmd_allow_failure "$log_file" "$@" || die "Command failed: $(command_string "$@")"
}

run_json_cmd() {
  local output_json="$1"
  local log_file="$2"
  shift 2
  mkdir -p "$(dirname "$output_json")" "$(dirname "$log_file")"
  log "+ $(command_string "$@") > $output_json"
  if is_dry_run; then
    return 0
  fi
  local stdout_tmp="$output_json.stdout.tmp"
  local stderr_tmp="$output_json.stderr.tmp"
  set +e
  "$@" > "$stdout_tmp" 2> "$stderr_tmp"
  local rc="$?"
  set -e
  cat "$stdout_tmp" | tee -a "$RUN_LOG" "$log_file"
  cat "$stderr_tmp" | tee -a "$RUN_LOG" "$log_file" >&2
  if [[ "$rc" -eq 0 ]]; then
    cp "$stdout_tmp" "$output_json"
  fi
  rm -f "$stdout_tmp" "$stderr_tmp"
  return "$rc"
}

marker_path() {
  printf '%s/%s.ok' "$MARKER_DIR" "$1"
}

mark_done() {
  local step="$1"
  if ! is_dry_run; then
    printf '%s\n' "$(timestamp)" > "$(marker_path "$step")"
  fi
}

step_done() {
  [[ -f "$(marker_path "$1")" ]]
}

append_summary() {
  mkdir -p "$(dirname "$SUMMARY_MD")"
  printf '%s\n' "$*" >> "$SUMMARY_MD"
}

json_read() {
  local path="$1"
  local expression="$2"
  local default_value="${3:-}"
  if [[ ! -f "$path" ]]; then
    printf '%s' "$default_value"
    return 0
  fi
  jq -r "$expression // \"$default_value\"" "$path"
}

csv_count() {
  local value="$1"
  local IFS=,
  local -a parts
  read -r -a parts <<< "$value"
  printf '%s' "${#parts[@]}"
}

run_step() {
  local step="$1"
  local function_name="$2"
  local description="$3"
  if step_done "$step"; then
    log "Skipping $step: marker exists."
    return 0
  fi
  log "Starting $step: $description"
  "$function_name"
  mark_done "$step"
  log "Completed $step."
}

training_status() {
  local run_id="$1"
  local manifest="$TRAIN_ROOT/$run_id/run.manifest.json"
  json_read "$manifest" '.status' "missing"
}

training_metrics_path() {
  local run_id="$1"
  printf '%s/%s/metrics.json' "$TRAIN_ROOT" "$run_id"
}

training_failed_text() {
  local run_id="$1"
  local extra_log="$2"
  local metrics
  metrics="$(training_metrics_path "$run_id")"
  {
    [[ -f "$TRAIN_ROOT/$run_id/run.manifest.json" ]] && cat "$TRAIN_ROOT/$run_id/run.manifest.json"
    [[ -f "$metrics" ]] && cat "$metrics"
    [[ -f "$extra_log" ]] && cat "$extra_log"
  } 2>/dev/null || true
}

failure_looks_like_oom() {
  local run_id="$1"
  local log_file="$2"
  training_failed_text "$run_id" "$log_file" | grep -Eiq 'out of memory|cuda.*memory|CUDA.*OOM|OOM|CUBLAS_STATUS_ALLOC_FAILED'
}

train_once() {
  local profile="$1"
  local variant="$2"
  local run_id="$3"
  local seq_len="$4"
  local batch_size="$5"
  local grad_accum="$6"
  local status_log="$7"
  local log_file="$8"
  local distributed_strategy="${9:-ddp}"

  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  export PYTORCH_CUDA_ALLOC_CONF
  run_cmd_allow_failure \
    "$log_file" \
    "$GTSM" train \
      --profile "$profile" \
      --dataset-manifest "$DATASET_MANIFEST" \
      --corpus-jsonl "$CORPUS_JSONL" \
      --variant-id "$variant" \
      --backend axolotl \
      --num-processes "$NUM_PROCESSES" \
      --distributed-strategy "$distributed_strategy" \
      --max-seq-length "$seq_len" \
      --per-device-batch-size "$batch_size" \
      --gradient-accumulation-steps "$grad_accum" \
      --status-log "$status_log" \
      --status-interval-seconds "$STATUS_INTERVAL_SECONDS" \
      --stream-logs \
      --log-tail-lines "$LOG_TAIL_LINES" \
      --internal-run-id "$run_id"

  if is_dry_run; then
    return 0
  fi

  local status
  status="$(training_status "$run_id")"
  if [[ "$status" == "completed" && -f "$MODEL_VARIANTS_DIR/$variant.manifest.json" ]]; then
    return 0
  fi
  return 1
}

benchmark_once() {
  local label="$1"
  local variant="$2"
  local limit="$3"
  local out_dir="$4"
  local status_log="$5"
  local log_file="$6"

  mkdir -p "$out_dir"
  if ! is_dry_run && [[ -f "$out_dir/benchmark.summary.json" ]]; then
    log "Using existing benchmark summary for $label: $out_dir/benchmark.summary.json"
    append_summary ""
    append_summary "## Benchmark: $label"
    "$GTSM" benchmark-summary --summary "$out_dir/benchmark.summary.json" | tee -a "$RUN_LOG" "$SUMMARY_MD"
    return 0
  fi

  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  export PYTORCH_CUDA_ALLOC_CONF
  if command -v nvidia-smi >/dev/null 2>&1; then
    run_cmd_allow_failure "$log_file" nvidia-smi || true
  fi

  local benchmark_args=(
    benchmark-generate
      --corpus-jsonl "$CORPUS_JSONL" \
      --limit "$limit" \
      --provider local \
      --model-variant "$variant" \
      --temperature 0 \
      --max-tokens "$MAX_TOKENS" \
      --max-prompt-help-chars "$MAX_PROMPT_HELP_CHARS" \
      --repair-invalid-xml \
      --num-processes "$NUM_PROCESSES" \
      --gpu-devices "$GPU_DEVICES" \
      --min-items-per-process 1 \
      --local-gpu-topology "$BENCHMARK_GPU_TOPOLOGY" \
      --local-offload-policy "$BENCHMARK_OFFLOAD_POLICY" \
      --local-gpu-memory-reserve-gib "$BENCHMARK_GPU_MEMORY_RESERVE_GIB" \
      --record-timeout-seconds "$BENCHMARK_RECORD_TIMEOUT_SECONDS" \
      --checkpoint-records "$out_dir/checkpoint.records.jsonl" \
      --wrappers-dir "$out_dir/wrappers" \
      --generation-records "$out_dir/generation.records.json" \
      --evaluation-report "$out_dir/evaluation.summary.json" \
      --benchmark-summary "$out_dir/benchmark.summary.json" \
      --status-log "$status_log"
  )
  if is_true "$BENCHMARK_RESUME_EXISTING"; then
    benchmark_args+=(--resume-existing)
  fi

  local strict_preflight=0
  if [[ "$BENCHMARK_GPU_TOPOLOGY" == "model-parallel" ]] || [[ "$BENCHMARK_OFFLOAD_POLICY" == "fail" ]]; then
    strict_preflight=1
  fi
  if is_true "$BENCHMARK_PREFLIGHT" && [[ "$label" == *candidate* ]] && [[ "$strict_preflight" == "1" ]]; then
    local preflight_dir="$out_dir/preflight"
    mkdir -p "$preflight_dir"
    if [[ ! -f "$preflight_dir/benchmark.summary.json" ]]; then
      run_cmd \
        "$log_file" \
        "$GTSM" benchmark-generate \
          --corpus-jsonl "$CORPUS_JSONL" \
          --limit 1 \
          --provider local \
          --model-variant "$variant" \
          --temperature 0 \
          --max-tokens "$MAX_TOKENS" \
          --max-prompt-help-chars "$MAX_PROMPT_HELP_CHARS" \
          --repair-invalid-xml \
          --num-processes "$NUM_PROCESSES" \
          --gpu-devices "$GPU_DEVICES" \
          --min-items-per-process 1 \
          --local-gpu-topology "$BENCHMARK_GPU_TOPOLOGY" \
          --local-offload-policy "$BENCHMARK_OFFLOAD_POLICY" \
          --local-gpu-memory-reserve-gib "$BENCHMARK_GPU_MEMORY_RESERVE_GIB" \
          --record-timeout-seconds "$BENCHMARK_RECORD_TIMEOUT_SECONDS" \
          --checkpoint-records "$preflight_dir/checkpoint.records.jsonl" \
          --resume-existing \
          --wrappers-dir "$preflight_dir/wrappers" \
          --generation-records "$preflight_dir/generation.records.json" \
          --evaluation-report "$preflight_dir/evaluation.summary.json" \
          --benchmark-summary "$preflight_dir/benchmark.summary.json" \
          --status-log "$preflight_dir/status.jsonl"
    fi
  fi

  run_cmd "$log_file" "$GTSM" "${benchmark_args[@]}"

  if is_dry_run; then
    return 0
  fi

  append_summary ""
  append_summary "## Benchmark: $label"
  "$GTSM" benchmark-summary --summary "$out_dir/benchmark.summary.json" | tee -a "$RUN_LOG" "$SUMMARY_MD"
}

benchmark_succeeded_count() {
  local summary_path="$1"
  json_read "$summary_path" '.succeeded' "0"
}

append_training_data_diagnostics() {
  local metrics_path="$1"
  [[ -f "$metrics_path" ]] || return 0
  append_summary ""
  append_summary "## Training data diagnostics"
  jq '{
    samples,
    total_corpus_records: .training_data_diagnostics.total_corpus_records,
    trainable_samples: .training_data_diagnostics.trainable_samples,
    missing_xml_path_count: .training_data_diagnostics.missing_xml_path_count,
    missing_xml_target_count: .training_data_diagnostics.missing_xml_target_count,
    empty_xml_target_count: .training_data_diagnostics.empty_xml_target_count,
    target_source_counts: .training_data_diagnostics.target_source_counts,
    missing_xml_target_examples: .training_data_diagnostics.missing_xml_target_examples
  }' "$metrics_path" | tee -a "$RUN_LOG" "$SUMMARY_MD"
}

run_qwen_dry_run_command() {
  local run_id="$1"
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  export PYTORCH_CUDA_ALLOC_CONF
  run_cmd \
    "$LOG_DIR/qwen-dry-run.log" \
    "$GTSM" train \
      --profile "$QWEN_PROFILE" \
      --dataset-manifest "$DATASET_MANIFEST" \
      --corpus-jsonl "$CORPUS_JSONL" \
      --variant-id "$QWEN_DRYRUN_VARIANT" \
      --backend axolotl \
      --num-processes "$NUM_PROCESSES" \
      --distributed-strategy "$QWEN_DISTRIBUTED_STRATEGY" \
      --max-seq-length 8192 \
      --per-device-batch-size 2 \
      --gradient-accumulation-steps 1 \
      --dry-run-backend \
      --internal-run-id "$run_id"
}

rebuild_corpus_for_low_samples() {
  log "Rebuilding corpus because trainable samples are below MIN_TRAIN_SAMPLES=$MIN_TRAIN_SAMPLES."
  append_summary ""
  append_summary "## Corpus rebuild"
  append_summary "- Trigger: low trainable sample count"
  append_summary "- Started: $(timestamp)"
  local extract_args=(
    extract-corpus
    --restart
    --max-workers "$EXTRACT_MAX_WORKERS"
    --output "$CORPUS_JSONL"
    --checkpoint "$CORPUS_CHECKPOINT"
  )
  if is_true "$NO_FETCH_DOCS"; then
    extract_args+=(--no-fetch-docs)
  fi
  extract_args+=(
    --resolve-containers
    --execute-containers
    --container-runtime "$CONTAINER_RUNTIME"
    --container-cache-dir "$CONTAINER_CACHE_DIR"
    --container-help-probe-mode "$CONTAINER_HELP_PROBE_MODE"
    --status-log "$LOG_DIR/extract-corpus.status.jsonl"
    --bioconda-checkout-sources
    --bioconda-ref "$BIOCONDA_REF"
  )
  if is_true "$DOCKER_USE_SUDO"; then
    extract_args+=(--docker-use-sudo)
  fi
  run_cmd \
    "$LOG_DIR/extract-corpus.log" \
    "$GTSM" "${extract_args[@]}"
  append_summary "- Finished: $(timestamp)"
}

preflight() {
  save_state
  append_summary "# GTSM Overnight Run $RUN_TAG"
  append_summary ""
  append_summary "- Started: $(timestamp)"
  append_summary "- Corpus: $CORPUS_JSONL"
  append_summary "- GPUs: $GPU_DEVICES"

  if is_dry_run; then
    run_cmd "$LOG_DIR/preflight.log" "$GTSM" benchmark-summary --help
    run_cmd "$LOG_DIR/preflight.log" "$GTSM" promote-candidate --help
    run_cmd "$LOG_DIR/preflight.log" "$GTSM" model-cache-info
    run_cmd "$LOG_DIR/preflight.log" "$GTSM" runtime-detect
    run_cmd "$LOG_DIR/preflight.log" python -m pip check
    run_cmd "$LOG_DIR/preflight.log" axolotl --version
    run_cmd "$LOG_DIR/preflight.log" "$GTSM" serve-stop --all-ports --force
    run_cmd "$LOG_DIR/preflight.log" nvidia-smi
    return 0
  fi

  command -v "$GTSM" >/dev/null || die "gtsm command not found: $GTSM"
  command -v jq >/dev/null || die "jq is required."
  command -v python >/dev/null || die "python is required."
  command -v axolotl >/dev/null || die "axolotl is required."
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required."

  local gpu_count
  gpu_count="$(csv_count "$GPU_DEVICES")"
  [[ "$gpu_count" -ge 4 ]] || die "Expected at least 4 configured GPU ids, got $GPU_DEVICES."

  [[ -f "$CORPUS_JSONL" ]] || die "Corpus JSONL not found: $CORPUS_JSONL"
  local corpus_records
  corpus_records="$(wc -l < "$CORPUS_JSONL" | tr -d ' ')"
  [[ "$corpus_records" -ge "$MIN_TRAIN_SAMPLES" ]] || die "Corpus has $corpus_records records, below MIN_TRAIN_SAMPLES=$MIN_TRAIN_SAMPLES."

  run_cmd "$LOG_DIR/preflight.log" "$GTSM" benchmark-summary --help
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" promote-candidate --help
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" model-cache-info
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" runtime-detect
  "$GTSM" runtime-detect | tee -a "$RUN_LOG" "$LOG_DIR/preflight.log" | jq -e '.cuda_available == true' >/dev/null || die "CUDA is not available according to gtsm runtime-detect."
  run_cmd "$LOG_DIR/preflight.log" python -m pip check
  run_cmd "$LOG_DIR/preflight.log" axolotl --version
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" serve-stop --all-ports --force
  run_cmd "$LOG_DIR/preflight.log" nvidia-smi
  append_summary "- Corpus records: $corpus_records"
}

qwen_dry_run() {
  local run_id="train-$RUN_TAG-qwen7b-dryrun"
  local metrics_path
  metrics_path="$(training_metrics_path "$run_id")"
  if ! is_dry_run && [[ -f "$metrics_path" ]]; then
    local existing_samples
    existing_samples="$(json_read "$metrics_path" '.samples' "0")"
    if [[ "$existing_samples" -ge "$MIN_TRAIN_SAMPLES" ]]; then
      log "Using existing Qwen dry-run metrics: $metrics_path"
      append_summary "- Dry-run trainable samples: $existing_samples"
      append_training_data_diagnostics "$metrics_path"
      return 0
    fi
    log "Existing Qwen dry-run metrics are low ($existing_samples samples); rerunning dry-run."
  fi

  run_qwen_dry_run_command "$run_id"

  if is_dry_run; then
    return 0
  fi

  local samples
  samples="$(json_read "$metrics_path" '.samples' "0")"
  append_training_data_diagnostics "$metrics_path"
  if [[ "$samples" -lt "$MIN_TRAIN_SAMPLES" ]] && is_true "$REBUILD_CORPUS_ON_LOW_SAMPLES"; then
    rebuild_corpus_for_low_samples
    run_qwen_dry_run_command "$run_id"
    samples="$(json_read "$metrics_path" '.samples' "0")"
    append_training_data_diagnostics "$metrics_path"
  fi
  [[ "$samples" -ge "$MIN_TRAIN_SAMPLES" ]] || die "Dry-run produced $samples samples, below MIN_TRAIN_SAMPLES=$MIN_TRAIN_SAMPLES."
  append_summary "- Dry-run trainable samples: $samples"
}

qwen_train() {
  local primary_run_id="train-$RUN_TAG-qwen7b"
  local safe_run_id="train-$RUN_TAG-qwen7b-safe"
  local primary_log="$LOG_DIR/qwen-train.log"
  local safe_log="$LOG_DIR/qwen-train-safe.log"

  if ! is_dry_run; then
    if [[ -n "$ACTIVE_QWEN_VARIANT" && -f "$MODEL_VARIANTS_DIR/$ACTIVE_QWEN_VARIANT.manifest.json" ]]; then
      log "Using existing active Qwen variant: $ACTIVE_QWEN_VARIANT"
      append_summary "- Qwen 7B variant: $ACTIVE_QWEN_VARIANT"
      return 0
    fi
    if [[ "$(training_status "$primary_run_id")" == "completed" && -f "$MODEL_VARIANTS_DIR/$QWEN_VARIANT.manifest.json" ]]; then
      ACTIVE_QWEN_VARIANT="$QWEN_VARIANT"
      save_state
      log "Recovered completed Qwen primary run: $ACTIVE_QWEN_VARIANT"
      append_summary "- Qwen 7B variant: $ACTIVE_QWEN_VARIANT"
      return 0
    fi
    if [[ "$(training_status "$safe_run_id")" == "completed" && -f "$MODEL_VARIANTS_DIR/$QWEN_SAFE_VARIANT.manifest.json" ]]; then
      ACTIVE_QWEN_VARIANT="$QWEN_SAFE_VARIANT"
      save_state
      log "Recovered completed Qwen safe run: $ACTIVE_QWEN_VARIANT"
      append_summary "- Qwen 7B variant: $ACTIVE_QWEN_VARIANT"
      return 0
    fi
  fi

  if train_once "$QWEN_PROFILE" "$QWEN_VARIANT" "$primary_run_id" 8192 2 1 "$LOG_DIR/qwen-train.status.jsonl" "$primary_log" "$QWEN_DISTRIBUTED_STRATEGY"; then
    ACTIVE_QWEN_VARIANT="$QWEN_VARIANT"
    save_state
    append_summary "- Qwen 7B variant: $ACTIVE_QWEN_VARIANT"
    return 0
  fi

  if failure_looks_like_oom "$primary_run_id" "$primary_log"; then
    log "Primary Qwen training appears to have hit OOM; retrying with safe settings."
    if train_once "$QWEN_PROFILE" "$QWEN_SAFE_VARIANT" "$safe_run_id" 4096 1 2 "$LOG_DIR/qwen-train-safe.status.jsonl" "$safe_log" "$QWEN_DISTRIBUTED_STRATEGY"; then
      ACTIVE_QWEN_VARIANT="$QWEN_SAFE_VARIANT"
      save_state
      append_summary "- Qwen 7B variant: $ACTIVE_QWEN_VARIANT"
      append_summary "- Qwen 7B used safe memory settings after OOM."
      return 0
    fi
  fi

  die "Qwen 7B training failed. Inspect $primary_log and $safe_log."
}

qwen_smoke() {
  [[ -n "$ACTIVE_QWEN_VARIANT" ]] || die "ACTIVE_QWEN_VARIANT is not set."
  local out_dir="$BENCHMARK_DIR/qwen7b/smoke"
  benchmark_once "qwen7b smoke" "$ACTIVE_QWEN_VARIANT" "$SMOKE_LIMIT" "$out_dir" "$LOG_DIR/qwen-smoke.status.jsonl" "$LOG_DIR/qwen-smoke.log"
  if is_dry_run; then
    return 0
  fi
  local succeeded
  succeeded="$(benchmark_succeeded_count "$out_dir/benchmark.summary.json")"
  [[ "$succeeded" -ge "$SMOKE_MIN_SUCCEEDED" ]] || die "Qwen smoke benchmark succeeded $succeeded/$SMOKE_LIMIT, below SMOKE_MIN_SUCCEEDED=$SMOKE_MIN_SUCCEEDED."
}

qwen_candidate() {
  [[ -n "$ACTIVE_QWEN_VARIANT" ]] || die "ACTIVE_QWEN_VARIANT is not set."
  benchmark_once "qwen7b candidate" "$ACTIVE_QWEN_VARIANT" "$CANDIDATE_LIMIT" "$BENCHMARK_DIR/qwen7b/candidate" "$LOG_DIR/qwen-candidate.status.jsonl" "$LOG_DIR/qwen-candidate.log"
}

qwen_baseline() {
  benchmark_once "qwen7b baseline" "$BASELINE_VARIANT" "$CANDIDATE_LIMIT" "$BENCHMARK_DIR/qwen7b/baseline" "$LOG_DIR/qwen-baseline.status.jsonl" "$LOG_DIR/qwen-baseline.log"
}

qwen_promotion() {
  local candidate="$BENCHMARK_DIR/qwen7b/candidate/benchmark.summary.json"
  local baseline="$BENCHMARK_DIR/qwen7b/baseline/benchmark.summary.json"
  local decision="$BENCHMARK_DIR/qwen7b/promotion.decision.json"
  if ! is_dry_run && [[ -f "$decision" ]]; then
    log "Using existing Qwen promotion decision: $decision"
    append_summary ""
    append_summary "## Qwen promotion"
    jq '{promote,metrics,reasons}' "$decision" | tee -a "$RUN_LOG" "$SUMMARY_MD"
    return 0
  fi

  run_json_cmd \
    "$decision.stdout.json" \
    "$LOG_DIR/qwen-promotion.log" \
    "$GTSM" promote-candidate \
      --candidate-summary "$candidate" \
      --baseline-summary "$baseline" \
      --decision-out "$decision" \
      --policy development
  if [[ -f "$decision" ]]; then
    append_summary ""
    append_summary "## Qwen promotion"
    jq '{promote,metrics,reasons}' "$decision" | tee -a "$RUN_LOG" "$SUMMARY_MD"
  fi
}

export_variant() {
  local label="$1"
  local variant="$2"
  local output_json="$3"
  local log_file="$4"
  [[ -n "$variant" ]] || die "No variant id set for export: $label"
  if ! is_dry_run && [[ -f "$output_json" ]]; then
    log "Using existing export result for $label: $output_json"
    append_summary ""
    append_summary "## Export: $label"
    jq '{variant_id,status,merged_path,gguf_path,gguf_paths,notes,ollama_modelfile_path,ollama_model_name}' "$output_json" | tee -a "$RUN_LOG" "$SUMMARY_MD"
    return 0
  fi

  run_json_cmd \
    "$output_json" \
    "$log_file" \
    "$GTSM" export-model \
      --variant-id "$variant" \
      --format all \
      --quantizations "$EXPORT_QUANTIZATIONS"
  append_summary ""
  append_summary "## Export: $label"
  if [[ -f "$output_json" ]]; then
    jq '{variant_id,status,merged_path,gguf_path,gguf_paths,notes,ollama_modelfile_path,ollama_model_name}' "$output_json" | tee -a "$RUN_LOG" "$SUMMARY_MD"
  fi
}

qwen_export() {
  if [[ ! -f "$BENCHMARK_DIR/qwen7b/candidate/benchmark.summary.json" ]] && ! is_true "$EXPORT_ON_BENCHMARK_FAILURE"; then
    die "Candidate benchmark missing and EXPORT_ON_BENCHMARK_FAILURE is disabled."
  fi
  export_variant "qwen7b" "$ACTIVE_QWEN_VARIANT" "$EXPORT_DIR/qwen7b.export.json" "$LOG_DIR/qwen-export.log"
}

stretch_allowed() {
  if is_dry_run; then
    is_true "$STRETCH_ON_SUCCESS"
    return $?
  fi
  is_true "$STRETCH_ON_SUCCESS" || return 1
  [[ -f "$(marker_path qwen_export)" ]] || return 1
  return 0
}

stretch_train() {
  if ! stretch_allowed; then
    log "Stretch training disabled or Qwen export incomplete; skipping."
    return 0
  fi

  local primary_run_id="train-$RUN_TAG-devstral24b"
  local safe_run_id="train-$RUN_TAG-devstral24b-safe"
  local deepspeed_run_id="train-$RUN_TAG-devstral24b-zero3"
  local deepspeed_offload_run_id="train-$RUN_TAG-devstral24b-zero3-offload"
  local fourbit_run_id="train-$RUN_TAG-devstral24b-4bit"
  local primary_log="$LOG_DIR/stretch-train.log"
  local safe_log="$LOG_DIR/stretch-train-safe.log"
  local deepspeed_log="$LOG_DIR/stretch-train-zero3.log"
  local deepspeed_offload_log="$LOG_DIR/stretch-train-zero3-offload.log"
  local fourbit_log="$LOG_DIR/stretch-train-4bit.log"

  if ! is_dry_run; then
    if [[ -n "$ACTIVE_STRETCH_VARIANT" && -f "$MODEL_VARIANTS_DIR/$ACTIVE_STRETCH_VARIANT.manifest.json" ]]; then
      log "Using existing active stretch variant: $ACTIVE_STRETCH_VARIANT"
      append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
      return 0
    fi
    if [[ "$(training_status "$primary_run_id")" == "completed" && -f "$MODEL_VARIANTS_DIR/$STRETCH_VARIANT.manifest.json" ]]; then
      ACTIVE_STRETCH_VARIANT="$STRETCH_VARIANT"
      save_state
      log "Recovered completed stretch primary run: $ACTIVE_STRETCH_VARIANT"
      append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
      return 0
    fi
    if [[ "$(training_status "$safe_run_id")" == "completed" && -f "$MODEL_VARIANTS_DIR/$STRETCH_SAFE_VARIANT.manifest.json" ]]; then
      ACTIVE_STRETCH_VARIANT="$STRETCH_SAFE_VARIANT"
      save_state
      log "Recovered completed stretch safe run: $ACTIVE_STRETCH_VARIANT"
      append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
      return 0
    fi
    if [[ "$(training_status "$deepspeed_run_id")" == "completed" && -f "$MODEL_VARIANTS_DIR/$STRETCH_DEEPSPEED_VARIANT.manifest.json" ]]; then
      ACTIVE_STRETCH_VARIANT="$STRETCH_DEEPSPEED_VARIANT"
      save_state
      log "Recovered completed stretch DeepSpeed ZeRO-3 run: $ACTIVE_STRETCH_VARIANT"
      append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
      return 0
    fi
    if [[ "$(training_status "$deepspeed_offload_run_id")" == "completed" && -f "$MODEL_VARIANTS_DIR/$STRETCH_DEEPSPEED_OFFLOAD_VARIANT.manifest.json" ]]; then
      ACTIVE_STRETCH_VARIANT="$STRETCH_DEEPSPEED_OFFLOAD_VARIANT"
      save_state
      log "Recovered completed stretch DeepSpeed ZeRO-3 offload run: $ACTIVE_STRETCH_VARIANT"
      append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
      return 0
    fi
    if [[ "$(training_status "$fourbit_run_id")" == "completed" && -f "$MODEL_VARIANTS_DIR/$STRETCH_4BIT_VARIANT.manifest.json" ]]; then
      ACTIVE_STRETCH_VARIANT="$STRETCH_4BIT_VARIANT"
      save_state
      log "Recovered completed stretch 4-bit fallback run: $ACTIVE_STRETCH_VARIANT"
      append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
      return 0
    fi
  fi

  if train_once "$STRETCH_PROFILE" "$STRETCH_VARIANT" "$primary_run_id" 8192 1 2 "$LOG_DIR/stretch-train.status.jsonl" "$primary_log" "$STRETCH_DISTRIBUTED_STRATEGY"; then
    ACTIVE_STRETCH_VARIANT="$STRETCH_VARIANT"
    save_state
    append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
    return 0
  fi

  if failure_looks_like_oom "$primary_run_id" "$primary_log"; then
    log "Primary stretch training appears to have hit OOM; retrying $STRETCH_DISTRIBUTED_STRATEGY with safe settings."
    if train_once "$STRETCH_PROFILE" "$STRETCH_SAFE_VARIANT" "$safe_run_id" 4096 1 4 "$LOG_DIR/stretch-train-safe.status.jsonl" "$safe_log" "$STRETCH_DISTRIBUTED_STRATEGY"; then
      ACTIVE_STRETCH_VARIANT="$STRETCH_SAFE_VARIANT"
      save_state
      append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
      append_summary "- Stretch used $STRETCH_DISTRIBUTED_STRATEGY safe memory settings after OOM."
      return 0
    fi

    if is_true "$STRETCH_DEEPSPEED_FALLBACK"; then
      log "Stretch $STRETCH_DISTRIBUTED_STRATEGY safe run failed; retrying with DeepSpeed ZeRO-3."
      if train_once "$STRETCH_PROFILE" "$STRETCH_DEEPSPEED_VARIANT" "$deepspeed_run_id" 4096 1 4 "$LOG_DIR/stretch-train-zero3.status.jsonl" "$deepspeed_log" "deepspeed-zero3"; then
        ACTIVE_STRETCH_VARIANT="$STRETCH_DEEPSPEED_VARIANT"
        save_state
        append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
        append_summary "- Stretch used DeepSpeed ZeRO-3 after FSDP/primary OOM."
        return 0
      fi
    fi

    if is_true "$STRETCH_DEEPSPEED_OFFLOAD_FALLBACK" && failure_looks_like_oom "$deepspeed_run_id" "$deepspeed_log"; then
      log "Stretch DeepSpeed ZeRO-3 appears to have hit OOM; retrying with ZeRO-3 CPU offload."
      if train_once "$STRETCH_PROFILE" "$STRETCH_DEEPSPEED_OFFLOAD_VARIANT" "$deepspeed_offload_run_id" 4096 1 4 "$LOG_DIR/stretch-train-zero3-offload.status.jsonl" "$deepspeed_offload_log" "deepspeed-zero3-offload"; then
        ACTIVE_STRETCH_VARIANT="$STRETCH_DEEPSPEED_OFFLOAD_VARIANT"
        save_state
        append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
        append_summary "- Stretch used DeepSpeed ZeRO-3 CPU offload after GPU-only ZeRO-3 OOM."
        return 0
      fi
    fi

    if is_true "$STRETCH_4BIT_FALLBACK"; then
      log "Full-precision stretch training failed; retrying explicit 4-bit stretch fallback."
      if train_once "$STRETCH_4BIT_PROFILE" "$STRETCH_4BIT_VARIANT" "$fourbit_run_id" 4096 1 4 "$LOG_DIR/stretch-train-4bit.status.jsonl" "$fourbit_log" "ddp"; then
        ACTIVE_STRETCH_VARIANT="$STRETCH_4BIT_VARIANT"
        save_state
        append_summary "- Stretch variant: $ACTIVE_STRETCH_VARIANT"
        append_summary "- Stretch used explicit 4-bit fallback."
        return 0
      fi
    fi
  fi

  die "Stretch training failed. Inspect $primary_log, $safe_log, $deepspeed_log, $deepspeed_offload_log, and $fourbit_log."
}

stretch_smoke() {
  stretch_allowed || return 0
  [[ -n "$ACTIVE_STRETCH_VARIANT" ]] || die "ACTIVE_STRETCH_VARIANT is not set."
  local out_dir="$BENCHMARK_DIR/stretch/smoke"
  benchmark_once "stretch smoke" "$ACTIVE_STRETCH_VARIANT" "$SMOKE_LIMIT" "$out_dir" "$LOG_DIR/stretch-smoke.status.jsonl" "$LOG_DIR/stretch-smoke.log"
  if is_dry_run; then
    return 0
  fi
  local succeeded
  succeeded="$(benchmark_succeeded_count "$out_dir/benchmark.summary.json")"
  [[ "$succeeded" -ge "$SMOKE_MIN_SUCCEEDED" ]] || die "Stretch smoke benchmark succeeded $succeeded/$SMOKE_LIMIT, below SMOKE_MIN_SUCCEEDED=$SMOKE_MIN_SUCCEEDED."
}

stretch_candidate() {
  stretch_allowed || return 0
  [[ -n "$ACTIVE_STRETCH_VARIANT" ]] || die "ACTIVE_STRETCH_VARIANT is not set."
  benchmark_once "stretch candidate" "$ACTIVE_STRETCH_VARIANT" "$CANDIDATE_LIMIT" "$BENCHMARK_DIR/stretch/candidate" "$LOG_DIR/stretch-candidate.status.jsonl" "$LOG_DIR/stretch-candidate.log"
}

stretch_promotion() {
  stretch_allowed || return 0
  local candidate="$BENCHMARK_DIR/stretch/candidate/benchmark.summary.json"
  local baseline="$BENCHMARK_DIR/qwen7b/candidate/benchmark.summary.json"
  local decision="$BENCHMARK_DIR/stretch/promotion.decision.json"
  if ! is_dry_run && [[ -f "$decision" ]]; then
    log "Using existing stretch promotion decision: $decision"
    append_summary ""
    append_summary "## Stretch promotion"
    jq '{promote,metrics,reasons}' "$decision" | tee -a "$RUN_LOG" "$SUMMARY_MD"
    return 0
  fi

  run_json_cmd \
    "$decision.stdout.json" \
    "$LOG_DIR/stretch-promotion.log" \
    "$GTSM" promote-candidate \
      --candidate-summary "$candidate" \
      --baseline-summary "$baseline" \
      --decision-out "$decision" \
      --policy development
  if [[ -f "$decision" ]]; then
    append_summary ""
    append_summary "## Stretch promotion"
    jq '{promote,metrics,reasons}' "$decision" | tee -a "$RUN_LOG" "$SUMMARY_MD"
  fi
}

stretch_export() {
  stretch_allowed || return 0
  export_variant "stretch" "$ACTIVE_STRETCH_VARIANT" "$EXPORT_DIR/stretch.export.json" "$LOG_DIR/stretch-export.log"
}

stop_stretch_candidate() {
  local variant="${ACTIVE_STRETCH_VARIANT:-$STRETCH_VARIANT}"
  log "Stopping stretch candidate processes for variant=$variant"
  if pgrep -a -u "$USER" -f "benchmark-generate .*${variant}.*stretch/candidate" >/dev/null 2>&1; then
    pkill -TERM -u "$USER" -f "benchmark-generate .*${variant}.*stretch/candidate" || true
  fi
  if pgrep -a -u "$USER" -f "scripts/gtsm_overnight_4xa100.sh resume" >/dev/null 2>&1; then
    pkill -TERM -u "$USER" -f "scripts/gtsm_overnight_4xa100.sh resume" || true
  fi
  sleep 5
  if pgrep -a -u "$USER" -f "benchmark-generate .*${variant}.*stretch/candidate" >/dev/null 2>&1; then
    log "Stretch candidate processes still running after TERM; sending KILL."
    pkill -KILL -u "$USER" -f "benchmark-generate .*${variant}.*stretch/candidate" || true
  fi
  log "Stop request complete."
}

print_status() {
  echo "run_tag=$RUN_TAG"
  echo "run_root=$RUN_ROOT"
  echo "dry_run=$DRY_RUN"
  echo
  if [[ -f "$STATE_ENV" ]]; then
    echo "state:"
    sed -n '1,120p' "$STATE_ENV"
    echo
  fi
  echo "steps:"
  local step
  local next_step=""
  for step in "${STEPS[@]}"; do
    if step_done "$step"; then
      printf '  [x] %s\n' "$step"
    else
      printf '  [ ] %s\n' "$step"
      [[ -z "$next_step" ]] && next_step="$step"
    fi
  done
  echo
  echo "next_step=${next_step:-complete}"
  echo
  for summary in \
    "$BENCHMARK_DIR/qwen7b/smoke/benchmark.summary.json" \
    "$BENCHMARK_DIR/qwen7b/candidate/benchmark.summary.json" \
    "$BENCHMARK_DIR/qwen7b/baseline/benchmark.summary.json" \
    "$BENCHMARK_DIR/stretch/smoke/benchmark.summary.json" \
    "$BENCHMARK_DIR/stretch/candidate/benchmark.summary.json"; do
    if [[ -f "$summary" ]]; then
      echo "summary: $summary"
      if command -v "$GTSM" >/dev/null 2>&1; then
        "$GTSM" benchmark-summary --summary "$summary" || true
      else
        echo "warning: GTSM command not found on PATH: $GTSM"
        jq '{attempted,succeeded,failed,quality:{validity:.quality.validity,throughput:.quality.throughput},startup}' "$summary" || true
      fi
      echo
    fi
  done
  if [[ -f "$SUMMARY_MD" ]]; then
    echo "summary_md=$SUMMARY_MD"
  fi
  if [[ -f "$RUN_LOG" ]]; then
    echo
    echo "run_log_tail:"
    tail -40 "$RUN_LOG"
  fi
}

if [[ "$MODE" == "status" ]]; then
  print_status
  exit 0
fi

if [[ "$MODE" == "stop-stretch-candidate" ]]; then
  stop_stretch_candidate
  exit 0
fi

save_state

run_step preflight preflight "environment, cache, corpus, and GPU checks"
run_step qwen_dry_run qwen_dry_run "verify full trainable sample count"
run_step qwen_train qwen_train "train full-corpus Qwen2.5-Coder-7B"
run_step qwen_smoke qwen_smoke "run 5-record Qwen smoke benchmark"
run_step qwen_candidate qwen_candidate "run Qwen candidate benchmark"
run_step qwen_baseline qwen_baseline "run old smoke baseline benchmark"
run_step qwen_promotion qwen_promotion "compare Qwen candidate to baseline"
run_step qwen_export qwen_export "export Qwen candidate artifacts"
run_step stretch_train stretch_train "train stretch model without a wall-clock timeout"
run_step stretch_smoke stretch_smoke "run stretch smoke benchmark"
run_step stretch_candidate stretch_candidate "run stretch candidate benchmark"
run_step stretch_promotion stretch_promotion "compare stretch candidate to Qwen candidate"
run_step stretch_export stretch_export "export stretch artifacts"

append_summary ""
append_summary "- Finished: $(timestamp)"
log "Overnight workflow complete."
