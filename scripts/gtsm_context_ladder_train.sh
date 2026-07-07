#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-run}"
case "$MODE" in
  launch|run|estimate|probe|train|status|tail|clean|classify-contamination) ;;
  *)
    echo "Usage: $0 {launch|run|estimate|probe|train|status|tail|clean|classify-contamination}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && /bin/pwd -P)"
cd "$REPO_ROOT"
SCRIPT_PATH="$REPO_ROOT/scripts/gtsm_context_ladder_train.sh"

CONTEXT_LADDER_ROOT="${CONTEXT_LADDER_ROOT:-.gtsm-cache/runs/context-ladder}"
CURRENT_FILE="$CONTEXT_LADDER_ROOT/current"

if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ "$MODE" == "launch" || "$MODE" == "run" || "$MODE" == "estimate" || "$MODE" == "classify-contamination" ]]; then
    RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
  elif [[ -f "$CURRENT_FILE" ]]; then
    RUN_TAG="$(cat "$CURRENT_FILE")"
  else
    echo "RUN_TAG is not set and no current context-ladder run exists." >&2
    exit 2
  fi
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must contain only letters, numbers, '.', '_', and '-'." >&2
  exit 2
fi

: "${GTSM:=.conda/gtsm/bin/gtsm}"
: "${PYTHON:=.conda/gtsm/bin/python}"
: "${PROFILE:=agentic-devstral-24b}"
: "${ARTIFACT_FORMAT:=mixed}"
: "${CORPUS_JSONL:=.gtsm-cache/datasets/tools-iuc-corpus.jsonl}"
: "${DATASET_MANIFEST:=config/dataset.manifest.json}"
: "${SOURCE_MODES:=all-raw,all-filtered}"
: "${SOURCE_MODE_PREFERENCE:=all-raw,all-filtered}"
: "${TEST_CONTEXT_MODE:=fixtures}"
: "${TEST_CONTEXT_MAX_CHARS:=4000}"
: "${TEST_CONTEXT_MAX_FILES:=6}"
: "${TEST_CONTEXT_MAX_FILE_BYTES:=64KB}"
: "${DISTRIBUTED_STRATEGIES:=deepspeed-zero3,fsdp}"
: "${CONTEXT_LADDER:=128k,96k,64k,48k,32k,24k,16k,12k,8k,4k,2k}"
: "${NUM_PROCESSES:=auto}"
: "${GPU_DEVICES:=auto}"
: "${GPU_PREFLIGHT:=1}"
: "${GPU_IDLE_MAX_MIB:=1024}"
: "${GPU_IDLE_MAX_UTIL:=20}"
: "${GPU_IDLE_WAIT_SECONDS:=300}"
: "${GPU_REQUIRE_NO_COMPUTE_APPS:=1}"
: "${MIN_CORPUS_RECORDS:=1}"
: "${PROBE_MAX_STEPS:=5}"
: "${PROBE_ONLY:=0}"
: "${ESTIMATE_ONLY:=0}"
: "${EXACT_TOKENIZER:=1}"
: "${EXACT_TOKENIZER_REQUIRED:=0}"
: "${ESTIMATE_PROGRESS_INTERVAL:=100}"
: "${ESTIMATE_LIMIT:=0}"
: "${ESTIMATE_WORKERS:=0}"
: "${ESTIMATE_LONGEST_SAMPLE_COUNT:=50}"
: "${CANDIDATE_MAX_OVERFLOW_FRACTION:=1.0}"
: "${CANDIDATE_MAX_OVERFLOW_SAMPLES:=1000000000}"
: "${TRAIN_BATCH_SIZE:=1}"
: "${TRAIN_GRAD_ACCUM:=2}"
: "${TRAIN_DATA_WORKERS:=8}"
: "${ATTN_IMPLEMENTATION:=xformers}"
: "${PAD_TO_SEQUENCE_LEN:=0}"
: "${STATUS_INTERVAL_SECONDS:=60}"
: "${LOG_TAIL_LINES:=25}"
: "${POST_EXPORT_QUANTIZATIONS:=q4_k_m}"
: "${POST_OLLAMA_CREATE:=0}"
: "${POST_EXPORT_ENV_DIR:=${GGUF_EXPORT_ENV_DIR:-}}"
: "${POST_EXPORT_LLAMA_CPP_DIR:=${LLAMA_CPP_DIR:-.gtsm-cache/llama.cpp}}"
: "${POST_EXPORT_GGUF_OUTTYPE:=${GGUF_OUTTYPE:-bf16}}"
: "${POST_EXPORT_PREPARE:=0}"
: "${POST_EXPORT_SYNC_OVERNIGHT:=0}"
: "${POST_OLLAMA_FROM_QUANTIZATION:=q4_k_m}"
: "${LAUNCH_BACKEND:=auto}"
: "${TRAINING_METHOD:=}"
: "${LEARNING_RATE:=}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
: "${CLEANUP_ORPHAN_TRAINING:=1}"
: "${SKIP_PREVIOUS_FAILED_PROBES:=1}"
: "${DRY_RUN:=0}"

export GTSM_TRAIN_DATA_WORKERS="$TRAIN_DATA_WORKERS"

RUN_ROOT="$CONTEXT_LADDER_ROOT/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
MARKER_DIR="$RUN_ROOT/markers"
RUN_LOG="$RUN_ROOT/run.log"
SUMMARY_MD="$RUN_ROOT/summary.md"
STATE_ENV="$RUN_ROOT/state.env"
PID_FILE="$RUN_ROOT/pipeline.pid"
TMUX_SESSION_FILE="$RUN_ROOT/tmux.session"
DETACH_LOG="$RUN_ROOT/detach.log"
STARTED_FILE="$RUN_ROOT/started_at.epoch"
APPROX_ESTIMATE_JSON="$RUN_ROOT/estimate.approx.json"
EXACT_ESTIMATE_JSON="$RUN_ROOT/estimate.exact.json"
APPROX_CANDIDATES_TSV="$RUN_ROOT/candidates.approx.tsv"
EXACT_CANDIDATES_TSV="$RUN_ROOT/candidates.exact.tsv"
CANDIDATES_TSV="$RUN_ROOT/candidates.tsv"
PROBE_REPORT_TSV="$RUN_ROOT/probes.tsv"
FULL_REPORT_TSV="$RUN_ROOT/full-training.tsv"
SELECTION_ENV="$RUN_ROOT/selection.env"

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$STATUS_DIR" "$MARKER_DIR"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$RUN_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
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
  if is_true "$DRY_RUN"; then
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
  if is_true "$DRY_RUN"; then
    printf '{}\n' > "$output_json"
    return 0
  fi
  local stdout_tmp="$output_json.stdout.tmp"
  local stderr_tmp="$output_json.stderr.tmp"
  set +e
  "$@" > "$stdout_tmp" 2> "$stderr_tmp"
  local rc="$?"
  set -e
  cat "$stderr_tmp" | tee -a "$RUN_LOG" "$log_file" >&2
  if [[ "$rc" -eq 0 ]]; then
    cp "$stdout_tmp" "$output_json"
  else
    cat "$stdout_tmp" >> "$log_file" || true
  fi
  rm -f "$stdout_tmp" "$stderr_tmp"
  return "$rc"
}

save_state() {
  {
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'PROFILE=%q\n' "$PROFILE"
    printf 'ARTIFACT_FORMAT=%q\n' "$ARTIFACT_FORMAT"
    printf 'CORPUS_JSONL=%q\n' "$CORPUS_JSONL"
    printf 'DATASET_MANIFEST=%q\n' "$DATASET_MANIFEST"
    printf 'SOURCE_MODES=%q\n' "$SOURCE_MODES"
    printf 'TEST_CONTEXT_MODE=%q\n' "$TEST_CONTEXT_MODE"
    printf 'TEST_CONTEXT_MAX_CHARS=%q\n' "$TEST_CONTEXT_MAX_CHARS"
    printf 'TEST_CONTEXT_MAX_FILES=%q\n' "$TEST_CONTEXT_MAX_FILES"
    printf 'TEST_CONTEXT_MAX_FILE_BYTES=%q\n' "$TEST_CONTEXT_MAX_FILE_BYTES"
    printf 'DISTRIBUTED_STRATEGIES=%q\n' "$DISTRIBUTED_STRATEGIES"
    printf 'CONTEXT_LADDER=%q\n' "$CONTEXT_LADDER"
    printf 'GPU_DEVICES=%q\n' "$GPU_DEVICES"
    printf 'GPU_PREFLIGHT=%q\n' "$GPU_PREFLIGHT"
    printf 'GPU_REQUIRE_NO_COMPUTE_APPS=%q\n' "$GPU_REQUIRE_NO_COMPUTE_APPS"
    printf 'NUM_PROCESSES=%q\n' "$NUM_PROCESSES"
    printf 'PROBE_MAX_STEPS=%q\n' "$PROBE_MAX_STEPS"
    printf 'ESTIMATE_LIMIT=%q\n' "$ESTIMATE_LIMIT"
    printf 'ESTIMATE_WORKERS=%q\n' "$ESTIMATE_WORKERS"
    printf 'ESTIMATE_LONGEST_SAMPLE_COUNT=%q\n' "$ESTIMATE_LONGEST_SAMPLE_COUNT"
    printf 'CANDIDATE_MAX_OVERFLOW_FRACTION=%q\n' "$CANDIDATE_MAX_OVERFLOW_FRACTION"
    printf 'CANDIDATE_MAX_OVERFLOW_SAMPLES=%q\n' "$CANDIDATE_MAX_OVERFLOW_SAMPLES"
    printf 'TRAIN_DATA_WORKERS=%q\n' "$TRAIN_DATA_WORKERS"
    printf 'POST_EXPORT_QUANTIZATIONS=%q\n' "$POST_EXPORT_QUANTIZATIONS"
    printf 'POST_OLLAMA_CREATE=%q\n' "$POST_OLLAMA_CREATE"
    printf 'POST_EXPORT_ENV_DIR=%q\n' "$POST_EXPORT_ENV_DIR"
    printf 'POST_EXPORT_LLAMA_CPP_DIR=%q\n' "$POST_EXPORT_LLAMA_CPP_DIR"
    printf 'POST_EXPORT_GGUF_OUTTYPE=%q\n' "$POST_EXPORT_GGUF_OUTTYPE"
    printf 'POST_EXPORT_PREPARE=%q\n' "$POST_EXPORT_PREPARE"
    printf 'POST_EXPORT_SYNC_OVERNIGHT=%q\n' "$POST_EXPORT_SYNC_OVERNIGHT"
    printf 'POST_OLLAMA_FROM_QUANTIZATION=%q\n' "$POST_OLLAMA_FROM_QUANTIZATION"
    printf 'SKIP_PREVIOUS_FAILED_PROBES=%q\n' "$SKIP_PREVIOUS_FAILED_PROBES"
    printf 'LAUNCH_BACKEND=%q\n' "$LAUNCH_BACKEND"
  } > "$STATE_ENV"
  printf '%s\n' "$RUN_TAG" > "$CURRENT_FILE"
}

append_summary() {
  printf '%s\n' "$*" >> "$SUMMARY_MD"
}

split_csv() {
  local value="$1"
  local IFS=,
  read -r -a SPLIT_CSV_RESULT <<< "$value"
}

nvidia_smi_retry() {
  local output=""
  local attempt
  for attempt in 1 2 3 4 5; do
    if output="$(nvidia-smi "$@" 2>&1)"; then
      printf '%s\n' "$output"
      return 0
    fi
    sleep 3
  done
  printf '%s\n' "$output" >&2
  return 1
}

resolve_gpus() {
  if [[ "$GPU_DEVICES" == "auto" ]]; then
    local gpu_query
    if ! gpu_query="$(nvidia_smi_retry --query-gpu=index --format=csv,noheader,nounits)"; then
      gpu_query="$(nvidia_smi_retry -L | sed -n 's/^GPU \([0-9][0-9]*\):.*/\1/p')" \
        || die "Could not query NVIDIA GPU ids with nvidia-smi."
    fi
    GPU_DEVICES="$(
      printf '%s\n' "$gpu_query" \
        | "$PYTHON" -c 'import sys; print(",".join(line.strip() for line in sys.stdin if line.strip()))'
    )"
    [[ -n "$GPU_DEVICES" ]] || die "No NVIDIA GPU ids found."
  fi
  if [[ "$NUM_PROCESSES" == "auto" ]]; then
    split_csv "$GPU_DEVICES"
    NUM_PROCESSES="${#SPLIT_CSV_RESULT[@]}"
  fi
}

assert_no_training_processes() {
  local matches
  matches="$(
    ps -u "$USER" -o pid=,cmd= \
      | grep -E 'gtsm train|torch\.distributed\.run|axolotl train|accelerate launch|deepspeed' \
      | grep -v grep \
      | grep -v "$SCRIPT_PATH" || true
  )"
  if [[ -n "$matches" ]]; then
    printf '%s\n' "$matches" | tee -a "$RUN_LOG" "$LOG_DIR/preflight.log"
    die "Training-like processes are already running."
  fi
}

gpus_are_idle() {
  local gpu_query
  if ! gpu_query="$(nvidia_smi_retry --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"; then
    local plain
    plain="$(nvidia_smi_retry)" || return 1
    printf '%s\n' "$plain" >&2
    if printf '%s\n' "$plain" | grep -Eq '[1-9][0-9]*MiB[[:space:]]*/'; then
      return 1
    fi
    return 0
  fi
  local ok=0
  while IFS=, read -r index memory util; do
    index="${index//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    if [[ "$memory" -gt "$GPU_IDLE_MAX_MIB" || "$util" -gt "$GPU_IDLE_MAX_UTIL" ]]; then
      printf 'gpu %s busy: memory=%s MiB util=%s%%\n' "$index" "$memory" "$util" >&2
      ok=1
    fi
  done <<< "$gpu_query"
  if is_true "$GPU_REQUIRE_NO_COMPUTE_APPS"; then
    local compute_apps
    if ! compute_apps="$(nvidia_smi_retry --query-compute-apps=pid,gpu_bus_id,used_memory,process_name --format=csv,noheader)"; then
      return 1
    fi
    compute_apps="$(printf '%s\n' "$compute_apps" | sed '/^[[:space:]]*$/d')"
    if [[ -n "$compute_apps" ]]; then
      printf 'gpu compute apps still present:\n%s\n' "$compute_apps" >&2
      ok=1
    fi
  fi
  return "$ok"
}

wait_for_idle_gpus() {
  local deadline=$((SECONDS + GPU_IDLE_WAIT_SECONDS))
  while (( SECONDS <= deadline )); do
    if gpus_are_idle; then
      return 0
    fi
    sleep 10
  done
  return 1
}

latest_status_file() {
  find "$STATUS_DIR" -type f -name '*.jsonl' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d ' ' -f 2-
}

cleanup_training_processes() {
  is_true "$CLEANUP_ORPHAN_TRAINING" || return 0
  local pids
  pids="$(
    ps -u "$USER" -o pid=,ppid=,cmd= \
      | "$PYTHON" -c '
import re
import sys

self_pid = int(sys.argv[1])
patterns = (
    re.compile(r"(^|[/\s])torchrun(\s|$)"),
    re.compile(r"python[\d.]*\s+(-u\s+)?-m\s+torch\.distributed\.run(\s|$)"),
    re.compile(r"python[\d.]*\s+(-u\s+)?-m\s+axolotl\.cli\.train(\s|$)"),
    re.compile(r"(^|[/\s])axolotl\s+train(\s|$)"),
    re.compile(r"(^|[/\s])accelerate\s+launch(\s|$)"),
    re.compile(r"(^|[/\s])deepspeed(\s|$)"),
    re.compile(r"(^|[/\s])pt_data_worker(\s|$)"),
)
for line in sys.stdin:
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
        continue
    try:
        pid = int(parts[0])
        ppid = int(parts[1])
    except ValueError:
        continue
    if pid == self_pid or ppid == self_pid:
        continue
    cmd = parts[2]
    if any(pattern.search(cmd) for pattern in patterns):
        print(pid)
' "$$" || true
  )"
  [[ -n "$pids" ]] || return 0
  log "Cleaning leftover training child processes: $pids"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  sleep 10
  # shellcheck disable=SC2086
  kill -KILL $pids 2>/dev/null || true
}

gpu_snapshot() {
  local log_file="$1"
  local label="$2"
  {
    printf '\n[%s] GPU snapshot: %s\n' "$(timestamp)" "$label"
    nvidia_smi_retry || true
    printf '\n[%s] GPU compute apps: %s\n' "$(timestamp)" "$label"
    nvidia_smi_retry --query-compute-apps=pid,gpu_bus_id,used_memory,process_name --format=csv,noheader || true
  } | tee -a "$RUN_LOG" "$log_file" >/dev/null
}

attempt_contaminated() {
  local path
  for path in "$@"; do
    [[ -f "$path" ]] || continue
    if grep -Eq 'Process [0-9]+ has [0-9.]+ GiB memory in use|\[Not Found\]|stale GPU|stale-context|contaminated' "$path"; then
      return 0
    fi
  done
  return 1
}

latest_jsonl_status() {
  local status_file="$1"
  [[ -f "$status_file" ]] || return 1
  "$PYTHON" - "$status_file" <<'PY'
from __future__ import annotations

import json
import sys

latest = {}
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            latest = json.loads(line)
        except json.JSONDecodeError:
            continue
if not latest:
    raise SystemExit(1)
print(json.dumps(latest, sort_keys=True))
PY
}

probe_previously_failed() {
  is_true "$SKIP_PREVIOUS_FAILED_PROBES" || return 1
  local seq_len="$1"
  local source_mode="$2"
  local strategy="$3"
  local safe_mode="${source_mode//[^A-Za-z0-9]/-}"
  local safe_strategy="${strategy//[^A-Za-z0-9]/-}"
  local status_file="$STATUS_DIR/probe-${seq_len}-${safe_mode}-${safe_strategy}.jsonl"
  if attempt_contaminated "$status_file"; then
    return 1
  fi
  local latest
  latest="$(latest_jsonl_status "$status_file")" || return 1
  "$PYTHON" - "$latest" <<'PY'
from __future__ import annotations

import json
import sys

payload = json.loads(sys.argv[1])
if payload.get("status") == "training-finished" and payload.get("run_status") == "failed":
    raise SystemExit(0)
raise SystemExit(1)
PY
}

launch_env_args() {
  LAUNCH_ENV_ARGS=(
    "RUN_TAG=$RUN_TAG"
    "CONTEXT_LADDER_ROOT=$CONTEXT_LADDER_ROOT"
    "GTSM=$GTSM"
    "PYTHON=$PYTHON"
    "PROFILE=$PROFILE"
    "ARTIFACT_FORMAT=$ARTIFACT_FORMAT"
    "CORPUS_JSONL=$CORPUS_JSONL"
    "DATASET_MANIFEST=$DATASET_MANIFEST"
    "SOURCE_MODES=$SOURCE_MODES"
    "SOURCE_MODE_PREFERENCE=$SOURCE_MODE_PREFERENCE"
    "TEST_CONTEXT_MODE=$TEST_CONTEXT_MODE"
    "TEST_CONTEXT_MAX_CHARS=$TEST_CONTEXT_MAX_CHARS"
    "TEST_CONTEXT_MAX_FILES=$TEST_CONTEXT_MAX_FILES"
    "TEST_CONTEXT_MAX_FILE_BYTES=$TEST_CONTEXT_MAX_FILE_BYTES"
    "DISTRIBUTED_STRATEGIES=$DISTRIBUTED_STRATEGIES"
    "CONTEXT_LADDER=$CONTEXT_LADDER"
    "NUM_PROCESSES=$NUM_PROCESSES"
    "GPU_DEVICES=$GPU_DEVICES"
    "GPU_PREFLIGHT=$GPU_PREFLIGHT"
    "GPU_IDLE_MAX_MIB=$GPU_IDLE_MAX_MIB"
    "GPU_IDLE_MAX_UTIL=$GPU_IDLE_MAX_UTIL"
    "GPU_IDLE_WAIT_SECONDS=$GPU_IDLE_WAIT_SECONDS"
    "GPU_REQUIRE_NO_COMPUTE_APPS=$GPU_REQUIRE_NO_COMPUTE_APPS"
    "MIN_CORPUS_RECORDS=$MIN_CORPUS_RECORDS"
    "PROBE_MAX_STEPS=$PROBE_MAX_STEPS"
    "PROBE_ONLY=$PROBE_ONLY"
    "ESTIMATE_ONLY=$ESTIMATE_ONLY"
    "EXACT_TOKENIZER=$EXACT_TOKENIZER"
    "EXACT_TOKENIZER_REQUIRED=$EXACT_TOKENIZER_REQUIRED"
    "ESTIMATE_PROGRESS_INTERVAL=$ESTIMATE_PROGRESS_INTERVAL"
    "ESTIMATE_LIMIT=$ESTIMATE_LIMIT"
    "ESTIMATE_WORKERS=$ESTIMATE_WORKERS"
    "ESTIMATE_LONGEST_SAMPLE_COUNT=$ESTIMATE_LONGEST_SAMPLE_COUNT"
    "CANDIDATE_MAX_OVERFLOW_FRACTION=$CANDIDATE_MAX_OVERFLOW_FRACTION"
    "CANDIDATE_MAX_OVERFLOW_SAMPLES=$CANDIDATE_MAX_OVERFLOW_SAMPLES"
    "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
    "TRAIN_GRAD_ACCUM=$TRAIN_GRAD_ACCUM"
    "TRAIN_DATA_WORKERS=$TRAIN_DATA_WORKERS"
    "GTSM_TRAIN_DATA_WORKERS=$TRAIN_DATA_WORKERS"
    "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
    "PAD_TO_SEQUENCE_LEN=$PAD_TO_SEQUENCE_LEN"
    "STATUS_INTERVAL_SECONDS=$STATUS_INTERVAL_SECONDS"
    "LOG_TAIL_LINES=$LOG_TAIL_LINES"
    "POST_EXPORT_QUANTIZATIONS=$POST_EXPORT_QUANTIZATIONS"
    "POST_OLLAMA_CREATE=$POST_OLLAMA_CREATE"
    "POST_EXPORT_ENV_DIR=$POST_EXPORT_ENV_DIR"
    "POST_EXPORT_LLAMA_CPP_DIR=$POST_EXPORT_LLAMA_CPP_DIR"
    "POST_EXPORT_GGUF_OUTTYPE=$POST_EXPORT_GGUF_OUTTYPE"
    "POST_EXPORT_PREPARE=$POST_EXPORT_PREPARE"
    "POST_EXPORT_SYNC_OVERNIGHT=$POST_EXPORT_SYNC_OVERNIGHT"
    "POST_OLLAMA_FROM_QUANTIZATION=$POST_OLLAMA_FROM_QUANTIZATION"
    "TRAINING_METHOD=$TRAINING_METHOD"
    "LEARNING_RATE=$LEARNING_RATE"
    "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
    "CLEANUP_ORPHAN_TRAINING=$CLEANUP_ORPHAN_TRAINING"
    "SKIP_PREVIOUS_FAILED_PROBES=$SKIP_PREVIOUS_FAILED_PROBES"
    "DRY_RUN=$DRY_RUN"
  )
}

preflight() {
  save_state
  if [[ ! -f "$STARTED_FILE" ]]; then
    date -u +%s > "$STARTED_FILE"
  fi
  : > "$SUMMARY_MD"
  append_summary "# GTSM context ladder run $RUN_TAG"
  append_summary ""
  append_summary "- Started: $(timestamp)"
  append_summary "- Profile: $PROFILE"
  append_summary "- Artifact format: $ARTIFACT_FORMAT"
  append_summary "- Corpus: $CORPUS_JSONL"
  append_summary "- Context ladder: $CONTEXT_LADDER"
  append_summary "- Source modes: $SOURCE_MODES"
  append_summary "- Test context: $TEST_CONTEXT_MODE, max chars $TEST_CONTEXT_MAX_CHARS, max files $TEST_CONTEXT_MAX_FILES, max file bytes $TEST_CONTEXT_MAX_FILE_BYTES"
  append_summary "- Strategies: $DISTRIBUTED_STRATEGIES"
  if [[ -n "$POST_EXPORT_ENV_DIR" ]]; then
    append_summary "- External GGUF export env: $POST_EXPORT_ENV_DIR"
    append_summary "- External llama.cpp dir: $POST_EXPORT_LLAMA_CPP_DIR"
  fi

  command -v "$GTSM" >/dev/null || die "gtsm command not found: $GTSM"
  command -v "$PYTHON" >/dev/null || die "python command not found: $PYTHON"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required."
  [[ -f "$CORPUS_JSONL" ]] || die "Corpus JSONL not found: $CORPUS_JSONL"
  [[ -f "$DATASET_MANIFEST" ]] || die "Dataset manifest not found: $DATASET_MANIFEST"

  resolve_gpus
  save_state
  local corpus_records
  corpus_records="$(wc -l < "$CORPUS_JSONL" | tr -d ' ')"
  [[ "$corpus_records" -ge "$MIN_CORPUS_RECORDS" ]] \
    || die "Corpus has $corpus_records records, below MIN_CORPUS_RECORDS=$MIN_CORPUS_RECORDS."
  assert_no_training_processes
  if is_true "$GPU_PREFLIGHT"; then
    wait_for_idle_gpus || die "GPUs did not become idle before run start."
  else
    log "Skipping GPU idle preflight because GPU_PREFLIGHT=$GPU_PREFLIGHT."
  fi
  run_cmd "$LOG_DIR/preflight.log" "$GTSM" list-train-profiles
  if is_true "$GPU_PREFLIGHT"; then
    run_cmd "$LOG_DIR/preflight.log" nvidia-smi
  elif ! run_cmd_allow_failure "$LOG_DIR/preflight.log" nvidia-smi; then
    log "Continuing after best-effort nvidia-smi snapshot failed because GPU_PREFLIGHT=$GPU_PREFLIGHT."
  fi
  append_summary "- Corpus records: $corpus_records"
  append_summary "- GPU devices: $GPU_DEVICES"
  append_summary "- Num processes: $NUM_PROCESSES"
}

estimate_approx() {
  if [[ -f "$APPROX_ESTIMATE_JSON" ]]; then
    log "Using existing approximate estimator output: $APPROX_ESTIMATE_JSON"
  else
    local estimate_args=(
      estimate-training-tokens
      --profile "$PROFILE"
      --corpus-jsonl "$CORPUS_JSONL"
      --artifact-format "$ARTIFACT_FORMAT"
      --source-context-mode all-filtered
      --compare-source-context-modes "$SOURCE_MODES"
      --source-context-budget-ladder
      --test-context-mode "$TEST_CONTEXT_MODE"
      --test-context-max-chars "$TEST_CONTEXT_MAX_CHARS"
      --test-context-max-files "$TEST_CONTEXT_MAX_FILES"
      --test-context-max-file-bytes "$TEST_CONTEXT_MAX_FILE_BYTES"
      --max-seq-lengths "$CONTEXT_LADDER"
      --progress-interval "$ESTIMATE_PROGRESS_INTERVAL"
      --workers "$ESTIMATE_WORKERS"
      --longest-sample-count "$ESTIMATE_LONGEST_SAMPLE_COUNT"
    )
    if [[ "$ESTIMATE_LIMIT" -gt 0 ]]; then
      estimate_args+=(--limit "$ESTIMATE_LIMIT")
    fi
    run_json_cmd \
      "$APPROX_ESTIMATE_JSON" \
      "$LOG_DIR/estimate.approx.log" \
      "$GTSM" "${estimate_args[@]}"
  fi
  write_candidates "$APPROX_ESTIMATE_JSON" "$APPROX_CANDIDATES_TSV"
  cp "$APPROX_CANDIDATES_TSV" "$CANDIDATES_TSV"
}

write_candidates() {
  local estimate_json="$1"
  local out_tsv="$2"
  "$PYTHON" - "$estimate_json" "$out_tsv" "$SOURCE_MODE_PREFERENCE" \
    "$CANDIDATE_MAX_OVERFLOW_FRACTION" "$CANDIDATE_MAX_OVERFLOW_SAMPLES" <<'PY'
import csv
import json
import sys

(
    estimate_json,
    out_tsv,
    source_mode_preference,
    max_overflow_fraction_raw,
    max_overflow_samples_raw,
) = sys.argv[1:6]
try:
    max_overflow_fraction = float(max_overflow_fraction_raw)
except ValueError:
    raise SystemExit(
        f"Invalid CANDIDATE_MAX_OVERFLOW_FRACTION={max_overflow_fraction_raw!r}"
    )
try:
    max_overflow_samples = int(max_overflow_samples_raw)
except ValueError:
    raise SystemExit(
        f"Invalid CANDIDATE_MAX_OVERFLOW_SAMPLES={max_overflow_samples_raw!r}"
    )
if max_overflow_fraction < 0:
    raise SystemExit("CANDIDATE_MAX_OVERFLOW_FRACTION must be nonnegative.")
if max_overflow_samples < 0:
    raise SystemExit("CANDIDATE_MAX_OVERFLOW_SAMPLES must be nonnegative.")

mode_order = {
    mode.strip(): index
    for index, mode in enumerate(source_mode_preference.split(","))
    if mode.strip()
}
with open(estimate_json, encoding="utf-8") as handle:
    data = json.load(handle)

rows = []
for estimate in data.get("estimates", []):
    source = estimate.get("source_context", {})
    mode = str(source.get("mode", ""))
    for threshold in estimate.get("thresholds", []):
        max_seq_length = int(threshold.get("max_seq_length", 0) or 0)
        overflow = int(threshold.get("over_max_seq_length", 0) or 0)
        samples = int(estimate.get("samples", 0) or 0)
        overflow_fraction = float(threshold.get("overflow_fraction", 0.0) or 0.0)
        rows.append(
            {
                "source_mode": mode,
                "source_max_chars": int(source.get("max_chars", 0) or 0),
                "source_max_files": int(source.get("max_files", 0) or 0),
                "max_seq_length": max_seq_length,
                "overflow": overflow,
                "overflow_fraction": overflow_fraction,
                "samples": samples,
                "token_p99": int(estimate.get("token_summary", {}).get("p99", 0) or 0),
                "token_max": int(estimate.get("token_summary", {}).get("max", 0) or 0),
                "mode_rank": mode_order.get(mode, 999),
            }
        )

passing = [
    row
    for row in rows
    if row["overflow"] <= max_overflow_samples
    and row["overflow_fraction"] <= max_overflow_fraction
]
passing.sort(
    key=lambda row: (
        -row["max_seq_length"],
        row["overflow"],
        row["overflow_fraction"],
        row["mode_rank"],
    )
)
with open(out_tsv, "w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "source_mode",
        "source_max_chars",
        "source_max_files",
        "max_seq_length",
        "overflow",
        "overflow_fraction",
        "samples",
        "token_p99",
        "token_max",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for row in passing:
        writer.writerow({key: row[key] for key in fieldnames})

if not passing:
    print(
        "No estimator candidates fit within overflow limits "
        f"(fraction <= {max_overflow_fraction}, samples <= {max_overflow_samples}).",
        file=sys.stderr,
    )
    rows.sort(key=lambda row: (row["overflow"], -row["max_seq_length"], row["mode_rank"]))
    for row in rows[:10]:
        print(
            "candidate "
            f"mode={row['source_mode']} seq={row['max_seq_length']} "
            f"overflow={row['overflow']}/{row['samples']} "
            f"fraction={row['overflow_fraction']:.6f} "
            f"p99={row['token_p99']} max={row['token_max']}",
            file=sys.stderr,
        )
    sys.exit(1)
PY
}

candidate_contexts_for_exact() {
  "$PYTHON" - "$CANDIDATES_TSV" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if not rows:
    sys.exit(1)
first = int(rows[0]["max_seq_length"])
lower = None
for row in rows[1:]:
    value = int(row["max_seq_length"])
    if value < first:
        lower = value
        break
values = [first]
if lower is not None:
    values.append(lower)
print(",".join(str(value) for value in values))
PY
}

merge_candidate_files() {
  local primary_tsv="$1"
  local fallback_tsv="$2"
  local out_tsv="$3"
  "$PYTHON" - "$primary_tsv" "$fallback_tsv" "$out_tsv" <<'PY'
import csv
import sys

primary_tsv, fallback_tsv, out_tsv = sys.argv[1:4]
fieldnames = [
    "source_mode",
    "source_max_chars",
    "source_max_files",
    "max_seq_length",
    "overflow",
    "overflow_fraction",
    "samples",
    "token_p99",
    "token_max",
]
seen = set()
rows = []
for path in (primary_tsv, fallback_tsv):
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (
                row.get("source_mode", ""),
                row.get("source_max_chars", ""),
                row.get("source_max_files", ""),
                row.get("max_seq_length", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append({field: row.get(field, "") for field in fieldnames})
with open(out_tsv, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
PY
}

estimate_exact_finalists() {
  is_true "$EXACT_TOKENIZER" || return 0
  local exact_contexts
  exact_contexts="$(candidate_contexts_for_exact)" || return 0
  local selected_source_mode
  selected_source_mode="$(awk -F '\t' 'NR==2 {print $1}' "$CANDIDATES_TSV")"
  if [[ -z "$selected_source_mode" ]]; then
    return 0
  fi
  local exact_args=(
    estimate-training-tokens
    --profile "$PROFILE"
    --corpus-jsonl "$CORPUS_JSONL"
    --artifact-format "$ARTIFACT_FORMAT"
    --source-context-mode "$selected_source_mode"
    --compare-source-context-modes "$SOURCE_MODES"
    --source-context-budget-ladder
    --test-context-mode "$TEST_CONTEXT_MODE"
    --test-context-max-chars "$TEST_CONTEXT_MAX_CHARS"
    --test-context-max-files "$TEST_CONTEXT_MAX_FILES"
    --test-context-max-file-bytes "$TEST_CONTEXT_MAX_FILE_BYTES"
    --max-seq-lengths "$exact_contexts"
    --exact-tokenizer
    --progress-interval "$ESTIMATE_PROGRESS_INTERVAL"
    --workers "$ESTIMATE_WORKERS"
    --longest-sample-count "$ESTIMATE_LONGEST_SAMPLE_COUNT"
  )
  if [[ "$ESTIMATE_LIMIT" -gt 0 ]]; then
    exact_args+=(--limit "$ESTIMATE_LIMIT")
  fi
  if run_json_cmd \
    "$EXACT_ESTIMATE_JSON" \
    "$LOG_DIR/estimate.exact.log" \
    "$GTSM" "${exact_args[@]}"; then
    if write_candidates "$EXACT_ESTIMATE_JSON" "$EXACT_CANDIDATES_TSV"; then
      merge_candidate_files "$EXACT_CANDIDATES_TSV" "$APPROX_CANDIDATES_TSV" "$CANDIDATES_TSV"
      log "Using exact-tokenizer finalists from $EXACT_CANDIDATES_TSV with approximate fallbacks from $APPROX_CANDIDATES_TSV"
    else
      is_true "$EXACT_TOKENIZER_REQUIRED" && die "Exact-tokenizer estimator produced no passing candidates."
      log "Exact-tokenizer estimator produced no passing candidates; keeping approximate candidates."
    fi
  else
    is_true "$EXACT_TOKENIZER_REQUIRED" && die "Exact-tokenizer estimator failed."
    log "Exact-tokenizer estimator failed; keeping approximate candidates."
  fi
}

training_extra_args() {
  TRAINING_EXTRA_ARGS=()
  if [[ -n "$TRAINING_METHOD" ]]; then
    TRAINING_EXTRA_ARGS+=(--training-method "$TRAINING_METHOD")
  fi
  if [[ -n "$LEARNING_RATE" ]]; then
    TRAINING_EXTRA_ARGS+=(--learning-rate "$LEARNING_RATE")
  fi
  if is_true "$PAD_TO_SEQUENCE_LEN"; then
    TRAINING_EXTRA_ARGS+=(--pad-to-sequence-len)
  else
    TRAINING_EXTRA_ARGS+=(--no-pad-to-sequence-len)
  fi
  if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
    TRAINING_EXTRA_ARGS+=(--attn-implementation "$ATTN_IMPLEMENTATION")
  fi
}

train_command_common() {
  local variant_id="$1"
  local run_id="$2"
  local seq_len="$3"
  local source_mode="$4"
  local source_chars="$5"
  local source_files="$6"
  local strategy="$7"
  shift 7
  training_extra_args
  TRAIN_COMMAND=(
    "$GTSM" train
    --profile "$PROFILE"
    --dataset-manifest "$DATASET_MANIFEST"
    --corpus-jsonl "$CORPUS_JSONL"
    --variant-id "$variant_id"
    --artifact-format "$ARTIFACT_FORMAT"
    --backend axolotl
    --num-processes "$NUM_PROCESSES"
    --distributed-strategy "$strategy"
    --max-seq-length "$seq_len"
    --per-device-batch-size "$TRAIN_BATCH_SIZE"
    --gradient-accumulation-steps "$TRAIN_GRAD_ACCUM"
    --source-context-mode "$source_mode"
    --source-context-max-chars "$source_chars"
    --source-context-max-files "$source_files"
    --test-context-mode "$TEST_CONTEXT_MODE"
    --test-context-max-chars "$TEST_CONTEXT_MAX_CHARS"
    --test-context-max-files "$TEST_CONTEXT_MAX_FILES"
    --test-context-max-file-bytes "$TEST_CONTEXT_MAX_FILE_BYTES"
    --status-interval-seconds "$STATUS_INTERVAL_SECONDS"
    --stream-logs
    --log-tail-lines "$LOG_TAIL_LINES"
    --internal-run-id "$run_id"
    "${TRAINING_EXTRA_ARGS[@]}"
    "$@"
  )
}

write_selection() {
  local seq_len="$1"
  local source_mode="$2"
  local source_chars="$3"
  local source_files="$4"
  local strategy="$5"
  local variant_id="$6"
  local run_id="$7"
  {
    printf 'SELECTED_MAX_SEQ_LENGTH=%q\n' "$seq_len"
    printf 'SELECTED_SOURCE_MODE=%q\n' "$source_mode"
    printf 'SELECTED_SOURCE_MAX_CHARS=%q\n' "$source_chars"
    printf 'SELECTED_SOURCE_MAX_FILES=%q\n' "$source_files"
    printf 'SELECTED_DISTRIBUTED_STRATEGY=%q\n' "$strategy"
    printf 'SELECTED_VARIANT_ID=%q\n' "$variant_id"
    printf 'SELECTED_RUN_ID=%q\n' "$run_id"
  } > "$SELECTION_ENV"
}

run_external_post_export() {
  local variant_id="$1"
  local log_file="$2"
  local helper="$REPO_ROOT/scripts/gtsm_llama_cpp_gguf.sh"
  local ollama_model_name="gtsm-${variant_id}-q4"

  if [[ ! -f "$helper" ]]; then
    log "External GGUF helper not found: $helper"
    return 1
  fi
  if [[ -z "$POST_EXPORT_ENV_DIR" ]]; then
    log "POST_EXPORT_ENV_DIR is empty; cannot run external post-export."
    return 1
  fi

  log "Running external GGUF post-export for $variant_id using $POST_EXPORT_ENV_DIR"
  if is_true "$POST_EXPORT_PREPARE"; then
    run_cmd_allow_failure "$log_file" \
      env \
        "ENV=$POST_EXPORT_ENV_DIR" \
        "LLAMA_CPP_DIR=$POST_EXPORT_LLAMA_CPP_DIR" \
        "GGUF_OUTTYPE=$POST_EXPORT_GGUF_OUTTYPE" \
        bash "$helper" prepare || return 1
  fi

  run_cmd_allow_failure "$log_file" \
    env \
      "ENV=$POST_EXPORT_ENV_DIR" \
      "LLAMA_CPP_DIR=$POST_EXPORT_LLAMA_CPP_DIR" \
      "GGUF_OUTTYPE=$POST_EXPORT_GGUF_OUTTYPE" \
      "EXPORT_QUANTIZATIONS=$POST_EXPORT_QUANTIZATIONS" \
      "VARIANT_ID=$variant_id" \
      bash "$helper" export || return 1

  run_cmd_allow_failure "$log_file" \
    env \
      "ENV=$POST_EXPORT_ENV_DIR" \
      "VARIANT_ID=$variant_id" \
      "RUN_TAG=$RUN_TAG" \
      "SYNC_OVERNIGHT_EXPORT=$POST_EXPORT_SYNC_OVERNIGHT" \
      "OLLAMA_MODEL_NAME=$ollama_model_name" \
      "OLLAMA_FROM_QUANTIZATION=$POST_OLLAMA_FROM_QUANTIZATION" \
      "OLLAMA_CREATE=$POST_OLLAMA_CREATE" \
      bash "$helper" finalize || return 1
}

run_training_attempt() {
  local label="$1"
  local seq_len="$2"
  local source_mode="$3"
  local source_chars="$4"
  local source_files="$5"
  local strategy="$6"
  local max_steps="$7"
  local export_outputs="$8"
  local safe_mode="${source_mode//[^A-Za-z0-9]/-}"
  local safe_strategy="${strategy//[^A-Za-z0-9]/-}"
  local variant_id="tools-iuc-devstral-24b-${ARTIFACT_FORMAT}-${safe_mode}-${seq_len}-${safe_strategy}-${RUN_TAG}"
  local run_id="train-${RUN_TAG}-${label}-${seq_len}-${safe_mode}-${safe_strategy}"
  local log_file="$LOG_DIR/${label}-${seq_len}-${safe_mode}-${safe_strategy}.log"
  local status_log="$STATUS_DIR/${label}-${seq_len}-${safe_mode}-${safe_strategy}.jsonl"
  local extra_args=(--status-log "$status_log")
  local use_external_post_export=0
  LAST_ATTEMPT_STATUS="failed"
  if [[ "$max_steps" -gt 0 ]]; then
    extra_args+=(--max-steps "$max_steps")
  fi
  if is_true "$export_outputs"; then
    if [[ -n "$POST_EXPORT_ENV_DIR" ]]; then
      use_external_post_export=1
    else
      extra_args+=(--post-export-quantizations "$POST_EXPORT_QUANTIZATIONS")
      extra_args+=(--post-ollama-model-name "gtsm-${variant_id}-q4")
      if is_true "$POST_OLLAMA_CREATE"; then
        extra_args+=(--post-ollama-create)
      fi
    fi
  fi
  train_command_common \
    "$variant_id" "$run_id" "$seq_len" "$source_mode" "$source_chars" "$source_files" "$strategy" \
    "${extra_args[@]}"

  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  export PYTORCH_CUDA_ALLOC_CONF
  if is_true "$GPU_PREFLIGHT"; then
    cleanup_training_processes
    gpu_snapshot "$log_file" "before $label seq_len=$seq_len source=$source_mode strategy=$strategy"
    wait_for_idle_gpus || die "GPUs did not become idle before $label seq_len=$seq_len source=$source_mode strategy=$strategy."
  fi
  if run_cmd_allow_failure "$log_file" "${TRAIN_COMMAND[@]}"; then
    write_selection "$seq_len" "$source_mode" "$source_chars" "$source_files" "$strategy" "$variant_id" "$run_id"
    if [[ "$use_external_post_export" -eq 1 ]]; then
      if ! run_external_post_export "$variant_id" "$log_file"; then
        LAST_ATTEMPT_STATUS="post-export-failed"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(timestamp)" "$label" "$seq_len" "$source_mode" "$source_chars" "$source_files" \
          "$strategy" "$LAST_ATTEMPT_STATUS" "$variant_id" >> "$FULL_REPORT_TSV"
        return 3
      fi
    fi
    LAST_ATTEMPT_STATUS="completed"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(timestamp)" "$label" "$seq_len" "$source_mode" "$source_chars" "$source_files" \
      "$strategy" "completed" "$variant_id" >> "$FULL_REPORT_TSV"
    return 0
  fi
  if attempt_contaminated "$log_file" "$status_log"; then
    LAST_ATTEMPT_STATUS="contaminated"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(timestamp)" "$label" "$seq_len" "$source_mode" "$source_chars" "$source_files" \
    "$strategy" "$LAST_ATTEMPT_STATUS" "$variant_id" >> "$FULL_REPORT_TSV"
  cleanup_training_processes
  if is_true "$GPU_PREFLIGHT"; then
    gpu_snapshot "$log_file" "after failed $label seq_len=$seq_len source=$source_mode strategy=$strategy"
    if ! wait_for_idle_gpus; then
      LAST_ATTEMPT_STATUS="contaminated"
      log "GPU cleanup did not reach idle after failed $label seq_len=$seq_len source=$source_mode strategy=$strategy; stopping ladder to avoid contaminating later candidates."
      return 2
    fi
  fi
  return 1
}

probe_and_train_candidates() {
  [[ -f "$CANDIDATES_TSV" ]] || die "Candidate TSV not found: $CANDIDATES_TSV"
  rm -f "$SELECTION_ENV"
  printf 'timestamp\tmax_seq_length\tsource_mode\tsource_max_chars\tsource_max_files\tstrategy\tstatus\ttoken_p99\n' > "$PROBE_REPORT_TSV"
  printf 'timestamp\tlabel\tmax_seq_length\tsource_mode\tsource_max_chars\tsource_max_files\tstrategy\tstatus\tvariant_id\n' > "$FULL_REPORT_TSV"
  while IFS=$'\t' read -r source_mode source_chars source_files seq_len overflow overflow_fraction samples token_p99 token_max; do
    [[ -n "$seq_len" ]] || continue
    split_csv "$DISTRIBUTED_STRATEGIES"
    local strategy
    for strategy in "${SPLIT_CSV_RESULT[@]}"; do
      strategy="${strategy//[[:space:]]/}"
      [[ -n "$strategy" ]] || continue
      if probe_previously_failed "$seq_len" "$source_mode" "$strategy"; then
        log "Skipping previously failed probe seq_len=$seq_len source=$source_mode strategy=$strategy"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(timestamp)" "$seq_len" "$source_mode" "$source_chars" "$source_files" "$strategy" "skipped-previous-failure" "$token_p99" >> "$PROBE_REPORT_TSV"
        continue
      fi
      log "Probing seq_len=$seq_len source=$source_mode strategy=$strategy"
      if run_training_attempt "probe" "$seq_len" "$source_mode" "$source_chars" "$source_files" "$strategy" "$PROBE_MAX_STEPS" 0; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(timestamp)" "$seq_len" "$source_mode" "$source_chars" "$source_files" "$strategy" "passed" "$token_p99" >> "$PROBE_REPORT_TSV"
        if is_true "$PROBE_ONLY"; then
          log "Probe-only mode selected $seq_len/$source_mode/$strategy."
          return 0
        fi
        log "Starting full training for seq_len=$seq_len source=$source_mode strategy=$strategy"
        if run_training_attempt "full" "$seq_len" "$source_mode" "$source_chars" "$source_files" "$strategy" 0 1; then
          write_summary "completed"
          return 0
        else
          local full_rc="$?"
          if [[ "$full_rc" -eq 2 ]]; then
            die "Stopping after contaminated or non-idle GPU state during full training for seq_len=$seq_len source=$source_mode strategy=$strategy."
          elif [[ "$full_rc" -eq 3 ]]; then
            write_summary "post-export-failed"
            die "Post-training export failed after successful training for seq_len=$seq_len source=$source_mode strategy=$strategy."
          fi
        fi
        log "Full training failed after passing probe; continuing to next candidate."
      else
        local attempt_rc="$?"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(timestamp)" "$seq_len" "$source_mode" "$source_chars" "$source_files" "$strategy" "${LAST_ATTEMPT_STATUS:-failed}" "$token_p99" >> "$PROBE_REPORT_TSV"
        if [[ "$attempt_rc" -eq 2 ]]; then
          die "Stopping after contaminated or non-idle GPU state for seq_len=$seq_len source=$source_mode strategy=$strategy."
        fi
      fi
    done
  done < <(tail -n +2 "$CANDIDATES_TSV")
  die "No candidate completed under GPU-only strategies."
}

write_summary() {
  local status="$1"
  {
    echo "# GTSM context ladder run $RUN_TAG"
    echo
    echo "- Status: $status"
    echo "- Profile: $PROFILE"
    echo "- Artifact format: $ARTIFACT_FORMAT"
    echo "- Context ladder: $CONTEXT_LADDER"
    echo "- Source modes: $SOURCE_MODES"
    echo "- Test context: $TEST_CONTEXT_MODE, max chars $TEST_CONTEXT_MAX_CHARS, max files $TEST_CONTEXT_MAX_FILES, max file bytes $TEST_CONTEXT_MAX_FILE_BYTES"
    echo "- Strategies: $DISTRIBUTED_STRATEGIES"
    if [[ -n "$POST_EXPORT_ENV_DIR" ]]; then
      echo "- External GGUF export env: $POST_EXPORT_ENV_DIR"
      echo "- External llama.cpp dir: $POST_EXPORT_LLAMA_CPP_DIR"
      echo "- External export quantizations: $POST_EXPORT_QUANTIZATIONS"
    fi
    echo "- Approximate estimate: $APPROX_ESTIMATE_JSON"
    echo "- Exact estimate: $EXACT_ESTIMATE_JSON"
    echo "- Candidates: $CANDIDATES_TSV"
    echo "- Probe report: $PROBE_REPORT_TSV"
    echo "- Training report: $FULL_REPORT_TSV"
    if [[ -f "$SELECTION_ENV" ]]; then
      echo
      echo "## Selection"
      sed 's/^/- /' "$SELECTION_ENV"
    fi
  } > "$SUMMARY_MD"
}

run_pipeline() {
  preflight
  estimate_approx
  estimate_exact_finalists
  write_summary "estimated"
  if is_true "$ESTIMATE_ONLY"; then
    log "Estimate-only mode complete."
    return 0
  fi
  probe_and_train_candidates
}

status() {
  echo "RUN_TAG=$RUN_TAG"
  echo "RUN_ROOT=$RUN_ROOT"
  if [[ -f "$STARTED_FILE" ]]; then
    "$PYTHON" - "$STARTED_FILE" <<'PY'
from __future__ import annotations

import sys
import time


def fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


try:
    started = int(open(sys.argv[1], encoding="utf-8").read().strip())
except Exception:
    raise SystemExit(0)
print(f"ELAPSED={fmt(time.time() - started)}")
PY
  fi
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "PID=$pid"
    if kill -0 "$pid" 2>/dev/null; then
      echo "PIPELINE_RUNNING=1"
    else
      echo "PIPELINE_RUNNING=0"
    fi
  fi
  if [[ -f "$TMUX_SESSION_FILE" ]]; then
    local tmux_session
    tmux_session="$(cat "$TMUX_SESSION_FILE")"
    echo "TMUX_SESSION=$tmux_session"
    if command -v tmux >/dev/null && tmux has-session -t "$tmux_session" 2>/dev/null; then
      echo "TMUX_SESSION_RUNNING=1"
    else
      echo "TMUX_SESSION_RUNNING=0"
    fi
  fi
  local latest_status
  latest_status="$(latest_status_file)"
  if [[ -n "$latest_status" && -f "$latest_status" ]]; then
    echo "LATEST_STATUS_FILE=$latest_status"
    "$PYTHON" - "$latest_status" <<'PY'
from __future__ import annotations

import json
import sys


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


path = sys.argv[1]
line = ""
with open(path, encoding="utf-8") as handle:
    for candidate in handle:
        if candidate.strip():
            line = candidate
if not line:
    raise SystemExit(0)
try:
    payload = json.loads(line)
except json.JSONDecodeError:
    print(f"LATEST_STATUS={line.strip()}")
    raise SystemExit(0)

status = payload.get("status", "")
run_status = payload.get("run_status", "")
if status:
    print(f"LATEST_STATUS={status}")
if run_status:
    print(f"LATEST_RUN_STATUS={run_status}")
progress = payload.get("progress") or {}
if isinstance(progress, dict):
    completed = progress.get("completed_units")
    total = progress.get("total_units")
    if completed is not None:
        suffix = f"/{total}" if total is not None else ""
        print(f"LATEST_PROGRESS={completed}{suffix}")
    elapsed = fmt(progress.get("elapsed_seconds"))
    if elapsed:
        print(f"LATEST_PROGRESS_ELAPSED={elapsed}")
    eta = fmt(progress.get("eta_seconds"))
    if eta:
        print(f"LATEST_PROGRESS_ETA={eta}")
    eta_timestamp = progress.get("eta_timestamp")
    if eta_timestamp:
        print(f"LATEST_PROGRESS_ETA_AT={eta_timestamp}")
last_stdout = str(payload.get("last_stdout_line") or "").strip()
last_stderr = str(payload.get("last_stderr_line") or "").strip()
if last_stdout:
    print(f"LATEST_STDOUT={last_stdout[-240:]}")
if last_stderr:
    print(f"LATEST_STDERR={last_stderr[-240:]}")
PY
  fi
  if [[ -f "$SUMMARY_MD" ]]; then
    cat "$SUMMARY_MD"
  fi
  if [[ -f "$PROBE_REPORT_TSV" ]]; then
    tail -n 5 "$PROBE_REPORT_TSV"
  fi
  if [[ -f "$FULL_REPORT_TSV" ]]; then
    tail -n 5 "$FULL_REPORT_TSV"
  fi
  return 0
}

tail_logs() {
  tail -n "${TAIL_LINES:-120}" -f "$RUN_LOG"
}

launch() {
  save_state
  date -u +%s > "$STARTED_FILE"
  launch_env_args
  local backend="$LAUNCH_BACKEND"
  if [[ "$backend" == "auto" ]]; then
    if command -v tmux >/dev/null; then
      backend="tmux"
    else
      backend="nohup"
    fi
  fi
  log "Launching detached context ladder run: $RUN_TAG with backend=$backend"
  if [[ "$backend" == "tmux" ]]; then
    command -v tmux >/dev/null || die "tmux launch backend requested but tmux is not installed."
    local tmux_session="gtsm-context-${RUN_TAG//[^A-Za-z0-9_-]/-}"
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
      die "tmux session already exists: $tmux_session"
    fi
    local tmux_command
    tmux_command="$(command_string env "${LAUNCH_ENV_ARGS[@]}" "$SCRIPT_PATH" run) >> $(printf '%q' "$DETACH_LOG") 2>&1"
    tmux new-session -d -s "$tmux_session" -c "$REPO_ROOT" "$tmux_command"
    printf '%s\n' "$tmux_session" > "$TMUX_SESSION_FILE"
    tmux display-message -p -t "$tmux_session" '#{pane_pid}' > "$PID_FILE" 2>/dev/null || true
    log "Detached tmux session $tmux_session, pid $(cat "$PID_FILE" 2>/dev/null || true), log $DETACH_LOG"
  elif [[ "$backend" == "nohup" ]]; then
    nohup env "${LAUNCH_ENV_ARGS[@]}" "$SCRIPT_PATH" run > "$DETACH_LOG" 2>&1 &
    echo "$!" > "$PID_FILE"
    rm -f "$TMUX_SESSION_FILE"
    log "Detached PID $(cat "$PID_FILE"), log $DETACH_LOG"
  else
    die "Unsupported LAUNCH_BACKEND=$LAUNCH_BACKEND. Use auto, tmux, or nohup."
  fi
}

clean() {
  cleanup_training_processes
  if is_true "$GPU_PREFLIGHT"; then
    wait_for_idle_gpus || true
  fi
}

case "$MODE" in
  launch) launch ;;
  run) run_pipeline ;;
  estimate)
    preflight
    estimate_approx
    estimate_exact_finalists
    write_summary "estimated"
    ;;
  probe)
    preflight
    [[ -f "$CANDIDATES_TSV" ]] || estimate_approx
    estimate_exact_finalists
    PROBE_ONLY=1 probe_and_train_candidates
    write_summary "probed"
    ;;
  train)
    preflight
    [[ -f "$CANDIDATES_TSV" ]] || estimate_approx
    estimate_exact_finalists
    probe_and_train_candidates
    ;;
  status) status ;;
  tail) tail_logs ;;
  clean) clean ;;
  classify-contamination)
    shift || true
    attempt_contaminated "$@"
    ;;
esac
