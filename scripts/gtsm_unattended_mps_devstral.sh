#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-run}"
case "$MODE" in
  launch|run|status|tail|export-corpus|watch|launch-watch) ;;
  *)
    echo "Usage: $0 {launch|run|status|tail|export-corpus|watch|launch-watch}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SCRIPT_PATH="$REPO_ROOT/scripts/gtsm_unattended_mps_devstral.sh"

: "${GTSM:=.conda/gtsm-mps/bin/gtsm}"
: "${PYTHON:=.conda/gtsm-mps/bin/python}"
: "${HF_CLI:=.conda/gtsm-mps/bin/hf}"
: "${MAMBA:=/Users/blanked2/miniforge3/bin/mamba}"
: "${BASE_MODEL:=mistralai/Devstral-Small-2505}"
: "${PROFILE:=mps-devstral-24b}"
: "${VARIANT_PREFIX:=tools-iuc-mps-devstral-24b-mixed}"
: "${CORPUS_JSONL:=.gtsm-cache/datasets/tools-iuc-corpus.jsonl}"
: "${CORPUS_CHECKPOINT:=.gtsm-cache/datasets/tools-iuc-corpus.checkpoint}"
: "${RETRY_MANIFEST:=.gtsm-cache/datasets/tools-iuc-corpus.retry-manifest.json}"
: "${DATASET_MANIFEST:=config/dataset.manifest.json}"
: "${EXTRACT_MAX_WORKERS:=8}"
: "${SOURCE_WORKERS:=8}"
: "${CONTAINER_RUNTIME:=docker}"
: "${CONTAINER_CACHE_DIR:=.gtsm-cache/containers}"
: "${CONTAINER_PREPARE_WORKERS:=2}"
: "${CONTAINER_PROBE_WORKERS:=4}"
: "${CONTAINER_IMAGE_TIMEOUT_SECONDS:=600}"
: "${CONTAINER_IMAGE_QUARANTINE_SECONDS:=86400}"
: "${CONTAINER_HELP_PROBE_MODE:=exploratory}"
: "${SOURCE_DOWNLOAD_TIMEOUT_SECONDS:=90}"
: "${BIOCONDA_REF:=master}"
: "${TOOLS_REF:=main}"
: "${GALAXY_SKILLS_REF:=main}"
: "${GALAXY_XSD_REF:=dev}"
: "${SOURCE_CONTEXT_MODE:=snippets}"
: "${SOURCE_CONTEXT_MAX_CHARS:=6000}"
: "${SOURCE_CONTEXT_MAX_FILES:=16}"
: "${TRAIN_MAX_SEQ_LENGTH:=8192}"
: "${TRAIN_BATCH_SIZE:=1}"
: "${TRAIN_GRAD_ACCUM:=2}"
: "${STATUS_INTERVAL_SECONDS:=30}"
: "${LOG_TAIL_LINES:=40}"
: "${BENCHMARK_LIMIT:=50}"
: "${BENCHMARK_MAX_WORKERS:=1}"
: "${BENCHMARK_NUM_PROCESSES:=1}"
: "${MAX_TOKENS:=4096}"
: "${MAX_PROMPT_HELP_CHARS:=4000}"
: "${EXPORT_QUANTIZATIONS:=q8_0,q6_k,q5_k_m,q4_k_m}"
: "${LLAMA_CPP_DIR:=.gtsm-cache/llama.cpp}"
: "${LLAMA_CPP_ENV:=.conda/gtsm-llama-cpp-mps}"
: "${HF_HOME:=$REPO_ROOT/.gtsm-cache/huggingface}"

UNATTENDED_ROOT=".gtsm-cache/runs/unattended"
CURRENT_FILE="$UNATTENDED_ROOT/current"

resolve_run_tag() {
  if [[ -n "${RUN_TAG:-}" ]]; then
    return 0
  fi
  if [[ "$MODE" == "launch" || "$MODE" == "run" ]]; then
    RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
  elif [[ -f "$CURRENT_FILE" ]]; then
    RUN_TAG="$(cat "$CURRENT_FILE")"
  else
    echo "RUN_TAG is not set and no current unattended run exists." >&2
    exit 2
  fi
}

resolve_run_tag
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must contain only letters, numbers, '.', '_', and '-'." >&2
  exit 2
fi

RUN_ROOT="$UNATTENDED_ROOT/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"
MARKER_DIR="$RUN_ROOT/markers"
EXPORT_DIR="$RUN_ROOT/exports"
BENCHMARK_DIR="$RUN_ROOT/benchmarks"
STATE_ENV="$RUN_ROOT/state.env"
RUN_LOG="$RUN_ROOT/run.log"
DETACH_LOG="$RUN_ROOT/detach.log"
PID_FILE="$RUN_ROOT/pipeline.pid"
SUMMARY_MD="$RUN_ROOT/summary.md"
STAGE_FILE="$RUN_ROOT/current-stage.txt"
STATUS_FILE="$RUN_ROOT/status.txt"
WATCH_PID_FILE="$RUN_ROOT/watch.pid"
WATCH_LOG="$RUN_ROOT/watch.log"
CORPUS_EXPORT_DIR="$EXPORT_DIR/corpus"
VARIANT_ID="${VARIANT_ID:-$VARIANT_PREFIX-$RUN_TAG}"
VARIANT_MANIFEST=".gtsm-cache/models/variants/$VARIANT_ID.manifest.json"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-train-$RUN_TAG-mps-devstral24b}"

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$MARKER_DIR" "$EXPORT_DIR" "$BENCHMARK_DIR"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

save_state() {
  {
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'VARIANT_ID=%q\n' "$VARIANT_ID"
    printf 'TRAIN_RUN_ID=%q\n' "$TRAIN_RUN_ID"
    printf 'GTSM=%q\n' "$GTSM"
    printf 'PYTHON=%q\n' "$PYTHON"
    printf 'HF_CLI=%q\n' "$HF_CLI"
    printf 'MAMBA=%q\n' "$MAMBA"
    printf 'BASE_MODEL=%q\n' "$BASE_MODEL"
    printf 'PROFILE=%q\n' "$PROFILE"
    printf 'HF_HOME=%q\n' "$HF_HOME"
    printf 'CORPUS_JSONL=%q\n' "$CORPUS_JSONL"
    printf 'DATASET_MANIFEST=%q\n' "$DATASET_MANIFEST"
    printf 'BENCHMARK_LIMIT=%q\n' "$BENCHMARK_LIMIT"
    printf 'EXPORT_QUANTIZATIONS=%q\n' "$EXPORT_QUANTIZATIONS"
  } > "$STATE_ENV"
}

run_cmd() {
  log "+ $*"
  "$@"
}

run_cmd_logged() {
  local output_log="$1"
  shift
  mkdir -p "$(dirname "$output_log")"
  log "+ $*"
  "$@" >> "$output_log" 2>&1
  local code=$?
  if [[ -s "$output_log" ]]; then
    log "Last log lines from $output_log:"
    tail -n 20 "$output_log" || true
  fi
  return "$code"
}

run_cmd_allow_failure() {
  local label="$1"
  local output_log="$2"
  shift 2
  if run_cmd_logged "$output_log" "$@"; then
    return 0
  else
    local code=$?
    log "Nonfatal step failed ($label), exit_code=$code. See $output_log"
    printf '%s\t%s\t%s\n' "$(timestamp)" "$label" "$output_log" >> "$RUN_ROOT/nonfatal-failures.tsv"
  fi
  return 0
}

begin_stage() {
  local stage="$1"
  echo "$stage" > "$STAGE_FILE"
  log "==> $stage"
}

complete_stage() {
  local stage="$1"
  touch "$MARKER_DIR/$stage.done"
  log "<== $stage"
}

run_stage() {
  local stage="$1"
  shift
  if [[ -f "$MARKER_DIR/$stage.done" ]]; then
    log "Skipping completed stage: $stage"
    return 0
  fi
  begin_stage "$stage"
  "$@"
  complete_stage "$stage"
}

json_get() {
  local path="$1"
  local expr="$2"
  jq -r "$expr" "$path"
}

write_summary() {
  local status="$1"
  {
    echo "# MPS Devstral unattended run"
    echo
    echo "- Status: $status"
    echo "- Run tag: $RUN_TAG"
    echo "- Variant id: $VARIANT_ID"
    echo "- Base model: $BASE_MODEL"
    echo "- Profile: $PROFILE"
    echo "- Corpus: $CORPUS_JSONL"
    echo "- Variant manifest: $VARIANT_MANIFEST"
    echo "- Run log: $RUN_LOG"
    echo "- Hugging Face cache: $HF_HOME"
    echo
    echo "## Artifacts"
    find "$EXPORT_DIR" -maxdepth 3 -type f -print 2>/dev/null | sort || true
    echo
    echo "## Benchmarks"
    find "$BENCHMARK_DIR" -maxdepth 3 -type f -name 'benchmark.summary.json' -print 2>/dev/null | sort || true
    if [[ -f "$RUN_ROOT/nonfatal-failures.tsv" ]]; then
      echo
      echo "## Nonfatal failures"
      cat "$RUN_ROOT/nonfatal-failures.tsv"
    fi
  } > "$SUMMARY_MD"
}

on_error() {
  local code=$?
  trap - ERR
  set +e
  log "Pipeline failed with exit code $code"
  ensure_corpus_after_failure
  echo "failed" > "$STATUS_FILE"
  write_summary "failed"
  exit "$code"
}

preflight() {
  export HF_HOME
  mkdir -p "$HF_HOME" "$(dirname "$CORPUS_JSONL")" "$CONTAINER_CACHE_DIR"
  save_state
  echo "$RUN_TAG" > "$CURRENT_FILE"

  run_cmd_logged "$LOG_DIR/runtime-detect.log" "$GTSM" runtime-detect
  run_cmd_logged "$LOG_DIR/docker-version.log" docker version
  run_cmd_logged "$LOG_DIR/mlx-device.log" "$PYTHON" -c "import mlx.core as mx; import mlx_lm; print(mx.default_device())"
  run_cmd_logged "$LOG_DIR/train-profiles.log" "$GTSM" list-train-profiles
  grep -q "$PROFILE" "$LOG_DIR/train-profiles.log"

  run_cmd_allow_failure "huggingface-whoami" "$LOG_DIR/hf-whoami.log" "$HF_CLI" auth whoami
}

download_model() {
  export HF_HOME
  run_cmd_logged "$LOG_DIR/hf-download.log" "$HF_CLI" download "$BASE_MODEL" --repo-type model
}

sync_sources() {
  run_cmd_logged "$LOG_DIR/init-workspace.log" "$GTSM" init-workspace
  run_cmd_logged "$LOG_DIR/sync-tools-iuc.log" "$GTSM" sync-tools-iuc --ref "$TOOLS_REF"
  run_cmd_logged "$LOG_DIR/sync-galaxy-skills.log" "$GTSM" sync-galaxy-skills --ref "$GALAXY_SKILLS_REF"
  run_cmd_logged "$LOG_DIR/sync-galaxy-xsd.log" "$GTSM" sync-galaxy-xsd --ref "$GALAXY_XSD_REF"
}

extract_corpus() {
  local args=(
    extract-corpus
    --max-workers "$EXTRACT_MAX_WORKERS"
    --source-workers "$SOURCE_WORKERS"
    --output "$CORPUS_JSONL"
    --checkpoint "$CORPUS_CHECKPOINT"
    --no-fetch-docs
    --resolve-containers
    --execute-containers
    --container-runtime "$CONTAINER_RUNTIME"
    --container-cache-dir "$CONTAINER_CACHE_DIR"
    --container-prepare-workers "$CONTAINER_PREPARE_WORKERS"
    --container-probe-workers "$CONTAINER_PROBE_WORKERS"
    --container-image-timeout-seconds "$CONTAINER_IMAGE_TIMEOUT_SECONDS"
    --container-image-quarantine-seconds "$CONTAINER_IMAGE_QUARANTINE_SECONDS"
    --container-image-quarantine-file "$CONTAINER_CACHE_DIR/image-quarantine.json"
    --container-help-probe-mode "$CONTAINER_HELP_PROBE_MODE"
    --source-download-timeout-seconds "$SOURCE_DOWNLOAD_TIMEOUT_SECONDS"
    --source-download-max-bytes 0
    --status-log "$LOG_DIR/extract-corpus.status.jsonl"
    --retry-manifest "$RETRY_MANIFEST"
    --bioconda-checkout-sources
    --bioconda-ref "$BIOCONDA_REF"
    --synthesize-udt-yaml
  )
  run_cmd_logged "$LOG_DIR/extract-corpus.log" "$GTSM" "${args[@]}"
  local line_count
  line_count="$(wc -l < "$CORPUS_JSONL" | tr -d ' ')"
  log "Corpus records: $line_count"
  [[ "$line_count" -gt 0 ]]
}

export_corpus() {
  [[ -f "$CORPUS_JSONL" ]]
  mkdir -p "$CORPUS_EXPORT_DIR"
  local records exported_corpus exported_checkpoint
  records="$(wc -l < "$CORPUS_JSONL" | tr -d ' ')"
  [[ "$records" -gt 0 ]]

  exported_corpus="$CORPUS_EXPORT_DIR/tools-iuc-corpus.jsonl"
  exported_checkpoint="$CORPUS_EXPORT_DIR/tools-iuc-corpus.checkpoint"
  cp "$CORPUS_JSONL" "$exported_corpus"
  [[ -f "$CORPUS_CHECKPOINT" ]] && cp "$CORPUS_CHECKPOINT" "$exported_checkpoint"
  [[ -f "$RETRY_MANIFEST" ]] && cp "$RETRY_MANIFEST" "$CORPUS_EXPORT_DIR/tools-iuc-corpus.retry-manifest.json"
  records="$(wc -l < "$exported_corpus" | tr -d ' ')"

  run_cmd_allow_failure "rebuild-execution-report" "$LOG_DIR/rebuild-execution-report.log" \
    "$GTSM" rebuild-execution-report \
      --corpus-jsonl "$exported_corpus" \
      --output "$CORPUS_EXPORT_DIR/tools-iuc-corpus.execution.json"

  if [[ -f "$CORPUS_EXPORT_DIR/tools-iuc-corpus.execution.json" ]]; then
    local diagnose_args=(
      diagnose-corpus
      --execution-report "$CORPUS_EXPORT_DIR/tools-iuc-corpus.execution.json"
      --corpus-jsonl "$exported_corpus"
      --diagnostics-dir "$CORPUS_EXPORT_DIR/diagnostics"
      --sample-limit 250
    )
    [[ -f "$exported_checkpoint" ]] && diagnose_args+=(--checkpoint "$exported_checkpoint")
    run_cmd_allow_failure "diagnose-corpus" "$LOG_DIR/diagnose-corpus.log" \
      "$GTSM" "${diagnose_args[@]}"
  fi

  {
    echo "{"
    printf '  "created_at": "%s",\n' "$(timestamp)"
    printf '  "run_tag": "%s",\n' "$RUN_TAG"
    printf '  "records": %s,\n' "$records"
    printf '  "corpus_jsonl": "%s",\n' "$exported_corpus"
    printf '  "export_dir": "%s",\n' "$CORPUS_EXPORT_DIR"
    printf '  "archive": "%s"\n' "$EXPORT_DIR/corpus-artifacts.tar.gz"
    echo "}"
  } > "$CORPUS_EXPORT_DIR/corpus-export-summary.json"

  tar -czf "$EXPORT_DIR/corpus-artifacts.tar.gz" -C "$CORPUS_EXPORT_DIR" .
  log "Corpus exported: $CORPUS_EXPORT_DIR ($records records)"
}

ensure_corpus_after_failure() {
  if [[ -f "$MARKER_DIR/export_corpus.done" ]]; then
    log "Corpus export already completed."
    return 0
  fi
  if [[ -f "$CORPUS_JSONL" && "$(wc -l < "$CORPUS_JSONL" | tr -d ' ')" -gt 0 ]]; then
    log "Failure fallback: exporting existing corpus."
    if export_corpus; then
      touch "$MARKER_DIR/export_corpus.done" || true
    else
      log "Failure fallback: corpus export failed."
    fi
    return 0
  fi
  log "Failure fallback: corpus missing; attempting source sync, extraction, and export."
  sync_sources || true
  extract_corpus || true
  if [[ -f "$CORPUS_JSONL" && "$(wc -l < "$CORPUS_JSONL" | tr -d ' ')" -gt 0 ]]; then
    if export_corpus; then
      touch "$MARKER_DIR/export_corpus.done" || true
    else
      log "Failure fallback: corpus export failed after rescue extraction."
    fi
  fi
}

train_model() {
  export HF_HOME
  run_cmd_logged "$LOG_DIR/train.status.log" \
    "$GTSM" train \
      --profile "$PROFILE" \
      --dataset-manifest "$DATASET_MANIFEST" \
      --corpus-jsonl "$CORPUS_JSONL" \
      --variant-id "$VARIANT_ID" \
      --artifact-format mixed \
      --backend mlx-lm \
      --num-processes 1 \
      --training-method lora \
      --max-seq-length "$TRAIN_MAX_SEQ_LENGTH" \
      --per-device-batch-size "$TRAIN_BATCH_SIZE" \
      --gradient-accumulation-steps "$TRAIN_GRAD_ACCUM" \
      --source-context-mode "$SOURCE_CONTEXT_MODE" \
      --source-context-max-chars "$SOURCE_CONTEXT_MAX_CHARS" \
      --source-context-max-files "$SOURCE_CONTEXT_MAX_FILES" \
      --status-log "$LOG_DIR/train.status.jsonl" \
      --status-interval-seconds "$STATUS_INTERVAL_SECONDS" \
      --stream-logs \
      --log-tail-lines "$LOG_TAIL_LINES" \
      --internal-run-id "$TRAIN_RUN_ID"
  [[ -f "$VARIANT_MANIFEST" ]]
}

artifact_dir() {
  json_get "$VARIANT_MANIFEST" '.artifact_dir'
}

quantize_method() {
  case "$1" in
    q8_0) echo "Q8_0" ;;
    q6_k) echo "Q6_K" ;;
    q5_k_m) echo "Q5_K_M" ;;
    q4_k_m) echo "Q4_K_M" ;;
    *) echo "$1" ;;
  esac
}

quantizer_path() {
  if [[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
    echo "$LLAMA_CPP_DIR/build/bin/llama-quantize"
  elif [[ -x "$LLAMA_CPP_DIR/build/bin/quantize" ]]; then
    echo "$LLAMA_CPP_DIR/build/bin/quantize"
  fi
}

write_ollama_modelfile() {
  local gguf_path="$1"
  local modelfile="$EXPORT_DIR/Modelfile"
  {
    printf 'FROM %s\n' "$REPO_ROOT/$gguf_path"
    printf 'PARAMETER temperature 0.1\n'
    printf 'PARAMETER num_ctx %s\n' "$TRAIN_MAX_SEQ_LENGTH"
  } > "$modelfile"
  log "Wrote Ollama Modelfile: $modelfile"
  if command -v ollama >/dev/null 2>&1 && ollama list >/dev/null 2>&1; then
    run_cmd_allow_failure "ollama-create" "$LOG_DIR/ollama-create.log" ollama create "gtsm-mps-devstral-24b-$RUN_TAG" -f "$modelfile"
  else
    log "Ollama is not installed or not running; skipped ollama create."
  fi
}

export_artifacts() {
  local adapter_dir
  adapter_dir="$(artifact_dir)"
  [[ -n "$adapter_dir" && -f "$adapter_dir/adapters.safetensors" ]]
  printf '%s\n' "$adapter_dir" > "$EXPORT_DIR/mlx-adapter-path.txt"

  run_cmd_allow_failure "mlx-to-peft" "$LOG_DIR/convert-adapter.log" \
    "$GTSM" convert-adapter \
      --from mlx \
      --to peft \
      --base-model "$BASE_MODEL" \
      --adapter-dir "$adapter_dir" \
      --output-dir "$EXPORT_DIR/peft-adapter"

  run_cmd_allow_failure "repo-export-model" "$LOG_DIR/repo-export-model.log" \
    "$GTSM" export-model \
      --variant-id "$VARIANT_ID" \
      --format all \
      --quantizations "$EXPORT_QUANTIZATIONS"

  run_cmd_allow_failure "mlx-fuse" "$LOG_DIR/mlx-fuse.log" \
    "$PYTHON" -m mlx_lm fuse \
      --model "$BASE_MODEL" \
      --adapter-path "$adapter_dir" \
      --save-path "$EXPORT_DIR/mlx-fused"

  run_cmd_allow_failure "mlx-fuse-gguf" "$LOG_DIR/mlx-fuse-gguf.log" \
    "$PYTHON" -m mlx_lm fuse \
      --model "$BASE_MODEL" \
      --adapter-path "$adapter_dir" \
      --save-path "$EXPORT_DIR/mlx-fused-gguf" \
      --export-gguf \
      --gguf-path devstral-f16.gguf

  local f16_gguf="$EXPORT_DIR/mlx-fused-gguf/devstral-f16.gguf"
  if [[ -f "$f16_gguf" ]]; then
    mkdir -p "$EXPORT_DIR/gguf"
    run_cmd_allow_failure "llama-cpp-prepare" "$LOG_DIR/llama-cpp-prepare.log" \
      env \
        CONDA="$MAMBA" \
        ENV="$LLAMA_CPP_ENV" \
        LLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
        LLAMA_CPP_CLEAN_BUILD="${LLAMA_CPP_CLEAN_BUILD:-0}" \
        bash scripts/gtsm_llama_cpp_gguf.sh prepare

    local quantizer
    quantizer="$(quantizer_path)"
    if [[ -n "$quantizer" ]]; then
      local method
      IFS=',' read -r -a methods <<< "$EXPORT_QUANTIZATIONS"
      for method in "${methods[@]}"; do
        method="${method// /}"
        [[ -n "$method" ]] || continue
        run_cmd_allow_failure "quantize-$method" "$LOG_DIR/quantize-$method.log" \
          "$quantizer" "$f16_gguf" "$EXPORT_DIR/gguf/devstral-$method.gguf" "$(quantize_method "$method")"
      done
      if [[ -f "$EXPORT_DIR/gguf/devstral-q4_k_m.gguf" ]]; then
        write_ollama_modelfile "$EXPORT_DIR/gguf/devstral-q4_k_m.gguf"
      else
        write_ollama_modelfile "$f16_gguf"
      fi
    else
      log "No llama.cpp quantizer found after prepare; skipped quantization."
      write_ollama_modelfile "$f16_gguf"
    fi
  else
    log "MLX GGUF was not produced; skipped direct quantization and Ollama Modelfile."
  fi
}

benchmark_one() {
  local artifact_format="$1"
  local out_dir="$BENCHMARK_DIR/$artifact_format"
  mkdir -p "$out_dir"
  run_cmd_allow_failure "benchmark-$artifact_format" "$LOG_DIR/benchmark-$artifact_format.log" \
    "$GTSM" benchmark-generate \
      --corpus-jsonl "$CORPUS_JSONL" \
      --artifact-format "$artifact_format" \
      --limit "$BENCHMARK_LIMIT" \
      --provider local \
      --model-variant "$VARIANT_ID" \
      --temperature 0.1 \
      --max-tokens "$MAX_TOKENS" \
      --max-workers "$BENCHMARK_MAX_WORKERS" \
      --num-processes "$BENCHMARK_NUM_PROCESSES" \
      --min-items-per-process 1 \
      --max-prompt-help-chars "$MAX_PROMPT_HELP_CHARS" \
      --source-context-mode "$SOURCE_CONTEXT_MODE" \
      --source-context-max-chars "$SOURCE_CONTEXT_MAX_CHARS" \
      --source-context-max-files "$SOURCE_CONTEXT_MAX_FILES" \
      --repair-invalid-xml \
      --resume-existing \
      --checkpoint-records "$out_dir/checkpoint.records.jsonl" \
      --wrappers-dir "$out_dir/wrappers" \
      --generation-records "$out_dir/generation.records.json" \
      --evaluation-report "$out_dir/evaluation.summary.json" \
      --benchmark-summary "$out_dir/benchmark.summary.json" \
      --status-log "$LOG_DIR/benchmark-$artifact_format.status.jsonl"
}

benchmark_model() {
  export HF_HOME
  benchmark_one xml
  benchmark_one udt-yaml
}

run_pipeline() {
  trap on_error ERR
  echo "running" > "$STATUS_FILE"
  save_state
  run_stage preflight preflight
  run_stage sync_sources sync_sources
  run_stage extract_corpus extract_corpus
  run_stage export_corpus export_corpus
  run_stage download_model download_model
  run_stage train_model train_model
  run_stage export_artifacts export_artifacts
  run_stage benchmark_model benchmark_model
  echo "completed" > "$STATUS_FILE"
  write_summary "completed"
  log "Pipeline completed. Summary: $SUMMARY_MD"
}

launch_pipeline() {
  save_state
  echo "$RUN_TAG" > "$CURRENT_FILE"
  "$PYTHON" - \
    "$RUN_TAG" \
    "$REPO_ROOT" \
    "$SCRIPT_PATH" \
    "$DETACH_LOG" \
    "$PID_FILE" \
    "$RUN_ROOT" \
    "$RUN_LOG" \
    "$STATUS_FILE" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

run_tag, repo_root, script_path, detach_log, pid_file, run_root, run_log, status_file = sys.argv[1:]
Path(detach_log).parent.mkdir(parents=True, exist_ok=True)
handle = Path(detach_log).open("ab", buffering=0)
env = os.environ.copy()
env["RUN_TAG"] = run_tag
cmd = ["bash", script_path, "run"]
if env.get("USE_CAFFEINATE", "1").lower() not in {"0", "false", "no", "off"}:
    caffeinate = shutil.which("caffeinate")
    if caffeinate:
        cmd = [caffeinate, "-dims", *cmd]
process = subprocess.Popen(
    cmd,
    cwd=repo_root,
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=handle,
    stderr=subprocess.STDOUT,
    close_fds=True,
    start_new_session=True,
)
Path(pid_file).write_text(str(process.pid) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "run_tag": run_tag,
            "pid": process.pid,
            "run_root": run_root,
            "run_log": run_log,
            "detach_log": detach_log,
            "status_file": status_file,
        },
        indent=2,
    )
)
PY
}

print_status() {
  echo "run_tag=$RUN_TAG"
  echo "run_root=$RUN_ROOT"
  echo "status=$(cat "$STATUS_FILE" 2>/dev/null || echo unknown)"
  echo "stage=$(cat "$STAGE_FILE" 2>/dev/null || echo none)"
  echo "pid=$(cat "$PID_FILE" 2>/dev/null || echo unknown)"
  if [[ -f "$PID_FILE" ]]; then
    ps -p "$(cat "$PID_FILE")" -o pid=,stat=,etime=,command= || true
  fi
  echo "run_log=$RUN_LOG"
  echo "summary=$SUMMARY_MD"
  echo "watch_pid=$(cat "$WATCH_PID_FILE" 2>/dev/null || echo none)"
}

tail_log() {
  print_status
  echo
  tail -n "${TAIL_LINES:-80}" "$RUN_LOG" 2>/dev/null || tail -n "${TAIL_LINES:-80}" "$DETACH_LOG"
}

watch_run() {
  save_state
  echo "$$" > "$WATCH_PID_FILE"
  log "Watchdog started for RUN_TAG=$RUN_TAG" >> "$WATCH_LOG"
  while true; do
    local status stage pid
    status="$(cat "$STATUS_FILE" 2>/dev/null || echo unknown)"
    stage="$(cat "$STAGE_FILE" 2>/dev/null || echo none)"
    pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
    printf '[%s] status=%s stage=%s pid=%s\n' "$(timestamp)" "$status" "$stage" "$pid" >> "$WATCH_LOG"

    if [[ -f "$MARKER_DIR/extract_corpus.done" && ! -f "$MARKER_DIR/export_corpus.done" ]]; then
      printf '[%s] exporting completed corpus\n' "$(timestamp)" >> "$WATCH_LOG"
      if export_corpus >> "$WATCH_LOG" 2>&1; then
        touch "$MARKER_DIR/export_corpus.done"
      fi
    fi

    if [[ "$status" == "failed" ]]; then
      ensure_corpus_after_failure >> "$WATCH_LOG" 2>&1 || true
      printf '[%s] watchdog exiting after failed run\n' "$(timestamp)" >> "$WATCH_LOG"
      return 0
    fi

    if [[ "$status" == "completed" ]]; then
      if [[ ! -f "$MARKER_DIR/export_corpus.done" ]]; then
        ensure_corpus_after_failure >> "$WATCH_LOG" 2>&1 || true
      fi
      printf '[%s] watchdog exiting after completed run\n' "$(timestamp)" >> "$WATCH_LOG"
      return 0
    fi

    if [[ -n "$pid" ]] && ! ps -p "$pid" >/dev/null 2>&1; then
      printf '[%s] pipeline pid is gone while status=%s; marking failed and rescuing corpus\n' "$(timestamp)" "$status" >> "$WATCH_LOG"
      echo "failed" > "$STATUS_FILE"
      ensure_corpus_after_failure >> "$WATCH_LOG" 2>&1 || true
      return 0
    fi
    sleep "${WATCH_INTERVAL_SECONDS:-120}"
  done
}

launch_watch() {
  save_state
  "$PYTHON" - "$RUN_TAG" "$REPO_ROOT" "$SCRIPT_PATH" "$WATCH_LOG" "$WATCH_PID_FILE" <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

run_tag, repo_root, script_path, watch_log, watch_pid_file = sys.argv[1:]
Path(watch_log).parent.mkdir(parents=True, exist_ok=True)
handle = Path(watch_log).open("ab", buffering=0)
env = os.environ.copy()
env["RUN_TAG"] = run_tag
process = subprocess.Popen(
    ["bash", script_path, "watch"],
    cwd=repo_root,
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=handle,
    stderr=subprocess.STDOUT,
    close_fds=True,
    start_new_session=True,
)
Path(watch_pid_file).write_text(str(process.pid) + "\n", encoding="utf-8")
print(json.dumps({"run_tag": run_tag, "watch_pid": process.pid, "watch_log": watch_log}, indent=2))
PY
}

case "$MODE" in
  launch)
    launch_pipeline
    ;;
  run)
    run_pipeline >> "$RUN_LOG" 2>&1
    ;;
  status)
    print_status
    ;;
  tail)
    tail_log
    ;;
  export-corpus)
    export_corpus
    ;;
  watch)
    watch_run
    ;;
  launch-watch)
    launch_watch
    ;;
esac
