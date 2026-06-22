#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-run}"
case "$MODE" in
  run|prepare|export|finalize|status) ;;
  *)
    echo "Usage: $0 {run|prepare|export|finalize|status}" >&2
    exit 2
    ;;
esac

: "${ENV:=.conda/gtsm-unsloth-export}"
: "${CONDA:=$HOME/miniforge3/bin/conda}"
: "${LLAMA_CPP_DIR:=.gtsm-cache/llama.cpp}"
: "${LLAMA_CPP_REPO:=https://github.com/ggml-org/llama.cpp}"
: "${LLAMA_CPP_REF:=master}"
: "${LLAMA_CPP_CLEAN_BUILD:=1}"
: "${LLAMA_CPP_NO_PULL:=0}"
: "${GGUF_OUTTYPE:=bf16}"
: "${EXPORT_QUANTIZATIONS:=q4_k_m}"
: "${VARIANT_ID:=}"
: "${RUN_TAG:=}"
: "${OVERNIGHT_ROOT:=.gtsm-cache/runs/overnight}"
: "${OVERNIGHT_EXPORT_JSON:=}"
: "${SYNC_OVERNIGHT_EXPORT:=1}"
: "${OLLAMA_MODEL_NAME:=}"
: "${OLLAMA_FROM_QUANTIZATION:=q4_k_m}"
: "${OLLAMA_CREATE:=0}"

if [[ ! -x "$CONDA" ]]; then
  CONDA="$(command -v conda || true)"
fi
if [[ -z "$CONDA" || ! -x "$CONDA" ]]; then
  echo "Could not find conda. Set CONDA=/path/to/conda." >&2
  exit 2
fi

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

log() {
  printf '[llama.cpp-gguf] %s\n' "$*"
}

ensure_env() {
  if [[ ! -x "$ENV/bin/python" ]]; then
    log "Creating export environment at $ENV"
    "$CONDA" create -y -p "$ENV" -c conda-forge \
      python=3.11 pip git cmake make cxx-compiler ninja
  else
    log "Ensuring build tools are installed in $ENV"
    "$CONDA" install -y -p "$ENV" -c conda-forge \
      git cmake make cxx-compiler ninja
  fi

  "$ENV/bin/python" -m pip install --upgrade pip
  "$ENV/bin/python" -m pip install \
    numpy sentencepiece safetensors tqdm pyyaml protobuf
}

sync_llama_cpp() {
  if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
    log "Cloning llama.cpp into $LLAMA_CPP_DIR"
    "$ENV/bin/git" clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
  elif ! is_true "$LLAMA_CPP_NO_PULL"; then
    log "Updating existing llama.cpp checkout"
    "$ENV/bin/git" -C "$LLAMA_CPP_DIR" fetch --tags --prune
  fi

  if [[ "$LLAMA_CPP_REF" != "master" || ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    log "Checking out llama.cpp ref $LLAMA_CPP_REF"
    "$ENV/bin/git" -C "$LLAMA_CPP_DIR" checkout "$LLAMA_CPP_REF"
  elif ! is_true "$LLAMA_CPP_NO_PULL"; then
    "$ENV/bin/git" -C "$LLAMA_CPP_DIR" checkout master
    "$ENV/bin/git" -C "$LLAMA_CPP_DIR" pull --ff-only
  fi
}

build_llama_cpp() {
  if is_true "$LLAMA_CPP_CLEAN_BUILD"; then
    log "Removing previous llama.cpp build directory"
    rm -rf "$LLAMA_CPP_DIR/build"
  fi

  log "Configuring llama.cpp without OpenMP or CUDA"
  "$ENV/bin/cmake" -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
    -DGGML_CUDA=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release

  log "Building llama.cpp quantizer"
  if ! "$ENV/bin/cmake" --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"; then
    "$ENV/bin/cmake" --build "$LLAMA_CPP_DIR/build" --target quantize -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
  fi

  if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" && ! -x "$LLAMA_CPP_DIR/build/bin/quantize" ]]; then
    echo "No llama.cpp quantizer found under $LLAMA_CPP_DIR/build/bin." >&2
    exit 1
  fi
  if [[ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]]; then
    echo "No convert_hf_to_gguf.py found in $LLAMA_CPP_DIR." >&2
    exit 1
  fi
}

quantizer_path() {
  if [[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
    echo "$LLAMA_CPP_DIR/build/bin/llama-quantize"
  elif [[ -x "$LLAMA_CPP_DIR/build/bin/quantize" ]]; then
    echo "$LLAMA_CPP_DIR/build/bin/quantize"
  fi
}

require_llama_cpp_tools() {
  if [[ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]]; then
    echo "No convert_hf_to_gguf.py found in $LLAMA_CPP_DIR. Run prepare first." >&2
    exit 1
  fi
  if [[ -z "$(quantizer_path)" ]]; then
    echo "No llama.cpp quantizer found under $LLAMA_CPP_DIR/build/bin. Run prepare first." >&2
    exit 1
  fi
}

variant_export_json() {
  echo ".gtsm-cache/models/exports/$VARIANT_ID/export.result.json"
}

resolved_run_tag() {
  if [[ -n "$RUN_TAG" ]]; then
    echo "$RUN_TAG"
  elif [[ -f "$OVERNIGHT_ROOT/current" ]]; then
    cat "$OVERNIGHT_ROOT/current"
  fi
}

resolved_overnight_export_json() {
  if [[ -n "$OVERNIGHT_EXPORT_JSON" ]]; then
    echo "$OVERNIGHT_EXPORT_JSON"
    return 0
  fi
  local tag
  tag="$(resolved_run_tag)"
  if [[ -n "$tag" ]]; then
    echo "$OVERNIGHT_ROOT/$tag/exports/qwen7b.export.json"
  fi
}

require_variant_id() {
  if [[ -z "$VARIANT_ID" ]]; then
    echo "VARIANT_ID is required." >&2
    exit 2
  fi
}

verify_export_result() {
  require_variant_id
  local export_json
  export_json="$(variant_export_json)"
  if [[ ! -f "$export_json" ]]; then
    echo "Export result not found: $export_json" >&2
    exit 1
  fi
  "$ENV/bin/python" - "$export_json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
status = data.get("status", "")
gguf_path = Path(str(data.get("gguf_path", "")))
if status != "completed":
    raise SystemExit(f"Expected completed export status, found {status!r} in {path}")
if not gguf_path.is_file() or gguf_path.stat().st_size == 0:
    raise SystemExit(f"GGUF path is missing or empty: {gguf_path}")
print(json.dumps({
    "variant_id": data.get("variant_id", ""),
    "status": status,
    "gguf_path": str(gguf_path),
    "size_bytes": gguf_path.stat().st_size,
    "notes": data.get("notes", []),
}, indent=2))
PY
}

sync_overnight_export() {
  if ! is_true "$SYNC_OVERNIGHT_EXPORT"; then
    return 0
  fi
  local target
  target="$(resolved_overnight_export_json)"
  if [[ -z "$target" ]]; then
    log "No RUN_TAG or overnight current file found; skipping overnight export JSON refresh."
    return 0
  fi
  mkdir -p "$(dirname "$target")"
  cp "$(variant_export_json)" "$target"
  log "Updated overnight export JSON: $target"
}

prepare() {
  ensure_env
  sync_llama_cpp
  build_llama_cpp
}

export_variant() {
  require_variant_id
  if [[ ! -x "$ENV/bin/gtsm" ]]; then
    "$ENV/bin/python" -m pip install -e . --no-deps
  fi
  require_llama_cpp_tools

  log "Exporting $VARIANT_ID to GGUF via user-space llama.cpp"
  GTSM_GGUF_BACKEND=llama.cpp \
  GTSM_LLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
  GTSM_LLAMA_CPP_OUTTYPE="$GGUF_OUTTYPE" \
    "$ENV/bin/gtsm" export-model \
      --variant-id "$VARIANT_ID" \
      --format gguf \
      --quantizations "$EXPORT_QUANTIZATIONS"
}

finalize_variant() {
  require_variant_id
  if [[ ! -x "$ENV/bin/gtsm" ]]; then
    "$ENV/bin/python" -m pip install -e . --no-deps
  fi

  log "Verifying completed GGUF export for $VARIANT_ID"
  verify_export_result

  if [[ -n "$OLLAMA_MODEL_NAME" ]]; then
    log "Generating Ollama Modelfile for $OLLAMA_MODEL_NAME"
    create_flag=()
    if is_true "$OLLAMA_CREATE"; then
      create_flag=(--create)
    fi
    "$ENV/bin/gtsm" export-ollama-model \
      --variant-id "$VARIANT_ID" \
      --model-name "$OLLAMA_MODEL_NAME" \
      --from-quantization "$OLLAMA_FROM_QUANTIZATION" \
      "${create_flag[@]}"
  else
    log "OLLAMA_MODEL_NAME is empty; skipping Ollama Modelfile generation."
  fi

  sync_overnight_export
}

status() {
  echo "ENV=$ENV"
  echo "CONDA=$CONDA"
  echo "LLAMA_CPP_DIR=$LLAMA_CPP_DIR"
  echo "LLAMA_CPP_REF=$LLAMA_CPP_REF"
  echo "GGUF_OUTTYPE=$GGUF_OUTTYPE"
  echo "EXPORT_QUANTIZATIONS=$EXPORT_QUANTIZATIONS"
  echo "VARIANT_ID=$VARIANT_ID"
  echo "RUN_TAG=$(resolved_run_tag)"
  echo "OVERNIGHT_EXPORT_JSON=$(resolved_overnight_export_json)"
  echo "SYNC_OVERNIGHT_EXPORT=$SYNC_OVERNIGHT_EXPORT"
  echo "OLLAMA_MODEL_NAME=$OLLAMA_MODEL_NAME"
  echo "OLLAMA_FROM_QUANTIZATION=$OLLAMA_FROM_QUANTIZATION"
  echo "OLLAMA_CREATE=$OLLAMA_CREATE"
  echo "converter=$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
  echo "quantizer=$(quantizer_path)"
}

case "$MODE" in
  run)
    prepare
    export_variant
    ;;
  prepare)
    prepare
    ;;
  export)
    export_variant
    ;;
  finalize)
    finalize_variant
    ;;
  status)
    status
    ;;
esac
