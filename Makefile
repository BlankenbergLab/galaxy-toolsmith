CONDA_EXE ?= $(HOME)/miniforge3/bin/conda
ENV_CREATE ?= $(CONDA_EXE) create
ENV_DIR ?= .conda/gtsm
PYTHON := $(ENV_DIR)/bin/python
GTSM ?= $(ENV_DIR)/bin/gtsm

MAX_WORKERS ?= 8
SOURCE_WORKERS ?= 8
TOOLS_REF ?= main
GALAXY_SKILLS_REF ?= main
GALAXY_XSD_REF ?= dev
BIOCONDA_REF ?= master
PROFILE ?= agentic-devstral-24b
CONTAINER_RUNTIME ?= auto
CONTAINER_CACHE_DIR ?= .gtsm-cache/containers
CONTAINER_SIF_EXEC_MODE ?= auto
CONTAINER_HELP_PROBE_MODE ?= exploratory
CONTAINER_PREPARE_WORKERS ?= 2
CONTAINER_PROBE_WORKERS ?= 4
CONTAINER_IMAGE_TIMEOUT_SECONDS ?= 300
CONTAINER_IMAGE_QUARANTINE_SECONDS ?= 86400
CONTAINER_IMAGE_QUARANTINE_FILE ?=
SOURCE_DOWNLOAD_TIMEOUT_SECONDS ?= 60
DOCKER_USE_SUDO ?= 0
NO_FETCH_DOCS ?= 1
STATUS_LOG ?=
CORPUS_JSONL ?= .gtsm-cache/datasets/tools-iuc-corpus.jsonl
CORPUS_CHECKPOINT ?= .gtsm-cache/datasets/tools-iuc-corpus.checkpoint
RETRY_MANIFEST ?= .gtsm-cache/datasets/tools-iuc-corpus.retry-manifest.json
SYNTHESIZE_UDT_YAML ?= 0
RESTART ?= 0
EXTRACT_RESTART_FLAG := $(if $(filter 1 true yes,$(RESTART)),--restart,)
EXTRACT_DOCKER_SUDO_FLAG := $(if $(filter 1 true yes,$(DOCKER_USE_SUDO)),--docker-use-sudo,)
EXTRACT_NO_FETCH_DOCS_FLAG := $(if $(filter 1 true yes,$(NO_FETCH_DOCS)),--no-fetch-docs,)
EXTRACT_STATUS_LOG_FLAG := $(if $(STATUS_LOG),--status-log $(STATUS_LOG),)
EXTRACT_SYNTHESIZE_UDT_FLAG := $(if $(filter 1 true yes,$(SYNTHESIZE_UDT_YAML)),--synthesize-udt-yaml,)
EXTRACT_CONTAINER_QUARANTINE_FILE_FLAG := $(if $(CONTAINER_IMAGE_QUARANTINE_FILE),--container-image-quarantine-file $(CONTAINER_IMAGE_QUARANTINE_FILE),)
GGUF_EXPORT_ENV_DIR ?= .conda/gtsm-unsloth-export
LLAMA_CPP_DIR ?= .gtsm-cache/llama.cpp
LLAMA_CPP_REF ?= master
LLAMA_CPP_CLEAN_BUILD ?= 1
GGUF_OUTTYPE ?= bf16
GGUF_QUANTIZATIONS ?= q4_k_m
GGUF_VARIANT_ID ?=
GGUF_RUN_TAG ?=
GGUF_OVERNIGHT_EXPORT_JSON ?=
GGUF_SYNC_OVERNIGHT_EXPORT ?= 1
GGUF_OLLAMA_MODEL_NAME ?= gtsm-tools-iuc-qwen25-7b-q4
GGUF_OLLAMA_FROM_QUANTIZATION ?= q4_k_m
GGUF_OLLAMA_CREATE ?= 0
PLANEMO_TEST_GALAXY_ROOT ?=
PLANEMO_TEST_INSTALL_GALAXY ?= 0
PLANEMO_TEST_TIMEOUT ?= 120
PLANEMO_TEST_ENGINE ?=
OLLAMA_TEST_GGUF ?=
OLLAMA_TEST_BIN ?=

.PHONY: env-linux install install-active doctor-active sync extract-corpus train prepare-llama-cpp-export export-gguf-llama-cpp finalize-gguf-export overnight-4xa100 overnight-4xa100-resume overnight-4xa100-status test test-optional-planemo test-optional-ollama lint docs-build clean clean-extract-corpus clean-containers clean-runs clean-models clean-cache

env-linux: $(PYTHON)

$(PYTHON):
	$(ENV_CREATE) -y -p $(ENV_DIR) -c conda-forge -c bioconda python=3.11 apptainer squashfuse libfuse3

install: env-linux
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[training,server,eval]"

install-active:
	@test -n "$$CONDA_PREFIX" || (echo "CONDA_PREFIX is not set; activate a conda env first." >&2; exit 2)
	$(CONDA_EXE) install -y -p "$$CONDA_PREFIX" -c conda-forge -c bioconda apptainer squashfuse libfuse3
	python -m pip install --upgrade pip
	python -m pip install -e ".[training,server,eval]"

doctor-active:
	@test -n "$$CONDA_PREFIX" || (echo "CONDA_PREFIX is not set; activate a conda env first." >&2; exit 2)
	@echo "CONDA_PREFIX=$$CONDA_PREFIX"
	@echo "python=$$(command -v python)"
	@python --version
	@echo "conda=$(CONDA_EXE)"
	@$(CONDA_EXE) --version
	@echo "gtsm=$$(command -v gtsm || true)"
	@gtsm --help >/dev/null
	@echo "apptainer=$$(command -v apptainer || true)"
	@apptainer --version
	@echo "squashfuse=$$(command -v squashfuse || true)"
	@echo "fusermount3=$$(command -v fusermount3 || true)"
	@echo "env_fusermount3=$$(test -x "$$CONDA_PREFIX/sbin/fusermount3" && echo "$$CONDA_PREFIX/sbin/fusermount3" || true)"
	@echo "dev_fuse=$$(test -e /dev/fuse && echo present || echo missing)"

sync:
	$(GTSM) init-workspace
	$(GTSM) sync-tools-iuc --ref $(TOOLS_REF)
	$(GTSM) sync-galaxy-skills --ref $(GALAXY_SKILLS_REF)
	$(GTSM) sync-galaxy-xsd --ref $(GALAXY_XSD_REF)

extract-corpus:
	$(GTSM) extract-corpus $(EXTRACT_RESTART_FLAG) \
		--max-workers $(MAX_WORKERS) \
		--source-workers $(SOURCE_WORKERS) \
		--output $(CORPUS_JSONL) \
		--checkpoint $(CORPUS_CHECKPOINT) \
		$(EXTRACT_NO_FETCH_DOCS_FLAG) \
		--resolve-containers \
		--execute-containers \
		--container-runtime $(CONTAINER_RUNTIME) \
		--container-cache-dir $(CONTAINER_CACHE_DIR) \
		--container-sif-exec-mode $(CONTAINER_SIF_EXEC_MODE) \
		--container-prepare-workers $(CONTAINER_PREPARE_WORKERS) \
		--container-probe-workers $(CONTAINER_PROBE_WORKERS) \
		--container-image-timeout-seconds $(CONTAINER_IMAGE_TIMEOUT_SECONDS) \
		--container-image-quarantine-seconds $(CONTAINER_IMAGE_QUARANTINE_SECONDS) \
		$(EXTRACT_CONTAINER_QUARANTINE_FILE_FLAG) \
		--source-download-timeout-seconds $(SOURCE_DOWNLOAD_TIMEOUT_SECONDS) \
		--container-help-probe-mode $(CONTAINER_HELP_PROBE_MODE) \
		$(EXTRACT_STATUS_LOG_FLAG) \
		--retry-manifest $(RETRY_MANIFEST) \
		--bioconda-checkout-sources \
		--bioconda-ref $(BIOCONDA_REF) \
		$(EXTRACT_SYNTHESIZE_UDT_FLAG) \
		$(EXTRACT_DOCKER_SUDO_FLAG)

train:
	$(GTSM) train --profile $(PROFILE) --corpus-jsonl $(CORPUS_JSONL)

prepare-llama-cpp-export:
	ENV=$(GGUF_EXPORT_ENV_DIR) \
	LLAMA_CPP_DIR=$(LLAMA_CPP_DIR) \
	LLAMA_CPP_REF=$(LLAMA_CPP_REF) \
	LLAMA_CPP_CLEAN_BUILD=$(LLAMA_CPP_CLEAN_BUILD) \
	GGUF_OUTTYPE=$(GGUF_OUTTYPE) \
	bash scripts/gtsm_llama_cpp_gguf.sh prepare

export-gguf-llama-cpp:
	@test -n "$(GGUF_VARIANT_ID)" || (echo "Set GGUF_VARIANT_ID=<variant-id>." >&2; exit 2)
	ENV=$(GGUF_EXPORT_ENV_DIR) \
	LLAMA_CPP_DIR=$(LLAMA_CPP_DIR) \
	GGUF_OUTTYPE=$(GGUF_OUTTYPE) \
	EXPORT_QUANTIZATIONS=$(GGUF_QUANTIZATIONS) \
	VARIANT_ID=$(GGUF_VARIANT_ID) \
	bash scripts/gtsm_llama_cpp_gguf.sh export

finalize-gguf-export:
	@test -n "$(GGUF_VARIANT_ID)" || (echo "Set GGUF_VARIANT_ID=<variant-id>." >&2; exit 2)
	ENV=$(GGUF_EXPORT_ENV_DIR) \
	VARIANT_ID=$(GGUF_VARIANT_ID) \
	RUN_TAG=$(GGUF_RUN_TAG) \
	OVERNIGHT_EXPORT_JSON=$(GGUF_OVERNIGHT_EXPORT_JSON) \
	SYNC_OVERNIGHT_EXPORT=$(GGUF_SYNC_OVERNIGHT_EXPORT) \
	OLLAMA_MODEL_NAME=$(GGUF_OLLAMA_MODEL_NAME) \
	OLLAMA_FROM_QUANTIZATION=$(GGUF_OLLAMA_FROM_QUANTIZATION) \
	OLLAMA_CREATE=$(GGUF_OLLAMA_CREATE) \
	bash scripts/gtsm_llama_cpp_gguf.sh finalize

overnight-4xa100:
	bash scripts/gtsm_overnight_4xa100.sh run

overnight-4xa100-resume:
	bash scripts/gtsm_overnight_4xa100.sh resume

overnight-4xa100-status:
	bash scripts/gtsm_overnight_4xa100.sh status

test:
	$(PYTHON) -m pytest -q tests/unit

test-optional-planemo:
	GTSM_TEST_LIVE_PLANEMO=1 \
	GTSM_TEST_PLANEMO_GALAXY_ROOT="$(PLANEMO_TEST_GALAXY_ROOT)" \
	GTSM_TEST_PLANEMO_INSTALL_GALAXY="$(PLANEMO_TEST_INSTALL_GALAXY)" \
	GTSM_TEST_PLANEMO_TIMEOUT="$(PLANEMO_TEST_TIMEOUT)" \
	GTSM_TEST_PLANEMO_ENGINE="$(PLANEMO_TEST_ENGINE)" \
	$(PYTHON) -m pytest -q -m planemo_live tests/integration

test-optional-ollama:
	@test -n "$(OLLAMA_TEST_GGUF)" || (echo "Set OLLAMA_TEST_GGUF=/absolute/path/model.gguf." >&2; exit 2)
	GTSM_TEST_LIVE_OLLAMA=1 \
	GTSM_TEST_OLLAMA_GGUF="$(OLLAMA_TEST_GGUF)" \
	GTSM_TEST_OLLAMA_BIN="$(OLLAMA_TEST_BIN)" \
	$(PYTHON) -m pytest -q -m ollama_live tests/integration

lint:
	$(PYTHON) -m ruff check src/galaxy_toolsmith tests --select E4,E7,E9,F

docs-build:
	$(PYTHON) -m pip install -r docs/requirements.txt
	$(PYTHON) -m mkdocs build --strict

clean: clean-extract-corpus
	rm -rf build dist *.egg-info site .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

clean-extract-corpus:
	rm -f $(CORPUS_JSONL) $(CORPUS_CHECKPOINT)
	rm -f .gtsm-cache/datasets/tools-iuc-corpus.index.json
	rm -f .gtsm-cache/datasets/tools-iuc-corpus.execution.json
	rm -rf .gtsm-cache/datasets/expanded

clean-containers:
	rm -rf $(CONTAINER_CACHE_DIR)

clean-runs:
	rm -rf .gtsm-cache/runs

clean-models:
	rm -rf .gtsm-cache/models

clean-cache:
	rm -rf .gtsm-cache
