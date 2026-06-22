# Quick Start

```bash
gtsm doctor
gtsm init-workspace
gtsm sync-tools-iuc --ref main
gtsm sync-galaxy-skills --ref main
gtsm sync-galaxy-xsd --ref dev
gtsm extract-corpus --max-workers 8
gtsm train --profile agentic-devstral-24b --corpus-jsonl .gtsm-cache/datasets/tools-iuc-corpus.jsonl
gtsm list-model-variants
gtsm export-model --variant-id <variant-id> --format all --quantizations q8_0,q6_k,q5_k_m,q4_k_m
gtsm generate-wrapper --tool-name my_tool --help-text-file help.txt --output my_tool.xml
```

For a Linux node with Apptainer/Singularity available, build the richer fine-tuning corpus with:

```bash
gtsm extract-corpus --max-workers 32 --resolve-containers --execute-containers --container-runtime auto
```

For remote training visibility, run `gtsm serve ...` and open `http://127.0.0.1:8765/monitor` in a browser.

## Common inspection commands

```bash
gtsm list-train-profiles
gtsm estimate-model-resources
gtsm list-promotion-policies
```

For model tradeoffs and default-selection guidance, see [Model Selection](models.md).
