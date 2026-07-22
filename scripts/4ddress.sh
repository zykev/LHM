#!/usr/bin/env bash
set -euo pipefail

unset LD_LIBRARY_PATH LD_PRELOAD

CUDA_VISIBLE_DEVICES="0" python -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml \
    "$@"

# Examples:
#   bash scripts/4ddress.sh --sample-id 00127_Inner_Take10_00128
#   bash scripts/4ddress.sh --sample-list Docs/4ddress_train_subset.txt
#   bash scripts/4ddress.sh --train-list Docs/train.txt --test-list Docs/test.txt
