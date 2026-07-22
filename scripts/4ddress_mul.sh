#!/usr/bin/env bash
set -euo pipefail

unset LD_LIBRARY_PATH LD_PRELOAD

CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml \
    "$@"

# The same --sample-id / --sample-list / --train-list --test-list arguments
# as scripts/4ddress.sh are forwarded to the trainer.
