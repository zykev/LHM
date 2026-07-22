#!/usr/bin/env bash
set -euo pipefail


# Default: official LHM-MINI pretrained weights.
# Override with: --checkpoint /path/to/step_00002000.pth
# Optional:      --save-render --output-dir ./outputs/4ddress_eval
# 4DDress uses square source/target crops: 512x512 and 384x384.
unset LD_LIBRARY_PATH LD_PRELOAD

CUDA_VISIBLE_DEVICES=1 python -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml \
    --sample-id 00127_Inner_Take10_00128 \
    --eval-only \
    --save-render \
    --checkpoint pretrained \
    "$@"
