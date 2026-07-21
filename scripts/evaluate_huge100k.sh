#!/usr/bin/env bash
set -euo pipefail

# Default checkpoint is official LHM-MINI.  Override with:
#   --checkpoint /path/to/checkpoint.pth
# Add --save-render to write renders; omit it for metrics only.
# HuGe scale is intentionally unused. Target is 384x640 by default; pass
# --render-width 512 to the geometry validator for source 512x848.
CUDA_VISIBLE_DEVICES=0 python -m LHM.launch train.human_lrm \
  --config configs/training/human-lrm-mini-static-eval.yaml \
  --eval-only \
  --checkpoint pretrained \
  --dataset-root .datasets/HuGe100K \
  --metadata-root ../LHM_Track/train_data/huge100k_lhm \
  "$@"
