#!/usr/bin/env bash
set -euo pipefail


# Default: official LHM-MINI pretrained weights, evaluated in FP32.
# Override with: --checkpoint /path/to/step_00002000.pth
# Optional:      --save-render --output-dir ./outputs/4ddress_eval
# Geometry validator default: target crop is 384x640.  Use --render-width 512
# there to validate the source crop at 512x848.
CUDA_VISIBLE_DEVICES="0" python -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml \
    --eval-only \
    --mixed-precision no \
    --checkpoint pretrained \
    "$@"
