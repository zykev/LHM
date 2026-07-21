#!/usr/bin/env bash
set -euo pipefail

# Default verification is the training target crop: width=384, height=640.
# To inspect the source crop instead, add: --render-width 512
# (width=512, height=848).  Pass --dataset, --root, and --sample-id.
CUDA_VISIBLE_DEVICES=0 python tools/validate_lhm_geometry.py "$@"
