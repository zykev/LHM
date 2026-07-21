#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="0" python -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml \
    "$@"
