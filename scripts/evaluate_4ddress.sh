#!/usr/bin/env bash
set -euo pipefail


# Default: official LHM-MINI pretrained weights.
# Override with: --checkpoint /path/to/step_00002000.pth
# Select data with --sample-id <bare-or-grouped-id> or --sample-list <txt>.
# The evaluator runs the official infer_single_view -> animation_infer path
# with the 24 real 4D-Dress target cameras and 1:1 LHM crop (512/384).
unset LD_LIBRARY_PATH LD_PRELOAD

CUDA_VISIBLE_DEVICES=1 python -m LHM.launch infer.human_lrm \
    --infer configs/inference/human-lrm-mini-4ddress-eval.yaml \
    model_name=LHM-MINI \
    --checkpoint pretrained \
    "$@"
