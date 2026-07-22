#!/usr/bin/env bash
unset LD_LIBRARY_PATH LD_PRELOAD

# Default verification is the same square 384x384 crop used by the 4D-Dress
# training and inference-evaluation paths. Override --aspect-hw only when
# inspecting a legacy 5:3 preprocessing result.
PYTHONPATH="${PYTHONPATH:-}:." CUDA_VISIBLE_DEVICES=0 python tools/validate_lhm_geometry.py \
    --dataset 4ddress \
    --root .datasets/4d-dress \
    --sample-id 00127_Inner/00127_Inner_Take10_00128 \
    --aspect-hw 1.0 \
    --output-dir ./outputs/geometry_4ddress_square \
    "$@"
  
