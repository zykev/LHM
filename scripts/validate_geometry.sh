#!/usr/bin/env bash
unset LD_LIBRARY_PATH LD_PRELOAD

# Default verification is a square 384x384 crop. This script intentionally
# does not modify the LHM train/evaluation pipeline. Override --aspect-hw
# 1.6666667 to reproduce the current 5:3 training crop.
PYTHONPATH="${PYTHONPATH:-}:." CUDA_VISIBLE_DEVICES=0 python tools/validate_lhm_geometry.py \
    --dataset 4ddress \
    --root .datasets/4d-dress \
    --sample-id 00127_Inner/00127_Inner_Take10_00128 \
    --aspect-hw 1.0 \
    --output-dir ./outputs/geometry_4ddress_square \
    "$@"
  
