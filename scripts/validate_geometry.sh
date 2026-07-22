#!/usr/bin/env bash
unset LD_LIBRARY_PATH LD_PRELOAD

# Default verification is the training target crop: width=384, height=640.
# To inspect the source crop instead, add: --render-width 512
# (width=512, height=848).
CUDA_VISIBLE_DEVICES=0 python tools/validate_lhm_geometry.py \
    --dataset 4ddress \
    --root .datasets/4d-dress \
    --sample-id 00127_Inner/00127_Inner_Take10_00128 \
    --output-dir ./outputs/geometry_4ddress \
    "$@"
  