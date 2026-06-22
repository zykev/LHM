unset LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml