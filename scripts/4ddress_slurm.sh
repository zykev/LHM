#!/bin/bash
#SBATCH -J lhm_4ddress_overfit
#SBATCH -p visualai
#SBATCH -w research21
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00

source /home/zychen/miniconda3/etc/profile.d/conda.sh
conda activate lhm

cd /home/zychen/Documents/LHM


unset LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml