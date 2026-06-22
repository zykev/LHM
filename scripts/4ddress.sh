# 清空系统级 LD_LIBRARY_PATH 中的 CUDA/cuDNN 路径，避免与 conda 环境自带的
# pip cudnn 包（nvidia-cudnn-cu11）混用不同版本的 cuDNN 子插件库导致 segfault。
# （torch 的 pip wheel 自带完整的 CUDA/cuDNN 运行库，无需依赖系统安装。）
unset LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 python -m LHM.launch train.human_lrm \
    --config configs/training/human-lrm-mini-4ddress.yaml