#!/usr/bin/env bash
set -e

# Put this script in LHM/; LHM_Track/ must be its sibling directory.
ENV_NAME="${1:-lhm}"
LHM_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACK_DIR="$(cd "${LHM_DIR}/../LHM_Track" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "${ENV_NAME}" python=3.10 pip
conda activate "${ENV_NAME}"

pip install -U pip setuptools wheel

# Sapiens and Track's SAM2 require 2.3.1; xformers 0.0.27 matches it.
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu118

conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
ln -sfn "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" "${CONDA_PREFIX}/bin/gcc"
ln -sfn "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" "${CONDA_PREFIX}/bin/g++"

# Install all remaining packages from both projects without replacing Torch.
REQ_FILE="$(mktemp)"
trap 'rm -f "$REQ_FILE"' EXIT
cat "${LHM_DIR}/requirements.txt" "${TRACK_DIR}/requirements.txt" \
    | grep -vE '^(torch|torchvision|xformers)==' \
    | grep -vE '^[[:space:]]*#|^[[:space:]]*$' \
    | sort -u > "${REQ_FILE}"
pip install -r "${REQ_FILE}"

pip uninstall -y basicsr
pip install git+https://github.com/XPixelGroup/BasicSR
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/ashawkey/diff-gaussian-rasterization/
pip install --no-build-isolation git+https://github.com/camenduru/simple-knn/
pip install --no-build-isolation -e "${TRACK_DIR}/engine/samurai/sam2"

echo "Done: conda activate ${ENV_NAME}"
