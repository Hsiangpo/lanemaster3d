#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] 检查Python"
python -V

echo "[2/4] 检查PyTorch与CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
PY

echo "[3/4] 检查GPU状态"
nvidia-smi

echo "[4/4] 检查关键目录"
pwd
ls -lah

echo "环境检查完成"

