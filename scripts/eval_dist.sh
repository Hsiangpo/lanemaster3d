#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/openlane/r50_960x720.py}"
CKPT_PATH="${2:?请传入checkpoint路径}"
OUT_DIR="${3:-experiments/eval_$(date +%Y%m%d_%H%M%S)}"
GPUS="${4:-2}"
LOG_DIR="${5:-logs}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [ "${GPUS}" -gt 1 ]; then
  PYTHONUNBUFFERED=1 torchrun --nproc_per_node="${GPUS}" tools/eval.py \
    --config "${CONFIG_PATH}" \
    --ckpt "${CKPT_PATH}" \
    --out "${OUT_DIR}" \
    --gpus "${GPUS}" \
    --local-run \
    --launcher ddp | tee "${LOG_FILE}"
else
  PYTHONUNBUFFERED=1 python tools/eval.py \
    --config "${CONFIG_PATH}" \
    --ckpt "${CKPT_PATH}" \
    --out "${OUT_DIR}" \
    --gpus "${GPUS}" \
    --local-run \
    --launcher none | tee "${LOG_FILE}"
fi
