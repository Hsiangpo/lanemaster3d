#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/openlane/r50_960x720.py}"
WORK_DIR="${2:-experiments/exp_$(date +%Y%m%d_%H%M%S)}"
GPUS="${3:-2}"
LOG_DIR="${4:-logs}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [ "${GPUS}" -gt 1 ]; then
  PYTHONUNBUFFERED=1 torchrun --nproc_per_node="${GPUS}" tools/train.py \
    --config "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --gpus "${GPUS}" \
    --local-run \
    --launcher ddp | tee "${LOG_FILE}"
else
  PYTHONUNBUFFERED=1 python tools/train.py \
    --config "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --gpus "${GPUS}" \
    --local-run \
    --launcher none | tee "${LOG_FILE}"
fi
