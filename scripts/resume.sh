#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/openlane/r50_960x720.py}"
WORK_DIR="${2:?请传入已有实验目录}"
GPUS="${3:-2}"
LOG_DIR="${4:-logs}"

LATEST_CKPT="${WORK_DIR}/latest.pth"
if [[ ! -f "${LATEST_CKPT}" ]]; then
  echo "未找到${LATEST_CKPT}"
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/resume_$(date +%Y%m%d_%H%M%S).log"

python tools/train.py \
  --config "${CONFIG_PATH}" \
  --work-dir "${WORK_DIR}" \
  --gpus "${GPUS}" | tee "${LOG_FILE}"

