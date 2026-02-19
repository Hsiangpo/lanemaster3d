#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-lm3d_train}"
CONFIG_PATH="${2:-configs/openlane/r50_960x720.py}"
WORK_DIR="${3:-experiments/lm3d_exp001}"
GPUS="${4:-2}"
LOG_DIR="${5:-logs}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/tmux_train_$(date +%Y%m%d_%H%M%S).log"
RUN_CMD="cd /home/sust/zhou/lanemaster3d && PYTHONUNBUFFERED=1 bash scripts/train_dist.sh ${CONFIG_PATH} ${WORK_DIR} ${GPUS} ${LOG_DIR} 2>&1 | tee ${LOG_FILE}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "会话已存在: ${SESSION_NAME}"
  echo "可用命令: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" "${RUN_CMD}"
echo "已启动训练会话: ${SESSION_NAME}"
echo "查看日志: tail -f ${LOG_FILE}"
echo "进入会话: tmux attach -t ${SESSION_NAME}"
