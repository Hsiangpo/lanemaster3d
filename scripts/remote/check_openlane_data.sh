#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-.}"
OPENLANE_ROOT="${PROJECT_ROOT}/data/OpenLane"

echo "检查数据目录: ${OPENLANE_ROOT}"
if [[ ! -d "${OPENLANE_ROOT}" ]]; then
  echo "错误: 未找到OpenLane目录"
  exit 1
fi

required_paths=(
  "${OPENLANE_ROOT}/images"
  "${OPENLANE_ROOT}/data_lists/training.txt"
  "${OPENLANE_ROOT}/data_lists/validation.txt"
)

for item in "${required_paths[@]}"; do
  if [[ ! -e "${item}" ]]; then
    echo "错误: 缺少 ${item}"
    exit 1
  fi
done

echo "OpenLane目录检查通过"
du -sh "${OPENLANE_ROOT}" || true
