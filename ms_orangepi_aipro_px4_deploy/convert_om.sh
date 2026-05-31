#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${DEPLOY_DIR}/artifacts/ppo_policy.mindir"
OUTPUT="${DEPLOY_DIR}/artifacts/ppo_policy"
SOC_VERSION="${SOC_VERSION:-Ascend310B1}"

if [ ! -f "${MODEL}" ]; then
  echo "MindIR not found: ${MODEL}"
  echo "Run export_mindir.py first."
  exit 1
fi

atc \
  --model="${MODEL}" \
  --framework=1 \
  --output="${OUTPUT}" \
  --input_format=ND \
  --input_shape="obs:1,135" \
  --soc_version="${SOC_VERSION}"

echo "exported: ${OUTPUT}.om"
