#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model name is required}
EXPERIMENT_DIR=${2:?experiment dir is required}
W_BITS=${3:-4}
A_BITS=${4:-4}
KV_BITS=${5:-4}

GROUP_SIZE=${GROUP_SIZE:-32}
MODEL_DTYPE=${MODEL_DTYPE:-auto}
ROTATION_COMPUTE_DTYPE=${ROTATION_COMPUTE_DTYPE:-fp64}
DEVICE=${DEVICE:-cuda}
ONLINE_HAD_GROUP_SIZE=${ONLINE_HAD_GROUP_SIZE:--1}
W_DOWN_HAD_GROUP_SIZE=${W_DOWN_HAD_GROUP_SIZE:--1}
QK_HAD_GROUP_SIZE=${QK_HAD_GROUP_SIZE:--1}
HIST_BINS=${HIST_BINS:-128}
TOP_K=${TOP_K:-20}
MODULES=${MODULES:-all}
MAX_LAYERS=${MAX_LAYERS:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
INCLUDE_LM_HEAD=${INCLUDE_LM_HEAD:-0}
NO_ROTATE=${NO_ROTATE:-0}

EXTRA_ARGS=()
if [[ -n "$MAX_LAYERS" ]]; then
  EXTRA_ARGS+=(--max-layers "$MAX_LAYERS")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ "$INCLUDE_LM_HEAD" == "1" || "$INCLUDE_LM_HEAD" == "true" || "$INCLUDE_LM_HEAD" == "True" ]]; then
  EXTRA_ARGS+=(--include-lm-head)
fi
if [[ "$NO_ROTATE" == "1" || "$NO_ROTATE" == "true" || "$NO_ROTATE" == "True" ]]; then
  EXTRA_ARGS+=(--no-rotate)
fi

python bfp_refactor/analyze_rotated_weight_distribution.py \
  --model "$MODEL" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --w-bits "$W_BITS" \
  --a-bits "$A_BITS" \
  --kv-bits "$KV_BITS" \
  --group-size "$GROUP_SIZE" \
  --dtype "$MODEL_DTYPE" \
  --rotation-compute-dtype "$ROTATION_COMPUTE_DTYPE" \
  --device "$DEVICE" \
  --w-down-had-group-size "$W_DOWN_HAD_GROUP_SIZE" \
  --qk-had-group-size "$QK_HAD_GROUP_SIZE" \
  --hist-bins "$HIST_BINS" \
  --top-k "$TOP_K" \
  --modules "$MODULES" \
  "${EXTRA_ARGS[@]}"
