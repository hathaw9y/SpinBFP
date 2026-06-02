#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model name is required}
W_BITS=${2:-4}
A_BITS=${3:-4}
KV_BITS=${4:-4}

MODEL_SLUG=$(basename "$MODEL")
OUTPUT_DIR=${OUTPUT_DIR:-"bfp_runs/${MODEL_SLUG}"}
BFP_GROUP_SIZE=${BFP_GROUP_SIZE:-32}
MODEL_DTYPE=${MODEL_DTYPE:-auto}
ROTATION_COMPUTE_DTYPE=${ROTATION_COMPUTE_DTYPE:-fp64}
ONLINE_HAD_GROUP_SIZE=${ONLINE_HAD_GROUP_SIZE:-32}
W_DOWN_HAD_GROUP_SIZE=${W_DOWN_HAD_GROUP_SIZE:-32}
QK_HAD_GROUP_SIZE=${QK_HAD_GROUP_SIZE:-32}
QK_MATMUL_BITS=${QK_MATMUL_BITS:-$KV_BITS}
AV_MATMUL_BITS=${AV_MATMUL_BITS:-$KV_BITS}
QK_MATMUL_BFP_GROUP_SIZE=${QK_MATMUL_BFP_GROUP_SIZE:-32}
AV_MATMUL_BFP_GROUP_SIZE=${AV_MATMUL_BFP_GROUP_SIZE:-32}
ROTATION_BLOCK_SIZE=${ROTATION_BLOCK_SIZE:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MAX_STEPS=${MAX_STEPS:-100}
MAX_LENGTH=${MAX_LENGTH:-2048}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
LR=${LR:-1.5}
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" train_rotation.py \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --w-bits "$W_BITS" \
  --a-bits "$A_BITS" \
  --kv-bits "$KV_BITS" \
  --bfp-group-size "$BFP_GROUP_SIZE" \
  --dtype "$MODEL_DTYPE" \
  --rotation-compute-dtype "$ROTATION_COMPUTE_DTYPE" \
  --online-had-group-size "$ONLINE_HAD_GROUP_SIZE" \
  --w-down-had-group-size "$W_DOWN_HAD_GROUP_SIZE" \
  --qk-had-group-size "$QK_HAD_GROUP_SIZE" \
  --qk-matmul-bits "$QK_MATMUL_BITS" \
  --qk-matmul-bfp-group-size "$QK_MATMUL_BFP_GROUP_SIZE" \
  --av-matmul-bits "$AV_MATMUL_BITS" \
  --av-matmul-bfp-group-size "$AV_MATMUL_BFP_GROUP_SIZE" \
  --rotation-block-size "$ROTATION_BLOCK_SIZE" \
  --max-length "$MAX_LENGTH" \
  --max-steps "$MAX_STEPS" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --learning-rate "$LR"
