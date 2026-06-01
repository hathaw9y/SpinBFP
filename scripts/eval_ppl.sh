#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model name is required}
NO_ROTATE=${NO_ROTATE:-0}
EXTRA_ARGS=()
if [[ "$NO_ROTATE" == "1" || "$NO_ROTATE" == "true" || "$NO_ROTATE" == "True" ]]; then
  EXTRA_ARGS+=(--no-rotate)
  if [[ $# -ge 5 ]]; then
    EXPERIMENT_DIR=${2:-}
    W_BITS=${3:-4}
    A_BITS=${4:-4}
    KV_BITS=${5:-4}
  else
    EXPERIMENT_DIR=""
    W_BITS=${2:-4}
    A_BITS=${3:-4}
    KV_BITS=${4:-4}
  fi
else
  EXPERIMENT_DIR=${2:?experiment dir is required}
  W_BITS=${3:-4}
  A_BITS=${4:-4}
  KV_BITS=${5:-4}
fi
DATASET=${DATASET:-wikitext2}
BATCH_SIZE=${BATCH_SIZE:-4}
EVAL_NSAMPLES=${EVAL_NSAMPLES:-256}
BFP_GROUP_SIZE=${BFP_GROUP_SIZE:-32}
BFP_EXPONENT_ROUNDING=${BFP_EXPONENT_ROUNDING:-floor}
MODEL_DTYPE=${MODEL_DTYPE:-auto}
ROTATION_COMPUTE_DTYPE=${ROTATION_COMPUTE_DTYPE:-fp64}
ONLINE_HAD_GROUP_SIZE=${ONLINE_HAD_GROUP_SIZE:-32}
W_DOWN_HAD_GROUP_SIZE=${W_DOWN_HAD_GROUP_SIZE:-32}
QK_HAD_GROUP_SIZE=${QK_HAD_GROUP_SIZE:-32}
QK_MATMUL_BITS=${QK_MATMUL_BITS:-$KV_BITS}
AV_MATMUL_BITS=${AV_MATMUL_BITS:-$KV_BITS}
QK_MATMUL_BFP_GROUP_SIZE=${QK_MATMUL_BFP_GROUP_SIZE:-32}
AV_MATMUL_BFP_GROUP_SIZE=${AV_MATMUL_BFP_GROUP_SIZE:-32}
ROTATION_BLOCK_SIZE=${ROTATION_BLOCK_SIZE:-32}
python eval_ppl.py \
  --model "$MODEL" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --w-bits "$W_BITS" \
  --a-bits "$A_BITS" \
  --kv-bits "$KV_BITS" \
  --bfp-group-size "$BFP_GROUP_SIZE" \
  --bfp-exponent-rounding "$BFP_EXPONENT_ROUNDING" \
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
  --dataset "$DATASET" \
  --batch-size "$BATCH_SIZE" \
  --eval-nsamples "$EVAL_NSAMPLES" \
  "${EXTRA_ARGS[@]}"
