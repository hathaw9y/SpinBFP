#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model name is required}
MODEL_SLUG=$(basename "$MODEL")
EXPERIMENT_DIR=${2:-"bfp_runs/${MODEL_SLUG}"}

W_BITS=${W_BITS:-16}
A_BITS=${A_BITS:-4}
KV_BITS=${KV_BITS:-4}
BFP_GROUP_SIZE=${BFP_GROUP_SIZE:-32}
W_GPTQ_BITS=${W_GPTQ_BITS:-4}
W_GPTQ_GROUP_SIZE=${W_GPTQ_GROUP_SIZE:-32}
W_GPTQ_DAMP_PCT=${W_GPTQ_DAMP_PCT:-0.01}
W_GPTQ_CLIP_RATIO=${W_GPTQ_CLIP_RATIO:-1.0}
CALIB_SAMPLES=${CALIB_SAMPLES:-128}
SEQLEN=${SEQLEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
SEED=${SEED:-0}
MODEL_DTYPE=${MODEL_DTYPE:-auto}
ONLINE_HAD_GROUP_SIZE=${ONLINE_HAD_GROUP_SIZE:-32}
W_DOWN_HAD_GROUP_SIZE=${W_DOWN_HAD_GROUP_SIZE:-32}
QK_HAD_GROUP_SIZE=${QK_HAD_GROUP_SIZE:-32}
QK_MATMUL_BITS=${QK_MATMUL_BITS:-$KV_BITS}
AV_MATMUL_BITS=${AV_MATMUL_BITS:-$KV_BITS}
QK_MATMUL_BFP_GROUP_SIZE=${QK_MATMUL_BFP_GROUP_SIZE:-32}
AV_MATMUL_BFP_GROUP_SIZE=${AV_MATMUL_BFP_GROUP_SIZE:-32}
ROTATION_BLOCK_SIZE=${ROTATION_BLOCK_SIZE:-0}
ROTATION_INIT=${ROTATION_INIT:-random_hadamard}
OUTPUT_PATH=${OUTPUT_PATH:-"${EXPERIMENT_DIR}/BFP_MGPTQ_${W_BITS}_${A_BITS}_${KV_BITS}_W${W_GPTQ_BITS}_G${W_GPTQ_GROUP_SIZE}.bin"}

python train_bfp_gptq.py \
  --model "$MODEL" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --output-path "$OUTPUT_PATH" \
  --w-bits "$W_BITS" \
  --a-bits "$A_BITS" \
  --kv-bits "$KV_BITS" \
  --bfp-group-size "$BFP_GROUP_SIZE" \
  --dtype "$MODEL_DTYPE" \
  --online-had-group-size "$ONLINE_HAD_GROUP_SIZE" \
  --w-down-had-group-size "$W_DOWN_HAD_GROUP_SIZE" \
  --qk-had-group-size "$QK_HAD_GROUP_SIZE" \
  --qk-matmul-bits "$QK_MATMUL_BITS" \
  --qk-matmul-bfp-group-size "$QK_MATMUL_BFP_GROUP_SIZE" \
  --av-matmul-bits "$AV_MATMUL_BITS" \
  --av-matmul-bfp-group-size "$AV_MATMUL_BFP_GROUP_SIZE" \
  --rotation-block-size "$ROTATION_BLOCK_SIZE" \
  --rotation-init "$ROTATION_INIT" \
  --w-gptq-bits "$W_GPTQ_BITS" \
  --w-gptq-group-size "$W_GPTQ_GROUP_SIZE" \
  --w-gptq-damp-pct "$W_GPTQ_DAMP_PCT" \
  --w-gptq-clip-ratio "$W_GPTQ_CLIP_RATIO" \
  --calib-samples "$CALIB_SAMPLES" \
  --seqlen "$SEQLEN" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED"
