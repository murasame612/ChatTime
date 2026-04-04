#!/usr/bin/env bash
set -euo pipefail

CODE_PATH="${CODE_PATH:-$(pwd)}"
MODEL_PATH="${MODEL_PATH:-$CODE_PATH/outputs/dam_1h_dx_small}"
DATASET_PATH="${DATASET_PATH:-$CODE_PATH/dataset/dam_1h_dx_sft_small}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
OUTPUT_PATH="${OUTPUT_PATH:-$MODEL_PATH/eval_${EVAL_SPLIT}_debug.json}"
GPU_ID="${GPU_ID:-${1:-}}"

MAX_SAMPLES="${MAX_SAMPLES:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_CONTEXT_FEATURES="${MAX_CONTEXT_FEATURES:-40}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

echo "Starting debug evaluation"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATASET_PATH=$DATASET_PATH"
echo "EVAL_SPLIT=$EVAL_SPLIT"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "NUM_SAMPLES=$NUM_SAMPLES"
echo "MAX_CONTEXT_FEATURES=$MAX_CONTEXT_FEATURES"
echo "LOG_INTERVAL=$LOG_INTERVAL"

python "$CODE_PATH/training/evaluate_dam_model.py" \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --split "$EVAL_SPLIT" \
  --output_path "$OUTPUT_PATH" \
  --max_samples "$MAX_SAMPLES" \
  --num_samples "$NUM_SAMPLES" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --temperature "$TEMPERATURE" \
  --max_context_features "$MAX_CONTEXT_FEATURES" \
  --log_interval "$LOG_INTERVAL"
