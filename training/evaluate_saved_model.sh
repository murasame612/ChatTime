#!/usr/bin/env bash
set -euo pipefail

CODE_PATH="${CODE_PATH:-$(pwd)}"
MODEL_DIR_INPUT="${MODEL_DIR_INPUT:-${1:-}}"
DATASET_PATH="${DATASET_PATH:-$CODE_PATH/dataset/dam_1h_dx_sft}"
SPLIT="${SPLIT:-${2:-test}}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_CONTEXT_FEATURES="${MAX_CONTEXT_FEATURES:-40}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
GPU_IDS="${GPU_IDS:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [ -z "$MODEL_DIR_INPUT" ]; then
  echo "Usage:"
  echo "  bash training/evaluate_saved_model.sh <model_or_log_dir> [split]"
  echo
  echo "Examples:"
  echo "  bash training/evaluate_saved_model.sh outputs/dam_1h_dx_multigpu test"
  echo "  bash training/evaluate_saved_model.sh logs/dam_1h_dx_multigpu validation"
  exit 1
fi

if [ ! -d "$MODEL_DIR_INPUT" ]; then
  echo "Input directory does not exist: $MODEL_DIR_INPUT" >&2
  exit 1
fi

resolve_model_dir() {
  local candidate="$1"

  if [ -f "$candidate/config.json" ] || [ -f "$candidate/adapter_config.json" ]; then
    echo "$candidate"
    return 0
  fi

  local checkpoint_dir=""
  checkpoint_dir="$(find "$candidate" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [ -n "$checkpoint_dir" ]; then
    echo "$checkpoint_dir"
    return 0
  fi

  return 1
}

MODEL_PATH="$(resolve_model_dir "$MODEL_DIR_INPUT" || true)"
if [ -z "$MODEL_PATH" ]; then
  echo "Could not resolve a loadable model directory from: $MODEL_DIR_INPUT" >&2
  echo "Expected one of:" >&2
  echo "  1. a merged model directory containing config/tokenizer files" >&2
  echo "  2. a log directory containing checkpoint-* subdirectories" >&2
  exit 1
fi

INPUT_BASENAME="$(basename "$MODEL_DIR_INPUT")"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_DIR_INPUT}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_ROOT/eval_${SPLIT}.json}"

echo "Resolved model_path=$MODEL_PATH"
echo "Using dataset_path=$DATASET_PATH"
echo "Using split=$SPLIT"
echo "Saving metrics to $OUTPUT_PATH"
echo "Using batch_size=$BATCH_SIZE"

CMD=(
  "$CODE_PATH/training/evaluate_dam_model.py"
)

EVAL_ARGS=(
  --model_path "$MODEL_PATH"
  --dataset_path "$DATASET_PATH"
  --split "$SPLIT"
  --output_path "$OUTPUT_PATH"
  --batch_size "$BATCH_SIZE"
  --num_samples "$NUM_SAMPLES"
  --top_k "$TOP_K"
  --top_p "$TOP_P"
  --temperature "$TEMPERATURE"
  --max_context_features "$MAX_CONTEXT_FEATURES"
  --log_interval "$LOG_INTERVAL"
)

if [ -n "$MAX_SAMPLES" ]; then
  EVAL_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [ -n "$GPU_IDS" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
  NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ID_ARRAY[@]}}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
fi

if [ "$NPROC_PER_NODE" -gt 1 ]; then
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "Launching torchrun with nproc_per_node=$NPROC_PER_NODE"
  torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "${CMD[@]}" \
    "${EVAL_ARGS[@]}"
else
  python "${CMD[@]}" "${EVAL_ARGS[@]}"
fi
