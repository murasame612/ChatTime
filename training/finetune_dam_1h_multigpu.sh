#!/usr/bin/env bash
set -euo pipefail

CODE_PATH="${CODE_PATH:-$(pwd)}"
INPUT_CSV="${INPUT_CSV:-$CODE_PATH/dam_dataset/Dam/DamProcess_1h.csv}"
MODEL_PATH="${MODEL_PATH:-$CODE_PATH/ChatTime-1-7B-Chat}"
DATASET_PATH="${DATASET_PATH:-$CODE_PATH/dataset/dam_1h_dx_sft}"
LOG_PATH="${LOG_PATH:-$CODE_PATH/logs/dam_1h_dx_multigpu}"
OUTPUT_PATH="${OUTPUT_PATH:-$CODE_PATH/outputs/dam_1h_dx_multigpu}"

GPU_IDS="${GPU_IDS:-${1:-0,1}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

HIST_LEN="${HIST_LEN:-96}"
PRED_LEN="${PRED_LEN:-48}"
STRIDE="${STRIDE:-24}"
CONTEXT_FEATURE_SCOPE="${CONTEXT_FEATURE_SCOPE:-target_only}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MAX_STEPS="${MAX_STEPS:--1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"

EVAL_SPLIT="${EVAL_SPLIT:-validation}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-100}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-$OUTPUT_PATH/eval_${EVAL_SPLIT}.json}"
EVAL_MAX_CONTEXT_FEATURES="${EVAL_MAX_CONTEXT_FEATURES:-40}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ID_ARRAY[@]}}"
echo "Launching torchrun with nproc_per_node=$NPROC_PER_NODE"

REBUILD_DATASET=0
if [ ! -f "$DATASET_PATH/train.jsonl" ]; then
  REBUILD_DATASET=1
elif [ -f "$DATASET_PATH/metadata.json" ]; then
  CURRENT_CONTEXT_SCOPE="$(python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); data=json.loads(p.read_text(encoding="utf-8")); print(data.get("context_feature_scope", "all"))' "$DATASET_PATH/metadata.json")"
  if [ "$CURRENT_CONTEXT_SCOPE" != "$CONTEXT_FEATURE_SCOPE" ]; then
    REBUILD_DATASET=1
    echo "Rebuilding dataset because context_feature_scope changed: $CURRENT_CONTEXT_SCOPE -> $CONTEXT_FEATURE_SCOPE"
  fi
fi

if [ "$REBUILD_DATASET" -eq 1 ]; then
  rm -rf "$DATASET_PATH"
  python "$CODE_PATH/training/build_dam_finetune_dataset.py" \
    --input_csv "$INPUT_CSV" \
    --output_path "$DATASET_PATH" \
    --hist_len "$HIST_LEN" \
    --pred_len "$PRED_LEN" \
    --stride "$STRIDE" \
    --target_prefix dx \
    --context_feature_scope "$CONTEXT_FEATURE_SCOPE"
fi

torchrun \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$CODE_PATH/training/finetune.py" \
  --code_path "$CODE_PATH" \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --log_path "$LOG_PATH" \
  --output_path "$OUTPUT_PATH" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --logging_steps "$LOGGING_STEPS" \
  --max_steps "$MAX_STEPS" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --load_in_4bit

python "$CODE_PATH/training/evaluate_dam_model.py" \
  --model_path "$OUTPUT_PATH" \
  --dataset_path "$DATASET_PATH" \
  --split "$EVAL_SPLIT" \
  --output_path "$EVAL_OUTPUT_PATH" \
  --max_samples "$EVAL_MAX_SAMPLES" \
  --max_context_features "$EVAL_MAX_CONTEXT_FEATURES"
