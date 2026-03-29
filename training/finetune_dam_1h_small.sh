#!/usr/bin/env bash
  set -euo pipefail

  CODE_PATH="${CODE_PATH:-$(pwd)}"
  INPUT_CSV="${INPUT_CSV:-$CODE_PATH/dam_dataset/Dam/DamProcess_1h.csv}"
  MODEL_PATH="${MODEL_PATH:-$CODE_PATH/ChatTime-1-7B-Chat}"
  DATASET_PATH="${DATASET_PATH:-$CODE_PATH/dataset/dam_1h_dx_sft_small}"
  LOG_PATH="${LOG_PATH:-$CODE_PATH/logs/dam_1h_dx_small}"
  OUTPUT_PATH="${OUTPUT_PATH:-$CODE_PATH/outputs/dam_1h_dx_small}"

  HIST_LEN="${HIST_LEN:-96}"
  PRED_LEN="${PRED_LEN:-48}"
  STRIDE="${STRIDE:-24}"

  LIMIT_TARGETS="${LIMIT_TARGETS:-10}"
  LIMIT_WINDOWS_PER_TARGET="${LIMIT_WINDOWS_PER_TARGET:-100}"

  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
  LORA_RANK="${LORA_RANK:-8}"
  LORA_ALPHA="${LORA_ALPHA:-16}"
  LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
  SAVE_STEPS="${SAVE_STEPS:-100}"
  LOGGING_STEPS="${LOGGING_STEPS:-10}"
  MAX_STEPS="${MAX_STEPS:--1}"
  DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"

  EVAL_SPLIT="${EVAL_SPLIT:-validation}"
  EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-100}"
  EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-$OUTPUT_PATH/eval_${EVAL_SPLIT}.json}"

  rm -rf "$DATASET_PATH"

  python "$CODE_PATH/training/build_dam_finetune_dataset.py" \
    --input_csv "$INPUT_CSV" \
    --output_path "$DATASET_PATH" \
    --hist_len "$HIST_LEN" \
    --pred_len "$PRED_LEN" \
    --stride "$STRIDE" \
    --target_prefix dx \
    --limit_targets "$LIMIT_TARGETS" \
    --limit_windows_per_target "$LIMIT_WINDOWS_PER_TARGET"

  python "$CODE_PATH/training/finetune.py" \
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
    --max_samples "$EVAL_MAX_SAMPLES"