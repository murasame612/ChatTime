  #!/usr/bin/env bash
  set -euo pipefail

  CODE_PATH="${CODE_PATH:-$(pwd)}"
  INPUT_CSV="${INPUT_CSV:-$CODE_PATH/dam_dataset/Dam/DamProcess_1h.csv}"
  MODEL_PATH="${MODEL_PATH:-$CODE_PATH/ChatTime-1-7B-Chat}"
  DATASET_PATH="${DATASET_PATH:-$CODE_PATH/dataset/dam_1h_dx_sft_small}"
  LOG_PATH="${LOG_PATH:-$CODE_PATH/logs/dam_1h_dx_small}"
  OUTPUT_PATH="${OUTPUT_PATH:-$CODE_PATH/outputs/dam_1h_dx_small}"

  python "$CODE_PATH/training/build_dam_finetune_dataset.py" \
    --input_csv "$INPUT_CSV" \
    --output_path "$DATASET_PATH" \
    --hist_len 96 \
    --pred_len 48 \
    --stride 24 \
    --target_prefix dx \
    --limit_targets 10 \
    --limit_windows_per_target 100

  python "$CODE_PATH/training/finetune.py" \
    --code_path "$CODE_PATH" \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --log_path "$LOG_PATH" \
    --output_path "$OUTPUT_PATH" \
    --max_seq_length 1024 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --logging_steps 10 \
    --dataset_num_proc 1 \
    --load_in_4bit