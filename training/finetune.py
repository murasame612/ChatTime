import argparse
import inspect
import numbers
import os
import sys
from pathlib import Path
from types import MethodType

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import TrainingArguments, LlamaTokenizer
from trl import SFTTrainer


def _infer_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def apply_training_step_compat_patch(trainer):
    """Handle scalar num_items_in_batch for newer Trainer call paths."""
    original_training_step = trainer.training_step
    training_step_signature = inspect.signature(original_training_step)
    if "num_items_in_batch" not in training_step_signature.parameters:
        return False

    def wrapped_training_step(self, *args, **kwargs):
        args = list(args)
        model = args[0] if args else kwargs.get("model", self.model)
        device = _infer_model_device(model)

        if len(args) >= 3 and isinstance(args[2], numbers.Real) and not torch.is_tensor(args[2]):
            args[2] = torch.tensor(args[2], device=device, dtype=torch.float32)

        value = kwargs.get("num_items_in_batch")
        if isinstance(value, numbers.Real) and not torch.is_tensor(value):
            kwargs["num_items_in_batch"] = torch.tensor(value, device=device, dtype=torch.float32)

        return original_training_step(*args, **kwargs)

    trainer.training_step = MethodType(wrapped_training_step, trainer)
    return True



def load_train_dataset(dataset_path):
    path = Path(dataset_path)

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(path), split="train")
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        raise ValueError(f"Unsupported dataset file format: {path}")

    if path.is_dir():
        # Support datasets saved by Dataset.save_to_disk / DatasetDict.save_to_disk.
        if (path / "dataset_info.json").exists() or (path / "state.json").exists() or (path / "dataset_dict.json").exists():
            dataset = load_from_disk(str(path))
            if isinstance(dataset, DatasetDict):
                if "train" not in dataset:
                    raise ValueError(f"No 'train' split found in dataset directory: {path}")
                return dataset["train"]
            return dataset

        # Support directories that contain train/validation jsonl files.
        train_jsonl = path / "train.jsonl"
        train_json = path / "train.json"
        train_csv = path / "train.csv"
        train_parquet = path / "train.parquet"
        if train_jsonl.exists():
            return load_dataset("json", data_files=str(train_jsonl), split="train")
        if train_json.exists():
            return load_dataset("json", data_files=str(train_json), split="train")
        if train_csv.exists():
            return load_dataset("csv", data_files=str(train_csv), split="train")
        if train_parquet.exists():
            return load_dataset("parquet", data_files=str(train_parquet), split="train")

    return load_dataset(dataset_path, split="train")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_path", type=str, required=True, default=None)
    parser.add_argument("--model_path", type=str, required=True, default=None)
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--log_path", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.00)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=None)

    args = parser.parse_args()

    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_id was provided, but CUDA is not available.")
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU cuda:{args.gpu_id}")

    sys.path.append(args.code_path)

    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"\nVocabulary number: {len(tokenizer.get_vocab())}\n")

    EOS_TOKEN = tokenizer.eos_token

    # load model
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # add lora to llama model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        # modules_to_save=["embed_tokens", "lm_head", ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
    )


    print(f"\nLoading dataset in {args.dataset_path}")
    dataset = load_train_dataset(args.dataset_path)
    dataset = dataset.map(
        lambda example: {"text": example["text"] + EOS_TOKEN},
        num_proc=1,
        desc="Append EOS token",
    )
    print(f"Dataset example: \n{dataset[0]['text']}\n")

    # train model
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        learning_rate=2e-4,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        save_total_limit=1,
        logging_first_step=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.random_seed,
        output_dir=args.log_path,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    )

    trainer_signature = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }

    optional_trainer_kwargs = {
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "dataset_num_proc": args.dataset_num_proc,
        "packing": False,
    }

    for key, value in optional_trainer_kwargs.items():
        if key in trainer_signature.parameters:
            trainer_kwargs[key] = value

    # TRL changed `tokenizer` to `processing_class` in newer versions.
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    if apply_training_step_compat_patch(trainer):
        print("Applied training_step compatibility patch for scalar num_items_in_batch.")

    # title Show current memory stats
    current_device = torch.cuda.current_device()
    gpu_stats = torch.cuda.get_device_properties(current_device)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = cuda:{current_device} ({gpu_stats.name}). Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.\n")

    trainer_stats = trainer.train()

    # title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.\n")

    # save model and tokenizer
    model.save_pretrained_merged(args.output_path, tokenizer)
