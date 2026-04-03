import argparse
import inspect
import numbers
import os
import sys
from types import MethodType

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

from unsloth import FastLanguageModel, is_bfloat16_supported
import numpy as np
import torch
from datasets import load_dataset
from transformers import TrainingArguments, LlamaTokenizer
from trl import SFTTrainer


def _infer_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _coerce_scalar_to_tensor(value, device):
    if isinstance(value, numbers.Real) and not torch.is_tensor(value):
        return torch.tensor(value, device=device, dtype=torch.float32)
    return value


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

        if len(args) >= 3:
            args[2] = _coerce_scalar_to_tensor(args[2], device)

        value = kwargs.get("num_items_in_batch")
        kwargs["num_items_in_batch"] = _coerce_scalar_to_tensor(value, device)

        return original_training_step(*args, **kwargs)

    trainer.training_step = MethodType(wrapped_training_step, trainer)
    return True


def force_single_gpu_trainer_state(training_args):
    """These scripts are written for single-GPU finetuning even on multi-GPU hosts."""
    if getattr(training_args, "local_rank", -1) == -1 and torch.cuda.is_available():
        training_args._n_gpu = 1


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

    parser.add_argument("--low_limit", type=float, default=-1)
    parser.add_argument("--high_limit", type=float, default=1)
    parser.add_argument("--n_tokens", type=int, default=10002)
    parser.add_argument("--prec", type=int, default=4)
    parser.add_argument("--time_sep", type=str, default=" ")
    parser.add_argument("--time_flag", type=str, default="###")
    parser.add_argument("--nan_flag", type=str, default="Nan")
    parser.add_argument("--gpu_id", type=int, default=None)

    args = parser.parse_args()

    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu_id was provided, but CUDA is not available.")
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU cuda:{args.gpu_id}")

    sys.path.append(args.code_path)
    from utils.tools import Discretizer, Serializer

    # construct vocabulary
    discretizer = Discretizer(low_limit=args.low_limit, high_limit=args.high_limit, n_tokens=args.n_tokens)
    serializer = Serializer(prec=args.prec, time_sep=args.time_sep, time_flag=args.time_flag, nan_flag=args.nan_flag)

    vocabulary = np.concatenate((discretizer.centers[1:-1], [np.NaN])).reshape(-1, 1)
    vocabulary = np.array([serializer.serialize(i) for i in vocabulary])
    print(f"\nVocabulary: \n{vocabulary}\n")

    # add token to llama tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Old model pieces: {len(tokenizer.get_vocab())}")
    tokenizer.add_tokens(vocabulary.tolist())
    print(f"New model pieces: {len(tokenizer.get_vocab())}")

    EOS_TOKEN = tokenizer.eos_token

    # load model
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        resize_model_vocab=len(tokenizer.get_vocab()),
    )

    # add lora to llama model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        modules_to_save=["embed_tokens", "lm_head", ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
    )


    # load dataset
    def formatting_func(example):
        return example["text"] + EOS_TOKEN


    print(f"\nLoading dataset in {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, split="train")
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
    force_single_gpu_trainer_state(training_args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=64,
        packing=False,
        formatting_func=formatting_func,
        args=training_args,
    )
    force_single_gpu_trainer_state(trainer.args)
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
