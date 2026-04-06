import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import ChatTime


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(y_true, y_pred):
    diff = y_pred - y_true
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


def truncate_context(context, max_context_features):
    if context is None or max_context_features is None or max_context_features <= 0:
        return context

    parts = [part.strip() for part in context.split(";") if part.strip()]
    if not parts:
        return context

    # Always keep the leading target declaration.
    head = parts[:1]
    tail = parts[1: 1 + max_context_features]
    return "; ".join(head + tail)


def is_distributed_run():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_distributed():
    if not is_distributed_run():
        return False
    if torch.distributed.is_initialized():
        return True
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    return True


def shard_samples(samples):
    if not is_distributed_run():
        return samples
    rank = get_rank()
    world_size = get_world_size()
    return samples[rank::world_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_context_features", type=int, default=40)
    parser.add_argument("--log_interval", type=int, default=5)
    args = parser.parse_args()

    init_distributed()
    rank = get_rank()
    world_size = get_world_size()

    dataset_dir = Path(args.dataset_path)
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    split_path = dataset_dir / f"{args.split}.jsonl"
    samples = load_jsonl(split_path)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    if not samples:
        raise ValueError(f"No samples found in {split_path}")

    total_samples = len(samples)
    local_samples = shard_samples(samples)
    if not local_samples:
        raise ValueError(f"No samples assigned to rank {rank} from {split_path}")

    model = ChatTime(
        model_path=args.model_path,
        hist_len=metadata["hist_len"],
        pred_len=metadata["pred_len"],
        max_pred_len=metadata["pred_len"],
        num_samples=args.num_samples,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    per_target_true = defaultdict(list)
    per_target_pred = defaultdict(list)
    prediction_records = []
    failed_samples = 0
    sample_times = []
    total_generated_tokens = 0
    max_generated_tokens = 0
    total_invalid_sample_outputs = 0
    eval_start = time.perf_counter()

    print(
        f"Starting evaluation: total_samples={total_samples}, local_samples={len(local_samples)}, "
        f"batch_size={args.batch_size}, num_samples={args.num_samples}, "
        f"max_context_features={args.max_context_features}, split={args.split}, "
        f"rank={rank}/{world_size}"
    )

    for batch_start in range(0, len(local_samples), args.batch_size):
        sample_batch = local_samples[batch_start: batch_start + args.batch_size]
        batch_eval_start = time.perf_counter()
        hist_batch = [np.array(sample["hist_data"], dtype=float) for sample in sample_batch]
        future_batch = [np.array(sample["future_data"], dtype=float) for sample in sample_batch]
        context_batch = [
            truncate_context(sample.get("context"), args.max_context_features) for sample in sample_batch
        ]

        pred_batch = model.predict_batch(hist_batch, context_batch)
        prediction_stats_batch = getattr(model, "last_batch_prediction_stats", [{} for _ in sample_batch])
        batch_elapsed = time.perf_counter() - batch_eval_start
        sample_times.extend([batch_elapsed / max(1, len(sample_batch))] * len(sample_batch))

        for sample, future_data, pred, prediction_stats in zip(
            sample_batch,
            future_batch,
            pred_batch,
            prediction_stats_batch,
        ):
            total_generated_tokens += prediction_stats.get("total_generated_tokens", 0)
            max_generated_tokens = max(
                max_generated_tokens,
                prediction_stats.get("max_generated_tokens_per_sequence", 0),
            )
            total_invalid_sample_outputs += prediction_stats.get("invalid_sample_outputs", 0)
            pred = np.array(pred[: len(future_data)], dtype=float)
            if pred.size == 0 or np.isnan(pred).all():
                failed_samples += 1
                prediction_records.append(
                    {
                        "target_col": sample["target_col"],
                        "forecast_time": sample["forecast_time"],
                        "context_feature_limit": args.max_context_features,
                        "status": "failed",
                        "elapsed_seconds": round(batch_elapsed / max(1, len(sample_batch)), 3),
                        "prediction_stats": prediction_stats,
                        "true": future_data.tolist(),
                        "pred": [],
                    }
                )
            else:
                per_target_true[sample["target_col"]].append(future_data)
                per_target_pred[sample["target_col"]].append(pred)
                prediction_records.append(
                    {
                        "target_col": sample["target_col"],
                        "forecast_time": sample["forecast_time"],
                        "context_feature_limit": args.max_context_features,
                        "elapsed_seconds": round(batch_elapsed / max(1, len(sample_batch)), 3),
                        "prediction_stats": prediction_stats,
                        "true": future_data.tolist(),
                        "pred": pred.tolist(),
                    }
                )

        processed = min(batch_start + len(sample_batch), len(local_samples))
        if processed % args.log_interval == 0 or processed == len(local_samples):
            elapsed = time.perf_counter() - eval_start
            avg_sample_seconds = elapsed / processed
            remaining = len(local_samples) - processed
            eta_seconds = remaining * avg_sample_seconds
            avg_generated_tokens = total_generated_tokens / processed if processed else 0.0
            print(
                f"Rank {rank} evaluated {processed}/{len(local_samples)} local samples | "
                f"avg_sample={avg_sample_seconds:.2f}s | "
                f"last_batch={batch_elapsed:.2f}s | "
                f"avg_generated_tokens={avg_generated_tokens:.1f} | "
                f"max_generated_tokens={max_generated_tokens} | "
                f"invalid_outputs={total_invalid_sample_outputs} | "
                f"eta={eta_seconds / 60:.1f}m"
            )

    local_result = {
        "successful_samples": len(local_samples) - failed_samples,
        "failed_samples": failed_samples,
        "sample_times": sample_times,
        "total_generated_tokens": total_generated_tokens,
        "max_generated_tokens": max_generated_tokens,
        "invalid_sample_outputs": total_invalid_sample_outputs,
        "prediction_records": prediction_records,
        "per_target_true": {
            target: [array.tolist() for array in arrays] for target, arrays in per_target_true.items()
        },
        "per_target_pred": {
            target: [array.tolist() for array in arrays] for target, arrays in per_target_pred.items()
        },
    }

    gathered_results = [local_result]
    if is_distributed_run():
        gathered_results = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_results, local_result)

    if rank != 0:
        if is_distributed_run():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    merged_per_target_true = defaultdict(list)
    merged_per_target_pred = defaultdict(list)
    merged_prediction_records = []
    merged_sample_times = []
    successful_samples = 0
    failed_samples = 0
    total_generated_tokens = 0
    max_generated_tokens = 0
    total_invalid_sample_outputs = 0

    for result in gathered_results:
        successful_samples += result["successful_samples"]
        failed_samples += result["failed_samples"]
        merged_sample_times.extend(result["sample_times"])
        total_generated_tokens += result["total_generated_tokens"]
        max_generated_tokens = max(max_generated_tokens, result["max_generated_tokens"])
        total_invalid_sample_outputs += result["invalid_sample_outputs"]
        merged_prediction_records.extend(result["prediction_records"])
        for target_col, arrays in result["per_target_true"].items():
            merged_per_target_true[target_col].extend(np.array(arr, dtype=float) for arr in arrays)
        for target_col, arrays in result["per_target_pred"].items():
            merged_per_target_pred[target_col].extend(np.array(arr, dtype=float) for arr in arrays)

    if not merged_per_target_true:
        raise ValueError("All evaluation samples failed to produce usable predictions.")

    all_true = np.concatenate([np.concatenate(v) for v in merged_per_target_true.values()])
    all_pred = np.concatenate([np.concatenate(v) for v in merged_per_target_pred.values()])
    overall_metrics = compute_metrics(all_true, all_pred)

    per_target_metrics = {}
    for target_col in sorted(merged_per_target_true):
        target_true = np.concatenate(merged_per_target_true[target_col])
        target_pred = np.concatenate(merged_per_target_pred[target_col])
        per_target_metrics[target_col] = compute_metrics(target_true, target_pred)

    result = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "world_size": world_size,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "max_context_features": args.max_context_features,
        "avg_sample_seconds": float(np.mean(merged_sample_times)) if merged_sample_times else 0.0,
        "max_sample_seconds": float(np.max(merged_sample_times)) if merged_sample_times else 0.0,
        "avg_generated_tokens_per_sample": (
            total_generated_tokens / len(merged_sample_times) if merged_sample_times else 0.0
        ),
        "max_generated_tokens_per_sequence": max_generated_tokens,
        "invalid_sample_outputs": total_invalid_sample_outputs,
        "overall": overall_metrics,
        "per_target": per_target_metrics,
        "metadata": metadata,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

    predictions_path = output_path.with_name(output_path.stem + "_predictions.json")
    predictions_path.write_text(json.dumps(merged_prediction_records, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Evaluation finished")
    print(json.dumps(result["overall"], ensure_ascii=True, indent=2))
    print(f"Saved metrics to {output_path}")
    print(f"Saved predictions to {predictions_path}")

    if is_distributed_run():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
