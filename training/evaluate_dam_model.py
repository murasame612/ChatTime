import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_context_features", type=int, default=40)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_path)
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    split_path = dataset_dir / f"{args.split}.jsonl"
    samples = load_jsonl(split_path)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    if not samples:
        raise ValueError(f"No samples found in {split_path}")

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

    for idx, sample in enumerate(samples, start=1):
        hist_data = np.array(sample["hist_data"], dtype=float)
        future_data = np.array(sample["future_data"], dtype=float)
        context = truncate_context(sample.get("context"), args.max_context_features)

        pred = model.predict(hist_data, context)
        pred = np.array(pred[: len(future_data)], dtype=float)

        per_target_true[sample["target_col"]].append(future_data)
        per_target_pred[sample["target_col"]].append(pred)

        prediction_records.append(
            {
                "target_col": sample["target_col"],
                "forecast_time": sample["forecast_time"],
                "context_feature_limit": args.max_context_features,
                "true": future_data.tolist(),
                "pred": pred.tolist(),
            }
        )

        if idx % 20 == 0:
            print(f"Evaluated {idx}/{len(samples)} samples")

    all_true = np.concatenate([np.concatenate(v) for v in per_target_true.values()])
    all_pred = np.concatenate([np.concatenate(v) for v in per_target_pred.values()])
    overall_metrics = compute_metrics(all_true, all_pred)

    per_target_metrics = {}
    for target_col in sorted(per_target_true):
        target_true = np.concatenate(per_target_true[target_col])
        target_pred = np.concatenate(per_target_pred[target_col])
        per_target_metrics[target_col] = compute_metrics(target_true, target_pred)

    result = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_samples": len(samples),
        "max_context_features": args.max_context_features,
        "overall": overall_metrics,
        "per_target": per_target_metrics,
        "metadata": metadata,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

    predictions_path = output_path.with_name(output_path.stem + "_predictions.json")
    predictions_path.write_text(json.dumps(prediction_records, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Evaluation finished")
    print(json.dumps(result["overall"], ensure_ascii=True, indent=2))
    print(f"Saved metrics to {output_path}")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
