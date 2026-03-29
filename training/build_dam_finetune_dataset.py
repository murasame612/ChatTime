import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


def format_scalar(value, precision):
    if pd.isna(value):
        return "Nan"
    return f"{float(value):.{precision}f}"


def build_context(sample_row, target_col, feature_cols, precision):
    parts = [f"target={target_col}"]

    for col in feature_cols:
        if col == target_col:
            continue
        parts.append(f"{col}={format_scalar(sample_row[col], precision)}")

    return "; ".join(parts)


def make_sample(df, target_col, end_idx, hist_len, pred_len, feature_cols, precision, time_col):
    hist_start = end_idx - hist_len
    pred_end = end_idx + pred_len

    target_hist = df[target_col].iloc[hist_start:end_idx].to_numpy(dtype=float)
    target_future = df[target_col].iloc[end_idx:pred_end].to_numpy(dtype=float)

    if np.isnan(target_hist).any() or np.isnan(target_future).any():
        return None

    discretizer = Discretizer()
    serializer = Serializer()

    target_full = np.concatenate([target_hist, target_future], axis=0)
    target_disc = discretizer.discretize(target_full, fit_length=hist_len)
    hist_disc = target_disc[:hist_len]
    future_disc = target_disc[hist_len:]

    context = build_context(
        sample_row=df.iloc[end_idx - 1],
        target_col=target_col,
        feature_cols=feature_cols,
        precision=precision,
    )

    prompt = getPrompt(
        flag="prediction",
        context=context,
        input=serializer.serialize(hist_disc),
        response=serializer.serialize(future_disc),
    )

    return {
        "text": prompt,
        "target_col": target_col,
        "forecast_time": str(df.iloc[end_idx][time_col]),
        "history_start_time": str(df.iloc[hist_start][time_col]),
        "history_end_time": str(df.iloc[end_idx - 1][time_col]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--hist_len", type=int, default=120)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--target_prefix", type=str, default="dx")
    parser.add_argument("--time_col", type=str, default="date")
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--limit_targets", type=int, default=None)
    parser.add_argument("--limit_windows_per_target", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.time_col not in df.columns:
        raise ValueError(f"Missing time column: {args.time_col}")

    df[args.time_col] = pd.to_datetime(df[args.time_col])
    df = df.sort_values(args.time_col).reset_index(drop=True)

    target_cols = [col for col in df.columns if col.startswith(args.target_prefix)]
    if not target_cols:
        raise ValueError(f"No target columns found with prefix '{args.target_prefix}'")

    if args.limit_targets is not None:
        target_cols = target_cols[:args.limit_targets]

    feature_cols = [col for col in df.columns if col != args.time_col]
    min_required_rows = args.hist_len + args.pred_len
    if len(df) < min_required_rows:
        raise ValueError(f"Dataset is too short. Need at least {min_required_rows} rows, got {len(df)}")

    samples = []
    for target_col in target_cols:
        per_target_count = 0
        for end_idx in range(args.hist_len, len(df) - args.pred_len + 1, args.stride):
            sample = make_sample(
                df=df,
                target_col=target_col,
                end_idx=end_idx,
                hist_len=args.hist_len,
                pred_len=args.pred_len,
                feature_cols=feature_cols,
                precision=args.precision,
                time_col=args.time_col,
            )
            if sample is None:
                continue

            samples.append(sample)
            per_target_count += 1

            if args.limit_windows_per_target is not None and per_target_count >= args.limit_windows_per_target:
                break

    if not samples:
        raise ValueError("No training samples were generated.")

    total = len(samples)
    train_end = int(total * args.train_ratio)
    valid_end = int(total * (args.train_ratio + args.valid_ratio))
    train_end = max(1, min(train_end, total))
    valid_end = max(train_end, min(valid_end, total))

    splits = {
        "train": samples[:train_end],
        "validation": samples[train_end:valid_end],
        "test": samples[valid_end:],
    }

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        with (output_path / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    metadata = {
        "input_csv": args.input_csv,
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "target_prefix": args.target_prefix,
        "context_mode": "last_observation_snapshot",
        "targets": target_cols,
        "num_samples": {split: len(rows) for split, rows in splits.items()},
    }
    with (output_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print(f"Saved dataset to {output_path}")
    print(f"Targets: {len(target_cols)}")
    print(f"Samples: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")
    print("Example:")
    print(splits["train"][0]["text"])


if __name__ == "__main__":
    main()
