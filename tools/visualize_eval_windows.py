import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a few evaluation windows from ChatTime outputs, mark each window midpoint, "
            "and optionally stitch midpoint values into a sparse trajectory."
        )
    )
    parser.add_argument("--eval_path", type=str, required=True, help="Path to eval_*.json")
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="Optional path to eval_*_predictions.json. Defaults to the companion file of eval_path.",
    )
    parser.add_argument("--target_col", type=str, default=None, help="Filter to a single target column")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "validation", "test"],
        help="Optional split override when reading dataset metadata",
    )
    parser.add_argument(
        "--num_windows",
        type=int,
        default=6,
        help="How many windows to visualize when window_indices is not specified",
    )
    parser.add_argument(
        "--window_indices",
        type=str,
        default=None,
        help="Comma-separated indices after filtering/sorting, for example: 0,5,10",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="even",
        choices=["even", "first", "last"],
        help="How to choose windows when window_indices is not specified",
    )
    parser.add_argument(
        "--midpoint_mode",
        type=str,
        default="center",
        choices=["center", "left"],
        help="How to define the midpoint index for each prediction window",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output png path. Defaults to <eval_dir>/<target>_window_view.png",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="16,7",
        help="Figure size as width,height",
    )
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_predictions_path(eval_path, predictions_path):
    if predictions_path:
        return Path(predictions_path)
    eval_path = Path(eval_path)
    return eval_path.with_name(eval_path.stem + "_predictions.json")


def parse_figsize(figsize_text):
    width_text, height_text = [part.strip() for part in figsize_text.split(",", 1)]
    return float(width_text), float(height_text)


def parse_window_indices(text):
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def infer_dataset_split(eval_result, split_override):
    return split_override or eval_result.get("split") or "validation"


def load_dataset_records(dataset_path, split_name):
    dataset_file = Path(dataset_path) / f"{split_name}.jsonl"
    rows = []
    with dataset_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def midpoint_offset(pred_len, midpoint_mode):
    if midpoint_mode == "left":
        return pred_len // 2
    return (pred_len - 1) / 2.0


def select_indices(total, requested_indices, count, selection):
    if total <= 0:
        return []
    if requested_indices is not None:
        valid = [idx for idx in requested_indices if 0 <= idx < total]
        return sorted(dict.fromkeys(valid))

    count = max(1, min(count, total))
    if selection == "first":
        return list(range(count))
    if selection == "last":
        return list(range(total - count, total))
    if count == 1:
        return [total // 2]
    raw = np.linspace(0, total - 1, num=count)
    return sorted(dict.fromkeys(int(round(x)) for x in raw))


def build_records(eval_result, prediction_records, split_override=None):
    dataset_path = eval_result.get("dataset_path")
    metadata = eval_result.get("metadata", {})
    split_name = infer_dataset_split(eval_result, split_override)
    dataset_rows = load_dataset_records(dataset_path, split_name)

    keyed_dataset = {}
    for row in dataset_rows:
        key = (row["target_col"], row["forecast_time"])
        keyed_dataset[key] = row

    enriched = []
    for record in prediction_records:
        if record.get("status") == "failed" or not record.get("pred"):
            continue
        key = (record["target_col"], record["forecast_time"])
        dataset_row = keyed_dataset.get(key)
        if dataset_row is None:
            continue
        enriched.append(
            {
                "target_col": record["target_col"],
                "forecast_time": pd.to_datetime(record["forecast_time"]),
                "history_start_time": pd.to_datetime(dataset_row["history_start_time"]),
                "history_end_time": pd.to_datetime(dataset_row["history_end_time"]),
                "hist_data": np.asarray(dataset_row["hist_data"], dtype=float),
                "true": np.asarray(record["true"], dtype=float),
                "pred": np.asarray(record["pred"], dtype=float),
            }
        )

    enriched.sort(key=lambda row: (row["target_col"], row["forecast_time"]))
    return enriched, metadata


def plot_windows(records, metadata, output_path, figsize, midpoint_mode):
    hist_len = int(metadata["hist_len"])
    pred_len = int(metadata["pred_len"])
    midpoint = midpoint_offset(pred_len, midpoint_mode)

    fig, (ax_windows, ax_midpoints) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    midpoint_times = []
    midpoint_true_values = []
    midpoint_pred_values = []

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(records))))
    for idx, (record, color) in enumerate(zip(records, colors)):
        hist_x = pd.date_range(
            start=record["history_start_time"],
            end=record["history_end_time"],
            periods=len(record["hist_data"]),
        )
        pred_x = pd.date_range(
            start=record["forecast_time"],
            periods=len(record["true"]),
            freq=record["forecast_time"] - record["history_end_time"],
        )
        mid_time = pred_x[0] + (pred_x[1] - pred_x[0]) * midpoint if len(pred_x) > 1 else pred_x[0]
        mid_idx = min(int(math.floor(midpoint)), len(record["true"]) - 1)

        label_prefix = f"{record['target_col']}[{idx}]"
        ax_windows.plot(hist_x, record["hist_data"], color=color, alpha=0.35, linewidth=1.0)
        ax_windows.plot(pred_x, record["true"], color=color, linewidth=2.0, label=f"{label_prefix} true")
        ax_windows.plot(pred_x, record["pred"], color=color, linestyle="--", linewidth=2.0, label=f"{label_prefix} pred")
        ax_windows.scatter([mid_time], [record["true"][mid_idx]], color=color, s=45, marker="o")
        ax_windows.scatter([mid_time], [record["pred"][mid_idx]], color=color, s=45, marker="x")
        ax_windows.axvline(mid_time, color=color, alpha=0.15, linewidth=1.0)

        midpoint_times.append(mid_time)
        midpoint_true_values.append(record["true"][mid_idx])
        midpoint_pred_values.append(record["pred"][mid_idx])

    ax_windows.set_title("Selected Forecast Windows With Midpoint Markers")
    ax_windows.set_ylabel("Value")
    ax_windows.grid(alpha=0.25)
    ax_windows.legend(ncol=2, fontsize=8)

    ax_midpoints.plot(midpoint_times, midpoint_true_values, color="#111111", marker="o", linewidth=2.0, label="midpoint true")
    ax_midpoints.plot(midpoint_times, midpoint_pred_values, color="#d97706", marker="o", linewidth=2.0, linestyle="--", label="midpoint pred")
    ax_midpoints.set_title("Sparse Midpoint Stitch")
    ax_midpoints.set_xlabel("Time")
    ax_midpoints.set_ylabel("Midpoint Value")
    ax_midpoints.grid(alpha=0.25)
    ax_midpoints.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    eval_result = load_json(args.eval_path)
    predictions_path = resolve_predictions_path(args.eval_path, args.predictions_path)
    prediction_records = load_json(predictions_path)

    split_name = infer_dataset_split(eval_result, args.split)
    records, metadata = build_records(eval_result, prediction_records, split_override=args.split)
    if args.target_col:
        records = [row for row in records if row["target_col"] == args.target_col]

    if not records:
        raise ValueError("No successful records matched the requested filters.")

    selected_indices = select_indices(
        total=len(records),
        requested_indices=parse_window_indices(args.window_indices),
        count=args.num_windows,
        selection=args.selection,
    )
    selected_records = [records[idx] for idx in selected_indices]

    target_name = args.target_col or "mixed_targets"
    default_name = f"{target_name}_{split_name}_window_view.png"
    output_path = Path(args.output_path) if args.output_path else Path(args.eval_path).with_name(default_name)

    plot_windows(
        records=selected_records,
        metadata=metadata,
        output_path=output_path,
        figsize=parse_figsize(args.figsize),
        midpoint_mode=args.midpoint_mode,
    )

    print(f"Loaded {len(records)} filtered successful windows")
    print(f"Selected indices: {selected_indices}")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
