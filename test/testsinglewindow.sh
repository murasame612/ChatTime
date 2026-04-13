for i in {0..30}
do
  python tools/visualize_eval_windows.py \
    --eval_path outputs/dam_1h_dx_multigpu/eval_test.json \
    --target_col dx_IN1-1-10M \
    --mode single \
    --window_indices $i \
    --output_path outputs/dam_1h_dx_multigpu/eval_test_window_$i.png
done