# VSL-30 V4.3 deployable model

The Edge AI registry exposes this as `vsl30_v4_3` and runs
`vsl30_v4_3_main.onnx` through the CPU-only VSL-30 ONNX predictor.

Input: `float32 [batch, 48, 75, 4]` named `keypoints`.
Output: `logits` for the 30 labels in `label_map.json`.

`deployment_config.json` records deployment settings, `export_report.json`
records artifact checksums and export equivalence, and `summary.json` contains
training/validation metadata. The training/export source is
`notebooks/vsl30_v4_3_train_evaluate_export.ipynb`.
