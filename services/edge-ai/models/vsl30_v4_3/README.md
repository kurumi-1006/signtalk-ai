# VSL-30 V4.3 deployable model

This directory contains the V4.3 deployment export for 30 Vietnamese sign
glosses. The Edge AI model registry exposes it as `vsl30_v4_3` and runs
`vsl30_v4_3_main.onnx` through the existing CPU-only VSL-30 ONNX predictor.

Input: `float32 [batch, 48, 75, 4]` named `keypoints` (the Edge service submits
one clip at a time).
Output: `logits` for the 30 labels in `label_map.json`.

`deployment_config.json` records deployment settings, `export_report.json`
records artifact checksums and export equivalence, and `summary.json` contains
training/validation metadata. The notebook that trains and exports this bundle
is `notebooks/VSL30_V4_3_RUN_ALL_TRAIN_EXPORT.ipynb`.
