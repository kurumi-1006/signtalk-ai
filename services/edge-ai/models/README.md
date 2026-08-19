# Edge AI model artifacts

This service ships with one model only: **VSL-30 V4.3**. It recognizes 30
Vietnamese sign glosses and is the sole model exposed by the Edge AI registry.

```text
models/vsl30_v4_3/
  best_vsl30_v4_3.pt
  vsl30_v4_3_main.onnx
  label_map.json
  deployment_config.json
  export_report.json
  summary.json
```

The runtime uses the CPU-only ONNX artifact. Its input is `float32 [batch, 48,
75, 4]` (`33 pose + 21 left hand + 21 right hand`, with `x/y/z/visibility`)
and its output is `logits` for the labels in `label_map.json`.

`ACTIVE_MODEL_ID` must remain `vsl30_v4_3`. The training and export source is
`notebooks/VSL30_V4_3_RUN_ALL_TRAIN_EXPORT.ipynb`.
