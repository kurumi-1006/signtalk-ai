# Edge AI model artifacts

Model binaries are normally kept out of Git. The deployable VSL-30 V4.3
artifacts are an intentional exception so a checkout can run the selected
30-gloss model without an additional download.

## Multi-VSL MViT-v2 (primary)

`multi_vsl_wacv_2025` contains the Multi-VSL WACV 2025 checkpoints. The
primary adapter uses MViT-v2-S one-view, with 16 RGB frames resized to 224×224
and ImageNet normalization. This is the model selected by `ACTIVE_MODEL_ID`.

V6.2 remains available as a rollback model. Its files live in:

```text
services/edge-ai/models/v6_2/
├── model_v6_2_fp32.onnx
├── labels.json
├── deployment_config.json
└── final_metrics.json
```

The V6.2 ONNX model requires four batch-one inputs:

```text
legacy_features: [1, 40, 456] float32
anchor_features: [1, 40, 456] float32
mask:            [1, 40] bool
video:           [1, 3, 16, 112, 112] float32
```

`V6Predictor` detects the ONNX input contract automatically, so the original
three-input V6 ONNX can still be used as a rollback by changing `MODEL_PATH`
and `LABELS_PATH`.

## Switching models for testing

The Edge service exposes a local model registry with Multi-VSL as the primary
model and V6.2 as a rollback.

```text
GET  /models
POST /models/multi_vsl_mvit_v2_1000/activate
POST /models/v6_2/activate
```

Set `ACTIVE_MODEL_ID` in `.env` to choose the startup default. To register a
future model, add its artifacts under `models/<id>/` and one definition in
`src/inference/model_registry.py`, with a predictor adapter for its input
contract.

## VSL low-shot metric encoder

`vsl_metric_lowshot` is the metric-learning model trained from the supplied
Kaggle notebooks. It is a retrieval model, not a softmax classifier. At runtime
the Edge service performs the exact training-time media pipeline:

1. MediaPipe Tasks `HolisticLandmarker` extracts 33 pose, 21 left-hand, 21
   right-hand and the fixed 40 face landmarks.
2. The 115-point sequence is shoulder-centred, shoulder-scale-normalized,
   rotation-normalized and resampled to `[48, 115, 4]` (`x, y, z, visible`).
3. The PyTorch encoder produces a 256-D embedding; cosine retrieval selects
   Top-7 stabilized prototypes, then mask-aware DTW reranks them to Top-3
   Vietnamese glosses.
4. The model's calibrated **pre-DTW** cosine and margin gate marks unmatched clips as
   `unknown`, which becomes `accepted: false` in the standard Edge response.

Required local artifacts (ignored by Git) are:

```text
models/vsl_metric_lowshot/
  best_vsl_metric_encoder.pt
  prototypes_stabilized.npz
  reference_sequences.npz
  inference_config.json
  labels.csv
  holistic_landmarker.task
```

Set `ACTIVE_MODEL_ID=vsl_metric_lowshot` and start with
`python -m src.main --serve`. Use `GET /health` to confirm the active model.

## VSL-30 keypoint classifier

`vsl30_keypoint_classifier` is the supplied 30-gloss softmax classifier. It
keeps the public `/predict` input contract unchanged: an MP4/WebM/MOV clip or
4--60 JPEG frames. The adapter extracts MediaPipe Holistic pose and hand
landmarks, shoulder-normalizes them, and resamples to the checkpoint's exact
`[1, 48, 75, 4]` float32 input (`33 pose + 21 left hand + 21 right hand`,
`x/y/z/visibility`). It returns the normal recognition event with the model's
Vietnamese class labels, top-3 probabilities, and a probability margin.

Required local artifacts (ignored by Git):

```text
models/vsl30_keypoint_classifier/
  best_vsl30_keypoint.pt
  label_map.json
```

It shares `models/vsl_metric_lowshot/holistic_landmarker.task` with the
existing keypoint pipeline. Set `ACTIVE_MODEL_ID=vsl30_keypoint_classifier`
to make it the startup model, or select it through `POST /models/vsl30_keypoint_classifier/activate`.

For V6.2, one MediaPipe pass creates the raw 76-landmark sequence. The predictor
derives both the protected legacy preprocessing and the anchor/bounded-gap
preprocessing from that same sequence.

## VSL-30 V4.3 (default)

`vsl30_v4_3` is the current default 30-gloss keypoint classifier. It uses the
same MediaPipe preprocessing and ONNX input contract as
`vsl30_keypoint_classifier`: `float32 [1, 48, 75, 4]` with 33 pose landmarks,
21 landmarks per hand, and `x/y/z/visibility` channels. The included ONNX
export has been verified against its PyTorch checkpoint.

The committed deployment bundle is:

```text
models/vsl30_v4_3/
  best_vsl30_v4_3.pt
  vsl30_v4_3_main.onnx
  label_map.json
  deployment_config.json
  export_report.json
  summary.json
```

Set `ACTIVE_MODEL_ID=vsl30_v4_3` (the default) or activate it at runtime with
`POST /models/vsl30_v4_3/activate`. The training and export source is saved in
`notebooks/VSL30_V4_3_RUN_ALL_TRAIN_EXPORT.ipynb`.
