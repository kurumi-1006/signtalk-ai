# VSL-30 keypoint classifier

This package runs on the UNO Q Edge AI service, not directly on an Arduino
microcontroller. Arduino should capture/send frames or trigger a request to
`POST /predict`; the Edge service extracts landmarks and runs this checkpoint.

Model input after preprocessing: float32 `[1, 48, 75, 4]`:

- 48 uniformly sampled time steps
- 75 landmarks: 33 pose, 21 left hand, 21 right hand
- coordinates `x, y, z` plus visibility

The deployment artifact is `vsl30_keypoint_classifier.onnx`; it runs with the
existing CPU-only ONNX Runtime and does not require PyTorch or CUDA packages.
The adapter uses MediaPipe's legacy Holistic API for pose and hand landmarks,
so it does not require a `holistic_landmarker.task` file.

Set `ACTIVE_MODEL_ID=vsl30_keypoint_classifier` before starting the Edge AI
service.
