# SignTalk Full Stack for UNO Q

Import this ZIP in Arduino App Lab, then press **Run**. App Lab builds and
starts the custom `signbridge-stack` Docker Brick:

- Edge AI upload API: `http://UNO_Q_IP:8082/predict`

The first run downloads/builds Docker images and can take several minutes.
The existing React web frontend must use `EXPO_PUBLIC_EDGE_AI_URL` set to
`http://UNO_Q_IP:8082`. This slim edition contains recognition only: no
database, NestJS API, Socket.IO, authentication, or event publishing.

The ARM-safe edition uses the classic MediaPipe Holistic graph so that frames
with no detected hand or face return an empty masked feature instead of
terminating the Edge AI process.

## V18 ONNX CPU

VSL-30 runs through ONNX Runtime CPU. This package intentionally excludes PyTorch and CUDA dependencies, avoiding multi-hundred-MB Triton/CUDA downloads on Arduino UNO Q.
