"""CPU-only ONNX inference adapter for the VSL-30 keypoint classifier."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from .types import Candidate, Prediction
SEQ_LEN = 48
N_POSE = 33
N_HAND = 21
N_POINTS = N_POSE + 2 * N_HAND
TRAIN_RAW_POINTS = 76
COORD_CLIP = 8.0
MIN_SHOULDER_SCALE = 0.08
# Exact 76-point extraction contract in camera.ipynb.  The raw hand points
# are interleaved and use its tip-to-MCP order, not MediaPipe's default order.
CAMERA_HAND_INDICES = (0, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17, 4, 3, 2, 1)
BODY_ANCHOR_INDICES = (0, 11, 12, 23, 24, 33)
# Keep the full duration of the gesture while benchmarking a smaller uniform
# landmark-extraction set. The model still receives its required 48 steps
# after keypoint interpolation. 48- and full-frame backups are retained.
MAX_MEDIAPIPE_FRAMES = 32


class Vsl30KeypointOnnxPredictor:
    """Runs the fixed `[1, 48, 75, 4]` VSL-30 keypoint contract on CPU."""

    def __init__(self, model_path: str, labels_path: str) -> None:
        checkpoint_path, label_path = Path(model_path), Path(labels_path)
        if not checkpoint_path.is_file() or not label_path.is_file():
            raise FileNotFoundError('VSL-30 artifacts are missing: classifier ONNX and label_map.json are required.')
        self.labels = self._load_labels(label_path)
        self.session = ort.InferenceSession(str(checkpoint_path), providers=['CPUExecutionProvider'])
        input_contract = self.session.get_inputs()[0]
        input_shape = input_contract.shape
        # V3 exports use a fixed batch dimension of 1 while V4.3 uses a
        # symbolic batch dimension. Inference always submits one clip, so the
        # three non-batch dimensions remain the compatibility contract.
        if (
            input_contract.name != 'keypoints'
            or len(input_shape) != 4
            or input_shape[1:] != [SEQ_LEN, N_POINTS, 4]
        ):
            raise ValueError(f'Unexpected VSL-30 ONNX input: {input_contract.name} {input_contract.shape}')
        self.model_name = 'VSL-30 keypoint classifier (ONNX CPU, 30 glosses)'
        self.input_names = {'video_or_jpeg_frames'}

    @staticmethod
    def _load_labels(path: Path) -> dict[int, str]:
        payload = json.loads(path.read_text(encoding='utf-8'))
        values = payload.get('idx_to_label') if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            raise ValueError('VSL-30 label_map.json must contain idx_to_label.')
        labels = {int(index): str(label).strip() for index, label in values.items()}
        if set(labels) != set(range(30)) or any(not label for label in labels.values()):
            raise ValueError('VSL-30 label_map.json must map every class 0 through 29 to a non-empty label.')
        return labels

    @staticmethod
    def _normalize_body(raw_xyz: np.ndarray) -> np.ndarray:
        """Match `normalize_body` in VSL30_Kaggle_V3_KEYPOINT_ONLY_FIXED.ipynb.

        The exported classifier was validated with this exact 75-point
        pose/left-hand/right-hand transform.  Do not use the low-shot model's
        115-point normalizer here: it has different shoulder filtering and
        visibility-resampling behaviour.
        """
        xyz = np.asarray(raw_xyz, dtype=np.float32)
        if xyz.ndim != 3 or xyz.shape[1:] != (N_POINTS, 3):
            raise ValueError(f'Expected raw landmarks [frames, {N_POINTS}, 3], got {xyz.shape}.')
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (np.linalg.norm(xyz, axis=-1) > 1e-8).astype(np.float32)[..., None]

        shoulder_ok = (mask[:, 11, 0] > 0.5) & (mask[:, 12, 0] > 0.5)
        if shoulder_ok.any():
            shoulder_centers = (xyz[shoulder_ok, 11] + xyz[shoulder_ok, 12]) / 2.0
            shoulder_distance = np.linalg.norm(
                xyz[shoulder_ok, 11, :2] - xyz[shoulder_ok, 12, :2], axis=1,
            )
            good = np.isfinite(shoulder_distance) & (shoulder_distance >= MIN_SHOULDER_SCALE)
        else:
            good = np.zeros(0, dtype=bool)

        if shoulder_ok.any() and good.any():
            center = np.median(shoulder_centers[good], axis=0)
            scale = float(np.median(shoulder_distance[good]))
            shoulder_vector = (xyz[shoulder_ok, 12, :2] - xyz[shoulder_ok, 11, :2])[good]
            theta = -float(np.median(np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])))
        else:
            visible_points = xyz[mask[..., 0] > 0.5]
            if len(visible_points):
                center = np.median(visible_points, axis=0)
                radial = np.linalg.norm(visible_points[:, :2] - center[:2], axis=1)
                radial = radial[radial > 1e-4]
                scale = float(np.median(radial)) if len(radial) else 0.25
            else:
                center = np.zeros(3, dtype=np.float32)
                scale = 0.25
            theta = 0.0

        scale = max(scale, MIN_SHOULDER_SCALE)
        centers_t = np.repeat(center[None], len(xyz), axis=0)
        if shoulder_ok.any():
            centers_t[shoulder_ok] = (xyz[shoulder_ok, 11] + xyz[shoulder_ok, 12]) / 2.0

        output = (xyz - centers_t[:, None, :]) / scale
        cosine, sine = np.cos(theta), np.sin(theta)
        rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float32)
        output[..., :2] = output[..., :2] @ rotation.T
        output = np.clip(output, -COORD_CLIP, COORD_CLIP) * mask
        return np.concatenate((output, mask), axis=-1).astype(np.float32)

    @staticmethod
    def _linear_resample(value: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
        """Match notebook `linear_resample`, including its visibility rule."""
        if len(value) == target_len:
            return value.astype(np.float32)
        if len(value) <= 1:
            return np.repeat(value[:1], target_len, axis=0).astype(np.float32)
        old_t = np.linspace(0.0, 1.0, len(value), dtype=np.float32)
        new_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        flat = value.reshape(len(value), -1)
        output = np.empty((target_len, flat.shape[1]), dtype=np.float32)
        for index in range(flat.shape[1]):
            output[:, index] = np.interp(new_t, old_t, flat[:, index])
        output = output.reshape(target_len, *value.shape[1:])
        output[..., 3] = np.clip(output[..., 3], 0.0, 1.0)
        output[..., :3] *= output[..., 3:4] > 0.2
        return output.astype(np.float32)

    @staticmethod
    def _camera_dataset_normalize(raw_76: np.ndarray) -> np.ndarray:
        """Reproduce camera.ipynb's SingleBody/SingleHandDictNormalize."""
        output = np.asarray(raw_76, dtype=np.float32).copy()
        if output.ndim != 3 or output.shape[1:] != (TRAIN_RAW_POINTS, 3):
            raise ValueError(f'Expected camera raw keypoints [frames, 76, 3], got {output.shape}.')
        for frame in output:
            body = frame[:34]
            anchor_x = [float(body[index, 0]) for index in BODY_ANCHOR_INDICES if body[index, 0] != 0]
            anchor_y = [float(body[index, 1]) for index in BODY_ANCHOR_INDICES if body[index, 1] != 0]
            if anchor_x and anchor_y:
                min_x, max_x = min(anchor_x), max(anchor_x)
                min_y, max_y = min(anchor_y), max(anchor_y)
                dx, dy = (max_x - min_x) * 1.6, (max_y - min_y) * 1.6
                if dx > 0 and dy > 0:
                    center_x, center_y = (max_x + min_x) / 2.0, (max_y + min_y) / 2.0
                    for point in body:
                        if point[0] != 0 or point[1] != 0:
                            point[0] = (point[0] - center_x) / dx
                            point[1] = (point[1] - center_y) / dy
            for side in (0, 1):
                indices = tuple(34 + 2 * index + side for index in range(N_HAND))
                hand = frame[list(indices)]
                hand_x = [float(point[0]) for point in hand if point[0] != 0]
                hand_y = [float(point[1]) for point in hand if point[1] != 0]
                if not hand_x or not hand_y:
                    continue
                min_x, max_x = min(hand_x), max(hand_x)
                min_y, max_y = min(hand_y), max(hand_y)
                dx, dy = max_x - min_x, max_y - min_y
                if dx <= 0 or dy <= 0:
                    continue
                for index in indices:
                    point = frame[index]
                    if point[0] != 0 or point[1] != 0:
                        point[0] = (point[0] - min_x) / dx - 0.5
                        point[1] = (point[1] - min_y) / dy - 0.5
        return output

    def _extract_dataset_npy(self, frames: list[np.ndarray]) -> np.ndarray:
        """Convert an uploaded video to the Kaggle extractor's `[T, 76, 3]` NPY contract."""
        # MediaPipe Tasks' HolisticLandmarker graph is not ABI-compatible with
        # every ARM build shipped by Arduino App Lab. The legacy Holistic API
        # provides the same normalized pose/hand landmarks required here, with
        # no external .task graph or TaskRunner binding.
        sequence: list[np.ndarray] = []
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            refine_face_landmarks=False,
        ) as holistic:
            for frame in frames:
                result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                def points(landmarks: object, count: int) -> np.ndarray:
                    output = np.zeros((count, 3), dtype=np.float32)
                    items = getattr(landmarks, 'landmark', landmarks) or []
                    for index, item in enumerate(items[:count]):
                        output[index] = (item.x, item.y, item.z)
                    return output

                # Match camera.ipynb: 33 pose + neck, then interleaved hands
                # in its tip-to-MCP order.  This is the source layout used to
                # create the supplied anh.npy training keypoints.
                pose = points(result.pose_landmarks, N_POSE)
                left = points(result.left_hand_landmarks, N_HAND)
                right = points(result.right_hand_landmarks, N_HAND)
                raw = np.zeros((TRAIN_RAW_POINTS, 3), dtype=np.float32)
                raw[:N_POSE] = pose
                if np.any(pose[11]) and np.any(pose[12]):
                    raw[33] = (pose[11] + pose[12]) / 2.0
                for index in range(N_HAND):
                    source_index = CAMERA_HAND_INDICES[index]
                    raw[34 + 2 * index] = left[source_index]
                    raw[35 + 2 * index] = right[source_index]
                sequence.append(raw)
        dataset_normalized = self._camera_dataset_normalize(np.stack(sequence))
        return dataset_normalized.astype(np.float32)

    @classmethod
    def _prepare_dataset_npy(cls, dataset_npy: np.ndarray) -> np.ndarray:
        """Apply the V3 model-stage transform after Kaggle NPY extraction."""
        return cls._linear_resample(cls._normalize_body(dataset_npy[:, :N_POINTS]))

    def predict_video(self, video_path: str) -> Prediction:
        started = time.perf_counter()
        capture = cv2.VideoCapture(video_path)
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(frames_per_second) or frames_per_second <= 1e-6:
            frames_per_second = 30.0
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        decode_ms = (time.perf_counter() - started) * 1_000
        prediction = self.predict_frames(frames, frames_per_second)
        return replace(prediction, diagnostics={
            'video_decode_ms': round(decode_ms, 2), **prediction.diagnostics,
        })

    def predict_frames(self, frames: list[np.ndarray], frames_per_second: float = 30.0) -> Prediction:
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        started = time.perf_counter()
        raw_frame_count = len(frames)
        sampled_indices = np.unique(
            np.linspace(0, raw_frame_count - 1, min(raw_frame_count, MAX_MEDIAPIPE_FRAMES)).round().astype(np.int64),
        )
        sampled_frames = [frames[int(index)] for index in sampled_indices]
        # The V3 validation pipeline supports variable sequence lengths, then
        # interpolates landmarks to the required 48 model steps below.
        dataset_npy = self._extract_dataset_npy(sampled_frames)
        sequence = self._prepare_dataset_npy(dataset_npy)
        preprocess_ms = (time.perf_counter() - started) * 1_000
        coverage = float(np.any(sequence[:, 33:75, 3] > 0.5, axis=1).mean())
        inference_started = time.perf_counter()
        logits = self.session.run(['logits'], {'keypoints': sequence[None].astype(np.float32)})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        ordered = np.argsort(probabilities)[::-1]
        candidates = tuple(Candidate(self.labels[int(index)], float(probabilities[index])) for index in ordered[:3])
        return Prediction(
            label=candidates[0].label,
            text=candidates[0].label,
            confidence=candidates[0].confidence,
            margin=float(probabilities[ordered[0]] - probabilities[ordered[1]]),
            landmark_coverage=coverage,
            top_k=candidates,
            diagnostics={
                'input_frames': raw_frame_count,
                'mediapipe_input_frames': len(sampled_frames),
                'dataset_npy_shape': list(dataset_npy.shape),
                'dataset_npy_contract': 'kaggle_extract_keypoint_vslfullfront_v3',
                'sampled_frames': SEQ_LEN,
                'preprocess_contract': 'vsl30_camera_ipynb_two_stage_v1',
                'keypoint_tensor_shape': [1, SEQ_LEN, N_POINTS, 4],
                'mediapipe_preprocess_ms': round(preprocess_ms, 2),
                'onnx_inference_ms': round((time.perf_counter() - inference_started) * 1_000, 2),
                'onnx_provider': self.session.get_providers()[0],
            },
        )
