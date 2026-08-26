"""CPU-only ONNX inference adapter for the VSL-30 keypoint classifier."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ..trace import trace
from .types import Candidate, Prediction

logger = logging.getLogger(__name__)
Array = NDArray[Any]
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
# UNO Q extracts a uniform 32-frame clip before MediaPipe to keep latency and
# memory bounded. V4.3 reconstruction and landmark resampling still produce
# the exact 48-step tensor expected by the ONNX model.
MAX_MEDIAPIPE_FRAMES = 32


class Vsl30KeypointOnnxPredictor:
    """Runs the V4.3 `[batch, 48, 75, 4]` keypoint contract on CPU."""

    def __init__(self, model_path: str, labels_path: str) -> None:
        checkpoint_path, label_path = Path(model_path), Path(labels_path)
        if not checkpoint_path.is_file() or not label_path.is_file():
            raise FileNotFoundError('VSL-30 artifacts are missing: classifier ONNX and label_map.json are required.')
        self.labels = self._load_labels(label_path)
        self.session = ort.InferenceSession(str(checkpoint_path), providers=['CPUExecutionProvider'])
        input_contract = self.session.get_inputs()[0]
        input_shape = input_contract.shape
        # Older exports use batch 1; V4.3 has a symbolic batch dimension.
        # The Edge service submits one clip, so the remaining dimensions are
        # the strict compatibility contract.
        if (
            input_contract.name != 'keypoints'
            or len(input_shape) != 4
            or input_shape[1:] != [SEQ_LEN, N_POINTS, 4]
        ):
            raise ValueError(f'Unexpected VSL-30 ONNX input: {input_contract.name} {input_contract.shape}')
        self.model_name = 'VSL-30 keypoint classifier (ONNX CPU, 30 glosses)'
        self.input_names = {'video_or_jpeg_frames'}
        logger.info(
            '[model] loaded model=%s path=%s labels=%d input=%s shape=%s providers=%s',
            self.model_name,
            checkpoint_path,
            len(self.labels),
            input_contract.name,
            input_shape,
            self.session.get_providers(),
        )

    @staticmethod
    def _load_labels(path: Path) -> dict[int, str]:
        payload = json.loads(path.read_text(encoding='utf-8'))
        values = payload.get('idx_to_label') if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            raise TypeError('VSL-30 label_map.json must contain idx_to_label.')
        labels = {int(index): str(label).strip() for index, label in values.items()}
        if set(labels) != set(range(30)) or any(not label for label in labels.values()):
            raise ValueError('VSL-30 label_map.json must map every class 0 through 29 to a non-empty label.')
        return labels

    @staticmethod
    def _normalize_body(raw_xyz: Array) -> Array:
        """Match V4.3 notebook `normalize_body`.

        The exported classifier was validated with this exact 75-point
        pose/left-hand/right-hand transform.  Do not use the low-shot model's
        115-point normalizer here: it has different shoulder filtering and
        visibility-resampling behaviour.
        """
        xyz = Vsl30KeypointOnnxPredictor._reconstruct_missing_keypoints(raw_xyz)
        if xyz.ndim != 3 or xyz.shape[1:] != (N_POINTS, 3):
            raise ValueError(f'Expected raw landmarks [frames, {N_POINTS}, 3], got {xyz.shape}.')
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        mask = (np.linalg.norm(xyz, axis=-1) > 1e-8).astype(np.float32)[..., None]

        # Use the shoulder midpoint as the body origin whenever both shoulders
        # exist. This removes camera translation before the classifier sees the
        # sequence, while the fallback below keeps partially visible clips usable.
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
        return cast(Array, np.concatenate((output, mask), axis=-1).astype(np.float32))

    @staticmethod
    def _reconstruct_missing_keypoints(raw_xyz: Array) -> Array:
        """Match V4.3's per-joint temporal reconstruction before normalization."""
        xyz = np.asarray(raw_xyz, dtype=np.float32)
        if xyz.ndim != 3 or xyz.shape[1:] != (N_POINTS, 3):
            raise ValueError(f'Expected raw landmarks [frames, {N_POINTS}, 3], got {xyz.shape}.')
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        if len(xyz) < 3:
            return xyz
        observed = np.linalg.norm(xyz, axis=-1) > 1e-8
        output = xyz.copy()
        timeline = np.arange(len(output), dtype=np.float32)
        for joint_index in range(N_POINTS):
            indices = np.flatnonzero(observed[:, joint_index])
            if len(indices) < 2:
                continue
            source_time = indices.astype(np.float32)
            for axis in range(3):
                # np.interp uses the nearest observed endpoint outside the
                # observed interval and linearly fills gaps inside it.
                output[:, joint_index, axis] = np.interp(
                    timeline,
                    source_time,
                    output[indices, joint_index, axis],
                ).astype(np.float32)
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _linear_resample(value: Array, target_len: int = SEQ_LEN) -> Array:
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
        # Coordinates are interpolated continuously, but visibility remains a
        # mask: suppress coordinates whose interpolated visibility is too low.
        output[..., 3] = np.clip(output[..., 3], 0.0, 1.0)
        output[..., :3] *= output[..., 3:4] > 0.2
        return output.astype(np.float32)

    @staticmethod
    def _camera_dataset_normalize(raw_76: Array) -> Array:
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

    def _extract_dataset_npy(self, frames: list[Array], trace_id: str = '-') -> Array:
        """Convert an uploaded video to the Kaggle extractor's `[T, 76, 3]` NPY contract."""
        # MediaPipe Tasks' HolisticLandmarker graph is not ABI-compatible with
        # every ARM build shipped by Arduino App Lab. The legacy Holistic API
        # provides the same normalized pose/hand landmarks required here, with
        # no external .task graph or TaskRunner binding.
        sequence: list[Array] = []
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            refine_face_landmarks=False,
        ) as holistic:
            trace(logger, trace_id, '04', 'MEDIAPIPE START', frame_count=len(frames), input_color='BGR -> RGB')
            for frame_index, frame in enumerate(frames, start=1):
                result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                def points(landmarks: Any, count: int) -> Array:
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
                visible_points = int(np.count_nonzero(np.linalg.norm(raw, axis=-1) > 1e-8))
                trace(
                    logger,
                    trace_id,
                    f'04.{frame_index:02d}',
                    'MEDIAPIPE FRAME COMPLETE',
                    frame=f'{frame_index}/{len(frames)}',
                    image_shape=frame.shape,
                    pose_points=N_POSE,
                    hand_points=2 * N_HAND,
                    visible_points=f'{visible_points}/{TRAIN_RAW_POINTS}',
                )
        dataset_normalized = self._camera_dataset_normalize(np.stack(sequence))
        trace(
            logger,
            trace_id,
            '05',
            'KEYPOINT EXTRACTION COMPLETE',
            raw_shape=np.stack(sequence).shape,
            normalized_shape=dataset_normalized.shape,
            visible_ratio=round(
                float(np.count_nonzero(np.linalg.norm(dataset_normalized, axis=-1) > 1e-8))
                / float(dataset_normalized.size / 3),
                6,
            ),
        )
        return dataset_normalized.astype(np.float32)

    @classmethod
    def _prepare_dataset_npy(cls, dataset_npy: Array) -> Array:
        """Apply the V4.3 model-stage transform after Kaggle NPY extraction."""
        return cls._linear_resample(cls._normalize_body(dataset_npy[:, :N_POINTS]))

    def predict_video(self, video_path: str, trace_id: str | None = None) -> Prediction:
        trace_id = trace_id or '-'
        trace(logger, trace_id, '01', 'VIDEO DECODE START', path=video_path)
        started = time.perf_counter()
        capture = cv2.VideoCapture(video_path)
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(frames_per_second) or frames_per_second <= 1e-6:
            frames_per_second = 30.0
        frames: list[Array] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        decode_ms = (time.perf_counter() - started) * 1_000
        trace(
            logger,
            trace_id,
            '02',
            'VIDEO DECODE COMPLETE',
            frame_count=len(frames),
            fps=round(frames_per_second, 3),
            elapsed_ms=round(decode_ms, 2),
        )
        prediction = self.predict_frames(frames, frames_per_second, trace_id)
        return replace(prediction, diagnostics={
            'video_decode_ms': round(decode_ms, 2), **prediction.diagnostics,
        })

    def predict_frames(
        self,
        frames: list[Array],
        frames_per_second: float = 30.0,
        trace_id: str | None = None,
    ) -> Prediction:
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        trace_id = trace_id or '-'
        trace(
            logger,
            trace_id,
            '03',
            'FRAME SAMPLING START',
            input_frames=len(frames),
            fps=round(frames_per_second, 3),
            max_mediapipe_frames=MAX_MEDIAPIPE_FRAMES,
        )
        started = time.perf_counter()
        raw_frame_count = len(frames)
        # Sample uniformly over the whole clip instead of taking only the first
        # 32 frames; late hand motion is often the discriminative part of a sign.
        sampled_indices = np.unique(
            np.linspace(0, raw_frame_count - 1, min(raw_frame_count, MAX_MEDIAPIPE_FRAMES)).round().astype(np.int64),
        )
        sampled_frames = [frames[int(index)] for index in sampled_indices]
        trace(
            logger,
            trace_id,
            '03.1',
            'FRAME SAMPLING COMPLETE',
            raw_frames=raw_frame_count,
            sampled_frames=len(sampled_frames),
            selected_indices=sampled_indices.tolist(),
        )
        # Keep the full clip duration through uniform sampling, then reproduce
        # the V4.3 landmark reconstruction/normalization/resampling contract.
        dataset_npy = self._extract_dataset_npy(sampled_frames, trace_id)
        trace(
            logger,
            trace_id,
            '06',
            'V4.3 NORMALIZATION START',
            input_shape=dataset_npy.shape,
            input_dtype=dataset_npy.dtype,
            contract='reconstruct -> normalize -> resample',
        )
        sequence = self._prepare_dataset_npy(dataset_npy)
        normalized_visible_ratio = float(np.mean(sequence[..., 3] > 0.5))
        trace(
            logger,
            trace_id,
            '07',
            'MODEL TENSOR READY',
            tensor_shape=[1, *sequence.shape],
            tensor_dtype=sequence.dtype,
            visible_ratio=round(normalized_visible_ratio, 6),
            value_min=round(float(sequence.min()), 6),
            value_max=round(float(sequence.max()), 6),
        )
        preprocess_ms = (time.perf_counter() - started) * 1_000
        coverage = float(np.any(sequence[:, 33:75, 3] > 0.5, axis=1).mean())
        trace(
            logger,
            trace_id,
            '08',
            'PREPROCESSING COMPLETE',
            elapsed_ms=round(preprocess_ms, 2),
            landmark_coverage=round(coverage, 6),
        )
        inference_started = time.perf_counter()
        trace(
            logger,
            trace_id,
            '09',
            'ONNX INFERENCE START',
            input_name='keypoints',
            input_shape=[1, SEQ_LEN, N_POINTS, 4],
            provider=self.session.get_providers()[0],
        )
        logits = self.session.run(['logits'], {'keypoints': sequence[None].astype(np.float32)})[0][0]
        onnx_ms = (time.perf_counter() - inference_started) * 1_000
        # Subtract the maximum logit before exponentiation for stable softmax on
        # the UNO Q CPU, where an unshifted exponent can overflow.
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        ordered = np.argsort(probabilities)[::-1]
        candidates = tuple(Candidate(self.labels[int(index)], float(probabilities[index])) for index in ordered[:3])
        trace(
            logger,
            trace_id,
            '09.1',
            'ONNX INFERENCE COMPLETE',
            logits_shape=logits.shape,
            elapsed_ms=round(onnx_ms, 2),
            top_1=f'{candidates[0].label} ({candidates[0].confidence:.6f})',
            top_2=f'{candidates[1].label} ({candidates[1].confidence:.6f})',
            top_3=f'{candidates[2].label} ({candidates[2].confidence:.6f})',
        )
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
                'dataset_npy_contract': 'kaggle_extract_keypoint_v4_3',
                'sampled_frames': SEQ_LEN,
                'preprocess_contract': 'vsl30_v4_3_reconstruct_normalize_resample_v1',
                'keypoint_tensor_shape': [1, SEQ_LEN, N_POINTS, 4],
                'mediapipe_preprocess_ms': round(preprocess_ms, 2),
                'onnx_inference_ms': round(onnx_ms, 2),
                'onnx_provider': self.session.get_providers()[0],
            },
        )
