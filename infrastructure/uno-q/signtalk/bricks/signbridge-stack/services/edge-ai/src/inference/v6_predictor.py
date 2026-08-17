from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]


class LandmarkLike(Protocol):
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Candidate:
    label: str
    confidence: float


@dataclass(frozen=True)
class Prediction:
    label: str
    text: str
    confidence: float
    margin: float
    landmark_coverage: float
    top_k: tuple[Candidate, ...]
    diagnostics: dict[str, object]
    accepted: bool | None = None


class V6Predictor:
    """V6/V6.2 ONNX inference using the training-time feature contracts."""

    def __init__(self, model_path: str, labels_path: str) -> None:
        model = Path(model_path)
        labels = Path(labels_path)
        if not model.is_file() or not labels.is_file():
            raise FileNotFoundError('V6 ONNX artifacts are missing. Set MODEL_PATH and LABELS_PATH in services/edge-ai/.env.')
        available = ort.get_available_providers()
        providers = [provider for provider in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if provider in available]
        self.session = ort.InferenceSession(str(model), providers=providers)
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.is_v62 = {'legacy_features', 'anchor_features', 'mask', 'video'} <= self.input_names
        if not self.is_v62 and not {'features', 'mask', 'video'} <= self.input_names:
            raise ValueError(f'Unsupported ONNX input contract: {sorted(self.input_names)}')
        self.model_name = 'V6.2 multimodal ONNX' if self.is_v62 else 'V6 multimodal ONNX'
        mapping = json.loads(labels.read_text(encoding='utf-8'))
        self.labels = (
            mapping['id_to_label']
            if isinstance(mapping, dict) and 'id_to_label' in mapping
            else mapping
        )
        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1, refine_face_landmarks=False)

    @staticmethod
    def _sample(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        positions = np.linspace(0, len(frames) - 1, count).round().astype(np.int64)
        return [frames[index] for index in positions]

    @staticmethod
    def _validate_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
        """Validate decoded video frames before VSL2-compatible cropping."""
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        for frame in frames:
            if frame.ndim != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
                raise ValueError('A submitted frame has invalid dimensions.')
        return frames

    @staticmethod
    def _point(landmark: LandmarkLike | None) -> np.ndarray:
        if landmark is None:
            return np.zeros(3, dtype=np.float32)
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    def _landmarks(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)
        output = np.zeros((76, 3), dtype=np.float32)
        pose = result.pose_landmarks.landmark if result.pose_landmarks else []
        for index in range(min(33, len(pose))):
            output[index] = self._point(pose[index])
        # VSL2's 34th body point is a stable shoulder-centre anchor.
        if len(pose) > 12:
            output[33] = (output[11] + output[12]) * 0.5
        left = result.left_hand_landmarks.landmark if result.left_hand_landmarks else []
        right = result.right_hand_landmarks.landmark if result.right_hand_landmarks else []
        for index in range(21):
            output[34 + index * 2] = self._point(left[index] if index < len(left) else None)
            output[35 + index * 2] = self._point(right[index] if index < len(right) else None)
        return output

    @staticmethod
    def _crop_square(frame: np.ndarray, center_x: float, center_y: float, side: int) -> np.ndarray:
        """Crop a square signer ROI, replicating only pixels outside the source frame."""
        height, width = frame.shape[:2]
        left = round(center_x * width - side / 2)
        top = round(center_y * height - side / 2)
        right = left + side
        bottom = top + side
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - width)
        pad_bottom = max(0, bottom - height)
        if pad_left or pad_top or pad_right or pad_bottom:
            frame = cv2.copyMakeBorder(
                frame,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )
        return frame[top + pad_top:bottom + pad_top, left + pad_left:right + pad_left]

    def _signer_cropped_frames(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, object]]:
        """Approximate VSL2's processed 224x224 signer crop from a raw upload.

        V6 was trained on the dataset's pre-cropped front-view videos, not the
        raw 1080px source video. A clip-level crop keeps the person and signing
        space at the scale seen by the RGB and keypoint branches during training.
        """
        self._validate_frames(frames)
        self.holistic.reset()
        try:
            samples = self._sample(frames, min(len(frames), 12))
            selected_indices = np.r_[np.arange(25), np.arange(34, 76)]
            points: list[np.ndarray] = []
            for frame in samples:
                landmarks = self._landmarks(frame)[selected_indices, :2]
                valid = np.any(np.abs(landmarks) > 1e-8, axis=1)
                if valid.any():
                    points.append(np.clip(landmarks[valid], 0.0, 1.0))
        finally:
            # Holistic tracking state must never carry over from crop probing to
            # feature extraction, or between independent uploaded videos.
            self.holistic.reset()

        height, width = frames[0].shape[:2]
        if not points:
            return frames, {
                'crop_mode': 'full_frame_fallback',
                'crop_width': width,
                'crop_height': height,
            }
        all_points = np.concatenate(points, axis=0)
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        center = (minimum + maximum) / 2
        # 1.40 retains head, torso and the complete signing space while making
        # the signer occupy a comparable fraction of VSL2 processed clips.
        side_fraction = max(float(np.max(maximum - minimum)) * 1.40, 0.45)
        side = max(2, round(side_fraction * max(height, width)))
        cropped = [self._crop_square(frame, float(center[0]), float(center[1]), side) for frame in frames]
        return cropped, {
            'crop_mode': 'mediapipe_signer_square_v1',
            'crop_width': int(cropped[0].shape[1]),
            'crop_height': int(cropped[0].shape[0]),
            'crop_center_x': round(float(center[0]), 4),
            'crop_center_y': round(float(center[1]), 4),
            'crop_side_fraction': round(side_fraction, 4),
        }

    @staticmethod
    def _sample_array(values: np.ndarray, count: int) -> np.ndarray:
        positions = np.linspace(0, len(values) - 1, count).round().astype(np.int64)
        return values[positions]

    @staticmethod
    def _legacy_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = V6Predictor._sample_array(raw, 40)
        mask = np.any(np.abs(coords) > 1e-8, axis=(1, 2))
        normalized = np.zeros_like(coords)
        for index, frame in enumerate(coords):
            points = frame[np.any(np.abs(frame) > 1e-8, axis=1)]
            if len(points):
                center = np.median(points, axis=0)
                scale = max(float(np.median(np.linalg.norm(points - center, axis=1))), 1e-5)
                normalized[index] = (frame - center) / scale
        velocity = np.diff(normalized, axis=0, prepend=normalized[:1])
        features = np.concatenate([normalized.reshape(40, -1), velocity.reshape(40, -1)], axis=1).astype(np.float32)
        return features, mask

    @staticmethod
    def _interpolate_short_gaps(raw: np.ndarray, max_gap: int = 3) -> tuple[np.ndarray, np.ndarray]:
        output = raw.copy()
        observed = np.any(np.abs(output) > 1e-8, axis=2)
        for joint in range(output.shape[1]):
            valid = np.flatnonzero(observed[:, joint])
            for left, right in pairwise(valid):
                gap = int(right - left - 1)
                if 0 < gap <= max_gap:
                    alpha = np.arange(1, gap + 1, dtype=np.float32)[:, None] / (gap + 1)
                    output[left + 1:right, joint] = output[left, joint] * (1 - alpha) + output[right, joint] * alpha
        return output, observed

    @staticmethod
    def _anchor_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reconstructed, observed = V6Predictor._interpolate_short_gaps(raw)
        positions = np.linspace(0, len(reconstructed) - 1, 40).round().astype(np.int64)
        coords = reconstructed[positions]
        observed = observed[positions]
        mask = observed.any(axis=1)
        normalized = np.zeros_like(coords)
        for index, frame in enumerate(coords):
            valid = np.any(np.abs(frame) > 1e-8, axis=1)
            if not valid.any():
                continue
            if valid[11] and valid[12]:
                center = (frame[11] + frame[12]) * 0.5
                scale = max(float(np.linalg.norm(frame[11] - frame[12])), 1e-4)
            else:
                points = frame[valid]
                center = np.median(points, axis=0)
                scale = max(float(np.median(np.linalg.norm(points - center, axis=1))), 1e-4)
            normalized[index, valid] = (frame[valid] - center) / scale
        velocity = np.diff(normalized, axis=0, prepend=normalized[:1])
        features = np.concatenate([normalized.reshape(40, -1), velocity.reshape(40, -1)], axis=1).astype(np.float32)
        return np.clip(features, -8.0, 8.0), mask

    def _keypoint_inputs(
        self,
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        raw = np.stack([self._landmarks(frame) for frame in frames]).astype(np.float32)
        hand_coverage = float(np.any(np.abs(raw[:, 34:]) > 1e-8, axis=(1, 2)).mean())
        legacy, legacy_mask = self._legacy_features(raw)
        if not self.is_v62:
            return legacy[None], legacy[None], legacy_mask[None], hand_coverage
        anchor, anchor_mask = self._anchor_features(raw)
        if not np.array_equal(legacy_mask, anchor_mask):
            raise ValueError('V6.2 legacy and anchor frame masks differ.')
        return legacy[None], anchor[None], anchor_mask[None], hand_coverage

    def _rgb_input(self, frames: list[np.ndarray]) -> np.ndarray:
        output = []
        for frame in self._sample(frames, 16):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)[8:120, 8:120].astype(np.float32) / 255.0
            output.append(rgb)
        video = np.stack(output)
        mean = np.array([.43216, .394666, .37645], dtype=np.float32)
        std = np.array([.22803, .22145, .216989], dtype=np.float32)
        return ((video - mean) / std).transpose(3, 0, 1, 2)[None].astype(np.float32)

    def predict_video(self, video_path: str) -> Prediction:
        started = time.perf_counter()
        capture = cv2.VideoCapture(video_path)
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        decode_ms = (time.perf_counter() - started) * 1_000
        prediction = self.predict_frames(frames)
        return replace(
            prediction,
            diagnostics={
                'video_decode_ms': round(decode_ms, 2),
                **prediction.diagnostics,
            },
        )

    def predict_frames(self, frames: list[np.ndarray]) -> Prediction:
        started = time.perf_counter()
        compatible_frames, crop_details = self._signer_cropped_frames(frames)
        crop_ms = (time.perf_counter() - started) * 1_000
        landmark_started = time.perf_counter()
        legacy_features, anchor_features, mask, hand_coverage = self._keypoint_inputs(
            compatible_frames,
        )
        landmark_ms = (time.perf_counter() - landmark_started) * 1_000
        rgb_started = time.perf_counter()
        video = self._rgb_input(compatible_frames)
        rgb_ms = (time.perf_counter() - rgb_started) * 1_000
        inputs = {'mask': mask, 'video': video}
        if self.is_v62:
            inputs.update({'legacy_features': legacy_features, 'anchor_features': anchor_features})
        else:
            inputs['features'] = legacy_features
        inference_started = time.perf_counter()
        logits = self.session.run(['logits'], inputs)[0][0]
        inference_ms = (time.perf_counter() - inference_started) * 1_000
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        top_indices = np.argsort(probabilities)[::-1][:3]
        def label_at(index: int) -> str:
            return self.labels[index] if isinstance(self.labels, list) else self.labels[str(index)]
        top_k = tuple(Candidate(label=label_at(int(item)), confidence=float(probabilities[item])) for item in top_indices)
        winner = top_k[0]
        margin = winner.confidence - top_k[1].confidence if len(top_k) > 1 else winner.confidence
        return Prediction(
            label=winner.label,
            text=winner.label,
            confidence=winner.confidence,
            margin=float(margin),
            landmark_coverage=hand_coverage,
            top_k=top_k,
            diagnostics={
                'input_frames': len(frames),
                'input_width': int(frames[0].shape[1]),
                'input_height': int(frames[0].shape[0]),
                'preprocess_contract': 'v6_processed_signer_crop_v1',
                'crop_signer_ms': round(crop_ms, 2),
                **crop_details,
                'mediapipe_features_ms': round(landmark_ms, 2),
                'rgb_tensor_ms': round(rgb_ms, 2),
                'onnx_inference_ms': round(inference_ms, 2),
                'hand_coverage': round(hand_coverage, 4),
                'onnx_provider': self.session.get_providers()[0],
            },
        )
