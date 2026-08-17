"""Inference adapter for the supplied VSL-30 keypoint classifier.

The checkpoint was trained on a 48-frame sequence of 75 MediaPipe Holistic
landmarks: 33 pose points, 21 left-hand points, and 21 right-hand points.  A
video or JPEG sequence is therefore converted to that exact shape before the
PyTorch classifier is invoked.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .types import Candidate, Prediction
from .vsl_metric_lowshot_predictor import (
    BODY_IDS,
    COORD_CLIP,
    RegionMLP,
    VslMetricLowShotPredictor,
    _hand_features,
    _motion,
)

SEQ_LEN = 48
N_POINTS = 75
REGION_DIM = 96
EMBED_DIM = 256


class AttentionPool(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.score = nn.Linear(dimension, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(value), dim=1)
        return (value * weights).sum(dim=1)


class Vsl30KeypointEncoder(nn.Module):
    """Architecture reconstructed from the checkpoint's state-dict contract."""

    def __init__(self) -> None:
        super().__init__()
        self.body_encoder = RegionMLP(len(BODY_IDS) * 7, REGION_DIM)
        self.hand_encoder = RegionMLP(230, REGION_DIM)
        self.side_embed = nn.Parameter(torch.zeros(2, REGION_DIM))
        self.gate = nn.Sequential(
            nn.Linear(3 * REGION_DIM, REGION_DIM), nn.GELU(), nn.Linear(REGION_DIM, 3),
        )
        self.fuse = nn.Sequential(
            nn.Linear(3 * REGION_DIM, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.GELU(),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(EMBED_DIM, EMBED_DIM, 3, padding=1, groups=EMBED_DIM),
            nn.GELU(),
            nn.Conv1d(EMBED_DIM, EMBED_DIM, 1),
            nn.Dropout(0.12),
        )
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, EMBED_DIM))
        layer = nn.TransformerEncoderLayer(
            EMBED_DIM, 8, 512, dropout=0.12, activation='gelu', batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=3)
        self.attn_pool = AttentionPool(EMBED_DIM)
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM), nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(),
            nn.Dropout(0.12), nn.Linear(EMBED_DIM, EMBED_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[1:]) != (SEQ_LEN, N_POINTS, 4):
            raise ValueError(f'Expected keypoints [batch, {SEQ_LEN}, {N_POINTS}, 4], got {tuple(value.shape)}')
        value = torch.nan_to_num(value.float())
        value = torch.cat((value[..., :3].clamp(-COORD_CLIP, COORD_CLIP), value[..., 3:4].clamp(0.0, 1.0)), dim=-1)

        def basic(region: torch.Tensor) -> torch.Tensor:
            xyz, mask = region[..., :3], region[..., 3:4]
            return torch.cat(((xyz * mask).flatten(-2), mask.flatten(-2), _motion(xyz, mask).flatten(-2)), dim=-1)

        body = self.body_encoder(basic(value[:, :, BODY_IDS, :]))
        left = self.hand_encoder(_hand_features(value[:, :, 33:54, :])) + self.side_embed[0]
        right = self.hand_encoder(_hand_features(value[:, :, 54:75, :])) + self.side_embed[1]
        regions = (body, left, right)
        gates = torch.softmax(self.gate(torch.cat(regions, dim=-1)), dim=-1)
        hidden = self.fuse(torch.cat(
            tuple(region * gates[..., index:index + 1] for index, region in enumerate(regions)), dim=-1,
        ))
        hidden = hidden + self.local_conv(hidden.transpose(1, 2)).transpose(1, 2) + self.pos
        return functional.normalize(self.head(self.attn_pool(self.temporal(hidden))), dim=-1, eps=1e-6)


class Vsl30KeypointClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kp_encoder = Vsl30KeypointEncoder()
        # ``classifier`` is retained because it is part of the source checkpoint.
        # In keypoint-only mode the training export uses the dedicated kp head.
        self.classifier = nn.Sequential(nn.LayerNorm(EMBED_DIM), nn.Dropout(0.12), nn.Linear(EMBED_DIM, 30))
        self.kp_classifier = nn.Linear(EMBED_DIM, 30)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        return self.kp_classifier(self.kp_encoder(keypoints))


class Vsl30KeypointPredictor:
    def __init__(self, model_path: str, labels_path: str) -> None:
        checkpoint_path, label_path = Path(model_path), Path(labels_path)
        if not checkpoint_path.is_file() or not label_path.is_file():
            raise FileNotFoundError('VSL-30 artifacts are missing: best_vsl30_keypoint.pt and label_map.json are required.')
        self.labels = self._load_labels(label_path)
        self.model = Vsl30KeypointClassifier().eval()
        self.model.load_state_dict(self._load_weights(checkpoint_path), strict=True)
        self.task_asset = Path(__file__).resolve().parents[2] / 'models' / 'vsl_metric_lowshot' / 'holistic_landmarker.task'
        if not self.task_asset.is_file():
            raise FileNotFoundError(f'Missing shared MediaPipe Holistic task asset: {self.task_asset}')
        self.model_name = 'VSL-30 keypoint classifier (MediaPipe Holistic, 30 glosses)'
        self.input_names = {'video_or_jpeg_frames'}

    @staticmethod
    def _load_weights(path: Path) -> dict[str, torch.Tensor]:
        # Never fall back to pickle execution for an externally supplied checkpoint.
        try:
            scalar = import_module('numpy._core.multiarray').scalar
        except ModuleNotFoundError:  # NumPy 1.x
            scalar = import_module('numpy.core.multiarray').scalar
        allowed: list[object] = [(scalar, 'numpy._core.multiarray.scalar'), np.dtype]
        float64_dtype = getattr(getattr(np, 'dtypes', object()), 'Float64DType', None)
        if float64_dtype is not None:
            allowed.append(float64_dtype)
        torch.serialization.add_safe_globals(allowed)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get('model_state'), dict):
            raise ValueError('VSL-30 checkpoint does not contain a model_state dictionary.')
        state = payload['model_state']
        if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
            raise ValueError('VSL-30 checkpoint contains an invalid model_state.')
        return state

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

    def _extract(self, frames: list[np.ndarray], source_indices: np.ndarray, frames_per_second: float) -> np.ndarray:
        # Reuse the training family's Holistic extraction, then omit the 40 face
        # landmarks because this classifier was trained on pose + hands only.
        extractor = object.__new__(VslMetricLowShotPredictor)
        extractor.task_asset = self.task_asset
        sequence = extractor._extract(frames, source_indices, frames_per_second)
        return sequence[:, :N_POINTS, :]

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
        prediction = self.predict_frames(frames, frames_per_second)
        return replace(prediction, diagnostics={
            'video_decode_ms': round((time.perf_counter() - started) * 1_000, 2), **prediction.diagnostics,
        })

    def predict_frames(self, frames: list[np.ndarray], frames_per_second: float = 30.0) -> Prediction:
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        started = time.perf_counter()
        raw_frame_count = len(frames)
        sampled_indices = VslMetricLowShotPredictor._uniform_indices(raw_frame_count)
        sampled_frames = [frames[int(index)] for index in sampled_indices]
        sequence = self._extract(sampled_frames, sampled_indices, frames_per_second)
        preprocess_ms = (time.perf_counter() - started) * 1_000
        coverage = float(np.any(sequence[:, 33:75, 3] > 0.5, axis=1).mean())
        inference_started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(torch.from_numpy(sequence[None])), dim=1)[0].numpy()
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
                'sampled_frames': SEQ_LEN,
                'preprocess_contract': 'vsl30_holistic_pose_hands_75_shoulder_normalized_v1',
                'keypoint_tensor_shape': [1, SEQ_LEN, N_POINTS, 4],
                'mediapipe_preprocess_ms': round(preprocess_ms, 2),
                'pytorch_inference_ms': round((time.perf_counter() - inference_started) * 1_000, 2),
            },
        )
