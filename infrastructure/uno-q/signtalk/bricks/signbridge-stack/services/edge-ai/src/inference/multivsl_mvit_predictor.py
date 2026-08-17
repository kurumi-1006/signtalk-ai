from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from .class_labels import load_optional_labels
from .multivsl.mvit_v2 import mvit_v2_s
from .v6_predictor import Candidate, Prediction


class MultiVslMvitPredictor:
    """MViT-v2-S adapter for the Multi-VSL WACV 2025 one-view checkpoint."""

    def __init__(self, model_path: str, labels_path: str) -> None:
        checkpoint = Path(model_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Multi-VSL checkpoint is missing: {checkpoint}')
        state_dict = torch.load(checkpoint, map_location='cpu')
        head_weight = state_dict.get('head.1.weight')
        if not isinstance(head_weight, torch.Tensor) or head_weight.ndim != 2:
            raise ValueError(f'MViT checkpoint has no valid classifier head: {checkpoint}')
        self.class_count = int(head_weight.shape[0])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = mvit_v2_s(num_classes=self.class_count)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.labels = load_optional_labels(labels_path, class_count=self.class_count, fallback_prefix='vsl_gloss_')
        self.model_name = 'Multi-VSL WACV 2025 MViT-v2-S (one-view, 1,000 classes)'
        self.input_names = {'video'}

    @staticmethod
    def _sample(frames: list[np.ndarray], count: int = 16) -> list[np.ndarray]:
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        indices = np.linspace(0, len(frames) - 1, count).round().astype(np.int64)
        return [frames[int(index)] for index in indices]

    def _input_tensor(self, frames: list[np.ndarray]) -> torch.Tensor:
        normalized_frames: list[np.ndarray] = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for frame in self._sample(frames):
            rgb = cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            normalized_frames.append((rgb.astype(np.float32) / 255.0 - mean) / std)
        video = np.stack(normalized_frames).transpose(3, 0, 1, 2)[None]
        return torch.from_numpy(video).to(self.device)

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
        prediction = self.predict_frames(frames)
        return replace(prediction, diagnostics={'video_decode_ms': round((time.perf_counter() - started) * 1_000, 2), **prediction.diagnostics})

    def predict_frames(self, frames: list[np.ndarray]) -> Prediction:
        preprocess_started = time.perf_counter()
        video = self._input_tensor(frames)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1_000
        inference_started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(video)['logits'][0], dim=0).cpu().numpy()
        inference_ms = (time.perf_counter() - inference_started) * 1_000
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_k = tuple(Candidate(label=self.labels[int(index)], confidence=float(probabilities[index])) for index in top_indices)
        winner = top_k[0]
        margin = winner.confidence - top_k[1].confidence
        return Prediction(
            label=winner.label,
            text=winner.label,
            confidence=winner.confidence,
            margin=float(margin),
            landmark_coverage=1.0,
            top_k=top_k,
            diagnostics={
                'input_frames': len(frames),
                'sampled_frames': 16,
                'preprocess_contract': 'multi_vsl_mvit_v2_one_view_224_imagenet_v1',
                'video_preprocess_ms': round(preprocess_ms, 2),
                'pytorch_inference_ms': round(inference_ms, 2),
                'device': str(self.device),
            },
        )
