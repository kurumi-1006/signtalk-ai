from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ImportError:  # Uno Q uses the ONNX runtime and does not install PyTorch.
    torch = None  # type: ignore[assignment]

    class _OptionalNN:
        class Module:
            pass

    nn = _OptionalNN()  # type: ignore[assignment]
    functional = object()  # type: ignore[assignment]

from .types import Candidate, Prediction

FACE_IDX = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87,
            14, 317, 402, 318, 324, 308, 33, 133, 159, 145, 362, 263, 386, 374, 70,
            63, 105, 66, 107, 336, 296, 334, 293, 300)
BODY_IDS = (0, 11, 12, 13, 14, 15, 16, 23, 24)
SEQ_LEN, MAX_RAW_FRAMES, N_POINTS = 48, 96, 115
COORD_CLIP, MIN_SHOULDER_SCALE, MIN_HAND_SCALE = 8.0, 0.08, 0.04


class RegionMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size), nn.LayerNorm(output_size), nn.GELU(),
            nn.Dropout(0.12), nn.Linear(output_size, output_size), nn.LayerNorm(output_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _motion(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(coords)
    output[:, 1:] = (coords[:, 1:] - coords[:, :-1]) * mask[:, 1:] * mask[:, :-1]
    return output


def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cosine = (a * b).sum(dim=-1) / (torch.linalg.norm(a, dim=-1).clamp_min(1e-6)
                                   * torch.linalg.norm(b, dim=-1).clamp_min(1e-6))
    return torch.acos(torch.nan_to_num(cosine).clamp(-0.9999, 0.9999))


def _hand_features(hand: torch.Tensor) -> torch.Tensor:
    xyz, mask = hand[..., :3], hand[..., 3:4]
    wrist, mcp9 = xyz[..., :1, :], xyz[..., 9:10, :]
    anchor_ok = (mask[..., :1, :] > .5) & (mask[..., 9:10, :] > .5)
    scale = torch.where(anchor_ok, torch.linalg.norm(mcp9 - wrist, dim=-1, keepdim=True).clamp_min(MIN_HAND_SCALE), torch.ones_like(mask[..., :1, :]))
    local = ((xyz - wrist) / scale * mask * (mask[..., :1, :] > .5)).clamp(-6.0, 6.0)
    fingers = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
               (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    values: list[torch.Tensor] = []
    validity = hand[..., 3] > .5
    for finger in fingers:
        for index in range(1, 4):
            a, b, c = finger[index - 1], finger[index], finger[index + 1]
            values.append(torch.where(validity[..., a] & validity[..., b] & validity[..., c], _angle(xyz[..., a, :] - xyz[..., b, :], xyz[..., c, :] - xyz[..., b, :]), torch.zeros_like(validity[..., a], dtype=xyz.dtype)))
    mcps = (1, 5, 9, 13, 17)
    rays = [xyz[..., point, :] - xyz[..., 0, :] for point in mcps]
    for index in range(4):
        values.append(torch.where(validity[..., 0] & validity[..., mcps[index]] & validity[..., mcps[index + 1]], _angle(rays[index], rays[index + 1]), torch.zeros_like(validity[..., 0], dtype=xyz.dtype)))
    values.append(torch.where(validity[..., 0] & validity[..., 5] & validity[..., 17], _angle(rays[1], rays[4]), torch.zeros_like(validity[..., 0], dtype=xyz.dtype)))
    return torch.cat(((xyz * mask).flatten(-2), mask.flatten(-2), local.flatten(-2), _motion(xyz, mask).flatten(-2), torch.stack(values, dim=-1)), dim=-1)


class VslMetricEncoder(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        region_dim, d_model = int(config['region_dim']), int(config['d_model'])
        self.body_encoder = RegionMLP(len(BODY_IDS) * 7, region_dim)
        self.hand_encoder = RegionMLP(230, region_dim)
        self.face_encoder = RegionMLP(40 * 7, region_dim)
        self.side_embed = nn.Parameter(torch.zeros(2, region_dim))
        self.gate = nn.Sequential(nn.Linear(4 * region_dim, region_dim), nn.GELU(), nn.Linear(region_dim, 4))
        self.fuse = nn.Sequential(nn.Linear(4 * region_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.local_conv = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model), nn.GELU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(.12))
        self.pos = nn.Parameter(torch.zeros(1, int(config['seq_len']), d_model))
        layer = nn.TransformerEncoderLayer(d_model, int(config['nhead']), int(config['ff_dim']), dropout=.12, activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(config['layers']))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(.12), nn.Linear(d_model, int(config['embed_dim'])))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(value.float())
        value = torch.cat((value[..., :3].clamp(-COORD_CLIP, COORD_CLIP), value[..., 3:4].clamp(0., 1.)), dim=-1)
        def basic(region: torch.Tensor) -> torch.Tensor:
            xyz, mask = region[..., :3], region[..., 3:4]
            return torch.cat(((xyz * mask).flatten(-2), mask.flatten(-2), _motion(xyz, mask).flatten(-2)), dim=-1)
        body = self.body_encoder(basic(value[:, :, BODY_IDS, :]))
        left = self.hand_encoder(_hand_features(value[:, :, 33:54, :])) + self.side_embed[0]
        right = self.hand_encoder(_hand_features(value[:, :, 54:75, :])) + self.side_embed[1]
        face = self.face_encoder(basic(value[:, :, 75:, :]))
        regions = (body, left, right, face)
        cat = torch.cat(regions, dim=-1)
        gates = torch.softmax(self.gate(cat), dim=-1)
        hidden = self.fuse(torch.cat(tuple(region * gates[..., index:index + 1] for index, region in enumerate(regions)), dim=-1))
        hidden = hidden + self.local_conv(hidden.transpose(1, 2)).transpose(1, 2) + self.pos[:, :hidden.size(1)]
        return functional.normalize(self.head(self.temporal(hidden).mean(dim=1)), dim=-1, eps=1e-6)


class VslMetricLowShotPredictor:
    """Faithful inference adapter for the Kaggle low-shot metric-learning export."""

    def __init__(self, model_path: str, labels_path: str) -> None:
        root = Path(model_path).parent
        onnx_path = root / 'vsl_metric_encoder.onnx'
        if not onnx_path.is_file():
            raise FileNotFoundError(f'Missing ONNX metric encoder: {onnx_path}')
        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        inference = json.loads((root / 'inference_config.json').read_text(encoding='utf-8'))
        self.cosine_threshold = float(inference['unknown_cosine_threshold'])
        self.margin_threshold = float(inference['unknown_margin_threshold'])
        self.dtw_topk = int(inference['dtw_topk'])
        self.cosine_weight = float(inference['cosine_weight'])
        self.dtw_weight = float(inference['dtw_weight'])
        prototype_file = root / 'prototypes_stabilized.npz'
        with np.load(prototype_file) as data:
            self.prototypes = data['prototypes'].astype(np.float32)
            self.prototype_labels = data['prototype_labels'].astype(np.int64)
        self.labels = self._load_labels(root / 'class_labels.json', self.prototype_labels)
        reference_path = root / 'reference_sequences.npz'
        if reference_path.is_file():
            with np.load(reference_path, allow_pickle=True) as reference:
                self.reference_sequences = reference['X']
                self.reference_labels = reference['y'].astype(np.int64)
        else:
            # Uno Q deployment intentionally omits the optional 125 MB DTW
            # dictionary. Cosine prototype retrieval remains fully available.
            self.reference_sequences = None
            self.reference_labels = None
        task_asset = root / 'holistic_landmarker.task'
        if not task_asset.is_file():
            raise FileNotFoundError(f'Missing MediaPipe task asset: {task_asset}')
        self.task_asset = task_asset
        self.model_name = 'VSL low-shot metric encoder (Holistic keypoints + prototype retrieval)'
        self.input_names = {'video_or_jpeg_frames'}

    @staticmethod
    def _load_labels(path: Path, prototype_labels: np.ndarray) -> dict[int, str]:
        labels = {int(class_id): str(label) for class_id, label in json.loads(path.read_text(encoding='utf-8')).items()}
        if set(int(value) for value in prototype_labels) - labels.keys():
            raise ValueError('class_labels.json cannot map every metric prototype class.')
        return labels

    @staticmethod
    def _uniform_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
        """Match the notebook's uniform_frame_indices(..., MAX_RAW_FRAMES)."""
        return [frames[int(index)] for index in VslMetricLowShotPredictor._uniform_indices(len(frames))]

    @staticmethod
    def _uniform_indices(frame_count: int) -> np.ndarray:
        if frame_count <= MAX_RAW_FRAMES:
            return np.arange(frame_count, dtype=np.int64)
        return np.unique(np.linspace(0, frame_count - 1, MAX_RAW_FRAMES).round().astype(np.int64))

    @staticmethod
    def _resample(value: np.ndarray) -> np.ndarray:
        if len(value) == SEQ_LEN:
            return value
        positions = np.linspace(0., 1., len(value))
        targets = np.linspace(0., 1., SEQ_LEN)
        output = np.empty((SEQ_LEN, N_POINTS, 4), dtype=np.float32)
        flat, target = value.reshape(len(value), -1), output.reshape(SEQ_LEN, -1)
        for index in range(flat.shape[1]):
            target[:, index] = np.interp(targets, positions, flat[:, index])
        return output

    @staticmethod
    def _normalize(value: np.ndarray) -> np.ndarray:
        value = np.nan_to_num(value.astype(np.float32), nan=0., posinf=0., neginf=0.)
        value[..., 3] = np.clip(value[..., 3], 0., 1.)
        value[..., :3] *= (value[..., 3:4] > 1e-6)
        xyz, mask = value[..., :3], value[..., 3:4]
        valid = (value[:, 11, 3] > .5) & (value[:, 12, 3] > .5)
        center: np.ndarray
        scale: float
        theta = 0.
        if valid.any():
            centers = (xyz[valid, 11] + xyz[valid, 12]) / 2
            distances = np.linalg.norm(xyz[valid, 11, :2] - xyz[valid, 12, :2], axis=1)
            good = np.isfinite(distances) & (distances >= MIN_SHOULDER_SCALE) & (distances <= 1.5)
            if good.any():
                center, scale = np.median(centers[good], axis=0), float(np.median(distances[good]))
                vector = (xyz[valid, 12, :2] - xyz[valid, 11, :2])[good]
                theta = -float(np.median(np.arctan2(vector[:, 1], vector[:, 0])))
            else:
                valid = np.zeros(len(value), dtype=bool)
        if not valid.any():
            points = xyz[mask[..., 0] > .5]
            if not len(points):
                return value
            center = np.median(points, axis=0)
            radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
            scale = float(np.median(radial[radial > 1e-4])) if (radial > 1e-4).any() else .25
        centers_t = np.repeat(center[None], len(value), axis=0)
        if valid.any():
            centers_t[valid] = (xyz[valid, 11] + xyz[valid, 12]) / 2
        xyz = (xyz - centers_t[:, None, :]) / float(np.clip(scale, MIN_SHOULDER_SCALE, 2.0))
        c, s = np.cos(theta), np.sin(theta)
        xyz[..., :2] = xyz[..., :2] @ np.array(((c, -s), (s, c)), dtype=np.float32).T
        value[..., :3] = np.clip(xyz * mask, -COORD_CLIP, COORD_CLIP)
        return value

    @staticmethod
    def _frame_masked_distance(query: np.ndarray, reference: np.ndarray) -> float:
        # The notebook reranks with pose + hands only, and weights hands 2×.
        points = np.r_[np.arange(33), np.arange(33, 75)]
        query_points, reference_points = query[points], reference[points]
        valid = (query_points[:, 3] > .5) & (reference_points[:, 3] > .5)
        if not valid.any():
            return 5.0
        weights = np.ones(len(points), dtype=np.float32)
        weights[33:] = 2.0
        distances = np.linalg.norm(query_points[valid, :3] - reference_points[valid, :3], axis=1)
        return float(np.sum(distances * weights[valid]) / (np.sum(weights[valid]) + 1e-12))

    @classmethod
    def _masked_dtw_distance(cls, query: np.ndarray, reference: np.ndarray) -> float:
        dynamic = np.full((len(query) + 1, len(reference) + 1), np.inf, dtype=np.float32)
        dynamic[0, 0] = 0.0
        for query_index in range(1, len(query) + 1):
            for reference_index in range(1, len(reference) + 1):
                dynamic[query_index, reference_index] = cls._frame_masked_distance(
                    query[query_index - 1], reference[reference_index - 1],
                ) + min(
                    dynamic[query_index - 1, reference_index],
                    dynamic[query_index, reference_index - 1],
                    dynamic[query_index - 1, reference_index - 1],
                )
        return float(dynamic[-1, -1] / (len(query) + len(reference)))

    def _dtw_rerank(
        self, sequence: np.ndarray, indices: np.ndarray, cosine: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        distances = []
        for index in indices:
            class_id = self.prototype_labels[int(index)]
            references = self.reference_sequences[self.reference_labels == class_id]
            distances.append(min(self._masked_dtw_distance(sequence, reference) for reference in references))
        dtw = np.asarray(distances, dtype=np.float32)
        if len(dtw) > 1 and dtw.max() > dtw.min():
            dtw_similarity = 1.0 - (dtw - dtw.min()) / (dtw.max() - dtw.min() + 1e-12)
        else:
            dtw_similarity = np.ones_like(dtw)
        final = self.cosine_weight * ((cosine[indices] + 1.0) / 2.0) + self.dtw_weight * dtw_similarity
        order = np.argsort(final)[::-1]
        return indices[order], cosine[indices][order], dtw[order], final[order]

    def _extract(
        self, frames: list[np.ndarray], source_indices: np.ndarray, frames_per_second: float,
    ) -> np.ndarray:
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.task_asset)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_face_detection_confidence=.45, min_face_landmarks_confidence=.45,
            min_pose_detection_confidence=.45, min_pose_landmarks_confidence=.45,
            min_hand_landmarks_confidence=.45,
        )
        sequence: list[np.ndarray] = []
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
            last_timestamp = -1
            for frame, source_index in zip(frames, source_indices, strict=True):
                timestamp = round(int(source_index) * 1_000.0 / frames_per_second)
                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))), timestamp)
                def points(landmarks: object, count: int, indices: tuple[int, ...] | None = None) -> np.ndarray:
                    output = np.zeros((count, 4), dtype=np.float32)
                    items = getattr(landmarks, 'landmark', landmarks) or []
                    for output_index, item_index in enumerate(indices or tuple(range(count))):
                        if item_index >= len(items):
                            break
                        item = items[item_index]
                        output[output_index] = (item.x, item.y, item.z, 1.)
                    return output
                sequence.append(np.concatenate((points(result.pose_landmarks, 33), points(result.left_hand_landmarks, 21), points(result.right_hand_landmarks, 21), points(result.face_landmarks, 40, FACE_IDX))))
        return self._resample(self._normalize(np.stack(sequence)))

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
        prediction = self.predict_frames(frames, frames_per_second=frames_per_second)
        return replace(
            prediction,
            diagnostics={
                'video_decode_ms': round((time.perf_counter() - started) * 1_000, 2),
                **prediction.diagnostics,
            },
        )

    def predict_frames(self, frames: list[np.ndarray], frames_per_second: float = 30.0) -> Prediction:
        if not frames:
            raise ValueError('The submitted clip has no decodable frames.')
        started = time.perf_counter()
        raw_frame_count = len(frames)
        sampled_indices = self._uniform_indices(raw_frame_count)
        sampled_frames = [frames[int(index)] for index in sampled_indices]
        sequence = self._extract(sampled_frames, sampled_indices, frames_per_second)
        preprocess_ms = (time.perf_counter() - started) * 1000
        coverage = float(np.any(sequence[:, 33:75, 3] > .5, axis=1).mean())
        embedding = self.session.run(None, {'keypoints': sequence[None].astype(np.float32)})[0][0]
        cosine = embedding @ self.prototypes.T
        shortlist = np.argsort(cosine)[::-1][:self.dtw_topk]
        raw_top, raw_margin = float(cosine[shortlist[0]]), float(cosine[shortlist[0]] - cosine[shortlist[1]])
        unknown = raw_top < self.cosine_threshold or raw_margin < self.margin_threshold
        if self.reference_sequences is None or self.reference_labels is None:
            order = shortlist
            dtw_distances = np.zeros(len(shortlist), dtype=np.float32)
            final_scores = (cosine[shortlist] + 1.0) / 2.0
        else:
            order, _reranked_cosine, dtw_distances, final_scores = self._dtw_rerank(
                sequence, shortlist, cosine,
            )
        candidates = tuple(
            Candidate(self.labels[int(self.prototype_labels[index])], float(final_scores[position]))
            for position, index in enumerate(order[:3])
        )
        winner = candidates[0]
        return Prediction(
            label='unknown' if unknown else winner.label, text='Không xác định' if unknown else winner.label,
            confidence=0.0 if unknown else winner.confidence,
            margin=float(np.clip(final_scores[0] - final_scores[1], 0., 1.)),
            landmark_coverage=coverage, top_k=candidates,
            diagnostics={'input_frames': raw_frame_count, 'mediapipe_input_frames': len(sampled_frames), 'sampled_frames': SEQ_LEN,
                         'preprocess_contract': 'vsl_lowshot_holistic_115_shoulder_normalized_v1',
                         'mediapipe_preprocess_ms': round(preprocess_ms, 2), 'raw_top1_cosine': round(raw_top, 6),
                         'raw_margin': round(raw_margin, 6), 'dtw_topk': self.dtw_topk,
                         'dtw_best_distance': round(float(dtw_distances[0]), 6), 'unknown_gate': unknown,
                         'device': str(self.device)},
            accepted=not unknown,
        )
