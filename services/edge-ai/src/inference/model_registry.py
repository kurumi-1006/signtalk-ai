from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import Prediction
from .vsl30_keypoint_onnx_predictor import Vsl30KeypointOnnxPredictor
from .vsl_metric_lowshot_predictor import VslMetricLowShotPredictor


class Predictor(Protocol):
    model_name: str
    input_names: set[str]
    def predict_video(self, video_path: str) -> Prediction: ...
    def predict_frames(self, frames: list[object]) -> Prediction: ...


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    display_name: str
    model_path: str
    labels_path: str
    predictor_type: str


def definitions(default_model_path: str, default_labels_path: str) -> dict[str, ModelDefinition]:
    return {
        'vsl_metric_lowshot': ModelDefinition(
            'vsl_metric_lowshot',
            'VSL low-shot metric encoder (MediaPipe Holistic)',
            './models/vsl_metric_lowshot/vsl_metric_encoder.onnx',
            './models/vsl_metric_lowshot/labels.csv',
            'vsl_metric_lowshot',
        ),
        'vsl30_keypoint_classifier': ModelDefinition(
            'vsl30_keypoint_classifier',
            'VSL-30 keypoint classifier (30 Vietnamese glosses)',
            './models/vsl30_keypoint_classifier/vsl30_keypoint_classifier.onnx',
            './models/vsl30_keypoint_classifier/label_map.json',
            'vsl30_keypoint_classifier',
        ),
        'vsl30_v4_3': ModelDefinition(
            'vsl30_v4_3',
            'VSL-30 V4.3 keypoint classifier (30 Vietnamese glosses)',
            './models/vsl30_v4_3/vsl30_v4_3_main.onnx',
            './models/vsl30_v4_3/label_map.json',
            'vsl30_keypoint_classifier',
        ),
    }


def create_predictor(definition: ModelDefinition) -> Predictor:
    if definition.predictor_type == 'vsl30_keypoint_classifier':
        return Vsl30KeypointOnnxPredictor(definition.model_path, definition.labels_path)
    return VslMetricLowShotPredictor(definition.model_path, definition.labels_path)
