from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import Prediction
from .vsl30_keypoint_onnx_predictor import Vsl30KeypointOnnxPredictor


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


def definitions() -> dict[str, ModelDefinition]:
    return {
        'vsl30_v4_3': ModelDefinition(
            'vsl30_v4_3',
            'VSL-30 V4.3 keypoint classifier (30 Vietnamese glosses)',
            './models/vsl30_v4_3/vsl30_v4_3_main.onnx',
            './models/vsl30_v4_3/label_map.json',
            'vsl30_keypoint_classifier',
        ),
    }


def create_predictor(definition: ModelDefinition) -> Predictor:
    return Vsl30KeypointOnnxPredictor(definition.model_path, definition.labels_path)
