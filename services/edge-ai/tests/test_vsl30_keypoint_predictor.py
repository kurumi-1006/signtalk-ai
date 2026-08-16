import json

import numpy as np
import pytest
import torch

from src.inference.types import Prediction
from src.inference.vsl30_keypoint_predictor import (
    N_POINTS,
    SEQ_LEN,
    Vsl30KeypointEncoder,
    Vsl30KeypointPredictor,
)


def test_encoder_rejects_an_input_shape_that_the_checkpoint_was_not_trained_on() -> None:
    with pytest.raises(ValueError, match='Expected keypoints'):
        Vsl30KeypointEncoder()(torch.zeros((1, SEQ_LEN, N_POINTS - 1, 4)))


def test_label_map_requires_exactly_the_checkpoint_vocabulary(tmp_path) -> None:
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'idx_to_label': {'0': 'Em'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='map every class'):
        Vsl30KeypointPredictor._load_labels(path)


def test_keypoint_tensor_produces_the_standard_top_k_output(monkeypatch) -> None:
    predictor = object.__new__(Vsl30KeypointPredictor)
    predictor.labels = {index: f'label-{index}' for index in range(30)}
    predictor.model = Vsl30KeypointEncoder().eval()
    # Use a classifier-shaped output to test the public adapter contract without
    # requiring MediaPipe or the external checkpoint in the source test suite.
    predictor.model = torch.nn.Sequential(predictor.model, torch.nn.Linear(256, 30)).eval()
    monkeypatch.setattr(predictor, '_extract', lambda *_: np.zeros((SEQ_LEN, N_POINTS, 4), dtype=np.float32))
    result = predictor.predict_frames([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)])
    assert isinstance(result, Prediction)
    assert len(result.top_k) == 3
    assert result.diagnostics['keypoint_tensor_shape'] == [1, 48, 75, 4]
