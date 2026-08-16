from pathlib import Path

import numpy as np

from src.inference.mock_predictor import MockPredictor
from src.inference.v6_predictor import V6Predictor
from src.inference.vsl_metric_lowshot_predictor import MAX_RAW_FRAMES, VslMetricLowShotPredictor
from src.storage.outbox import Outbox
from src.transport.signer import DeviceSigner


def test_predictor() -> None:
    assert MockPredictor().predict().confidence == 0.95


def test_v6_square_crop_preserves_requested_size() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cropped = V6Predictor._crop_square(frame, center_x=0.5, center_y=0.5, side=400)
    assert cropped.shape == (400, 400, 3)


def test_signer_is_deterministic() -> None:
    assert DeviceSigner('x').sign('POST', '/x', 't', 'n', {'a': 1}) == DeviceSigner('x').sign('POST', '/x', 't', 'n', {'a': 1})

def test_outbox(tmp_path: Path) -> None:
    outbox = Outbox(str(tmp_path / 'outbox.db'))
    outbox.put('1', '{}')
    assert len(outbox.pending()) == 1
    outbox.mark_sent('1')
    assert not outbox.pending()


def test_lowshot_uniform_frame_sampling_matches_training_limit() -> None:
    frames = [np.full((1, 1, 1), index, dtype=np.uint8) for index in range(240)]
    sampled = VslMetricLowShotPredictor._uniform_frames(frames)
    assert len(sampled) == MAX_RAW_FRAMES
    assert int(sampled[0][0, 0, 0]) == 0
    assert int(sampled[-1][0, 0, 0]) == 239
