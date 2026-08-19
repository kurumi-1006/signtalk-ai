from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import Settings
from .inference.model_registry import Predictor, create_predictor, definitions
from .transport.http import ApiTransport
from .transport.signer import DeviceSigner

logger = logging.getLogger(__name__)


def _video_for_inference(source: Path) -> Path:
    if source.suffix.casefold() not in {'.mp4', '.webm', '.mov'}:
        raise ValueError('Upload an MP4, WebM, or MOV video.')
    # OpenCV in MediaPipe's dependency already decodes browser video formats.
    # Avoid bundling a second 25 MB ffmpeg binary on the constrained Uno Q.
    return source


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title='SignTalk Edge AI')
    app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.cors_origins.split(',')], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    available_models = definitions()
    if settings.active_model_id not in available_models:
        raise ValueError(f'Unknown ACTIVE_MODEL_ID: {settings.active_model_id}')
    active_model_id = settings.active_model_id
    predictor: Predictor = create_predictor(available_models[active_model_id])
    predictor_lock = Lock()
    transport = ApiTransport(settings.api_url, settings.device_id, settings.device_key_id, DeviceSigner(settings.device_secret))
    publisher = ThreadPoolExecutor(max_workers=1, thread_name_prefix='signtalk-publisher')
    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='signtalk-inference')

    def publish_event(event: dict[str, object]) -> None:
        try:
            transport.send(event)
        except Exception:
            logger.exception('Recognition succeeded but background publishing failed')

    @app.get('/health')
    def health() -> dict[str, object]:
        return {'status': 'ok', 'model_id': active_model_id, 'model': predictor.model_name, 'inputs': sorted(predictor.input_names)}

    @app.get('/models')
    def models() -> dict[str, object]:
        return {'active_model_id': active_model_id, 'models': [
            {'id': model.id, 'name': model.display_name, 'type': model.predictor_type}
            for model in available_models.values()
        ]}

    @app.post('/models/{model_id}/activate')
    def activate_model(model_id: str) -> dict[str, object]:
        nonlocal active_model_id, predictor
        if model_id not in available_models:
            raise HTTPException(status_code=404, detail=f'Unknown model: {model_id}')
        # Create before swapping, so a broken model never takes down inference.
        replacement = create_predictor(available_models[model_id])
        with predictor_lock:
            predictor = replacement
            active_model_id = model_id
        return {'active_model_id': active_model_id, 'model': predictor.model_name}

    @app.post('/predict')
    async def predict(
        device_id: str | None = Form(default=None),
        video: UploadFile | None = File(default=None),  # noqa: B008
        frames: list[UploadFile] = File(default=[]),  # noqa: B008
    ) -> dict[str, object]:
        request_started = time.perf_counter()
        request_diagnostics: dict[str, object] = {}
        if device_id and device_id != settings.device_id:
            raise HTTPException(status_code=409, detail='device_id must match the Edge AI device configuration.')
        with predictor_lock:
            active_predictor = predictor
            model_id = active_model_id
        try:
            if frames:
                decode_started = time.perf_counter()
                if not 4 <= len(frames) <= 60:
                    raise ValueError('Submit between 4 and 60 JPEG frames.')
                decoded_frames: list[np.ndarray] = []
                for frame_upload in frames:
                    if frame_upload.content_type and not frame_upload.content_type.startswith('image/'):
                        raise ValueError('Every submitted frame must be an image.')
                    frame_payload = await frame_upload.read()
                    decoded = cv2.imdecode(np.frombuffer(frame_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None:
                        decoded_frames.append(decoded)
                if len(decoded_frames) < 4:
                    raise ValueError('The browser did not submit enough decodable frames.')
                request_diagnostics = {
                    'input_mode': 'jpeg_frames',
                    'received_frames': len(frames),
                    'decoded_frames': len(decoded_frames),
                    'jpeg_decode_ms': round((time.perf_counter() - decode_started) * 1_000, 2),
                }
                prediction = await asyncio.get_running_loop().run_in_executor(
                    inference_executor,
                    active_predictor.predict_frames,
                    decoded_frames,
                )
            elif video is not None:
                if video.content_type and not video.content_type.startswith('video/'):
                    raise ValueError('Submit an MP4/WebM video clip.')
                suffix = Path(video.filename or 'clip.mp4').suffix or '.mp4'
                payload = await video.read()
                if len(payload) < 1_024:
                    raise ValueError('The recorded clip is empty or too small.')
                with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                    temporary.write(payload)
                    temporary_path = Path(temporary.name)
                inference_path = temporary_path
                try:
                    preparation_started = time.perf_counter()
                    inference_path = _video_for_inference(temporary_path)
                    preparation_ms = (time.perf_counter() - preparation_started) * 1_000
                    prediction = await asyncio.get_running_loop().run_in_executor(
                        inference_executor,
                        active_predictor.predict_video,
                        str(inference_path),
                    )
                    request_diagnostics = {
                        'input_mode': 'video_upload',
                        'upload_bytes': len(payload),
                        'source_extension': suffix.casefold(),
                        'video_prepare_ms': round(preparation_ms, 2),
                    }
                finally:
                    inference_path.unlink(missing_ok=True)
                    temporary_path.unlink(missing_ok=True)
            else:
                raise ValueError('Submit either a video clip or a sequence of JPEG frames.')
        except Exception as error:
            logger.exception('%s prediction failed', active_predictor.model_name)
            raise HTTPException(status_code=422, detail=f'Unable to recognize this clip: {error}') from error
        accepted = (
            prediction.accepted
            if prediction.accepted is not None
            else prediction.confidence >= settings.min_word_confidence
            and prediction.margin >= settings.min_word_margin
        ) and prediction.landmark_coverage >= settings.min_landmark_coverage
        event = {
            'schemaVersion': 1,
            'eventId': str(uuid4()),
            'eventType': 'recognition.confirmed',
            'deviceId': settings.device_id,
            'occurredAt': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            'payload': {
                'label': prediction.label,
                'text': prediction.text,
                'confidence': prediction.confidence,
                'margin': prediction.margin,
                'landmarkCoverage': prediction.landmark_coverage,
                'accepted': accepted,
                'topK': [{'label': item.label, 'confidence': item.confidence} for item in prediction.top_k],
            },
        }
        publisher.submit(publish_event, event)
        diagnostics = {
            **request_diagnostics,
            **prediction.diagnostics,
            'model_id': model_id,
            'model_name': active_predictor.model_name,
            'confidence': round(prediction.confidence, 6),
            'margin': round(prediction.margin, 6),
            'accepted': accepted,
            'total_request_ms': round((time.perf_counter() - request_started) * 1_000, 2),
        }
        return {
            'event': event,
            'published': False,
            'publish_status': 'queued',
            'diagnostics': diagnostics,
        }

    return app
