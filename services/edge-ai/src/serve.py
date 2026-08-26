from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from numpy.typing import NDArray

from .config.settings import Settings
from .inference.model_registry import Predictor, create_predictor, definitions
from .trace import trace

logger = logging.getLogger(__name__)
Array = NDArray[Any]


def _video_for_inference(source: Path) -> Path:
    if source.suffix.casefold() not in {'.mp4', '.webm', '.mov'}:
        raise ValueError('Upload an MP4, WebM, or MOV video.')
    # OpenCV in MediaPipe's dependency already decodes browser video formats.
    # Avoid bundling a second 25 MB ffmpeg binary on the constrained Uno Q.
    return source


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title='SignTalk AI Edge AI')
    app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.cors_origins.split(',')], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    available_models = definitions()
    # This service deliberately ships one model only.  Do not let an old App
    # Lab brick environment select the removed vsl30_keypoint_classifier.
    active_model_id = 'vsl30_v4_3'
    predictor: Predictor = create_predictor(available_models[active_model_id])
    predictor_lock = Lock()
    # Keep CPU-heavy MediaPipe/ONNX work off FastAPI's event loop. One worker
    # also prevents concurrent requests from competing for UNO Q memory/CPU.
    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='signtalk-ai-inference')

    @app.get('/health')
    def health() -> dict[str, object]:
        return {'status': 'ok', 'model_id': active_model_id, 'model': predictor.model_name, 'inputs': sorted(predictor.input_names)}

    @app.get('/models')
    def models() -> dict[str, object]:
        return {'active_model_id': active_model_id, 'models': [
            {'id': model.id, 'name': model.display_name, 'type': model.predictor_type}
            for model in available_models.values()
        ]}

    @app.post('/predict')
    async def predict(
        video: UploadFile | None = File(default=None),  # noqa: B008
        frames: list[UploadFile] = File(default=[]),  # noqa: B008
    ) -> dict[str, object]:
        request_started = time.perf_counter()
        request_id = uuid4().hex[:12]
        request_diagnostics: dict[str, object] = {}
        with predictor_lock:
            active_predictor = predictor
            model_id = active_model_id
        trace(
            logger,
            request_id,
            '00',
            'REQUEST RECEIVED',
            model_id=model_id,
            input_frames=len(frames),
            has_video=bool(video),
            model=active_predictor.model_name,
        )
        try:
            if frames:
                decode_started = time.perf_counter()
                if not 4 <= len(frames) <= 60:
                    raise ValueError('Submit between 4 and 60 JPEG frames.')
                trace(logger, request_id, '01', 'INPUT VALIDATION', input_mode='jpeg_frames', frame_count=len(frames))
                decoded_frames: list[Array] = []
                total_bytes = 0
                for frame_index, frame_upload in enumerate(frames):
                    if frame_upload.content_type and not frame_upload.content_type.startswith('image/'):
                        raise ValueError('Every submitted frame must be an image.')
                    frame_payload = await frame_upload.read()
                    total_bytes += len(frame_payload)
                    decoded = cv2.imdecode(np.frombuffer(frame_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None:
                        decoded_frames.append(decoded)
                    else:
                        logger.warning(
                            '[predict][%s] frame=%d decode_failed bytes=%d',
                            request_id,
                            frame_index,
                            len(frame_payload),
                        )
                if len(decoded_frames) < 4:
                    raise ValueError('The browser did not submit enough decodable frames.')
                request_diagnostics = {
                    'input_mode': 'jpeg_frames',
                    'received_frames': len(frames),
                    'decoded_frames': len(decoded_frames),
                    'upload_bytes': total_bytes,
                    'jpeg_decode_ms': round((time.perf_counter() - decode_started) * 1_000, 2),
                }
                trace(
                    logger,
                    request_id,
                    '02',
                    'JPEG DECODE COMPLETE',
                    decoded_frames=f'{len(decoded_frames)}/{len(frames)}',
                    total_bytes=total_bytes,
                    elapsed_ms=request_diagnostics['jpeg_decode_ms'],
                    first_frame_shape=decoded_frames[0].shape,
                )
                prediction = await asyncio.get_running_loop().run_in_executor(
                    inference_executor,
                    active_predictor.predict_frames,
                    decoded_frames,
                    30.0,
                    request_id,
                )
            elif video is not None:
                if video.content_type and not video.content_type.startswith('video/'):
                    raise ValueError('Submit an MP4/WebM video clip.')
                suffix = Path(video.filename or 'clip.mp4').suffix or '.mp4'
                payload = await video.read()
                if len(payload) < 1_024:
                    raise ValueError('The recorded clip is empty or too small.')
                trace(
                    logger,
                    request_id,
                    '01',
                    'VIDEO INPUT VALIDATION',
                    filename=video.filename or 'clip.mp4',
                    content_type=video.content_type or 'unknown',
                    upload_bytes=len(payload),
                )
                with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                    temporary.write(payload)
                    temporary_path = Path(temporary.name)
                inference_path = temporary_path
                try:
                    preparation_started = time.perf_counter()
                    inference_path = _video_for_inference(temporary_path)
                    preparation_ms = (time.perf_counter() - preparation_started) * 1_000
                    trace(
                        logger,
                        request_id,
                        '02',
                        'VIDEO PREPARATION COMPLETE',
                        source_extension=suffix.casefold(),
                        elapsed_ms=round(preparation_ms, 2),
                    )
                    prediction = await asyncio.get_running_loop().run_in_executor(
                        inference_executor,
                        active_predictor.predict_video,
                        str(inference_path),
                        request_id,
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
            logger.exception('[TRACE][%s] %s prediction failed', request_id, active_predictor.model_name)
            raise HTTPException(status_code=422, detail=f'Unable to recognize this clip: {error}') from error
        # `accepted` is a product decision, not just the model's top-1 label:
        # confidence, class margin, and visible-hand coverage must all pass the
        # configured thresholds before the UI treats the gloss as reliable.
        accepted = (
            prediction.accepted
            if prediction.accepted is not None
            else prediction.confidence >= settings.min_word_confidence
            and prediction.margin >= settings.min_word_margin
        ) and prediction.landmark_coverage >= settings.min_landmark_coverage
        trace(
            logger,
            request_id,
            '10',
            'ACCEPTANCE DECISION',
            label=prediction.label,
            confidence=round(prediction.confidence, 6),
            margin=round(prediction.margin, 6),
            landmark_coverage=round(prediction.landmark_coverage, 6),
            confidence_threshold=settings.min_word_confidence,
            margin_threshold=settings.min_word_margin,
            coverage_threshold=settings.min_landmark_coverage,
            accepted=accepted,
        )
        event: dict[str, Any] = {
            'schemaVersion': 1,
            'eventId': str(uuid4()),
            'eventType': 'recognition.confirmed',
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
        trace(
            logger,
            request_id,
            '11',
            'RESPONSE COMPLETE',
            event_type=event['eventType'],
            label=event['payload']['label'],
            text=event['payload']['text'],
            confidence=event['payload']['confidence'],
            accepted=accepted,
            total_request_ms=diagnostics['total_request_ms'],
        )
        return {
            'event': event,
            'diagnostics': diagnostics,
        }

    return app
