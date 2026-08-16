import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from .config.settings import Settings
from .inference.mock_predictor import MockPredictor
from .serve import create_app
from .storage.outbox import Outbox
from .transport.http import ApiTransport
from .transport.signer import DeviceSigner


def event(settings: Settings) -> dict[str, object]:
    prediction = MockPredictor().predict()
    return {'schemaVersion': 1, 'eventId': str(uuid4()), 'eventType': 'recognition.confirmed', 'deviceId': settings.device_id, 'occurredAt': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), 'payload': {'label': prediction.label, 'text': prediction.text, 'confidence': prediction.confidence}}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--interval', type=int, default=3)
    parser.add_argument('--port', type=int, help='Override the configured HTTP port when serving.')
    parser.add_argument('--raw-class-ids', action='store_true', help='Serve Multi-VSL predictions as stable vsl_gloss IDs.')
    args = parser.parse_args()
    settings = Settings.model_validate({})
    if args.serve:
        if args.port:
            settings = settings.model_copy(update={'edge_port': args.port})
        if args.raw_class_ids:
            os.environ['MULTI_VSL_LABELS_PATH'] = ''
        import uvicorn
        uvicorn.run(create_app(settings), host=settings.edge_host, port=settings.edge_port)
        return
    logging.basicConfig(level=settings.log_level, format='%(message)s')
    outbox = Outbox(settings.outbox_db_path)
    transport = ApiTransport(settings.api_url, settings.device_id, settings.device_key_id, DeviceSigner(settings.device_secret))
    while True:
        payload = event(settings)
        outbox.put(str(payload['eventId']), json.dumps(payload))
        for event_id, item in outbox.pending():
            if transport.send(json.loads(item)):
                outbox.mark_sent(event_id)
                logging.info(json.dumps({'event': 'sent', 'eventId': event_id}))
        time.sleep(args.interval)
if __name__ == '__main__':
    main()
