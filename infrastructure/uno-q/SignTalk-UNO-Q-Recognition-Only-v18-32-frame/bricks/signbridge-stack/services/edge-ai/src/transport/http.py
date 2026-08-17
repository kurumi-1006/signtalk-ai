from datetime import datetime, timezone
from uuid import uuid4
import httpx
from .signer import DeviceSigner
class ApiTransport:
    def __init__(self, api_url: str, device_id: str, key_id: str, signer: DeviceSigner): self.api_url, self.device_id, self.key_id, self.signer = api_url.rstrip('/'), device_id, key_id, signer
    def send(self, event: dict[str, object]) -> bool:
        timestamp, nonce, path = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), str(uuid4()), '/recognition/events'
        signature = self.signer.sign('POST', path, timestamp, nonce, event)
        response = httpx.post(f'{self.api_url}{path}', json=event, headers={'x-device-id': self.device_id, 'x-key-id': self.key_id, 'x-timestamp': timestamp, 'x-nonce': nonce, 'x-signature': signature}, timeout=10)
        return response.is_success
