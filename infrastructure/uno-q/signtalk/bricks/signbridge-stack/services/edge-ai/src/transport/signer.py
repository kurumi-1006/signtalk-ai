import hashlib
import hmac
import json
class DeviceSigner:
    def __init__(self, secret: str): self.secret = secret.encode()
    def sign(self, method: str, path: str, timestamp: str, nonce: str, body: object) -> str:
        # Match Node's JSON.stringify used by the NestJS HMAC guard: compact
        # separators and literal UTF-8 characters in the request body.
        canonical = '\n'.join([method, path, timestamp, nonce, json.dumps(body, ensure_ascii=False, separators=(',', ':'))])
        return hmac.new(self.secret, canonical.encode(), hashlib.sha256).hexdigest()
