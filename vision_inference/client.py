"""Baseten JSON inference with an optional multipart HTTP fallback."""
import base64
from urllib.parse import urlparse
import httpx
from .contract import filter_detections


class VisionClient:
    def __init__(self, url, key, fallback_url=None, *, classes, checkpoint_sha256, transport=None):
        if url:
            parsed = urlparse(url)
            if parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.api.baseten.co'):
                raise ValueError('YOLO_BASETEN_PREDICT_URL must be an HTTPS Baseten endpoint')
            if not key:
                raise ValueError('Baseten vision inference needs an API key')
        if fallback_url:
            parsed = urlparse(fallback_url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError('Invalid YOLO_FALLBACK_URL')
        if not url and not fallback_url:
            raise ValueError('Configure a Baseten YOLO endpoint or fallback endpoint')
        self.url, self.key, self.fallback_url = url, key, fallback_url
        self.classes, self.checkpoint_sha256 = classes, checkpoint_sha256
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10), transport=transport)

    def validate(self, result, width, height):
        if not isinstance(result, dict):
            raise ValueError("Expected a vision response object")
        if result.get('checkpoint_sha256', self.checkpoint_sha256) != self.checkpoint_sha256:
            raise ValueError('Unexpected YOLO checkpoint')
        result = dict(result)
        result['detections'] = filter_detections(result, width, height, self.classes)
        return result

    async def predict(self, data, width, height):
        if self.url:
            try:
                response = await self.http.post(self.url, headers={'Authorization': f'Api-Key {self.key}'},
                    json={'image': base64.b64encode(data).decode()})
                response.raise_for_status()
                result = self.validate(response.json(), width, height)
                return dict(result, provider='baseten', model_ref=self.url)
            except (httpx.HTTPError, ValueError, TypeError, KeyError):
                if not self.fallback_url:
                    raise
        response = await self.http.post(self.fallback_url, files={'file': ('frame.jpg',data,'application/octet-stream')})
        response.raise_for_status()
        result = self.validate(response.json(), width, height)
        return dict(result, provider='fallback', model_ref=self.fallback_url)

    async def close(self):
        await self.http.aclose()
