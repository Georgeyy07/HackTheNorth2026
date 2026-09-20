"""Async transport for the stateless Baseten ensemble; no torch in the backend."""
import hashlib
from pathlib import Path
import httpx
import numpy as np

ENSEMBLE_SHA = hashlib.sha256(Path(__file__).with_name('ensemble.json').read_bytes()).hexdigest()


def validate_prediction(result, count):
    if result.get('ensemble_sha256') != ENSEMBLE_SHA or result.get('patch_indices') != [61, 62, 63]:
        raise ValueError('Unexpected deployed ensemble or patch geometry')
    q = np.asarray(result['quality_probability'], dtype=float)
    p = np.asarray(result['disturbance_probability'], dtype=float)
    valid = np.asarray(result['patch_valid'])
    if q.shape != (count, 3, 3) or p.shape != (count, 3) or valid.shape != (count, 3) or valid.dtype != bool:
        raise ValueError('Invalid inference response shape')
    if not (np.isfinite(q).all() and np.isfinite(p).all() and ((q >= 0) & (q <= 1)).all()
            and ((p >= 0) & (p <= 1)).all() and np.allclose(q.sum(-1), 1, atol=1e-5)):
        raise ValueError('Invalid inference probabilities')
    return q, p, valid


class BasetenClient:
    def __init__(self, url, key):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not parsed.hostname or not parsed.hostname.endswith('.api.baseten.co'):
            raise ValueError('Expected an HTTPS Baseten prediction URL')
        self.url = url
        self.client = httpx.AsyncClient(headers={'Authorization': f'Api-Key {key}'},
                                       timeout=httpx.Timeout(120, connect=10))

    async def predict(self, payload):
        # A failed request leaves stream state unchanged; the device can retry its batch.
        response = await self.client.post(self.url, json=payload)
        response.raise_for_status()
        result = response.json()
        validate_prediction(result, len(payload['windows']))
        return result

    async def close(self):
        await self.client.aclose()
