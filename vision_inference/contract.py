"""Shared image/detection contract for Baseten, fallback and the API."""
import base64
import binascii
from io import BytesIO
import math

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 16_000_000


def decode_image(encoded):
    from PIL import Image
    if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 100:
        raise ValueError('Expected a base64 JPEG/PNG image of at most 4 MiB')
    if encoded.startswith('data:image/'):
        encoded = encoded.split(',', 1)[-1]
    try:
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError('Image exceeds 4 MiB')
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            if im.format not in ('JPEG', 'PNG') or width * height > MAX_PIXELS:
                raise ValueError('Expected JPEG/PNG of at most 16 megapixels')
            im.verify()
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid image') from exc
    return data, width, height


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def filter_detections(result, width, height, classes, pothole_conf=.4, crack_conf=.2):
    if not isinstance(result, dict) or result.get('stub') or result.get('error'):
        raise ValueError('Vision service returned an error or a synthetic stub')
    detections = result.get('detections')
    if not isinstance(detections, list) or len(detections) > 1000:
        raise ValueError('Invalid detection list')
    kept = []
    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError('Invalid detection')
        cls, confidence, box = (detection.get(k) for k in ('cls', 'conf', 'box'))
        if type(cls) is not int or cls not in classes or not number(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid class or confidence')
        if not isinstance(box, list) or len(box) != 4 or not all(number(v) for v in box):
            raise ValueError('Expected finite pixel xyxy bounding boxes')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Bounding box is outside the submitted image')
        label = classes[cls]
        if detection.get('label', label) != label:
            raise ValueError('Model class names differ from configured weights')
        threshold = crack_conf if label == 'crack' else pothole_conf
        if confidence >= threshold:
            kept.append(dict(cls=cls, label=label, conf=float(confidence), box=box))
    return kept
