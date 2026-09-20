"""FastAPI fallback adapted from adding-dockerfile; missing weights fail startup."""
import base64
from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from vision_inference.contract import MAX_IMAGE_BYTES


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        nonlocal predictor
        if predictor is None:
            from vision_inference.model import YoloPredictor
            weights = os.environ.get('YOLO_WEIGHTS', str(Path(__file__).resolve().parents[1]/'models/yolo26/best.pt'))
            predictor = YoloPredictor(weights)
        if os.environ.get('SENTRY_DSN'):
            import sentry_sdk
            sentry_sdk.init(dsn=os.environ['SENTRY_DSN'], send_default_pii=False, traces_sample_rate=.1)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    def health():
        return dict(status='ok', model_loaded=predictor is not None)

    @app.post('/predict')
    async def predict(file: UploadFile):
        contents = await file.read(MAX_IMAGE_BYTES + 1)
        await file.close()
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(413, 'Image exceeds 4 MiB')
        try:
            return await run_in_threadpool(predictor.predict, {'image': base64.b64encode(contents).decode()})
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
