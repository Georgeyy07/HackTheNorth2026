"""Baseten runs the stateless network; FastAPI owns all per-drive state."""
from pathlib import Path


class Model:
    def __init__(self, data_dir, **kwargs):
        self.data_dir = Path(data_dir)

    def load(self):
        import torch
        from imu_inference.model import EnsemblePredictor
        torch.set_num_threads(2)
        self.predictor = EnsemblePredictor(self.data_dir/'models', device='cpu')

    def predict(self, model_input):
        return self.predictor.predict(model_input)
