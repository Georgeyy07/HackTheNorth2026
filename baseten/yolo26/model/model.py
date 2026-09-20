from pathlib import Path


class Model:
    def __init__(self, **kwargs):
        self.data_dir = Path(kwargs['data_dir'])

    def load(self):
        import torch
        from vision_inference.model import YoloPredictor
        torch.set_num_threads(2)
        self.predictor = YoloPredictor(self.data_dir/'best.pt')

    def predict(self, payload):
        return self.predictor.predict(payload)
