import torch
from src.utils import file_operations, logger
from src.utils.device import get_device
from torch.amp.autocast_mode import autocast
from src.models.unet_3d import UNet3D
from src.inference.base_predictor import BasePredictor

log = logger.get_logger()


class EnsemblePredictor(BasePredictor):
    def __init__(self, model_paths: list[str]):
        self.models = []
        self.device = get_device()

        for p in model_paths:
            state_dict, _, _, _ = file_operations.load_torch(p)
            model = UNet3D(in_channels=3, num_classes=4)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        log.info(f"Loaded {len(self.models)} models for ensemble")

    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = imgs.to(self.device).float()

        avg_probs = None
        with torch.no_grad():
            for model in self.models:
                with autocast(device_type=self.device.type):
                    logits = model(imgs)
                probs = torch.softmax(logits, dim=1)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

        if avg_probs is None:
            raise ValueError("No models were loaded for prediction.")

        return avg_probs / len(self.models)  # averaged probabilities
