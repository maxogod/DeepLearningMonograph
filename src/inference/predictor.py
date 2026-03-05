from torch import nn
import torch


class Predictor:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        imgs = imgs.to(self.device).float()

        outputs = self.model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds
