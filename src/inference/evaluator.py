from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test: DataLoader,
    ):
        self.model = model
        self.test_loader = test

        self.iou_metric = MeanIoU(include_background=True, reduction="mean")
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def validate_model(self) -> Tuple[float, float]:
        self.model.eval()
        self.iou_metric.reset()
        self.dice_metric.reset()

        with torch.no_grad():
            for imgs, masks in self.test_loader:
                imgs = imgs.to(self.device).float()
                masks = masks.to(self.device)

                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True)
                targets = torch.argmax(masks, dim=1, keepdim=True)

                self.iou_metric(preds, targets)
                self.dice_metric(preds, targets)

        mean_iou, _ = self.iou_metric.aggregate()
        mean_dice, _ = self.dice_metric.aggregate()

        mean_iou = mean_iou.item()
        mean_dice = mean_dice.item()

        return mean_iou, mean_dice
