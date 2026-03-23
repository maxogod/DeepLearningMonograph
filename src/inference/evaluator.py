from typing import Tuple
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
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

        self.iou_metric = MeanIoU(
            include_background=True, reduction="mean", get_not_nans=False
        )
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, imgs: torch.Tensor) -> torch.Tensor:
        use_amp = self.device.type == "cuda"
        with autocast(device_type="cuda", enabled=use_amp):
            return self.model(imgs)

    def validate_model(self) -> Tuple[float, float]:
        self.iou_metric.reset()
        self.dice_metric.reset()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for imgs, masks in tqdm(self.test_loader):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                try:
                    outputs = self._forward(imgs)
                except torch.OutOfMemoryError:
                    if self.device.type != "cuda" or imgs.shape[0] == 1:
                        raise
                    torch.cuda.empty_cache()
                    chunked_outputs = []
                    for idx in range(imgs.shape[0]):
                        chunked_outputs.append(self._forward(imgs[idx : idx + 1]))
                    outputs = torch.cat(chunked_outputs, dim=0)

                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                targets = torch.argmax(masks, dim=1)

                preds = F.one_hot(preds, num_classes=4).permute(0, 4, 1, 2, 3).float()
                targets = (
                    F.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()
                )

                self.dice_metric(preds, targets)
                self.iou_metric(preds, targets)

        mean_iou = self.iou_metric.aggregate().item()  # type: ignore
        mean_dice = self.dice_metric.aggregate().item()  # type: ignore

        return mean_iou, mean_dice
