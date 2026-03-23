from typing import Tuple
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from src.utils import logger
from src.utils.device import get_device

log = logger.get_logger()


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test: DataLoader,
    ):
        self.model = model
        self.test_loader = test

        self.iou_metric = MeanIoU(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=True
        )

        self.dice_per_class = DiceMetric(
            include_background=False, reduction="mean_batch", get_not_nans=True
        )

        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, imgs: torch.Tensor) -> torch.Tensor:
        use_amp = self.device.type == "cuda"
        with autocast(device_type=self.device.type, enabled=use_amp):
            return self.model(imgs)

    def validate_model(self) -> Tuple[float, float]:
        self.iou_metric.reset()
        self.dice_metric.reset()
        self.dice_per_class.reset()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for imgs, masks in tqdm(self.test_loader):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                try:
                    outputs = self.predict_with_tta(imgs)
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
                self.dice_per_class(preds, targets)
                self.iou_metric(preds, targets)

        mean_iou = self.iou_metric.aggregate().item()  # type: ignore
        mean_dice, not_nans = self.dice_metric.aggregate()  # type: ignore
        per_class, _ = self.dice_per_class.aggregate()

        log.info(f"Per-class Dice (NCR, ED, ET): {per_class.tolist()}")
        log.info(f"Not-NaN batches: {not_nans}")

        return mean_iou, mean_dice.item()

    def predict_with_tta(self, imgs: torch.Tensor) -> torch.Tensor:
        preds = torch.softmax(self.model(imgs), dim=1)

        for flip_dim in [2, 3, 4]:  # flip D, H, W
            flipped = torch.flip(imgs, dims=[flip_dim])
            preds += torch.softmax(self.model(flipped), dim=1).flip(dims=[flip_dim])

        return preds / 4.0  # average over original + 3 flips
