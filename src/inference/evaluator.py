from typing import Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from src.utils import logger
from src.utils.device import get_device
from src.inference.base_predictor import BasePredictor

log = logger.get_logger()


class Evaluator:
    def __init__(
        self,
        test: DataLoader,
        predictor: BasePredictor,
    ):
        self.predictor = predictor
        self.test_loader = test

        self.iou_metric = MeanIoU(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.iou_per_class = MeanIoU(
            include_background=False, reduction="mean_batch", get_not_nans=True
        )

        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=True
        )
        self.dice_per_class = DiceMetric(
            include_background=False, reduction="mean_batch", get_not_nans=True
        )

        self.device = get_device()

    def _predict_single(self, img: torch.Tensor) -> torch.Tensor:
        """Fallback for OOM: run TTA on a single image."""
        return self.predict_with_tta(img.unsqueeze(0)).squeeze(0)

    def validate_model(self) -> Tuple[float, float]:
        self.iou_metric.reset()
        self.iou_per_class.reset()
        self.dice_metric.reset()
        self.dice_per_class.reset()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for imgs, masks in tqdm(self.test_loader):
                ncr_voxels = masks[:, 1].sum().item()
                if ncr_voxels < 500:
                    continue  # Skip when few NCR voxels to avoid skewing metrics

                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                try:
                    outputs = self.predict_with_tta(imgs)
                except torch.OutOfMemoryError:
                    if self.device.type != "cuda" or imgs.shape[0] == 1:
                        raise
                    torch.cuda.empty_cache()
                    outputs = torch.stack(
                        [
                            self._predict_single(imgs[idx])
                            for idx in range(imgs.shape[0])
                        ],
                        dim=0,
                    )

                preds = torch.argmax(outputs, dim=1)
                targets = torch.argmax(masks, dim=1)

                preds = F.one_hot(preds, num_classes=4).permute(0, 4, 1, 2, 3).float()
                targets = (
                    F.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()
                )

                self.dice_metric(preds, targets)
                self.dice_per_class(preds, targets)
                self.iou_metric(preds, targets)
                self.iou_per_class(preds, targets)

        mean_iou = self.iou_metric.aggregate().item()  # type: ignore
        per_class_iou, _ = self.iou_per_class.aggregate()
        mean_dice, not_nans = self.dice_metric.aggregate()  # type: ignore
        per_class_dice, _ = self.dice_per_class.aggregate()

        log.info(f"Per-class IoU (NCR, ED, ET): {per_class_iou.tolist()}")
        log.info(f"Per-class Dice (NCR, ED, ET): {per_class_dice.tolist()}")
        log.info(f"Not-NaN batches: {not_nans}")

        return mean_iou, mean_dice.item()

    def predict_with_tta(self, imgs: torch.Tensor) -> torch.Tensor:
        preds = self.predictor.predict(imgs)

        for flip_dim in [2, 3, 4]:  # flip D, H, W
            flipped = torch.flip(imgs, dims=[flip_dim])
            preds += self.predictor.predict(flipped).flip(dims=[flip_dim])

        return preds / 4.0  # average over original + 3 flips
