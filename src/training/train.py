from os import path
import time
from torch import amp, nn, optim
import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from src.config.config import Config
from src.utils import file_operations
from src.utils.logger import get_logger

logger = get_logger()


class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        criterion: nn.Module,
        lr: float,
        train: DataLoader,
        test: DataLoader | None = None,
    ):
        self.config = config
        file_operations.mkdir(self.config.file_paths.model_save_path)
        self.save_path = path.join(
            self.config.file_paths.model_save_path, self.config.train_config.model_name
        )
        self.best_save_path = path.join(
            self.config.file_paths.model_save_path,
            "best_",
            self.config.train_config.model_name,
        )

        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = amp.GradScaler("cuda")  # type: ignore

        self.train_loader = train
        self.test_loader = test

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, num_epochs: int):
        best_loss = float("inf")

        self.model.to(self.device)
        for epoch in range(num_epochs):
            time_start = time.time()

            loss = self._fit_epoch()
            val_loss = self._validate_epoch()

            time_end = time.time()

            logger.debug(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {time_end - time_start:.2f}s"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                file_operations.save_torch(
                    self.best_save_path,
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                )

        file_operations.save_torch(
            self.save_path, self.model.state_dict(), self.optimizer.state_dict()
        )

    def _fit_epoch(self) -> float:
        self.model.train()

        train_loss = 0.0
        for imgs, masks in self.train_loader:
            imgs = imgs.to(self.device).float()
            masks = masks.to(self.device).float()

            self.optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def _validate_epoch(self) -> float:
        if self.test_loader is None:
            return 0.0

        self.model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in self.test_loader:
                imgs = imgs.to(self.device).float()
                masks = masks.to(self.device).float()

                with autocast(device_type="cuda"):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)

                val_loss += loss.item()

        return val_loss / len(self.test_loader)


# probs = torch.softmax(outputs, dim=1)  # probabilities
# preds = torch.argmax(probs, dim=1)
# correct = (preds == masks).float()
# accuracy = correct.sum() / correct.numel()

# def compute_iou(preds, masks, num_classes):
#     ious = []

#     for cls in range(num_classes):
#         pred_cls = (preds == cls)
#         mask_cls = (masks == cls)

#         intersection = (pred_cls & mask_cls).sum().float()
#         union = (pred_cls | mask_cls).sum().float()

#         if union == 0:
#             continue

#         ious.append(intersection / union)

#     return torch.mean(torch.stack(ious))
# iou = compute_iou(preds, masks, num_classes=4)

# OR install torchmetrics
# from torchmetrics.classification import MulticlassAccuracy
# from torchmetrics.classification import MulticlassJaccardIndex
# accuracy_metric = MulticlassAccuracy(num_classes=4)
# iou_metric = MulticlassJaccardIndex(num_classes=4)
