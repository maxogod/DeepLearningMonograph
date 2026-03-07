from os import path
from tqdm import tqdm
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
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scaler: amp.GradScaler,
        train: DataLoader,
        test: DataLoader | None = None,
        start_epoch: int = 0,
    ):
        self.save_model = config.train_config.save_model
        if config.train_config.save_model:
            file_operations.mkdir(config.file_paths.model_save_path)
            self.save_path = path.join(
                config.file_paths.model_save_path,
                config.train_config.model_name,
            )
            self.best_save_path = path.join(
                config.file_paths.model_save_path,
                f"best_{config.train_config.model_name}",
            )

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler

        self.start_epoch = start_epoch
        self.train_loader = train
        self.test_loader = test

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)

        self.model.to(self.device)
        self.criterion.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.scaler = amp.GradScaler(self.device_type)  # type: ignore

    def fit(self, num_epochs: int):
        logger.info(
            f"Starting training for {num_epochs} epochs on device: {self.device_type}"
        )
        best_loss = float("inf")

        epoch_bar = tqdm(
            range(self.start_epoch, num_epochs), desc="Training", unit="epoch"
        )

        for current_epoch in epoch_bar:
            time_start = time.time()

            loss = self._fit_epoch()
            val_loss = self._validate_epoch()

            time_end = time.time()

            epoch_bar.set_postfix(
                {
                    "train_loss": f"{loss:.4f}",
                    "val_loss": f"{(val_loss or 0.0):.4f}",
                    "time": f"{time_end - time_start:.2f}s",
                }
            )

            if val_loss is not None and val_loss < best_loss and self.save_model:
                best_loss = val_loss
                self._save_checkpoint(self.best_save_path, current_epoch)

        if self.save_model:
            self._save_checkpoint(self.save_path, num_epochs - 1)

    def _fit_epoch(self) -> float:
        self.model.train()

        epoch_bar = tqdm(self.train_loader, desc="Batches", leave=False)

        train_loss = 0.0
        for imgs, masks in epoch_bar:
            imgs = imgs.to(self.device).float()
            masks = masks.to(self.device).float()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device_type):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < prev_scale:  # Detect overflow
                logger.warning("Gradient overflow detected. Skipping step.")

            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def _validate_epoch(self) -> float | None:
        if self.test_loader is None:
            return None

        self.model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in self.test_loader:
                imgs = imgs.to(self.device).float()
                masks = masks.to(self.device).float()

                with autocast(device_type=self.device_type):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)

                val_loss += loss.item()

        return val_loss / len(self.test_loader)

    def _save_checkpoint(self, path: str, epoch: int):
        file_operations.save_torch(
            path,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.scaler.state_dict(),
            epoch,
        )


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
