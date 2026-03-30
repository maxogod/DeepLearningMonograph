from os import path
from tqdm import tqdm
import time
from torch import amp, nn, optim
import torch
import numpy as np
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from src.config.config import Config
from src.utils import file_operations
from src.utils.device import get_device_type
from src.utils.logger import get_logger


logger = get_logger()


class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scaler: amp.GradScaler,  # type: ignore
        train: DataLoader,
        test: DataLoader | None = None,
        start_epoch: int = 0,
    ):
        self.save_model = config.train_config.save_model
        self.resume_training = config.train_config.resume_training
        self.resume_from = config.train_config.resume_from
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
            self.loss_history_path = config.train_config.loss_history_path
            file_operations.mkdir(self.loss_history_path)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train_config.num_epochs - start_epoch,
            eta_min=config.train_config.eta_min_lr,
        )
        self.scaler = scaler

        self.start_epoch = start_epoch
        self.train_loader = train
        self.test_loader = test

        self.device_type = get_device_type()
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
        epoch_history, train_loss_history, val_loss_history = (
            self._load_existing_loss_history()
        )

        best_loss = float("inf")
        if val_loss_history:
            finite_val_losses = np.array(val_loss_history, dtype=np.float32)
            finite_val_losses = finite_val_losses[np.isfinite(finite_val_losses)]
            if finite_val_losses.size > 0:
                best_loss = float(finite_val_losses.min())

        epoch_bar = tqdm(
            range(self.start_epoch, num_epochs), desc="Training", unit="epoch"
        )

        for current_epoch in epoch_bar:
            time_start = time.time()

            loss = self._fit_epoch()
            val_loss = self._validate_epoch()
            self.scheduler.step()

            # Keep aligned arrays for plotting and checkpoint-side persistence.
            epoch_history.append(current_epoch)
            train_loss_history.append(loss)
            val_loss_history.append(np.nan if val_loss is None else val_loss)

            time_end = time.time()

            epoch_bar.set_postfix(
                {
                    "train_loss": f"{loss:.4f}",
                    "val_loss": f"{(val_loss or 0.0):.4f}",
                    "time": f"{time_end - time_start:.2f}s",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if val_loss is not None and val_loss < best_loss and self.save_model:
                best_loss = val_loss
                self._save_checkpoint(
                    self.best_save_path,
                    current_epoch,
                    epoch_history,
                    train_loss_history,
                    val_loss_history,
                )

        if self.save_model:
            self._save_checkpoint(
                self.save_path,
                num_epochs - 1,
                epoch_history,
                train_loss_history,
                val_loss_history,
            )

    def _fit_epoch(self) -> float:
        self.model.train()

        epoch_bar = tqdm(self.train_loader, desc="Batches", leave=False)

        train_loss = 0.0
        for imgs, masks in epoch_bar:
            ncr_voxels = masks[:, 1].sum().item()
            if ncr_voxels < 500:
                continue

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

    def _save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        epoch_history: list[int],
        train_loss_history: list[float],
        val_loss_history: list[float],
    ) -> None:
        file_operations.save_torch(
            checkpoint_path,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.scaler.state_dict(),
            epoch,
        )

        history_rows = [
            [float(ep), train_l, val_l]
            for ep, train_l, val_l in zip(
                epoch_history,
                train_loss_history,
                val_loss_history,
            )
        ]
        loss_history = np.array(history_rows, dtype=np.float32)
        checkpoint_name = path.splitext(path.basename(checkpoint_path))[0]
        loss_path = path.join(self.loss_history_path, f"{checkpoint_name}_losses.npy")
        file_operations.save_npy(loss_path, loss_history)

    def _load_existing_loss_history(
        self,
    ) -> tuple[list[int], list[float], list[float]]:
        if not self.save_model or not self.resume_training or self.start_epoch <= 0:
            return [], [], []

        checkpoint_name = path.splitext(path.basename(self.resume_from))[0]
        loss_path = path.join(
            self.loss_history_path,
            f"{checkpoint_name}_losses.npy",
        )

        if not path.exists(loss_path):
            logger.warning(
                f"Expected resume loss history not found: {loss_path}. Starting with empty history."
            )
            return [], [], []

        history = file_operations.load_npy(loss_path)
        if history.ndim != 2 or history.shape[1] < 3:
            logger.warning(
                f"Invalid loss history format in {loss_path}; starting with empty history."
            )
            return [], [], []

        # Keep only epochs already completed before the resume point.
        history = history[history[:, 0] < float(self.start_epoch)]
        if history.size == 0:
            logger.info(
                f"Found {loss_path}, but no rows before resume epoch {self.start_epoch}."
            )
            return [], [], []

        logger.info(
            f"Loaded {len(history)} previous loss rows from {loss_path} for resumed training."
        )
        return (
            history[:, 0].astype(np.int64).tolist(),
            history[:, 1].astype(np.float32).tolist(),
            history[:, 2].astype(np.float32).tolist(),
        )
