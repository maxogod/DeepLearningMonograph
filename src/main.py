import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from typing import Tuple
from src.architecture.unet_3d_checkpoint import UNet3DCheckpoint
from src.dataset.lazy_loader import LazyBratsDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def dice_score_multiclass(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    """
    pred: (B, C, D, H, W)
    target: (B, C, D, H, W) one-hot
    """
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean()

# Training function


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, scaler: GradScaler) -> Tuple[float, float]:
    model.train()
    epoch_loss, epoch_dice = 0.0, 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device).float()
        masks = masks.to(device).float()
        imgs = imgs.permute(0, 4, 1, 2, 3)
        masks = masks.permute(0, 4, 1, 2, 3)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_dice += dice_score_multiclass(outputs, masks).item()

    return epoch_loss / len(loader), epoch_dice / len(loader)

# Validation function


def validate_epoch(model: nn.Module, loader: DataLoader, criterion) -> Tuple[float, float]:
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()
            imgs = imgs.permute(0, 4, 1, 2, 3)
            masks = masks.permute(0, 4, 1, 2, 3)

            with autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_score_multiclass(outputs, masks).item()
    return val_loss / len(loader), val_dice / len(loader)

# Main training loop


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 10, lr: float = 1e-4):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda")

    for epoch in range(num_epochs):
        time_start = time.time()
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, scaler)
        val_loss, val_dice = validate_epoch(model, val_loader, criterion)
        time_end = time.time()

        print(f"[Time: {time_end-time_start}] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        torch.save(model.state_dict(), f"unet3d_brats_amp_chp{epoch}.pth")

    # Save model
    torch.save(model.state_dict(), "unet3d_brats_amp.pth")


if __name__ == "__main__":
    print("Running")

    train_dataset = LazyBratsDataset(
        "./data/preprocessed/split_brats_nii/train", lazy_chunk_size=30)
    val_dataset = LazyBratsDataset(
        "./data/preprocessed/split_brats_nii/val", lazy_chunk_size=20)
    print("Datasets created")

    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)
    print("Loaders created")

    model = UNet3DCheckpoint(in_channels=3, num_classes=4).to(device)
    print("Model instantiated and will start training")
    train_model(model, train_loader, val_loader, num_epochs=90, lr=1e-4)
