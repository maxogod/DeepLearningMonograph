from glob import glob
from os import path
import sys
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from src.utils import file_operations
from src.utils.consts import (
    FLAIR_SUFFIX,
    PREPROC_TEST,
    PREPROC_TRAIN,
    SEG_SUFFIX,
    T1CE_SUFFIX,
    T2_SUFFIX,
)
import src.utils.logger as logger
from src.config.config import Config
from src.preprocessing.preprocessor import Preprocessor
from src.models.unet_3d import UNet3D
from src.training.loss_functions import WeightedDiceFocalLoss
from src.training.train import Trainer
from src.inference.evaluator import Evaluator
from src.inference.predictor import Predictor
from src.dataset.brats_dataset import BraTSDataset


def preprocess(config: Config):
    preprocessor = Preprocessor(
        config.file_paths.raw_data,
        config.file_paths.preproc_data,
        config.preprocessing_config.split_ratio,
    )
    preprocessor.preprocess()


def train(config: Config):
    train_dataset = BraTSDataset(
        path.join(config.file_paths.preproc_data, PREPROC_TRAIN)
    )
    test_dataset = BraTSDataset(path.join(config.file_paths.preproc_data, PREPROC_TEST))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
        num_workers=config.loader_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=False,
        num_workers=config.loader_workers,
    )

    model = UNet3D(in_channels=3, num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=config.train_config.learning_rate)
    criterion = WeightedDiceFocalLoss()

    if config.train_config.resume_training:
        model_state, optimizer_state = file_operations.load_torch(
            config.train_config.resume_from
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainer = Trainer(
        config, model, optimizer, criterion, train=train_loader, test=test_loader
    )
    trainer.fit(num_epochs=config.train_config.num_epochs)


def evaluate(config: Config) -> tuple[float, float]:
    test_dataset = BraTSDataset(path.join(config.file_paths.preproc_data, PREPROC_TEST))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=False,
        num_workers=config.loader_workers,
    )

    model_state_dict, _ = file_operations.load_torch(
        config.validation_config.model_path
    )
    model = UNet3D(in_channels=3, num_classes=4)
    model.load_state_dict(model_state_dict)

    evaluator = Evaluator(model, test_loader)
    return evaluator.validate_model()


def predict(config: Config, rmi_folder: str):
    log = logger.get_logger()

    # Load model
    model_state_dict, _ = file_operations.load_torch(
        config.validation_config.model_path
    )
    model = UNet3D(in_channels=3, num_classes=4)
    model.load_state_dict(model_state_dict)

    # Find NII files in the RMI folder
    t1ce_file = sorted(glob(rmi_folder + "*/*" + T1CE_SUFFIX))[0]
    t2_file = sorted(glob(rmi_folder + "*/*" + T2_SUFFIX))[0]
    flair_file = sorted(glob(rmi_folder + "*/*" + FLAIR_SUFFIX))[0]
    seg_file = sorted(glob(rmi_folder + "*/*" + SEG_SUFFIX))[0]

    log.info(f"T1CE: {t1ce_file}")
    log.info(f"T2:   {t2_file}")
    log.info(f"FLAIR:{flair_file}")
    log.info(f"SEG:  {seg_file}")

    # Preprocess NII files (same pipeline as Preprocessor._process_nii)
    scaler = MinMaxScaler(feature_range=(0, 1))

    def scale_volume(volume: np.ndarray) -> np.ndarray:
        original_shape = volume.shape
        return scaler.fit_transform(volume.reshape(-1, original_shape[-1])).reshape(
            original_shape
        )

    # Process segmentation mask: crop, remap class 4→3, one-hot encode
    seg_img = file_operations.load_nii(seg_file).astype(np.uint8)
    seg_img = seg_img[56:184, 56:184, 13:141]  # Crop to 128x128x128
    seg_img[seg_img == 4] = 3

    seg_onehot = (
        torch.nn.functional.one_hot(torch.from_numpy(seg_img).long(), num_classes=4)
        .numpy()
        .astype(np.uint8)
    )  # (H, W, D, 4)

    # Process image volumes: scale to [0,1], stack, crop
    t1ce_img = scale_volume(file_operations.load_nii(t1ce_file)).astype(np.float32)
    t2_img = scale_volume(file_operations.load_nii(t2_file)).astype(np.float32)
    flair_img = scale_volume(file_operations.load_nii(flair_file)).astype(np.float32)

    volume = np.stack(
        [t1ce_img, t2_img, flair_img], axis=3
    )  # (H_full, W_full, D_full, 3)
    volume = volume[56:184, 56:184, 13:141]  # Crop to (128, 128, 128, 3)

    # Convert to tensor with same permutation as BraTSDataset: (C, D, H, W)
    x = torch.from_numpy(volume).permute(3, 2, 0, 1).float()  # (3, D, H, W)
    x = x.unsqueeze(0)  # (1, 3, D, H, W) - add batch dim

    # Run prediction
    predictor = Predictor(model)
    preds = predictor.predict(x)  # (1, D, H, W) - class indices

    # Remove batch dimension: (D, H, W)
    preds_np = preds.squeeze(0).cpu().numpy()

    # --- Visualization ---
    # Pick the middle slice along depth axis (axis 2 of the original numpy layout H,W,D)
    # In numpy (H,W,D,...) layout, depth is axis 2
    # In model output (D,H,W), depth is axis 0
    slice_idx = volume.shape[2] // 2  # middle depth slice

    # Volume slices: (H, W) for each modality
    t1ce_slice = volume[:, :, slice_idx, 0]
    t2_slice = volume[:, :, slice_idx, 1]
    flair_slice = volume[:, :, slice_idx, 2]

    # Ground truth segmentation slice: one-hot (H, W, 4) → color image
    seg_slice = seg_onehot[:, :, slice_idx, :]  # (H, W, 4)
    gt_seg_img = _seg_to_color(seg_slice)

    # Predicted segmentation slice: (D, H, W) → pick depth slice → (H, W)
    pred_slice = preds_np[slice_idx, :, :]  # depth axis is 0 in (D,H,W)
    pred_seg_img = _class_indices_to_color(pred_slice)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(t1ce_slice, cmap="gray")
    axes[0].set_title("T1CE")
    axes[0].axis("off")

    axes[1].imshow(t2_slice, cmap="gray")
    axes[1].set_title("T2")
    axes[1].axis("off")

    axes[2].imshow(flair_slice, cmap="gray")
    axes[2].set_title("FLAIR")
    axes[2].axis("off")

    axes[3].imshow(gt_seg_img)
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    axes[4].imshow(pred_seg_img)
    axes[4].set_title("Prediction")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()


def _seg_to_color(seg_onehot_slice: np.ndarray) -> np.ndarray:
    """Convert a one-hot encoded segmentation slice (H, W, 4) to an RGB image."""
    h, w, _ = seg_onehot_slice.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Class 0: background (black), Class 1: red, Class 2: green, Class 3: blue
    colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
    for c, color in colors.items():
        mask = seg_onehot_slice[:, :, c] > 0
        color_img[mask] = color
    return color_img


def _class_indices_to_color(class_slice: np.ndarray) -> np.ndarray:
    """Convert a class index slice (H, W) to an RGB image."""
    h, w = class_slice.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
    for c, color in colors.items():
        mask = class_slice == c
        color_img[mask] = color
    return color_img


def main():
    config = Config()
    logger.setup_logging(config.environment)
    log = logger.get_logger()
    torch.manual_seed(config.random_seed)

    if config.preprocessing_config.preprocess:
        preprocess(config)

    if config.train_config.train:
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        log.info(f"torch.version.cuda: {torch.version.cuda}")  # type: ignore
        log.info(f"compiled archs: {torch.cuda.get_arch_list()}")

        log.info("Starting training...")
        train(config)

    if config.validation_config.evaluate:
        log.info("Starting evaluation...")
        mean_iou, mean_dice = evaluate(config)
        log.info(
            f"Validation results - Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}"
        )

    if config.validation_config.predict:
        if len(sys.argv) < 2:
            log.info("Prediction usage: python -m src.main <rmi_folder>")
            sys.exit(1)
        rmi_folder = sys.argv[1]
        log.info(f"Starting prediction on RMI folder: {rmi_folder}")
        predict(config, rmi_folder)


if __name__ == "__main__":
    main()
