from glob import glob
from os import path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
from src.utils import file_operations
from src.utils.consts import (
    FLAIR_SUFFIX,
    PREPROC_TEST,
    PREPROC_TRAIN,
    SEG_NPY_SUFFIX,
    SEG_SUFFIX,
    T1CE_SUFFIX,
    T2_SUFFIX,
    VOLUME_NPY_SUFFIX,
)
from src.utils.logger import get_logger


logger = get_logger()


class Preprocessor:
    def __init__(
        self, data_path: str, save_path: str, split_ratio: float = 0.0
    ) -> None:
        self.t1ce_files = sorted(glob(data_path + "*/*" + T1CE_SUFFIX))
        self.t2_files = sorted(glob(data_path + "*/*" + T2_SUFFIX))
        self.flair_files = sorted(glob(data_path + "*/*" + FLAIR_SUFFIX))
        self.seg_files = sorted(glob(data_path + "*/*" + SEG_SUFFIX))
        logger.debug(
            f"Found {len(self.t1ce_files)} T1CE files, {len(self.t2_files)} T2 files, {len(self.flair_files)} FLAIR files, and {len(self.seg_files)} SEG files."
        )

        self.save_path = save_path
        file_operations.mkdir(path.join(save_path, PREPROC_TRAIN))
        file_operations.mkdir(path.join(save_path, PREPROC_TEST))

        self.split_ratio = split_ratio

        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self, amount: int | None = None) -> None:
        count = 0
        train_count = 0

        for t1ce_file, t2_file, flair_file, seg_file in zip(
            self.t1ce_files, self.t2_files, self.flair_files, self.seg_files
        ):
            if count % 50 == 0:
                logger.debug(f"Processing file {count} and continuing...")

            result = self._process_nii(t1ce_file, t2_file, flair_file, seg_file)
            if result is None:
                continue
            volume, seg_img = result

            # Save preprocessed data
            saved_train = self._save_train_test(volume, seg_img, count)

            if saved_train:
                train_count += 1
            count += 1
            if amount is not None and count >= amount:
                break
        logger.debug(
            f"Preprocessed: {count} imgs & train/test ratio: {train_count}/{count-train_count}."
        )

    def _process_nii(
        self, t1ce_file: str, t2_file: str, flair_file: str, seg_file: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        # Process segmentation mask
        seg_img = file_operations.load_nii(seg_file).astype(np.uint8)
        seg_img = seg_img[56:184, 56:184, 13:141]  # Crop to 128x128x128
        seg_img[seg_img == 4] = 3

        total = seg_img.size
        background = np.sum(seg_img == 0)  # Unlabeled voxels
        foreground_fraction = 1 - (background / total)

        if foreground_fraction < 0.01:
            logger.warning(
                f"Skipping {seg_file} due to insufficient foreground pixels."
            )
            return None

        seg_img = (
            F.one_hot(torch.from_numpy(seg_img).long(), num_classes=4)
            .numpy()
            .astype(np.uint8)
        )  # one-hot requires long tensor

        # Process and image volumes
        t1ce_img = self._scale_volume(file_operations.load_nii(t1ce_file)).astype(
            np.float32
        )
        t2_img = self._scale_volume(file_operations.load_nii(t2_file)).astype(
            np.float32
        )
        flair_img = self._scale_volume(file_operations.load_nii(flair_file)).astype(
            np.float32
        )

        volume = np.stack([t1ce_img, t2_img, flair_img], axis=3)
        volume = volume[56:184, 56:184, 13:141]  # Crop to 128x128x128

        return volume, seg_img

    def _scale_volume(self, volume: np.ndarray) -> np.ndarray:
        original_shape = volume.shape
        scaled_volume = self.scaler.fit_transform(
            volume.reshape(-1, original_shape[-1])
        ).reshape(original_shape)
        return scaled_volume

    def _save_train_test(self, volume: np.ndarray, seg: np.ndarray, index: int) -> bool:
        folder = PREPROC_TRAIN
        if self.split_ratio != 0.0 and np.random.rand() >= self.split_ratio:
            folder = PREPROC_TEST

        save_path = path.join(self.save_path, folder)
        file_operations.save_npy(f"{save_path}/{index}{VOLUME_NPY_SUFFIX}", volume)
        file_operations.save_npy(f"{save_path}/{index}{SEG_NPY_SUFFIX}", seg)

        return folder == PREPROC_TRAIN  # Whether it was assigned to train or test


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    from src.config.config import Environment

    setup_logging(Environment.DEVELOPMENT)

    preprocessor = Preprocessor("data/raw/training/", "data/tmp/")
    preprocessor.preprocess(amount=1)

    vol = file_operations.load_npy("./data/tmp/train/0_volume.npy")
    seg = file_operations.load_npy("./data/tmp/train/0_seg.npy")

    vol_min = vol.min()
    vol_max = vol.max()
    if vol_min == 0.0 and vol_max == 1.0:
        print("Volume is correctly min-max scaled from 0 to 1.")
    else:
        print(f"Volume min: {vol_min}, max: {vol_max} - not scaled correctly.")

    seg_sum = seg.sum(axis=-1)
    if np.all(seg_sum == 1) and np.all((seg == 0) | (seg == 1)):
        print("Segmentation is correctly one-hot encoded.")
    else:
        print("Segmentation is not one-hot encoded.")
        print(f"Sum check: all sums ==1? {np.all(seg_sum == 1)}")
        print(f"Binary check: all 0 or 1? {np.all((seg == 0) | (seg == 1))}")

    print(f"Volume shape: {vol.shape}, Segmentation shape: {seg.shape}")
