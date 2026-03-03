from glob import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
from src.utils import file_operations
from src.utils.consts import (
    FLAIR_SUFFIX,
    SEG_SUFFIX,
    T1CE_SUFFIX,
    T2_SUFFIX,
    VOLUME_SUFFIX,
)
from src.utils.logger import get_logger


logger = get_logger()


class Preprocessor:
    def __init__(self, data_path: str) -> None:
        self.t1ce_files = sorted(glob(data_path + "*/*" + T1CE_SUFFIX))
        self.t2_files = sorted(glob(data_path + "*/*" + T2_SUFFIX))
        self.flair_files = sorted(glob(data_path + "*/*" + FLAIR_SUFFIX))
        self.seg_files = sorted(glob(data_path + "*/*" + SEG_SUFFIX))
        logger.debug(
            f"Found {len(self.t1ce_files)} T1CE files, {len(self.t2_files)} T2 files, {len(self.flair_files)} FLAIR files, and {len(self.seg_files)} SEG files."
        )

        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess(self, save_path: str):
        count = 0
        for t1ce_file, t2_file, flair_file, seg_file in zip(
            self.t1ce_files, self.t2_files, self.flair_files, self.seg_files
        ):
            t1ce_img = self.scale_volume(file_operations.load_nii(t1ce_file)).astype(
                np.float32
            )
            t2_img = self.scale_volume(file_operations.load_nii(t2_file)).astype(
                np.float32
            )
            flair_img = self.scale_volume(file_operations.load_nii(flair_file)).astype(
                np.float32
            )
            volume = np.stack([t1ce_img, t2_img, flair_img], axis=3)
            volume = volume[56:184, 56:184, 13:141]  # Crop to 128x128x128

            seg_img = file_operations.load_nii(seg_file).astype(np.uint8)
            seg_img = seg_img[56:184, 56:184, 13:141]  # Crop to 128x128x128
            seg_img[seg_img == 4] = 3
            seg_img = (
                F.one_hot(torch.from_numpy(seg_img), num_classes=4)
                .numpy()
                .astype(np.uint8)
            )

            file_operations.save_npy(f"{save_path}/{count}{VOLUME_SUFFIX}", volume)
            count += 1

    def scale_volume(self, volume: np.ndarray) -> np.ndarray:
        original_shape = volume.shape
        scaled_volume = self.scaler.fit_transform(
            volume.reshape(-1, original_shape[-1])
        ).reshape(original_shape)
        return scaled_volume
