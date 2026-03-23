from glob import glob
from os import path
from typing import Dict, Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
from src.utils import file_operations
from src.utils.consts import FLAIR_SUFFIX, SEG_SUFFIX, T1CE_SUFFIX, T2_SUFFIX
from src.utils.device import get_device


class Predictor:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

        self.device = get_device()
        self.model.to(self.device)

    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        imgs = imgs.to(self.device).float()

        outputs = self.model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def prepare_from_rmi_folder(
        self, rmi_folder: str
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict[str, str]]:
        files = self._find_nii_files(rmi_folder)

        seg_img = file_operations.load_nii(files["seg"]).astype(np.uint8)
        seg_img = seg_img[56:184, 56:184, 13:141]
        seg_img[seg_img == 4] = 3

        seg_onehot = (
            torch.nn.functional.one_hot(torch.from_numpy(seg_img).long(), num_classes=4)
            .numpy()
            .astype(np.uint8)
        )

        t1ce_img = self._scale_volume(file_operations.load_nii(files["t1ce"]))
        t2_img = self._scale_volume(file_operations.load_nii(files["t2"]))
        flair_img = self._scale_volume(file_operations.load_nii(files["flair"]))

        volume = np.stack([t1ce_img, t2_img, flair_img], axis=3).astype(np.float32)
        volume = volume[56:184, 56:184, 13:141]

        x = torch.from_numpy(volume).permute(3, 2, 0, 1).float().unsqueeze(0)
        return x, volume, seg_onehot, files

    def _scale_volume(self, volume: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(0, 1))
        original_shape = volume.shape
        return scaler.fit_transform(volume.reshape(-1, original_shape[-1])).reshape(
            original_shape
        )

    def _find_nii_files(self, rmi_folder: str) -> Dict[str, str]:
        file_patterns = {
            "t1ce": T1CE_SUFFIX,
            "t2": T2_SUFFIX,
            "flair": FLAIR_SUFFIX,
            "seg": SEG_SUFFIX,
        }
        files: Dict[str, str] = {}

        for key, suffix in file_patterns.items():
            candidates = sorted(glob(path.join(rmi_folder, f"*{suffix}")))
            if not candidates:
                candidates = sorted(
                    glob(path.join(rmi_folder, "**", f"*{suffix}"), recursive=True)
                )
            if not candidates:
                raise FileNotFoundError(
                    f"No file found for suffix '{suffix}' in folder '{rmi_folder}'"
                )
            files[key] = candidates[0]

        return files
