import os
from typing import Any, Dict, cast
from nibabel.loadsave import load
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch


# --- Misc ---


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# --- Raw Images ---


def load_nii(file_path: str) -> np.ndarray:
    nii_image: Nifti1Image = cast(Nifti1Image, load(file_path))
    return nii_image.get_fdata()


# --- Preprocessed Data ---


def save_npy(file_path: str, data: np.ndarray):
    np.save(file_path, data)


def load_npy(file_path: str) -> np.ndarray:
    return np.load(file_path)


# --- Model Checkpoints ---


def save_torch(file_path: str, state_dict: Dict[str, Any]) -> None:
    torch.save(state_dict, file_path)


def load_torch(file_path: str) -> Dict[str, Any]:
    return torch.load(file_path)
