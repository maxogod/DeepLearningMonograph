import os
from typing import Any, Dict, Tuple, cast
from nibabel.loadsave import load
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch

MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
SCALER_STATE_DICT = "scaler_state_dict"
EPOCH_STATE = "epoch_state"

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


def save_torch(
    file_path: str,
    model: Dict,
    optimizer: Dict,
    scaler: Dict,
    epoch: int,
) -> None:
    torch.save(
        {
            MODEL_STATE_DICT: model,
            OPTIMIZER_STATE_DICT: optimizer,
            SCALER_STATE_DICT: scaler,
            EPOCH_STATE: epoch,
        },
        file_path,
    )


def load_torch(
    file_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_dict = torch.load(file_path, map_location=device)
    return (
        saved_dict[MODEL_STATE_DICT],
        saved_dict[OPTIMIZER_STATE_DICT],
        saved_dict[SCALER_STATE_DICT],
        saved_dict[EPOCH_STATE],
    )
