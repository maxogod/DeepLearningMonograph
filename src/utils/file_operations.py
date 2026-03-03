import os
from typing import cast
from nibabel.loadsave import load
from nibabel.nifti1 import Nifti1Image
import numpy as np


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_nii(file_path: str) -> np.ndarray:
    nii_image: Nifti1Image = cast(Nifti1Image, load(file_path))
    return nii_image.get_fdata()
