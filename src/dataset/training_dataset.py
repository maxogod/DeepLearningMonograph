from glob import glob
from torch.utils.data import Dataset
import torch
import numpy as np

from src.utils import file_operations
from src.utils.consts import SEG_NPY_SUFFIX, VOLUME_NPY_SUFFIX


class BraTSDataset(Dataset):
    def __init__(self, data_path: str):
        self.volume_files = sorted(glob(data_path + "*/*" + VOLUME_NPY_SUFFIX))
        self.seg_files = sorted(glob(data_path + "*/*" + SEG_NPY_SUFFIX))

        if len(self.volume_files) != len(self.seg_files):
            raise ValueError(
                f"Number of volume files ({len(self.volume_files)}) does not match number of segmentation files ({len(self.seg_files)})."
            )

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        vol = file_operations.load_npy(self.volume_files[idx])  # [D, H, W, 3]
        seg = file_operations.load_npy(self.seg_files[idx])  # [D, H, W, 4]

        vol = np.transpose(vol, (3, 0, 1, 2))  # [3, D, H, W]
        seg = np.transpose(seg, (3, 0, 1, 2))  # [4, D, H, W]

        x = torch.from_numpy(vol).float()
        y = torch.from_numpy(seg).float()

        return x, y


"""
dataset = BratsDataset(
    data_path
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)


for images, masks in loader:
    # images: [B, C, H, W]
    # masks:  [B, H, W]
    outputs = model(images)
    loss = criterion(outputs, masks)
"""
