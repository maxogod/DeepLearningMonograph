from glob import glob
from torch.utils.data import Dataset
import torch

from src.utils import file_operations
from src.utils.consts import SEG_NPY_SUFFIX, VOLUME_NPY_SUFFIX
from src.utils.logger import get_logger

logger = get_logger()


class BraTSDataset(Dataset):
    def __init__(self, data_path: str):
        self.volume_files = sorted(glob(data_path + "*" + VOLUME_NPY_SUFFIX))
        self.seg_files = sorted(glob(data_path + "*" + SEG_NPY_SUFFIX))

        if len(self.volume_files) != len(self.seg_files):
            raise ValueError(
                f"Number of volume files ({len(self.volume_files)}) does not match number of segmentation files ({len(self.seg_files)})."
            )
        logger.debug(
            f"Found {len(self.volume_files)} volume files and {len(self.seg_files)} segmentation files."
        )

    def __len__(self):
        return len(self.volume_files)

    def __getitem__(self, idx):
        vol = file_operations.load_npy(self.volume_files[idx])  # [H, W, D, 3]
        seg = file_operations.load_npy(self.seg_files[idx])  # [H, W, D, 4]

        x = torch.from_numpy(vol).permute(3, 2, 0, 1).float()  # [3, D, H, W]
        y = torch.from_numpy(seg).permute(3, 2, 0, 1).float()  # [4, D, H, W]

        return x, y


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    from src.config.config import Environment
    from torch.utils.data import DataLoader

    setup_logging(Environment.DEVELOPMENT)

    dataset = BraTSDataset(data_path="data/preproc/")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    batch = next(iter(loader))

    x, y = batch
    print(f"Volume batch shape: {x.shape}, Segmentation batch shape: {y.shape}")
    print(f"Volume dtype: {x.dtype}, Segmentation dtype: {y.dtype}")
