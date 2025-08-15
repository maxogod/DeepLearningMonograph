import numpy as np
import tqdm
from torch.utils.data import Dataset
import src.utils.consts as consts
import glob
import random

VOLUMES_IDX = 0
MASKS_IDX = 1


class LazyBratsDataset(Dataset):
    def __init__(self, data_path, lazy_chunk_size=50):
        super().__init__()
        self.data_path = data_path
        self.chunk_size = lazy_chunk_size
        self.files, self.indices = self._get_file_paths()

        self.data = {}
        self.chunk_start = 0
        self.chunk_end = lazy_chunk_size if lazy_chunk_size > len(
            self.files[MASKS_IDX]) else len(self.files[MASKS_IDX])

        self._load_chunk()

    def _get_file_paths(self):
        volumes = sorted(glob.glob(self.data_path +
                         consts.VOLUMES_PATH + "/*.npy"))
        masks = sorted(glob.glob(self.data_path +
                       consts.MASKS_PATH + "/*.npy"))

        files = (volumes, masks)
        indices = [i for i in range(len(volumes))]
        random.shuffle(indices)

        return files, indices

    def __len__(self):
        return len(self.files[MASKS_IDX])

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        if idx not in self.data:
            if idx >= len(self):
                raise ValueError(
                    f"index {idx} not valid for length {len(self)}")
            self._move_chunk(idx)
            self._load_chunk()
        return self.data[idx]

    def _move_chunk(self, idx):
        self.chunk_start = idx
        self.chunk_end = min(idx + self.chunk_size, len(self))

    def _load_chunk(self):
        self.data.clear()
        for p in range(self.chunk_start, self.chunk_end):
            volume = np.load(self.files[VOLUMES_IDX][p])
            mask = np.load(self.files[MASKS_IDX][p])
            self.data[p] = (volume, mask)


if __name__ == "__main__":
    dataset = LazyBratsDataset("./data/preprocessed/split_brats_nii/train")
    for _ in tqdm.tqdm(dataset, "Loopinn"):
        continue
