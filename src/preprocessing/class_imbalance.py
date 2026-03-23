import numpy as np
from glob import glob
from tqdm import tqdm
from src.utils import logger
from src.utils.consts import SEG_NPY_SUFFIX

log = logger.get_logger()


def measure_class_inbalance(data_dir: str):
    path = data_dir + "*" + SEG_NPY_SUFFIX
    seg_files = sorted(glob(path))
    log.info(f"Found {len(seg_files)} segmentation files in {path}.")
    counts = np.zeros(4)
    for f in tqdm(seg_files):
        seg = np.load(f)  # shape (128,128,128,4) one-hot
        counts += seg.sum(axis=(0, 1, 2))

    freqs = counts / counts.sum()
    log.info(f"Class frequencies: {freqs}")

    # Inverse-frequency weights (excluding background):
    inv = 1.0 / (freqs + 1e-6)
    inv[0] = 0.05  # suppress background
    inv = inv / inv[1:].sum() * 0.95  # renormalize foreground to sum to 0.95
    log.info(f"Suggested weights: {inv}")
