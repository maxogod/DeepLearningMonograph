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


def voxel_count_percentage_per_class(
    data_dir: str, segs_idx: list[int] = [8, 50, 38, 11]
):
    test_segs = sorted(glob(data_dir + "*" + SEG_NPY_SUFFIX))
    for idx in segs_idx:
        seg = np.load(test_segs[idx])  # (128,128,128,4)
        voxel_counts = seg.sum(axis=(0, 1, 2))
        total = seg.shape[0] * seg.shape[1] * seg.shape[2]
        log.info(f"Sample {idx}:")
        log.info(
            f"  NCR voxels: {int(voxel_counts[1])} ({100*voxel_counts[1]/total:.2f}%)"
        )
        log.info(
            f"  ED  voxels: {int(voxel_counts[2])} ({100*voxel_counts[2]/total:.2f}%)"
        )
        log.info(
            f"  ET  voxels: {int(voxel_counts[3])} ({100*voxel_counts[3]/total:.2f}%)"
        )
