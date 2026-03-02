from glob import glob
from src.utils.consts import FLAIR_SUFFIX, SEG_SUFFIX, T1CE_SUFFIX, T2_SUFFIX
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
