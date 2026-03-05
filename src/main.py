from glob import glob
from os import path
import sys
import torch
from torch.utils.data import DataLoader
from src.utils import file_operations
from src.utils.consts import (
    FLAIR_SUFFIX,
    PREPROC_TEST,
    PREPROC_TRAIN,
    SEG_SUFFIX,
    T1CE_SUFFIX,
    T2_SUFFIX,
)
import src.utils.logger as logger
from src.config.config import Config
from src.preprocessing.preprocessor import Preprocessor
from src.models.unet_3d import UNet3D
from src.training.loss_functions import WeightedDiceFocalLoss
from src.training.train import Trainer
from src.inference.evaluator import Evaluator
from src.dataset.brats_dataset import BraTSDataset


def preprocess(config: Config):
    preprocessor = Preprocessor(
        config.file_paths.raw_data,
        config.file_paths.preproc_data,
        config.preprocessing_config.split_ratio,
    )
    preprocessor.preprocess()


def train(config: Config):
    train_dataset = BraTSDataset(
        path.join(config.file_paths.preproc_data, PREPROC_TRAIN)
    )
    test_dataset = BraTSDataset(path.join(config.file_paths.preproc_data, PREPROC_TEST))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = UNet3D(in_channels=3, num_classes=4)
    criterion = WeightedDiceFocalLoss()

    trainer = Trainer(
        config, model, criterion, 1e-4, train=train_loader, test=test_loader
    )
    trainer.fit(num_epochs=config.train_config.num_epochs)


def evaluate(config: Config) -> tuple[float, float]:
    test_dataset = BraTSDataset(path.join(config.file_paths.preproc_data, PREPROC_TEST))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    model_state_dict, _ = file_operations.load_torch(
        config.validation_config.model_path
    )
    model = UNet3D(in_channels=3, num_classes=4)
    model.load_state_dict(model_state_dict)

    evaluator = Evaluator(model, test_loader)
    return evaluator.validate_model()


def predict(config: Config, rmi_folder: str):
    model_state_dict, _ = file_operations.load_torch(
        config.validation_config.model_path
    )
    model = UNet3D(in_channels=3, num_classes=4)
    model.load_state_dict(model_state_dict)

    t1ce_file = sorted(glob(rmi_folder + "*/*" + T1CE_SUFFIX))[0]
    t2_file = sorted(glob(rmi_folder + "*/*" + T2_SUFFIX))[0]
    flair_file = sorted(glob(rmi_folder + "*/*" + FLAIR_SUFFIX))[0]
    seg_file = sorted(glob(rmi_folder + "*/*" + SEG_SUFFIX))[0]


def main():
    config = Config()
    logger.setup_logging(config.environment)
    log = logger.get_logger()
    torch.manual_seed(config.random_seed)

    if config.preprocessing_config.preprocess:
        preprocess(config)

    if config.train_config.train:
        log.info("CUDA available:", torch.cuda.is_available())
        log.info("torch.version.cuda:", torch.version.cuda)  # type: ignore
        log.info("compiled archs:", torch.cuda.get_arch_list())

        log.info("Starting training...")
        train(config)

    if config.validation_config.evaluate:
        log.info("Starting evaluation...")
        mean_iou, mean_dice = evaluate(config)
        log.info(
            f"Validation results - Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}"
        )

    if config.validation_config.predict:
        if len(sys.argv) < 2:
            log.info("Prediction usage: python -m src.main <rmi_folder>")
            sys.exit(1)
        rmi_folder = sys.argv[1]
        log.info(f"Starting prediction on RMI folder: {rmi_folder}")
        predict(config, rmi_folder)


if __name__ == "__main__":
    main()
