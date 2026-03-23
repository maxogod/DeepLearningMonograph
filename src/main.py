from os import path
import sys
import torch
from torch.utils.data import DataLoader
from src.utils import file_operations
from src.utils.consts import (
    PREPROC_TEST,
    PREPROC_TRAIN,
)
import src.utils.logger as logger
from src.config.config import Config
from src.preprocessing.preprocessor import Preprocessor
from src.models.unet_3d import UNet3D
from src.training.loss_functions import WeightedDiceFocalLoss
from src.training.train import Trainer
from src.inference.evaluator import Evaluator
from src.inference.predictor import Predictor
from src.inference.plotter import InferencePlotter
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
    log = logger.get_logger()

    # Load model
    model_state_dict, _ = file_operations.load_torch(
        config.validation_config.model_path
    )
    model = UNet3D(in_channels=3, num_classes=4)
    model.load_state_dict(model_state_dict)

    predictor = Predictor(model)
    x, volume, seg_onehot, files = predictor.prepare_from_rmi_folder(rmi_folder)
    log.info(f"T1CE: {files['t1ce']}")
    log.info(f"T2:   {files['t2']}")
    log.info(f"FLAIR:{files['flair']}")
    log.info(f"SEG:  {files['seg']}")

    preds = predictor.predict(x)

    plotter = InferencePlotter()
    plotter.plot_prediction(volume=volume, seg_onehot=seg_onehot, preds=preds)


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
