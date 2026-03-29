from os import path
import sys
import torch
from torch import optim, amp
from torch.utils.data import DataLoader
from src.preprocessing.class_imbalance import measure_class_inbalance
from src.utils import file_operations
from src.utils.device import get_device_type
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
from src.utils.loss_plotter import plot_loss_history


def preprocess(config: Config):
    preprocessor = Preprocessor(
        config.file_paths.raw_data,
        config.file_paths.preproc_data,
        config.preprocessing_config.split_ratio,
    )
    preprocessor.preprocess()


def train(config: Config):
    train_dataset = BraTSDataset(
        path.join(config.file_paths.preproc_data, PREPROC_TRAIN), config=config
    )
    test_dataset = BraTSDataset(
        path.join(config.file_paths.preproc_data, PREPROC_TEST), config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=True,
        num_workers=config.loader_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train_config.batch_size,
        shuffle=False,
        num_workers=config.loader_workers,
    )

    model = UNet3D(in_channels=3, num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=config.train_config.learning_rate)
    criterion = WeightedDiceFocalLoss(config.train_config.weighted_loss)
    scaler = amp.GradScaler(get_device_type())  # type: ignore

    start_epoch = 0
    if config.train_config.resume_training:
        model_state, optimizer_state, scaler_state, epoch = file_operations.load_torch(
            config.train_config.resume_from
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.train_config.learning_rate

        scaler.load_state_dict(scaler_state)
        start_epoch = epoch

    trainer = Trainer(
        config,
        model,
        optimizer,
        criterion,
        scaler,
        train=train_loader,
        test=test_loader,
        start_epoch=start_epoch,
    )
    trainer.fit(num_epochs=config.train_config.num_epochs)


def evaluate(config: Config) -> tuple[float, float]:
    test_dataset = BraTSDataset(
        path.join(config.file_paths.preproc_data, PREPROC_TEST), config=config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.validation_config.batch_size,
        shuffle=False,
        num_workers=config.loader_workers,
    )

    model_state_dict, _, _, _ = file_operations.load_torch(
        config.validation_config.model_path
    )
    model = UNet3D(in_channels=3, num_classes=4)
    model.load_state_dict(model_state_dict)

    evaluator = Evaluator(model, test_loader)
    return evaluator.validate_model()


def predict(config: Config, rmi_folder: str):
    log = logger.get_logger()

    # Load model
    model_state_dict, _, _, _ = file_operations.load_torch(
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

    if config.preprocessing_config.print_class_imbalance:
        preproc_folder = (
            PREPROC_TRAIN
            if len(sys.argv) > 1 and sys.argv[1] == "train"
            else PREPROC_TEST
        )
        measure_class_inbalance(config.file_paths.preproc_data + preproc_folder)

    if config.preprocessing_config.preprocess:
        preprocess(config)

    if config.train_config.train:
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        log.info(f"torch.version.cuda: {torch.version.cuda}")  # type: ignore
        log.info(f"compiled archs: {torch.cuda.get_arch_list()}")

        log.info("Starting training...")
        train(config)

    if config.validation_config.evaluate:
        log.info("Starting evaluation...")
        mean_iou, mean_dice = evaluate(config)
        log.info(
            f"Validation results - Mean IoU: {mean_iou:.2f}, Mean Dice: {mean_dice:.2f}"
        )

    if config.validation_config.plot_loss:
        model_name = path.splitext(path.basename(config.validation_config.model_path))[
            0
        ]
        loss_history_file = path.join(
            config.train_config.loss_history_path,
            f"{model_name}_losses.npy",
        )

        if not path.exists(loss_history_file):
            log.warning(
                f"Loss history file not found for model '{model_name}': {loss_history_file}"
            )
        else:
            log.info(f"Plotting loss history from: {loss_history_file}")
            plot_loss_history(loss_history_file)

    if config.validation_config.predict:
        if len(sys.argv) < 2:
            log.info("Prediction usage: python -m src.main <rmi_folder>")
            sys.exit(1)
        rmi_folder = sys.argv[1]
        log.info(f"Starting prediction on RMI folder: {rmi_folder}")
        predict(config, rmi_folder)


if __name__ == "__main__":
    main()
