from os import path
import sys
import torch
from torch.utils.data import DataLoader
from src.utils.consts import PREPROC_TEST, PREPROC_TRAIN
import src.utils.logger as logger
from src.config.config import Config
from src.preprocessing.preprocessor import Preprocessor
from src.models.unet_3d import UNet3D
from src.training.loss_functions import WeightedDiceFocalLoss
from src.training.train import Trainer
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

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    model = UNet3D(in_channels=3, num_classes=4)
    criterion = WeightedDiceFocalLoss()

    trainer = Trainer(
        config, model, criterion, 1e-4, train=train_loader, test=test_loader
    )
    trainer.fit(num_epochs=config.train_config.num_epochs)


def predict(config: Config, rmi_folder: str):
    pass


def main():
    config = Config()
    logger.setup_logging(config.environment)
    torch.manual_seed(config.random_seed)

    if config.preprocessing_config.preprocess:
        preprocess(config)

    if config.train_config.train:
        print("CUDA available:", torch.cuda.is_available())
        print("torch.version.cuda:", torch.version.cuda)
        print("compiled archs:", torch.cuda.get_arch_list())
        train(config)

    if config.validation_config.evaluate:
        pass

    if config.validation_config.predict:
        if len(sys.argv) < 2:
            print("Prediction usage: python -m src.main <rmi_folder>")
            sys.exit(1)
        rmi_folder = sys.argv[1]
        predict(config, rmi_folder)


if __name__ == "__main__":
    main()
