import os
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict
import yaml


class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


@dataclass
class PreprocessingConfig:
    preprocess: bool
    split_ratio: float


@dataclass
class TrainConfig:
    train: bool
    batch_size: int
    num_epochs: int
    save_model: bool
    model_name: str


@dataclass
class ValidationConfig:
    model_name: str
    evaluate: bool
    predict: bool


@dataclass
class FilePaths:
    raw_data: str
    preproc_data: str
    model_save_path: str


class Config:
    def __init__(self, config_path: str = "./config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict: Dict[str, Any] = yaml.safe_load(f)

        # Environment
        env_str = config_dict.get("environment", "development")
        self.environment = Environment(env_str)

        # Global
        self.random_seed = config_dict.get("random_seed", 42)

        # Nested configs
        self.preprocessing_config = PreprocessingConfig(
            **config_dict.get("preprocessing_config", {})
        )

        self.train_config = TrainConfig(**config_dict.get("train_config", {}))

        self.validation_config = ValidationConfig(
            **config_dict.get("validation_config", {})
        )

        self.file_paths = FilePaths(**config_dict.get("file_paths", {}))
