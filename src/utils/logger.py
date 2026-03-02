import logging
import logging.config
from src.config.config import Environment

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["console"],
    },
}


def setup_logging(env: Environment):
    config = LOGGING_CONFIG.copy()

    level = "DEBUG" if env == Environment.DEVELOPMENT else "INFO"

    config["handlers"]["console"]["level"] = level
    config["root"]["level"] = level

    logging.config.dictConfig(config)


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)
