import sys
import src.utils.logger as logger
from src.config.config import Config
from src.preprocessing.preprocessor import Preprocessor


def main():
    config = Config()
    logger.setup_logging(config.environment)

    if config.preprocessing_config.preprocess:
        preprocessor = Preprocessor(
            config.file_paths.raw_data,
            config.file_paths.preproc_data,
            config.preprocessing_config.split_ratio,
        )
        preprocessor.preprocess()

    if config.train_config.train:
        pass

    if config.validation_config.evaluate:
        pass

    if config.validation_config.predict:
        if len(sys.argv) < 2:
            print("Prediction usage: python -m src.main <rmi_folder>")
            sys.exit(1)
        _ = sys.argv[1]
        pass


if __name__ == "__main__":
    main()
