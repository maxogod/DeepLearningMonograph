import sys
import src.utils.logger as logger
from src.config.config import Config
from src.preprocessing.preprocessor import Preprocessor

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.main <data_path> <save_path>")
        sys.exit(1)

    config = Config()
    logger.setup_logging(config.environment)

    data_path = sys.argv[1]
    save_path = sys.argv[2]
    preprocessor = Preprocessor(data_path, save_path)
    preprocessor.preprocess()
